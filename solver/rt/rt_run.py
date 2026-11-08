"""Solve the RT formulation (rt_model.py) at D = 4 from an hp state mapped into RT gauge (gg_to_rt.py), continue it in
D with EEG's protocol (steps of 1/2, the initial guess linearly extrapolated from the two previous solutions), and
compare Delta with hp/data/delta_all.csv.

    python3 solver/rt/rt_run.py --profile PROFILE.npz --grids 32x61,48x81 --cont_grid 32x61 --D_to 10 --mp 160 \
        --out DIR [--mutate=coef]

Exits 1 when |Delta_RT - Delta_hp| > --tol (1e-8) at the start D or at D_to, on the last grid. --mutate=coef scales the
coefficient (D - 3)(D - 2)/2 of Omega_3 by 1 + 1e-3; the run must then exit 1 with Delta moved by more than 1e-6.
--mp BITS polishes the start and end solves with the residual in arb (rt_mp.py) and the f64 Jacobian; without it the
f64 rounding along the weak direction moves Delta by 3e-4 at D = 10 on the 48x81 grid. The continuation itself is f64.
mu at D = 4 is also compared with Reiterer and Trubowitz's enclosure (arXiv:1203.3766v1, Eq. (3), radius 1e-80).

--solver mp (issue #105) does every solve, the continuation's included, with rt_mpsolve.newton_mp: the residual in arb
at --mp bits (default 192), each Newton step by flexible GMRES in arb preconditioned by the f64 LU of the f64 Jacobian
(built once per D, at the predictor) applied with double-double triangular solves. It continues past D = 10, where
kappa(J) exceeds 2^53 on grids that resolve Delta.

    python3 solver/rt/rt_run.py --start STATE.npz --grids 32x81 --solver mp --D_to 30 \
        --D_extra 20.043653,29.984746 --check_D 15,20.043653,29.984746 --tol 1e-10 --out DIR --tag TAG [--mutate=coef]

--start: an rt_run.py state (any grid, resampled to the first of --grids), solved first at --D_start (default its D).
--D_extra adds D values to the steps of --dD (the hp rows nearest 20 and 30); the predictor is linear in D through the
two previous solutions. --check_D lists where |Delta_RT - Delta_hp| <= --tol is checked (the start D always is, when
hp/data/delta_all.csv has that D). States are saved with their double-double parts (zh, zl; z = zh). Without --D_to only
the start D is solved: with --mutate=coef from an unmutated state, Delta must move by more than 1e-6 and the run exit 1.

Issue #145 adds these options. --start_prev A.npz,B.npz names older states (increasing D, before --start's), resampled
to the continuation grid, that seed the predictor's history, so a continuation restarts from its last saved states.
--pred_order k predicts by Lagrange extrapolation in D through the last k + 1 solutions (1, the default, is EEG's
linear predictor; 2 at --dD 1 starts at the residual of the linear one at --dD 0.5, 1.8e-4 at D = 30 on 32x101).
--extras_off_path solves the --D_extra points off the path: each is predicted from the steps of --dD and is not used
to predict later steps. --patient lets Newton continue through a stalled max|F| far above the tolerance, up to
--it_max steps (#105's rule stops it after step 3, which ended 32x121 solves at D = 32 at max|F| = 1.5e-26 while
max|dz| still fell). --full_below X takes every Newton step with max|dz| < X in full, without halving it.
--newton_tol (default 1e-40, #105's) is Newton's target max|F|; with --patient a stall within 1e6 of it is accepted.
Each record also carries the seconds in products and in the preconditioner per Newton step, the process's peak
resident memory (rss_peak_gb) and newton_tol.

Issue #163 adds --horizon K: the discretization with K Taylor rows at the horizon (rt_model.py's horizon rows, which
exclude a singular branch s^lambda with Re lambda < K). The default 0 is the collocation of the issues above. States
and records carry `horizon`. --mutate=norows runs the same command with the rows dropped (K = 0), the control that
must bring back the collocation's miss.
"""
import os
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")          # a shared machine: two BLAS threads
import sys
import csv
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_model import RT

MU_RT4 = 0.16830707896344996951013497904285742072100199080892966476395293134873313662587505
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def delta_hp_parts(D):
    """(Delta_q0, Delta_q1) of the hp/data/delta_all.csv row at D, or None."""
    with open(os.path.join(REPO, "hp", "data", "delta_all.csv")) as f:
        for r in csv.DictReader(f):
            if abs(float(r["D"]) - D) < 1e-9:
                return float(r["Delta_q0"]), float(r["Delta_q1"] or 0.0)
    return None


def delta_hp(D):
    p = delta_hp_parts(D)
    return None if p is None else p[0] + p[1]


def rss_peak_gb():
    import resource
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(m / (2 ** 30 if sys.platform == "darwin" else 2 ** 20), 2)     # bytes on macOS, KiB on Linux


def predict(hist, Dn, order):
    """The guess at Dn: Lagrange extrapolation in D through the last order + 1 entries (D, z) of hist."""
    pts = hist[-(order + 1):]
    if len(pts) == 1:
        return pts[0][1]
    if len(pts) == 2:                                   # EEG's linear predictor, as in issue #105's runs
        (Da, za), (Db, zb) = pts
        return zb + (zb - za) * (Dn - Db) / (Db - Da)
    zg = None
    for i, (Di, zi) in enumerate(pts):
        w = 1.0
        for j, (Dj, _) in enumerate(pts):
            if j != i:
                w *= (Dn - Dj) / (Di - Dj)
        zg = zi * w if zg is None else zg + zi * w
    return zg


def load_state(path):
    d = np.load(path)
    zh = d["zh"] if "zh" in d else d["z"]
    zl = d["zl"] if "zl" in d else np.zeros_like(zh)
    return zh, zl, float(d["D"]), int(d["Nt"]), int(d["N"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None)
    ap.add_argument("--start", default=None, help="an rt_run.py state to start from (instead of --profile)")
    ap.add_argument("--D_start", type=float, default=None, help="with --start: the first D solved (default its D)")
    ap.add_argument("--grids", default="32x41")
    ap.add_argument("--cont_grid", default=None, help="grid of the continuation (default: the first)")
    ap.add_argument("--D_to", type=float, default=None)
    ap.add_argument("--dD", type=float, default=0.5)
    ap.add_argument("--D_extra", default="", help="extra D values of the continuation, comma separated")
    ap.add_argument("--check_D", default=None, help="D values to check against hp (default: the start and D_to)")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="rt")
    ap.add_argument("--mutate", default=None)
    ap.add_argument("--horizon", type=int, default=0,
                    help="issue #163: K > 0 replaces the collocated rows of Omega_2,3,4 at the K points next to xi = -1 "
                    "by the Taylor coefficients of orders 1..K at xi = -1 (rt_model.py); --mutate=norows drops them")
    ap.add_argument("--solver", default="f64", choices=["f64", "mp"])
    ap.add_argument("--mp", type=int, default=0, help="bits of the arb residual (f64 solver: final solves only, 0: none;"
                    " mp solver: every solve, default 192)")
    ap.add_argument("--max_products", type=int, default=80)
    ap.add_argument("--refresh_dz", type=float, default=1e-3,
                    help="mp solver: rebuild the f64 Jacobian and LU after a Newton step with max|dz| above this")
    ap.add_argument("--refresh_ratio", type=float, default=1.5,
                    help="mp solver: rebuild them after an inner solve with this many times the products of the first")
    ap.add_argument("--start_prev", default="", help="with --start: older states, comma separated in increasing D, "
                    "that seed the predictor's history")
    ap.add_argument("--pred_order", type=int, default=1,
                    help="predictor: Lagrange extrapolation in D through the last pred_order + 1 solutions")
    ap.add_argument("--it_max", type=int, default=12, help="mp solver: Newton steps at most")
    ap.add_argument("--newton_tol", type=float, default=1e-40, help="mp solver: Newton stops at max|F| below this")
    ap.add_argument("--patient", action="store_true",
                    help="mp solver: a stalled max|F| stops Newton only near the tolerance (else after step 3)")
    ap.add_argument("--full_below", type=float, default=None,
                    help="mp solver: take the full Newton step when its max|dz| is below this (no step halving)")
    ap.add_argument("--extras_off_path", action="store_true",
                    help="solve the --D_extra points off the path (not used to predict later steps)")
    a = ap.parse_args()
    if (a.profile is None) == (a.start is None):
        ap.error("give one of --profile and --start")
    if a.solver == "mp" and not a.mp:
        a.mp = 192
    if a.start_prev and not a.start:
        ap.error("--start_prev needs --start")
    checks = [float(v) for v in a.check_D.split(",")] if a.check_D else None
    os.makedirs(a.out, exist_ok=True)
    coef = 1.0 + 1e-3 if a.mutate == "coef" else 1.0
    HK = 0 if a.mutate == "norows" else a.horizon          # --mutate=norows drops the horizon rows
    tag = a.tag + (f"_mut{a.mutate}" if a.mutate in ("coef", "norows") else "")
    log = open(os.path.join(a.out, f"{tag}.log"), "a")
    rec = open(os.path.join(a.out, f"{tag}.jsonl"), "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    grids = [tuple(int(v) for v in g.split("x")) for g in a.grids.split(",")]
    if a.solver == "mp":
        import flint
        from rt_mp import RTmp
        import rt_mpsolve as mps

    def save(z, D, rt):
        f = os.path.join(a.out, f"{tag}_D{D:.4f}_{rt.Nt}x{rt.N}.npz")
        if z.dtype == object:
            zh, zl = mps.to_dd(z)
            np.savez(f, z=zh, zh=zh, zl=zl, D=D, Nt=rt.Nt, N=rt.N, horizon=rt.K)
        else:
            np.savez(f, z=z, D=D, Nt=rt.Nt, N=rt.N, horizon=rt.K)

    def solve_mp(rt, z, D, label):
        """rt_mpsolve.newton_mp at D on rt's grid from z (f64 or arb). Returns (z_arb, ok, record)."""
        t0 = time.time()
        rm = RTmp(D, rt.Nt, rt.N, prec=a.mp, coef_mut=coef, horizon=HK)
        zf = mps.to_f64(z) if z.dtype == object else z
        J = rt.jacobian(zf)
        t_jac = time.time() - t0
        held = [mps.LUdd(J, overwrite=True)]
        del J
        pivot_ratio, t_lu = held[0].pivot_ratio, held[0].secs
        make = lambda zz: mps.LUdd(rt.jacobian(mps.to_f64(zz)), overwrite=True)
        # newton_mp holds the only reference, so a rebuild frees the old factors first
        zm, ok, info = mps.newton_mp(rm, z, held.pop(), tol=a.newton_tol, it_max=a.it_max, max_products=a.max_products,
                                     make_prec=make, refresh_dz=a.refresh_dz, refresh_ratio=a.refresh_ratio,
                                     log=say, stall_any=not a.patient, full_below=a.full_below)
        d0, d1 = rm.delta_m(zm)
        parts = delta_hp_parts(D)
        diff = None
        if parts is not None:
            diff = float((d0 - flint.arb(parts[0]) - flint.arb(parts[1])).mid())
        r = dict(D=D, grid=[rt.Nt, rt.N], L=rt.L, solver="mp", prec=a.mp, mu=float(zm[-1].mid()),
                 Delta_xi0=float(d0.mid()), Delta_xi1=float(d1.mid()), Delta_str=d0.mid().str(30, radius=False),
                 Delta_hp=(parts[0] + parts[1]) if parts else None, diff=diff, newton=info["res"],
                 products=info["products"], dz=info["dz"], lam=info["lam"], refresh=info["refresh"],
                 pivot_ratio=pivot_ratio, ok=bool(ok), t_jac=round(t_jac, 1), t_lu=round(t_lu, 1),
                 t_refresh=round(info["t_refresh"], 1), t_newton=round(sum(info["secs"]), 1),
                 secs=round(time.time() - t0, 1), label=label, mutate=a.mutate, step_secs=info["secs"],
                 t_prod=info["t_prod"], t_prec=info["t_prec"], pred_order=a.pred_order, rss_peak_gb=rss_peak_gb(),
                 newton_tol=a.newton_tol, horizon=rt.K)
        rec.write(json.dumps(r) + "\n")
        rec.flush()
        say(f"  {label} D {D:.6f} grid {rt.Nt}x{rt.N} L {rt.L}: {len(info['res']) - 1} Newton steps, max|F| "
            f"{info['res'][0]:.1e} -> {info['res'][-1]:.1e}, products {info['products']}, mu {float(zm[-1].mid()):.15f}, "
            f"Delta {d0.mid().str(20, radius=False)} (xi=1: {float(d1.mid()):.15f}), hp {r['Delta_hp']}, diff "
            f"{diff if diff is not None else float('nan'):.2e}, LU pivot ratio {pivot_ratio:.1e}, J {t_jac:.0f} s, "
            f"LU {t_lu:.0f} s, Newton {r['t_newton']:.0f} s (rebuilds {len(info['refresh'])}, {info['t_refresh']:.0f} s)")
        return zm, ok, r

    def solve(rt, z, D, label):
        if a.solver == "mp":
            return solve_mp(rt, z, D, label)
        t0 = time.time()
        z, ok, hist = rt.newton(z, tol=1e-11, it_max=15, verbose=False)
        d0, d1 = rt.delta(z)
        mp_hist = None
        if a.mp and label in ("start", "end"):
            # the residual in arb (rt_mp.py), f64 Jacobian: removes f64's rounding along the weak direction
            from rt_mp import RTmp, newton_mixed, to_f64
            rm = RTmp(D, rt.Nt, rt.N, prec=a.mp, coef_mut=coef, horizon=HK)
            zm, ok, mp_hist = newton_mixed(rm, z, tol=1e-28, it_max=40, log=lambda s: None)
            dm0, dm1 = rm.delta_m(zm)
            d0, d1 = float(dm0.mid()), float(dm1.mid())
            z = to_f64(zm)
        ref = delta_hp(D)
        r = dict(D=D, grid=[rt.Nt, rt.N], L=rt.L, mu=float(z[-1]), Delta_xi0=float(d0), Delta_xi1=float(d1),
                 Delta_hp=ref, diff=(float(d0 - ref) if ref is not None else None), newton=[float(h) for h in hist],
                 mixed=mp_hist, ok=bool(ok), secs=round(time.time() - t0, 1), label=label, mutate=a.mutate)
        if mp_hist:
            say(f"    mixed refinement ({a.mp} bits): {len(mp_hist) - 1} its, max|F| {mp_hist[0]:.1e} -> "
                f"{mp_hist[-1]:.1e}")
        rec.write(json.dumps(r) + "\n")
        rec.flush()
        say(f"  {label} D {D:.3f} grid {rt.Nt}x{rt.N} L {rt.L}: newton {len(hist) - 1} its, max|F| {hist[-1]:.1e}, "
            f"mu {z[-1]:.15f}, Delta {d0:.13f} (xi=1: {d1:.13f}), hp {ref}, diff "
            f"{(d0 - ref) if ref is not None else float('nan'):.2e}, {r['secs']} s")
        return z, ok, r

    failed = False

    def check(r, where):
        nonlocal failed
        if r["Delta_hp"] is None:
            if a.solver == "mp":
                say(f"  CHECK {where}: no hp row at D = {r['D']}; FAIL")
                failed = True
            return
        bad = abs(r["diff"]) > a.tol or not r["ok"]
        say(f"  CHECK {where}: |Delta_RT - Delta_hp| = {abs(r['diff']):.2e} {'>' if bad else '<='} {a.tol:g}"
            f" {'FAIL' if bad else 'ok'}")
        failed |= bad

    # ------------------------------------------------ start D: map (or load), then each grid
    if a.start:
        zh, zl, Ds, Nts, Ns = load_state(a.start)
        D0 = a.D_start if a.D_start is not None else Ds
        src = RT(Ds, Nts, Ns)
        say(f"== {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC  start {a.start} (D {Ds}, {Nts}x{Ns}) "
            f"D_start {D0} grids {grids} solver {a.solver} prec {a.mp} mutate {a.mutate} horizon {HK}")
        rt = RT(D0, *grids[0], coef_mut=coef, horizon=HK)
        z = zh if (Nts, Ns) == grids[0] else src.resample_to(zh, rt)
        zlo = zl if (Nts, Ns) == grids[0] else src.resample_to(zl, rt)
        if a.solver == "mp":
            flint.ctx.prec = a.mp
            z = mps.from_dd(z, zlo)
    else:
        from gg_to_rt import GGMap
        g = GGMap(a.profile, verbose=False)
        D0 = g.D
        say(f"== {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC  profile {a.profile} D {D0} grids {grids} "
            f"mutate {a.mutate}  map: lam {g.lam:.15f} mu {g.mu:.15f} {g.check}")
        rt = RT(D0, *grids[0], coef_mut=coef, horizon=HK)
        z, err, s = g.to_rt_phased(rt)
        say(f"  mapped initial guess: inversion error {err:.1e}, t shift {s:.4f}, max|F(z0)| "
            f"{np.abs(rt.residual(z)[0]).max():.2e}")
    states = {}
    for i, gr in enumerate(grids):
        if i > 0:
            new = RT(D0, *gr, coef_mut=coef, horizon=HK)
            zf = mps.to_f64(z) if z.dtype == object else z
            z = rt.resample_to(zf, new)
            rt = new
        z, ok, r = solve(rt, z, D0, "start")
        states[gr] = z
        # the start D is checked when hp has its row, or when --check_D lists it
        if a.solver == "mp" and (r["Delta_hp"] is not None or (checks and any(abs(D0 - c) < 1e-9 for c in checks))):
            check(r, f"D = {D0} grid {gr[0]}x{gr[1]}")
    zlast = mps.to_f64(z) if z.dtype == object else z
    if abs(D0 - 4) < 1e-12:
        say(f"  mu - mu_RT(D=4) = {zlast[-1] - MU_RT4:.2e}")
    if a.solver == "f64":
        check(r, f"D = {D0}")
    save(z, D0, rt)
    if a.D_to is None:
        say(f"== exit {1 if failed else 0}")
        sys.exit(1 if failed else 0)

    # ------------------------------------------------ continuation in D
    cg = tuple(int(v) for v in a.cont_grid.split("x")) if a.cont_grid else grids[0]
    rtc = RT(D0, *cg, coef_mut=coef, horizon=HK)
    if cg in states:
        zc = states[cg]
    else:
        zc = rt.resample_to(zlast, rtc)
    hist = []
    for path in [p for p in a.start_prev.split(",") if p]:      # older states seed the predictor's history
        ph, pl, Dp_, Ntp, Np = load_state(path)
        src = RT(Dp_, Ntp, Np)
        if (Ntp, Np) != cg:
            ph, pl = src.resample_to(ph, rtc), src.resample_to(pl, rtc)
        hist.append((Dp_, mps.from_dd(ph, pl) if a.solver == "mp" else ph))
        say(f"  predictor history: {path} (D {Dp_}, {Ntp}x{Np})")
    hist.append((D0, zc))
    extras = [round(float(v), 9) for v in a.D_extra.split(",") if v]
    Ds_ = list(np.arange(D0 + a.dD, a.D_to + 1e-9, a.dD)) + [a.D_to] + extras
    Ds_ = sorted(set(round(float(v), 9) for v in Ds_ if D0 + 1e-9 < v <= a.D_to + 1e-9))
    D = D0
    for Dn in Ds_:
        off = a.extras_off_path and Dn in extras
        zg = predict(hist, Dn, a.pred_order)
        rtn = RT(Dn, *cg, coef_mut=coef, horizon=HK)
        zn, ok, r = solve(rtn, zg, Dn, "extra" if off else "cont")
        if not ok:
            say("  continuation step failed; stopping")
            failed = True
            break
        if checks is not None and any(abs(Dn - c) < 1e-9 for c in checks):
            check(r, f"D = {Dn} grid {cg[0]}x{cg[1]}")
        save(zn, Dn, rtn)
        if off:
            continue
        zc, rtc = zn, rtn
        hist = hist[-a.pred_order:] + [(Dn, zc)]
        D = Dn
    # ------------------------------------------------ the end point on every grid
    if a.solver == "mp":
        for gr in grids:
            if gr == cg:
                continue
            new = RT(D, *gr, coef_mut=coef, horizon=HK)
            zz, ok, r = solve(new, rtc.resample_to(mps.to_f64(zc), new), D, "end")
            save(zz, D, new)
            if checks is None or any(abs(D - c) < 1e-9 for c in checks):
                check(r, f"D = {D} grid {gr[0]}x{gr[1]}")
        if checks is None and not failed:
            check(r, f"D = {D}")
    else:
        for gr in grids:
            if gr == cg:
                zz, ok, r = solve(rtc, zc, D, "end")
            else:
                new = RT(D, *gr, coef_mut=coef, horizon=HK)
                zz, ok, r = solve(new, rtc.resample_to(zc, new), D, "end")
                save(zz, D, new)
        check(r, f"D = {D}")
    say(f"== exit {1 if failed else 0}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
