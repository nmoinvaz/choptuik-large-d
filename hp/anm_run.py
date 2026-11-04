#!/usr/bin/env python3
"""Chained steps of the exact Taylor continuation in D (notes/QUAD_DOUBLE.md 13.15-13.16): from a converged pinned
state, the branch's Taylor coefficients to order N (anm_step.py's loop: Solver.residual_jet for the right-hand sides,
the Jacobian's LU at the base, one refinement per order), the step a = min(a_max, h_max) with
a_max = (eps |z_1| / |z_N|)^{1/(N-1)}, the prediction z(a), the corrector at D + a with T75's recipe (scan_hp.py's
"dnp": deflated_newton, then up to 4 plain Newton steps from below 1e-12; the tolerance grows 1.45x per unit of D
from the base's, and a stall below --accept_stall is accepted as T75 accepted its), and on convergence the new base;
on failure the step is halved, three times at most. Every
converged state is saved (npz, hpkernel.driver.save's layout) and every step recorded (jsonl), in a task folder of the
raw-data repository. p1's odd harmonics in (kfreeze, 13] stay at the first base's values throughout (the pinned
formulation of hp/pinning.py; their coefficients are zero).

    python3 anm_run.py RAWDIR OUTDIR --state T75/pin7_D73.2657.npz --kfreeze 3 [--N 24] [--eps 1e-16] [--h_max 2]
        [--steps 20] [--D_max 200] [--corrector dnp|deflated|newton] [--tol 1e-20] [--tol_growth 1.45]
        [--accept_stall 1e-19] [--junk_only 0] [--threads 4] [--w 24] [--trend T75] [--tol_qd 1e-28]
        [--qd_it 12] [--qd_w 0]

--corrector qd runs the quad-double Newton of the kernel (hpkernel.driver.newton_qd, Solver.newton_q) on the pinned
system (pinning.PinnedQ: the held harmonics' offset subtracted from the quad-double residual and added to the step, as
PinnedSolver does in double-double) from the double-double prediction, to --tol_qd on max |F| in at most --qd_it steps
(the quadratic phase begins only below |dz| ~ 1e-11, 13.19, so 8 cut off converging iterations). The base is polished the
same way first. The result's first two parts are the next base (the jets are double-double) and all four parts are
saved (save_q). Where the double-double Newton takes no step from a residual of 1e-14 in the horizon block's weak
directions (13.17), the quad-double one sees those directions. (Freezing the held harmonics as directions on the
unpinned kernel instead diverges: the rows the pinning drops stay, 13.19.) --qd_w W > 0 starts that Newton deflated
(newton_qd's phase 'all': the step without its components along the Jacobian's W weakest right singular directions),
and plain Newton finishes once that phase stalls, so a converged state solves the whole pinned system.

--trend FOLDER holds the harmonics on a trend instead of at the base's values (T75's pin-to-trend, in the
continuation's terms): for each held harmonic k the coefficients (a_k, b_k) of p1 across the stored states of FOLDER are
fitted as |v| = C n^s (n = D - 2) when their sign is constant and the fit's residual in ln|v| is below 0.05 (harmonics
3-9 at D 71-73: residuals 1e-5 to 5e-3); the others stay constant. The trend is taken relative to the base, v(D) =
v(D0) (n/n0)^s, so the base state is unchanged, its Taylor coefficients in h enter the jet state order by order (the
held rows of each right-hand side carry them, as PinnedSolver's offset carries the values), and the corrector at D + a
holds the harmonics at the trend's values there. Records carry the exponents.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import make_solver, newton_solve, deflated_newton, delta_correction, newton_qd, qparts, rnorm, save, save_q  # noqa: E402
from multidouble import dd_add, dd_scale, dd_norm  # noqa: E402
from pinning import PinnedQ, PinnedSolver, band_coeffs, band_field  # noqa: E402
from raw_out import require_raw  # noqa: E402
from taylor_predict import formula_delta  # noqa: E402


def load_state(path):
    d = np.load(path)
    m = json.loads(str(d["meta"]))
    return d["zh"].astype(float), d["zl"].astype(float), m


# D is carried in mpmath precision: the solver forms n = D - 2 in double-double from it, and at fixed data the
# residual changes by about 2e6 per unit of D (13.16), so the float64 rounding of D + a (7e-15 at D 73) would put
# 1e-8 into every prediction (found on the pilot's first run, 2026-11-04 00:10 UTC)
mp.mp.dps = 50


def coefficients(s, zh, zl, N, held=None):
    """z_1..z_N from the stored LU at (zh, zl): [(zh_k, zl_k)], norms, the refinement residuals. held = (fn, rows):
    fn(k) gives the held harmonics' order-k Taylor coefficients as a field (hi, lo) on the horizon block's p rows (the
    slice rows), added to the right-hand side there as PinnedSolver's offset is, so that z_k carries their motion"""
    Zh, Zl, norms, refine = [zh], [zl], [float(dd_norm(zh, zl))], []
    for k in range(1, N + 1):
        Hs, Ls = s.residual_jet(Zh, Zl, k + 1, 0)
        bh, bl = -np.asarray(Hs[k]), -np.asarray(Ls[k])
        if held is not None:
            fn, rows = held
            th, tl = fn(k)
            bh, bl = bh.copy(), bl.copy()
            ah, al = dd_add(bh[rows], bl[rows], np.asarray(th), np.asarray(tl))
            bh[rows], bl[rows] = ah, al
        xh, xl = s.lu_solve(bh, bl)
        jh, jl = s.jvp(zh, zl, xh, xl)[:2]
        r0h, r0l = dd_add(bh, bl, -jh, -jl)
        ch, cl = s.lu_solve(r0h, r0l)
        xh, xl = dd_add(xh, xl, ch, cl)
        jh, jl = s.jvp(zh, zl, xh, xl)[:2]
        r1h, r1l = dd_add(bh, bl, -jh, -jl)
        rh, rl = -bh, -bl
        Zh.append(xh); Zl.append(xl)
        norms.append(float(dd_norm(xh, xl)))
        refine.append((float(dd_norm(r0h, r0l)), float(dd_norm(r1h, r1l))))
    return Zh, Zl, norms, refine


def main():
    args = list(sys.argv[1:])
    opt = dict(state="T75/pin7_D73.2657.npz", kfreeze=3, N=24, eps=1e-16, h_max=2.0, steps=20, D_max=200.0, corrector="dnp",
               tol=1e-20, tol_growth=1.45, accept_stall=1e-17, junk_only=0, threads=4, w=24, trend="", tol_qd=1e-28, qd_it=12, qd_w=0)
    for k in list(opt):
        f = "--" + k
        if f in args:
            i = args.index(f); opt[k] = type(opt[k])(args[i + 1]); del args[i:i + 2]
    raw, out = args[0], args[1]
    os.makedirs(out, exist_ok=True)
    rec_path = os.path.join(out, "anm_run.jsonl")
    require_raw(rec_path)
    zh, zl, m = load_state(os.path.join(raw, opt["state"]))
    D, M, nodes = mp.mpf(m.get("D_str", repr(m["D"]))), int(m["M"]), list(m["nodes"])
    kt, kc, ks = m.get("kt", 36), m.get("kc", 146), m.get("ks", 130)
    kf, N = opt["kfreeze"], opt["N"]
    D_start = D
    junk_only = bool(opt["junk_only"])

    def tol_at(Dx):
        return opt["tol"] * opt["tol_growth"] ** float(Dx - D_start)

    last_q = {}

    def correct(spx, ph, pl, tol):
        """T75's corrector (scan_hp.py dnp): deflated Newton, then plain Newton steps from below 1e-12 if they lower |F|;
        returns zh, zl, |F|, steps, accepted (converged, or stalled below accept_stall)"""
        if opt["corrector"] == "qd":
            sq = PinnedQ(spx.s, spx.ch, spx.cl)
            z4, hist, conv = newton_qd(sq, qparts(ph, pl), tol=opt["tol_qd"], it_max=opt["qd_it"], w=opt["qd_w"], verbose=True)
            last_q["z4"] = z4
            Fh, Fl = spx.residual(z4[0], z4[1])
            Fc = float(rnorm(Fh, Fl))
            fq = float(hist[-1]["F"]) if hist else float("nan")
            print(f"      quad-double Newton: {len(hist)} steps, max|F| {fq:.1e}, {'converged' if conv else 'not converged'}; double-double |F| of its "
                  f"first two parts {Fc:.1e}", flush=True)
            return np.asarray(z4[0]), np.asarray(z4[1]), Fc, len(hist), bool(conv), 0.0
        if opt["corrector"] == "newton":
            ch, cl, hist = newton_solve(spx, ph, pl, tol=tol, it_max=12, verbose=False)
        else:
            ch, cl, hist = deflated_newton(spx, ph, pl, tol=tol, it_max=30, w=opt["w"], verbose=False, junk_only=junk_only)
            if opt["corrector"] == "dnp" and hist and hist[-1]["F"] < 1e-12:
                zh3, zl3, h3 = newton_solve(spx, ch, cl, tol=1e-40, it_max=4, verbose=False)
                if h3 and h3[-1]["F"] < hist[-1]["F"]:
                    ch, cl = zh3, zl3
                    hist = hist + h3
        Fh, Fl = spx.residual(ch, cl)
        Fc = float(rnorm(Fh, Fl))
        # a stall below accept_stall is accepted (T75's rule, at 1e-19 there; here 1e-17: the pinned horizon block's
        # floor after a prediction at h <= 1, 13.16), with the linearised Delta correction of the remaining residual
        # recorded as that point's Delta uncertainty
        dcorr = float(delta_correction(spx, Fh, Fl)) if Fc > tol and np.isfinite(Fc) else 0.0
        return ch, cl, Fc, len(hist), bool(Fc <= tol or Fc <= opt["accept_stall"]), dcorr
    mk = lambda Dx: make_solver(Dx, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=opt["threads"], kfreeze=kf)  # noqa: E731
    ks_held = [k for k in range(kf + 1, 14) if k % 2 == 1]
    base_coeffs = band_coeffs(zh[:M], zl[:M], M, ks_held)         # [(a_k, b_k)] at the base, exact there
    # the trend: exponents s of |v| ~ n^s per held coefficient, from the stored states of --trend (0 = held constant)
    slopes = [[0.0, 0.0] for _ in ks_held]
    if opt["trend"]:
        import glob
        ns, vals = [], []
        for f in sorted(glob.glob(os.path.join(raw, opt["trend"], "pin*_D*.npz")))[::4]:
            dd = np.load(f); mm = json.loads(str(dd["meta"]))
            if int(mm["M"]) != M:
                continue
            ns.append(float(mm["D"]) - 2.0)
            vals.append(band_coeffs(dd["zh"][:M], dd["zl"][:M], M, ks_held))
        ns = np.array(ns)
        for i, k in enumerate(ks_held):
            for j in (0, 1):
                v = np.array([c[i][j] for c in vals])
                if len(v) >= 6 and (np.all(v > 0) or np.all(v < 0)):
                    sl, c0 = np.polyfit(np.log(ns), np.log(np.abs(v)), 1)
                    res = float(np.max(np.abs(np.log(np.abs(v)) - (c0 + sl * np.log(ns)))))
                    if res < 0.05:
                        slopes[i][j] = float(sl)
        print("trend exponents (a, b) per held harmonic " + ", ".join(f"{k}: {sa:+.2f}/{sb:+.2f}" for k, (sa, sb) in zip(ks_held, slopes)), flush=True)
    n0 = float(D) - 2.0

    def held_at(Dx):
        """the held coefficients at D = Dx on the trend, relative to the base"""
        r = float(Dx) - 2.0
        return [(a * (r / n0) ** sa, b * (r / n0) ** sb) for (a, b), (sa, sb) in zip(base_coeffs, slopes)]

    def held_jet(j):
        """the order-j Taylor coefficients in h of the held values, v(D0) binom(s, j) n0^-j"""
        out = []
        for (a, b), (sa, sb) in zip(base_coeffs, slopes):
            ca = math.comb if False else None  # noqa: F841
            def binom(sx, jj):
                r = 1.0
                for t in range(jj):
                    r *= (sx - t) / (t + 1)
                return r
            out.append((a * binom(sa, j) / n0 ** j, b * binom(sb, j) / n0 ** j))
        return out

    def wrap_at(sx, Dx):
        hh, hl = band_field(held_at(Dx), ks_held, M)
        return PinnedSolver(sx, M, hh, hl)
    wrap = lambda sx: wrap_at(sx, D)  # noqa: E731
    s = mk(D); segs = s.plan(zh, zl, 1e-32, 0.05); s.set_segs(segs); sp = wrap(s)
    Kb = (len(zh) - 1) // (4 * M) - 1                                   # the horizon block
    rows = slice(Kb * 4 * M, Kb * 4 * M + M)
    # the held harmonics as directions to freeze in the quad-double Newton: unit band fields in the p1 block
    Fh, Fl = sp.residual(zh, zl)
    F0 = float(rnorm(Fh, Fl))
    if F0 > opt["tol"]:
        zh, zl, F0, _, _, _ = correct(sp, zh, zl, opt["tol"])
    print(f"base {opt['state']}: D {mp.nstr(D, 12)}, M {M}, pin {kf} (held {ks_held}), |F| {F0:.1e}; N {N}, eps {opt['eps']}, h_max {opt['h_max']}, "
          f"corrector {opt['corrector']} (junk_only {junk_only}), tol {opt['tol']:.1e} x {opt['tol_growth']}^(D - {mp.nstr(D_start, 8)}), "
          f"stalls accepted below {opt['accept_stall']:.0e}", flush=True)
    a = None
    for step in range(opt["steps"]):
        if D >= opt["D_max"]:
            break
        t0 = time.time()
        dh, dl, Fh, Fl, pmin, pmax = sp.newton(zh, zl)
        Zh, Zl, norms, refine = coefficients(s, zh, zl, N, held=(lambda j: band_field(held_jet(j), ks_held, M), rows) if any(any(sl) for sl in slopes) else None)
        t_coef = time.time() - t0
        a_max = (opt["eps"] * norms[1] / norms[N]) ** (1.0 / (N - 1)) if norms[N] else opt["h_max"]
        room = float(opt["D_max"] - D)
        a = min(a_max, opt["h_max"], room) if a is None else min(a, a_max, opt["h_max"], room)
        ratios = [norms[k - 1] / norms[k] if norms[k] else None for k in range(2, N + 1)]
        rec = dict(step=step, D0=float(D), D0_str=str(D), pin=kf, N=N, pivots=(float(pmin), float(pmax)), norms=norms, a_max=a_max,
                   t_coef=t_coef, refine_last=refine[-1], ratios_2_5=ratios[:4], trend=opt["trend"], slopes=slopes, held=ks_held)
        print(f"step {step}: D {mp.nstr(D, 12)}, pivots {pmin:.1e}..{pmax:.1e}, |z_1| {norms[1]:.2e}, |z_N| {norms[N]:.2e}, ratios 2-5 "
              + " ".join(f"{r:.1f}" for r in ratios[:4] if r) + f", a_max {a_max:.3f}, {t_coef:.0f} s", flush=True)
        converged = False
        for attempt in range(4):
            ph, pl = np.array(zh), np.array(zl)
            for k in range(1, N + 1):
                ph, pl = dd_add(ph, pl, *dd_scale(Zh[k], Zl[k], a ** k))
            Dn = D + mp.mpf(a)                 # exact: the series variable is a to double-double precision
            sn = mk(Dn)
            segs_n = segs
            sn.set_segs(segs_n)
            spn = wrap_at(sn, Dn)
            Fh, Fl = spn.residual(ph, pl)
            Fp = float(rnorm(Fh, Fl))
            if np.isfinite(Fp) and Fp <= 1e-6:
                segs_n = sn.plan(ph, pl, 1e-32, 0.05)
                sn.set_segs(segs_n)
                Fh, Fl = spn.residual(ph, pl); Fp = float(rnorm(Fh, Fl))
            t1 = time.time()
            tol = tol_at(Dn)
            if np.isfinite(Fp):
                ch, cl, Fc, nsteps, accepted, dcorr = correct(spn, ph, pl, tol)
            else:
                ch, cl, Fc, nsteps, accepted, dcorr = ph, pl, float("nan"), 0, False, 0.0
            t_corr = time.time() - t1
            fd = float(formula_delta(float(Dn)))
            att = dict(attempt=attempt, a=a, D=float(Dn), D_str=str(Dn), F_pred=Fp, F_final=Fc, tol=tol, steps=nsteps, accepted=accepted, t_corr=t_corr,
                       Delta=float(ch[-1] + cl[-1]), dDelta_formula=float(ch[-1] + cl[-1]) - fd, dDelta_corr=dcorr,
                       moved=float(dd_norm(*dd_add(ch, cl, -ph, -pl))))
            rec.setdefault("attempts", []).append(att)
            print(f"    a {a:.3f} -> D {mp.nstr(Dn, 12)}: predicted |F| {Fp:.2e}, {opt['corrector']} {nsteps} steps -> {Fc:.2e} (tol {tol:.1e}"
                  f"{', stall accepted, Delta correction ' + format(dcorr, '+.1e') if accepted and Fc > tol else ''}), "
                  f"Delta {att['Delta']:.12f} (formula diff {att['dDelta_formula']:+.1e}), "
                  f"moved {att['moved']:.1e}, {t_corr:.0f} s", flush=True)
            if accepted:
                converged = True
                break
            a *= 0.5
        rec["converged"] = converged
        with open(rec_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if not converged:
            print("stopped: no convergence after three halvings", flush=True)
            break
        D, zh, zl, s, sp, segs = Dn, ch, cl, sn, spn, segs_n
        # re-base the trend on the new state (its harmonics sit on the trend's values at D, to the corrector's floor)
        base_coeffs = band_coeffs(zh[:M], zl[:M], M, ks_held)
        n0 = float(D) - 2.0
        meta_out = dict(D=float(D), D_str=str(D), M=M, nodes=nodes, kt=kt, kc=kc, ks=ks, kfreeze=kf, pin_held=ks_held, N=N, a=a, source=opt["state"])
        if opt["corrector"] == "qd" and "z4" in last_q:
            save_q(os.path.join(out, f"anm_pin{kf}_D{float(D):.4f}.npz"), last_q["z4"], meta_out)
        else:
            save(os.path.join(out, f"anm_pin{kf}_D{float(D):.4f}.npz"), zh, zl, meta_out)
        a = min(2 * a, opt["h_max"])          # the next step may grow again
    print("done", flush=True)


if __name__ == "__main__":
    main()
