#!/usr/bin/env python3
"""Continuation in D with the double-double solver.

    python3 scan_hp.py OUT.jsonl START.npz D_END [--dD 0.5] [--M 48] [--kt 36] [--kc 40] [--ks 100]
                       [--nodes 0.02,0.05,...] [--tag name] [--maxdD 1.0]

START.npz: a converged solution (hpkernel.driver.save format: zh, zl, meta with D, M, nodes). Every converged point
is appended to OUT.jsonl and saved as <tag>_D<D>.npz next to OUT. A --nodes / --M different from the start file's
re-nodes / resamples the start solution first (and re-converges it at its own D).
"""
import argparse
import json
import os
import time

# the f64 linear algebra here is small (QR of L x w); BLAS threads would only compete with the kernel's thread pool
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from hpkernel.driver import QdhSolver, deflated_newton, delta_correction, hermite, lm_solve, load, make_solver, newton_solve, plow_odd, predict, renode, resample_z, save, tangent
from pinning import PinnedSolver, band_coeffs, band_field, predict_dd, dd_add


def tails(s, zh, zl, nodes, M):
    """max over fields/lanes of |c_k| x^k for the last 3 orders: centre series at nodes[0], horizon series at
    u = 1 - nodes[-1]"""
    out = []
    for which, x in ((0, nodes[0]), (1, 1 - nodes[-1])):
        h, l = s.series(zh, zl, which)
        c = np.abs((h + l).reshape(-1, 4, M)).max(axis=(1, 2))
        k = np.arange(len(c))
        out.append(float((c * x ** k)[-3:].max()))
    return out


def kc_needed(s, zh, zl, x0, M, kc, margin):
    """The centre-series order past which the terms |c_k| x0^k (largest over fields and tau points) stay below 1e-45 of
    the largest for three orders, plus margin, at most kc (hp/kc_check.py; with margin 30 the residual, the Jacobian and
    the Newton step are bitwise those of kc = 420 at D = 36-61, and the double-double Jacobian 2-4 times cheaper)."""
    h, l = s.series(zh, zl, 0)
    c = np.abs((np.asarray(h) + np.asarray(l)).reshape(-1, 4, M)).max(axis=(1, 2))
    t = c * x0 ** np.arange(len(c))
    for k in range(20, len(t) - 3):
        if (t[k:k + 3] < 1e-45 * t.max()).all():
            return min(k + margin, kc)
    return kc


def same_bits(a, b):
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def sensitivity(s, M):
    """dDelta/dF (needs the LU of the last Newton step): max |y| per residual block"""
    L = s.size
    e = np.zeros(L)
    e[-1] = 1.0
    yh, yl = s.lu_solve_t(e, np.zeros(L))
    nb = (L - 1) // (4 * M)
    return [float(np.abs(yh[b * 4 * M:(b + 1) * 4 * M]).max()) for b in range(nb)]


def certify(D, zh, zl, s, a, nodes, M):
    """the converged point (zh, zl) checked in the layout a.certify_nodes without a Newton step: the state cut to the
    layout (driver.renode; the node states it keeps, the centre and horizon data and Delta are unchanged), its residual
    there, and from that layout's double-double Jacobian and LU the sensitivity max|dDelta/dF| per block, the
    linearised Delta correction -y.F and the first-order bound sum|y_i F_i| (y = J^-T e_Delta)"""
    cn = [float(v) for v in a.certify_nodes.split(",")]
    if any(x not in nodes for x in cn):
        raise SystemExit(f"--certify_nodes {cn} must be a subset of the run's nodes {nodes}")
    t0 = time.time()
    ch, cl = renode(s, zh, zl, nodes, cn, M)
    sc = make_solver(D, M, nodes=cn, kt=a.kt, kc=a.kc, ks=a.certify_ks, threads=a.threads, kfreeze=a.kfreeze,
                     horizon_alpha=a.horizon_alpha)
    sc.set_segs(sc.plan(ch, cl, a.tol_step, a.hmax))
    sc.newton(ch, cl)                       # Jacobian and LU at the point; the step is not taken
    Fh, Fl = sc.residual(ch, cl)
    F = np.asarray(Fh) + np.asarray(Fl)
    L = sc.size
    e = np.zeros(L)
    e[-1] = 1.0
    yh, yl = sc.lu_solve_t(e, np.zeros(L))
    y = np.asarray(yh) + np.asarray(yl)
    _, th = tails(sc, ch, cl, cn, M)
    return dict(nodes=cn, ks=a.certify_ks, F=float(np.abs(F).max()), sens=max(sensitivity(sc, M)),
                dDelta_est=delta_correction(sc, Fh, Fl), bound=float(np.sum(np.abs(y * F))), tail_h=th,
                seconds=round(time.time() - t0, 1))


def freeze_p1(zh, zl, a, M):
    """p1's harmonics above kfreeze set to zero (they are pinned there by the frozen formulation)"""
    if not a.kfreeze:
        return zh, zl
    lo, lo2 = plow_odd(M, a.kfreeze)
    zh = zh.copy()
    zl = zl.copy()
    zh[:M] = (lo + lo2) @ (zh[:M] + zl[:M])
    zl[:M] = 0.0
    return zh, zl


def gmres_opts(a):
    """options of driver.gmres_refine from --step_tol / --max_products (driver.GMRES defaults otherwise)"""
    return {k: v for k, v in (("step_tol", a.step_tol), ("max_products", a.max_products)) if v is not None}


def step_split(hist):
    """seconds of the Newton steps by part (residual, Jacobian, LU, solves, products) and the products, summed over a
    point's history (driver.newton_step's records)"""
    out = {k: round(sum(r.get(k, 0.0) for r in hist if r.get(k) == r.get(k)), 3) for k in ("t_res", "t_jac", "t_lu", "t_solve", "t_prod")}
    out["products"] = sum(r.get("products", 0) for r in hist)
    return out


def accept_stall(a):
    """--accept_stall F_MAX,DDELTA_MAX as a pair, or None"""
    if not a.accept_stall:
        return None
    f, d = (float(v) for v in a.accept_stall.split(","))
    return f, d


def solve_at(D, zh, zl, a, nodes, M, it_max, pin=None):
    """pin: None, or (coeffs, ks) - p1's harmonics ks (all in (a.pin, a.kfreeze]) held at the given cos/sin
    coefficients (PinnedSolver), the rest solved"""
    s = make_solver(D, M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads,
                    kfreeze=a.pin if pin is not None else a.kfreeze, horizon_alpha=a.horizon_alpha)
    s.set_segs(s.plan(zh, zl, a.tol_step, a.hmax))
    if a.qdh:
        if a.jac != "dd":
            raise SystemExit("--qdh needs --jac dd")
        s = QdhSolver(s)
    if pin is not None:
        # outside QdhSolver: the pin is an offset on the residual, whichever precision the horizon block has
        hh, hl = band_field(pin[0], pin[1], M)
        s = PinnedSolver(s, M, hh, hl)
    t0 = time.time()
    jkw = {} if a.jac == "dd" else dict(jac=a.jac, gmres=gmres_opts(a), dd_last=a.dd_last)
    if a.corrector in ("dn", "dnp"):
        zh2, zl2, hist = deflated_newton(s, zh, zl, tol=a.tol, it_max=it_max, w=a.w, verbose=a.verbose,
                                         junk_only=a.junk_only, delta_tol=a.delta_tol, accept_stall=accept_stall(a), **jkw)
    else:
        zh2, zl2, hist = newton_solve(s, zh, zl, tol=a.tol, it_max=it_max, verbose=a.verbose, **jkw)
    if a.corrector == "dnp" and hist and hist[-1]["F"] < a.polish_from:
        # polish: plain Newton to the bottom (deflated Newton stops at the first iterate under tol, or stalls just
        # above it, and leaves that point's error in the predictor's history)
        zh3, zl3, h3 = newton_solve(s, zh2, zl2, tol=1e-40, it_max=a.polish_its, verbose=a.verbose, **jkw)
        if h3 and h3[-1]["F"] < hist[-1]["F"]:
            zh2, zl2 = zh3, zl3
            hist = hist + [dict(r, polish=True) for r in h3]
    # accepted at a stall (--accept_stall), possibly polished below that point's |F| afterwards
    stall_ok = any(r.get("accept_stall") and r.get("accepted") for r in hist)
    ok = bool(hist) and (hist[-1]["F"] < a.tol or bool(hist[-1].get("accepted")) or stall_ok)
    if not ok and a.lm:      # (the wrappers pass their residual, pin offset and quad-double horizon, to lm_steps)
        print(f"   D={D}: Newton stopped at |F| {hist[-1]['F']:.1e}; Levenberg-Marquardt from the predictor", flush=True)
        zh2, zl2, h2 = lm_solve(s, zh, zl, tol=a.tol, it_max=a.lm, verbose=a.verbose, newton_after=1e-22)
        hist = hist + h2
        ok = hist[-1]["F"] < a.tol
    return s, zh2, zl2, hist, ok, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("start")
    ap.add_argument("D_end", type=float)
    ap.add_argument("--dD", type=float, default=0.5)
    ap.add_argument("--maxdD", type=float, default=1.0)
    ap.add_argument("--dD0", type=float, default=None, help="first step (default dD)")
    ap.add_argument("--history", default="", help="comma list of converged npz files at earlier D (same M), oldest first: "
                    "re-noded to the scan's layout and used for the predictor")
    ap.add_argument("--predictor", default="lagrange", choices=["lagrange", "lagrange_dd", "hermite"],
                    help="lagrange: extrapolation through the last 4 converged points (default; the J-based tangent "
                    "is unreliable along near-null directions at larger D); lagrange_dd: the same in double-double "
                    "(the f64 version rounds z at 1e-16, which the horizon block amplifies to |F| ~ 2e-7 at D ~ 61 "
                    "even for dD = 0, and limits the step); hermite: exact tangent + cubic Hermite")
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--kt", type=int, default=36)
    ap.add_argument("--kc", type=int, default=40)
    ap.add_argument("--kc_auto", type=int, default=0, help="lower kc after a converged point to the order where the "
                    "centre series' terms stay below 1e-45, plus this margin (30 is bitwise safe at D = 36-61), if the "
                    "residual and the Jacobian there are bitwise unchanged; 0 (default) keeps kc as given (it is "
                    "still raised when the tail check asks)")
    ap.add_argument("--ks", type=int, default=100)
    ap.add_argument("--nodes", default=None)
    ap.add_argument("--tag", default="hp")
    ap.add_argument("--tol", type=float, default=1e-27)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--it_max", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--kfreeze", type=int, default=None, help="pin p1 harmonics above this to zero and drop the matching of "
                    "p's harmonics above it at the horizon block (removes the near-null centre directions)")
    ap.add_argument("--horizon_alpha", type=int, default=0, choices=[0, 1, 2],
                    help="the horizon series in alpha = a - 1 (1; a lower residual floor at large D, the same solution; "
                    "notes/LARGE_D_MIXED.md section 10), 2 its control; 0 (default) as before")
    ap.add_argument("--accept_stall", default="", help="F_MAX,DDELTA_MAX: accept a point whose deflated Newton ends "
                    "above --tol if |F| <= F_MAX and its linearised Delta correction is at most DDELTA_MAX in size "
                    "(driver.deflated_newton accept_stall)")
    ap.add_argument("--qdh", type=int, default=0, choices=[0, 1],
                    help="1: the residual's horizon block in quad-double (driver.QdhSolver, notes/QUAD_DOUBLE.md section "
                         "11): the double-double residual floor of large D removed; needs --jac dd")
    ap.add_argument("--corrector", default="dn", choices=["dn", "dnp", "newton"],
                    help="dn: deflated Newton (default); dnp: deflated Newton, then plain Newton to the bottom")
    ap.add_argument("--polish_its", type=int, default=4, help="dnp: plain Newton steps after the deflated phase")
    ap.add_argument("--polish_from", type=float, default=1e-12, help="dnp: polish only from |F| below this")
    ap.add_argument("--w", type=int, default=24, help="weak directions frozen in the deflated phase")
    ap.add_argument("--delta_tol", type=float, default=1e-14, help="junk_only: accept a stalled step if the linearised "
                    "Delta correction is below this")
    ap.add_argument("--junk_hold", type=int, default=None, help="predictor: carry p1 harmonics above this over from the "
                    "last point instead of extrapolating them")
    ap.add_argument("--junk_trend", type=int, default=0, help="with junk_hold: put the held harmonics on a least-squares "
                    "line through the last N points instead of the last value")
    ap.add_argument("--junk_only", action="store_true", help="freeze only weak directions without Delta content, and keep "
                    "them frozen (they stay on the predictor's smooth trend)")
    ap.add_argument("--pin", type=int, default=None, help="with --kfreeze K: hold p1's harmonics in (pin, K] on a "
                    "polynomial trend in D instead of solving them (they are the weakest directions); after each "
                    "point one full-system Newton step measures their values for the trend and the pin's Delta error")
    ap.add_argument("--pin_from", default="trend", choices=["trend", "seed"], help="--pin: where the held values come "
                    "from: trend (default), the polynomial trend through values measured by a full-system step after "
                    "each point; seed, the predictor's seed itself (at D 67.74 this converges from seeds where the "
                    "trend and the unpinned corrector stall; hp/corrector_diag.py, T71)")
    ap.add_argument("--pin_deg", type=int, default=1, help="--pin: degree of the trend")
    ap.add_argument("--pin_n", type=int, default=6, help="--pin: number of measured points in the trend fit")
    ap.add_argument("--lm", type=int, default=15, help="Levenberg-Marquardt iterations after a failed Newton (0: off)")
    ap.add_argument("--Dlist", default="", help="explicit D values to visit in order (comma list or @file); "
                    "D_end is then ignored")
    ap.add_argument("--jac", default="dd", choices=["dd", "f64", "f64h", "f64g", "f64hg"],
                    help="Newton step's Jacobian (driver.newton_step; notes/LARGE_D_MIXED.md): dd exact (default); f64 "
                    "mixed precision; f64h with the horizon block in double-double; f64g / f64hg with the GMRES "
                    "correction loop (exact products); the sensitivities then come from the step's f64 LU")
    ap.add_argument("--step_tol", type=float, default=None, help="f64g / f64hg: the loop's step-change tolerance")
    ap.add_argument("--max_products", type=int, default=None, help="f64g / f64hg: the loop's product budget per step")
    ap.add_argument("--dd_last", action="store_true", help="mixed --jac: one exact Newton step once |F| < tol "
                    "(driver.exact_last_step: Delta to the exact Newton's accuracy at D ~ 60, and the exact LU for the "
                    "sensitivities)")
    ap.add_argument("--schedule", default="", help="node changes during the scan: 'D1=x,x,...;D2=x,x,...' "
                    "(at the first converged point with D >= D_i the solution is re-noded and re-converged)")
    # step control (hp/step_test.py); the defaults are the scans' rules
    ap.add_argument("--pred_n", type=int, default=4, help="--predictor lagrange_dd: points in the extrapolation "
                    "(default 4: cubic)")
    ap.add_argument("--grow_its", type=int, default=8, help="grow dD after a point that took at most this many Newton "
                    "iterations, polish excluded (default 8; a large value grows after every converged point)")
    ap.add_argument("--grow", type=float, default=1.25, help="dD growth factor")
    ap.add_argument("--fail_factor", type=float, default=0.5, help="dD factor after a failed step")
    ap.add_argument("--min_dD", type=float, default=1e-3, help="stop when dD falls below this")
    ap.add_argument("--tol_growth", type=float, default=1.0, help="multiply --tol by this per unit of D from the start "
                    "(the residual floor rises about 1.4-1.46x per unit at D = 45-61); 1 (default): fixed tol")
    ap.add_argument("--max_points", type=int, default=0, help="stop after this many converged points (0: no limit)")
    ap.add_argument("--max_hours", type=float, default=0.0, help="stop when this much wall time has passed (0: no limit)")
    ap.add_argument("--max_span", type=float, default=0.0, help="stop when D is this far from the start (0: no limit)")
    ap.add_argument("--max_fail_row", type=int, default=0, help="stop after this many failed steps in a row (0: no limit)")
    ap.add_argument("--attempts", default="", help="jsonl: one record per attempted step, converged or not, and a final "
                    "record with the reason the scan ended")
    ap.add_argument("--certify_nodes", default="", help="check each converged point in this node layout too (a subset of "
                    "the run's nodes, e.g. 0.3,0.4,0.5,0.6): the state cut to it, its residual there and that layout's "
                    "dDelta/dF, with no Newton step (Newton does not converge at 0.6 matching at D ~ 60, but the 0.7 "
                    "solution sits at the 0.6 floor; comms 2026-10-23 12:43 local). Recorded as 'cert'")
    ap.add_argument("--certify_ks", type=int, default=160, help="--certify_nodes: horizon series order there")
    a = ap.parse_args()
    T_start = time.time()
    zh, zl, meta = load(a.start)
    D, M0, nodes0 = meta["D"], meta["M"], list(meta["nodes"])
    M = a.M or M0
    nodes = [float(v) for v in a.nodes.split(",")] if a.nodes else nodes0
    if M != M0:
        zh, zl = resample_z(zh, M0, M, len(nodes0))
    if nodes != nodes0:
        s0 = make_solver(D, M, nodes=nodes0, kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads)
        zh, zl = renode(s0, zh, zl, nodes0, nodes, M)
    zh, zl = freeze_p1(zh, zl, a, M)
    s, zh, zl, hist, ok, dt = solve_at(D, zh, zl, a, nodes, M, a.it_max)
    print(f"(re-)converged start D={D} M={M} nodes={nodes}: Delta {zh[-1]:.16f} ok {ok} ({dt:.0f}s)", flush=True)
    if not ok:
        raise SystemExit("start did not converge")
    sched = []
    for item in filter(None, a.schedule.split(";")):
        d, xs = item.split("=")
        sched.append((float(d), [float(v) for v in xs.split(",")]))
    sched.sort()
    outdir = os.path.dirname(os.path.abspath(a.out))
    s_of = lambda Dx: make_solver(Dx, M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads, kfreeze=a.kfreeze)
    hist_z = []
    hist_dd = []        # (D, zh, zl) of the converged points, for --predictor lagrange_dd
    pin_meas = []       # (D, cos/sin coefficients of p1's pinned band), measured by a full-system Newton step
    for f in filter(None, a.history.split(",")):
        hh, hl, hm = load(f)
        if hm["M"] != M:
            raise SystemExit(f"{f}: M {hm['M']} != {M}")
        if list(hm["nodes"]) != nodes:
            s0 = make_solver(hm["D"], M, nodes=list(hm["nodes"]), kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads)
            hh, hl = renode(s0, hh, hl, list(hm["nodes"]), nodes, M)
        hh, hl = freeze_p1(hh, hl, a, M)
        sh, hh, hl, hst, okh, _ = solve_at(hm["D"], hh, hl, a, nodes, M, a.it_max)
        print(f"history point D={hm['D']}: ok {okh}, Delta {hh[-1]:.16f}", flush=True)
        if okh:
            hist_z.append((hm["D"], hh + hl, None))
            hist_dd.append((hm["D"], hh, hl))
            if a.pin is not None and a.kfreeze and a.pin < a.kfreeze:
                pin_meas.append((hm["D"], np.ravel(band_coeffs(hh[:M], hl[:M], M, list(range(a.pin + 2, a.kfreeze + 1, 2))))))
    t0_ = tangent(s_of, s, D, zh, zl) if a.predictor == "hermite" or not hist_z else None
    hist_z.append((D, zh + zl, t0_))
    hist_dd.append((D, zh, zl))
    if a.pin is not None:
        if not a.kfreeze or a.pin >= a.kfreeze:
            raise SystemExit("--pin needs --kfreeze K with pin < K")
        pin_ks = list(range(a.pin + 2, a.kfreeze + 1, 2))
        pin_meas.append((D, np.ravel(band_coeffs(zh[:M], zl[:M], M, pin_ks))))
    dD = a.dD0 or a.dD
    targets = []
    if a.Dlist:
        txt = open(a.Dlist[1:]).read() if a.Dlist.startswith("@") else a.Dlist
        targets = [float(v) for v in txt.replace("\n", ",").split(",") if v.strip()]
        a.D_end = targets[-1]
    direction = 1.0 if a.D_end > D else -1.0
    D_start, tol0, n_ok, fail_row = D, a.tol, 0, 0

    def attempt(**kw):
        if a.attempts:
            with open(a.attempts, "a") as f:
                f.write(json.dumps(dict(kw, t_wall=round(time.time() - T_start, 1))) + "\n")

    end_reason = "D_end reached"
    while (a.D_end - D) * direction > 1e-12:
        if a.max_hours and time.time() - T_start > 3600 * a.max_hours:
            end_reason = f"max_hours {a.max_hours}"
            break
        while targets and (targets[0] - D) * direction <= 1e-12:
            targets.pop(0)
        goal = targets[0] if targets else a.D_end
        Dn = D + direction * min(dD, abs(goal - D))
        Dn = round(Dn, 6)
        zpl = None
        if a.predictor == "lagrange_dd" and len(hist_dd) > 1:
            zp, zpl = predict_dd(hist_dd, Dn, a.pred_n)
        elif a.predictor in ("lagrange", "lagrange_dd") and len(hist_z) > 1:
            zp = predict([(h_[0], h_[1]) for h_ in hist_z], Dn)
            if a.junk_hold:
                # p1 harmonics above junk_hold: carried over from the last point (they are frozen junk; extrapolating
                # them compounds their noise from point to point)
                lo, lo2 = plow_odd(M, a.junk_hold)
                P = lo + lo2
                zlast = hist_z[-1][1]
                zp = zp.copy()
                held = zlast[:M] - P @ zlast[:M]
                if a.junk_trend and len(hist_z) >= 3:
                    # least-squares line through the held harmonics of the last junk_trend points: their values
                    # jitter from point to point at |F| / sigma, and a straight fit averages that out
                    pts_ = hist_z[-a.junk_trend:]
                    Dh_ = np.array([h_[0] for h_ in pts_])
                    Y = np.array([h_[1][:M] - P @ h_[1][:M] for h_ in pts_])
                    A_ = np.vander(Dh_ - Dh_[-1], 2)
                    coef, *_ = np.linalg.lstsq(A_, Y, rcond=None)
                    held = coef[0] * (Dn - Dh_[-1]) + coef[1]
                zp[:M] = P @ zp[:M] + held
        elif len(hist_z) > 1 and hist_z[-2][2] is not None:
            (D0, z0, t0), (D1, z1, t1) = hist_z[-2], hist_z[-1]
            zp = hermite(D0, z0, t0, D1, z1, t1, Dn)
        else:
            D1, z1, t1 = hist_z[-1]
            zp = z1 + (Dn - D1) * t1
        if zpl is None:
            zpl = np.zeros_like(zp)
        pin = None
        if a.pin is not None and a.pin_from == "seed":
            # the pinned band held at the seed's: the predictor's extrapolation of the converged points' own band
            # (with lagrange_dd, Lagrange through their pinned values), so no full-system step feeds the trend
            cz = np.ravel(band_coeffs(zp[:M], zpl[:M], M, pin_ks))
            pin = ([(cz[2 * i], cz[2 * i + 1]) for i in range(len(pin_ks))], pin_ks)
        elif a.pin is not None:
            # the pinned band on a polynomial trend through the last pin_n measured points (constant at first)
            Dm = np.array([q[0] for q in pin_meas[-a.pin_n:]])
            Cm = np.array([q[1] for q in pin_meas[-a.pin_n:]])
            deg = min(a.pin_deg, len(Dm) - 1)
            cn = np.polyval(np.polyfit(Dm - Dm[-1], Cm, deg), Dn - Dm[-1]) if deg > 0 else Cm[-1]
            co = [(cn[2 * i], cn[2 * i + 1]) for i in range(len(pin_ks))]
            cz = np.ravel(band_coeffs(zp[:M], zpl[:M], M, pin_ks))
            bh, bl = band_field([(co[i][0] - cz[2 * i], co[i][1] - cz[2 * i + 1]) for i in range(len(pin_ks))], pin_ks, M)
            zp = zp.copy(); zpl = zpl.copy()
            zp[:M], zpl[:M] = dd_add(zp[:M], zpl[:M], bh, bl)
            pin = (co, pin_ks)
        a.tol = tol0 * a.tol_growth ** abs(Dn - D_start)
        s, zh2, zl2, hist, ok, dt = solve_at(Dn, zp, zpl, a, nodes, M, a.it_max, pin=pin)
        attempt(D_from=D, D_to=Dn, dD=abs(Dn - D), ok=ok, its=len(hist), polish=sum(1 for r in hist if r.get("polish")),
                F_seed=hist[0]["F"] if hist else None, F_last=hist[-1]["F"] if hist else None, tol=a.tol,
                seconds=round(dt, 1), Delta=float(zh2[-1]) if ok else None, Delta_lo=float(zl2[-1]) if ok else None)
        if not ok:
            fail_row += 1
            print(f"D={Dn}: no convergence (|F| {hist[-1]['F'] if hist else float('nan'):.1e}); dD {dD} -> "
                  f"{dD * a.fail_factor}", flush=True)
            dD *= a.fail_factor
            if dD < a.min_dD:
                attempt(end=f"step below min_dD {a.min_dD}", n_ok=n_ok, D=D)
                raise SystemExit("step too small")
            if a.max_fail_row and fail_row >= a.max_fail_row:
                end_reason = f"{fail_row} failed steps in a row"
                break
            continue
        n_ok += 1
        fail_row = 0
        D, zh, zl = Dn, zh2, zl2
        hist_z.append((D, zh + zl, tangent(s_of, s, D, zh, zl) if a.predictor == "hermite" else None))
        hist_dd.append((D, zh, zl))
        pin_rec = {}
        if pin is not None:
            # one full-system (kfreeze) Newton step from the pinned solution: its residual and linearised Delta
            # correction measure what the pin costs; the stepped point's band is the measurement for the trend
            sf = make_solver(D, M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads, kfreeze=a.kfreeze)
            sf.set_segs(s.get_segs())
            if a.qdh:
                sf = QdhSolver(sf)
            fdh, fdl, fFh, fFl, _, _ = sf.newton(zh, zl)
            zmh, zml = dd_add(zh, zl, np.asarray(fdh), np.asarray(fdl))
            pin_meas.append((D, np.ravel(band_coeffs(zmh[:M], zml[:M], M, pin_ks))))
            pin_rec = dict(pin=a.pin, F_full=float(np.abs(np.asarray(fFh) + np.asarray(fFl)).max()),
                           dDelta_pin=delta_correction(sf, fFh, fFl), Delta_full1=float(zmh[-1]), Delta_full1_lo=float(zml[-1]))
        # the stored LU is from the last Newton call, made at the converged z
        sens = sensitivity(s, M)
        tc, th = tails(s, zh, zl, nodes, M)
        rec = dict(D=D, Delta=float(zh[-1]), Delta_lo=float(zl[-1]), F=hist[-1]["F"], its=len(hist), seconds=round(dt, 1),
                   M=M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, tol_step=a.tol_step, steps=sum(len(g) - 1 for g in s.get_segs()),
                   piv=hist[0].get("piv"), sens=sens, tail_c=tc, tail_h=th, dDelta_est=hist[-1].get("dDelta_est"),
                   tol=a.tol, kfreeze=a.kfreeze, corrector=a.corrector)
        if rec["dDelta_est"] is None:
            rec["dDelta_est"] = delta_correction(s, *s.residual(zh, zl))
        rec["polish"] = sum(1 for r in hist if r.get("polish"))
        if a.qdh:
            z2, z3 = s.parts(zh, zl)
            rec.update(qdh=1, Delta_q2=float(z2[-1]))
        if a.jac != "dd":
            # sens_lu: the LU the sensitivities came from (the last step's: f64 for a mixed step, dd after dd_finish
            # switched); at D >= 50 an f64 LU's dDelta/dF is wrong by orders of magnitude (notes/LARGE_D_MIXED.md)
            rec.update(jac=a.jac, split=step_split(hist), sens_lu=hist[-1].get("jac", a.jac),
                       switched_dd=any(r.get("switch_dd") for r in hist))
        rec.update(pin_rec)
        if a.certify_nodes:
            rec["cert"] = certify(D, zh, zl, s, a, nodes, M)
            c = rec["cert"]
            print(f"   certified at nodes {c['nodes']}: |F| {c['F']:.1e}, dDelta/dF max {c['sens']:.1e}, "
                  f"dDelta_est {c['dDelta_est']:+.1e}, bound sum|y F| {c['bound']:.1e} (here: {rec['F'] * max(sens):.1e} "
                  f"as |F| x max dDelta/dF)", flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        save(os.path.join(outdir, f"{a.tag}_D{D:.4f}.npz"), zh, zl, dict(D=D, M=M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks),
             parts=s.parts(zh, zl) if a.qdh else None)
        print(f"D={D:.4f} Delta={zh[-1]:.16f}{zl[-1]:+.2e} its {len(hist)} |F| {hist[-1]['F']:.1e} {dt:.0f}s "
              f"piv {rec['piv'][0]:.1e}..{rec['piv'][1]:.1e} dDelta/dF max {max(sens):.1e} tails {tc:.0e} {th:.0e}"
              + (f" pin: |F_full| {pin_rec['F_full']:.1e} dDelta_pin {pin_rec['dDelta_pin']:+.1e}" if pin_rec else ""), flush=True)
        if tc > 1e-33:
            a.kc = int(a.kc * 1.25) + 2
            print(f"   centre series tail {tc:.1e}: kc -> {a.kc}", flush=True)
        elif a.kc_auto:
            k_need = kc_needed(s, zh, zl, nodes[0], M, a.kc, a.kc_auto)
            if k_need < a.kc - 10:      # hysteresis: check (two Jacobians) only when it saves at least 10 orders
                s_new = make_solver(D, M, nodes=nodes, kt=a.kt, kc=k_need, ks=a.ks, threads=a.threads,
                                    kfreeze=a.kfreeze)
                s_new.set_segs(s.get_segs())
                if same_bits(s.residual(zh, zl), s_new.residual(zh, zl)) and \
                        same_bits(s.jacobian(zh, zl), s_new.jacobian(zh, zl)):
                    print(f"   kc {a.kc} -> {k_need} (residual and Jacobian bitwise unchanged)", flush=True)
                    a.kc = k_need
                else:
                    print(f"   kc stays {a.kc}: kc {k_need} changes the residual or the Jacobian", flush=True)
        if th > 1e-33:
            a.ks = int(a.ks * 1.25) + 2
            print(f"   horizon series tail {th:.1e}: ks -> {a.ks}", flush=True)
        its_main = len(hist) - rec["polish"]
        if its_main <= a.grow_its:
            dD = min(a.maxdD, dD * a.grow)
        elif its_main >= 14:
            dD *= 0.8
        if a.max_points and n_ok >= a.max_points:
            end_reason = f"max_points {a.max_points}"
            break
        if a.max_span and abs(D - D_start) >= a.max_span - 1e-12:
            end_reason = f"max_span {a.max_span}"
            break
        while sched and (D - sched[0][0]) * direction >= -1e-9:
            new_nodes = sched.pop(0)[1]
            if new_nodes == nodes:
                continue
            zh, zl = renode(s, zh, zl, nodes, new_nodes, M)
            s, zh, zl, hist, ok, dt = solve_at(D, zh, zl, a, new_nodes, M, a.it_max + 4)
            print(f"re-noded at D={D}: {nodes} -> {new_nodes}: Delta {zh[-1]:.16f} ok {ok} ({dt:.0f}s)", flush=True)
            if not ok:
                raise SystemExit("re-noding did not converge")
            with open(a.out, "a") as f:
                f.write(json.dumps(dict(D=D, Delta=float(zh[-1]), Delta_lo=float(zl[-1]), F=hist[-1]["F"], its=len(hist),
                                        M=M, nodes=new_nodes, kt=a.kt, kc=a.kc, ks=a.ks, renoded=True,
                                        sens=sensitivity(s, M), tail_c=tails(s, zh, zl, new_nodes, M)[0],
                                        tail_h=tails(s, zh, zl, new_nodes, M)[1])) + "\n")
            nodes = new_nodes
            s_of = lambda Dx: make_solver(Dx, M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads, kfreeze=a.kfreeze)
            hist_z = [(D, zh + zl, tangent(s_of, s, D, zh, zl))]   # one point: Euler with the tangent
            hist_dd = [(D, zh, zl)]                                 # (the old layout's points cannot be mixed in)
            dD = min(dD, 0.5)
    attempt(end=end_reason, n_ok=n_ok, D=D)
    print(f"scan ended: {end_reason} ({n_ok} converged points, D = {D})", flush=True)


if __name__ == "__main__":
    main()
