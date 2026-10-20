#!/usr/bin/env python3
"""The critical exponent gamma(D) = 1/lam at double-double precision, from saved double-double states.

lam is the growth rate of the unstable perturbation e^{lam tau} (periodic) of the DSS solution: the root of
det M(lam), where M(lam) = Solver.mode_matrix is the exact Jacobian of the multiple-shooting residual for
perturbations (every tau-derivative of a perturbation gains + lam, Delta held fixed, no phase row), built by
forward-mode duals in double-double (hp/src/model.rs: Ctx::lam, dtau, psolve with the lam-shifted horizon LUs).
The f64 solver does the same with mode_matrix / gamma_root.py; this is its double-double version.

For each state the solver is rebuilt, with the kfreeze-13 cut from D = 35 on (as in the runs; without the cut, weak
directions bias the root by up to 2e-9 at D = 61), --kf=none|13|auto overrides (auto: the layout with the smaller
residual); det M is bracketed around the rational formula's lam = 1/gamma(D)
and the sign change located with brentq on sign * exp(ln|det| - ref). --polish[=N] first re-converges the state in the
rebuilt layout (up to N Newton steps, default 4; the up6 states at D = 27-35 come down from |F| ~ 1e-12 to 1e-25, and lam does
not change). States whose residual still exceeds --fmax (default 1e-20) are skipped and recorded as such.

    python3 gamma_dd.py out.jsonl STATE.npz [STATE.npz ...] [--threads=N] [--M48 | --Mhi=64] [--kf=none|13|auto] [--fmax=1e-20] [--polish[=N]]

--M48 (or --Mhi=N) also resamples the state to M = 48 (N) tau points, re-converges it (Newton, then Levenberg-Marquardt
if Newton stalls) and repeats the root search, bracketed afresh from the formula (self-convergence; the record's M48
field). The M = 48 solve uses the kfreeze-13 cut from D = 35 on (without it Newton from the resampled state stalls at
D = 40), with the resampled p1 projected onto the cut first; --kf48=none|13 overrides.
Needs the hpkernel wheel built from this tree (hp/build.sh) for mode_det.
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import make_solver, load, lm_solve, newton_solve, plow_odd, resample_z  # noqa: E402

# README rational formula for gamma (solver/formula_all.py), used only for the starting guess
E = [0.5, 1.8083629635e1, 1.5241588758e2, -5.2341588541e1, 9.9259594505, -9.8386450353e1, 7.4529506074e1,
     -4.8272358903e1]
FQ = [1, 3.7683686025e1, 3.4245505235e2]


def gamma_formula(D):
    x = 1.0 / (D - 2)
    return sum(c * x ** i for i, c in enumerate(E)) / sum(c * x ** i for i, c in enumerate(FQ))


def build(meta, zh, zl, threads, M=None, kfreeze=None):
    s = make_solver(meta["D"], M or meta["M"], nodes=meta["nodes"], kt=meta.get("kt", 36), kc=meta.get("kc", 420),
                    ks=meta.get("ks", 100), threads=threads, kfreeze=kfreeze)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    fh, fl = s.residual(zh, zl)
    return s, float(np.abs(np.asarray(fh) + np.asarray(fl)).max())


def best_solver(meta, zh, zl, threads, M=None, force=None):
    kfs = (None, 13) if force in (None, "auto") else ((None,) if force == "none" else (int(force),))
    cands = [(build(meta, zh, zl, threads, M, kf), kf) for kf in kfs]
    (s, F), kf = min(cands, key=lambda c: c[0][1])
    return s, F, kf


def find_lam(s, zh, zl, lam0, log=None):
    cache = {}

    def sd(lam):
        if lam not in cache:
            t = time.time()
            sg, la, pmin, pmax = s.mode_det(zh, zl, float(lam), 0.0)
            cache[lam] = (sg, la)
            if log:
                log(f"    lam {lam:.12f}: sign {sg:+.0f} ln|det| {la:.6f} pivots {pmin:.2e}..{pmax:.2e} ({time.time() - t:.1f} s)")
        return cache[lam]

    for w in (3e-4, 2e-3, 1e-2, 4e-2):
        grid = [lam0 - w, lam0, lam0 + w]
        sg = [sd(l)[0] for l in grid]
        ch = [i for i in range(2) if sg[i] != sg[i + 1]]
        if ch:
            i = min(ch, key=lambda i: abs(0.5 * (grid[i] + grid[i + 1]) - lam0))
            a, b = grid[i], grid[i + 1]
            ref = 0.5 * (sd(a)[1] + sd(b)[1])
            f = lambda l: sd(l)[0] * math.exp(sd(l)[1] - ref)
            lam = brentq(f, a, b, xtol=1e-14, rtol=1e-15, maxiter=60)
            return lam, len(ch), len(cache)
    return None, 0, len(cache)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    threads = int(opts.get("threads", 1))
    out = args[0]
    log = lambda m: print(m, flush=True)
    for f in args[1:]:
        zh, zl, meta = load(f)
        D = meta["D"]
        t0 = time.time()
        # the kfreeze-13 cut from D = 35 on: without it weak (junk) directions bias the det root by up to 2e-9 at
        # D = 61 (four states within dD = 0.005 scatter by 2e-9 about a line; with the cut by 3e-11)
        s, F, kf = best_solver(meta, zh, zl, threads, force=opts.get("kf", "13" if D >= 35 else "none"))
        F_in = F
        if "polish" in opts:
            # re-converge in the rebuilt layout (its segment plan differs from the scan's)
            zh, zl, hist = newton_solve(s, zh, zl, tol=1e-29, it_max=int(opts["polish"]) if opts["polish"] != "1" else 4,
                                        verbose=True)
            fh, fl = s.residual(zh, zl)
            F = float(np.abs(np.asarray(fh) + np.asarray(fl)).max())
        lam0 = 1.0 / gamma_formula(D)
        log(f"D {D:.6f}: |F| {F:.1e} (kfreeze {kf}), formula lam {lam0:.9f}")
        fmax = float(opts.get("fmax", 1e-20))
        if F > fmax:
            rec = dict(D=D, M=meta["M"], kfreeze=kf, F=F, lam=None, gamma=None, lam_formula=lam0,
                       skipped=f"|F| {F:.1e} > {fmax:.0e}: state not converged in this layout", state=os.path.basename(f))
            log(json.dumps(rec))
            with open(out, "a") as fh_:
                fh_.write(json.dumps(rec) + "\n")
            continue
        lam, nch, nev = find_lam(s, zh, zl, lam0, log)
        rec = dict(D=D, M=meta["M"], kfreeze=kf, F=F, **({"F_in": F_in} if "polish" in opts else {}), lam=lam, gamma=(1 / lam if lam else None), lam_formula=lam0,
                   n_signchanges=nch, n_det=nev, state=os.path.basename(f), seconds=round(time.time() - t0, 1))
        if ("M48" in opts or "Mhi" in opts) and lam:
            # self-convergence: resample to Mhi (default 48) collocation points in tau, re-converge, find lam
            Mhi = int(opts.get("Mhi", 48))
            K = len(meta["nodes"])
            z48h, z48l = resample_z(np.asarray(zh) + np.asarray(zl), meta["M"], Mhi, K)
            m48 = dict(meta, M=Mhi)
            kf48 = opts.get("kf48", "13" if D >= 35 else "none")  # at D = 40 Newton from the resampled state stalls without the cut
            if kf48 != "none":
                # the frozen formulation pins p1's harmonics above the cut: project the resampled p1 onto k <= kf
                a_, b_ = plow_odd(Mhi, int(kf48))
                z48h[:Mhi] = (a_ + b_) @ z48h[:Mhi]
            s48, F48, kf48 = best_solver(m48, z48h, z48l, threads, M=Mhi, force=kf48)
            log(f"  M = {Mhi}: resampled |F| {F48:.1e} (kfreeze {kf48}); Newton")
            z48h, z48l, hist = newton_solve(s48, z48h, z48l, tol=max(10 * F, 1e-27), it_max=12, verbose=True)
            if hist[-1]["F"] > fmax:
                log("  Newton stalled; Levenberg-Marquardt")
                z48h, z48l, hist = lm_solve(s48, z48h, z48l, tol=max(10 * F, 1e-27), it_max=20, verbose=True,
                                            newton_after=1e-22)
            fh, fl = s48.residual(z48h, z48l)
            F48 = float(np.abs(np.asarray(fh) + np.asarray(fl)).max())
            lam48 = None
            if F48 <= fmax:
                lam48, nch48, _ = find_lam(s48, z48h, z48l, lam0, log)  # bracketed independently of lam
            rec.update(M48=dict(M=Mhi, kfreeze=kf48, F=F48, Delta_diff=float((z48h[-1] - zh[-1]) + (z48l[-1] - zl[-1])),
                                lam=lam48, lam_diff=(lam48 - lam) if lam48 else None))
        log(json.dumps(rec))
        with open(out, "a") as fh_:
            fh_.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
