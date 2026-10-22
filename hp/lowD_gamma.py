#!/usr/bin/env python3
"""gamma(D) = 1/lam below D = 4 from double-double states (hp/data/lowD_states/*.npz): the root of det M(lam) by
hp/gamma_dd.find_lam, with the starting guess interpolated from a table of earlier low-D values (the f64 kc 47 sweep,
solver/data/scan_lowD3_kc47.jsonl, and the README value at D = 4) instead of the rational formula, which is not fitted
below D = 4. The guess only places the bracket; the root is located by sign changes of det M(lam).

    python3 lowD_gamma.py OUT.jsonl STATE.npz [STATE.npz ...] [--threads=2] [--polish=N] [--lam0=L] [--xtol=1e-11]

--polish=N: re-converge the state first in the rebuilt layout (N Newton steps), as gamma_dd.py --polish.
The solver is rebuilt with the state's own centre/horizon series lengths (meta kc, ks) and a fresh step plan.
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
from hpkernel.driver import load, make_solver, newton_solve  # noqa: E402


def find_lam(s, zh, zl, lam0, log=None, xtol=1e-11, widths=(1e-4, 1e-3, 1e-2, 4e-2)):
    """root of det M(lam) near lam0 (as hp/gamma_dd.find_lam, with a tighter first bracket and xtol, since each det
    costs one double-double Jacobian and LU: about 6-8 evaluations from a guess within 1e-4)"""
    cache = {}

    def sd(lam):
        if lam not in cache:
            t = time.time()
            sg, la, pmin, pmax = s.mode_det(zh, zl, float(lam), 0.0)
            cache[lam] = (sg, la)
            if log:
                log(f"    lam {lam:.12f}: sign {sg:+.0f} ln|det| {la:.6f} pivots {pmin:.2e}..{pmax:.2e} ({time.time() - t:.1f} s)")
        return cache[lam]

    for w in widths:
        grid = [lam0 - w, lam0, lam0 + w]
        sg = [sd(l)[0] for l in grid]
        ch = [i for i in range(2) if sg[i] != sg[i + 1]]
        if ch:
            i = min(ch, key=lambda i: abs(0.5 * (grid[i] + grid[i + 1]) - lam0))
            a, b = grid[i], grid[i + 1]
            ref = 0.5 * (sd(a)[1] + sd(b)[1])
            f = lambda l: sd(l)[0] * math.exp(sd(l)[1] - ref)
            lam = brentq(f, a, b, xtol=xtol, rtol=1e-15, maxiter=60)
            return lam, len(ch), len(cache)
    return None, 0, len(cache)

# guesses for lam = 1/gamma: f64 kc 47 values (scan_lowD3_kc47.jsonl) and D = 4
TABLE = [(3.6425, 1 / 0.34213711), (3.675, 1 / 0.34590386), (3.7, 1 / 0.34864959), (3.7938, 1 / 0.35792558),
         (4.0, 1 / 0.37396062), (4.25, 1 / 0.38807371), (4.5, 1 / 0.39853855), (5.0, 1 / 0.41312694)]


def lam_guess(D):
    Ds = np.array([t[0] for t in TABLE])
    ls = np.array([t[1] for t in TABLE])
    if D < Ds[0]:
        c = np.polyfit(Ds[:3], ls[:3], 2)
        return float(np.polyval(c, D))
    return float(np.interp(D, Ds, ls))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    threads = int(opts.get("threads", 2))
    out = args[0]
    log = lambda m: print(m, flush=True)
    for f in args[1:]:
        zh, zl, meta = load(f)
        D, M = meta["D"], meta["M"]
        t0 = time.time()
        s = make_solver(D, M, nodes=meta["nodes"], kt=meta.get("kt", 36), kc=meta.get("kc", 60), ks=meta.get("ks", 100),
                        threads=threads)
        s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
        if "polish" in opts:
            zh, zl, _ = newton_solve(s, zh, zl, tol=1e-29, it_max=int(opts["polish"]), verbose=True)
        fh, fl = s.residual(zh, zl)
        F = float(np.abs(np.asarray(fh) + np.asarray(fl)).max())
        lam0 = float(opts["lam0"]) if "lam0" in opts else lam_guess(D)
        log(f"{os.path.basename(f)}: D {D} M {M} |F| {F:.1e}, guess lam {lam0:.6f}")
        lam, nch, nev = find_lam(s, zh, zl, lam0, log, xtol=float(opts.get("xtol", 1e-11)))
        rec = dict(D=D, M=M, F=F, lam=lam, gamma=(1 / lam if lam else None), lam_guess=lam0, n_signchanges=nch,
                   n_det=nev, state=os.path.basename(f), seconds=round(time.time() - t0, 1))
        log(json.dumps(rec))
        with open(out, "a") as fh_:
            fh_.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
