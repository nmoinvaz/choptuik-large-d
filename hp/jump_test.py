#!/usr/bin/env python3
"""Can the large-D solver jump in D instead of crawling? (The question behind reaching D = 200: continuation from
D = 61 in steps of 0.002-0.03 would take months, and the precision needed rises with D, so a far point has to be
reached in a few large jumps.)

The saved scan states at D = 36-45 (kfreeze 13, M = 32, nodes 0.3-0.7) are fitted component by component with
polynomials in t = 1/D, and the fit is evaluated at a far target D whose converged scan state exists. From that
seed, driver.deflated_newton (w = 24, as the scan) solves to 10 times the stored state's |F|. Reported: the seed's
|F|, whether it converged, the iterations, and the Delta found against the stored one.

    cd hp && python3 jump_test.py [--fit=36,45] [--targets=47.0082,50.0075,55.0351,60.0020] [--deg=4,6]
"""
import glob
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import deflated_newton, load, make_solver  # noqa: E402

KC = 220          # enough at D >= 36 (kc_check.py: 209 at D = 36 with margin 30)


def states():
    out = {}
    for p in glob.glob(os.path.join(HERE, "data", "states", "up*_D*.npz")):
        out[float(re.search(r"_D([0-9.]+)\.npz$", p).group(1))] = p
    return out


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    lo, hi = (float(x) for x in opts.get("fit", "36,45").split(","))
    targets = [float(x) for x in opts.get("targets", "47.0082,50.0075,55.0351,60.0020").split(",")]
    degs = [int(x) for x in opts.get("deg", "4,6").split(",")]
    avail = states()
    fitD = sorted(d for d in avail if lo <= d <= hi)
    Z = []
    for d in fitD:
        zh, zl, m = load(avail[d])
        Z.append(np.asarray(zh) + np.asarray(zl))
    Z = np.array(Z)
    t = 1.0 / np.array(fitD)
    print(f"fit on {len(fitD)} states, D = {fitD[0]}..{fitD[-1]}", flush=True)
    for Dt in targets:
        zh_t, zl_t, mt = load(avail[min(avail, key=lambda d: abs(d - Dt))])
        D = mt["D"]
        mk = lambda: make_solver(D, mt["M"], nodes=mt["nodes"], kt=mt["kt"], kc=min(KC, mt["kc"]), ks=mt["ks"],
                                 threads=2, kfreeze=13)
        s0 = mk()
        s0.set_segs(s0.plan(zh_t, zl_t, 1e-32, 0.05))
        F0 = float(np.max(np.abs(np.asarray(s0.residual(zh_t, zl_t)[0]))))
        tol = max(10 * F0, 1e-28)
        for deg in degs:
            c = np.polyfit(t, Z, deg)
            seed = np.polyval(c, 1.0 / D) if Z.ndim == 1 else np.array([np.polyval(c[:, j], 1.0 / D) for j in range(Z.shape[1])])
            s = mk()
            s.set_segs(s.plan(seed, np.zeros_like(seed), 1e-32, 0.05))
            Fs = float(np.max(np.abs(np.asarray(s.residual(seed, np.zeros_like(seed))[0]))))
            dz = float(np.max(np.abs(seed - (np.asarray(zh_t) + np.asarray(zl_t)))))
            t0 = time.time()
            try:
                zh, zl, hist = deflated_newton(s, seed.copy(), np.zeros_like(seed), tol=tol, it_max=30, w=24,
                                               verbose=False)
                Ff = hist[-1].get("F", float("nan")) if hist else float("nan")
                ok = Ff <= tol
                dD = (zh[-1] - zh_t[-1]) + (zl[-1] - zl_t[-1])
                res = f"{'converged' if ok else 'NOT converged'} in {len(hist)} its, |F| {Ff:.1e}, Delta - stored {dD:+.1e}"
            except Exception as e:  # noqa: BLE001
                res = f"failed: {str(e)[:120]}"
            print(f"D {lo:g}-{hi:g} -> {D}: jump {D - hi:.1f}, degree {deg}: seed |F| {Fs:.1e} (max |z - stored| {dz:.1e}); "
                  f"{res}; {time.time() - t0:.0f} s; tol {tol:.0e}", flush=True)


if __name__ == "__main__":
    main()
