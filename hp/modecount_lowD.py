#!/usr/bin/env python3
"""A double-double state at low D for the mode count (notes/MODE_COUNT.md), seeded from an f64 solution.

    python3 modecount_lowD.py F64_STATE.npy [--kc=39] [--M=32,48] [--ckc=60] [--ks=100] [--out=hp/data/modecount_state_D<D>_M<M>.npz]

The double-double states in hp/data/states start at D = 8.25. This seeds hpkernel from the f64 solver's state (default
solver/data/neB_kc39_D4.0000.npy: kc 39, nodes 0.02 ... 0.6, matched at x_K = 0.6) as hp/proto/val_lowD.py did at
D = 7 and 7.75: seed_from_f64 on the same nodes, Newton at each M in turn (resampled in tau between them), and saves the
last one. Then count with

    python3 hp/modecount_wind.py hp/data/modecount_state_D4.0000_M48.npz --h0=0.6 --lam=2.6740700

(lam_phys(4) = 1/0.3739610, solver/CONTEXT.md). Prints Delta against the f64 value and |F| after each Newton.
"""
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import make_solver, seed_from_f64, newton_solve, save, resample_z  # noqa: E402

NL = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    path = args[0] if args else os.path.join(HERE, "..", "solver", "data", "neB_kc39_D4.0000.npy")
    kc = int(opts.get("kc", 39))
    Ms = [int(m) for m in opts.get("M", "32,48").split(",")]
    ckc, ks = int(opts.get("ckc", 60)), int(opts.get("ks", 100))
    D = float(re.findall(r"D([0-9.]+)\.npy", path)[0])
    z64 = np.load(path)
    t0 = time.time()
    D, zh, zl = seed_from_f64(path, Ms[0], kc=kc, nodes=NL)
    print(f"D {D}: seeded at M {Ms[0]} from {os.path.basename(path)} (f64 Delta {z64[-1]:.12f})", flush=True)
    Mo = Ms[0]
    for M in Ms:
        if M != Mo:
            zh, zl = resample_z(np.asarray(zh) + np.asarray(zl), Mo, M, len(NL))
        s = make_solver(D, M, nodes=NL, kt=36, kc=ckc, ks=ks)
        s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
        zh, zl, hist = newton_solve(s, zh, zl, tol=1e-29, it_max=12, verbose=True)
        F = hist[-1]["F"]
        print(f"D {D} M {M}: Delta {zh[-1] + zl[-1]:.15f}, minus f64 {zh[-1] + zl[-1] - z64[-1]:+.2e}, |F| {F:.1e}, "
              f"{len(hist)} Newton steps, {time.time() - t0:.0f} s", flush=True)
        Mo = M
    out = opts.get("out", os.path.join(HERE, "data", f"modecount_state_D{D:.4f}_M{Mo}.npz"))
    save(out, zh, zl, dict(D=D, M=Mo, nodes=NL, kt=36, kc=ckc, ks=ks, source=os.path.basename(path)))
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
