#!/usr/bin/env python3
"""Wall time of one Solver.residual_jet_q call (the quad-double jet of the large-D chain's series, nout coefficients)
against the thread count, at a stored large-D base, for comparing two builds back to back (notes/QUAD_DOUBLE.md 13.24).
The higher coefficients are the base scaled by 2^-10k: the call's cost does not depend on their values.

    python3 jet_scale.py RAWDIR [--state anm-mac/pred_q/A2/anm_pin3_D128.5559.npz] [--threads 1,4,6] [--nout 24]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import load_q, make_solver  # noqa: E402

mp.mp.dps = 50


def main():
    args = list(sys.argv[1:])
    opt = dict(state="anm-mac/pred_q/A2/anm_pin3_D128.5559.npz", threads="1,4,6", nout="24")
    for k in list(opt):
        f = "--" + k
        if f in args:
            i = args.index(f); opt[k] = args[i + 1]; del args[i:i + 2]
    zq, m = load_q(os.path.join(args[0], opt["state"]))
    zq = tuple(np.ascontiguousarray(np.asarray(p)) for p in zq)
    D, M, nout = mp.mpf(m["D_str"]), int(m["M"]), int(opt["nout"])
    zs = [zq] + [tuple(np.ascontiguousarray(p * 2.0 ** (-10 * k)) for p in zq) for k in range(1, nout)]
    print(f"state {opt['state']}, D {mp.nstr(D, 10)}, M {M}, nout {nout}", flush=True)
    for th in [int(t) for t in opt["threads"].split(",")]:
        s = make_solver(D, M, nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=th, kfreeze=3)
        s.set_segs(s.plan(zq[0], zq[1], 1e-32, 0.05))
        t0 = time.time()
        s.residual_jet_q(zs, nout)
        print(f"threads {th}: {time.time() - t0:.2f} s", flush=True)


if __name__ == "__main__":
    main()
