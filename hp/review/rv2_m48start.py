#!/usr/bin/env python3
"""Review RV2: the M = 48 residual of a saved M = 32 state interpolated to 48 tau points in double-double (no Newton),
and the linearised Delta change -dDelta/dF . F that the M = 48 system asks for. If the M = 32 solution were not resolved
in tau, the interpolated state would not satisfy the M = 48 collocation equations.

    python3 hp/review/rv2_m48start.py STATE.npz [--M 48] [--kfreeze 13]
"""
import argparse
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hpkernel.driver import block_norms, delta_correction, load, make_solver  # noqa: E402
from rv2_resolve import resample_dd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("state")
ap.add_argument("--M", type=int, default=48)
ap.add_argument("--kfreeze", type=int, default=13)
a = ap.parse_args()
zh, zl, meta = load(a.state)
D, M0, nodes = meta["D"], meta["M"], meta["nodes"]
for M in (M0, a.M):
    h, l = (zh, zl) if M == M0 else resample_dd(zh, zl, M0, M, len(nodes))
    s = make_solver(D, M, nodes=nodes, kt=36, kc=meta["kc"], ks=meta["ks"], threads=1, kfreeze=a.kfreeze or None)
    s.set_segs(s.plan(h, l, 1e-32, 0.05))
    s.newton(h, l)
    Fh, Fl = s.residual(h, l)
    print(f"D {D} M {M}: |F| per block " + " ".join(f"{v:.1e}" for v in block_norms(s, Fh, Fl))
          + f"; linearised Delta change {delta_correction(s, Fh, Fl):+.1e}", flush=True)
