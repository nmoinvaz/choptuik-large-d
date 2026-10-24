"""A double-double solver with one 'botched conversion' each: a constant of the discretisation kept in double precision
(its low word dropped), as happens when a double code is converted to double-double and some inputs are still built in
double. Each variant re-solves the converged state (deflated Newton, then Newton, to the floor); printed: the shift of
Delta from the clean re-solve and the |F| reached.

  dmat   the tau-derivative matrix (spectral differentiation) in double
  phase  the phase-condition weights in double
  n      n = D - 2 in double (D-dependent: exact at D = 4 and at every integer D)
  plow   the projectors of the frozen harmonics in double (states from D = 35)
  all    all of them

    python3 leak_test.py STATE.npz [STATE ...]
"""
import sys
import time

import mpmath as mp
import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hpkernel.driver import (Solver, dd_from_mp, deflated_newton, dmat_odd, load, newton_solve,  # noqa: E402
                             phase_weights, plow_odd)


def build(D, M, nodes, kt, kc, ks, kf, leak, threads=2):
    with mp.workdps(40):
        nh, nl = dd_from_mp([mp.mpf(D) - 2])
    nh, nl = float(nh[0]), float(nl[0])
    dh, dl = dmat_odd(M)
    ph, pl = phase_weights(M)
    a = b = None
    if kf:
        a, b = plow_odd(M, kf)
    dh, dl, ph, pl = (np.array(v, dtype=float) for v in (dh, dl, ph, pl))
    if leak in ("dmat", "all"):
        dl = np.zeros_like(dl)
    if leak in ("phase", "all"):
        pl = np.zeros_like(pl)
    if leak in ("n", "all"):
        nl = 0.0
    if kf and leak in ("plow", "all"):
        b = np.zeros_like(np.asarray(b))
    return Solver(M, nh, nl, dh, dl, ph, pl, list(nodes), [], kt, kc, ks, threads, a, b, None)


def solve(s, zh, zl):
    zh1, zl1, _ = deflated_newton(s, zh, zl, tol=1e-40, it_max=20, w=16, verbose=False)
    zh2, zl2, _ = newton_solve(s, zh1, zl1, tol=1e-40, it_max=6, verbose=False)
    Fh, Fl = s.residual(zh2, zl2)
    return zh2, zl2, float(np.max(np.abs(np.asarray(Fh) + np.asarray(Fl))))


for path in sys.argv[1:]:
    zh, zl, m = load(path)
    D, M = m["D"], m["M"]
    kf = 13 if D >= 35 else None
    t = time.time()
    s = build(D, M, m["nodes"], m["kt"], m["kc"], m["ks"], kf, "clean")
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    segs = s.get_segs()
    z0h, z0l, F0 = solve(s, zh, zl)
    print(f"D {D:.4f}: clean Delta {z0h[-1]!r} + {z0l[-1]:.3e}, |F| {F0:.1e} ({time.time() - t:.0f} s)", flush=True)
    for leak in ("n", "phase", "plow", "dmat", "all"):
        if leak == "plow" and not kf:
            continue
        t = time.time()
        sl = build(D, M, m["nodes"], m["kt"], m["kc"], m["ks"], kf, leak)
        sl.set_segs(segs)
        z1h, z1l, F1 = solve(sl, z0h, z0l)
        d = (float(z1h[-1]) - float(z0h[-1])) + (float(z1l[-1]) - float(z0l[-1]))
        print(f"  {leak:5s} in double: Delta shift {d:+.3e} (|F| {F1:.1e}, {time.time() - t:.0f} s)", flush=True)
