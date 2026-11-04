#!/usr/bin/env python3
"""Release the held harmonics: the quad-double Newton (newton_qd, plain) of the full system (kfreeze 13, every p1
harmonic solved) from a pinned chain state, as T75's protocol checks its pinned states (README: "Delta after one
full-system Newton step"; notes/QUAD_DOUBLE.md 12, the full-system convergence of T75's states). Prints the
full-system residual at the pinned state, each step, Delta before and after with the first step's dDelta, and the
1/D series fitted on D <= 67.77 (dd_fits_mp, degree 20) at that D for reference. --kfreeze K (default 13) frees the
harmonics above K only; --kfreeze 3 reproduces the chain's own system, whose residual at its state must be at the floor
(the control).
    cd hp && python3 full_release.py RAWDIR STATE [STATE ...] [--kfreeze 13] [--threads 2] [--it 10]
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import make_solver, newton_qd, load_q, qmax  # noqa: E402
from pinning import PinnedQ, PinnedSolver, band_coeffs, band_field  # noqa: E402
import dd_fits_mp as F  # noqa: E402
mp.mp.dps = 60
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = dict(kfreeze=13, threads=2, it=10)
for k in list(opt):
    if f"--{k}" in sys.argv:
        opt[k] = int(sys.argv[sys.argv.index(f"--{k}") + 1])
        args = [a for a in args if a != str(opt[k])]
raw, states = args[0], args[1:]
pts, _ = F.load_points()
Ds = F.clean(pts, sorted(d for d in pts if 10 <= d <= 67.77), verbose=False)
ev, _, _ = F.fit(pts, Ds, "1/D", 20)
for st in states:
    z, m = load_q(os.path.join(raw, st))
    D, M = (mp.mpf(m["D_str"]) if "D_str" in m else mp.mpf(float(m["D"]))), int(m["M"])
    s = make_solver(D, M, nodes=list(m["nodes"]), kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130), threads=opt["threads"],
                    kfreeze=opt["kfreeze"])
    s.set_segs(s.plan(np.asarray(z[0]), np.asarray(z[1]), 1e-32, 0.05))
    kf_state = int(m.get("kfreeze", 3))
    if opt["kfreeze"] == kf_state:
        # the chain's own system: its frozen rows read coefficient - held value, the held values being the state's own
        ks_held = [k for k in range(kf_state + 1, 14) if k % 2 == 1]
        hh, hl = band_field(band_coeffs(np.asarray(z[0])[:M], np.asarray(z[1])[:M], M, ks_held), ks_held, M)
        ps = PinnedSolver(s, M, hh, hl)
        s = PinnedQ(s, ps.ch, ps.cl)
    F0 = s.residual_q(*z)
    d0 = sum(mp.mpf(float(p[-1])) for p in z)
    print(f"{st}: D {mp.nstr(D, 12)}, M {M}, kfreeze {opt['kfreeze']}: residual at the pinned state {qmax(F0[:2]):.2e}, Delta {mp.nstr(d0, 17)}", flush=True)
    z2, hist, conv = newton_qd(s, z, tol=1e-40, it_max=opt["it"], verbose=True)
    d1 = sum(mp.mpf(float(p[-1])) for p in z2)
    ser = ev(D)
    first = hist[0].get("dDelta") if hist else None
    print(f"  {'converged' if conv else 'not converged'} in {len(hist)} steps; Delta {mp.nstr(d1, 17)}, moved {float(d1 - d0):+.3e}"
          f" (first step's dDelta {first if first is None else format(first, '+.3e')}); series 1/D 20 {mp.nstr(ser, 17)}:"
          f" pinned - series {float(d0 - ser):+.3e}, released - series {float(d1 - ser):+.3e}", flush=True)
