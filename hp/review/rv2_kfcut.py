#!/usr/bin/env python3
"""Review RV2: the size of the kfreeze-13 cut at a saved state, by pinning instead of freeing p1's harmonic 15.

In the kfreeze-13 formulation p1's harmonics above 13 are pinned at zero (and p's matching above 13 dropped). Pinned
at their TRUE values instead, the pinned system's solution is the true one (the dropped matching equations then hold
by themselves). So Delta_true - Delta_13 = Delta_pinned(a*) - Delta_pinned(0) ~ sum_k dDelta/da_k a_k*, dominated by
k = 15. This measures dDelta/da_15 (cos and sin parts) by pinning harmonic 15 at +-amp (hpkernel's pinning offset,
hp/pinning.PinnedSolver, on a kfreeze-13 solver) and re-converging with Newton, and estimates a_15* by continuing the
geometric decay of the measured harmonics 9, 11, 13. (Freeing harmonic 15 directly, kfreeze 15, fails at D = 60: its
singular value is so small that the first Newton step leaves the basin.)

    python3 hp/review/rv2_kfcut.py STATE.npz [--amp 1e-8] [--its 8]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hpkernel.driver import dd_add, load, make_solver, newton_solve  # noqa: E402
from pinning import PinnedSolver, band_coeffs, band_field  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--amp", type=float, default=1e-8)
    ap.add_argument("--its", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    zh, zl, meta = load(a.state)
    D, M, nodes = meta["D"], meta["M"], meta["nodes"]
    co = band_coeffs(zh[:M], zl[:M], M, [9, 11, 13, 15])
    mag = [float(np.hypot(*c)) for c in co]
    r = mag[2] / mag[1]
    print(f"D {D}: |p1_k| k = 9, 11, 13, 15: " + ", ".join(f"{m:.3e}" for m in mag) + f"; ratio 13/11 {r:.3f}", flush=True)
    est15 = (co[2][0] * r, co[2][1] * r)            # harmonic 13 continued by one more decay step
    base = None
    out = {}
    for label, h in (("zero", (0.0, 0.0)), ("cos", (a.amp, 0.0)), ("sin", (0.0, a.amp))):
        s = make_solver(D, M, nodes=nodes, kt=36, kc=meta["kc"], ks=meta["ks"], threads=1, kfreeze=13)
        s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
        hh, hl = band_field([h], [15], M)
        ps = PinnedSolver(s, M, hh, hl)
        z0h, z0l = zh.copy(), zl.copy()
        z0h[:M], z0l[:M] = dd_add(z0h[:M], z0l[:M], hh, hl)
        z1h, z1l, hist = newton_solve(ps, z0h, z0l, tol=1e-40, it_max=a.its, verbose=False)
        Fh, Fl = ps.residual(z1h, z1l)
        F = float(np.abs(Fh + Fl).max())
        dl = float((z1h[-1] - zh[-1]) + (z1l[-1] - zl[-1]))
        c15 = band_coeffs(z1h[:M], z1l[:M], M, [15])[0]
        out[label] = dict(F=F, dDelta=dl, its=len(hist), p1_15=c15)
        print(f"  pinned harmonic 15 at {h}: |F| {F:.1e} after {len(hist)} its, Delta - saved {dl:+.3e}, "
              f"p1_15 now ({c15[0]:+.2e}, {c15[1]:+.2e})  ({time.time() - t0:.0f}s)", flush=True)
        if label == "zero":
            base = dl
    dc = (out["cos"]["dDelta"] - base) / a.amp
    ds = (out["sin"]["dDelta"] - base) / a.amp
    eff = dc * est15[0] + ds * est15[1]
    print(f"dDelta/da_15: cos {dc:+.2e}, sin {ds:+.2e} per unit amplitude; a_15 estimated ({est15[0]:+.2e}, {est15[1]:+.2e}); "
          f"cut effect Delta_true - Delta_13 ~ {eff:+.1e}", flush=True)
    rec = dict(state=os.path.basename(a.state), D=D, mags=mag, est15=est15, dDelta_dcos=dc, dDelta_dsin=ds, cut=eff,
               runs=out, amp=a.amp)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
