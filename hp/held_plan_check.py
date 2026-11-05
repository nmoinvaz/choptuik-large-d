#!/usr/bin/env python3
"""Why two converged chain states at the same D differ in Delta (notes/QUAD_DOUBLE.md 13.22): the pinned system of
hp/anm_run.py depends on the held harmonics (each chain re-bases them on its own previous state, in float64) and on
the Taylor plan (each chain plans from its own prediction). This re-solves the pinned quad-double system from state B
with state A's held values and plan, and with each one swapped, and prints Delta against A's:

  run 1: held A, plan A, from B   -> Delta1; if the difference to A is gone, held values and plan explain it
  run 2: held B, plan A, from B   -> Delta2 - Delta1 is the held values' effect
  run 3: held A, plan B, from A   -> Delta3 - Delta(A) is the plan's effect

    python3 held_plan_check.py RAWDIR A.npz B.npz [--threads 2] [--kfreeze 3]

A and B are paths relative to RAWDIR, four-part states with D_str in their meta (anm_run.py's saves). The control is
run 2: it differs from run 1 unless the held values are equal.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import make_solver, newton_qd  # noqa: E402
from pinning import PinnedQ, PinnedSolver, band_coeffs, band_field  # noqa: E402

mp.mp.dps = 50


def load4(path):
    d = np.load(path)
    m = json.loads(str(d["meta"]))
    z = tuple(np.ascontiguousarray(np.asarray(d[k], dtype=float)) for k in ("zh", "zl", "z2", "z3"))
    return z, m


def delta(z):
    return mp.mpf(float(z[0][-1])) + mp.mpf(float(z[1][-1])) + mp.mpf(float(z[2][-1])) + mp.mpf(float(z[3][-1]))


def main():
    args = list(sys.argv[1:])
    opt = dict(threads=2, kfreeze=3)
    for k in list(opt):
        f = "--" + k
        if f in args:
            i = args.index(f); opt[k] = int(args[i + 1]); del args[i:i + 2]
    raw, pa, pb = args[0], args[1], args[2]
    za, ma = load4(os.path.join(raw, pa))
    zb, mb = load4(os.path.join(raw, pb))
    assert ma["D_str"] == mb["D_str"], (ma["D_str"], mb["D_str"])
    D, M = mp.mpf(ma["D_str"]), int(ma["M"])
    ks_held = list(ma.get("pin_held", [k for k in range(opt["kfreeze"] + 1, 14) if k % 2 == 1]))
    s = make_solver(D, M, nodes=ma["nodes"], kt=ma["kt"], kc=ma["kc"], ks=ma["ks"], threads=opt["threads"], kfreeze=opt["kfreeze"])
    plan_a, plan_b = s.plan(za[0], za[1], 1e-32, 0.05), s.plan(zb[0], zb[1], 1e-32, 0.05)
    ha, hb = band_coeffs(za[0][:M], za[1][:M], M, ks_held), band_coeffs(zb[0][:M], zb[1][:M], M, ks_held)
    dheld = max(abs(x - y) for p, q in zip(ha, hb) for x, y in zip(p, q))
    print(f"D {mp.nstr(D, 15)}, M {M}: Delta(A) - Delta(B) = {mp.nstr(delta(za) - delta(zb), 5)}; held values differ by "
          f"{dheld:.3e} at most (harmonics {ks_held}); plans {sum(len(g) - 1 for g in plan_a)} and "
          f"{sum(len(g) - 1 for g in plan_b)} steps", flush=True)

    def solve(held, plan, z0, label):
        s.set_segs(plan)
        hh, hl = band_field(held, ks_held, M)
        sp = PinnedSolver(s, M, hh, hl)
        z4, hist, conv = newton_qd(PinnedQ(s, sp.ch, sp.cl), z0, tol=1e-28, it_max=8, verbose=False)
        d = delta(z4)
        print(f"  {label}: {len(hist)} steps, {'converged' if conv else 'NOT converged'}, Delta - Delta(A) = "
              f"{mp.nstr(d - delta(za), 5)}", flush=True)
        return d, conv

    d1, c1 = solve(ha, plan_a, zb, "run 1, held A, plan A, from B")
    d2, c2 = solve(hb, plan_a, zb, "run 2, held B, plan A, from B")
    d3, c3 = solve(ha, plan_b, za, "run 3, held A, plan B, from A")
    print(f"held values' effect (run 2 - run 1): {mp.nstr(d2 - d1, 5)}; plan's effect (run 3 - A): "
          f"{mp.nstr(d3 - delta(za), 5)}", flush=True)
    sys.exit(0 if (c1 and c2 and c3) else 1)


if __name__ == "__main__":
    main()
