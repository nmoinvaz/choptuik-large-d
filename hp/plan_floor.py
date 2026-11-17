#!/usr/bin/env python3
"""Is a low-D state's residual floor the plan's or the state's? Its double-double max |F| on its own plan and on
plans that differ from it near x = c and in hmax (#945).

    cd hp && python3 plan_floor.py OUT.jsonl STATE.npz [STATE.npz ...] [--threads 2] [--tol_step 1e-30] [--hmax 0.05]
        [--below 1e-2]

For each state (hp/lowD_dd.py's layout: zh, zl and meta with D, M, nodes, kt, kc, ks and its plan segs), max |F| of
Solver.residual on: (a) its own plan; (b) that plan with the steps of the segments near x = c split 4, 8 and 16 times
(sonic_valley.refine_near); (c) a plan made from the state at --tol_step with hmax halved; (d) the control, its own
plan coarsened by 4 (each segment keeps every 4th of its x points and its ends). A residual that falls with the
splitting and with hmax says the plan sets the floor; one that stays at the own plan's value says the state does. The
control must not fall below (a): a coarser plan cannot remove truncation. One record per state and plan in OUT.jsonl.
--by_block: on the own plan, max |F| of each block of 4M rows (node block j, the last row Delta's) and the 8 largest
rows as (block, field, tau index). --newton JAC (f64g, f64, f64h, f64hg, dd): Newton from the state on its own plan
with that step (hpkernel.driver.newton_solve, tol 1e-28, --it_max steps), one record per step: a residual that falls
below the own plan's floor says the step's accuracy, not the state, set it.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import sonic_valley  # noqa: E402
from hpkernel.driver import load, make_solver, newton_solve  # noqa: E402
from raw_out import FLAG, HELP, require_raw  # noqa: E402


def coarsen(segs, r):
    """each segment's x points thinned to every r-th, its ends kept"""
    out = []
    for g in segs:
        g = list(g)
        keep = g[::r]
        if keep[-1] != g[-1]:
            keep.append(g[-1])
        out.append(keep)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("states", nargs="+")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tol_step", type=float, default=1e-30)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--below", type=float, default=1e-2)
    ap.add_argument("--by_block", action="store_true")
    ap.add_argument("--newton", default=None, choices=["f64g", "f64", "f64h", "f64hg", "dd"])
    ap.add_argument("--it_max", type=int, default=6)
    ap.add_argument("--skip_plans", action="store_true", help="only --by_block and --newton")
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    require_raw(a.out, allow=a.allow_unraw)
    for path in a.states:
        zh, zl, m = load(path)
        zh, zl = np.asarray(zh), np.asarray(zl)
        D, M, nodes = float(m["D"]), int(m["M"]), list(m["nodes"])
        s = make_solver(D, M, nodes=nodes, kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=a.threads)
        own = [list(g) for g in m["segs"]]
        z = zh + zl
        plans = []
        if not a.skip_plans:
            plans = [("own", own)]
            plans += [(f"split {r} near x = c", sonic_valley.refine_near(own, z, M, nodes, r, a.below)) for r in (4, 8, 16)]
            s.set_segs(own)
            plans.append((f"replanned, hmax {a.hmax / 2:g}", s.plan(zh, zl, a.tol_step, a.hmax / 2)))
            plans.append(("control: own coarsened by 4", coarsen(own, 4)))
        print(f"{os.path.basename(path)}: D {D}, M {M}", flush=True)
        for name, segs in plans:
            s.set_segs(segs)
            fh, fl = s.residual(zh, zl)
            F = float(np.abs(np.asarray(fh) + np.asarray(fl)).max())
            steps = sum(len(g) - 1 for g in segs)
            print(f"  {name:32s} {steps:8d} steps  max|F| {F:.3e}", flush=True)
            with open(a.out, "a") as fo:
                fo.write(json.dumps(dict(state=os.path.basename(path), source=path, D=D, M=M, plan=name, steps=steps,
                                         F=F)) + "\n")

        if a.by_block:
            s.set_segs(own)
            fh, fl = s.residual(zh, zl)
            F = np.abs(np.asarray(fh) + np.asarray(fl))
            B = 4 * M
            nb = (len(F) - 1) // B
            blocks = [float(F[j * B:(j + 1) * B].max()) for j in range(nb)] + [float(F[-1])]
            print("  max|F| by block of 4M rows (the last: Delta's row): " + ", ".join(f"{j}:{v:.1e}" for j, v in enumerate(blocks)), flush=True)
            top = np.argsort(F)[::-1][:8]
            for i in top:
                j, r = divmod(int(i), B)
                print(f"    row {int(i)}: block {j}, field {r // M}, tau index {r % M}: |F| {F[i]:.3e}", flush=True)
            with open(a.out, "a") as fo:
                fo.write(json.dumps(dict(state=os.path.basename(path), D=D, M=M, by_block=blocks,
                                         top=[dict(row=int(i), block=int(i) // B, field=(int(i) % B) // M, tau=int(i) % M,
                                                   F=float(F[i])) for i in top])) + "\n")
        if a.newton:
            s.set_segs(own)
            _, _, hist = newton_solve(s, zh, zl, tol=1e-28, it_max=a.it_max, verbose=True, jac=a.newton)
            with open(a.out, "a") as fo:
                for h in hist:
                    fo.write(json.dumps(dict(state=os.path.basename(path), D=D, M=M, newton=a.newton,
                                             **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                for k, v in h.items() if k in ("it", "F", "lam", "dz", "failed", "products")})) + "\n")


if __name__ == "__main__":
    sys.exit(main())
