#!/usr/bin/env python3
"""A chain state's pinned residual on its run's plan, and the state saved with that plan (#589).

    cd hp && python3 plan_state.py RAWDIR STATE [--base BASE --out OUT] [--threads 2]

anm_run.py plans once, at its run's first base (Solver.plan at --plan_tol 1e-32, 0.05, from the base's first two
parts), and older states were saved without that plan. With --base, the plan is made again at BASE (or taken from
BASE's stored plan if it has one) and STATE is saved with it as OUT (eight parts, its meta's segs and plan_from set),
so that near_null.py and held_homotopy.py factor the run's own system. Without --base the state's stored plan is used.
STATE, BASE and OUT are relative to RAWDIR; OUT must be in a task folder of the raw-data repository (hp/raw_out.py).

Printed: the pinned max |F| (renode_o.pinned: the held offset at the state's own coefficients, the base polish's
system) by block, with the kernel the path gives (hpkernel.__file__ is printed). A state on its run's plan and the
rounded projector it was solved with sits at its run's converged level; that is #589's control.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpkernel  # noqa: E402
from hpkernel.driver import make_solver  # noqa: E402
from octo_corrector import load_o, save_o  # noqa: E402
from raw_out import require_raw  # noqa: E402
from renode_o import blocks, fmax, pinned  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("state")
    ap.add_argument("--base", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if bool(a.base) != bool(a.out):
        sys.exit("--base and --out go together")
    if a.out:
        require_raw(os.path.join(a.raw, a.out))
    z, m = load_o(os.path.join(a.raw, a.state))
    M, nodes, kf = int(m["M"]), list(m["nodes"]), int(m.get("kfreeze", 3))
    s = make_solver(m.get("D_str", repr(float(m["D"]))), M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146),
                    ks=m.get("ks", 130), threads=a.threads, kfreeze=kf)
    print(f"hpkernel {hpkernel.__file__}", flush=True)
    if a.base:
        zb, mb = load_o(os.path.join(a.raw, a.base))
        if list(mb["nodes"]) != nodes or int(mb["M"]) != M:
            sys.exit(f"{a.base}: another layout ({mb['nodes']}, M {mb['M']})")
        if "segs" in mb:
            segs, how = mb["segs"], f"{a.base}'s stored plan"
        else:
            segs, how = s.plan(zb[0], zb[1], 1e-32, 0.05), f"planned at {a.base} (1e-32, 0.05)"
    elif "segs" in m:
        segs, how = m["segs"], "the state's stored plan"
    else:
        sys.exit(f"{a.state} has no stored plan: give --base (its run's first base) and --out")
    s.set_segs(segs)
    F = pinned(s, z, M, kf).residual_o(z)
    print(f"{a.state}: D {float(m['D']):.10g}, nodes {nodes}; plan {how}, {sum(len(sg) - 1 for sg in segs)} steps; pinned "
          f"max|F| {fmax(F):.3e}; by block: " + "; ".join(f"{k} {v:.2e}" for k, v in blocks(F, M, len(nodes), kf).items()),
          flush=True)
    if a.out:
        meta = dict(m, segs=[[float(x) for x in sg] for sg in segs], parts=8, source=a.state, plan_from=a.base)
        save_o(os.path.join(a.raw, a.out), z, meta)
        print(f"saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
