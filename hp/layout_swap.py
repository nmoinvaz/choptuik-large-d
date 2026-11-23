#!/usr/bin/env python3
"""#1078: a low-D state moved to another state's layout (its nodes and its tau map), then re-converged there.

    python3 layout_swap.py IN.npz OUT.npz --like OTHER.npz [--threads 4] [--polish 12] [--hmax 0.05] [--tol 1e-20]
        [--part nodes|map|both] [--ntol 1e-27] [--allow_unraw]

At D 3.044375 (21 nodes to x 0.975, tau map s0 1.9635, L 0.4) the mode matrix's horizon block is singular to working
accuracy, while at D 3.15 (17 nodes to x 0.9, map s0 1.8326, L 0.7) it is well conditioned (dh370, #1078). Moving
each state to the other's layout tells the layout's part from the D's. Nodes: driver.renode (Taylor integration from
the nearest old node, the centre series below the first, the horizon series beyond the last); tau map:
tau_map.map_state (trigonometric interpolation in the source's s, in f64); then driver.newton_solve on OUT's layout and
step plan (hmax), as lowD_gamma.py --polish. --part moves only the nodes or only the map. kt, kc, ks stay IN's.

Prints |F| before and after the polish and Delta on both layouts; exit 1 if |F| stays above --tol. Controls:
--like IN (the identity: |F| after the polish at IN's level, Delta unchanged) and --polish 0 on a real move (exit 1).
OUT must be in a raw-data folder (raw_out.py)."""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import load, newton_solve, renode, save  # noqa: E402
from lowD_gamma import build_solver  # noqa: E402
from raw_out import FLAG, HELP, require_raw  # noqa: E402
from tau_map import map_state  # noqa: E402


def fmax(s, zh, zl):
    fh, fl = s.residual(zh, zl)
    return float(np.abs(np.asarray(fh) + np.asarray(fl)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--like", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--polish", type=int, default=12)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--tol", type=float, default=1e-20)
    ap.add_argument("--ntol", type=float, default=1e-27)   # Newton's stop: above the dd floor (3e-29 at D 3.15, #1078)
    ap.add_argument("--part", choices=("nodes", "map", "both"), default="both")
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    require_raw(a.out, allow=a.allow_unraw)
    t0 = time.time()
    zh, zl, meta = load(a.inp)
    like = load(a.like)[2]
    D, M = meta["D"], meta["M"]
    assert like["M"] == M, "the two states need the same M"
    nodes_old, map_old = list(meta["nodes"]), meta.get("map")
    nodes_new = list(like["nodes"]) if a.part in ("nodes", "both") else nodes_old
    map_new = like.get("map") if a.part in ("map", "both") else map_old
    print(f"IN {os.path.basename(a.inp)}: D {D} M {M}, {len(nodes_old)} nodes {nodes_old[0]}..{nodes_old[-1]}, "
          f"map {map_old}", flush=True)
    print(f"OUT layout ({a.part}): {len(nodes_new)} nodes {nodes_new[0]}..{nodes_new[-1]}, map {map_new}", flush=True)
    s_old, _ = build_solver(D, M, meta, a.threads)
    s_old.set_segs(s_old.plan(zh, zl, 1e-32, a.hmax))
    F_in, delta_in = fmax(s_old, zh, zl), float(zh[-1] + zl[-1])
    if nodes_new != nodes_old:
        zh, zl = renode(s_old, zh, zl, nodes_old, nodes_new, M)
    if map_new != map_old:
        zh, zl = map_state(zh, zl, M, M, len(nodes_new), None if map_old is None else tuple(map_old),
                           None if map_new is None else tuple(map_new))
    meta_new = dict(meta, nodes=nodes_new, map=map_new, layout_from=os.path.basename(a.like), layout_part=a.part)
    s, grid = build_solver(D, M, meta_new, a.threads)
    s.set_segs(s.plan(zh, zl, 1e-32, a.hmax))
    F_moved = fmax(s, zh, zl)
    if a.polish > 0:
        zh, zl, _ = newton_solve(s, zh, zl, tol=a.ntol, it_max=a.polish, verbose=True)
    F_out, delta_out = fmax(s, zh, zl), float(zh[-1] + zl[-1])
    save(a.out, zh, zl, meta_new)
    rec = dict(inp=os.path.basename(a.inp), like=os.path.basename(a.like), part=a.part, D=D, M=M, n=len(zh) - 1,
               F_in=F_in, F_moved=F_moved, F_out=F_out, delta_in=delta_in, delta_out=delta_out,
               d_delta=delta_out - delta_in, polish=a.polish, hmax=a.hmax, map=grid, seconds=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True)
    ok = F_out <= a.tol
    print(f"{'CONVERGED' if ok else 'NOT CONVERGED'}: |F| {F_in:.2e} (IN) -> {F_moved:.2e} (moved) -> {F_out:.2e}; "
          f"Delta {delta_in:.15f} -> {delta_out:.15f} ({delta_out - delta_in:+.2e})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
