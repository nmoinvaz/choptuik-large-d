#!/usr/bin/env python3
"""One segment of an hp state marched by band.py, against hp's own next node (notes/LOW_D_SONIC.md, check P1).

    python3 segment.py OUT.jsonl STATE.npz --band 0.01,0.02 [--warp none|test|design] [--A 1.5e-3] [--eps0 2e-3]
        [--kappa 1.0] [--tol 1e-13] [--mutate none|nowarp] [--allow_unraw]

The node state of an hp state at x_A is marched to x_B on the curves of the chosen warp (band.py), with steps of kappa
over the largest transport rate, and compared with hp's node state at x_B, which hp's Taylor integration of the same
node state reproduces to its residual. The warp changes only the curves marched on, so a converged state (in its tau
resolution) must come back to rounding whatever phi. Exit 1 when max |difference| / max |field| > --tol (the record
holds each field's relative difference, the steps and the smallest g_p, g_m and c - x on the curves).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "hp")))
import numpy as np  # noqa: E402

from band import Band, Warp, design_phi, node_full  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("state")
    ap.add_argument("--band", default="0.01,0.02")
    ap.add_argument("--warp", default="test")
    ap.add_argument("--A", type=float, default=1.5e-3)
    ap.add_argument("--eps0", type=float, default=2e-3)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=1e-13)
    ap.add_argument("--mutate", default="none")
    ap.add_argument("--allow_unraw", action="store_true")
    a = ap.parse_args()
    from raw_out import require_raw
    require_raw(a.out, allow=a.allow_unraw)
    d = np.load(a.state)
    meta = json.loads(str(d["meta"]))
    z = d["zh"] + d["zl"]
    M, nodes, D = meta["M"], list(meta["nodes"]), float(meta["D"])
    xa, xb = (float(v) for v in a.band.split(","))
    ia = nodes.index(xa)
    if abs(nodes[ia + 1] - xb) > 1e-15:
        raise SystemExit("the band must be two consecutive nodes")
    sa, sb = node_full(z, M, ia), node_full(z, M, ia + 1)
    om = 2 * np.pi / z[-1]
    N = 2 * M
    th = 2 * np.pi * np.arange(N) / N
    if a.warp == "none":
        phi = np.zeros(N)
    elif a.warp == "test":
        phi = a.A * (np.sin(2 * th + 0.3) + 0.3 * np.cos(4 * th - 1.1))
    else:
        prof = []
        Band(D, N, Warp.none(xa, xb, N)).plan(sa, om, kappa=a.kappa, record=prof)
        phi, _ = design_phi([(xa, sa)] + prof, om, a.eps0, kmax=M // 2)
    bd = Band(D, N, Warp(xa, xb, phi), mutate=a.mutate)
    t = time.time()
    rec = dict(state=a.state, D=D, M=M, band=[xa, xb], warp=a.warp, A=a.A, eps0=a.eps0, kappa=a.kappa,
               mutate=a.mutate, max_phi=float(np.abs(phi).max()), tol=a.tol)
    prof = []
    try:
        pts, se, gp, gm = bd.plan(sa, om, kappa=a.kappa, record=prof)
        rel = np.abs(se - sb).max(axis=1) / np.abs(sb).max(axis=1)
        x = [bd.warp.at(zz)[0] for zz, _ in prof]
        nat = float(min(np.min(st[3] - xx) for (zz, st), xx in zip(prof, x)))
        rec.update(steps=len(pts) - 1, min_gp=gp, min_gm=gm, min_c_minus_x=nat, rel=[float(v) for v in rel],
                   err=float(rel.max()), seconds=round(time.time() - t, 1))
    except (FloatingPointError, RuntimeError) as e:
        rec.update(error=str(e), err=float("inf"))
    rec["pass"] = bool(rec["err"] <= a.tol)
    with open(a.out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    sys.exit(0 if rec["pass"] else 1)


if __name__ == "__main__":
    main()
