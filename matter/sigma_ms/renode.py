#!/usr/bin/env python3
"""A state of run_eta.py carried to another node set (#916): the seed of a run that tests the x direction.

The centre data, the horizon data and Delta are kept. The state at each new node that is not an old node is the
kernel's own Taylor integration (Solver.integrate) from the old node below it, through the old state's planned
steps. So the new state solves the system on the new nodes up to the integration's error, and run_eta.py --nodes
... --from converges on it in a step or two.

    PYTHONPATH=<site> python3 renode.py STATE.npz --nodes 0.001,0.003,... --out NEW.npz [--threads 2]

The first and the last node must stay, since the centre and horizon series are evaluated there.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smdriver as sd  # noqa: E402


def renode(z, meta, new, threads=2):
    """(zh, zl) on the nodes `new` from the state z with its meta (nodes, segs, eta, D, M, series orders)"""
    old, M = list(meta["nodes"]), int(meta["M"])
    assert new[0] == old[0] and new[-1] == old[-1], "the first and the last node must stay"
    assert all(a < b for a, b in zip(new, new[1:])), "the nodes must increase"
    eta = None if meta["eta"] == "inf" else float(meta["eta"])
    s = sd.make_solver(eta, M, nodes=old, kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], threads=threads,
                       D=float(meta.get("D", 4.0)))
    s.set_segs(meta["segs"])
    zh, zl = np.asarray(z[0], float), np.asarray(z[1], float)
    K = len(old)
    sth, stl = zh[2 * M:2 * M + 5 * M * K].reshape(K, 5 * M), zl[2 * M:2 * M + 5 * M * K].reshape(K, 5 * M)
    segs = s.get_segs()
    oh, ol = [], []
    for x in new:
        if x in old:
            i = old.index(x)
            oh.append(sth[i])
            ol.append(stl[i])
            continue
        i = max(j for j in range(K) if old[j] < x)
        xs = [p for p in segs[i] if p < x - 1e-14] + [x]
        assert abs(xs[0] - old[i]) < 1e-15, "the planned steps do not start at the node"
        h, lo = s.integrate(zh, zl, np.ascontiguousarray(sth[i]), np.ascontiguousarray(stl[i]), xs)
        oh.append(np.asarray(h))
        ol.append(np.asarray(lo))
    o = 2 * M + 5 * M * K
    return (np.concatenate([zh[:2 * M]] + oh + [zh[o:]]), np.concatenate([zl[:2 * M]] + ol + [zl[o:]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    z, meta = sd.load(a.state)
    new = [float(x) for x in a.nodes.split(",")]
    zh, zl = renode(z, meta, new, a.threads)
    meta2 = dict(meta, nodes=new, segs=[], tag=str(meta.get("tag", "")) + "_renoded")
    sd.save(a.out, zh, zl, meta2)
    print(f"renode: {len(meta['nodes'])} nodes -> {len(new)}, {len(zh)} unknowns, Delta {zh[-1]:.15f} -> {a.out}")


if __name__ == "__main__":
    main()
