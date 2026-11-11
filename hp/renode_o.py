#!/usr/bin/env python3
"""An eight-part chain state re-noded to another node layout and saved in eight parts, a start for anm_run.py's base
polish on that layout (#489).

    cd hp && python3 renode_o.py RAWDIR STATE OUT --nodes 0.3,0.4,0.5,0.6,0.7,0.8 [--last horizon|integrate|f64]
        [--threads 2] [--mutate sign]

STATE and OUT are relative to RAWDIR; OUT must be in a task folder of the raw-data repository (hp/raw_out.py).
hpkernel.driver.renode gives each new node's state in double-double (beyond="horizon": a node past the old last one
from the horizon series, summed in f64). The centre data, the horizon data, Delta and the nodes both layouts share keep
all eight stored parts. Each new node is then refined in octo-double, from the centre out, by its own block of the
unfrozen kernel's residual (Solver.residual_o, kfreeze off). A block's rows are the state arriving at its target node
minus that node, and the arriving state does not depend on the target, so adding the rows makes the node the arriving
state in one step. A new node inside the old range takes the centre series or the Taylor steps from the node before.
The last node, if new, takes --last: 'horizon' (the default) the horizon series at u0 = 1 - x_last from the stored
horizon data, which is renode's choice carried to eight parts; 'integrate' the Taylor steps from the node before;
'f64' renode's state as it stands. The plan keeps each stored segment whose two ends are adjacent nodes of both
layouts, and plans the others (Solver.plan at 1e-32, 0.05, as anm_run.py plans a base).

Printed: the stored state's pinned residual on its own layout (the held offset at the state's own coefficients, as
anm_run.py's base polish forms it), which must be the run's converged value; the gap between the last node from the
horizon series and from the Taylor steps, by field and harmonic class (p's harmonics k <= kfreeze are matched at the
horizon, the higher ones are not); and the start's pinned residual by block.

Check (exit 1): after its refinement each refined node's block rows are below 1e-60 (max |F|), which holds only if
the rows are the arriving state minus the node with unit weight. Control --mutate sign subtracts the rows instead,
and the check must fail.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hpkernel.driver import make_solver, renode  # noqa: E402
from multidouble import od_axpy  # noqa: E402
from near_null import harmonics  # noqa: E402
from octo_corrector import PinnedO, load_o, o8, save_o  # noqa: E402
from pinning import PinnedSolver, band_coeffs, band_field  # noqa: E402
from raw_out import require_raw  # noqa: E402

FIELDS = (("p", 1), ("m", 1), ("a", 0), ("c", 0))


def pinned(s, z, M, kf):
    """PinnedO on s with the held offset at z's own coefficients of p1's harmonics (kf, 13], in eight parts: the base
    polish's system at the state's D (the fit's increment is zero there)"""
    ks = [k for k in range(kf + 1, 14) if k % 2 == 1]
    c = band_coeffs(z[0][:M], z[1][:M], M, ks, exact=True, rest=[np.asarray(p)[:M] for p in z[2:]])
    h = band_field(c, ks, M, parts=8)
    sp = PinnedSolver(s, M, h[0], h[1], h_rest=h[2:])
    return PinnedO(s, sp.ch, sp.cl, c8=sp.c8)


def fmax(F, rows=slice(None)):
    return float(np.abs(np.asarray(F[0])[rows] + np.asarray(F[1])[rows]).max())


def blocks(F, M, K, kf):
    """max |F| per residual block, the horizon block's p rows split into matched (k <= kf) and held harmonics"""
    f = np.asarray(F[0]) + np.asarray(F[1])
    out = {}
    for b in range(K + 1):
        name = "centre>n0" if b == 0 else (f"horizon>n{K - 1}" if b == K else f"n{b - 1}>n{b}")
        out[name] = float(np.abs(f[4 * M * b:4 * M * (b + 1)]).max())
    hm = harmonics(f[4 * M * K:4 * M * K + M], 1)
    out[f"horizon p k<={kf}"] = max(v for k, v in hm.items() if k <= kf)
    out[f"horizon p k>{kf}"] = max(v for k, v in hm.items() if k > kf)
    out["phase"] = float(abs(f[-1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("state")
    ap.add_argument("out")
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--last", default="horizon", choices=("horizon", "integrate", "f64"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--mutate", default="", choices=("", "sign"))
    a = ap.parse_args()
    out_path = os.path.join(a.raw, a.out)
    require_raw(out_path)
    sign = -1.0 if a.mutate == "sign" else 1.0
    z, m = load_o(os.path.join(a.raw, a.state))
    M, old = int(m["M"]), [float(x) for x in m["nodes"]]
    new = [float(x) for x in a.nodes.split(",")]
    D = m.get("D_str", repr(float(m["D"])))
    kt, kc, ks, kf = m.get("kt", 36), m.get("kc", 146), m.get("ks", 130), int(m.get("kfreeze", 3))
    K0, K = len(old), len(new)
    mk = lambda nodes, kfz: make_solver(D, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=a.threads, kfreeze=kfz)  # noqa: E731

    # the stored state on its own layout and plan
    s0 = mk(old, kf)
    s0.set_segs(m["segs"])
    F0 = pinned(s0, z, M, kf).residual_o(z)
    print(f"{a.state}: D {float(m['D']):.10g}, nodes {old}, its pinned max|F| {fmax(F0):.2e}", flush=True)
    print("  by block: " + "; ".join(f"{k} {v:.2e}" for k, v in blocks(F0, M, K0, kf).items()), flush=True)

    # renode in double-double; the shared parts keep all eight
    zh, zl = renode(s0, z[0], z[1], old, new, M, beyond="horizon")
    L = 4 * M * (K + 1) + 1
    zn = [np.zeros(L) for _ in range(8)]
    zn[0][:], zn[1][:] = zh, zl
    node = lambda i: slice(2 * M + 4 * M * i, 2 * M + 4 * M * (i + 1))  # noqa: E731
    for q in range(2, 8):
        zn[q][:2 * M] = z[q][:2 * M]
        zn[q][2 * M + 4 * M * K:] = z[q][2 * M + 4 * M * K0:]
    refine = []
    for i, x in enumerate(new):
        if x in old:
            for q in range(8):
                zn[q][node(i)] = z[q][node(old.index(x))]
        else:
            refine.append(i)
    print(f"renode to {new}: new nodes {[new[i] for i in refine]}, last node by {a.last}", flush=True)

    # the plan: stored segments between nodes adjacent in both layouts, the others planned
    s1 = mk(new, None)
    planned = s1.plan(zn[0], zn[1], 1e-32, 0.05)
    stored = {(sg[0], sg[-1]): sg for sg in m["segs"]}
    segs = [stored.get((sg[0], sg[-1]), sg) for sg in planned]
    kept = [i for i, sg in enumerate(planned) if (sg[0], sg[-1]) in stored]
    s1.set_segs(segs)
    print(f"plan: {sum(len(sg) - 1 for sg in segs)} steps, segments kept {kept}, planned "
          f"{[i for i in range(len(segs)) if i not in kept]} ({[len(segs[i]) - 1 for i in range(len(segs)) if i not in kept]} steps)", flush=True)

    ok, rec, gap = True, [], None
    for i in refine:
        F = o8(s1.residual_o(list(o8(zn))))
        if i == K - 1:
            # the last node both ways: from the Taylor steps (block K - 1) and from the horizon series (block K)
            dv = (F[0][4 * M * K:4 * M * (K + 1)] - F[0][4 * M * (K - 1):4 * M * K]) + (F[1][4 * M * K:4 * M * (K + 1)] - F[1][4 * M * (K - 1):4 * M * K])
            gap = {}
            for j, (f, par) in enumerate(FIELDS):
                h = harmonics(dv[j * M:(j + 1) * M], par)
                gap[f] = dict(all=float(np.abs(dv[j * M:(j + 1) * M]).max()),
                              low=max(v for k, v in h.items() if k <= kf), high=max(v for k, v in h.items() if k > kf))
            print(f"gap at node {new[i]:g}, horizon series minus Taylor steps: "
                  + "; ".join(f"{f} {g['all']:.2e} (k<={kf} {g['low']:.2e}, k>{kf} {g['high']:.2e})" for f, g in gap.items()), flush=True)
            if a.last == "f64":
                continue
        b = K if (i == K - 1 and a.last == "horizon") else i
        rows = slice(4 * M * b, 4 * M * (b + 1))
        before = fmax(F, rows)
        upd = o8(od_axpy(sign, *[np.ascontiguousarray(F[q][rows]) for q in range(8)],
                         *[np.ascontiguousarray(zn[q][node(i)]) for q in range(8)]))
        for q in range(8):
            zn[q][node(i)] = upd[q]
        after = fmax(o8(s1.residual_o(list(o8(zn)))), rows)
        good = after < 1e-60
        ok &= good
        rec.append(dict(node=new[i], block=b, F_before=before, F_after=after))
        how = "horizon series" if b == K else "centre series" if b == 0 else "Taylor steps"
        print(f"node {new[i]:g} refined by block {b} ({how}): its rows {before:.2e} -> {after:.2e} ({'ok' if good else 'FAIL'})", flush=True)

    zn = o8(zn)
    s1p = mk(new, kf)
    s1p.set_segs(segs)
    Fp = pinned(s1p, zn, M, kf).residual_o(zn)
    bl = blocks(Fp, M, K, kf)
    print(f"start, pinned max|F| {fmax(Fp):.2e}; by block: " + "; ".join(f"{k} {v:.2e}" for k, v in bl.items()), flush=True)
    print(f"start Delta {zn[0][-1]:.17f} {zn[1][-1]:+.4e} (the stored state's)", flush=True)
    meta = {k: v for k, v in m.items() if k != "segs"}
    meta.update(nodes=new, segs=[[float(x) for x in sg] for sg in segs], parts=8, source=a.state,
                renode=dict(nodes_old=old, last=a.last, refined=rec, gap=gap, start_blocks=bl, mutate=a.mutate))
    save_o(out_path, zn, meta)
    print(f"saved {a.out}", flush=True)
    print("check:", "pass" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
