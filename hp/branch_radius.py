#!/usr/bin/env python3
"""Measurement B of notes/QUAD_DOUBLE.md 13.11: the branch's radius of convergence in D from the stored states of one
layout (T75: 562 states at D 70.63-73.27, M 32, nodes 0.3-0.7). Every component of the state z(D) and Delta are fitted
by Chebyshev polynomials over the window; for a function analytic inside the Bernstein ellipse of parameter rho about
the window the coefficients fall as rho^-k, so the decay over the orders above the noise floor gives rho and the
distance to the nearest singularity, R_D = (half-width) (rho + 1/rho) / 2 (the ellipse's semi-major axis).

    cd hp && python3 branch_radius.py RAWDIR [--folder T75] [--degree 16] [--out OUT.json]
"""
import argparse
import glob
import json
import os

import numpy as np
from numpy.polynomial import chebyshev as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--folder", default="T75")
    ap.add_argument("--degree", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.raw, a.folder, "*.npz")))
    Ds, Z = [], []
    layout = None
    for p in paths:
        d = np.load(p); m = json.loads(str(d["meta"]))
        key = (m["M"], tuple(m["nodes"]), len(d["zh"]))
        if layout is None:
            layout = key
        if key != layout:
            continue
        Ds.append(m["D"]); Z.append(d["zh"] + d["zl"])
    Ds = np.array(Ds); Z = np.array(Z)
    order = np.argsort(Ds); Ds, Z = Ds[order], Z[order]
    lo, hi = Ds[0], Ds[-1]
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    x = (Ds - mid) / half
    print(f"{len(Ds)} states of layout M {layout[0]}, nodes {list(layout[1])}: D {lo:.4f}..{hi:.4f}, centre {mid:.3f}, half-width {half:.3f}")
    deg = a.degree
    V = C.chebvander(x, deg)
    coef, res, rank, sv = np.linalg.lstsq(V, Z, rcond=None)          # (deg + 1, L)
    scale = np.abs(Z).max(axis=0) + 1e-300
    rel = np.abs(coef) / scale                                         # relative Chebyshev coefficients per component
    fit_err = np.abs(V @ coef - Z).max(axis=0) / scale
    # decay rate: the slope of ln|c_k| over the orders where c_k is above the noise floor
    out = {"folder": a.folder, "n_states": int(len(Ds)), "D_lo": float(lo), "D_hi": float(hi), "centre": float(mid), "half_width": float(half), "degree": deg}

    def rate(c):
        k = np.arange(len(c))
        floor = max(c[-3:].max(), 1e-30)                                # the tail's level: the noise floor
        ok = (c > 10 * floor) & (k >= 1)
        if ok.sum() < 4:
            return None, int(ok.sum()), float(floor)
        kk, cc = k[ok], np.log(c[ok])
        slope = np.polyfit(kk, cc, 1)[0]
        rho = np.exp(-slope)
        return float(rho), int(ok.sum()), float(floor)

    rho_D, nD, floor_D = rate(rel[:, -1])
    RD = lambda rho: half * (rho + 1 / rho) / 2 if rho else None
    out["Delta"] = {"rho": rho_D, "orders_used": nD, "floor": floor_D, "R_D": RD(rho_D), "coef_rel": [float(v) for v in rel[:, -1]], "fit_err": float(fit_err[-1])}
    print(f"Delta: relative Chebyshev coefficients {' '.join(f'{v:.1e}' for v in rel[:, -1])}")
    print(f"Delta: rho {rho_D}, orders used {nD}, noise floor {floor_D:.1e}, R_D {RD(rho_D)}; fit error {fit_err[-1]:.1e}")
    rhos = []
    for i in range(Z.shape[1] - 1):
        if scale[i] < 1e-12 * scale.max():
            continue
        r, n, f = rate(rel[:, i])
        if r:
            rhos.append((r, i, n, f))
    rhos.sort()
    H = layout[0]; K = len(layout[1])
    def name(i):
        if i < H: return f"p1[{i}]"
        if i < 2 * H: return f"c0[{i - H}]"
        if i < 2 * H + 4 * H * K:
            j = i - 2 * H; node, rem = divmod(j, 4 * H); f, t = divmod(rem, H)
            return f"node{node} {'pmac'[f]}[{t}]"
        j = i - 2 * H - 4 * H * K
        return f"{'m_h' if j < H else 'a_h'}[{j % H}]"
    out["state"] = {"smallest_rho": [{"rho": r, "component": name(i), "R_D": RD(r), "orders_used": n, "floor": f} for r, i, n, f in rhos[:8]],
                    "median_rho": float(np.median([r for r, *_ in rhos])) if rhos else None, "n_components": len(rhos)}
    print(f"state: {len(rhos)} components measured; median rho {out['state']['median_rho']:.1f} (R_D {RD(out['state']['median_rho']):.1f}); the smallest:")
    for e in out["state"]["smallest_rho"]:
        print(f"   {e['component']:14s} rho {e['rho']:7.1f}  R_D {e['R_D']:7.1f}  orders {e['orders_used']}  floor {e['floor']:.1e}")
    # per block, over every component whose first three coefficients are above the noise floor: the ratio
    # c_2/c_3 (a direct estimate of rho from the low orders, which are always resolved), its median and the number of
    # components with a ratio below 10 (a nearby singularity or a non-smooth dependence on D)
    blocks = {}
    for i in range(Z.shape[1] - 1):
        if scale[i] < 1e-12 * scale.max():
            continue
        c = rel[:, i]
        floor = max(c[-3:].max(), 1e-30)
        if c[3] > 10 * floor and c[2] > 0:
            blocks.setdefault(name(i).split("[")[0], []).append(c[2] / c[3])
    out["blocks"] = {}
    print("per block: components with c_3 above the floor, median c_2/c_3 (an estimate of rho), smallest, count below 10")
    for b, v in blocks.items():
        v = sorted(v)
        out["blocks"][b] = {"n": len(v), "median_ratio": float(np.median(v)), "min_ratio": float(v[0]), "below_10": int(sum(x < 10 for x in v))}
        print(f"   {b:10s} {len(v):3d}  median {np.median(v):7.1f}  min {v[0]:6.1f}  below 10: {sum(x < 10 for x in v)}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
