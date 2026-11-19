#!/usr/bin/env python3
"""The cut-plane series of notes/FORMULA_FORMS.md beside the quadratic formula, on the same points (post hoc).

    cd solver/quadform && python3 cut_compare.py [--file=data/formula_q24.json] [--out=data/cut_compare.json]

On the points of algfit.load_points with D <= 67.77: the stored series (hp/data/formula_cut.json, 29 constants, fitted
to an earlier table), the same series refitted with K + 1 = 24, 26 and 29 constants (weighted least squares at 1024
bits, as hp/formula_cut_fit.py), and the quadratic formula of --file. For each: chi2/dof and the largest |miss|/sigma
on the fitted points, then the largest |miss| and |miss|/sigma on the points above 67.77, which no fit saw, and for
the quadratic the same by kind of point.
"""
import json
import os
import sys

from flint import arb, arb_mat

import algfit as A
import formula as F

ARGS = A.ARGS
HELD = [(67.77, 130), (130, 245.64)]


def zeta_row(D, K, ns=-9.0):
    n = arb(D) - 2
    u = (2 * n - (1 + arb(ns))) / (1 - arb(ns))
    return A._cheb(2 * (u - (u * u - 1).sqrt()) - 1, K)


def refit(pts, K):
    rows = [[t / arb(s) for t in zeta_row(D, K)] for D, _, s, _ in pts]
    W = arb_mat(rows)
    Wt = W.transpose()
    N, g = Wt * W, Wt * arb_mat([[y / arb(s)] for _, y, s, _ in pts])
    c = arb_mat([[arb(N[i, j].mid()) for j in range(K + 1)] for i in range(K + 1)]).solve(
        arb_mat([[arb(g[i, 0].mid())] for i in range(K + 1)]))
    c = [arb(c[i, 0].mid()) for i in range(K + 1)]
    return lambda D: sum((ci * t for ci, t in zip(c, zeta_row(D, K))), arb(0))


def report(name, n_const, fn, fit_pts, all_pts):
    r = [float((fn(D) - y) / arb(s)) for D, y, s, _ in fit_pts]
    row = dict(name=name, n_const=n_const, chi2dof=sum(v * v for v in r) / (len(r) - n_const),
               max_over_sigma=max(abs(v) for v in r), held={})
    for lo, hi in HELD:
        h = [(abs(float(fn(D) - y)), s) for D, y, s, _ in all_pts if lo < D <= hi + 1e-9]
        row["held"][f"{lo:g}-{hi:g}"] = dict(max_miss=max(m for m, _ in h), max_over_sigma=max(m / s for m, s in h),
                                            n=len(h))
    print(f"  {name:34s} N {n_const:2d} chi2/dof {row['chi2dof']:.3g} max/sigma {row['max_over_sigma']:.3g}; held out: "
          + "; ".join(f"D {k} {v['max_miss']:.1e} ({v['max_over_sigma']:.3g} sigma)" for k, v in row["held"].items()),
          flush=True)
    return row


def main():
    allp = A.load_points()
    fit_pts = [p for p in allp if p[0] <= 67.77 + 1e-9]
    print(f"{len(fit_pts)} points to D 67.77, {len(allp) - len(fit_pts)} above")
    rows = [report("cut series, stored (earlier table)", 29, A.plant_fn("T_cut"), fit_pts, allp)]
    for K in (23, 25, 28):
        rows.append(report(f"cut series, refitted, K = {K}", K + 1, refit(fit_pts, K), fit_pts, allp))
    path = ARGS.get("file", os.path.join(A.HERE, "data", "formula_q24.json"))
    Q = json.load(open(path))
    rows.append(report(f"quadratic {tuple(Q['degrees_pqr'])}, float64", Q["n_const"],
                       lambda D: arb(F.f64_value(Q, D)), fit_pts, allp))
    kinds = {}
    for D, y, s, k in allp:                                 # the quadratic against the unseen points, by kind
        if D > 67.77 + 1e-9:
            m = abs(F.f64_value(Q, D) - float(y))
            kinds.setdefault(k, []).append((m / s, m, D))
    by_kind = {}
    for k, v in kinds.items():
        w = max(v)
        by_kind[k] = dict(n=len(v), max_miss=max(m for _, m, _ in v), worst_over_sigma=w[0], worst_at=w[2],
                          over_5_sigma=sum(a > 5 for a, _, _ in v),
                          over_5_sigma_D=[min((d for a, _, d in v if a > 5), default=None),
                                          max((d for a, _, d in v if a > 5), default=None)])
        print(f"  unseen {k:9s} n {len(v):4d} max miss {by_kind[k]['max_miss']:.1e} worst {w[0]:.3g} sigma at D {w[2]:.4f}; "
              f"{by_kind[k]['over_5_sigma']} beyond 5 sigma, D {by_kind[k]['over_5_sigma_D']}")
    json.dump(dict(points=len(fit_pts), rows=rows, quadratic_unseen_by_kind=by_kind,
                   produced_by="solver/quadform/cut_compare.py"),
              open(ARGS.get("out", os.path.join(A.HERE, "data", "cut_compare.json")), "w"), indent=1)


if __name__ == "__main__":
    sys.argv.insert(1, "cut_compare") if len(sys.argv) < 2 or sys.argv[1].startswith("--") else None
    main()
