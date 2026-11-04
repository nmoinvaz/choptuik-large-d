#!/usr/bin/env python3
"""Fit Delta(D) over every computed point, 3.04 <= D <= 67.8, as one Chebyshev series in the variable zeta that maps
the D-plane cut along [D_s, 3] onto the unit disc (notes/FORMULA_FORMS.md, "One formula for every D"; hp/formula_cut.py evaluates it):

    Delta(D) = sum_{k=0}^{K} c_k T_k(2 zeta - 1),   zeta = u - sqrt(u^2 - 1),   u = (2 n - (1 + n_s)) / (1 - n_s),

n = D - 2, n_s = D_s - 2 (default -9: the cut [-7, 3]); zeta = 0 at D = infinity and 1 at D = 3, so
Delta_inf = sum (-1)^k c_k and Delta(3) = sum c_k. The weighted least squares run at 256 bits (python-flint): the
weighted design has condition 1e11-1e14, and float64 fits of it are wrong. The points: hp/data/delta_dd.csv (no
unresolved runs) and the resolved best rows of hp/data/delta_dd_lowD.csv; sigma is Delta_err (low-D rows: the change
from the lower M), floored at 1e-15; the continuation points that hp/dd_fits_mp.clean drops (60 digits, D >= 10) are
left out and listed in the output. Writes hp/data/formula_cut.json.
    cd hp && python3 formula_cut_fit.py [--K 28] [--ns -9]
"""
import csv
import json
import os
import sys

import numpy as np
from flint import arb, arb_mat, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
ctx.prec = 256
K = int(sys.argv[sys.argv.index("--K") + 1]) if "--K" in sys.argv else 28
NS = float(sys.argv[sys.argv.index("--ns") + 1]) if "--ns" in sys.argv else -9.0


def points():
    """{D: (hi, lo, sigma, run)} for every resolved double-double point"""
    pts = {}
    for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_dd.csv"))):
        if "unresolved" in r["run"]:
            continue
        D = round(float(r["D"]), 6)
        if r["run"].startswith("lowD"):
            err = float(r["Delta_err"]) if r["Delta_err"] else (1.2e-9 if r["run"] == "lowD-M80" else 1e-12)
        else:
            err = float(r["Delta_err"])
        pts[D] = (float(r["Delta_hi"]), float(r["Delta_lo"]), max(err, 1e-15), r["run"])
    for r in csv.DictReader(l for l in open(os.path.join(HERE, "data", "delta_dd_lowD.csv")) if not l.startswith("#")):
        if r["best"] == "1" and r["resolved"] == "1":
            D = round(float(r["D"]), 6)
            if D not in pts:
                c = r["change_from_lower_M"]
                pts[D] = (float(r["Delta_hi"]), float(r["Delta_lo"] or 0), max(abs(float(c)) if c else 1.2e-9, 1e-15), "lowD-M" + r["M"])
    return pts


def dropped():
    """the continuation points D >= 10 that the production cleaning drops (dd_fits_mp.clean)"""
    import dd_fits_mp as F
    pts, _ = F.load_points()
    Ds = sorted(d for d in pts if d >= 10)
    return sorted(set(round(d, 6) for d in Ds) - set(round(d, 6) for d in F.clean(pts, Ds, verbose=False)))


def zeta(d):
    n = arb(float(d)) - 2
    u = (2 * n - (1 + arb(NS))) / (1 - arb(NS))
    return u - (u * u - 1).sqrt()


def cheb(x, k):
    T = [arb(1), x]
    for _ in range(k - 1):
        T.append(2 * x * T[-1] - T[-2])
    return T[:k + 1]


def main():
    pts = points()
    drop = set(dropped())
    Ds = [d for d in sorted(pts) if d not in drop]
    rows, rhs = [], []
    for d in Ds:
        hi, lo, s, _ = pts[d]
        T = cheb(2 * zeta(d) - 1, K)
        rows.append([t / arb(s) for t in T])
        rhs.append((arb(hi) + arb(lo)) / arb(s))
    W = arb_mat(rows)
    Wt = W.transpose()
    N = Wt * W
    g = Wt * arb_mat([[v] for v in rhs])
    Nm = arb_mat([[arb(N[i, j].mid()) for j in range(N.ncols())] for i in range(N.nrows())])
    gm = arb_mat([[arb(g[i, 0].mid())] for i in range(g.nrows())])
    c = Nm.solve(gm)
    c = [arb(c[i, 0].mid()) for i in range(c.nrows())]
    res = []
    for d in Ds:
        hi, lo, s, _ = pts[d]
        T = cheb(2 * zeta(d) - 1, K)
        v = sum((ci * t for ci, t in zip(c, T)), arb(0))
        res.append(float((v - arb(hi) - arb(lo)) / arb(s)))
    res = np.array(res)
    dinf = sum((ci * (1 if k % 2 == 0 else -1) for k, ci in enumerate(c)), arb(0))
    d3 = sum(c, arb(0))
    out = dict(form="Delta(D) = sum_k c_k T_k(2 zeta - 1); zeta = u - sqrt(u^2 - 1); u = (2 (D - 2) - (1 + n_s)) / (1 - n_s)",
               n_s=NS, D_cut=[NS + 2, 3.0], K=K, c=[ci.mid().str(40, radius=False) for ci in c],
               Delta_inf=dinf.mid().str(20, radius=False), Delta_3=d3.mid().str(12, radius=False),
               points=len(Ds), dropped=sorted(drop), max_abs_residual_over_sigma=float(np.abs(res).max()),
               chi2_per_dof=float(np.sum(res ** 2) / (len(Ds) - K - 1)),
               produced_by="hp/formula_cut_fit.py (python-flint, 256 bits) from hp/data/delta_dd.csv and delta_dd_lowD.csv")
    json.dump(out, open(os.path.join(HERE, "data", "formula_cut.json"), "w"), indent=1)
    print(f"K {K}, n_s {NS:g}: {len(Ds)} points ({len(drop)} dropped by the cleaning), max |r|/sigma {np.abs(res).max():.2f}, "
          f"chi2/dof {out['chi2_per_dof']:.3f}; Delta_inf {out['Delta_inf']}, Delta(3) {out['Delta_3']}")


if __name__ == "__main__":
    main()
