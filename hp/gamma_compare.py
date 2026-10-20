#!/usr/bin/env python3
"""Compare the double-double gamma(D) (gamma_dd.py output) with the f64 points (solver/rational_fit.gamma_data,
interpolated to the same D by a local polynomial through the 6 nearest points) and with the README rational formula,
and fit lam = 1/gamma in 1/n for the D = infinity limit (free lam_inf, and lam_inf = 2 fixed).

    python3 gamma_compare.py GAMMA_DD.jsonl [more.jsonl ...]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.join(HERE, "..", "solver")
sys.path.insert(0, HERE)
from gamma_dd import gamma_formula  # noqa: E402


def f64_points():
    cwd = os.getcwd()
    sys.path.insert(0, SOLVER)
    os.chdir(os.path.join(SOLVER, "data"))
    try:
        from rational_fit import gamma_data
        D, g, s = gamma_data()
    finally:
        os.chdir(cwd)
    o = np.argsort(D)
    return D[o], g[o], s[o]


def interp(Dp, gp, D, k=6, deg=4):
    i = np.argsort(np.abs(Dp - D))[:k]
    if not (Dp[i].min() <= D <= Dp[i].max()):
        return None
    c = np.polyfit(Dp[i] - D, gp[i], deg)
    return c[-1]


def main():
    rows = {}
    for f in sys.argv[1:]:
        for l in open(f):
            r = json.loads(l)
            if r.get("lam"):
                rows[round(r["D"], 6)] = r
    Dp, gp, sp = f64_points()
    print(f"{'D':>9s} {'gamma (double-double)':>22s} {'- f64 interp':>13s} {'- formula':>11s}  M  kfreeze  M48 dlam")
    for d in sorted(rows):
        r = rows[d]
        gi = interp(Dp, gp, d)
        m48 = r.get("M48", {}).get("lam_diff")
        print(f"{d:9.4f} {r['gamma']:22.16f} {('%13.2e' % (r['gamma'] - gi)) if gi is not None else '            -'} "
              f"{r['gamma'] - gamma_formula(d):11.2e}  {r['M']}  {str(r['kfreeze']):>4s}  {('%.1e' % m48) if m48 is not None else '-'}")
    # fits of lam(n), n = D - 2, in powers of 1/n (columns scaled by n0 = smallest n in the fit, for conditioning);
    # states with |F| > 1e-20 are left out. Free fits (lam_inf + k powers) against fits with lam_inf = 2 fixed and
    # k + 1 powers (the same number of parameters): equal residuals mean the data cannot tell lam_inf from 2.
    good = {}
    for d in sorted(rows):  # one point per 0.5 in D (the seed clusters near D = 60-61 would otherwise dominate)
        if rows[d]["F"] <= 1e-20 and all(abs(d - e) >= 0.5 for e in good):
            good[d] = rows[d]
    if len(good) >= 6:
        D = np.array(sorted(good))
        lam = np.array([good[d]["lam"] for d in D])
        n = D - 2
        for lo in (16, 20, 24, 30):
            m = D >= lo
            n0 = n[m].min()
            for k in range(4, 9):
                if k + 5 > m.sum():
                    break
                A = np.stack([np.ones(m.sum())] + [(n0 / n[m]) ** p for p in range(1, k + 1)], 1)
                c, *_ = np.linalg.lstsq(A, lam[m], rcond=None)
                r = lam[m] - A @ c
                s_ = max(np.sqrt((r ** 2).sum() / max(m.sum() - k - 1, 1)), 3e-11)  # 3e-11: scatter of four states at D = 61.02
                err = s_ * np.sqrt(np.linalg.inv(A.T @ A)[0, 0])
                B = np.stack([(n0 / n[m]) ** p for p in range(1, k + 2)], 1)
                c2, *_ = np.linalg.lstsq(B, lam[m] - 2, rcond=None)
                r2 = lam[m] - 2 - B @ c2
                print(f"D >= {lo:2d} ({m.sum():2d} points), {k} powers of 1/n: lam_inf - 2 = {c[0] - 2:+.2e} +- {err:.1e} "
                      f"(rms residual {np.sqrt((r ** 2).mean()):.1e}; with lam_inf = 2 fixed and {k + 1} powers: "
                      f"{np.sqrt((r2 ** 2).mean()):.1e}, a1 = {c2[0] * n0:.5f})")

if __name__ == "__main__":
    main()
