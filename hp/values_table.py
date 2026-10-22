#!/usr/bin/env python3
"""The gamma column of the README's "Values" table, and Delta(4), from the double-double data.

- gamma at D = 10-60: the double-double growth rates (data/gamma_dd.jsonl and data/local/gamma_dd.jsonl, states with
  |F| <= 1e-20), taken directly where D is a computed point and otherwise interpolated in D by a local polynomial
  (degree 4 through the 6 nearest points; the error column is its difference from degree 5 through 7 points).
- gamma at D = 70, 100, 1000: extrapolated from fits of lam = 1/gamma in powers of 1/n (n = D - 2) with lam_inf = 2
  fixed, on D >= 24 and D >= 30 with 5-7 powers (as gamma_compare.py); the spread over those fits is printed. These
  assume gamma_inf = 1/2, which the free fits support to about 5e-7 (README, "The formula").
- D = 4: Delta and gamma from the low-D double-double runs at M = 80 (data/lowD_dd.jsonl, data/lowD_gamma.jsonl).

    cd hp && python3 values_table.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(paths):
    rows = {}
    for p in paths:
        for line in open(os.path.join(HERE, p)):
            if line.strip():
                r = json.loads(line)
                if r.get("lam") and r.get("F", 1) <= 1e-20:
                    rows[round(r["D"], 6)] = r
    return rows


def interp(Dp, yp, D, k, deg):
    i = np.argsort(np.abs(Dp - D))[:k]
    c = np.polyfit(Dp[i] - D, yp[i], deg)
    return c[-1]


def main():
    rows = load(["data/gamma_dd.jsonl", "data/local/gamma_dd.jsonl"])
    Dp = np.array(sorted(rows))
    lam = np.array([rows[d]["lam"] for d in Dp])
    print("gamma from the double-double growth rates")
    for D in (10, 12, 15, 20, 30, 40, 50, 60):
        if D in rows:
            print(f"  D = {D:4d}: gamma = {rows[D]['gamma']:.10f}  (computed point)")
            continue
        a, b = interp(Dp, lam, D, 6, 4), interp(Dp, lam, D, 7, 5)
        print(f"  D = {D:4d}: gamma = {1 / a:.10f}  (interpolated; degree 4 vs 5: {abs(1 / a - 1 / b):.1e})")

    good = {}
    for d in Dp:                                   # one point per 0.5 in D, as gamma_compare.py
        if all(abs(d - e) >= 0.5 for e in good):
            good[d] = rows[d]["lam"]
    D = np.array(sorted(good))
    L = np.array([good[d] for d in D])
    n = D - 2
    preds = {70: [], 100: [], 1000: []}
    for lo in (24, 30):
        m = D >= lo
        n0 = n[m].min()
        for k in (5, 6, 7):
            B = np.stack([(n0 / n[m]) ** p for p in range(1, k + 1)], 1)
            c, *_ = np.linalg.lstsq(B, L[m] - 2, rcond=None)
            for Dq in preds:
                nq = Dq - 2
                preds[Dq].append(1 / (2 + sum(c[p - 1] * (n0 / nq) ** p for p in range(1, k + 1))))
    print("gamma extrapolated with lam_inf = 2 fixed (D >= 24 and 30; 5-7 powers of 1/n)")
    for Dq, v in preds.items():
        v = np.array(v)
        print(f"  D = {Dq:4d}: gamma = {np.median(v):.8f}  (spread over the 6 fits {v.max() - v.min():.1e})")

    def last(path, D, M):
        out = None
        for line in open(os.path.join(HERE, path)):
            if line.strip():
                r = json.loads(line)
                if abs(r.get("D", 0) - D) < 1e-9 and r.get("M") == M:
                    out = r
        return out
    rd, rg = last("data/lowD_dd.jsonl", 4.0, 80), last("data/lowD_gamma.jsonl", 4.0, 80)
    print("D = 4 from the low-D double-double runs (M = 80)")
    print(f"  Delta = {rd['Delta'] if rd and 'Delta' in rd else rd}")
    print(f"  gamma = {rg['gamma'] if rg else None}")


if __name__ == "__main__":
    main()
