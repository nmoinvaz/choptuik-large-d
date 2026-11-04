#!/usr/bin/env python3
"""The local log-slope L(D) = (D - 3) Delta'/Delta of the low-D points (data/delta_dd_lowD.csv, the largest M at each
D, central differences between neighbours, D <= --Dmax) and of the sqrt(D - 3) formula (plot_lowD_values.C_DELTA). A
power law Delta = c (D - 3)^alpha has L = alpha at every D; the formula's L goes to 0 as 0.71 sqrt(D - 3). Prints the
table and the largest |measured - formula|.
    cd hp && python3 lowD_logslope.py [--Dmax=3.3]
"""
import csv
import math
import os
import sys
import plot_lowD_values as P
HERE = os.path.dirname(os.path.abspath(__file__))
Dmax = float(dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a).get("Dmax", 3.3))
rows = [r for r in csv.DictReader(l for l in open(os.path.join(HERE, "data", "delta_dd_lowD.csv")) if not l.startswith("#"))]
best = {}
for r in rows:
    D, M = float(r["D"]), int(r["M"])
    if D <= Dmax and (D not in best or M > best[D][0]):
        best[D] = (M, float(r["Delta_hi"]) + float(r["Delta_lo"] or 0))
Ds = sorted(best)
worst = 0.0
print("| D (midpoint) | s = sqrt(D - 3) | L measured | L of the formula |")
print("|---|---|---|---|")
for a, b in zip(Ds, Ds[1:]):
    Dm = (a + b) / 2
    L = (Dm - 3) * (best[b][1] - best[a][1]) / (b - a) / ((best[a][1] + best[b][1]) / 2)
    h = 1e-6
    Lf = (Dm - 3) * (P.formula(P.C_DELTA, Dm + h) - P.formula(P.C_DELTA, Dm - h)) / (2 * h) / P.formula(P.C_DELTA, Dm)
    worst = max(worst, abs(L - Lf))
    print(f"| {Dm:.6f} | {math.sqrt(Dm - 3):.4f} | {L:.5f} | {Lf:.5f} |")
print(f"largest |measured - formula| {worst:.1e}; L of the formula at 3.0425, 3.04, 3.02: "
      + ", ".join(f"{(D - 3) * (P.formula(P.C_DELTA, D + 1e-6) - P.formula(P.C_DELTA, D - 1e-6)) / 2e-6 / P.formula(P.C_DELTA, D):.4f}" for D in (3.0425, 3.04, 3.02)))
