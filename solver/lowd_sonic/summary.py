#!/usr/bin/env python3
"""Table of hybrid.py records (the converged D of one or more OUT.jsonl files, sorted by D): Delta, its first and second
differences in D, the band's smallest c - x (on the warped curves' step points) and smallest g_p, and the zero of the
line through the two smallest |c - x| on either side of zero (where the patch c < x appears).

    python3 summary.py A.jsonl [B.jsonl ...]
"""
import json
import sys

rows = {}
for f in sys.argv[1:]:
    for line in open(f):
        r = json.loads(line)
        if r.get("converged"):
            rows[r["D"]] = r
D = sorted(rows, reverse=True)
print(f"{'D':>9} {'M':>4} {'Delta':>20} {'dDelta/dD':>12} {'d2Delta/dD2':>12} {'min(c-x)':>12} {'min g_p':>10} its")
prev = None
for i, d in enumerate(D):
    r = rows[d]
    d1 = d2 = ""
    if i >= 1:
        d1 = f"{(rows[D[i - 1]]['Delta'] - r['Delta']) / (D[i - 1] - d):12.6f}"
    if i >= 2:
        a, b, c = D[i - 2], D[i - 1], d
        fa, fb, fc = rows[a]["Delta"], rows[b]["Delta"], rows[c]["Delta"]
        d2 = f"{2 * (fa / ((a - b) * (a - c)) + fb / ((b - a) * (b - c)) + fc / ((c - a) * (c - b))):12.4f}"
    print(f"{d:9.5f} {r['M']:4d} {r['Delta']:20.16f} {d1:>12} {d2:>12} {r['min_c_minus_x']:12.4e} {r['min_gp']:10.3e} {r['its']}")
pos = [d for d in D if rows[d]["min_c_minus_x"] > 0]
neg = [d for d in D if rows[d]["min_c_minus_x"] < 0]
if pos and neg:
    a, b = min(pos), max(neg)
    ga, gb = rows[a]["min_c_minus_x"], rows[b]["min_c_minus_x"]
    print(f"min(c - x) changes sign between D {b} and {a}; the line through them is zero at D {a - ga * (a - b) / (ga - gb):.6f}")
