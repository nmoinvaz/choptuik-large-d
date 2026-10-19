#!/usr/bin/env python3
"""Double-double scan values against the f64 data set (data_all.load_all) at the same D (to 1e-6 in D).

    python3 compare_old.py SCAN.jsonl [SCAN2.jsonl ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/claude/dss-residual-kernel/solver")
os.chdir("/home/claude/dss-residual-kernel/solver/data")
from data_all import load_all  # noqa: E402

D, y, s, g = load_all()
recs = {}
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l)
        recs[round(r["D"], 6)] = r
print(f"{'D':>8s} {'Delta (DD)':>20s} {'f64 data':>18s} {'DD - f64':>10s} {'f64 sigma':>9s}")
for d in sorted(recs):
    i = np.argmin(abs(D - d))
    if abs(D[i] - d) < 1e-6:
        r = recs[d]
        v = r["Delta"] + r["Delta_lo"]
        print(f"{d:8.4f} {v:20.16f} {y[i]:18.12f} {v - y[i]:+10.2e} {s[i]:9.1e}")
