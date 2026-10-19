#!/usr/bin/env python3
"""f64 data set (data_all.load_all) against the double-double scans, by local polynomial interpolation of the DD values
(degree 5 through the 6 nearest DD points; interpolation error ~1e-11 for spacing <= 0.35).

    python3 compare_interp.py SCAN.jsonl [...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/claude/dss-residual-kernel/solver")
os.chdir("/home/claude/dss-residual-kernel/solver/data")
from data_all import load_all  # noqa: E402

D, y, s, g = load_all()
pts = {}
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l)
        pts[round(r["D"], 6)] = r["Delta"] + r["Delta_lo"]
Dd = np.array(sorted(pts))
yd = np.array([pts[d] for d in Dd])


def interp(d, deg=5):
    i = np.argsort(np.abs(Dd - d))[:deg + 1]
    i.sort()
    x = Dd[i] - d
    c = np.polyfit(x, yd[i], deg)
    # error estimate: difference to degree deg-1 through the deg nearest
    j = np.argsort(np.abs(Dd - d))[:deg]
    c2 = np.polyfit(Dd[j] - d, yd[j], deg - 1)
    return c[-1], abs(c[-1] - c2[-1]), np.abs(x).max()


rows = []
for d, v, sg in zip(D, y, s):
    if Dd.min() <= d <= Dd.max():
        val, err, span = interp(d)
        rows.append((d, v - val, sg, err))
print(f"{'D':>8s} {'f64 - DD':>10s} {'f64 sigma':>9s} {'interp err':>10s}")
for d, dv, sg, e in rows:
    print(f"{d:8.4f} {dv:+10.2e} {sg:9.1e} {e:10.1e}")
