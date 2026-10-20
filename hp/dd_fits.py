#!/usr/bin/env python3
"""Large-D model fits on the double-double values only.

    python3 dd_fits.py [Dmin] [Dmax_fit]        (scan jsonl files from /home/claude/dss/hp_scan)
"""
import glob
import json
import sys

import numpy as np
from numpy.polynomial import polynomial as P

Dmin = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
Dfit = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
pts = {}
for f in sorted(glob.glob("/home/claude/dss/hp_scan/scan_up*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        pts[round(r["D"], 6)] = r["Delta"] + r["Delta_lo"]
D = np.array(sorted(pts))
y = np.array([pts[d] for d in D])
m = (D >= Dmin)
D, y = D[m], y[m]
fit = D <= Dfit
hold = ~fit
print(f"{len(D)} DD points, D = {D.min():.2f} .. {D.max():.2f}; fit on D <= {Dfit}")


def lsq(X, y):
    # scaled least squares in extended precision via QR on the f64 design (columns normalised)
    s = np.abs(X).max(0)
    c, *_ = np.linalg.lstsq(X / s, y, rcond=None)
    return c / s


def report(name, X, Xh, lim_index=0):
    c = lsq(X, y[fit])
    r = X @ c - y[fit]
    out = f"{name:34s} max|res| {np.abs(r).max():.1e}"
    if hold.any():
        out += f"  held-out max {np.abs(Xh @ c - y[hold]).max():.1e}"
    print(out + f"  limit {c[lim_index]:.8f}")
    return c


x = 1 / D[fit]
xh = 1 / D[hold]
for deg in (12, 14, 16, 18):
    report(f"1/D series deg {deg}", np.vander(x, deg + 1, increasing=True), np.vander(xh, deg + 1, increasing=True))
for deg in (20, 24, 28):
    t = (D[fit] - 2) ** -0.5
    th = (D[hold] - 2) ** -0.5
    report(f"n^-1/2 series deg {deg}", np.vander(t, deg + 1, increasing=True), np.vander(th, deg + 1, increasing=True))


def predictions(Ds=(40, 45, 50, 60, 80, 100)):
    rows = []
    for deg in (14, 16, 18):
        c = lsq(np.vander(x, deg + 1, increasing=True), y[fit])
        rows.append((f"1/D deg {deg}", [np.polyval(c[::-1], 1 / d) for d in Ds], c[0]))
    for deg in (20, 24, 28):
        c = lsq(np.vander((D[fit] - 2) ** -0.5, deg + 1, increasing=True), y[fit])
        rows.append((f"n^-1/2 deg {deg}", [np.polyval(c[::-1], (d - 2) ** -0.5) for d in Ds], c[0]))
    print(f"{'model':16s} " + " ".join(f"{'D=' + str(d):>16s}" for d in Ds) + f" {'limit':>12s}")
    for name, v, lim in rows:
        print(f"{name:16s} " + " ".join(f"{a:16.12f}" for a in v) + f" {lim:12.8f}")


if __name__ == "__main__":
    predictions()
