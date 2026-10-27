#!/usr/bin/env python3
"""What the quad-double points at D = 68.5-73 (data/delta_qd.csv) say beyond the double-double data: the 1/D and
1/n series fitted on the double-double points (dd_fits_mp, 60 digits) up to --Dfit, evaluated at the quad-double
D, against the quad-double values, at 25 digits. The residuals' spread over the series' degrees is the extrapolation's
own uncertainty; a non-perturbative term c e^(-a (D - 1)) could only be seen above it, so the table ends with the
bound on |c| that the residuals give for a = 0.25, 0.5 and 1 (|residual| / e^(-a (D - 1)) at the first point).

    cd hp && python3 qd_remainder.py [--Dfit=67.77] [--degs=16,18,20]
"""
import csv
import os
import sys

import mpmath as mp

import dd_fits_mp as F

mp.mp.dps = 60
HERE = os.path.dirname(os.path.abspath(__file__))
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
Dfit = float(opts.get("Dfit", 67.77))
degs = [int(v) for v in opts.get("degs", "16,18,20").split(",")]
pts, _ = F.load_points()
Ds = sorted(d for d in pts if 10 <= d <= Dfit)
Ds = F.clean(pts, Ds, verbose=False)
qd = {}
for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_qd.csv"))):
    qd[float(r["D"])] = sum(mp.mpf(r[f"Delta_q{q}"]) for q in range(4))
print(f"fit on {len(Ds)} double-double points 10 <= D <= {Dfit}; quad-double points at D = {', '.join(f'{d:g}' for d in sorted(qd))}")
res = {}
for model in ("1/D", "1/n"):
    for deg in degs:
        ev, mres, _ = F.fit(pts, Ds, model, deg)
        row = {D: qd[D] - ev(mp.mpf(D)) for D in sorted(qd)}
        res[(model, deg)] = row
        print(f"{model:>4} deg {deg:2d}  fit residual {float(mres):.1e}  qd - fit: " + " ".join(f"{float(row[D]):+9.1e}" for D in sorted(qd)))
print(f"{'':16}{'':25}D:        " + " ".join(f"{D:9g}" for D in sorted(qd)))
spread = {D: max(abs(v[D]) for v in res.values()) for D in qd}
print("largest |qd - fit| over the models:    " + " ".join(f"{float(spread[D]):9.1e}" for D in sorted(qd)))
D0 = min(qd)
for a in (0.25, 0.5, 1.0):
    print(f"a = {a:4}: a term c e^(-a(D-1)) at D = {D0:g} is {float(mp.exp(-a * (D0 - 1))):.1e} c, so |c| < {float(spread[D0] / mp.exp(-a * (D0 - 1))):.1e} "
          f"(the quad-double point itself is good to 1e-32: the bound is the series' extrapolation, not the point)")
