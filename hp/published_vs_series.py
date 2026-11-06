#!/usr/bin/env python3
"""The large-D gap between our chain and arXiv:2611.07007, against a reference that is neither: the 1/D and 1/n series
fitted on our full-system double-double points 10 <= D <= --Dfit (dd_fits_mp, as chain_vs_series.py, 60 digits),
extrapolated to each published D >= --Dmin. For each D it prints published - series, ours - series (ours from
data/delta_published_compare.csv: the chain interpolated, compare_published.py) and the series' spread over models and
degrees. If the published values follow the series and ours fall below it, the gap is our held harmonics' bias
(notes/QUAD_DOUBLE.md 13.19); the reverse puts it on the published side. --plant E adds E to the published values
(the control: published - series must move by E).
    cd hp && python3 published_vs_series.py [--Dfit=67.77] [--Dmin=75] [--degs=16,18,20] [--plant=0] [--every=10]
"""
import csv
import os
import sys

import mpmath as mp

import dd_fits_mp as F

mp.mp.dps = 60
HERE = os.path.dirname(os.path.abspath(__file__))
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
Dfit, Dmin = float(opts.get("Dfit", 67.77)), float(opts.get("Dmin", 75))
degs = [int(v) for v in opts.get("degs", "16,18,20").split(",")]
plant, every = mp.mpf(opts.get("plant", "0")), int(opts.get("every", 10))
pts, _ = F.load_points()
Ds = F.clean(pts, sorted(d for d in pts if 10 <= d <= Dfit), verbose=False)
fits = {}
for model in ("1/D", "1/n"):
    for deg in degs:
        fits[(model, deg)] = F.fit(pts, Ds, model, deg)[0]
rows = [r for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_published_compare.csv")))
        if r["published_ref"] == "EEG2026" and float(r["D"]) >= Dmin and r["ours_Delta"]]
print(f"series fitted on {len(Ds)} points 10 <= D <= {Dfit}, models 1/D, 1/n, degrees {degs}; plant {float(plant):.1e}")
print("| D | published - series | ours - series | series spread | ours - published |")
print("|---|---|---|---|---|")
for i, r in enumerate(rows):
    D = mp.mpf(r["D"])
    if i % every and float(r["D"]) not in (80.0, 100.0, 150.0, 200.0):
        continue
    s = [ev(D) for ev in fits.values()]
    mean = sum(s) / len(s)
    pub, ours = mp.mpf(r["published_Delta"]) + plant, mp.mpf(r["ours_Delta"])
    print(f"| {r['D']} | {float(pub - mean):+.2e} | {float(ours - mean):+.2e} | {float(max(s) - min(s)):.1e} | "
          f"{float(ours - pub):+.2e} |", flush=True)
