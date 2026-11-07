#!/usr/bin/env python3
"""The quad-double ANM chain's Delta (D 79-86, harmonics 5-13 of p1 held on T75's trend, notes/QUAD_DOUBLE.md 13.19)
against the 1/D and 1/n series fitted on the double-double points (dd_fits_mp, 60 digits) up to --Dfit, as
qd_remainder.py tests the unpinned quad-double points at 68.5-73 (13.12.5). The series' spread over models and
degrees is the extrapolation's own uncertainty at each D; a chain value outside it is a bias of the held harmonics
or a term the series lack. The rows of data/delta_qd.csv come first as the reference. --plant E adds E to every chain
value (the control: it must show as a residual E above the spread).
    cd hp && python3 chain_vs_series.py RAWDIR [--Dfit=67.77] [--degs=16,18,20] [--plant=0]
        [--folders=anm-mac/run_trend_pin3_qd3,...,anm-mac/run_trend_pin3_qd9]
"""
import csv
import glob
import json
import os
import sys
import mpmath as mp
import numpy as np
import dd_fits_mp as F
mp.mp.dps = 60
HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
if not args:
    sys.exit(__doc__)
raw = args[0]
Dfit = float(opts.get("Dfit", 67.77))
degs = [int(v) for v in opts.get("degs", "16,18,20").split(",")]
plant = mp.mpf(opts.get("plant", "0"))
folders = opts.get("folders", ",".join(f"anm-mac/run_trend_pin3_qd{k}" for k in (3, 4, 6, 7, 8, 9))).split(",")

pts, _ = F.load_points()
Ds = F.clean(pts, sorted(d for d in pts if 10 <= d <= Dfit), verbose=False)
ref = {}
for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_qd.csv"))):
    ref[float(r["D"])] = sum(mp.mpf(r[f"Delta_q{q}"]) for q in range(4))
chain = {}
for fo in folders:
    # the corrected states only (a run with --save_pred also stores its predictions, pred_pin3_D*.npz)
    for p in sorted(glob.glob(os.path.join(raw, fo, "anm_pin3_D*.npz"))):
        d = np.load(p)
        m = json.loads(str(d["meta"]))
        parts = [d[k] for k in ("zh", "zl", "z2", "z3") if k in d]
        D = mp.mpf(m.get("D_str", repr(m["D"])))
        chain[float(D)] = (D, sum(mp.mpf(float(v[-1])) for v in parts) + plant, fo)
fits = {}
for model in ("1/D", "1/n"):
    for deg in degs:
        ev, mres, _ = F.fit(pts, Ds, model, deg)
        fits[(model, deg)] = ev
        print(f"fit {model} degree {deg}: residual {float(mres):.1e}")
print(f"fit on {len(Ds)} double-double points 10 <= D <= {Dfit}; plant {float(plant):.1e}")
print("| D | source | Delta | " + " | ".join(f"{m} {g}" for m, g in fits) + " | spread | |mean residual| / spread |")
print("|---|---|---|" + "---|" * (len(fits) + 2))
rows = [(D, ref[D], "delta_qd.csv") for D in sorted(ref)] + [(v[0], v[1], os.path.basename(v[2])) for _, v in sorted(chain.items())]
for D, val, src in rows:
    r = [val - ev(mp.mpf(D)) for ev in fits.values()]
    spread = max(r) - min(r)
    mean = sum(r) / len(r)
    print(f"| {mp.nstr(D, 10)} | {src} | {mp.nstr(val, 16)} | " + " | ".join(f"{float(x):+.2e}" for x in r)
          + f" | {float(spread):.1e} | {float(abs(mean) / spread) if spread else float('inf'):.1f} |")
