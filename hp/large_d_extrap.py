"""Delta at new large-D points against the 1/(D-1) series fitted on the existing points (10 <= D <= Dfit),
degrees 14, 16, 18 (dd_fits_mp, 60 digits). 

    cd hp && python3 large_d_extrap.py RUN.jsonl [Dfit=61.03]

The points beyond Dfit test the extrapolation of the fit (notes: README "The large-D limit")."""
import json
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dd_fits_mp as F  # noqa: E402

mp.mp.dps = 60
F.MODELS["1/(D-1)"] = lambda D: 1 / (mp.mpf(D) - 1)
run = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
Dfit = float(sys.argv[2]) if len(sys.argv) > 2 else 61.03
pts, _ = F.load_points()
Ds = F.clean(pts, sorted(pts))
fitD = [d for d in Ds if 10 <= d <= Dfit]
evs = {K: F.fit(pts, fitD, "1/(D-1)", K)[0] for K in (14, 16, 18)}
print(f"fit on {len(fitD)} points, D = {fitD[0]:.2f} .. {fitD[-1]:.4f}")
for r in run:
    if "Delta" not in r:
        continue
    v = mp.mpf(r["Delta"]) + mp.mpf(r.get("Delta_lo", 0.0))
    diffs = "  ".join(f"K{K} {mp.nstr(v - evs[K](r['D']), 3):>10s}" for K in evs)
    print(f"D {r['D']:9.5f}  Delta {mp.nstr(v, 17)}  new - fit: {diffs}")
