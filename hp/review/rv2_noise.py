#!/usr/bin/env python3
"""Review RV2: how noisy are the double-double points, by D band? Two estimates that do not use the solver:

  1. residuals of a global fit (1/D, degree 18 and 20, cleaned points 10..61.03): rms and max per band;
  2. local smoothness: for each point, the residual of a degree-6 polynomial fitted (in D) to its 6 nearest
     neighbours on each side, excluding the point itself (leave-one-out). For a smooth function plus iid noise sigma
     this residual has rms about sigma * sqrt(1 + h), h ~ 0.6-1 for this design, so it measures the high-frequency
     part of the per-point error; slow, smooth errors (discretisation bias) are invisible to it.

    python3 hp/review/rv2_noise.py [--break]     (--break adds 1e-15 iid noise to D in 45..55 and must show it)
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import arb, arb_mat  # noqa: E402

import rv2_fits as RF  # noqa: E402

pts, run = RF.load()
if "--break" in sys.argv:
    rng = random.Random(3)
    for d in pts:
        if 45 <= d <= 55:
            pts[d] += arb(rng.gauss(0, 1e-15))
allD = sorted(pts)
keep, drop = RF.clean(pts, [d for d in allD if 10 <= d <= 61.03])
BANDS = ((10, 20), (20, 27), (27, 35.25), (35.25, 37), (37, 45), (45, 55), (55, 60.1), (60.1, 61.03))


def band_stats(vals):
    out = []
    for lo, hi in BANDS:
        v = [x for d, x in vals if lo <= d < hi or (hi == 61.03 and d == hi)]
        if v:
            rms = (sum(x * x for x in v) / len(v)) ** 0.5
            out.append(f"{lo}-{hi}: n {len(v):3d} rms {rms:.1e} max {max(abs(x) for x in v):.1e}")
    return out


for deg, model in ((18, "1/D"), (20, "1/D"), (20, "1/n")):
    f = RF.Fit(pts, keep, model, deg)
    _, res, c = f.solve(deg)
    r = [(d, float((sum((f.rows[i][k] * c[k, 0] for k in range(deg + 1)), arb(0)) - pts[d]).mid()))
         for i, d in enumerate(keep)]
    print(f"global fit {model} degree {deg} on {len(keep)} cleaned points (max {res:.1e}):")
    for s in band_stats(r):
        print("   " + s)

# local leave-one-out smoothness, in exact-ish arithmetic (arb), on all points 8.25..61.03 (not cleaned)
NB, DEG = 6, 6
loo = []
for i, d in enumerate(allD):
    js = [j for j in range(max(0, i - NB), min(len(allD), i + NB + 1)) if j != i]
    if len(js) < 2 * NB:
        continue
    x0 = allD[i]
    sc = max(abs(allD[j] - x0) for j in js)
    X = arb_mat([[arb((allD[j] - x0) / sc) ** k for k in range(DEG + 1)] for j in js])
    y = arb_mat([[pts[allD[j]]] for j in js])
    c = (X.transpose() * X).solve(X.transpose() * y)
    loo.append((d, float((c[0, 0] - pts[d]).mid())))
print(f"local leave-one-out (degree {DEG} in D through {NB} neighbours on each side), all points:")
for s in band_stats(loo):
    print("   " + s)
loo_c = [(d, x) for d, x in loo if d in set(keep)]
print("the same, cleaned points only:")
for s in band_stats(loo_c):
    print("   " + s)
big = sorted(loo, key=lambda t: -abs(t[1]))[:12]
print("largest leave-one-out residuals: " + ", ".join(f"{d:.4f} {run[d]} {x:+.1e}{' (cleaned)' if d not in keep else ''}"
                                                       for d, x in big))

# do the linearised corrections -dDelta/dF . F recorded by the runs predict the leave-one-out residuals?
est = {}
for line in list(open(RF.CSV))[1:]:
    c = line.strip().split(",")
    if c[5]:
        est[float(c[0])] = float(c[5])
print("points with |dDelta_est| > 3e-16: leave-one-out residual (fit - Delta) vs the recorded correction "
      "(Delta_true - Delta ~ dDelta_est, so the two should agree if the estimate is the error):")
for d, x in loo:
    if abs(est.get(d, 0)) > 3e-16:
        print(f"   D {d:10.6f} {run[d]:10s} loo {x:+.1e}  dDelta_est {est[d]:+.1e}")

# cross-machine check (README: the second machine agrees with the cloud points to 2-4e-16 at D = 59.84-60.05): each
# local point against a degree-6 fit through the 12 nearest CLOUD points; the same for each cloud point (leave-one-out
# within the cloud set) as the reference level
cloud = sorted(d for d in allD if not run[d].startswith("local"))
local = sorted(d for d in allD if run[d].startswith("local") and d <= 60.1)


def interp_from(src, d, nb=12, deg=6):
    js = sorted(sorted((x for x in src if x != d), key=lambda x: abs(x - d))[:nb])
    sc = max(abs(x - d) for x in js)
    X = arb_mat([[arb((x - d) / sc) ** k for k in range(deg + 1)] for x in js])
    y = arb_mat([[pts[x]] for x in js])
    c = (X.transpose() * X).solve(X.transpose() * y)
    return float((c[0, 0] - pts[d]).mid())


lv = [interp_from(cloud, d) for d in local]
cv = [interp_from(cloud, d) for d in cloud if 59.0 <= d <= 60.1]
print(f"cross-machine: {len(local)} local points in {local[0]:.4f}..{local[-1]:.4f} vs cloud interpolant: "
      f"max {max(abs(v) for v in lv):.1e} rms {(sum(v * v for v in lv) / len(lv)) ** 0.5:.1e}; cloud leave-one-out "
      f"on 59-60.1: max {max(abs(v) for v in cv):.1e} rms {(sum(v * v for v in cv) / len(cv)) ** 0.5:.1e}")
