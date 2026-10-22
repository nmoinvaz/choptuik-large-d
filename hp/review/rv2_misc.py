#!/usr/bin/env python3
"""Review RV2: the small numbers quoted in notes/NUMERICS_REVIEW.md that no other rv2 script prints.

    python3 hp/review/rv2_misc.py          (about 2 minutes: item 2 runs hp/dd_fits_mp.py's own cleaning, in mpmath)

1. the error the fits' loader (hp/dd_fits_mp.load_points) makes by parsing the CSV's shortest-repr strings at 60 digits;
2. the cleaning sets of dd_fits_mp.clean (repository code, that loader) and rv2_fits.clean (exact values), and the point
   where they differ;
3. the degree-18 1/D limit on the cleaned 10..61.03 points with exact and with decimal-parsed values, against the
   P(0) printed in hp/data/planted_dinf.log;
4. leave-one-out residuals of the two kept points with bad centre-series tails (D = 23.952249, 26.82972);
5. sign statistics of the recorded dDelta_est (run up6; points above 60.1);
6. the double-double 2 pi of hp/src/dd.rs.
Exits non-zero if the loader error is absent (it would mean the CSV or the loader changed) or the 2 pi split is wrong.
"""
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
HP = os.path.dirname(HERE)
sys.path.insert(0, HP)
sys.path.insert(0, HERE)
import dd_fits_mp as F  # noqa: E402  (sets mp.dps = 60)
import rv2_fits as RF  # noqa: E402
from flint import arb, arb_mat  # noqa: E402

rows = [line.strip().split(",") for line in list(open(RF.CSV))[1:]]
fails = 0

# 1
err = [(float(r[0]), float((mp.mpf(r[1]) + mp.mpf(r[2])) - (mp.mpf(float(r[1])) + mp.mpf(float(r[2]))))) for r in rows]
rms = lambda v: (sum(x * x for x in v) / len(v)) ** 0.5  # noqa: E731
print(f"1. loader parse error: max {max(abs(e) for _, e in err):.2e}, rms {rms([e for _, e in err]):.2e}; "
      f"D > 45: max {max(abs(e) for d, e in err if d > 45):.2e}, rms {rms([e for d, e in err if d > 45]):.2e}")
fails += max(abs(e) for _, e in err) < 1e-17

# 2
mpts, _ = F.load_points()
ept, run = RF.load()
for dmin, dmax in ((10, 60.1), (10, 61.03)):
    Ds = sorted(d for d in mpts if dmin <= d <= dmax)
    theirs = set(Ds) - set(F.clean(mpts, Ds, verbose=False))
    mine = set(RF.clean(ept, sorted(d for d in ept if dmin <= d <= dmax))[1])
    print(f"2. cleaning {dmin}..{dmax}: dd_fits_mp drops {len(theirs)}, rv2 (exact values) drops {len(mine)}; "
          f"only dd_fits_mp: {sorted(theirs - mine)}, only rv2: {sorted(mine - theirs)}")
Ds = sorted(d for d in mpts if 10 <= d <= 60.1)
ev, _, _ = F.fit(mpts, Ds, "1/D", 18)
r = [ev(d) - mpts[d] for d in Ds]
f = RF.Fit(ept, Ds, "1/D", 18)
_, _, c = f.solve(18)
r2 = [float((sum((f.rows[i][k] * c[k, 0] for k in range(19)), arb(0)) - ept[d]).mid()) for i, d in enumerate(Ds)]
i = Ds.index(10.75)
med = lambda v, i: (sorted(v[j] for j in range(i - 2, i + 3) if j != i)[1] + sorted(v[j] for j in range(i - 2, i + 3) if j != i)[2]) / 2  # noqa: E731
print(f"   D = 10.75, residual minus neighbours' median: decimal-parsed {float(r[i] - med(r, i)):.3e}, exact {r2[i] - med(r2, i):.3e}"
      " (threshold 1e-15)")

# 3
for dec in (False, True):
    RF.DECIMAL = dec
    pts, _ = RF.load()
    keep, _ = RF.clean(pts, [d for d in sorted(pts) if 10 <= d <= 61.03])
    lim, res, _ = RF.Fit(pts, keep, "1/D", 18).solve(18)
    print(f"3. degree-18 1/D limit, {len(keep)} cleaned points 10..61.03, {'decimal-parsed' if dec else 'exact'}: "
          f"{lim.str(15, radius=False)} (max residual {res:.4e})")
RF.DECIMAL = False
print("   hp/data/planted_dinf.log: " + next(line.strip() for line in open(os.path.join(HP, "data", "planted_dinf.log"))
                                          if line.startswith("base:")))

# 4
allD = sorted(ept)
for target in (23.952249, 26.82972):
    i = allD.index(target)
    js = [j for j in range(i - 6, i + 7) if j != i]
    sc = max(abs(allD[j] - target) for j in js)
    X = arb_mat([[arb((allD[j] - target) / sc) ** k for k in range(7)] for j in js])
    y = arb_mat([[ept[allD[j]]] for j in js])
    cc = (X.transpose() * X).solve(X.transpose() * y)
    print(f"4. D = {target} ({run[target]}): leave-one-out residual {float((cc[0, 0] - ept[target]).mid()):+.1e}, "
          f"spacing {allD[i + 1] - allD[i]:.3f}")

# 5
for label, sel in (("up6", lambda r: r[6] == "up6"), ("D > 60.1", lambda r: float(r[0]) > 60.1)):
    v = [float(r[5]) for r in rows if sel(r) and r[5]]
    print(f"5. dDelta_est, {label}: {len(v)} points, {sum(x > 0 for x in v)} positive, {sum(x < 0 for x in v)} negative, "
          f"mean {sum(v) / len(v):+.2e}, sum |.| {sum(abs(x) for x in v):.1e}")

# 6
with mp.workdps(50):
    e = 2 * mp.pi - mp.mpf(6.283185307179586) - mp.mpf(2.4492935982947064e-16)
print(f"6. 2 pi - (hi + lo) of dd.rs TWO_PI: {mp.nstr(e, 3)}")
fails += abs(e) > 1e-31
print("FAIL" if fails else "OK")
sys.exit(1 if fails else 0)
