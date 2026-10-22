#!/usr/bin/env python3
"""Review RV2: a planted-truth test of the Delta_inf bar whose truth is NOT produced by the fitting code.

hp/planted_dinf.py builds its truth from the degree-18 1/D fit to the real points (a fixture made by the code under
test, and inside the fitted model class) and adds iid Gaussian noise. Here:

  truth  R(D) = the README's rational formula P8/Q2 in eps = 1/(D-2) (fitted to the f64 points 4 <= D <= 30.2 by
         solver/formula_all.py, so independent of the double-double data and of the 1/D fits); its limit is
         a0 = 1.6438975180 exactly; poles at D = -3.2 and -11.0, close to the singularity the real data suggest.
         It matches the real double-double points to <= 6e-8, so it has the real function's large-scale shape.
  noise  N1: the real data's own residual pattern: Delta_i - P18(D_i), P18 the degree-18 1/D fit to the cleaned
             real points (so the junk points, the kfreeze-cut bias at 35.35-37 and the noisier points above 60.1
             are transplanted at their own D);
         N2: iid Gaussian with the leave-one-out rms per band (hp/review/rv2_noise.py), no junk;
         N3: N1 plus a smooth systematic error: +3e-15 on every point above D = 60.1 (the size of the largest
             recorded corrections there).
Each plant goes through the same 48-fit selection (rv2_fits.grid: dmax 60.1 and 61.03, dmin 10, 12, 15, cleaning
on / off, 1/D and 1/(D-2), the two lowest degrees within 2.5x of the group's best max residual).

    python3 hp/review/rv2_planted.py
Pass: the planted limit inside the 48-fit range for N1-N3.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import arb  # noqa: E402

import rv2_fits as RF  # noqa: E402

A = ["1.6438975180", "4.0378045077e1", "2.5233285989e2", "2.0673502321e2", "-3.5709850575e2", "3.1347320996e2",
     "-4.1891374574e2", "3.2695252782e2", "-1.6050467699e2"]
Bc = ["1", "1.8251453626e1", "6.8182050167e1"]
A = [arb(a) for a in A]
Bc = [arb(b) for b in Bc]
LIMIT = A[0]


def R(d):
    e = 1 / (arb(RF.fmpq(*float(d).as_integer_ratio())) - 2)
    num = sum((A[k] * e ** k for k in range(len(A))), arb(0))
    den = sum((Bc[k] * e ** k for k in range(len(Bc))), arb(0))
    return num / den


def main():
    pts, run = RF.load()
    allD = sorted(pts)
    keep, _ = RF.clean(pts, [d for d in allD if 10 <= d <= 61.03])
    f = RF.Fit(pts, keep, "1/D", 18)
    _, _, c = f.solve(18)
    lo_, hi_ = f.a, f.b

    def P18(d):
        u = (2 * RF.tvar(d, "1/D") - (lo_ + hi_)) / (hi_ - lo_)
        return sum((c[k, 0] * u ** k for k in range(19)), arb(0))

    resid = {d: pts[d] - P18(d) for d in allD}
    shape = max(abs(float((R(d) - pts[d]).mid())) for d in allD if d >= 10)
    print(f"truth R(D) = P8/Q2, limit {RF.fmt(LIMIT)}; max |R - real Delta| on 10..61.03: {shape:.1e}")
    loo = {(10, 27): 1e-16, (27, 35.25): 1.2e-15, (35.25, 45): 1e-15, (45, 60.1): 1e-17, (60.1, 62): 4e-16}
    rng = random.Random(7)

    def iid(d):
        for (a, b), s in loo.items():
            if a <= d < b:
                return arb(rng.gauss(0, s))
        return arb(0)

    plants = {
        "N1 real residual pattern": {d: R(d) + resid[d] for d in allD},
        "N2 iid per-band noise": {d: R(d) + iid(d) for d in allD},
        "N3 N1 + 3e-15 above 60.1": {d: R(d) + resid[d] + (arb("3e-15") if d > 60.1 else 0) for d in allD},
    }
    bad = 0
    for name, p in plants.items():
        rows = RF.grid(p, dmaxes=(60.1, 61.03))
        lo, hi, n = RF.span(rows)
        cen = (lo + hi) / 2
        inside = lo <= LIMIT.mid() <= hi
        bad += not inside
        print(f"{name:28s} {n} fits: range {float((lo - LIMIT).mid()):+.2e} .. {float((hi - LIMIT).mid()):+.2e} about "
              f"the planted limit; centre off by {float((cen - LIMIT).mid()):+.2e}; half-width {float((hi - lo) / 2):.1e}; "
              f"{'inside' if inside else 'OUTSIDE'}", flush=True)
        for r in rows:
            if r["use"] and r["clean"]:
                print(f"     D {r['dmin']:>2}..{r['dmax']:<5} {r['model']:3s} deg {r['deg']}: "
                      f"{float((r['lim'] - LIMIT).mid()):+.2e} (res {r['res']:.1e})")
    print("PASS" if not bad else f"FAIL ({bad} of {len(plants)} plants outside the range)")


if __name__ == "__main__":
    main()
