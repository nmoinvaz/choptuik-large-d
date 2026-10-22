#!/usr/bin/env python3
"""Planted-truth test of the Delta_inf error bar (README: Delta_inf = 1.64389375 +- 1.5e-8, the range of the 48 fits
of hp/dinf_budget.py on D <= 60.1 and 61.03).

Real data cannot grade the pipeline, because the answer is unknown. Here synthetic Delta(D) with a KNOWN limit is
generated on exactly the real D grid, with noise and junk points like the real ones, and run through exactly the
production path (dinf_budget.grid, the same cleaning, models, degrees and selection). The question is whether the
planted limit lands inside the quoted range.

Plants (frozen before the first run). Base: P(t), t = 1/D, the degree-18 fit in 1/D to the real cleaned points
(D = 10-61.03); its limit P(0) is known exactly. Each plant adds s h(t) with h(0) = 0, so the planted limit is P(0),
and |s h(0.1)| = 1e-3 (structure the polynomial does not contain, of a size comparable to the high orders of the real
function), except T6:
    T1  none                                   (noise and junk only; plumbing)
    T2  t^3/(1 + t/0.12)                       pole at t = -0.12 (D = -8.3)
    T3  Re t^3/(1 - t/ts), ts = 0.06 e^{2 pi i/3}   complex pole pair at |t| = 0.06 (|D| = 17), near the data
    T4  t^2 (sqrt(1 + t/0.1) - 1)              square-root branch point at t = -0.1
    T5  t^2 log(1 + t/0.05)                    log branch point at t = -0.05 (D = -20)
    T6  t^{5/2}, |s h(0.1)| = 1e-6             non-analytic at t = 0: outside the fitted model class (informative)
Noise: Gaussian, sigma 1e-16 for D <= 60.1 and 7e-16 above (the real estimate: about 1e-16, a few 1e-15 above 60.1);
junk: 20 points in D = 27-36 offset by +-(1-9)e-15, like the 27 points the real cleaning drops. Fixed seed.
Pass criteria (frozen): for T1-T5 the planted limit lies inside [min, max] of the quoted selection's limits, and
within the quoted half-width 1.5e-8 of that range's centre; the cleaning drops every junk point when it is on.
T6 is reported, not graded (the README states the bar assumes analyticity in 1/D).

    python3 planted_dinf.py [T1 T2 ...]     (about 4-6 minutes per plant; writes hp/data/planted_dinf.json)
"""
import json
import os
import random
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dd_fits_mp as F  # noqa: E402
import dinf_budget as B  # noqa: E402

mp.mp.dps = 60
BAR = mp.mpf("1.5e-8")
TS = mp.mpf("0.06") * mp.expj(2 * mp.pi / 3)
PLANTS = {
    "T1": (None, None),
    "T2": (lambda t: t ** 3 / (1 + t / mp.mpf("0.12")), "1e-3"),
    "T3": (lambda t: mp.re(t ** 3 / (1 - t / TS)), "1e-3"),
    "T4": (lambda t: t ** 2 * (mp.sqrt(1 + t / mp.mpf("0.1")) - 1), "1e-3"),
    "T5": (lambda t: t ** 2 * mp.log(1 + t / mp.mpf("0.05")), "1e-3"),
    "T6": (lambda t: t ** mp.mpf("2.5"), "1e-6"),
}


def make(allD, P, h, size, seed=1):
    rng = random.Random(seed)
    s = 0 if h is None else mp.mpf(size) / abs(h(mp.mpf("0.1")))
    pts = {}
    for d in allD:
        t = 1 / mp.mpf(d)
        sig = 1e-16 if d <= 60.1 else 7e-16
        pts[d] = P(mp.mpf(d)) + (s * h(t) if h else 0) + mp.mpf(rng.gauss(0, sig))
    cand = [d for d in allD if 27 <= d <= 36]
    junk = sorted(rng.sample(cand, 20))
    for d in junk:
        pts[d] += mp.mpf(rng.choice((-1, 1)) * rng.randint(1, 9)) * mp.mpf("1e-15")
    return pts, junk


def main():
    T0 = time.time()
    which = sys.argv[1:] or list(PLANTS)
    real, _ = F.load_points()
    allD = sorted(real)
    base = F.clean(real, [d for d in allD if 10 <= d <= 61.03], verbose=False)
    ev, res, _ = F.fit(real, base, "1/D", 18)
    L = F.limit(ev)
    print(f"base: degree-18 1/D fit to the real cleaned points, residual {mp.nstr(res, 3)}; planted limit "
          f"P(0) = {mp.nstr(L, 15)}  ({time.time() - T0:.0f} s)", flush=True)
    out_path = os.path.join(HERE, "data", "planted_dinf.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    graded = bad = 0
    for name in which:
        h, size = PLANTS[name]
        pts, junk = make(allD, ev, h, size)
        kept = F.clean(pts, [d for d in allD if 10 <= d <= 61.03], verbose=False)
        caught = all(d not in kept for d in junk)
        rows = B.grid(pts, allD, verbose=False, dmaxes=(60.1, 61.03))
        lo, hi, nfit = B.quoted(rows)
        c = (lo + hi) / 2
        floor = max(r[6] for r in rows if r[7] and r[0] > 50)
        inside = lo <= L <= hi
        inbar = abs(c - L) <= BAR
        ok = inside and inbar and caught
        tag = "info" if name == "T6" else ("pass" if ok else "FAIL")
        if name != "T6":
            graded += 1
            bad += not ok
        print(f"{name}: {nfit} fits, range {mp.nstr(lo - L, 3)} .. {mp.nstr(hi - L, 3)} about the planted limit; "
              f"centre off by {mp.nstr(c - L, 3)} (bar {mp.nstr(BAR, 2)}); inside range {inside}; within bar {inbar}; "
              f"junk caught {caught} ({len(junk)}); worst used residual {mp.nstr(floor, 2)} -> {tag}  "
              f"({time.time() - T0:.0f} s)", flush=True)
        results[name] = {"lo_minus_L": float(lo - L), "hi_minus_L": float(hi - L), "centre_minus_L": float(c - L),
                         "nfit": nfit, "inside": inside, "within_bar": bool(inbar), "junk_caught": caught,
                         "worst_used_residual": float(floor), "verdict": tag}
        json.dump(results, open(out_path, "w"), indent=1)
    print(f"{graded - bad} of {graded} graded plants passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
