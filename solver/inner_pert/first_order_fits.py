"""The fitted first-order coefficients, for comparison with the inner-problem derivation (notes/FIRST_ORDER.md):
    Delta = Delta_inf + Delta_1/n + ...,  omega = 2 pi/Delta = omega_inf + omega_1/n + ...  (omega_1 = -omega_inf Delta_1/Delta_inf),
    lam = 1/gamma = 2 + lam_1/n + ...  (lam_inf = 2 fixed, and free).
The first coefficient is the same in 1/n and in 1/D (1/D = 1/n - 2/n^2 + ...), but the higher ones differ, so both
series are fitted and their spread is part of the uncertainty.

Delta: the double-double points (hp/data/delta_dd.csv, cleaned as hp/dd_fits_mp.py --clean), least squares in
mpmath (60 digits) with polynomials of degree 12-20 in t = 1/n or 1/D on D >= Dmin. lam: the double-double growth
rates (hp/data/gamma_dd.jsonl, hp/data/local/gamma_dd.jsonl), one point per 0.5 in D, |F| <= 1e-20, as
hp/gamma_compare.py, polynomials in 1/n of degree 4-8 with lam_inf = 2 fixed (and free) on D >= 16-30.

    python3 first_order_fits.py
"""
import json
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
HP = os.path.join(HERE, "..", "..", "hp")
sys.path.insert(0, HP)
from dd_fits_mp import load_points, clean  # noqa: E402

mp.mp.dps = 60

# Degrees of the Delta fits. Degree 20 was dropped after the planted-truth test (--planted): on synthetic data with a
# known Delta_1 every fit of degree 12-18 recovers it within 1.5e-5, while the degree-20 fits miss by 4e-5 to 5e-4
# (21 parameters overfit the 1e-14 noise; on the real data the degree-20 fit on D >= 16 is the outlier).
DEGS = (12, 14, 16, 18)


def polyfit(ts, ys, deg, fix0=None):
    """least squares in powers of t/t0 (t0 = max t); returns coefficients in powers of t"""
    t0 = max(ts)
    lo = 0 if fix0 is None else 1
    X = mp.matrix([[(t / t0) ** k for k in range(lo, deg + 1)] for t in ts])
    y = mp.matrix([v - (fix0 or 0) for v in ys])
    c = mp.qr_solve(X, y)[0]
    coef = ([fix0] if fix0 is not None else []) + [c[k - lo] / t0 ** k for k in range(lo, deg + 1)]
    res = max(abs(sum(coef[k] * t ** k for k in range(deg + 1)) - v) for t, v in zip(ts, ys))
    return coef, res


def delta_fits():
    pts, _ = load_points()
    Ds = clean(pts, sorted(pts), verbose=False)
    print(f"Delta: {len(Ds)} cleaned double-double points, D = {Ds[0]:.2f} .. {Ds[-1]:.2f}")
    rows = []
    for var, tf in (("1/n", lambda D: 1 / (mp.mpf(D) - 2)), ("1/D", lambda D: 1 / mp.mpf(D))):
        for Dmin in (10, 12, 16):
            for deg in DEGS:
                sel = [d for d in Ds if d >= Dmin]
                if len(sel) < deg + 8:
                    continue
                coef, res = polyfit([tf(d) for d in sel], [pts[d] for d in sel], deg)
                d0, d1 = coef[0], coef[1]
                w0 = 2 * mp.pi / d0
                w1 = -w0 * d1 / d0
                rows.append((var, Dmin, deg, d0, d1, w1, res))
                print(f"  {var} D >= {Dmin:2d} deg {deg:2d}: Delta_inf {mp.nstr(d0, 12)}  Delta_1 {mp.nstr(d1, 10)}  "
                      f"omega_1 {mp.nstr(w1, 10)}  omega_1/omega_inf {mp.nstr(w1 / w0, 10)}  max|res| {mp.nstr(res, 2)}")
    d1s = [r[4] for r in rows if r[6] <= 1e-13]
    w1s = [r[5] for r in rows if r[6] <= 1e-13]
    print(f"  fits with max|res| <= 1e-13: Delta_1 in [{mp.nstr(min(d1s), 8)}, {mp.nstr(max(d1s), 8)}], "
          f"omega_1 in [{mp.nstr(min(w1s), 8)}, {mp.nstr(max(w1s), 8)}]")
    return rows


def lam_fits():
    rows = {}
    for f in (os.path.join(HP, "data", "gamma_dd.jsonl"), os.path.join(HP, "data", "local", "gamma_dd.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("lam"):
                rows[round(r["D"], 6)] = r
    good = {}
    for d in sorted(rows):
        if rows[d]["F"] <= 1e-20 and all(abs(d - e) >= 0.5 for e in good):
            good[d] = rows[d]
    Ds = sorted(good)
    print(f"lambda: {len(Ds)} double-double points, D = {Ds[0]:.2f} .. {Ds[-1]:.2f}")
    out = []
    for Dmin in (16, 20, 24, 30):
        sel = [d for d in Ds if d >= Dmin]
        ts = [1 / (mp.mpf(d) - 2) for d in sel]
        ys = [mp.mpf(good[d]["lam"]) for d in sel]
        for deg in (4, 5, 6, 7, 8, 9, 10):
            if len(sel) < deg + 5:
                continue
            c, res = polyfit(ts, ys, deg, fix0=mp.mpf(2))
            cf, resf = polyfit(ts, ys, deg)
            out.append((Dmin, deg, c[1], cf[0], cf[1]))
            print(f"  D >= {Dmin:2d}, degree {deg}: lam_inf = 2 fixed: lam_1 {mp.nstr(c[1], 8)} (max|res| {mp.nstr(res, 2)});"
                  f"  free: lam_inf - 2 {mp.nstr(cf[0] - 2, 3)}, lam_1 {mp.nstr(cf[1], 8)} (max|res| {mp.nstr(resf, 2)})")
    return out


def planted():
    """Planted truth: the same fits on synthetic data with a known first coefficient, at the same D points.
    Delta: the README's rational formula for Delta(D) (denominator zeros at D = -3.2 and -11.0, like the data's
    singularity near D = -9 to -11), plus Gaussian noise 1e-14; its 1/n coefficient is a1 - a0 b1.
    lam: 1/gamma with the README's rational gamma(D), noise 1e-12 on lam; its 1/n coefficient is -4 (e1 - f1/2).
    Pass criteria, fixed before the first run: Delta_1 recovered within 2e-5 by every fit with max|res| <= 1e-13;
    lam_1 recovered within 2e-4 by the degree 9-10 fits on D >= 24, 30.
    History: the first run (degrees 14-20) FAILED on the degree-20 Delta fits (errors 4e-5 to 5e-4, all other fits
    within 1.5e-5); degree 20 was then removed from the estimator (DEGS), not the threshold changed
    (logs/first_order_fits_planted_first.log)."""
    import random
    random.seed(1)
    a = [mp.mpf(v) for v in ("1.6438975180", "40.378045077", "252.33285989", "206.73502321", "-357.09850575",
                             "313.47320996", "-418.91374574", "326.95252782", "-160.50467699")]
    b = [mp.mpf(1), mp.mpf("18.251453626"), mp.mpf("68.182050167")]
    e = [mp.mpf(v) for v in ("0.5", "18.083629635", "152.41588758", "-52.341588541", "9.9259594505", "-98.386450353",
                             "74.529506074", "-48.272358903")]
    f = [mp.mpf(1), mp.mpf("37.683686025"), mp.mpf("342.45505235")]
    rat = lambda num, den, eps: sum(c * eps ** k for k, c in enumerate(num)) / sum(c * eps ** k for k, c in enumerate(den))
    D1_true = a[1] - a[0] * b[1]
    L1_true = -4 * (e[1] - f[1] / 2)
    pts, _ = load_points()
    Ds = clean(pts, sorted(pts), verbose=False)
    syn = {d: rat(a, b, 1 / (mp.mpf(d) - 2)) + mp.mpf(random.gauss(0, 1e-14)) for d in Ds}
    ok = True
    print(f"planted Delta: Delta_1 = {mp.nstr(D1_true, 12)}")
    for var, tf in (("1/n", lambda D: 1 / (mp.mpf(D) - 2)), ("1/D", lambda D: 1 / mp.mpf(D))):
        for Dmin in (10, 12, 16):
            for deg in DEGS:
                sel = [d for d in Ds if d >= Dmin]
                coef, res = polyfit([tf(d) for d in sel], [syn[d] for d in sel], deg)
                err = coef[1] - D1_true
                flag = ""
                if res <= 1e-13:
                    flag = "ok" if abs(err) <= 2e-5 else "MISS"
                    ok &= abs(err) <= 2e-5
                print(f"  {var} D >= {Dmin:2d} deg {deg:2d}: Delta_1 - planted {mp.nstr(err, 3):>10s}  max|res| {mp.nstr(res, 2)}  {flag}")
    rows = {}
    for fn in (os.path.join(HP, "data", "gamma_dd.jsonl"), os.path.join(HP, "data", "local", "gamma_dd.jsonl")):
        for l in open(fn):
            r = json.loads(l)
            if r.get("lam"):
                rows[round(r["D"], 6)] = r
    good = []
    for d in sorted(rows):
        if rows[d]["F"] <= 1e-20 and all(abs(d - x) >= 0.5 for x in good):
            good.append(d)
    print(f"planted lam: lam_1 = {mp.nstr(L1_true, 12)}")
    for Dmin in (24, 30):
        sel = [d for d in good if d >= Dmin]
        ts = [1 / (mp.mpf(d) - 2) for d in sel]
        ys = [1 / rat(e, f, t) + mp.mpf(random.gauss(0, 1e-12)) for t in ts]
        for deg in (8, 9, 10):
            c, res = polyfit(ts, ys, deg, fix0=mp.mpf(2))
            err = c[1] - L1_true
            flag = ("ok" if abs(err) <= 2e-4 else "MISS") if deg >= 9 else ""
            ok &= abs(err) <= 2e-4 or deg < 9
            print(f"  D >= {Dmin}, degree {deg}: lam_1 - planted {mp.nstr(err, 3):>10s}  max|res| {mp.nstr(res, 2)}  {flag}")
    print("PLANTED: PASS" if ok else "PLANTED: FAIL")
    return ok


if __name__ == "__main__":
    if "--planted" in sys.argv[1:]:
        sys.exit(0 if planted() else 1)
    if "lam" not in sys.argv[1:]:
        delta_fits()
    lam_fits()
