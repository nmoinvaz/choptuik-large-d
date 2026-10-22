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
            for deg in (12, 14, 16, 18, 20):
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
    d1s = [r[4] for r in rows if r[6] < 1e-12]
    w1s = [r[5] for r in rows if r[6] < 1e-12]
    print(f"  fits with max|res| < 1e-12: Delta_1 in [{mp.nstr(min(d1s), 8)}, {mp.nstr(max(d1s), 8)}], "
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


if __name__ == "__main__":
    if "lam" not in sys.argv[1:]:
        delta_fits()
    lam_fits()
