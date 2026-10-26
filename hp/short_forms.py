#!/usr/bin/env python3
"""The shortest 1/D series for Delta on a range of D, with its error.

    cd hp && python3 short_forms.py [--selftest]

For each range (D0, D1) and number of terms m: Delta = c_0 + c_1 / D + ... + c_{m-1} / D^{m-1}, the coefficients
by least squares (mpmath, Chebyshev basis in 1/D, dd_fits_mp.fit) on the double-double continuation points
(hp/data/delta_dd.csv, 10 <= D <= 61.03 as the README's fits, cleaned as dd_fits_mp.clean) with D in the range, or, for a range above the last
point (D > 61 is not computed; the T48 points above 61.03 are not used in the fits, README "The large-D limit"), on
the points with D >= D0 / 2. The error quoted is
  - inside the computed range: the largest |fit - point| over the range's points (an interpolation error);
  - above it: the spread of the series' values over the range between fits that differ in the points' lower cut
    (D >= D0/2, D0/3, 20) and in one extra term, plus the fit's own largest residual: a fit spread, not a bound,
    which assumes Delta analytic in 1/D (README "Is the period analytic in 1/D?").
The table lists, per range, the fewest terms whose error is below 1e-4, 1e-6 and 1e-8, with each form's
coefficients (rounded two digits past the error).
--selftest: the same procedure on a planted analytic function 1.6 + 10/D - 30/D^2 + 200/D^3 sampled on the points'
D with 1e-15 noise must recover its values on 70-200 within the quoted spread (controls: with 0.01 / sqrt(D) added,
not analytic in 1/D, the spread on 100-200 at the table's 8-term form must widen at least 100-fold, which is how a
non-analytic term would show; and a value off by twice the spread must lie outside it).
"""
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dd_fits_mp import clean, fit, load_points  # noqa: E402

mp.mp.dps = 60
RANGES = [(10, 20), (20, 40), (40, 61), (70, 100), (100, 200)]
LEVELS = (1e-4, 1e-6, 1e-8)


def series_coeffs(ev, m):
    """c_0..c_{m-1} of the fitted polynomial in 1/D (exactly: a degree m-1 polynomial in x = 1/D sampled at m points)"""
    xs = [mp.mpf(1) / (100 * (k + 1)) for k in range(m)]
    V = mp.matrix([[x ** j for j in range(m)] for x in xs])
    y = mp.matrix([ev(1 / x) for x in xs])
    return list(mp.lu_solve(V, y))


def error_in_range(pts, Ds, ev, D0, D1):
    return max(abs(ev(d) - pts[d]) for d in Ds if D0 <= d <= D1)


def spread_above(pts, Ds, D0, D1, m):
    """fit spread on [D0, D1] over lower cuts and one extra term, plus the largest fit residual"""
    grid = [mp.mpf(D0) + (D1 - D0) * k / 40 for k in range(41)]
    vals, resmax = [], mp.mpf(0)
    for cut in (D0 / 2, D0 / 3, 20):
        Dl = [d for d in Ds if d >= cut]
        for mm in (m, m + 1):
            ev, res, _ = fit(pts, Dl, "1/D", mm - 1)
            vals.append([ev(g) for g in grid])
            resmax = max(resmax, res)
    sp = max(max(v[i] for v in vals) - min(v[i] for v in vals) for i in range(len(grid)))
    return sp + resmax


def table(pts, Ds, out=print):
    rows = []
    for D0, D1 in RANGES:
        inside = D1 <= max(Ds)
        chosen = {}
        for m in range(1, 13):
            if inside:
                Dl = [d for d in Ds if D0 <= d <= D1]
                ev, res, _ = fit(pts, Dl, "1/D", m - 1)
                err = error_in_range(pts, Ds, ev, D0, D1)
            else:
                Dl = [d for d in Ds if d >= D0 / 2]
                ev, res, _ = fit(pts, Dl, "1/D", m - 1)
                err = spread_above(pts, Ds, D0, D1, m)
            for lv in LEVELS:
                if lv not in chosen and err < lv:
                    chosen[lv] = (m, err, series_coeffs(ev, m))
            if len(chosen) == len(LEVELS):
                break
        rows.append((D0, D1, inside, chosen))
        out(f"D {D0}-{D1} ({'interpolation error' if inside else 'fit spread'}): " + "; ".join(
            f"{lv:.0e}: {chosen[lv][0]} terms (err {float(chosen[lv][1]):.1e})" if lv in chosen else f"{lv:.0e}: not reached with 12 terms"
            for lv in LEVELS))
        for lv in LEVELS:
            if lv in chosen:
                m, err, c = chosen[lv]
                digits = int(-mp.log10(err)) + 2
                terms = [f"{float(c[0]):.{digits}f}"] + [f"{'+' if float(c[j]) >= 0 else '-'} {abs(float(c[j])):.{digits}g}/D^{j}" for j in range(1, m)]
                out(f"   {lv:.0e} form ({m} terms, error {float(err):.1e}): Delta = " + " ".join(terms))
    return rows


def selftest():
    pts0, _ = load_points()
    Ds = sorted(d for d in pts0 if d >= 10)
    f = lambda D: mp.mpf("1.6") + 10 / mp.mpf(D) - 30 / mp.mpf(D) ** 2 + 200 / mp.mpf(D) ** 3  # noqa: E731
    noise = lambda i: mp.mpf((-1) ** i) * mp.mpf("1e-15") * ((i * 7919) % 101) / 101  # noqa: E731
    pts = {d: f(d) + noise(i) for i, d in enumerate(Ds)}
    ok = True
    for D0, D1 in ((70, 100), (100, 200)):
        Dl = [d for d in Ds if d >= D0 / 2]
        for m in (3, 4, 5):
            ev, _, _ = fit(pts, Dl, "1/D", m - 1)
            sp = spread_above(pts, Ds, D0, D1, m)
            miss = max(abs(ev(mp.mpf(D0) + (D1 - D0) * k / 40) - f(mp.mpf(D0) + (D1 - D0) * k / 40)) for k in range(41))
            good = miss <= sp
            ok &= good
            print(f"planted, D {D0}-{D1}, {m} terms: miss {float(miss):.1e} within spread {float(sp):.1e}: {'ok' if good else 'WRONG'}")
    # controls: (1) with a D^-1/2 term added, which the 1/D fits cannot represent, the spread must widen by at least
    # 100x against the analytic case at the table's 8-term form (the spread reacts to non-analyticity; it still covered
    # the miss here, 2.7e-6 against 1.1e-5, so it is a detector, not a bound); (2) a value nudged by twice the spread
    # must lie outside it.
    g = lambda D: f(D) + mp.mpf("0.01") / mp.sqrt(mp.mpf(D))  # noqa: E731
    ptg = {d: g(d) + noise(i) for i, d in enumerate(Ds)}
    sp_a = spread_above(pts, Ds, 100, 200, 8)
    sp_g = spread_above(ptg, Ds, 100, 200, 8)
    ev, _, _ = fit(ptg, [d for d in Ds if d >= 50], "1/D", 7)
    miss = max(abs(ev(mp.mpf(100 + 100 * k / 40)) - g(mp.mpf(100 + 100 * k / 40))) for k in range(41))
    c1 = sp_g > 100 * sp_a
    print(f"control 1 (plus 0.01/sqrt(D)), D 100-200, 8 terms: spread {float(sp_g):.1e} against the analytic case's "
          f"{float(sp_a):.1e} (miss {float(miss):.1e}): {'widens, as it must' if c1 else 'WRONG: no reaction'}")
    c2 = not (2 * sp_a <= sp_a)
    print(f"control 2: a value off by twice the spread is outside it: {'ok' if c2 else 'WRONG'}")
    ok &= c1 and c2
    print("selftest", "ok" if ok else "FAILED")
    return ok


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    pts, _ = load_points()
    # the fits' points: 10 <= D <= 61.03, as the README's (the T48 points above 61.03 are not used in fits)
    Ds = sorted(d for d in pts if 10 <= d <= 61.03)
    Ds = clean(pts, Ds, verbose=False)
    print(f"{len(Ds)} points, D {min(Ds)} .. {max(Ds)}")
    table(pts, Ds)


if __name__ == "__main__":
    main()
