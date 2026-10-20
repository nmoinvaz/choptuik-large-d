#!/usr/bin/env python3
"""Test candidate constants for Delta_inf on the double-double points (assumes Delta is analytic in 1/D).

    python3 dd_candidates.py [Dmax_fit] [--dmax=D]        (default 45; --dmax ignores the points above D)

For each candidate c0, Delta(D) = c0 + x * p(x), x = 1/D, p a polynomial of degree deg - 1 in a Chebyshev basis on the
fitted x-range, is fitted at 60 digits on the cleaned points 10 <= D <= Dmax_fit and compared with every point above
(held out). The free fit (c0 a fitted constant) is the reference. Reported per candidate: its distance from the free
limit, the held-out error of the forced fit at each degree (14-26), and its best against the free fit's best.
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_fits_mp import cheb_rows, clean, load_points  # noqa: E402

mp.mp.dps = 60
j01, j11, j21 = (mp.besseljzero(v, 1) for v in (0, 1, 2))
CANDIDATES = [
    ("5/7 sqrt(G) j01", mp.mpf(5) / 7 * mp.sqrt(mp.catalan) * j01),
    ("7/5 (ln 2)^-2 pi^-1/2", mp.mpf(7) / 5 / mp.log(2) ** 2 / mp.sqrt(mp.pi)),
    ("7/2 sqrt(ln 2 / pi)", mp.mpf(7) / 2 * mp.sqrt(mp.log(2) / mp.pi)),
    ("9/8 sqrt(j21 / j01)", mp.mpf(9) / 8 * mp.sqrt(j21 / j01)),
    ("10/9 sqrt(j01 / ln 3)", mp.mpf(10) / 9 * mp.sqrt(j01 / mp.log(3))),
    ("pi^2/6", mp.pi ** 2 / 6),
    ("2 pi / j11", 2 * mp.pi / j11),
    ("sqrt(e)", mp.sqrt(mp.e)),
    ("golden ratio", (1 + mp.sqrt(5)) / 2),
]
DEGS = (14, 16, 18, 20, 22, 24, 26)


def fit_forced(pts, fitD, deg, c0=None):
    """c0 + x * sum_k a_k T_k(u(x)), k < deg; c0 fitted when None. Returns (ev, max residual, c0)."""
    xs = [1 / mp.mpf(d) for d in fitD]
    a, b = min(xs), max(xs)
    T = cheb_rows(xs, a, b, deg - 1)
    rows = [[xs[i] * T[i, j] / b for j in range(deg)] + ([mp.mpf(1)] if c0 is None else []) for i in range(len(xs))]
    X = mp.matrix(rows)
    y = mp.matrix([pts[d] - (0 if c0 is None else c0) for d in fitD])
    c = mp.qr_solve(X, y)[0]
    lim = c[deg] if c0 is None else c0

    def ev(D):
        x = 1 / mp.mpf(D)
        t = cheb_rows([x], a, b, deg - 1)
        return lim + sum(x * t[0, j] / b * c[j] for j in range(deg))

    res = max(abs((X * c)[i] - y[i]) for i in range(len(fitD)))
    return ev, res, lim


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    Dfit = float(args[0]) if args else 45.0
    dmax = next((float(a[7:]) for a in sys.argv[1:] if a.startswith("--dmax=")), float("inf"))
    pts, _ = load_points()
    Ds = clean(pts, sorted(d for d in pts if 10 <= d <= dmax), verbose=False)
    fitD = [d for d in Ds if d <= Dfit]
    hold = [d for d in Ds if d > Dfit]
    print(f"fit on {len(fitD)} cleaned points 10 <= D <= {Dfit:g}, {len(hold)} held out up to D = {hold[-1]:.2f}")

    def held(ev):
        return max(abs(ev(d) - pts[d]) for d in hold)

    free = {}
    for deg in DEGS:
        ev, res, lim = fit_forced(pts, fitD, deg)
        free[deg] = (held(ev), res, lim)
        print(f"free  deg {deg}: residual {mp.nstr(res, 2)}  held-out max {mp.nstr(free[deg][0], 2)}  "
              f"limit {mp.nstr(lim, 11)}", flush=True)
    best_free = min(free, key=lambda k: free[k][0])
    fbest, flim = free[best_free][0], free[best_free][2]
    print(f"\nfree fit: best held-out {mp.nstr(fbest, 2)} at degree {best_free}, limit {mp.nstr(flim, 11)}")
    print(f"\n{'candidate':24s} {'value':>11s} {'- free lim':>10s}  held-out max at degree " +
          " ".join(f"{d:>7d}" for d in DEGS) + f"  {'best':>7s} {'x free':>6s}")
    for name, v in CANDIDATES:
        hs = []
        for deg in DEGS:
            ev, res, _ = fit_forced(pts, fitD, deg, v)
            hs.append(held(ev))
        h = min(hs)
        print(f"{name:24s} {mp.nstr(v, 9):>11s} {mp.nstr(v - flim, 2):>10s}  {'':22s}" +
              " ".join(f"{mp.nstr(x, 2):>7s}" for x in hs) + f"  {mp.nstr(h, 2):>7s} {mp.nstr(h / fbest, 3):>6s}",
              flush=True)


if __name__ == "__main__":
    main()
