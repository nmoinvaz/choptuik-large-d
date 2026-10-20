"""Can the double-double Delta(D) be consistent with Delta -> 0 (the Ecker-Ecker-Grumiller conjecture)?

Models, each fitted on the cleaned points 10 <= D <= 45 (least squares at 40 digits, Chebyshev basis) and tested on
the held-out points 45 < D <= 61.03:
  analytic:   Delta = poly_deg(1/D)                       (limit free; the README's form)
  power:      Delta = D^-p poly_deg(1/D),  p > 0          (limit 0)
  log:        Delta = poly_deg(1/ln D) with no constant   (limit 0)
  sqrt:       Delta = poly_deg(D^-1/2) with no constant   (limit 0)
Reports the fit residual (max |model - point| on the fit range) and the held-out max error.
    python3 zero_fits.py
"""
import os
import sys
import mpmath as mp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hp"))
import dd_fits_mp as F
mp.mp.dps = 40
pts, _ = F.load_points()
Ds = sorted(d for d in pts if 10 <= d)
Ds = F.clean(pts, Ds, verbose=False)
fitD = [d for d in Ds if d <= 45]
hold = [d for d in Ds if d > 45]
print(f"fit on {len(fitD)} points 10..45, held out {len(hold)} points 45..{hold[-1]:.3f}", flush=True)


def lsq(basis, deg, Dl):
    """basis(D) -> list of deg+1 column values; returns ev(D)"""
    X = mp.matrix([basis(d) for d in Dl])
    y = mp.matrix([pts[d] for d in Dl])
    c = mp.qr_solve(X, y)[0]
    ev = lambda D: sum(v * c[j] for j, v in enumerate(basis(D)))
    return ev


def cheb(t, a, b, deg):
    u = (2 * t - (a + b)) / (b - a)
    T = [mp.mpf(1), u]
    for _ in range(deg - 1):
        T.append(2 * u * T[-1] - T[-2])
    return T[: deg + 1]


def report(name, basis, deg):
    ev = lsq(basis, deg, fitD)
    r = max(abs(ev(d) - pts[d]) for d in fitD)
    h = max(abs(ev(d) - pts[d]) for d in hold)
    lim = ev(mp.mpf(10) ** 30)
    print(f"{name:34s} deg {deg:2d}: fit residual {mp.nstr(r, 2):>8s}   held-out max {mp.nstr(h, 2):>8s}   "
          f"value at D = 1e30: {mp.nstr(lim, 8)}", flush=True)
    return h


tD = lambda D: 1 / mp.mpf(D)
a1, b1 = tD(45), tD(10)
for deg in (14, 16, 18):
    report("analytic poly(1/D)", lambda D, deg=deg: cheb(tD(D), a1, b1, deg), deg)
for p in ("0.01", "0.03", "0.1", "0.3", "1"):
    pp = mp.mpf(p)
    for deg in (16, 20):
        report(f"D^-{p} poly(1/D)", lambda D, deg=deg, pp=pp: [v * mp.mpf(D) ** (-pp) for v in cheb(tD(D), a1, b1, deg)], deg)
tl = lambda D: 1 / mp.log(D)
a2, b2 = tl(45), tl(10)
for deg in (12, 16, 20):
    # zero constant: columns t * T_j(t)
    report("1/lnD * poly(1/lnD)  (-> 0)", lambda D, deg=deg: [tl(D) * v for v in cheb(tl(D), a2, b2, deg - 1)], deg)
ts = lambda D: mp.mpf(D) ** -0.5
a3, b3 = ts(45), ts(10)
for deg in (16, 20, 24):
    report("D^-1/2 * poly(D^-1/2)  (-> 0)", lambda D, deg=deg: [ts(D) * v for v in cheb(ts(D), a3, b3, deg - 1)], deg)
