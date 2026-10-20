"""Follow-up to zero_fits.py: the logarithmic models in more detail.
  ln0:   Delta = (1/ln D) poly_{deg-1}(1/ln D)       (limit 0, logarithmic approach)
  lnfree: Delta = poly_deg(1/ln D)                 (limit free)
  mixed: Delta = poly_16(1/D) + c / ln D           (an extra slowly decaying piece)
    python3 zero_fits2.py"""
import os
import sys
import mpmath as mp
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "zero_fits.py")).read().split("tD = lambda")[0].split('"""', 2)[2])
tl = lambda D: 1 / mp.log(D)
a2, b2 = tl(45), tl(10)
for deg in (14, 18):
    report("ln0: (1/lnD) poly(1/lnD)", lambda D, deg=deg: [tl(D) * v for v in cheb(tl(D), a2, b2, deg - 1)], deg)
for deg in (14, 16, 18, 20):
    ev = report("lnfree: poly(1/lnD)", lambda D, deg=deg: cheb(tl(D), a2, b2, deg), deg)
# limit of lnfree at t = 0 printed separately
for deg in (14, 16, 18, 20):
    basis = lambda D, deg=deg: cheb(tl(D), a2, b2, deg)
    X = mp.matrix([basis(d) for d in fitD]); y = mp.matrix([pts[d] for d in fitD]); c = mp.qr_solve(X, y)[0]
    t0 = cheb(mp.mpf(0), a2, b2, deg)
    print(f"   lnfree deg {deg}: value at 1/lnD = 0 (D = infinity): {mp.nstr(sum(v * c[j] for j, v in enumerate(t0)), 8)}", flush=True)
tD = lambda D: 1 / mp.mpf(D)
a1, b1 = tD(45), tD(10)
for deg in (14, 16):
    basis = lambda D, deg=deg: cheb(tD(D), a1, b1, deg) + [tl(D)]
    X = mp.matrix([basis(d) for d in fitD]); y = mp.matrix([pts[d] for d in fitD]); c = mp.qr_solve(X, y)[0]
    ev = lambda D: sum(v * c[j] for j, v in enumerate(basis(D)))
    h = max(abs(ev(d) - pts[d]) for d in hold)
    lim = sum(v * c[j] for j, v in enumerate(cheb(mp.mpf(0), a1, b1, deg)))
    print(f"mixed poly{deg}(1/D) + c/lnD: c = {mp.nstr(c[deg + 1], 3)}, held-out max {mp.nstr(h, 2)}, analytic part at D = inf {mp.nstr(lim, 9)}", flush=True)
