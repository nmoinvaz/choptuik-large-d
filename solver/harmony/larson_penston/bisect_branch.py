"""Bisect y* on a sonic branch by the sign of (w - 1/3) where the backward solution breaks down (or at y_end);
print the density plateau (rho at the smallest y where |w - 1/3| < 1e-3) as the rho0 guess.
usage: bisect_branch.py <scriptdir> <branch> <ylo> <yhi> <iters>"""
import sys
sys.path.insert(0, sys.argv[1])
from flint import arb, ctx
from lp_core import *
import lp_core
ctx.prec = 160
br = sys.argv[2]; lo, hi, iters = arb(sys.argv[3]), arb(sys.argv[4]), int(sys.argv[5])
yend = arb('1e-7')

def back(ys, n=28, maxsteps=3000):
    rc, wc, _ = sonic_series(ys, n, br)
    rad = lp_core.radius_est([rc, wc], n)
    ds = arb(min(0.3 * rad, 0.3 * float(ys.mid())))
    y = mid(ys - ds); r = mid(evalpoly(rc, -ds)); w = mid(evalpoly(wc, -ds))
    plateau = None
    for s in range(maxsteps):
        if abs(float((w - arb(1) / 3).mid())) < 1e-3:
            plateau = (float(y.mid()), float(r.mid()))
        try:
            c1, c2 = step_series(y, r, w, n)
        except ValueError:
            break
        rad = lp_core.radius_est([c1, c2], n)
        h = rad * 10 ** (-30.0 / n) * 0.8
        if h < 1e-6 * float(y.mid()):
            break
        last = float((y - yend).mid()) <= h
        hh = (yend - y) if last else arb(-h)
        r, w = mid(evalpoly(c1, hh)), mid(evalpoly(c2, hh)); y = yend if last else mid(y + hh)
        if last:
            break
    return float((w - arb(1) / 3).mid()), float(y.mid()), plateau

slo = back(lo)[0] > 0
for i in range(iters):
    m = mid((lo + hi) / 2)
    s, yb, pl = back(m)
    if (s > 0) == slo:
        lo = m
    else:
        hi = m
    print(i, m.mid().str(20, radius=False), 'sign', '+' if s > 0 else '-', 'ybreak %.3g' % yb, 'plateau', pl, flush=True)
