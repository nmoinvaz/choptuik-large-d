"""Scan y* on a sonic branch: integrate backward from the sonic point toward y -> 0 and record y^3 (w - 1/3)
at y_end (regular-at-origin solutions are its zeros) or the y where the backward solution breaks down.
usage: scan_branch.py <scriptdir> <branch> <ylo> <yhi> <N> <yend>"""
import sys, math
sys.path.insert(0, sys.argv[1])
from flint import arb, ctx
from lp_core import *
import lp_core
ctx.prec = 80
br = sys.argv[2]; lo, hi, N = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]); yend = arb(sys.argv[6])

def back(ys, n=20, maxsteps=400):
    rc, wc, _ = sonic_series(arb(ys), n, br)
    rad = lp_core.radius_est([rc, wc], n)
    ds = arb(min(0.3 * rad, 0.3 * ys))
    y = mid(arb(ys) - ds); r = mid(evalpoly(rc, -ds)); w = mid(evalpoly(wc, -ds))
    for s in range(maxsteps):
        try:
            c1, c2 = step_series(y, r, w, n)
        except ValueError:
            return None, (float(y.mid()), float((w - arb(1) / 3).mid()), 0.0)
        rad = lp_core.radius_est([c1, c2], n)
        h = rad * 10 ** (-20.0 / n) * 0.8
        if h < 1e-6 * float(y.mid()):
            return None, (float(y.mid()), float((w - arb(1) / 3).mid()), float((1 - y * y * w * w).mid()))
        last = float((y - yend).mid()) <= h
        hh = (yend - y) if last else arb(-h)
        r, w = mid(evalpoly(c1, hh)), mid(evalpoly(c2, hh)); y = yend if last else mid(y + hh)
        if last:
            return float((yend ** 3 * (w - arb(1) / 3)).mid()), float(r.mid())
    return None, float(y.mid())

for i in range(N + 1):
    ys = lo + (hi - lo) * i / N
    v, extra = back(ys)
    if v is None:
        print('%.5f' % ys, 'breakdown at y=%.4g  w-1/3=%+.3e  B=%+.3e' % extra, flush=True)
    else:
        print('%.5f' % ys, 'reached y_end: y3(w-1/3)=%.4e rho=%.5g' % (v, extra), flush=True)
