"""Outer region of the LP solution: integrate from the sonic point to large y; print rho*y^2 and u = y(w - 1).
usage: lp_far.py <scriptdir> <bits> <n> <y*> <branch> <ds>"""
import sys
sys.path.insert(0, sys.argv[1])
from flint import arb, ctx
from lp_core import *
ctx.prec = int(sys.argv[2]); n = int(sys.argv[3])
ys = arb(sys.argv[4]); br = sys.argv[5]; ds = arb(sys.argv[6])
rc, wc, _ = sonic_series(ys, n, br)
y = mid(ys + ds); r, w = mid(evalpoly(rc, ds)), mid(evalpoly(wc, ds))
for k in range(1, 9):
    Y = arb(10) ** k
    r, w, st = integrate(y, r, w, Y, n=n)
    y = Y
    print('y=1e%d' % k, 'rho*y^2 =', (r * Y * Y).mid().str(30, radius=False), ' u=y(w-1) =', (Y * (w - 1)).mid().str(30, radius=False), 'steps', st, flush=True)
