"""Two-sided shooting for the LP (or Hunter) solution: unknowns (rho0, y*), match (rho, w) at y_m.
usage: lp_shoot.py <scriptdir> <bits> <n> <ym> <ya> <dsonic> <branch> <ys_guess> <rho0_guess> [mut]"""
import sys, time
sys.path.insert(0, sys.argv[1])
from flint import arb, ctx
from lp_core import *
bits, n = int(sys.argv[2]), int(sys.argv[3])
ctx.prec = bits
ym, ya, ds = arb(sys.argv[4]), arb(sys.argv[5]), arb(sys.argv[6])
branch = sys.argv[7]
x = [arb(sys.argv[9]), arb(sys.argv[8])]
mut = arb(sys.argv[10]) if len(sys.argv) > 10 else 1

def left(rho0):
    rc, wc = origin_series(rho0, n, mut)
    r, w = evalpoly(rc, ya), evalpoly(wc, ya)
    r, w, _ = integrate(ya, mid(r), mid(w), ym, n=n, mut=mut)
    return r, w

def right(ys):
    rc, wc, _ = sonic_series(ys, n, branch, mut)
    r, w = evalpoly(rc, -ds), evalpoly(wc, -ds)
    r, w, _ = integrate(mid(ys - ds), mid(r), mid(w), ym, n=n, mut=mut)
    return r, w

def F(v):
    rl, wl = left(v[0]); rr, wr = right(v[1])
    return [mid(rl - rr), mid(wl - wr)]

t0 = time.time()
eps = arb(2) ** (-bits // 2)
for it in range(12):
    f = F(x)
    nf = max(abs(float(f[0].mid())), abs(float(f[1].mid())))
    print('it', it, 'rho0', x[0].mid().str(12), 'y*', x[1].mid().str(12), '|F|', '%.3e' % nf, '%.1fs' % (time.time() - t0), flush=True)
    if nf < 2.0 ** (-bits + 12):
        break
    J = []
    for j in range(2):
        xp = list(x); xp[j] = x[j] + eps
        fp = F(xp)
        J.append([(fp[0] - f[0]) / eps, (fp[1] - f[1]) / eps])
    a11, a21 = J[0]; a12, a22 = J[1]
    det = a11 * a22 - a12 * a21
    d0 = (f[0] * a22 - a12 * f[1]) / det
    d1 = (a11 * f[1] - a21 * f[0]) / det
    x = [mid(x[0] - d0), mid(x[1] - d1)]
digits = int(bits * 0.30103) - 4
print('RESULT branch', branch, 'bits', bits, 'n', n, 'ym', sys.argv[4], 'ya', sys.argv[5], 'ds', sys.argv[6])
print('rho0 =', x[0].mid().str(digits, radius=False))
print('y*   =', x[1].mid().str(digits, radius=False))
