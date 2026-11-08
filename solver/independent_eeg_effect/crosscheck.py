"""Second route to the rms ratios of effect.py, with exact x-integrals and mpmath quadrature in theta.

Shares with effect.py only fields.py (the transcription) and the parameters. The x-mean of each square
is integrated exactly by sympy, the beta~ derivatives come from sympy differentiation of the closed form
of Eq. (16) (not its Fourier series), and the theta-mean is mpmath.quad (tanh-sinh), not a uniform grid.

    python3 crosscheck.py 25 50 200   # prints rms(change)/rms(corrected NNLO) for f, Omega, Pi, in %
"""
import sys

import mpmath as mp
import sympy as sp

from fields import b, x, printed, corrections
from effect import Q, DEL, A, MU0

mp.mp.dps = 20
th = sp.Symbol('th', real=True)
e = sp.Symbol('e', positive=True)
bt = A * sp.re(sp.exp(sp.I * th) / (1 + Q * sp.exp(sp.I * (2 * th + DEL))))
bt = sp.expand_complex(bt)
bk = [sp.lambdify(th, sp.diff(bt, th, k) / sp.Float(2 * MU0, 20)**k, 'mpmath') for k in range(5)]

P, C = printed(), corrections()
Pi3c = P['Pi3'] + C['dPi3']
f3c = P['f3'] + C['df3']
Om3c = 2 * b[0] * Pi3c + P['Om3_rest']
h = sp.Rational(1, 2)
ex = dict(
    f=(-(e * P['f1'] + e**2 * P['f2'] + e**3 * f3c), -e**3 * C['df3']),
    Om=(e * P['Om1'] + e**2 * P['Om2'] + e**3 * Om3c, -2 * e**3 * b[0] * C['dPi3']),
    # Pi over sqrt(eps), the common factor cancels in the ratio.
    Pi=(b[0] + e * P['Pi2'] + e**2 * Pi3c, -e**2 * C['dPi3']))
args = tuple(b[:5]) + (e,)
xmean = {k: tuple(sp.lambdify(args, sp.integrate(sp.expand(v**2), (x, 0, 1)), 'mpmath') for v in pair)
         for k, pair in ex.items()}


def tmean(g, eps):
    bb = lambda t: [f(t) for f in bk]
    tq = [mp.pi * k / 4 for k in range(9)]
    return mp.quad(lambda t: g(*bb(t), eps), tq) / (2 * mp.pi)


for D in [int(a) for a in sys.argv[1:]] or [25, 50, 200]:
    eps = mp.mpf(1) / (D - 1)
    out = []
    for k in ('f', 'Om', 'Pi'):
        gb, gc = xmean[k]
        out.append(f'{k} {mp.nstr(100 * mp.sqrt(tmean(gc, eps) / tmean(gb, eps)), 8)}')
    print(f'D {D}: ' + '  '.join(out), flush=True)
