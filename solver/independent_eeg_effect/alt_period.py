"""Alternative reading of effect.py's measure: rms over tau in [0, Delta(D)) instead of one period of beta~.

Delta(D) = 1.6439 + 10.37/D is the paper's fit of the echoing period. Eq. (16) has period 4 pi mu0.

    python3 alt_period.py     # prints rms(change)/rms(corrected NNLO) for f, Omega, Pi, in %, both spans
"""
import numpy as np

import effect as E

fn = E.build()
xg, wg = np.polynomial.legendre.leggauss(24)
xq, wq = (xg + 1) / 2, wg / 2
D0 = 4 * np.pi * E.MU0
for D in (25, 50, 200):
    eps = 1 / (D - 1)
    for lab, span in (('one period', D0), ('Delta(D) fit', 1.6439 + 10.37 / D)):
        th = 2 * np.pi * (span / D0) * np.arange(4096) / 4096
        base, pr, lo, nlo, flip, chg, _ = E.fields_on(eps, fn, th, xq, '')
        rms = lambda g: np.sqrt(np.mean(g**2 @ wq))
        print(D, f'{lab:14s}', ' '.join(f'{k} {100 * rms(chg[k]) / rms(base[k]):.4g}' for k in ('f', 'Om', 'Pi')))
