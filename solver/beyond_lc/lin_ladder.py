"""The harmonic ladder in the linear (c = b = 1) free-wave problem, in the two geometries.

For harmonic k (s = i k omega) of Proposition 2's exact solution:
  standard:  sine of the angle between the centre-regular and the light-cone-analytic solution vectors (p, m) at the
             matching point x_m = 0.7 (lin_angle.angle), i.e. how strongly the matching sees harmonic k's obstruction;
  candidate (c): the jump |w(z0 + i0) - w(z0 - i0)| / |w(z0 + i0)| of the centre-regular solution w = 2F1(a, b; kappa; z)
             across the Stokes line at z0 = x0^2 = n (xi0 = 1), the quantity a complex-path solver would drive to zero.
Prints both against n and the local power law between n = 39 and 59, to compare with the measured weak directions of
hp/ (README: n^-10.5, n^-14, n^-17.5, n^-21 for k = 7, 9, 11, 13) and max|dDelta/dF| ~ n^21.7 (sens_fit.py).

Usage: python3 lin_ladder.py
"""
import math

import mpmath as mp

from lin_angle import angle, W

mp.mp.dps = 50


def jump(n, k, xi0=1.0):
    s = 1j * k * W
    a = (1 + s) / 2
    b = (2 + s) / 2
    kap = mp.mpf(n + 1) / 2
    z0 = mp.mpf(xi0) ** 2 * n
    d = mp.mpf(10) ** -40
    wp = mp.hyp2f1(a, b, kap, mp.mpc(z0, d))
    wm = mp.hyp2f1(a, b, kap, mp.mpc(z0, -d))
    return abs(wp - wm) / abs(wp)


if __name__ == '__main__':
    ns = [19, 29, 39, 49, 59, 99, 199]
    print('standard: angle at x_m = 0.7          columns n =', ns, ' | local power n=39..59')
    for k in [1, 3, 5, 7, 9, 11, 13]:
        v = [angle(n, k, mp.mpf('0.7')) for n in ns]
        p = math.log(float(v[2] / v[4])) / math.log(59 / 39)
        print(f'  k={k:2d}', ' '.join(f'{float(x):.1e}' for x in v), f' | n^{-p:+.1f}', flush=True)
    print('candidate (c): relative jump at z0 = n columns n =', ns, ' | local power n=39..59')
    for k in [1, 3, 5, 7, 9, 11, 13]:
        v = [jump(n, k) for n in ns]
        p = math.log(float(v[2] / v[4])) / math.log(59 / 39)
        print(f'  k={k:2d}', ' '.join(f'{float(x):.1e}' for x in v), f' | n^{-p:+.1f}', flush=True)
