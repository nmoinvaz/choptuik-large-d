"""How does the double-double solver's sensitivity max|dDelta/dF| grow with n = D - 2: exponentially (e^{n/2}, the
premise of a beyond-the-light-cone solver) or as a power of n?

Reads hp/data/delta_dd.csv and fits ln s = a n + p ln n + c (and the one-parameter-family special cases) by least
squares over windows of D. From D = 35 on the scan froze the centre data's tau-harmonics above 13, so the set of
active harmonics is fixed there; below, all 16 odd harmonics were active (README, "The double-double solver").
Also fits the growth rate of the accepted residual |F| (a tolerance schedule that follows the floor) and prints
1/x_m for the last node x_m = 0.7, the inward growth factor of the b-mode (b - 1 ~ x^{-n}) per unit of n.

Usage: python3 sens_fit.py [--mutate=exp]   (the mutation replaces s by a planted e^{n/2} n^2 law and must be
                                              classified as exponential; exits 1 if the classifier fails)
"""
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, '..', '..', 'hp', 'data', 'delta_dd.csv')


def load():
    rows = list(csv.DictReader(open(CSV)))
    D = np.array([float(r['D']) for r in rows])
    s = np.array([float(r['max_dDelta_dF']) for r in rows])
    F = np.array([float(r['F']) for r in rows])
    return D, s, F


def fits(n, y):
    out = {}
    A = np.vstack([n, np.log(n), np.ones_like(n)]).T
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    out['free'] = (c[0], c[1], float(np.sqrt(np.mean((A @ c - y) ** 2))))
    A = np.vstack([n, np.ones_like(n)]).T
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    out['exp'] = (c[0], 0.0, float(np.sqrt(np.mean((A @ c - y) ** 2))))
    A = np.vstack([np.log(n), np.ones_like(n)]).T
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    out['power'] = (0.0, c[0], float(np.sqrt(np.mean((A @ c - y) ** 2))))
    c = np.linalg.lstsq(A, y - 0.5 * n, rcond=None)[0]
    out['e^(n/2) n^p'] = (0.5, c[0], float(np.sqrt(np.mean((A @ c + 0.5 * n - y) ** 2))))
    return out


def main():
    mutate = any(a == '--mutate=exp' for a in sys.argv[1:])
    D, s, F = load()
    if mutate:
        rng = np.random.default_rng(1)
        s = np.exp(0.5 * (D - 2)) * (D - 2) ** 2 * np.exp(0.02 * rng.standard_normal(len(D)))
        print('MUTATION: s replaced by a planted e^{n/2} n^2 law (2% scatter)')
    ok = True
    print('window          N    free fit a n + p ln n (a, p, rms) | exp (a, rms) | power (p, rms) | e^(n/2) n^p (p, rms)')
    for lo, hi in [(40, 61.1), (45, 61.1), (36, 61.1), (15, 26.5)]:
        m = (D >= lo) & (D <= hi) & (s > 0)
        n = D[m] - 2
        f = fits(n, np.log(s[m]))
        fr, ex, pw, eh = f['free'], f['exp'], f['power'], f['e^(n/2) n^p']
        print(f'D {lo:4.0f}-{hi:4.1f}  {m.sum():4d}   a={fr[0]:6.3f} p={fr[1]:6.2f} rms={fr[2]:.3f} | a={ex[0]:.3f} rms={ex[2]:.3f}'
              f' | p={pw[1]:5.2f} rms={pw[2]:.3f} | p={eh[1]:6.2f} rms={eh[2]:.3f}')
        if lo == 40:
            verdict = 'power law' if pw[2] < 0.5 * eh[2] else ('exponential' if eh[2] < 0.5 * pw[2] else 'undecided')
            print(f'   frozen regime D = 40-61: {verdict} (power-law rms {pw[2]:.3f} vs e^(n/2) n^p rms {eh[2]:.3f})')
            if mutate and verdict != 'exponential':
                ok = False
            if not mutate and verdict != 'power law':
                ok = False
    # local growth per unit of D, binned
    print('\nlocal growth of s per unit of D (least squares in 3-unit bins), and the power law n^p it implies')
    for lo in range(36, 61, 3):
        m = (D >= lo) & (D < lo + 3) & (s > 0)
        if m.sum() < 5:
            continue
        g = np.polyfit(D[m], np.log(s[m]), 1)[0]
        nmid = lo + 1.5 - 2
        print(f'  D {lo}-{lo + 3}: x{math.exp(g):.3f} per unit  (n^p with p = {g * nmid:.1f})')
    m = (D >= 36) & (D <= 60.1) & (F > 0)
    g = np.polyfit(D[m], np.log(F[m]), 1)[0]
    print(f'\naccepted residual |F|, D = 36-60.1: x{math.exp(g):.3f} per unit of D'
          f' (README: floor rises about 1.4-fold); 1/x_m for x_m = 0.7: {1 / 0.7:.4f}')
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
