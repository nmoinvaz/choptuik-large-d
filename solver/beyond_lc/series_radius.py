"""Radii of convergence of the double-double solver's centre series (in x about 0) and horizon series (in u = 1 - x
about the light cone), from the saved scan states: how far the centre-regular solution and the light-cone-analytic
solution continue as analytic functions, i.e. the distance to their nearest (complex) singularity.

For each state the series coefficients c_k (max over the four fields and the tau points) are computed with the
installed hpkernel (Solver.series), and the radius is estimated by a least-squares fit of ln c_k = -k ln R + b over the
upper half of the computed orders (a root test). Usage: python3 series_radius.py [--mutate]
(--mutate replaces the coefficients by those of a planted 1/(1 - x/0.5) and must return R = 0.5; exits 1 otherwise)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HP = os.path.join(HERE, '..', '..', 'hp')
sys.path.insert(0, HP)
from hpkernel.driver import load, make_solver  # noqa: E402


def radius(c, frac=0.5):
    k = np.arange(len(c))
    m = (k >= int(len(c) * frac)) & (c > 0)
    A = np.vstack([k[m], np.ones(m.sum())]).T
    slope = np.linalg.lstsq(A, np.log(c[m]), rcond=None)[0][0]
    return float(np.exp(-slope))


def main():
    if '--mutate' in sys.argv:
        R = radius(np.array([0.5 ** -k * (1 + 0.1 * np.sin(k)) for k in range(200)]))
        print(f'planted R = 0.5 -> {R:.4f}', 'PASS' if abs(R - 0.5) < 0.01 else 'FAIL')
        return 0 if abs(R - 0.5) < 0.01 else 1
    states = ['up2_D17.0180', 'up7_D40.0201', 'up7_D45.0507', 'up8_D50.0075', 'up10_D55.0351', 'up13_D60.0020']
    out = []
    print(f"{'state':16s} {'D':>8s} {'kc':>4s} {'R centre':>9s} {'ks':>4s} {'R horizon':>9s}")
    for st in states:
        p = os.path.join(HP, 'data', 'states', st + '.npz')
        if not os.path.exists(p):
            continue
        zh, zl, meta = load(p)
        D, M = meta['D'], meta['M']
        kf = 13 if D >= 35 else None
        s = make_solver(D, M, nodes=list(meta['nodes']), kt=meta['kt'], kc=420, ks=240, threads=1, kfreeze=kf)
        rec = dict(state=st, D=D)
        for which, name in ((0, 'centre'), (1, 'horizon')):
            h, l = s.series(zh, zl, which)
            c = np.abs((np.asarray(h) + np.asarray(l)).reshape(-1, 4, M)).max(axis=(1, 2))
            rec['R_' + name] = radius(c)
            rec['k_' + name] = len(c) - 1
        out.append(rec)
        print(f"{st:16s} {D:8.4f} {rec['k_centre']:4d} {rec['R_centre']:9.3f} {rec['k_horizon']:4d} {rec['R_horizon']:9.3f}",
              flush=True)
    with open(os.path.join(HERE, 'series_radius.json'), 'w') as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
