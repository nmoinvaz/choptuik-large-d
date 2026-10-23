"""Tabulate toy_results.jsonl (toy.py) and evaluate the pre-registered pass criterion of notes/BEYOND_LIGHT_CONE.md
section 0: fit ln kappa = a n + p ln n + c over n = 20, 60, 100, 200 for each method; 'polynomial' if a < 0.05, 'like
e^{n/2}' if a > 0.3; the candidate C passes if a_C < 0.05 and a_S(0.7) > 0.3 (and its controls fail, toy.py --plant).

Also prints hp_node_test.jsonl (the full-equation test) against the fast-mode prediction exp(n [Phi(x_m) - Phi(0.7)]).

Usage: python3 toy_table.py [--selftest]   (--selftest feeds a planted e^{0.5 n} n^2 kappa and a planted n^3 kappa
                                             through the classifier and exits 1 unless they come out 'e^{n/2}' and
                                             'polynomial')
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OMEGA = 2 * math.pi / 1.6438937


def phi(x):
    return math.log(x) - 0.5 * math.log(1 - x * x)


def fit(ns, ks):
    n = np.array(ns, float)
    y = np.log(np.array(ks, float))
    A = np.vstack([n, np.log(n), np.ones_like(n)]).T
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    return c[0], c[1]


def classify(a):
    return 'polynomial' if a < 0.05 else ('like e^(n/2)' if a > 0.3 else 'in between')


def selftest():
    ns = [20, 60, 100, 200]
    a1, _ = fit(ns, [math.exp(0.5 * n) * n ** 2 for n in ns])
    a2, _ = fit(ns, [n ** 3 for n in ns])
    ok = classify(a1) == 'like e^(n/2)' and classify(a2) == 'polynomial'
    print(f'selftest: planted e^(n/2) n^2 -> a = {a1:.3f} ({classify(a1)}); planted n^3 -> a = {a2:.3f} ({classify(a2)})')
    print('SELFTEST PASS' if ok else 'SELFTEST FAIL')
    return 0 if ok else 1


def main():
    if '--selftest' in sys.argv:
        return selftest()
    recs = [json.loads(l) for l in open(os.path.join(HERE, 'toy_results.jsonl'))]
    sst = (0.0, OMEGA)
    by = {}
    for r in recs:
        if abs(r['sstar'][0] - sst[0]) > 1e-9 or abs(r['sstar'][1] - sst[1]) > 1e-6:
            continue
        by.setdefault(r['method'], {})[r['n']] = r     # later lines (reruns) replace earlier ones
    order = ['S:0.5', 'S:0.6', 'S:0.7', 'S:0.7071067811865476', 'S:0.8', 'S:0.9', 'C:1', 'A:1.5']
    print('kappa_J: error in s per unit relative perturbation of the matched states; kappa_emp: with relative noise '
          'injected after every step (max of 3 draws); P_min: digits for |s - s*| < 1e-8')
    print(f"{'method':10s} {'n':>4s} {'steps':>6s} {'kappa_J':>9s} {'kappa_emp':>9s} {'P_min':>5s} {'prediction':>11s}")
    res = {}
    for m in order:
        if m not in by:
            continue
        rows = by[m]
        ns = sorted(rows)
        for n in ns:
            r = rows[n]
            pred = ''
            if m.startswith('S:'):
                pred = f"e^{n * phi(float(m[2:])):+.1f}"
            elif m.startswith('A:'):
                zs = float(m[2:])
                pred = f"e^{0.5 * n * math.log(zs / (zs - 1)):+.1f}"
            ke = r.get('kappa_emp')
            print(f"{m[:10]:10s} {n:4d} {r['steps']:6d} {r['kappa_J']:9.2e} {ke if ke is None else '%9.2e' % ke:>9s}"
                  f" {str(r.get('P_min')):>5s} {pred:>11s}")
        if len(ns) >= 3:
            aJ, pJ = fit(ns, [rows[n]['kappa_J'] for n in ns])
            aE, pE = fit(ns, [rows[n]['kappa_emp'] for n in ns])
            res[m] = (aJ, aE)
            print(f"   fit ln kappa = a n + p ln n + c:  kappa_J a = {aJ:+.3f} (p = {pJ:+.1f}), kappa_emp a = {aE:+.3f}"
                  f" (p = {pE:+.1f})  -> {classify(max(aJ, aE))}")
    print()
    if 'C:1' in res and 'S:0.7' in res:
        aC = max(res['C:1'])
        aS = max(res['S:0.7'])
        c1 = aC < 0.05
        c2 = aS > 0.3
        print(f'pre-registered criterion: candidate polynomial (a_C = {aC:+.3f} < 0.05): {c1}; standard at x_m = 0.7 like '
              f'e^(n/2) (a_S = {aS:+.3f} > 0.3): {c2}')
        print('criterion ' + ('MET' if (c1 and c2) else 'NOT MET') +
              ('' if c2 else ': the standard layout at the hp/ matching point is not exponentially ill-conditioned, so '
               'the candidate has nothing exponential to remove'))
    p = os.path.join(HERE, 'hp_node_test.jsonl')
    if os.path.exists(p):
        print('\nfull equations (hp_node_test.py): max|dDelta/dF| against the last node x_m')
        nr = [json.loads(l) for l in open(p)]
        base = {r['state']: r for r in nr if abs(r['x_last'] - 0.7) < 1e-9}
        print(f"{'D':>8s} {'x_m':>5s} {'max dDelta/dF':>13s} {'ln ratio':>9s} {'predicted':>9s} {'|F| reached':>11s}"
              f" {'dDelta_est':>10s}")
        for r in sorted(nr, key=lambda r: (r['D'], r['x_last'])):
            b = base.get(r['state'])
            lr = math.log(r['sens_max'] / b['sens_max']) if b else float('nan')
            print(f"{r['D']:8.4f} {r['x_last']:5.2f} {r['sens_max']:13.3e} {lr:+9.2f} {r['pred_log_ratio_fast']:+9.2f}"
                  f" {r['F_final']:11.1e} {r['dDelta_est']:+10.1e}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
