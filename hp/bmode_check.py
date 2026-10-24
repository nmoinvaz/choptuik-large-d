"""Does the horizon b-mode gain G = x_m^-(n-1) explain the large-D floor, sigma_max and the f64 Jacobian error?
(notes/LARGE_D_MIXED.md section 10). The mass mode b - 1 ~ x^-(n-1) grows by G from the horizon to the last node x_m.

    cd hp && python3 bmode_check.py [--mutate]

Reads only repository data files (read-only):
  solver/beyond_lc/hp_node_test.jsonl   Newton floors at last node x_m = 0.6 .. 0.8
  hp/data/mixed_precision/largeD.json   sigma_max and f64-vs-dd Jacobian error, D = 8.25 .. 60
  hp/data/delta_dd.csv                   accepted |F| along the continuation
Control (--mutate): G with half the exponent, x_m^-(n-1)/2; it must fail the same tolerances.
"""
import csv
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS_DD = 2.0 ** -106   # double-double unit (about 1.2e-32)
EPS64 = 2.0 ** -52


def gain(n, xm, mutate=False):
    if mutate:   # control: a wrong rate (half the b-mode exponent)
        return xm ** (-(n - 1) / 2)
    return xm ** (-(n - 1))


def main():
    mutate = '--mutate' in sys.argv
    ok = True
    print('1. Newton floors (node test): |F| / (eps_dd G), converged runs only (F_final < F_renode)')
    ratios = []
    for line in open(f'{REPO}/solver/beyond_lc/hp_node_test.jsonl'):
        r = json.loads(line)
        n, xm = r['n'], r['x_last']
        conv = r['F_final'] < r['F_renode'] * 0.999
        q = r['F_final'] / (EPS_DD * gain(n, xm, mutate))
        tag = 'converged' if conv else 'no step (re-noded start)'
        print(f"   D={r['D']:.2f} x_m={xm:.2f}  F={r['F_final']:.2e}  G={gain(n, xm, mutate):.2e}  F/(eps G)={q:6.2f}  {tag}")
        if conv:
            ratios.append(q)
    spread = max(ratios) / min(ratios)
    print(f'   converged runs: F/(eps G) in [{min(ratios):.2f}, {max(ratios):.2f}], spread x{spread:.1f}')
    ok &= spread < 30   # a law over 7 decades of G should hold to a factor of a few

    print('2. sigma_max and f64 Jacobian error (mixed_precision/largeD.json)')
    sr, er = [], []
    for r in json.load(open(f'{REPO}/hp/data/mixed_precision/largeD.json')):
        n = r['D'] - 2
        G = gain(n, 0.7, mutate)
        s, e = r['sigma_max'] / G, r['jac_f64_vs_hi'] / (EPS64 * G)
        print(f"   D={r['D']:6.2f} G={G:.2e} sigma_max/G={s:7.2f}  (J64-Jdd)/max|J| / (eps64 G)={e:8.2f}")
        if r['D'] > 15:
            sr.append(s)
            er.append(e)
    print(f'   D > 15: sigma_max/G in [{min(sr):.2f}, {max(sr):.2f}]; f64 error/(eps64 G) in [{min(er):.2f}, {max(er):.2f}]')
    ok &= max(sr) / min(sr) < 2 and max(er) / min(er) < 3

    print('3. accepted |F| along the continuation, D = 36-60.1: fitted growth per unit D vs 1/0.7')
    xs, ys = [], []
    for row in csv.DictReader(open(f'{REPO}/hp/data/delta_dd.csv')):
        D, F = float(row['D']), float(row['F'])
        if 36 <= D <= 60.1 and F > 0:
            xs.append(D)
            ys.append(math.log(F))
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    print(f'   {len(xs)} points: x{math.exp(b):.3f} per unit D (b-mode: x{1 / 0.7:.4f}); note: |F| here is'
          ' tolerance-limited, so this is only a consistency check')
    print('ALL CONSISTENT' if ok else 'INCONSISTENT')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
