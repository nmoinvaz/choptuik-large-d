#!/usr/bin/env python3
"""Controls for check.py (issue #165): planted profiles first, then every mutation.

Planted profiles are written in the data file's format (7 significant digits in |F_nu/F_1|, 5 decimals in the phase,
relative noise at the file's own truncation estimate per nu) and read back through check.load and check.analyze, the
code path of the real data. Pass criteria, fixed before the first run on the data:

  exact    a_k = w^k, w = 0.0551 e^{0.0985 i} (Eq. (16), paper's q, d, A* = sqrt2 A).
           Eq.(16) deviation < 1e-4 at every nu <= 19; (S62) recovers q, d within 2e-5; one-zero |b| < 1e-5 and its
           deviation < 1e-4; ratio-modulus drift max |D| < 1e-6.
  smooth   a_k = w^k (1 + 0.003 k^2). Must FAIL the exact criterion: Eq.(16) deviation > 1e-2 somewhere.
  kink     a_k = w^k, a_6 (nu = 13) x 1.05. Must FAIL the one-zero 3% criterion.
  branch   a_k = rho^k (1 + alpha/nu^2)/(1 + alpha), nu = 2k+1, a pole with a weaker branch point at it.
           Drift class 'pow' with p+1 in [3.5, 4.5]; |gamma| < 0.02; Pade [L/2] error > 1e-4 for L <= 2.
  pole2    a_k = (rho^k + c sigma^k)/(1 + c), sigma = 0.3 rho, two poles. Drift class 'exp' with lam within 10% of
           sqrt(0.3) per unit nu; Pade [L/2] error < 1e-4 for 1 <= L <= 5.
  rho = 0.05745 e^{0.0728 i}; alpha and c are set so that F_3/F_1 equals the data's.

Usage: controls.py [--data PATH] [--out DIR]. Exit 0 when every planted criterion holds and every mutation flips its
target check.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check as ck  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

TARGET = {
    'A_conj': 'I.A ', 'ellipse_axes': 'I.ell h on the ellipse', 'ellipse_ode': "I.ell h'' + h",
    'B_sign': 'I.B ', 'C_sign': 'I.C ', 'D_delta': 'I.D ', 'E_noi': 'I.E ', 'E_sqrt': 'I.E ',
    'pole_sign': 'P.pole Eq.(16) is singular', 'ratio_plus': 'P.ratio Eq.(16)', 'delta0': 'P.delta no time shift',
    'treversal': 'P.delta time reversal', 'data_kink': 'N.onezero within 3% of every |F_nu|, nu <= 19 (A* = sqrt2 A)',
    'data_A': 'N.A A = |F_1|', 'data_frame': 'N.S62',
}


def write_planted(path, a, err, A=2.72, label=''):
    nus = list(range(1, 2 * len(a), 2))
    rng = np.random.default_rng(165)
    F = {}
    for kk, nu in enumerate(nus):
        e = err.get(nu, 1e-3)
        F[nu] = a[kk] * (1 + e * (rng.standard_normal() + 1j * rng.standard_normal()) / math.sqrt(2)) \
            if nu > 1 else 1.0 + 0j
    S = sum(abs(v) ** 2 for v in F.values())
    with open(path, 'w') as f:
        f.write(f'== planted {label}, A = {A}\n')
        f.write(f'<F^2> = {A * A / 2 * S:.6f}   <bt^2> = N0 = {A * A * S:.6f}   (planted)\n')
        f.write(' nu  |F_nu/F_1|     arg(F_nu/F_1)  A-terms last/|F|   r_nu=F_{nu+2}/F_nu  |r_nu|    arg r_nu  rate\n')
        for nu in nus:
            f.write(f' {nu:2d}  {abs(F[nu]):.6e}  {np.angle(F[nu]):+.5f}       {10:2d}   {err.get(nu, 0):.1e}'
                    f'    0 0 0 0\n')
        f.write("paper's fit (planted file: no derived lines)\n")


def planted_sets(rdat):
    w = ck.PAPER['q'] * np.exp(1j * ck.PAPER['delta'])
    rho = 0.05745 * np.exp(0.0728j)
    K = 13
    kk = np.arange(K)
    nu = 2 * kk + 1
    out = {}
    out['exact'] = w ** kk
    out['smooth'] = w ** kk * (1 + 0.003 * kk ** 2)
    a = (w ** kk).astype(complex)
    a[6] *= 1.05
    out['kink'] = a
    # branch: r_1 = rho (1 + alpha/9)/(1 + alpha) = rdat -> alpha
    c = rdat / rho
    alpha = (c - 1) / (1 / 9 - c)
    out['branch'] = rho ** kk * (1 + alpha / nu ** 2) / (1 + alpha)
    sig = 0.3 * rho
    cc = (rho - rdat) / (rdat - sig)
    out['pole2'] = (rho ** kk + cc * sig ** kk) / (1 + cc)
    return out


def run_planted(data, outdir, log):
    blk = ck.load(data)[0]
    rdat = blk['F'][3] / blk['F'][1]
    sets = planted_sets(rdat)
    ok_all = True
    res = {}
    for name, a in sets.items():
        p = os.path.join(outdir, f'planted_{name}.txt')
        # exact: A* = 3.8455 = sqrt2 A, so the paper's Eq.(16) is the planted truth (the first run used A = 2.72)
        A = ck.PAPER['Astar'] / math.sqrt(2) if name == 'exact' else 2.72
        write_planted(p, a, blk['err'], A=A, label=name)
        b = ck.load(p)[0]
        o = ck.analyze(b)
        dev16 = max(abs(abs(v) - 1) + abs(np.angle(v)) for v in o['dev16_paper'].values())
        q, d, As, r = o['s62_repoN0']
        bq = o['r3'] - o['r1']
        oz = max(abs(abs(v) - 1) for v in o['onezero_sqrt2A'].values())
        dr = o['drift']
        maxD = float(np.max(np.abs(dr['D'])))
        pd2 = {L: e for (L, M), e in o['pade'].items() if M == 2}
        line = (f'{name:7s} dev16 {dev16:.2e}  S62 q {q:.6f} d {d:.6f}  |b| {abs(bq):.2e}  one-zero {oz:.2e}  '
                f'drift max|D| {maxD:.2e} class {dr["cls"]} p+1 {dr["p1"]:.2f} lam {dr["lam"]:.3f} '
                f'gamma {o["gamma"]:+.4f}  Pade[L/2] ' + ' '.join(f'{L}:{e:.1e}' for L, e in sorted(pd2.items())))
        if name == 'exact':
            ok = (dev16 < 1e-4 and abs(q - ck.PAPER['q']) < 2e-5 and abs(d - ck.PAPER['delta']) < 2e-5
                  and abs(bq) < 1e-5 and oz < 1e-4 and maxD < 1e-6)
        elif name == 'smooth':
            ok = dev16 > 1e-2  # the exact criterion must fail
        elif name == 'kink':
            ok = oz >= 0.03  # the one-zero criterion must fail
        elif name == 'branch':
            ok = (dr['cls'] == 'pow' and 3.5 <= dr['p1'] <= 4.5 and abs(o['gamma']) < 0.02
                  and all(pd2[L] > 1e-4 for L in (0, 1, 2)))
        elif name == 'pole2':
            ok = (dr['cls'] == 'exp' and abs(dr['lam'] / math.sqrt(0.3) - 1) < 0.1
                  and all(pd2[L] < 1e-4 for L in range(1, 6)))
        ok_all &= ok
        res[name] = ok
        log(f'  [{"PASS" if ok else "FAIL"}] planted {line}')
    return ok_all, res


def run_mutations(data, outdir, log):
    py = sys.executable
    base_json = os.path.join(outdir, 'check_base.json')
    subprocess.run([py, os.path.join(HERE, 'check.py'), '--data', data, '--json', base_json],
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    base = {r['name']: r['ok'] for r in json.load(open(base_json))['checks']}
    ok_all = True
    for mut, tgt in TARGET.items():
        j = os.path.join(outdir, f'check_mut_{mut}.json')
        p = subprocess.run([py, os.path.join(HERE, 'check.py'), '--data', data, '--mutate', mut, '--json', j],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        rows = json.load(open(j))['checks']
        flipped = [r['name'] for r in rows if r['name'] in base and r['ok'] != base[r['name']]]
        hit = [n for n in flipped if n.startswith(tgt)]
        ok = bool(hit) and p.returncode != 0
        ok_all &= ok
        log(f'  [{"PASS" if ok else "FAIL"}] mutation {mut:12s} exit {p.returncode}; target "{tgt.strip()}" '
            f'{"flipped" if hit else "NOT flipped"}; all flipped: {flipped}')
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=ck.DEFAULT_DATA)
    ap.add_argument('--out', required=True, help='output folder (the raw-data task folder)')
    ap.add_argument('--skip_mutations', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'controls.log'), 'w')

    def log(s):
        print(s)
        logf.write(s + '\n')
        logf.flush()

    log('planted profiles (criteria in the docstring, fixed before the data run)')
    okp, _ = run_planted(a.data, a.out, log)
    okm = True
    if not a.skip_mutations:
        log('mutations (each must flip its target check and make check.py exit non-zero)')
        okm = run_mutations(a.data, a.out, log)
    log(f'controls verdict: {"PASS" if okp and okm else "FAIL"}')
    sys.exit(0 if okp and okm else 1)


if __name__ == '__main__':
    main()
