"""Where the double-double solver matches decides its conditioning: a test on the full equations.

The hp/ solver matches the centre-side integration to the horizon series at its last node x_m = 0.7. The analysis in
notes/BEYOND_LIGHT_CONE.md predicts that the light-cone obstruction seen at x_m scales like the fast mode
exp(-n Phi(x_m)), Phi(x) = ln x - ln(1 - x^2)/2 (Phi = 0 at x = 1/sqrt 2), so the sensitivity max|dDelta/dF| should
change by about exp(n [Phi(x_m') - Phi(0.7)]) when the last node moves to x_m', and that the horizon series' inward
b-mode (b - 1 ~ x^-n) sets a residual floor growing like x_m^-n.

For each saved scan state (nodes 0.3..0.7, M = 32, harmonics above 13 frozen as the scan ran), this script re-nodes the
solution to a layout whose last node is x_m' (hpkernel.driver.renode: Taylor integration from the old nodes), re-converges
it with plain double-double Newton to its floor, and records the floor |F|, Delta, the linearised Delta correction and
max|dDelta/dF| per residual block (as hp/scan_hp.py's sensitivity()).

Uses the installed hpkernel (not rebuilt). Output: hp_node_test.jsonl (one line per (D, x_m'), resumable).
Usage: python3 hp_node_test.py [--states up7_D40.0201,up8_D50.0075,up13_D60.0020] [--last 0.6,0.65,0.7,0.75,0.8]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HP = os.path.join(HERE, '..', '..', 'hp')
sys.path.insert(0, HP)
from hpkernel.driver import load, make_solver, newton_solve, renode, delta_correction, dd_norm  # noqa: E402


def phi(x):
    return math.log(x) - 0.5 * math.log(1 - x * x)


def sensitivity(s, M):
    L = s.size
    e = np.zeros(L)
    e[-1] = 1.0
    yh, yl = s.lu_solve_t(e, np.zeros(L))
    nb = (L - 1) // (4 * M)
    return [float(np.abs(yh[b * 4 * M:(b + 1) * 4 * M]).max()) for b in range(nb)]


def tails(s, zh, zl, nodes, M):
    out = []
    for which, x in ((0, nodes[0]), (1, 1 - nodes[-1])):
        h, l = s.series(zh, zl, which)
        c = np.abs((np.asarray(h) + np.asarray(l)).reshape(-1, 4, M)).max(axis=(1, 2))
        k = np.arange(len(c))
        out.append(float((c * x ** k)[-3:].max()))
    return out


def layout(xl):
    base = [0.3, 0.4, 0.5, 0.6, 0.7]
    if xl <= 0.7:
        return [x for x in base if x < xl - 1e-9] + [xl]
    return base + [xl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', default='up7_D40.0201,up8_D50.0075,up13_D60.0020')
    ap.add_argument('--last', default='0.7,0.6,0.8,0.65,0.75')
    ap.add_argument('--kc', type=int, default=420)
    ap.add_argument('--ks', type=int, default=160)
    ap.add_argument('--kfreeze', type=int, default=13)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(HERE, 'hp_node_test.jsonl'))
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r['state'], r['x_last']))
    for st in a.states.split(','):
        zh0, zl0, meta = load(os.path.join(HP, 'data', 'states', st + '.npz'))
        D, M, nodes0 = meta['D'], meta['M'], list(meta['nodes'])
        n = D - 2
        s0 = make_solver(D, M, nodes=nodes0, kt=meta['kt'], kc=a.kc, ks=a.ks, threads=a.threads, kfreeze=a.kfreeze)
        s0.set_segs(s0.plan(zh0, zl0, 1e-32, 0.05))
        for xl in [float(v) for v in a.last.split(',')]:
            if (st, xl) in done:
                continue
            t0 = time.time()
            nodes = layout(xl)
            zh, zl = renode(s0, zh0, zl0, nodes0, nodes, M)
            s = make_solver(D, M, nodes=nodes, kt=meta['kt'], kc=a.kc, ks=a.ks, threads=a.threads, kfreeze=a.kfreeze)
            s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
            F0 = dd_norm(*s.residual(zh, zl))
            zh2, zl2, hist = newton_solve(s, zh, zl, tol=1e-40, it_max=7, verbose=False)
            # make sure the stored LU is at the final iterate
            s.newton(zh2, zl2)
            Fh, Fl = s.residual(zh2, zl2)
            sens = sensitivity(s, M)
            rec = dict(state=st, D=D, n=n, x_last=xl, nodes=nodes, F_renode=F0, F_hist=[h['F'] for h in hist],
                       F_final=dd_norm(Fh, Fl), Delta=float(zh2[-1]), Delta_lo=float(zl2[-1]),
                       dDelta_est=delta_correction(s, Fh, Fl), sens_blocks=sens, sens_max=max(sens),
                       tails=tails(s, zh2, zl2, nodes, M), pred_log_ratio_fast=n * (phi(xl) - phi(0.7)),
                       pred_log_ratio_floor=-n * math.log(xl / 0.7), seconds=round(time.time() - t0, 1))
            with open(a.out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"D={D:.4f} x_last={xl}: |F| renode {F0:.1e} -> {rec['F_final']:.1e} (min {min(rec['F_hist'] + [rec['F_final']]):.1e})"
                  f"  Delta {rec['Delta']:.16f}  dDelta_est {rec['dDelta_est']:+.1e}  max dDelta/dF {rec['sens_max']:.2e}"
                  f"  blocks {['%.0e' % v for v in sens]}  tails {['%.0e' % v for v in rec['tails']]}  ({rec['seconds']} s)",
                  flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
