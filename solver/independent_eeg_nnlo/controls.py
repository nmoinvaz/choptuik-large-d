#!/usr/bin/env python3
"""Mutation controls for check.py. Each row runs check.py once and compares its exit code and its set of failing
(order, equation) pairs with the expectation written here before the run. Exits 0 only if every row matches.

Usage (from the repository root): python3 solver/independent_eeg_nnlo/controls.py [--out DIR] [--full]
--full adds the two summary-mode runs (about 3 minutes each).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# (args, expected exit, expected failing (order, equation) pairs). Orders are powers of eps.
ROWS = [
    # NLO control, unmutated and with one printed NLO coefficient changed.
    (['--nlo'], 0, set()),
    (['--nlo', '--mutate=S40'], 1, {('eps^2', 'S37')}),
    (['--nlo', '--mutate=S41'], 1, {('eps^1', 'S36'), ('eps^2', 'S37')}),
    (['--nlo', '--mutate=S42'], 1, {('eps^1', 'S36'), ('eps^3/2', 'S38'), ('eps^3/2', 'S39')}),
    (['--nlo', '--mutate=S43'], 1, {('eps^3/2', 'S38')}),
    # The claimed set, unmutated and with one coefficient changed.
    (['--set=claimed'], 0, set()),
    (['--set=claimed', '--mutate=claimPi3'], 1, {('eps^5/2', 'S39'), ('eps^3', 'S37')}),
    (['--set=claimed', '--mutate=claimC4'], 1, {('eps^3', 'S37')}),
    (['--set=claimed', '--mutate=S49'], 1, {('eps^3', 'S37')}),
    (['--set=claimed', '--mutate=S45'], 1, {('eps^2', 'S36'), ('eps^3', 'S37')}),
    # The printed set, unmutated and with one printed NNLO coefficient changed.
    (['--set=printed'], 1, {('eps^5/2', 'S39')}),
    (['--set=printed', '--mutate=S45'], 1, {('eps^2', 'S36'), ('eps^3', 'S37'), ('eps^5/2', 'S39')}),
    (['--set=printed', '--mutate=S46'], 1, {('eps^5/2', 'S39'), ('eps^3', 'S37')}),
    (['--set=printed', '--mutate=S49'], 1, {('eps^3', 'S37'), ('eps^5/2', 'S39')}),
    # Reading B (plain beta = sqrt(eps) beta-tilde) fails already at NLO through (S43).
    (['--set=printedB'], 1, {('eps^3/2', 'S38'), ('eps^2', 'S36'), ('eps^5/2', 'S39'), ('eps^3', 'S37')}),
    (['--set=derived'], 0, set()),
    # Numerical eps-scaling of the (S39) residual.
    (['--numeric'], 0, None),
    (['--numeric', '--set=printed'], 1, None),
    (['--numeric', '--mutate=claimPi3'], 1, None),
]
FULL = [
    ([], 0, None),
    (['--mutate=claimPi3'], 6, None),     # 2 claimed fails, 4 derived != claimed
]


def run(args, out):
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as fh:
        jpath = fh.name
    cmd = [sys.executable, os.path.join(HERE, 'check.py')] + args + ['--json=' + jpath]
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(jpath) as fh:
        rep = json.load(fh)
    os.unlink(jpath)
    if out:
        tag = '_'.join(a.lstrip('-').replace('=', '-') for a in args) or 'summary'
        with open(os.path.join(out, 'check_%s.log' % tag), 'w') as fh:
            fh.write('$ python3 check.py %s\n%s%s' % (' '.join(args), p.stdout, p.stderr))
    fails = set()
    for k, v in rep.items():
        if isinstance(v, dict) and 'fails' in v:
            fails = {tuple(q) for q in v['fails']}
    return p.returncode, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='')
    ap.add_argument('--full', action='store_true')
    a = ap.parse_args()
    rows = ROWS + (FULL if a.full else [])
    bad = 0
    lines = []
    for args, rc_exp, fails_exp in rows:
        rc, fails = run(args, a.out)
        ok = rc == rc_exp and (fails_exp is None or fails == fails_exp)
        bad += not ok
        line = '%-4s %-40s exit %d (expected %d)  fails %s%s' % (
            'ok' if ok else 'BAD', ' '.join(args) or '(summary)', rc, rc_exp,
            sorted(fails) if fails_exp is not None else '-',
            '' if ok or fails_exp is None else '  expected %s' % sorted(fails_exp))
        print(line, flush=True)
        lines.append(line)
    print('\n%d of %d controls as expected' % (len(rows) - bad, len(rows)))
    if a.out:
        with open(os.path.join(a.out, 'controls.log'), 'w') as fh:
            fh.write('\n'.join(lines) + '\n\n%d of %d controls as expected\n' % (len(rows) - bad, len(rows)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
