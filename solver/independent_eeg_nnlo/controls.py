#!/usr/bin/env python3
"""Mutation controls for check.py. Each row runs check.py once and compares its exit code and its set of failing
(order, equation) pairs with the expectation written here before the run. Exits 0 only if every row matches.

Usage (from the repository root): python3 solver/independent_eeg_nnlo/controls.py [--out DIR] [--full] [--psi3]
--full adds the two summary-mode runs (about 3 minutes each). --psi3 runs only the rows of psi3.py (the order past the
NNLO, issue agent-comms#40); they are part of every other run too.
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
# psi3.py, the order past the NNLO (issue agent-comms#40): Psi_3, Pi_4, Omega_4, f_4. Same row format. --quick forms
# the residuals, the boundary values (S29), (S34) and the linear part only, so each row's failing orders are those of
# the mutated closed form alone. Expectations written before the first run.
PSI3_ROWS = [
    ([], 0, set()),
    (['--quick'], 0, set()),
    (['--quick', '--mutate=Psi3'], 1, {('eps^5/2', 'S38'), ('eps^7/2', 'S39')}),
    (['--quick', '--mutate=Psi3c'], 1, {('eps^5/2', 'S38'), ('eps^7/2', 'S39')}),
    (['--quick', '--mutate=Pi4'], 1, {('eps^7/2', 'S39'), ('eps^3', 'S36')}),
    (['--quick', '--mutate=Pi4b'], 1, {('eps^7/2', 'S39'), ('eps^3', 'S36')}),
    # A constant in Pi_4 leaves (S39) alone; (S34) and (S36) through 2 bt Pi_4 catch it.
    (['--quick', '--mutate=Pi4c'], 1, {('eps^3', 'S36')}),
    (['--quick', '--mutate=Omega4'], 1, {('eps^3', 'S36'), ('eps^4', 'S37')}),
    (['--quick', '--mutate=f4'], 1, {('eps^4', 'S37')}),
    # A constant in f_4 leaves every residual alone; only (S29) f_4(1) = 0 catches it.
    (['--quick', '--mutate=f4c'], 1, set()),
    # The full mode with a mutation: the derivation no longer equals the closed form.
    (['--mutate=Psi3'], 1, {('eps^5/2', 'S38'), ('eps^7/2', 'S39')}),
    # A transcription error in this file's (S36): the residuals of the closed forms and the derivation both move.
    (['--quick', '--mutate=eqS36'], 1, {('eps^1', 'S36'), ('eps^2', 'S36'), ('eps^3', 'S36')}),
    (['--crosscheck'], 0, set()),
    (['--crosscheck', '--mutate=eqS36'], 1, {('eps^1', 'S36'), ('eps^2', 'S36'), ('eps^3', 'S36')}),
    (['--numeric'], 0, None),
    (['--numeric', '--mutate=Psi3'], 1, None),
    (['--numeric', '--mutate=Pi4'], 1, None),
    (['--numeric', '--mutate=Omega4'], 1, None),
    (['--numeric', '--mutate=f4'], 1, None),
]


def run(args, out, script='check.py'):
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as fh:
        jpath = fh.name
    cmd = [sys.executable, os.path.join(HERE, script)] + args + ['--json=' + jpath]
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(jpath) as fh:
        rep = json.load(fh)
    os.unlink(jpath)
    if out:
        tag = '_'.join(a.lstrip('-').replace('=', '-') for a in args) or 'summary'
        stem = script[:-3]
        with open(os.path.join(out, '%s_%s.log' % (stem, tag)), 'w') as fh:
            fh.write('$ python3 %s %s\n%s%s' % (script, ' '.join(args), p.stdout, p.stderr))
    fails = set()
    for k, v in rep.items():
        if isinstance(v, dict) and 'fails' in v:
            fails = {tuple(q) for q in v['fails']}
    return p.returncode, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--psi3', action='store_true')
    a = ap.parse_args()
    rows = [] if a.psi3 else [('check.py',) + r for r in ROWS + (FULL if a.full else [])]
    rows += [('psi3.py',) + r for r in PSI3_ROWS]
    bad = 0
    lines = []
    for script, args, rc_exp, fails_exp in rows:
        rc, fails = run(args, a.out, script)
        ok = rc == rc_exp and (fails_exp is None or fails == fails_exp)
        bad += not ok
        line = '%-4s %-8s %-40s exit %d (expected %d)  fails %s%s' % (
            'ok' if ok else 'BAD', script[:-3], ' '.join(args) or '(summary)', rc, rc_exp,
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
