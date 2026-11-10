#!/usr/bin/env python3
"""Runs every check of solver/eeg_profile once unmutated (must pass) and once per mutation (must fail).

    python3 mutation_controls.py [--skip-finite] [--unmutate=NAME]
                                                     --skip-finite: without hpkernel, leave out finiteD_ratios.py

Exit status 1 if any unmutated check fails or any mutation passes. finiteD_ratios.py's unmutated run is not part of it:
its agreement check fails on the real data (notes/EEG_PROFILE.md 3.3); only its orientation check is used here.
rational_fit.py's rows come from rational_controls.py, which gets --unmutate=NAME (gauge, drop or A): with it one
rational mutation is not applied, so this script must exit 1.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
skip_finite = "--skip-finite" in sys.argv


def run(script, *a):
    p = subprocess.run([PY, os.path.join(HERE, script), *a], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


cases = [("forms_check.py", (), 0)] + [("forms_check.py", (f"--mutate={m}",), 1)
                                       for m in ("sign_q", "fourier_phase", "pole_dist", "norm", "recip", "ellipse",
                                                 "atanh")]
cases += [("tuned_profile.py", ("--plant",), 0), ("tuned_profile.py", ("--plant", "--mutate=drift"), 1),
          ("tuned_profile.py", ("--plant", "--mutate=sign_q"), 1)]
bad = 0
for script, a, want in cases:
    code, _ = run(script, *a)
    ok = (code == 0) == (want == 0)
    bad += not ok
    print(f"{'ok  ' if ok else 'BAD '} {script} {' '.join(a) or '(unmutated)'}: exit {code}, expected "
          f"{'0' if want == 0 else 'non-zero'}", flush=True)
code, out = run("rational_controls.py", *(a for a in sys.argv[1:] if a.startswith("--unmutate=")))
rows = [ln for ln in out.splitlines() if ln.startswith(("ok  ", "BAD "))]
print("\n".join(rows) if rows else f"BAD  rational_controls.py: exit {code}, no rows", flush=True)
# a row that misbehaves, a run without rows, or an exit status that disagrees with its rows
bad += sum(ln.startswith("BAD ") for ln in rows) or (code != 0 or not rows)
if not skip_finite:
    for a, want in (((), "PASS"), (("--mutate=reverse",), "FAIL")):
        code, out = run("finiteD_ratios.py", *a)
        line = next((ln for ln in out.splitlines() if ln.startswith("check (orientation)")), "")
        ok = line.endswith(want)
        bad += not ok
        print(f"{'ok  ' if ok else 'BAD '} finiteD_ratios.py {' '.join(a) or '(unmutated)'}: orientation check "
              f"{line.split(': ')[-1] or 'missing'}, expected {want}", flush=True)
print(f"\n{'all controls behave' if not bad else f'{bad} control(s) misbehave'}")
sys.exit(1 if bad else 0)
