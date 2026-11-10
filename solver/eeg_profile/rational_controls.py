#!/usr/bin/env python3
"""Runs every check of rational_fit.py once unmutated (must pass) and once per mutation (must fail)
(notes/EEG_PROFILE.md section 8).

    python3 rational_controls.py [--log=FILE] [--unmutate=NAME]

Exit status 1 if any unmutated check fails or any mutation passes. The planted meromorphic profile (--plant=poles)
enters with its first half only, "not marching", which must pass. Its second half, "converged", fails as stated
(|theta_2(4) - theta_2(3)| = 0.025 against 0.02; notes/EEG_PROFILE.md 8.3), and that run's exit status is not used.
The real-data runs use --fast (pade and ls only). --unmutate=NAME (gauge, drop or A) is the control of this script: it
runs the cases of that mutation without applying it, with what they expect unchanged, so the script must exit 1.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOG = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--log=")), None)
UNMUTATE = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--unmutate=")), None)
logf = open(LOG, "w") if LOG else None


def say(msg):
    print(msg, flush=True)
    if logf:
        logf.write(msg + "\n")
        logf.flush()


def run(*a):
    p = subprocess.run([PY, os.path.join(HERE, "rational_fit.py"), *a], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def lines(out, prefix):
    return [ln for ln in out.splitlines() if ln.startswith(prefix)]


# (arguments, expected exit, checks that must PASS, checks that must FAIL)
cases = [
    (("--plant=rational",), 0, ["P1 pade", "P2 pade", "P3 pade", "P1 ls", "P2 ls", "P3 ls"], []),
    (("--plant=rational", "--mutate=gauge"), 1, ["P1 pade", "P2 pade", "P1 ls", "P2 ls"], ["P3 pade", "P3 ls"]),
    (("--plant=rational", "--mutate=drop"), 1, [], ["P2 pade", "P2 ls"]),
    (("--plant=branch",), 0, ["M branch"], []),
    (("--plant=poles",), None, ["M poles, not marching"], []),
    (("--fast",), 0, ["S", "D"], []),
    (("--fast", "--mutate=gauge"), 1, [], ["S"]),
    (("--fast", "--mutate=A"), 1, [], ["D"]),
]
if UNMUTATE and not any(f"--mutate={UNMUTATE}" in a for a, *_ in cases):
    sys.exit(f"--unmutate={UNMUTATE}: no case has this mutation")
bad = 0
for a, want, must_pass, must_fail in cases:
    a = tuple(x for x in a if x != f"--mutate={UNMUTATE}")  # the named mutation is not applied
    code, out = run(*a)
    verdict = {}
    for ln in lines(out, "check "):
        name, rest = ln[len("check "):].split(":", 1)
        verdict[name] = rest.rstrip().endswith("PASS")
    ok = want is None or (code == 0) == (want == 0)
    ok &= all(verdict.get(n) is True for n in must_pass) and all(verdict.get(n) is False for n in must_fail)
    bad += not ok
    say(f"{'ok  ' if ok else 'BAD '} rational_fit.py {' '.join(a)}: exit {code}"
        + (f", expected {'0' if want == 0 else 'non-zero'}" if want is not None else ", not used")
        + (f"; must pass {must_pass}" if must_pass else "") + (f"; must fail {must_fail}" if must_fail else ""))
say(f"\n{'all controls behave' if not bad else f'{bad} control(s) misbehave'}")
sys.exit(1 if bad else 0)
