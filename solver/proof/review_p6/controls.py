"""The planted errors M1-M7 of notes/P6_route3_review.md 2 and 6, run one at a time: each must exit 1, except M7 as
fixed in the protocol, which does not fire (notes 6) and is replaced by M7'.

    python3 controls.py          about 3 min; exits 0 iff every mutation fails its check and the unmutated runs pass
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

RUNS = [
    ('none', 'ampcheck.py', [], 0),
    ('M6 droppair', 'ampcheck.py', ['--mutate=droppair'], 1),
    ('none', 'explicit.py', ['--quick'], 0),
    ('M1 demix', 'explicit.py', ['--quick', '--mutate=demix'], 1),
    ('M2 dropfar', 'explicit.py', ['--quick', '--mutate=dropfar'], 1),
    ('M4 g2', 'explicit.py', ['--quick', '--mutate=g2'], 1),
    ('none', 'interior.py', ['--J1=100'], 0),
    ('M3 nuprefac', 'interior.py', ['--J1=100', '--mutate=nuprefac'], 1),
    ('M5 dropii', 'interior.py', ['--J1=100', '--mutate=dropii'], 1),
    ('none', 'head.py', ['--check'], 0),
    ('M7 a0 (did not fire, see notes 6)', 'head.py', ['--check', '--mutate=a0'], 0),
    ("M7' b7", 'head.py', ['--check', '--mutate=b7'], 1),
]



def main():
    ok = True
    print("| mutation | script | expected exit | exit | lines that fired |")
    print("|---|---|---|---|---|")
    for name, script, args, want in RUNS:
        p = subprocess.run([PY, os.path.join(HERE, script)] + args, capture_output=True, text=True, cwd=HERE)
        got = p.returncode
        good = (got == 0) if want == 0 else (got != 0)
        ok &= good
        fired = [l.strip() for l in p.stdout.splitlines() if any(k in l for k in ('FAIL', 'UNBOUNDED', ': NO', 'NO'))]
        fired = '; '.join(f[:90] for f in fired[:3]) if want else '-'
        print(f"| {name} | {script} {' '.join(args)} | {want} | {got} | {fired} |", flush=True)
    print(f"controls.py {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
