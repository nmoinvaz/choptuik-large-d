"""Runs every check of proof/B3_borel.md with its mutation controls and writes one log per run, then a summary.

    python3 b3_checks_all.py [--quick] [--only=name1,name2]
(from solver/proof; --quick skips the slow strip checks; --only reruns the named runs, e.g. after a restart, and the
summary covers those runs only)

Expected exit codes are stated per run: a check exits 0 on the real data and non-zero under its mutation. The summary
line says whether every run behaved as expected; the script exits non-zero otherwise.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = "--quick" in sys.argv
RUNS = [  # (log name, command, expected exit code)
    ("b3_crosscheck_m1", ["b3_crosscheck_m1.py", "12"], 0),
    ("b3_crosscheck_m1_mut_omega_S", ["b3_crosscheck_m1.py", "12", "--mutate=omega_S"], 1),
    ("b3_crosscheck_M", ["b3_crosscheck_M.py"], 0),
    ("b3_crosscheck_M_mut_scale_G", ["b3_crosscheck_M.py", "--mutate=scale_G"], 1),
    ("b3_gevrey_certificate_rho0.75", ["b3_gevrey_certificate.py", "--rho=0.75"], 0),
    ("b3_gevrey_certificate_rho1.05", ["b3_gevrey_certificate.py", "--rho=1.05"], 0),
    ("b3_gevrey_certificate_lc_rho1.05", ["b3_gevrey_certificate.py", "--gauge=lc", "--rho=1.05"], 0),
    ("b3_gevrey_certificate_mutate_m0", ["b3_gevrey_certificate.py", "--rho=0.75", "--mutate=m0"], 1),
    ("b3_old_check_controls", ["b3_old_check_controls.py"], 0),
]
if not QUICK:
    RUNS += [
        ("b3_strip_check_M1", ["b3_strip_check.py", "b3_outer_m1_lc_K120.json", "--xs=0.5,0.8,0.95,1",
                               "--names=D,B,G"], 0),
        ("b3_strip_check_planted_1.3", ["b3_strip_check.py", "b3_outer_m1_lc_K120.json", "--xs=0.5,0.95", "--names=D",
                                        "--plant=1.3,1e-3"], 1),
        ("b3_strip_check_planted_2.2", ["b3_strip_check.py", "b3_outer_m1_lc_K120.json", "--xs=0.5,0.95", "--names=D",
                                        "--plant=2.2,1e-3"], 1),
        ("b3_strip_check_M3", ["b3_strip_check.py", "b3_outer_M3_K100.json", "--names=D1,D3,B0,B2,G0"], None),
        ("b3_taylor_rational", ["b3_taylor_rational.py", "5"], 0),
        ("b3_taylor_rational_mut_eq", ["b3_taylor_rational.py", "3", "--mutate=eq"], 1),   # order 3 suffices
    ]
ONLY = [a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")]
if ONLY:
    RUNS = [r for r in RUNS if r[0] in ONLY[0]]
env = dict(os.environ, TMPDIR=os.environ.get("TMPDIR", "/tmp"))
allok = True
for name, cmd, expect in RUNS:
    with open(os.path.join(HERE, name + ".log"), "w") as f:
        r = subprocess.run([sys.executable, os.path.join(HERE, cmd[0])] + cmd[1:], stdout=f, stderr=subprocess.STDOUT,
                           cwd=HERE, env=env)
    ok = expect is None or r.returncode == expect
    allok &= ok
    print(f"{name:36s} exit {r.returncode} (expected {expect if expect is not None else 'any: reported'}) "
          f"{'ok' if ok else 'UNEXPECTED'}", flush=True)
print("ALL AS EXPECTED" if allok else "SOME RUN NOT AS EXPECTED")
sys.exit(0 if allok else 1)
