"""Runs the Borel-plane scans and pair fits quoted in proof/B3_borel.md, Section 5, and writes them to one log.

    python3 b3_scan_all.py > b3_scan_all.log
    python3 b3_scan_all.py --set=M5 > b3_scan_M5.log      (the M = 5 file, b3_outer_M5_K70.json)

Inputs: b3_outer_m1_lc_K120.json, b3_outer_m1_centre_K120.json (b3_outer_m1.py), b3_outer_M3_K100.json
(b3_outer_M.py). Each line of the log comes from b3_borel_scan.py or b3_pair_fit.py, run as a subprocess.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LC, CE, M3 = (os.path.join(HERE, f) for f in ("b3_outer_m1_lc_K120.json", "b3_outer_m1_centre_K120.json",
                                              "b3_outer_M3_K100.json"))
RUNS = [
    ("planted-truth test of the scanner", ["b3_borel_scan.py", "none", "planted", "--K=120", "--n=4"]),
    ("M = 1, light-cone gauge, c(0) = 1 - sum G_k(0) eps^k", ["b3_borel_scan.py", LC, "g0", "--n=6"]),
    ("M = 1, centre gauge, light-cone position X*", ["b3_borel_scan.py", CE, "xstar", "--n=6"]),
    ("M = 1, light-cone gauge, D at x = 0.5", ["b3_borel_scan.py", LC, "D", "--x=0.5", "--n=6"]),
    ("M = 1, light-cone gauge, D at x = 0.8", ["b3_borel_scan.py", LC, "D", "--x=0.8", "--n=6"]),
    ("M = 1, light-cone gauge, B at x = 0.95", ["b3_borel_scan.py", LC, "B", "--x=0.95", "--n=6"]),
    ("M = 1, light-cone gauge, D at x = 1", ["b3_borel_scan.py", LC, "D", "--x=1", "--n=6"]),
    ("M = 1, centre gauge, D at x = 0.5", ["b3_borel_scan.py", CE, "D", "--x=0.5", "--n=4"]),
    ("M = 3, light-cone gauge, harmonic 0 of c(0)", ["b3_borel_scan.py", M3, "g0:0", "--n=6"]),
    ("M = 3, light-cone gauge, harmonic 2 of c(0)", ["b3_borel_scan.py", M3, "g0:2", "--n=6"]),
    ("M = 3, light-cone gauge, D1 at x = 0.8", ["b3_borel_scan.py", M3, "D1", "--x=0.8", "--n=6"]),
    ("M = 3, light-cone gauge, D1 at x = 0.95", ["b3_borel_scan.py", M3, "D1", "--x=0.95", "--n=6"]),
    ("pair fit, planted pair at +-i pi", ["b3_pair_fit.py", "--planted=0,3.14159265358979"]),
    ("pair fit, M = 1 light-cone gauge c(0)", ["b3_pair_fit.py", LC, "g0"]),
    ("pair fit, M = 1 centre gauge X*", ["b3_pair_fit.py", CE, "xstar"]),
]
M5 = os.path.join(HERE, "b3_outer_M5_K70.json")
if "--set=M5" in sys.argv:
    RUNS = [(f"M = 5, light-cone gauge, harmonic {mu} of c(0)", ["b3_borel_scan.py", M5, f"g0:{mu}", "--n=6"])
            for mu in (0, 2, 4)]
    RUNS += [(f"M = 5, light-cone gauge, D1 at x = {x}", ["b3_borel_scan.py", M5, "D1", f"--x={x}", "--n=6"])
             for x in ("0.5", "0.8")]
for label, cmd in RUNS:
    print(f"==== {label}: python3 {' '.join(os.path.basename(c) for c in cmd)}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, cmd[0])] + cmd[1:], capture_output=True, text=True)
    print(r.stdout.rstrip() + (("\n" + r.stderr.rstrip()) if r.returncode else ""), flush=True)
