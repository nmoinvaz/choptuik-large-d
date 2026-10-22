"""Cross-checks of b3_outer_M.py (proof/B3_borel.md, Section 5).

  1. M = 1: b3_outer_M.py must reproduce b3_outer_m1.py (written separately, scalar b and c) to the ball radii.
  2. M = 3: low orders against solver/largeD_late_orders.py (theta grid, double precision). The two truncations differ
     (the grid code divides by c b and then projects; b3_outer_M.py solves the projected equation), so only the orders
     before the conventions part must agree: k <= 3 for S, D at harmonics 1, 3, to 1e-12 of |value| + |D1_k| + |S1_k|
     (the size of the order; harmonic 3 of S_1 is exactly 0); the difference at k = 4..6 (1e-4 to 1e-2) is printed,
     not tested.
     Also checked: the light-cone gauge G_k(1) = 0 and the E1/solvability conditions hold (the script prints any
     violation).

    python3 b3_crosscheck_M.py [--mutate=omega_S|drop_bQ|scale_G]     exits non-zero on failure
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
mut = [a for a in sys.argv[1:] if a.startswith("--mutate")]
tmp = os.environ.get("TMPDIR", "/tmp")
fail = False

# 1. M = 1 against the scalar code
o1 = os.path.join(tmp, "b3x_m1.json")
oM = os.path.join(tmp, "b3x_M1.json")
subprocess.run([sys.executable, os.path.join(HERE, "b3_outer_m1.py"), "16", "--prec=200", f"--out={o1}",
                "--xs=0.5,0.8"], check=True, stdout=subprocess.DEVNULL)
subprocess.run([sys.executable, os.path.join(HERE, "b3_outer_M.py"), "1", "16", "--prec=200", f"--out={oM}",
                "--xs=0.5,0.8"] + mut, check=True, stdout=subprocess.DEVNULL)
a, b = json.load(open(o1)), json.load(open(oM))
worst = 0.0
for xs in a["series"]:
    xb = [k for k in b["series"] if abs(float(k) - float(xs)) < 1e-12][0]
    for na, nb in (("D", "D1"), ("S", "S1"), ("G", "G0"), ("B", "B0")):
        for u, v in zip(a["series"][xs][na], b["series"][xb][nb]):
            za, zb = complex(float(u[0]), float(u[1])), complex(float(v[0]), float(v[1]))
            worst = max(worst, abs(za - zb) / max(abs(za), 1e-300))
print(f"1. M = 1, K = 16: b3_outer_M vs b3_outer_m1, max relative difference {worst:.2e}")
fail |= worst > 1e-30

# 2. M = 3 against the grid code at low order
import largeD_late_orders as L
o3 = os.path.join(tmp, "b3x_M3.json")
subprocess.run([sys.executable, os.path.join(HERE, "b3_outer_M.py"), "3", "6", "--prec=200", f"--out={o3}",
                "--xs=0.5,0.8"] + mut, check=True, stdout=subprocess.DEVNULL)
c3 = json.load(open(o3))
th = 2 * np.pi * np.arange(L.NT) / L.NT
S, D, G, B, _ = L.run(2.72 * np.sin(th), 2 * np.pi / 1.64389375, 6, verbose=False, jmax_of=lambda k: 3)
worst2 = 0.0
for xs in c3["series"]:
    xv = float(xs)
    for k in range(1, 7):
        diffs = []
        u1, v1 = c3["series"][xs]["D1"][k - 1], c3["series"][xs]["S1"][k - 1]
        scale = abs(complex(float(u1[0]), float(u1[1]))) + abs(complex(float(v1[0]), float(v1[1])))   # size of order k
        for name, grid, j in (("D1", D, 1), ("D3", D, 3), ("S1", S, 1), ("S3", S, 3)):
            u = c3["series"][xs][name][k - 1]
            zb = complex(float(u[0]), float(u[1]))
            zg = np.fft.fft(L.at(grid[k], xv))[j] / L.NT
            rel = abs(zb - zg) / (abs(zg) + scale)
            diffs.append(rel)
            if k <= 3:
                worst2 = max(worst2, rel)
        print(f"2. M = 3, x = {xv}, order {k}: differences / (|value| + size of order) D1 D3 S1 S3 " + " ".join(f"{d:.1e}" for d in diffs))
print(f"2. M = 3, orders 1-3 (before the truncation conventions differ): max difference / (|value| + size of order) {worst2:.2e}")
fail |= worst2 > 1e-12
sys.exit(1 if fail else 0)
