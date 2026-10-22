"""Cross-check of b3_outer_m1.py (ball arithmetic, harmonic coefficients) against solver/largeD_late_orders.py (double
precision, theta grid) for the M = 1 model at low order, where double precision is still accurate. The two codes share
no code; they share the equations (README, Proposition 1) and the data F = 2.72 sin(theta), w = 2 pi/1.64389375.

    python3 b3_crosscheck_m1.py [K=12] [--mutate=omega_S|drop_bQ|scale_G]

Exits non-zero if the relative difference of D_k(x), G_k(x), B_{k+1}(x) (x = 0.5, 0.8) exceeds 1e-8 at any k <= K.
With --mutate (passed to the ball code only) it must fail.
"""
import os
import subprocess
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import largeD_late_orders as L

K = int([a for a in sys.argv[1:] if not a.startswith("--")][0]) if [a for a in sys.argv[1:] if not a.startswith("--")] else 12
mut = [a for a in sys.argv[1:] if a.startswith("--mutate")]
out = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"b3_xcheck_K{K}.json")
cmd = [sys.executable, os.path.join(HERE, "b3_outer_m1.py"), str(K), "--prec=128", f"--out={out}", "--xs=0.5,0.8"] + mut
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
ball = json.load(open(out))

th = 2 * np.pi * np.arange(L.NT) / L.NT
S, D, G, B, _ = L.run(2.72 * np.sin(th), 2 * np.pi / 1.64389375, K, verbose=False, jmax_of=lambda k: 1)


def harm(p, xv, j):
    return np.fft.fft(L.at(p, xv))[j] / L.NT


worst = 0.0
for xs in ("0.5", "0.8"):
    xv = float(xs)
    ser = ball["series"][[k for k in ball["series"] if abs(float(k) - xv) < 1e-12][0]]
    for name, grid, j, shift in (("D", D, 1, 0), ("G", G, 0, 0), ("B", B, 0, 1)):
        for k in range(1, K + 1):
            a = complex(float(ser[name][k - 1][0]), float(ser[name][k - 1][1]))
            g = harm(grid[k + shift], xv, j)
            rel = abs(a - g) / max(abs(g), 1e-300)
            worst = max(worst, rel)
print(f"max relative difference over k <= {K}, x = 0.5, 0.8, D/G/B: {worst:.2e}")
sys.exit(0 if worst < 1e-8 else 1)
