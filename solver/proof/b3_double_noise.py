"""Roundoff in the double-precision outer recursion used by outer_borel.py (proof/B3_borel.md, Section 5.4).

G_k(1) = 0 holds exactly at every order (the light-cone gauge, imposed by G_k's constant term). In
solver/largeD_late_orders.py the constant is computed in double precision as minus the sum of the polynomial's other
coefficients, which are much larger, so |G_k(1)| measures the absolute roundoff of the order. Printed for M = 1 and
compared with the size of the order (max |G_k| over the coefficients) and with the ball-arithmetic value of D_k(1).

    python3 b3_double_noise.py [K=30]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import largeD_late_orders as L

K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
th = 2 * np.pi * np.arange(L.NT) / L.NT
S, D, G, B, _ = L.run(2.72 * np.sin(th), 2 * np.pi / 1.64389375, K, verbose=False, jmax_of=lambda k: 1)
ball = None
fn = os.path.join(HERE, "b3_outer_m1_lc_K120.json")
if os.path.exists(fn):
    d = json.load(open(fn))
    key = [k for k in d["series"] if abs(float(k) - 1.0) < 1e-12][0]
    ball = [complex(float(a), float(b)) for a, b, r in d["series"][key]["D"]]
print(" k   |G_k(1)| (exact: 0)   max coeff of G_k   ratio      D_k(1) harmonic 1: double vs ball, rel. diff")
for k in range(5, K + 1, 4):
    g1 = abs(np.fft.fft(L.at(G[k], 1.0))[0] / L.NT)
    gmax = np.abs(G[k]).max()
    dd = np.fft.fft(L.at(D[k], 1.0))[1] / L.NT
    rel = abs(dd - ball[k - 1]) / abs(ball[k - 1]) if ball else float("nan")
    print(f"{k:3d}   {g1:.3e}             {gmax:.3e}          {g1 / gmax:.1e}    {rel:.2e}")
