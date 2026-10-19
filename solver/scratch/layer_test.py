"""Boundary-layer test: evaluate the horizon (SSH) series of converged solutions at u = 1 - x = eta / sqrt(n) and see
whether the fields collapse onto a universal profile in eta when scaled with the amplitude (p, m ~ n^-1/2).

    python3 layer_test.py
"""
import glob
import re

import numpy as np

from spec_long import make_long
from spec_c import freeze_steps
from rust_solver import NODES_L


def horizon(D, z, nodes):
    S = freeze_steps(make_long(23, D - 2, nodes=nodes, Ks=60), z[-1])
    parts, _ = S.split(z[None, :])
    parts = [p[0] for p in parts]
    mS = S.phys(parts[-2][None, :], "odd")
    aS = S.phys(parts[-1][None, :], "even")
    om = np.array([2 * np.pi / z[-1]])
    P, M, A, C = S.ssh_series(mS, aS, om)
    return S, (P, M, A, C)


def h1(f):
    F = np.fft.rfft(f) / len(f)
    return 2 * np.abs(F[1]), np.angle(F[1])


files = {}
for f in sorted(glob.glob("neB_rk2_D*.npy")):
    files[float(re.findall(r"D([0-9.]+)\.npy", f)[0])] = (f, NODES_L)
for f in sorted(glob.glob("ne7_js_rk3_s111111118_D*.npy")):
    files[float(re.findall(r"D([0-9.]+)\.npy", f)[0])] = (f, NODES_L + [0.7])
if __name__ == "__main__":
    pick = [d for d in sorted(files) if any(abs(d - t) < 0.3 for t in (8, 12, 16, 20, 24, 28, 30.1, 34.2, 38.2))]
    etas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    print("first harmonic of sqrt(n)*m, sqrt(n)*p and of n*(1-c), n*(a^2-1) at u = eta/sqrt(n)")
    for D in pick:
        f, nodes = files[D]
        n = D - 2
        S, (P, M, A, C) = horizon(D, np.load(f), nodes)
        row = []
        for eta in etas:
            u = eta / np.sqrt(n)
            p, m, a, c = (S.eval_series(Q, u)[0] for Q in (P, M, A, C))
            row.append((h1(np.sqrt(n) * m)[0], h1(np.sqrt(n) * p)[0], np.mean(n * (1 - c)), np.mean(n * (a ** 2 - 1))))
        print(f"D={D:6.2f} n={n:5.2f}  sqrt(n)m: " + " ".join(f"{r[0]:6.3f}" for r in row)
              + "  | sqrt(n)p: " + " ".join(f"{r[1]:6.3f}" for r in row))
        print(f"{'':16s} <n(1-c)>: " + " ".join(f"{r[2]:6.3f}" for r in row) + "  | <n(a^2-1)>: " + " ".join(f"{r[3]:6.3f}" for r in row))
