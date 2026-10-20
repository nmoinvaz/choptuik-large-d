#!/usr/bin/env python3
"""Data for echo.svg and spectrum.svg: the centre profile F(theta) = sqrt(n) dp/dx(x = 0, theta) of the critical solution
at a few dimensions, as complex Fourier coefficients F_k (F(theta) = sum_k Re(F_k e^{i k theta}), so |F_k| is the
amplitude of harmonic k), and the echo period Delta.

D = 4 comes from the f64 solver's state solver/data/neB_kc39_D4.0000.npy (39 harmonics; the first block of the state
is the centre slope p1 as spectral coefficients, the last entry is Delta). The other dimensions come from the
double-double runs' saved states (/home/claude/dss/hp_scan/up*_D*.npz, not in the repo), read through
hp/largeD_profiles.py; their x^1 centre coefficient is p1 on the half period grid (32 points, harmonics up to 31;
from D = 35 on the runs pin harmonics above 13 to zero).

    python3 echo_data.py            (writes solver/data/echo_spectrum.npz)
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "hp"))
H = "/home/claude/dss/hp_scan/"
OUT = os.path.join(HERE, "data", "echo_spectrum.npz")
KMAX = 39


def f64_state(path, D):
    z = np.load(path)
    kc = 39
    m = len(range(1, kc + 1, 2))
    c = z[:m] + 1j * z[m:2 * m]                       # spectral coefficients of p1 at k = 1, 3, ..., kc
    F = np.zeros(KMAX + 1, complex)
    F[1:kc + 1:2] = 2 * c * np.sqrt(D - 2)            # p1(theta) = 2 Re sum c_k e^{ik theta}; F = sqrt(n) p1
    return F, float(z[-1])


def dd_state(path):
    from largeD_profiles import solver_for, harmonics
    s, zh, zl, meta = solver_for(path)
    M, D = meta["M"], meta["D"]
    ch, cl = s.series(zh, zl, 0)
    C = (np.asarray(ch) + np.asarray(cl)).reshape(-1, 4, M)
    Fh = harmonics(C[1, 0] * np.sqrt(D - 2), True)    # F(theta) = sum_k Re(Fh_k e^{ik theta})
    F = np.zeros(KMAX + 1, complex)
    F[:min(len(Fh), KMAX + 1)] = Fh[:KMAX + 1]
    return F, float(zh[-1] + zl[-1]), D


def main():
    Ds, Fs, Deltas = [], [], []
    F, dl = f64_state(os.path.join(HERE, "data", "neB_kc39_D4.0000.npy"), 4.0)
    Ds.append(4.0); Fs.append(F); Deltas.append(dl)
    for t in (10, 12, 20, 30, 40, 60):
        fs = sorted(glob.glob(H + f"up*_D{t}.*.npz"))
        if not fs:
            print("no state near D =", t)
            continue
        F, dl, D = dd_state(fs[0])
        Ds.append(D); Fs.append(F); Deltas.append(dl)
        print(f"D {D:.4f}  Delta {dl:.10f}  |F1| {abs(F[1]):.4f}  |F3/F1| {abs(F[3] / F[1]):.4f}", flush=True)
    np.savez(OUT, D=np.array(Ds), F=np.array(Fs), Delta=np.array(Deltas))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
