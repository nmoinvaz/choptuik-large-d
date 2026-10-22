"""Harmonic content of a DSSEll solution: |F_k| of the centre data (q or p1, c0), of p, c at the first nodes and of the
horizon data, plus min c(tau, 0) and the psi-constraint violation (a resolution diagnostic).
usage: python ell_spectrum.py sol.npy D l kc"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD

f, D, l, kc = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
z = np.load(f)
S = DSSEll(kc=kc, nodes=NODES_STD, n=D - 2, ell=l, Delta_ref=z[-1])
fl = S.fields(z)


def amp(y):
    return np.abs(np.fft.rfft(y)) / len(y) * 2


ks = [1, 3, 5, 7, 9, 11, 13, 15, 19, 23, 27, 31, 39]
ks = [k for k in ks if k <= kc]
print(f"D {D}  Delta {z[-1]:.8f}  min c0 {fl['c0'].min():.4f}  max c0 {fl['c0'].max():.4f}  constraint {S.constraint(z):.2e}")
print("harmonic       " + " ".join(f"{k:8d}" for k in ks))
rows = [("q (centre)", fl["d"]), ("c0 (even, k-1)", None), ("p @ 0.02", fl["states"][0][0]),
        ("p @ 0.1", fl["states"][2][0]), ("c @ 0.05", None), ("m @ 0.5", fl["states"][7][1]), ("m(1)", fl["mS"])]
for name, y in rows:
    if name.startswith("c0"):
        A = amp(fl["c0"]); print(f"{name:14s} " + " ".join(f"{A[k - 1]:8.1e}" for k in ks)); continue
    if name.startswith("c @"):
        A = amp(fl["states"][1][3]); print(f"{name:14s} " + " ".join(f"{A[k - 1]:8.1e}" for k in ks)); continue
    A = amp(y)
    print(f"{name:14s} " + " ".join(f"{A[k]:8.1e}" for k in ks))
