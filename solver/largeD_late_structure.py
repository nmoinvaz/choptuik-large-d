"""structure of the nonlinear late orders (solver/largeD_late_orders.py recursion, F = A sin theta)."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import largeD_late_orders as L
NT = L.NT
th = 2 * np.pi * np.arange(NT) / NT
A = float(sys.argv[1]) if len(sys.argv) > 1 else 2.72
om = (float(sys.argv[2]) if len(sys.argv) > 2 else 1.0) * 2 * np.pi / 1.64389375
K = int(sys.argv[3]) if len(sys.argv) > 3 else 28
S, D, G, B, st = L.run(A * np.sin(th), om, K, verbose=False)
def nrm(p, xv):
    return np.abs(L.at(p, xv)).max()
print(f"A {A} omega {om:.4f}")
print(" k   ratio/k at x=0.25    0.5     0.75     1.0  | top-degree share at x=1 | harmonic peak j of D_k(1) | log-slope of harmonics")
for k in range(8, K):
    r = [nrm(D[k + 1], xv) / nrm(D[k], xv) / k for xv in (0.25, 0.5, 0.75, 1.0)]
    deg = D[k + 1].shape[0] - 1
    nz = [i for i in range(D[k + 1].shape[0]) if np.abs(D[k + 1][i]).max() > 0]
    top = np.abs(D[k + 1][nz[-1]]).max() / nrm(D[k + 1], 1.0)
    h = np.abs(np.fft.rfft(L.at(D[k + 1], 1.0))) / NT
    jp = int(np.argmax(h))
    print(f"{k:3d}  " + "  ".join(f"{v:7.3f}" for v in r) + f"  |  deg {nz[-1]} share {top:.3f}  |  j_peak {jp:3d}", flush=True)
# coefficient structure: |coefficient of x^j in D_k| for the last order, by j
k = K
cf = [np.abs(D[k][i]).max() for i in range(D[k].shape[0])]
print("order", k, "x-power coefficient magnitudes:", " ".join(f"{i}:{c:.1e}" for i, c in enumerate(cf) if c > 0))
