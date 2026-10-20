"""Newton (finite differences) for a zero of the M = 1 inner Stokes amplitude X_+(A, omega) = S_+ v_{+,1}.
    python3 inner_zero.py A0 wf0 [J]"""
import sys
import numpy as np
import mpmath as mp
from inner_scan import stokes
from inner_stokes import W0
A, wf = float(sys.argv[1]), float(sys.argv[2]); J = int(sys.argv[3]) if len(sys.argv) > 3 else 150
def X(A, wf, J=J):
    gam, S = stokes(mp.mpf(A), mp.mpf(wf) * W0, 1, J, mus=(1,))
    return complex(S[0][0]), complex(gam[0])
for it in range(8):
    x0, g0 = X(A, wf); h = 1e-4
    xa = (X(A + h, wf)[0] - x0) / h; xw = (X(A, wf + h)[0] - x0) / h
    Mt = np.array([[xa.real, xw.real], [xa.imag, xw.imag]])
    dA, dw = np.linalg.solve(Mt, [-x0.real, -x0.imag])
    print(f"it {it}: A {A:.8f} wf {wf:.8f} (omega {wf * float(W0):.6f})  |X| {abs(x0):.3e}  gamma {g0:.5f}", flush=True)
    A, wf = A + dA, wf + dw
    if abs(x0) < 1e-12: break
x200, g = X(A, wf, 220)
print(f"check at J = 220: |X| {abs(x200):.3e}")
