"""Zeros of the one-harmonic (M = 1) inner Stokes jump: grid over (A, w), F = A sin(theta).
Prints R_1 = J_1(xi1)/(2 pi i K' f_lin(xi1) 2F_1) (linear normalization; zeros of J are what matter)."""
import sys
import numpy as np
from lateral_np import jump, Fharm, W0
xi1 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
As = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
wfs = np.array([0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0])
res = np.zeros((len(As), len(wfs)), complex)
for i, A in enumerate(As):
    for k, wf in enumerate(wfs):
        try:
            R, JD, JS, err = jump(1, W0 * wf, Fharm(A), r1=xi1)
            res[i, k] = R[1]
        except Exception as ex:
            res[i, k] = np.nan
    print(f"A {A}: " + "  ".join(f"{v.real:+.2e}{v.imag:+.2e}i" for v in res[i]), flush=True)
np.save(f"scanM1_{xi1}.npy", res)
print("w/w_inf columns:", wfs)
