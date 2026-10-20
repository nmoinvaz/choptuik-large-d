"""The zero of the one-harmonic (M = 1) inner Stokes jump, F = A sin(theta): solves J_1 = 0 for (A, omega) at several
matching radii xi1 (found: A = 2.54215, omega = 0.77699 omega_inf, the same at xi1 = 0.2-0.275).
    python3 zero_M1.py"""
import numpy as np
from scipy.optimize import root
from lateral_np import jump, Fharm, W0
def F(v, xi1):
    A, wf = v
    R, JD, JS, err = jump(1, W0 * wf, Fharm(A), r1=xi1)
    z = R[1] * xi1 ** (-A * A)
    return [z.real, z.imag]
for xi1 in (0.2, 0.225, 0.275):
    sol = root(F, [2.54, 0.777], args=(xi1,), method="lm", options={"xtol": 1e-10})
    print(f"xi1 {xi1}: A {sol.x[0]:.5f} w/w_inf {sol.x[1]:.5f} |fun| {np.abs(sol.fun).max():.1e} {sol.success}", flush=True)
