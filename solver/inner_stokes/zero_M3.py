"""Zeros of the two-harmonic (M = 3) inner Stokes jump: unknowns A, B3, C3, w for F = A sin t + B3 sin 3t + C3 cos 3t;
conditions J_1(xi1) = J_3(xi1) = 0 (scaled by xi1^{-2<F^2>} and by the linear multiplier sizes).
    python3 zero_M3.py xi1 A B3 C3 wf"""
import sys
import numpy as np
from scipy.optimize import least_squares
from lateral_np import jump, Fharm, W0, Kp


def resid(v, xi1):
    A, B3, C3, wf = v
    w = W0 * wf
    Fh = Fharm(A, B3, C3)
    R, JD, JS, err = jump(3, w, Fh, r1=xi1)
    F2 = A * A / 2 + (B3 * B3 + C3 * C3) / 2
    sc = xi1 ** (-2 * F2) * np.exp(1 / (2 * xi1 * xi1))
    z1 = JD[1] * sc / abs(2j * np.pi * Kp(1j * w) * 2 * Fh[1])
    z3 = JD[3] * sc / abs(2j * np.pi * Kp(3j * w))
    out = [z1.real, z1.imag, z3.real, z3.imag]
    print("   ", np.round(v, 6), " |r| %.2e" % np.abs(out).max(), flush=True)
    return out


if __name__ == "__main__":
    xi1 = float(sys.argv[1]); x0 = [float(a) for a in sys.argv[2:6]]
    sol = least_squares(resid, x0, args=(xi1,), method="lm", xtol=1e-10, ftol=1e-12, max_nfev=60)
    A, B3, C3, wf = sol.x
    print(f"xi1 {xi1}: A {A:.6f} B3 {B3:.6f} C3 {C3:.6f} w/w_inf {wf:.6f} (w {wf * W0:.5f}, Delta {2 * np.pi / (wf * W0):.5f}) "
          f"F3/F1 {np.hypot(B3, C3) / A:.4f}  <F^2> {(A * A + B3 * B3 + C3 * C3) / 2:.4f}  max|resid| {np.abs(sol.fun).max():.1e} {sol.status}")
