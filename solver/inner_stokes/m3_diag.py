"""Conditioning of the M = 3 de-mixed residual: Jacobian singular values at two near-zeros, and the leakage of the
harmonic-3 condition into the harmonic-1 one.   python3 -u m3_diag.py"""
import numpy as np
import inner_np as I


def xfrom(A, wf, ratio):
    z = (A / 2j) * ratio
    return np.array([A, wf, -2 * z.imag, 2 * z.real])


pts = {"agent A (xi1 0.2)": xfrom(2.72122, 1.01344, 0.052992 + 0.004082j),
       "coordinator (xi1 0.2)": xfrom(2.708217, 1.000219, 0.053766 + 0.004183j)}
for name, x in pts.items():
    r0 = I.residual_demixed(x, 3)
    J = np.zeros((4, 4))
    for i in range(4):
        h = 1e-7 * max(1, abs(x[i]))
        xp = x.copy(); xp[i] += h
        J[:, i] = (I.residual_demixed(xp, 3) - r0) / h
    U, s, Vt = np.linalg.svd(J)
    print(f"{name}: x = {np.round(x, 8)}  r = {np.round(r0, 3)}")
    print(f"   singular values {s}  (condition {s[0] / s[-1]:.1e})")
    print(f"   smallest right singular vector (dA, dwf, dB3, dC3) = {np.round(Vt[-1], 6)}")
    print(f"   largest right singular vector                     = {np.round(Vt[0], 6)}")
    print(f"   |dr_1/dB3|, |dr_1/dC3| = {np.hypot(J[0, 2], J[1, 2]):.3e}, {np.hypot(J[0, 3], J[1, 3]):.3e};  "
          f"|dr_3/dB3|, |dr_3/dC3| = {np.hypot(J[2, 2], J[3, 2]):.3e}, {np.hypot(J[2, 3], J[3, 3]):.3e};  "
          f"|dr_1/dA|, |dr_1/dwf| = {np.hypot(J[0, 0], J[1, 0]):.3e}, {np.hypot(J[0, 1], J[1, 1]):.3e}", flush=True)
