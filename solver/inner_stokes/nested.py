"""Well-conditioned M = 3 selection: nested solve.

The de-mixed residual's harmonic-1 component carries a leakage of the harmonic-3 condition about 1e4 times larger
than r_3 itself (|K'_3/K'_1| ~ e^{pi w} times the finite-xi de-mixing error), so the joint 4 x 4 solve has condition
~1e10 and stalls on a valley of false near-zeros where a small S_3 cancels S_1 in r_1. Here instead:
  inner: at fixed (A, w) solve S_3 = 0 for F_3 to |r_3| ~ 1e-12 (well conditioned, 2 unknowns), so the leakage is
         ~1e-8 and r_1 measures S_1 alone;
  outer: Newton (finite differences of inner solves) on (A, w/w_inf) for r_1 = 0.
Run at several xi1: a genuine zero must not move.

    python3 -u nested.py A wf [xi1]
"""
import sys
import time
import numpy as np
from scipy.optimize import least_squares
import inner_np as I

T0 = time.time()


def inner(A, wf, F3, r1):
    """F3: complex harmonic coefficient guess. Returns (F3, r_1 complex, |r_3|)."""
    def res(v):
        x = np.array([A, wf, v[0], v[1]])
        return I.residual_demixed(x, 3, r1=r1)[2:]
    v0 = np.array([-2 * F3.imag, 2 * F3.real])
    sol = least_squares(res, v0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=60)
    x = np.array([A, wf, sol.x[0], sol.x[1]])
    r = I.residual_demixed(x, 3, r1=r1)
    F3n = sol.x[0] / 2j + sol.x[1] / 2
    return F3n, r[0] + 1j * r[1], np.abs(r[2:]).max()


def main():
    A, wf = float(sys.argv[1]), float(sys.argv[2])
    r1 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    F3 = (A / 2j) * (0.0532 + 0.0041j)
    for it in range(8):
        F3, R, r3 = inner(A, wf, F3, r1)
        print(f"it {it}: A {A:.7f} w/w_inf {wf:.7f}  r_1 {R:.4e}  |r_3| {r3:.1e}  F3/F1 {F3 / (A / 2j):.6f}  "
              f"[{time.time() - T0:.0f} s]", flush=True)
        if abs(R) < 1e-6:
            break
        hA, hw = 1e-4, 1e-4
        _, RA, _ = inner(A + hA, wf, F3, r1)
        _, Rw, _ = inner(A, wf + hw, F3, r1)
        Jm = np.array([[(RA - R).real / hA, (Rw - R).real / hw], [(RA - R).imag / hA, (Rw - R).imag / hw]])
        sv = np.linalg.svd(Jm, compute_uv=False)
        step = -np.linalg.solve(Jm, [R.real, R.imag])
        print(f"      2x2 Jacobian singular values {sv[0]:.3e} {sv[1]:.3e};  step dA {step[0]:+.2e} dwf {step[1]:+.2e}", flush=True)
        A, wf = A + step[0], wf + step[1]
    w = wf * I.W0
    print(f"RESULT xi1 {r1}: A {A:.6f}  w/w_inf {wf:.6f}  (w {w:.5f}, Delta {2 * np.pi / w:.5f})  F3/F1 {F3 / (A / 2j):.6f}  "
          f"|F3/F1| {abs(F3 / (A / 2j)):.5f}  <F^2> {A * A / 2 + 2 * abs(F3) ** 2:.4f}  r_1 {abs(R):.1e}", flush=True)


if __name__ == "__main__":
    main()
