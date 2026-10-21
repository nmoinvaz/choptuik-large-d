"""Poincare-Miranda dry run (NOT rigorous) for the M = 1 inner zero, with the double-precision lateral solver.
G(A, wf) = R_1 xi1^{-A^2} (lateral_np normalization); zero at A ~ 2.54215, wf ~ 0.77699.
Preconditioned H(p) = J0^{-1} G(p* + p) on the box |p_i| <= h; reports the edge margins the rigorous enclosures must beat."""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inner_stokes"))
from lateral_np import jump, Fharm, W0  # noqa: E402

XI1 = 0.2


def G(A, wf):
    R, JD, JS, err = jump(1, W0 * wf, Fharm(A), r1=XI1)
    z = R[1] * XI1 ** (-A * A)
    return np.array([z.real, z.imag])


def jac(p, h=1e-6):
    g0 = G(*p)
    return np.array([(G(p[0] + h, p[1]) - g0) / h, (G(p[0], p[1] + h) - g0) / h]).T, g0


def main():
    t0 = time.time()
    p = np.array([2.54215, 0.77699])
    for it in range(6):
        J, g0 = jac(p)
        step = -np.linalg.solve(J, g0)
        p = p + step
        print(f"newton {it}: p = {p[0]:.12f}, {p[1]:.12f}  |G| {np.abs(g0).max():.2e}  [{time.time() - t0:.0f} s]", flush=True)
        if np.abs(step).max() < 1e-12:
            break
    J0, g0 = jac(p)
    sv = np.linalg.svd(J0, compute_uv=False)
    print(f"zero: A = {p[0]:.10f}, w/w_inf = {p[1]:.10f} (w = {p[1] * W0:.8f}); J0 singular values {sv[0]:.3e}, {sv[1]:.3e}")
    Ji = np.linalg.inv(J0)
    for h in (1e-3, 1e-5):
        npts = 9
        ts = np.linspace(-h, h, npts)
        marg = []
        for i, sgn in ((0, -1), (0, 1), (1, -1), (1, 1)):
            vals = []
            for t in ts:
                q = np.array([t, t])
                q[i] = sgn * h
                H = Ji @ G(*(p + q))
                vals.append(sgn * H[i])
            marg.append(min(vals))
        # what |G| uncertainty is allowed: |J0^{-1}| maps enclosure radius r of G to <= |J0^{-1}|_inf r
        ninv = np.abs(Ji).sum(axis=1).max()
        print(f"box h = {h:.0e}: min edge margins (should all be > 0) {np.round(marg, 9).tolist()}; "
              f"allowed enclosure radius of G ~ {min(marg) / ninv:.2e}  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
