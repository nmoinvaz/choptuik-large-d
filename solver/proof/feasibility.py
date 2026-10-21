#!/usr/bin/env python3
"""Feasibility numbers for a computer-assisted proof of the inner zero (PROOF.md, layer A). NOT itself rigorous.

1. Growth of the amplitude-series coefficients sigma_k of S_1/S_1^(1) in x = A^2 (scan25phys, K = 25): the ratios
   |sigma_k/sigma_{k+1}| grow linearly in k (about 0.60 k + 2.5), the signature of an entire function of order 1 in x.
   Printed: the fit, the size of the terms |sigma_k| x0^k at the zero x0 = A0^2, and the order K at which a direct
   summation at x0 would reach a given accuracy if the linear-ratio law continues (it is a model, not a bound).
2. The 2 x 2 outer Jacobian of the M = 5 harmonic-1 residual r_1(A, w) near its zero, from the hier.py logs (fixed-A runs
   at w/w_inf = 0.95, 1.00, 1.05 and the converged zero), and the enclosure accuracy a Krawczyk / interval-Newton test
   would need for a box of given radius: eps <= sigma_min * radius (to first order).

    python3 feasibility.py
"""
import glob
import json
import os
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
mp.mp.dps = 40


def growth():
    fn = [f for f in sorted(glob.glob(os.path.join(HERE, "..", "inner_amp", "scan25phys", "*.json")))
          if abs(json.load(open(f))["wf"] - 1) < 1e-9][0]
    r = json.load(open(fn))
    sig = [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in r["sigma_str"]]
    k = np.arange(3, len(sig) - 1)
    rat = np.array([float(abs(sig[i] / sig[i + 1])) for i in k])
    a, b = np.polyfit(k, rat, 1)
    print(f"|sigma_k/sigma_k+1|, k = 3..{len(sig) - 2}: {np.round(rat, 3)}")
    print(f"linear fit: ratio = {a:.4f} k + {b:.3f}  (increments {np.round(np.diff(rat), 3)})")
    x0 = 2.7196 ** 2
    terms = [float(abs(s)) * x0 ** i for i, s in enumerate(sig)]
    print(f"terms |sigma_k| x0^k at x0 = {x0:.3f}: {np.round(terms, 1)}")
    # extrapolate the terms with the linear-ratio model
    t, kk = terms[-1], len(sig) - 1
    for target in (1e-2, 1e-6, 1e-10):
        while t > target:
            t *= x0 / (a * kk + b)
            kk += 1
        print(f"  model: terms fall below {target:g} at k ~ {kk} (K = 2k + 1 ~ {2 * kk + 1})")


def krawczyk():
    # r_1 at A = 2.72 (complex), from logs/hier_M5_fixA_w0{95,100,105} and the converged M = 5 --schur zero
    r95, r100, r105 = 74.436 - 310.78j, 6.4841 + 1.8536j, 271.31 + 328.69j
    A0, w0 = 2.718579, 0.999228
    Jw = (r105 - r95) / 0.10                          # d r_1 / d(w/w_inf), central difference (h = 0.05)
    JA = (r100 - Jw * (1.0 - w0)) / (2.72 - A0)       # from r_1(zero) = 0 and the linear model
    J = np.array([[JA.real, Jw.real], [JA.imag, Jw.imag]])
    s = np.linalg.svd(J, compute_uv=False)
    curv = (r105 + r95 - 2 * r100) / 0.05 ** 2
    print(f"M = 5 outer Jacobian (Re/Im rows; columns A, w/w_inf):\n{np.round(J, 1)}")
    print(f"singular values {s[0]:.0f}, {s[1]:.0f} (condition {s[0] / s[1]:.2f}); |d2 r_1/dw2| ~ {abs(curv):.0f}")
    for rad in (1e-3, 1e-4, 1e-6):
        print(f"  box radius {rad:g} in (A, w/w_inf): enclosure error of r_1 must be below ~{s[1] * rad:.3g}"
              f" (curvature term {abs(curv) * rad ** 2:.2g})")
    print("  for comparison: double-precision floor of r_1 with --schur at M = 5 is ~8e-7; the M = 3 -> 5 change of the")
    print("  zero is 0.0164 in w/w_inf, i.e. a truncation effect of ~60 in r_1; M = 5 -> 7 is expected ~20x smaller.")


if __name__ == "__main__":
    growth()
    print()
    krawczyk()
