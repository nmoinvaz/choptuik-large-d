"""Free large-D fits of gamma(D) and lambda = 1/gamma on the f64 points: is gamma_inf = 1/2 forced by the data?
Weighted Chebyshev polynomial fits in x = 1/n (n = D - 2) on Dmin <= D <= 40.35, limit = value at x = 0.
    python3 g_fits.py"""
import numpy as np
from numpy.polynomial import chebyshev as C
import os
d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gamma_points.npz")); D, g, s = d["D"], d["g"], d["s"]
s = np.maximum(s, 1e-9)
def fit(y, w, x, deg, c0=None):
    lo, hi = 0.0, x.max()
    t = 2 * (x - lo) / (hi - lo) - 1
    V = C.chebvander(t, deg)
    t0 = C.chebvander(np.array([-1.0]), deg)[0]
    if c0 is None:
        c, *_ = np.linalg.lstsq(V * w[:, None], y * w, rcond=None)
    else:   # constrain value at x = 0: eliminate via Lagrange (augment with a heavy row)
        A = np.vstack([V * w[:, None], 1e9 * t0]); b = np.r_[y * w, 1e9 * c0]
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = (V @ c - y) * w
    return t0 @ c, (r @ r) / max(1, len(y) - deg - 1), c, (lo, hi)
for Dmin in (6, 8, 10, 12):
    m = (D >= Dmin)
    x = 1 / (D[m] - 2); y = g[m]; w = 1 / s[m]
    print(f"Dmin {Dmin} ({m.sum()} pts):")
    for deg in range(4, 12):
        if m.sum() - deg < 6: break
        L, chi, c, _ = fit(y, w, x, deg)
        L2, chi2, _, _ = fit(y, w, x, deg, c0=0.5)
        Ll, chil, _, _ = fit(1 / y, w * y * y, x, deg)
        print(f"   deg {deg:2d}: gamma_inf free {L:.7f} (chi2/dof {chi:7.2f})  forced 1/2 chi2/dof {chi2:7.2f}   lambda_inf free {Ll:.6f}")
