"""Look for the coefficient rule ("factory") of Delta(D) = sum_k c_k x^k, and for closed forms it would sum to.

1. Weighted polynomial fits in x (x = 1/n or 1/(D-1)) of increasing degree: coefficients with 1-sigma errors,
   ratios c_k / c_{k-1} (Domb-Sykes: if c_k ~ A r^-k k^(a-1), the ratio -> (1/r)(1 + (a-1)/k), so the plot against
   1/k is a straight line; its intercept gives the radius r, its slope the exponent a, and the series sums to
   A (1 - x/r)^-a + ...).
2. Closed-form families fitted directly (few parameters), with their Delta_inf.
Data: best(Dmin, Dmax) = old precise data below D = 17.2 plus the trusted NE setups above (x_K <= 0.6, not 0.7).
"""
import numpy as np
from scipy.optimize import least_squares
from data_ne import load


def best(Dmin=8.0, Dmax=99.0, err_lo=1e-8):
    rows = {}
    for k, r in load().items():                      # x_K = 0.6, kc 23 plain (old below 17.2, NE above)
        if Dmin <= k <= min(Dmax, 25.5):
            rows[k] = (r["Delta"], err_lo if k < 17.2 else 3e-8)
    for setup in (dict(xK=0.5), dict(xK=0.6, kc=31, smooth=True), dict(xK=0.4)):
        try:
            for k, r in load(**setup).items():
                if r["src"] == "ne" and 25.5 < k <= Dmax and k >= Dmin:
                    rows.setdefault(k, (r["Delta"], 1e-6))
        except Exception:
            pass
    D = np.array(sorted(rows)); y = np.array([rows[k][0] for k in D]); s = np.array([rows[k][1] for k in D])
    return D, y, s


def polyfit_w(x, y, s, m):
    A = np.vander(x, m + 1, increasing=True) / s[:, None]
    c, *_ = np.linalg.lstsq(A, y / s, rcond=None)
    cov = np.linalg.pinv(A.T @ A)
    chi2 = np.sum((A @ c - y / s) ** 2) / max(len(y) - m - 1, 1)
    return c, np.sqrt(np.diag(cov) * max(chi2, 1.0)), chi2


if __name__ == "__main__":
    import sys
    Dmax = float(sys.argv[1]) if len(sys.argv) > 1 else 99
    D, y, s = best(8.0, Dmax)
    n = D - 2
    print(f"{len(D)} points, D = {D.min():.2f}..{D.max():.2f}")
    for name, x in (("1/n", 1 / n), ("1/(D-1)", 1 / (D - 1))):
        print(f"\nx = {name}")
        for m in range(5, 12):
            c, e, chi2 = polyfit_w(x, y, s, m)
            print(f"  m={m:2d} chi2/dof {chi2:8.1e}  c0 = {c[0]:.6f} +- {e[0]:.0e}   c1..: " +
                  " ".join(f"{v:+.4g}" for v in c[1:7]) + "   ratios: " + " ".join(f"{c[k] / c[k - 1]:+.2f}" for k in range(2, min(m, 7) + 1)))
    print("\nclosed forms (fit D >= 8):")
    fams = {
        "L + a/(n+b)": (lambda p, n: p[0] + p[1] / (n + p[2]), [1.6, 10, 1]),
        "L + a (n+b)^-p": (lambda p, n: p[0] + p[1] * (n + p[2]) ** -p[3], [1.6, 10, 1, 1]),
        "L + a/(n+b) + c/(n+b)^2": (lambda p, n: p[0] + p[1] / (n + p[2]) + p[3] / (n + p[2]) ** 2, [1.6, 10, 1, 0]),
        "L + a/n + b/n^1.5 + c/n^2": (lambda p, n: p[0] + p[1] / n + p[2] * n ** -1.5 + p[3] / n ** 2, [1.6, 10, -10, 5]),
        "L*(1 + a/(n+b))^p": (lambda p, n: p[0] * (1 + p[1] / (n + p[2])) ** p[3], [1.6, 5, 1, 1]),
        "sqrt(L^2 + a/(n+b))": (lambda p, n: np.sqrt(p[0] ** 2 + p[1] / (n + p[2])), [1.6, 30, 1]),
    }
    for name, (f, p0) in fams.items():
        for Dlo in (8, 12):
            m = D >= Dlo
            r = least_squares(lambda p: (f(p, n[m]) - y[m]) / s[m], p0, max_nfev=20000)
            res = (f(r.x, n[m]) - y[m])
            print(f"  {name:28s} D>={Dlo:2d}: L = {r.x[0]:.5f}  rms {np.sqrt(np.mean(res ** 2)):.0e}  max {np.abs(res).max():.0e}  params " +
                  " ".join(f"{v:+.4g}" for v in r.x[1:]))
