"""One formula for all D >= 4: rational functions of x = 1/n = 1/(D - 2), fitted on every computed point 4 <= D <= 30.2
(data_all.load_all: kc 39 + Richardson below D = 8, step-converged above), checked on the held-out points D = 34-40.

    Delta(D) = P7(x) / Q2(x),          10 coefficients
    gamma(D) = P7(x) / Q2(x), P7(0) = 1/2,  9 coefficients

Found by rational_fit.compact / compact_gamma as the smallest forms that reproduce every point within its precision.

    python3 ../formula_all.py            (from solver/data)
    from formula_all import delta, gamma
"""
import numpy as np

from rational_fit import data, fit_rational, fit_rational_p0, gamma_data, rat

FIT_MIN, FIT_MAX = 4.0, 30.2


def fit_delta():
    D, y, s, _ = data()
    m = (D >= FIT_MIN) & (D <= FIT_MAX)
    p, q, chi = fit_rational(1 / (D[m] - 2), y[m], 1 / np.maximum(s[m], 1e-7), 7, 2)
    return p, q


def fit_gamma():
    D, g, s = gamma_data()
    m = (D >= FIT_MIN) & (D <= FIT_MAX)
    p, q, chi = fit_rational_p0(1 / (D[m] - 2), g[m], 1 / np.maximum(s[m], 1e-8), 7, 2, 0.5)
    return p, q


_PD, _QD = None, None
_PG, _QG = None, None


def delta(D):
    global _PD, _QD
    if _PD is None:
        _PD, _QD = fit_delta()
    return rat(_PD, _QD, 1 / (np.asarray(D, float) - 2))


def gamma(D):
    global _PG, _QG
    if _PG is None:
        _PG, _QG = fit_gamma()
    return rat(_PG, _QG, 1 / (np.asarray(D, float) - 2))


def main():
    pd, qd = fit_delta()
    pg, qg = fit_gamma()
    print("Delta: p =", " ".join(f"{v:+.10e}" for v in pd))
    print("       q = 1", " ".join(f"{v:+.10e}" for v in qd))
    print("gamma: p =", " ".join(f"{v:+.10e}" for v in pg))
    print("       q = 1", " ".join(f"{v:+.10e}" for v in qg))
    print("poles (D):", [round(2 + 1 / r.real, 3) for r in np.roots(np.r_[1, qd][::-1])],
          [round(2 + 1 / r.real, 3) for r in np.roots(np.r_[1, qg][::-1])])
    D, y, s, _ = data()
    Dg, g, sg = gamma_data()
    print("\n| D range | Delta: max |formula - computed| | precision | gamma: max |formula - computed| | precision |")
    for lo, hi in ((4, 4.3), (4.3, 5), (5, 10), (10, 20), (20, 30.25), (30.25, 36), (36, 41)):
        m, mg = (D >= lo) & (D < hi), (Dg >= lo) & (Dg < hi)
        ed = np.abs(rat(pd, qd, 1 / (D[m] - 2)) - y[m]).max()
        eg = np.abs(rat(pg, qg, 1 / (Dg[mg] - 2)) - g[mg]).max() if mg.any() else np.nan
        print(f"| {lo}-{hi} | {ed:.1e} | {s[m].max():.0e} | {eg:.1e} | {sg[mg].max() if mg.any() else np.nan:.0e} |")
    print("\nvalues:")
    for d in (4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 40, 50, 100, 1000):
        print(f"| {d} | {rat(pd, qd, 1 / (d - 2)):.8f} | {rat(pg, qg, 1 / (d - 2)):.8f} |")
    print(f"| inf | {pd[0]:.6f} | 0.5 |")


if __name__ == "__main__":
    main()
