"""The fitted formulas for Delta(D) and gamma(D): coefficients, accuracy, and held-out check (the numbers in README.md).

    python3 ../formula.py            (from solver/data)

Delta(D) = Delta_inf + sum_k c_k / D^k  (degree 10, weighted, fitted on 10 <= D <= 30.2, final_fit.step_converged)
gamma(D) = 1/2       + sum_k g_k / D^k  (degree 10, gamma_inf fixed to 1/2, same points)
"""
import numpy as np

from final_fit import step_converged, weighted_series

FIT_MAX = 30.2


def fits():
    D, y, s, g = step_converged()
    m = (D >= 10) & (D <= FIT_MAX)
    cD, chiD = weighted_series(1 / D[m], y[m], 1 / s[m], 10)
    mg = m & np.isfinite(g)
    cG, chiG = weighted_series(1 / D[mg], g[mg], np.ones(mg.sum()), 10, c0=0.5)
    return D, y, s, g, cD, chiD, cG


def main():
    D, y, s, g, cD, chiD, cG = fits()
    fD = lambda d: np.polyval(cD[::-1], 1 / np.asarray(d, float))
    fG = lambda d: np.polyval(cG[::-1], 1 / np.asarray(d, float))
    m = (D >= 10) & (D <= FIT_MAX)
    print(f"Delta: chi2/dof {chiD:.2f}, max |fit - data| on 10..{FIT_MAX}: {np.abs(fD(D[m]) - y[m]).max():.1e}")
    h = D > FIT_MAX
    if h.any():
        print(f"  held out D = {D[h].min():.2f}..{D[h].max():.2f}: max |series - data| {np.abs(fD(D[h]) - y[h]).max():.1e}")
    mg = m & np.isfinite(g)
    print(f"gamma: max |fit - data| on 10..{FIT_MAX}: {np.abs(fG(D[mg]) - g[mg]).max():.1e}")
    print("\n| k | c_k (Delta) | g_k (gamma) |\n|---|---|---|")
    for k in range(11):
        print(f"| {k} | {cD[k]:+.10e} | {cG[k]:+.10e} |")
    for nterm in (2, 3, 4, 5):
        err = [np.abs(np.polyval(cD[:nterm][::-1], 1 / D[(D >= lo) & m]) - y[(D >= lo) & m]).max() for lo in (10, 20)]
        errg = [np.abs(np.polyval(cG[:nterm][::-1], 1 / D[(D >= lo) & mg]) - g[(D >= lo) & mg]).max() for lo in (10, 20)]
        print(f"first {nterm} terms: Delta max err D>=10 {err[0]:.1e}, D>=20 {err[1]:.1e}; gamma {errg[0]:.1e}, {errg[1]:.1e}")


if __name__ == "__main__":
    main()
