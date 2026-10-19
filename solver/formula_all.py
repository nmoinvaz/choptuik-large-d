"""One formula for all D >= 4: rational functions of x = 1/n = 1/(D - 2), fitted on every computed point 4 <= D <= 30.2
(data_all.load_all: kc 39 + Richardson below D = 8, step-converged above), checked on the held-out points D = 34-40.

    Delta(D) = P8(x) / Q2(x),          11 coefficients
    gamma(D) = P7(x) / Q2(x), P7(0) = 1/2,  9 coefficients

rational_fit.compact / compact_gamma find the smallest forms within the point precision (Delta: P7/Q2, 10 numbers, to
~2e-7); one more numerator term makes Delta ten times tighter (<= 2e-8 on 4-30), so that is the one used.

    python3 ../formula_all.py            (from solver/data)
    from formula_all import delta, gamma
"""
import numpy as np

from rational_fit import data, fit_rational, fit_rational_p0, gamma_data, rat

FIT_MIN, FIT_MAX = 4.0, 30.2


def fit_delta():
    D, y, s, _ = data()
    m = (D >= FIT_MIN) & (D <= FIT_MAX)
    p, q, chi = fit_rational(1 / (D[m] - 2), y[m], 1 / np.maximum(s[m], 1e-7), 8, 2)   # P7/Q2: 2e-7, P8/Q2: 2e-8
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


def _tex(v):
    if v == 0.5:
        return "$+0.5$ (fixed)"
    e = int(np.floor(np.log10(abs(v))))
    if e in (0,):
        return f"${v:+.10f}$"
    return f"${v / 10 ** e:+.10f} \\times 10^{{{e}}}$"


def readme_tables():
    """The README coefficient table and values table for the current fit (markdown)."""
    pd, qd = fit_delta()
    pg, qg = fit_gamma()
    rows = ["| $k$ | $a_k$ ($\\Delta$, numerator) | $b_k$ ($\\Delta$, denominator) | $e_k$ ($\\gamma$, numerator) | "
            "$f_k$ ($\\gamma$, denominator) |", "|---|---|---|---|---|"]
    for k in range(max(len(pd), len(pg))):
        a = _tex(pd[k]) if k < len(pd) else ""
        b = "$1$" if k == 0 else (_tex(qd[k - 1]) if k - 1 < len(qd) else "")
        e = _tex(pg[k]) if k < len(pg) else ""
        f = "$1$" if k == 0 else (_tex(qg[k - 1]) if k - 1 < len(qg) else "")
        rows.append(f"| {k} | {a} | {b} | {e} | {f} |")
    print("\n".join(rows))
    print()
    vals = ["| $D$ | $\\Delta$ | $\\gamma$ |", "|---|---|---|"]
    for d, nd in ((4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (10, 7), (12, 7), (15, 7), (20, 7), (30, 7), (40, 6), (50, 5),
                  (100, 5), (1000, 4)):
        vals.append(f"| {d} | {rat(pd, qd, 1 / (d - 2)):.{nd}f} | {rat(pg, qg, 1 / (d - 2)):.{nd}f} |")
    vals.append("| $\\infty$ | $1.64395 \\pm 0.00008$ | $1/2$ |")
    print("\n".join(vals))
    print("\nlimit a0 =", f"{pd[0]:.6f}", " poles:", [round(2 + 1 / r.real, 2) for r in np.roots(np.r_[1, qd][::-1])],
          [round(2 + 1 / r.real, 2) for r in np.roots(np.r_[1, qg][::-1])],
          " leading: Delta", f"{pd[1] - pd[0] * qd[0]:.3f}/n, gamma", f"{pg[1] - qg[0] / 2:.4f}/n")
