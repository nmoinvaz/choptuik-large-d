"""Delta_inf from all step-converged data: x_K 0.6 RK x2 (Richardson with RK x1 where available, junk-solved above
17.2) up to D = 30.2, plus x_K 0.7 RK x4 junk-solved above that. Weighted fits in x / max(x) for x = 1/D, 1/(D-1),
1/(D-2), several degrees and lower cuts; candidate limits forced for comparison.

    python3 ../final_fit.py            (from solver/data)
    from final_fit import step_converged
"""
import glob
import json

import numpy as np

from data_best import load as old_load


def step_converged(Dmin=8.0, hi_from=30.2):
    """(D, Delta, sigma, gamma) arrays: x_K 0.6 RK x2 (+ Richardson) up to hi_from, x_K 0.7 RK x4 junk-solved above."""
    o = old_load()
    rk1 = {k: r["Delta"] for k, r in o.items() if k < 17.2}
    rk2, hi, gam = {}, {}, {}
    for f in sorted(glob.glob("scan_ne*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if not r.get("ok") or r.get("kc", 23) != 23 or r.get("smooth"):
                continue
            k = round(r["D"], 4)
            xk = r.get("xK", 0.6)
            rk = r.get("rk", 1)
            if abs(xk - 0.6) < 1e-9:
                if rk == 2:
                    rk2[k] = r["Delta"]
                    gam[k] = r.get("gamma")
                elif rk == 1 and (r.get("js") or (17.2 <= k <= 23.4)):
                    if r.get("js") or k not in rk1:
                        rk1[k] = r["Delta"]
            elif abs(xk - 0.7) < 1e-9 and rk >= 4 and r.get("js") and k > hi_from:
                hi[k] = r["Delta"]
                gam[k] = r.get("gamma")
    D, y, s, g = [], [], [], []
    for k in sorted(rk2):
        if k < Dmin or k > hi_from:
            continue
        D.append(k)
        y.append(rk2[k] + ((rk2[k] - rk1[k]) / 15 if k in rk1 else 0.0))
        s.append(3e-11 if k < 22 else 5e-10 if k < 26 else 1e-9 if k < 29.5 else 3e-8)
        g.append(gam.get(k))
    for k in sorted(hi):
        D.append(k)
        y.append(hi[k])
        s.append(1e-7)
        g.append(gam.get(k))
    return np.array(D), np.array(y), np.array(s), np.array([np.nan if v is None else v for v in g], float)


def weighted_series(x, y, w, m, c0=None):
    """Weighted polynomial fit in x / max(x); returns coefficients of x and chi2/dof."""
    sc = x.max()
    A = np.vander(x / sc, m + 1, increasing=True)
    if c0 is None:
        c, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        chi = np.sum(((A @ c - y) * w) ** 2) / (len(y) - m - 1)
    else:
        c, *_ = np.linalg.lstsq(A[:, 1:] * w[:, None], (y - c0) * w, rcond=None)
        chi = np.sum(((A[:, 1:] @ c - (y - c0)) * w) ** 2) / (len(y) - m)
        c = np.r_[c0, c]
    return c / sc ** np.arange(m + 1), chi


if __name__ == "__main__":
    D, y, s, _ = step_converged()
    print(f"{len(D)} points: D {D.min():.2f}..{D.max():.2f} ({sum(D > 30.2)} x_K 0.7 RK x4 above 30.2)")
    cands = {"pi^2/6": np.pi ** 2 / 6, "1.6438": 1.6438, "1.6440": 1.6440, "1.6442": 1.6442}
    for lo in (8, 10, 12):
        m_ = D >= lo
        for xn, xf in (("1/D", lambda d: 1 / d), ("1/(D-1)", lambda d: 1 / (d - 1)), ("1/(D-2)", lambda d: 1 / (d - 2))):
            x = xf(D[m_])
            w = 1 / s[m_]
            row = [f"{weighted_series(x, y[m_], w, m)[0][0]:.6f}({weighted_series(x, y[m_], w, m)[1]:.1f})" for m in (8, 9, 10, 11)]
            cc = [f"{name}:{weighted_series(x, y[m_], w, 10, L)[1]:.1f}" for name, L in cands.items()]
            print(f"D>={lo:2d} {xn:8s} free c0(chi2/dof) m=8..11: " + " ".join(row) + "  | forced m=10 chi2/dof: " + " ".join(cc))
