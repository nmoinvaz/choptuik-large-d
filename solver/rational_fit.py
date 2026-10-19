"""Rational (Pade-type) fits Delta(D) = P_m(x) / Q_k(x), x = 1/D (or 1/(D-1), 1/(D-2)), Q(0) = 1, on all D >= 4.

The degree-10 power series has coefficients growing ~9x per order with alternating signs (c3/c2 .. c5/c4 ~ -8.4 .. -9.4),
i.e. a singularity near 1/D ~ -0.11 (D ~ -9): a pole that a rational form can carry with few parameters, which may
also cover 4 <= D < 10 where the power series fails.

    python3 ../rational_fit.py            (from solver/data)
"""
import numpy as np
from scipy.optimize import least_squares

from data_best import load as old_load
from final_fit import step_converged

FIT_MAX = 30.2


def data():
    """(D, Delta, sigma, gamma) over D = 3.7 .. 40: data_all.load_all (kc 39 + Richardson below D = 8)."""
    from data_all import load_all
    return load_all()


def data_old():
    """(D, Delta, sigma, gamma): old single-setup values for 3.69 <= D < 8 (scatter ~5e-7: D = 6.75-7.5 off by up to
    3.8e-6, D = 4.1-4.75 at kc 31, under-resolved by up to 8e-6), step-converged above."""
    D, y, s, g = step_converged(Dmin=8.0)
    o = old_load()
    lo = np.array(sorted(d for d in o if 3.69 <= d < 8.0))
    return (np.r_[lo, D], np.r_[[o[d]["Delta"] for d in lo], y], np.r_[np.full(len(lo), 3e-11), s],
            np.r_[[o[d].get("gamma") if o[d].get("gamma") is not None else np.nan for d in lo], g])


def rat(p, q, x):
    return np.polyval(p[::-1], x) / np.polyval(np.r_[1.0, q][::-1], x)


def fit_rational(x, y, w, m, k):
    """Weighted fit of P_m / Q_k (Q(0) = 1) in t = x / max(x): linearised start (P - y Q = 0), then nonlinear LS."""
    sc = x.max()
    t = x / sc
    A = np.hstack([np.vander(t, m + 1, increasing=True), -(y[:, None] * np.vander(t, k + 1, increasing=True)[:, 1:])])
    c0, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    f = lambda c: (rat(c[:m + 1], c[m + 1:], t) - y) * w
    r = least_squares(f, c0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
    c = r.x
    chi = np.sum(r.fun ** 2) / (len(y) - m - k - 1)
    p, q = c[:m + 1] / sc ** np.arange(m + 1), c[m + 1:] / sc ** np.arange(1, k + 1)
    return p, q, chi


def main():
    D, y, s, g = data()
    fit, hold = D <= FIT_MAX, D > FIT_MAX
    print(f"{fit.sum()} fitted points D = {D[fit].min():.2f}..{FIT_MAX}, {hold.sum()} held out D = {D[hold].min():.2f}.."
          f"{D[hold].max():.2f}")
    rows = []
    for xn, xf in (("1/D", lambda d: 1 / d), ("1/(D-1)", lambda d: 1 / (d - 1)), ("1/(D-2)", lambda d: 1 / (d - 2))):
        for Dlo in (4.0, 8.0):
            m_ = fit & (D >= Dlo)
            for m in range(2, 9):
                for k in range(1, 7):
                    if m + k > 11:
                        continue
                    try:
                        p, q, chi = fit_rational(xf(D[m_]), y[m_], 1 / s[m_], m, k)
                    except Exception:
                        continue
                    poles = np.roots(np.r_[1.0, q][::-1])
                    real_bad = [1 / r.real for r in poles if abs(r.imag) < 1e-9 and r.real > 0]   # poles at physical D
                    ho = rat(p, q, xf(D[hold])) - y[hold]
                    # interior cross-check: refit without 22 < D <= 30.2 and predict it
                    cv = m_ & (D <= 22)
                    pc, qc, _ = fit_rational(xf(D[cv]), y[cv], 1 / s[cv], m, k)
                    tst = fit & (D > 22)
                    cv_err = np.abs(rat(pc, qc, xf(D[tst])) - y[tst]).max()
                    rows.append((chi, xn, Dlo, m, k, p[0], np.sqrt(np.mean(ho ** 2)), cv_err, real_bad))
    rows.sort(key=lambda r: (r[3] + r[4], r[0]))
    print(f"{'x':8s} {'D>=':>4s} {'m':>2s} {'k':>2s} {'params':>6s} {'chi2/dof':>9s} {'Delta_inf':>10s} "
          f"{'held-out rms':>12s} {'CV(22->30) max':>14s}  poles at D > 0")
    best = [r for r in rows if not r[8]]                         # no pole at physical D
    best = [r for i, r in enumerate(best) if sum(1 for q in best[:i] if q[3] + q[4] == r[3] + r[4]) < 4]
    for chi, xn, Dlo, m, k, d0, ho, cv, bad in best:
        print(f"{xn:8s} {Dlo:4.0f} {m:2d} {k:2d} {m + k + 1:6d} {chi:9.2f} {d0:10.6f} {ho:12.1e} {cv:14.1e}  "
              f"{', '.join(f'{b:.1f}' for b in bad) if bad else '-'}")


if __name__ == "__main__":
    main()


def compact(targets=(1e-5, 1e-6, 1e-7, 1e-8), Dmin=4.0, npar_max=13):
    """Smallest rational form that reproduces every computed Delta for D >= 4 to a given accuracy (sigma floored at the
    target), with its Delta_inf and held-out error. python3 -c 'import rational_fit as r; r.compact()'"""
    D, y, s, g = data()
    fit, hold = (D <= FIT_MAX) & (D >= Dmin), D > FIT_MAX
    for tgt in targets:
        best = None
        for npar in range(4, npar_max + 1):
            for xn, xf in (("1/D", lambda d: 1 / d), ("1/(D-1)", lambda d: 1 / (d - 1)), ("1/(D-2)", lambda d: 1 / (d - 2))):
                for m in range(1, npar - 1):
                    k = npar - 1 - m
                    try:
                        p, q, chi = fit_rational(xf(D[fit]), y[fit], 1 / np.maximum(s[fit], tgt), m, k)
                    except Exception:
                        continue
                    poles = np.roots(np.r_[1.0, q][::-1])
                    if any(abs(r.imag) < 1e-9 and r.real > 0 for r in poles):
                        continue
                    dev = np.abs(rat(p, q, xf(D[fit])) - y[fit])
                    err = dev.max()
                    if np.all(dev <= np.maximum(tgt, 2 * s[fit])) and (best is None or err < best[0]):
                        best = (err, xn, m, k, p, q, np.abs(rat(p, q, xf(D[hold])) - y[hold]).max())
            if best:
                break
        if best:
            err, xn, m, k, p, q, ho = best
            print(f"target {tgt:.0e}: {m + k + 1} parameters, P{m}/Q{k} in {xn}: max error {err:.1e} on D = {Dmin}..30.2, "
                  f"held-out max {ho:.1e}, Delta_inf {p[0]:.6f}")
            print("   p =", " ".join(f"{v:+.10e}" for v in p))
            print("   q = 1", " ".join(f"{v:+.10e}" for v in q))
        else:
            print(f"target {tgt:.0e}: none up to {npar_max} parameters")


def fit_rational_p0(x, y, w, m, k, p0):
    """As fit_rational with P(0) = p0 fixed (gamma: p0 = 1/2)."""
    sc = x.max()
    t = x / sc
    A = np.hstack([np.vander(t, m + 1, increasing=True)[:, 1:], -(y[:, None] * np.vander(t, k + 1, increasing=True)[:, 1:])])
    c0, *_ = np.linalg.lstsq(A * w[:, None], (y - p0) * w, rcond=None)
    f = lambda c: (rat(np.r_[p0, c[:m]], c[m:], t) - y) * w
    r = least_squares(f, c0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
    c = r.x
    chi = np.sum(r.fun ** 2) / (len(y) - m - k)
    p = np.r_[p0, c[:m]] / sc ** np.arange(m + 1)
    q = c[m:] / sc ** np.arange(1, k + 1)
    return p, q, chi


def gamma_data():
    """(D, gamma, sigma): kc 39 RK x2 below D = 8 (sigma from the kc 31 -> 39 step), step-converged points above
    (sigma 1e-9, x_K 0.7 points above 30.2: 1e-7)."""
    from data_all import _load
    D, y, s, g = data()
    k31, k39 = _load("scan_ne6_lowD_kc31_rk2.jsonl"), _load("scan_ne6_lowD_kc39_rk2.jsonl")
    sg = []
    for d in D:
        k = round(d, 4)
        if d < 8 and k in k31 and k in k39 and k31[k].get("gamma") and k39[k].get("gamma"):
            sg.append(max(1e-9, abs(k39[k]["gamma"] - k31[k]["gamma"]) / 13))
        else:
            sg.append(1e-9 if d <= FIT_MAX else 1e-7)
    m = np.isfinite(g)
    return D[m], g[m], np.array(sg)[m]


def compact_gamma(targets=(1e-6, 1e-7, 1e-8), Dmin=4.0, npar_max=11):
    """Smallest rational form with gamma_inf = 1/2 reproducing every gamma for D >= Dmin to a target accuracy."""
    D, g, s = gamma_data()
    fit, hold = (D <= FIT_MAX) & (D >= Dmin), D > FIT_MAX
    for tgt in targets:
        best = None
        for npar in range(2, npar_max + 1):
            for xn, xf in (("1/D", lambda d: 1 / d), ("1/(D-1)", lambda d: 1 / (d - 1)), ("1/(D-2)", lambda d: 1 / (d - 2))):
                for m in range(1, npar):
                    k = npar - m
                    try:
                        p, q, chi = fit_rational_p0(xf(D[fit]), g[fit], 1 / np.maximum(s[fit], tgt), m, k, 0.5)
                    except Exception:
                        continue
                    poles = np.roots(np.r_[1.0, q][::-1])
                    if any(abs(r.imag) < 1e-9 and r.real > 0 for r in poles):
                        continue
                    dev = np.abs(rat(p, q, xf(D[fit])) - g[fit])
                    if np.all(dev <= np.maximum(tgt, 2 * s[fit])) and (best is None or dev.max() < best[0]):
                        best = (dev.max(), xn, m, k, p, q, np.abs(rat(p, q, xf(D[hold])) - g[hold]).max() if hold.any() else np.nan)
            if best:
                break
        if best:
            err, xn, m, k, p, q, ho = best
            print(f"gamma target {tgt:.0e}: {m + k} free parameters, P{m}/Q{k} in {xn} (P(0) = 1/2): max error {err:.1e} on "
                  f"D = {Dmin}..30.2, held-out max {ho:.1e}")
            print("   p =", " ".join(f"{v:+.10e}" for v in p))
            print("   q = 1", " ".join(f"{v:+.10e}" for v in q))
        else:
            print(f"gamma target {tgt:.0e}: none up to {npar_max} parameters")
