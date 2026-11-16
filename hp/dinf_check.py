#!/usr/bin/env python3
"""Two checks of Delta_inf that need no planted offset (#906): Bulirsch-Stoer extrapolation and Bayesian model
averaging of the fit grid. Both run on the double-double points (D <= 61.03, dd_fits_mp.load_points) and on the six
plants of planted_dinf.py, through the same code path, before the real data.

  bst   Bulirsch-Stoer tables (rational extrapolation in h, exponent omega = 1) on sub-sequences of the cleaned points:
        N points nearest to N values of t evenly spaced on [1/Dmax, 1/Dmin], for N in 8..16, Dmin in 10, 15, 20,
        Dmax 61.03, h = 1/D and 1/(D - 2). Each table's estimate is its last entry. The estimate is the median of all
        sub-sequences; the spread is half the range of the middle half of them (an estimator spread, not a bar).
        (R. Bulirsch, J. Stoer, Numer. Math. 6, 413 (1964); the form of M. Henkel, G. Schuetz, J. Phys. A 21, 2617
        (1988), as given by K. Krebs et al., cond-mat/9402019, section 3, Eqs. 3.1-3.3.)
  aic   Akaike-weighted average over the fits of dinf_budget.py's grid on D <= 60.1 and 61.03: model 1/D and 1/(D-2),
        degree 14-22, Dmin 10, 12, 15, cleaning on and off, 120 fits. Each fit is weighted least squares with the
        noise of planted_dinf.py (1e-16 for D <= 60.1, 7e-16 above). Its weight is exp(-AIC/2), with
        AIC = chi^2 + 2 k + 2 N_cut, k = degree + 1 and N_cut the points of D <= 61.03 the fit leaves out. The mean is
        sum w a, and the variance sum w s^2 + sum w a^2 - mean^2, s the fit's own standard error of its limit.
        (W. I. Jay, E. T. Neil, Phys. Rev. D 103, 114502 (2021), arXiv:2008.01069, Eqs. 10, 18 and the 2 N_cut term.)

Criteria (frozen in #906 before the first run). T1-T5 are graded and T6 (non-analytic at 1/D = 0) is reported only, as
in planted_dinf.py.
  bst  the median lies within 1.5e-8 of the planted limit for T1-T5.
  aic  the mean lies within 0.3e-8 of the planted limit with no offset, and within 2 standard deviations, for T1-T5.
A method that fails a graded plant is dropped. Then the real value is printed next to the README's
1.643893746 +- 1.5e-8, and a disagreement is reported, not fixed.

    python3 dinf_check.py bst|aic [--plants T1,T2,...] [--real 0|1] [--mutate MUT]

--mutate omega2 runs bst with omega = 2 (a wrong exponent for a series in 1/D) and must fail a graded plant;
--mutate noweight runs aic with equal weights and must fail a graded plant. Exit 0 when every graded plant passes.
"""
import os
import statistics
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dd_fits_mp as F  # noqa: E402
import dinf_budget as B  # noqa: E402
import planted_dinf as P  # noqa: E402

mp.mp.dps = 60
README = (mp.mpf("1.643893746"), mp.mpf("1.5e-8"))
BAR_BST = mp.mpf("1.5e-8")
BAR_AIC = mp.mpf("0.3e-8")


def bst(hs, ys, omega=1):
    """Bulirsch-Stoer table on (h_i, T_i), h decreasing toward 0; returns the last entry T_{N-1}^{(0)}"""
    n = len(ys)
    prev2 = [mp.mpf(0)] * (n + 1)                    # T_{m-2}, with T_{-1} = 0
    prev = list(ys)                                   # T_{m-1}
    for m in range(1, n):
        cur = []
        for i in range(n - m):
            d = prev[i + 1] - prev[i]
            den2 = prev[i + 1] - prev2[i + 1]
            ratio = (hs[i] / hs[i + m]) ** omega
            q = ratio * (1 - d / den2) - 1 if den2 != 0 else ratio - 1
            cur.append(prev[i + 1] + d / q)
        prev2, prev = prev, cur
    return prev[0]


def subsequence(Ds, N, dmin, dmax):
    """the N distinct points nearest to N values of t = 1/D evenly spaced on [1/dmax, 1/dmin], by increasing D"""
    avail = sorted(d for d in Ds if dmin <= d <= dmax)
    ts = [1 / mp.mpf(dmax) + (1 / mp.mpf(dmin) - 1 / mp.mpf(dmax)) * j / (N - 1) for j in range(N)]
    out = set()
    for t in ts:
        out.add(min(avail, key=lambda d: abs(1 / mp.mpf(d) - t)))
    return sorted(out)


def run_bst(pts, Ds, omega=1):
    est = []
    for model in ("1/D", "1/n"):
        for dmin in (10, 15, 20):
            for N in range(8, 17):
                sub = subsequence(Ds, N, dmin, 61.03)
                hs = [F.MODELS[model](d) for d in sub]
                est.append(bst(hs, [pts[d] for d in sub], omega))
    est.sort()
    med = est[len(est) // 2] if len(est) % 2 else (est[len(est) // 2 - 1] + est[len(est) // 2]) / 2
    q = len(est) // 4
    return med, (est[-q - 1] - est[q]) / 2, len(est)


def sigma(d):
    return mp.mpf("1e-16") if d <= 60.1 else mp.mpf("7e-16")


def fits_weighted(pts, Ds, model):
    """{deg: (limit, chi^2, standard error of the limit)} for dinf_budget's degrees, weighted least squares"""
    X, a, b = B.design(Ds, model)
    w = [1 / sigma(d) ** 2 for d in Ds]
    y = [pts[d] for d in Ds]
    N, K = len(Ds), B.DMAX + 1
    G = [[mp.fsum(w[i] * X[i, j] * X[i, k] for i in range(N)) for k in range(K)] for j in range(K)]
    r = [mp.fsum(w[i] * X[i, j] * y[i] for i in range(N)) for j in range(K)]
    t0 = F.cheb_rows([mp.mpf(0)], a, b, B.DMAX)
    out = {}
    for deg in B.DEGS:
        m = deg + 1
        Gm = mp.matrix([row[:m] for row in G[:m]])
        c = mp.lu_solve(Gm, mp.matrix(r[:m]))
        chi2 = mp.fsum(w[i] * (mp.fsum(X[i, j] * c[j] for j in range(m)) - y[i]) ** 2 for i in range(N))
        v = mp.lu_solve(Gm, mp.matrix([t0[0, j] for j in range(m)]))
        var = mp.fsum(t0[0, j] * v[j] for j in range(m))
        out[deg] = (mp.fsum(t0[0, j] * c[j] for j in range(m)), chi2, mp.sqrt(var))
    return out


def run_aic(pts, allD, equal=False):
    full = [d for d in allD if d <= 61.03]
    rows = []
    for dmax in (60.1, 61.03):
        for dmin in (10, 12, 15):
            base = [d for d in allD if dmin <= d <= dmax]
            for clean in (True, False):
                Ds = F.clean(pts, base, verbose=False) if clean else base
                for model in ("1/D", "1/n"):
                    for deg, (lim, chi2, se) in fits_weighted(pts, Ds, model).items():
                        rows.append((lim, chi2 + 2 * (deg + 1) + 2 * (len(full) - len(Ds)), se))
    amin = min(r[1] for r in rows)
    w = [mp.mpf(1) if equal else mp.exp(-(r[1] - amin) / 2) for r in rows]
    W = mp.fsum(w)
    w = [x / W for x in w]
    mean = mp.fsum(wi * r[0] for wi, r in zip(w, rows))
    var = mp.fsum(wi * r[2] ** 2 for wi, r in zip(w, rows)) + mp.fsum(wi * r[0] ** 2 for wi, r in zip(w, rows)) - mean ** 2
    top = max(w)
    return mean, mp.sqrt(max(var, 0)), len(rows), top


def main():
    args = sys.argv[1:]
    if not args or args[0] not in ("bst", "aic"):
        raise SystemExit(__doc__)
    what = args[0]
    opt = {"plants": ",".join(P.PLANTS), "real": "1", "mutate": ""}
    for i in range(1, len(args), 2):
        opt[args[i].lstrip("-")] = args[i + 1]
    omega = 2 if opt["mutate"] == "omega2" else 1
    equal = opt["mutate"] == "noweight"
    T0 = time.time()
    real, _ = F.load_points()
    allD = sorted(real)
    base = F.clean(real, [d for d in allD if 10 <= d <= 61.03], verbose=False)
    ev, res, _ = F.fit(real, base, "1/D", 18)
    L = F.limit(ev)
    print(f"{what}{' (mutation ' + opt['mutate'] + ')' if opt['mutate'] else ''}: planted limit P(0) = "
          f"{mp.nstr(L, 15)} ({time.time() - T0:.0f} s)", flush=True)
    graded = bad = 0
    for name in [p for p in opt["plants"].split(",") if p]:
        h, size = P.PLANTS[name]
        pts, _ = P.make(allD, ev, h, size)
        if what == "bst":
            Ds = F.clean(pts, [d for d in allD if 10 <= d <= 61.03], verbose=False)
            est, spread, n = run_bst(pts, Ds, omega)
            ok = abs(est - L) <= BAR_BST
            line = f"median of {n} tables off by {mp.nstr(est - L, 3)}, middle-half spread {mp.nstr(spread, 2)}"
        else:
            est, sd, n, top = run_aic(pts, allD, equal)
            ok = abs(est - L) <= BAR_AIC and abs(est - L) <= 2 * sd
            line = (f"{n} fits, mean off by {mp.nstr(est - L, 3)}, sd {mp.nstr(sd, 2)} "
                    f"({mp.nstr((est - L) / sd, 2) if sd else 'inf'} sd), largest weight {mp.nstr(top, 3)}")
        tag = "info" if name == "T6" else ("pass" if ok else "FAIL")
        if name != "T6":
            graded += 1
            bad += not ok
        print(f"{name}: {line} -> {tag} ({time.time() - T0:.0f} s)", flush=True)
    print(f"{graded - bad} of {graded} graded plants passed", flush=True)
    if opt["real"] == "1" and not opt["mutate"]:
        if what == "bst":
            Ds = F.clean(real, [d for d in allD if 10 <= d <= 61.03], verbose=False)
            est, spread, n = run_bst(real, Ds, omega)
            print(f"real: median of {n} tables {mp.nstr(est, 12)}, middle-half spread {mp.nstr(spread, 2)}")
        else:
            est, sd, n, top = run_aic(real, allD, equal)
            print(f"real: {n} fits, mean {mp.nstr(est, 12)}, sd {mp.nstr(sd, 2)}, largest weight {mp.nstr(top, 3)}")
        dev = est - README[0]
        print(f"against the README's {mp.nstr(README[0], 10)} +- {mp.nstr(README[1], 2)}: {mp.nstr(dev, 3)} "
              f"({'inside' if abs(dev) <= README[1] else 'OUTSIDE'} the bar) ({time.time() - T0:.0f} s)", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
