"""Extrapolation in k_c of the l = 1 root lambda_1 and of the zero D_c (#1103), planted tests first.

usage: python i1103_extrap.py OUT_JSON [--planted] [--mutate=NAME]
  Models, fitted by weighted least squares (sigma = #1101's root scatter 1.75e-4, or the stated one):
    geo    r(k) = r_inf + A rho^(k/8)                      (tau truncation of an analytic solution)
    alg    r(k) = r_inf + B k^-p                           (a weak singularity, or a fixed-order discretisation)
    rk4    r(k) = r_inf + B k^-4                           (freeze_from: RK4 step h ~ 1/k_c, error ~ h^4)
    geo+rk4 r(k) = r_inf + A rho^(k/8) + B k^-4           (both, rho free)
  Planted: sequences on k_c 87..135 (step 8) from each model with Gaussian noise of 1.75e-4; for each fit, the Monte
  Carlo spread of r_inf over 400 noise draws, the coverage of the truth by +- 2 spreads, and the bias of a wrong model.
  Pass criteria (written before the fits on real data): the true model's r_inf lies within 2 of its own spreads of the
  truth in at least 85% of draws; and geo fitted to geo+rk4 data must miss (its bias over 2 spreads), which is the
  control that has to fail. --mutate=noRK fits geo in place of geo+rk4 everywhere (the planted test must fail).
  Every quoted spread is a Monte Carlo spread over noise draws at the assumed scatter, a fit spread, not an error bar;
  the spread between models is the model uncertainty, also not a bound.
"""
import json, sys
import numpy as np
from scipy.optimize import least_squares

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")
SIG = 1.75e-4


def model(name, k, th):
    k = np.asarray(k, float)
    if name == "geo":
        return th[0] + th[1] * th[2] ** (k / 8)
    if name == "alg":
        return th[0] + th[1] * (k / 100.0) ** (-th[2])
    if name == "rk4":
        return th[0] + th[1] * (k / 100.0) ** -4
    if name == "geo+rk4":
        return th[0] + th[1] * th[2] ** (k / 8) + th[3] * (k / 100.0) ** -4
    raise ValueError(name)


def fit(name, k, r, sig=SIG):
    if MUT == "noRK" and name == "geo+rk4":
        name = "geo"
    k = np.asarray(k, float); r = np.asarray(r, float)
    span = r[-1] - r[0]
    starts = {"geo": [[r[-1], -span * 4, 0.5], [r[-1], -span * 40, 0.3], [r[-1], -span, 0.8]],
              "alg": [[r[-1], -span, 4.0], [r[-1], -span, 8.0], [r[-1], -span, 2.0]],
              "rk4": [[r[-1], -span]],
              "geo+rk4": [[r[-1], -span * 4, 0.5, -span * 0.1], [r[-1], -span * 40, 0.3, -span], [r[-1], -span, 0.6, -span]]}
    bounds = {"geo": ([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, 0.999]),
              "alg": ([-np.inf, -np.inf, 0.3], [np.inf, np.inf, 40.0]),
              "rk4": ([-np.inf, -np.inf], [np.inf, np.inf]),
              "geo+rk4": ([-np.inf, -np.inf, 1e-3, -np.inf], [np.inf, np.inf, 0.999, np.inf])}
    best = None
    for s in starts[name]:
        try:
            o = least_squares(lambda th: (model(name, k, th) - r) / sig, s, bounds=bounds[name], max_nfev=4000)
        except Exception:
            continue
        if best is None or o.cost < best.cost:
            best = o
    dof = max(1, len(k) - len(best.x))
    return dict(model=name, theta=[float(v) for v in best.x], r_inf=float(best.x[0]),
                chi2_dof=float(2 * best.cost / dof), n=len(k))


def mc(name, k, truth_fn, sig=SIG, ndraw=400, rng=None):
    rng = rng or np.random.default_rng(7)
    base = truth_fn(np.asarray(k, float))
    vals = [fit(name, k, base + sig * rng.standard_normal(len(k)), sig)["r_inf"] for _ in range(ndraw)]
    return np.array(vals)


def planted():
    k = np.arange(87, 136, 8)
    truths = {
        "geo": (lambda kk: 0.05 - 0.6 * 0.55 ** (kk / 8), 0.05),
        "rk4": (lambda kk: 0.065 - 0.016 * (kk / 100.0) ** -4 * 1.0, 0.065),
        "geo+rk4": (lambda kk: 0.065 - 3.0 * 0.40 ** (kk / 8) - 0.005 * (kk / 100.0) ** -4, 0.065),
    }
    ok = True
    rows = []
    for tname, (fn, rinf) in truths.items():
        for fname in ("geo", "alg", "rk4", "geo+rk4"):
            v = mc(fname, k, fn, ndraw=200)
            med, spr = float(np.median(v)), float(np.std(v))
            cover = float(np.mean(np.abs(v - rinf) <= 2 * spr)) if spr > 0 else 0.0
            rows.append(dict(truth=tname, fit=fname, median=med, spread=spr, bias=med - rinf, coverage=cover))
            print(f"planted truth {tname:8s} fit {fname:8s}: r_inf median {med:+.5f} (truth {rinf:+.5f}, bias "
                  f"{med - rinf:+.5f}), MC spread {spr:.5f}, |bias|/spread {abs(med - rinf) / max(spr, 1e-12):.1f}")
    get = lambda t, f: next(r for r in rows if r["truth"] == t and r["fit"] == f)
    c1 = get("geo", "geo"); c2 = get("rk4", "rk4"); c3 = get("geo+rk4", "geo+rk4"); ctl = get("geo+rk4", "geo")
    p1 = all(abs(c["bias"]) <= 2 * c["spread"] for c in (c1, c2, c3))
    p2 = abs(ctl["bias"]) > 2 * ctl["spread"]
    print(f"planted: each true model within 2 of its spreads of the truth -> {'pass' if p1 else 'FAIL'}; "
          f"control, geo fitted to geo+rk4 data misses by {abs(ctl['bias']) / ctl['spread']:.1f} spreads -> "
          f"{'pass (it fails as it must)' if p2 else 'FAIL (the control did not fire)'}")
    ok = p1 and p2
    return ok, rows


def main():
    ok, prow = planted()
    print("planted tests:", "PASS" if ok else "FAIL", flush=True)
    if "--planted" in sys.argv or not ok:
        sys.exit(0 if ok else 1)
    out = sys.argv[1]
    # the stored roots (raw I1025-*: point_summary / secant logs; k_c 135 at D 23.0 is a non-robust bracket of width
    # 1.9e-4, so its weight uses that width), and the zeros D_c(k_c) of #1025's quadratic fits
    seqs = {
        "root D 23.0": (np.array([87, 95, 103, 111, 119, 127, 135]),
                        np.array([-0.02540416, 0.00870734, 0.02650553, 0.03556371, 0.04106103, 0.04571076, 0.04957509]),
                        np.array([SIG] * 6 + [2.0e-4])),
        "root D 23.0625": (np.array([95, 103, 111, 127, 135, 143]),
                           np.array([-0.05862411, -0.02851154, -0.01247904, 0.00364182, 0.00939087, 0.01513874]),
                           np.array([SIG, SIG, SIG, 4.2e-4, 2.8e-4, SIG])),
        "D_c": (np.array([87, 95, 103, 111, 119, 127, 135]),
                np.array([22.97890, 23.00863, 23.03095, 23.04680, 23.05777, 23.06768, 23.07598]), None),
    }
    res = dict(planted=prow, real={})
    for nm, (k, r, s) in seqs.items():
        if s is None:
            # D_c's scatter: the root scatter over |d lambda_1 / dD| ~ 0.64 at k_c 135 (D 23.0 -> 23.0625)
            s = np.full(len(k), SIG / 0.64)
        res["real"][nm] = {}
        print(f"\n{nm}: " + " ".join(f"{a}:{b:.6f}" for a, b in zip(k, r)))
        st = np.diff(r)
        print("  steps " + " ".join(f"{v:+.5f}" for v in st) + "; ratios " + " ".join(f"{b / a:.3f}" for a, b in zip(st[:-1], st[1:])))
        for fname in ("geo", "alg", "rk4", "geo+rk4"):
            for lo in (0, 2, 3):
                kk, rr, ss = k[lo:], r[lo:], s[lo:]
                if len(kk) <= {"geo": 3, "alg": 3, "rk4": 2, "geo+rk4": 4}[fname]:
                    continue
                f = fit(fname, kk, rr, ss)
                rng = np.random.default_rng(11)
                base = model(f["model"], kk, f["theta"])
                v = np.array([fit(fname, kk, base + ss * rng.standard_normal(len(kk)), ss)["r_inf"] for _ in range(200)])
                f["mc_spread"] = float(np.std(v))
                res["real"][nm][f"{fname}_from{kk[0]}"] = f
                print(f"  {fname:8s} k_c >= {kk[0]:3d}: r_inf {f['r_inf']:+.5f}  MC spread {f['mc_spread']:.5f}  "
                      f"chi2/dof {f['chi2_dof']:.2f}  theta " + " ".join(f"{v:.3g}" for v in f["theta"][1:]))
    with open(out, "w") as fh:
        json.dump(res, fh)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
