"""Two-variable planted test for the zero D_c (#1103): truncation geometric in k_c plus RK4 error ~ (kappa / k_c)^4 at
the measured rates, and what a k_c-only (geometric) extrapolation of D_c(k_c) gives against the planted truth.

usage: python i1103_twovar.py OUT_JSON [--mutate=zeroB]
  Rates: the truncation's per-rung ratio rho (k_c -> k_c + 8) is set to the measured values, 0.35 (the low-harmonic
  residual band, i1103_bands.py) and 0.5 (the early root steps, k_c 87 -> 103, which the truncation dominates); the RK
  part is B (k_c / 100)^-4 at kappa 0.7 (freeze_from: h ~ kappa / k_c). Amplitudes A, B and the truth D_c* are fitted
  to #1025's zeros D_c(k_c) (k_c 87-135) with rho fixed; then 400 noisy copies (sigma = 1.75e-4 / 0.64, the root
  scatter of #1101 over |d lambda_1 / dD| at k_c 135) are fitted by:
    geo      D_c(k) = D_inf + a rho^(k/8), rho free            (a k_c-only geometric extrapolation)
    last     D_c(135)                                         (no extrapolation)
    geo+rk4  D_inf + a rho^(k/8) + b (k/100)^-4, rho free    (the two-variable model)
  and the bias of each against D_c* is the answer. Pass criterion (set before the real fits): the two-variable fit's
  median lies within 2 of its MC spreads of D_c*; the control --mutate=zeroB plants rho-only data with B = 0, where
  geo must be unbiased (so a geo bias in the main test is the RK term's), and the main test must then report no
  geo bias (it does fail as a test of the RK term).
"""
import json, sys
import numpy as np
from i1103_extrap import fit, model

MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")
K = np.array([87, 95, 103, 111, 119, 127, 135])
DC = np.array([22.97890, 23.00863, 23.03095, 23.04680, 23.05777, 23.06768, 23.07598])
SIG = 1.75e-4 / 0.64


def fixed_rho_fit(rho, k, d):
    """D_c* + a rho^(k/8) + b (k/100)^-4, rho fixed: linear least squares"""
    X = np.stack([np.ones(len(k)), rho ** (k / 8.0), (k / 100.0) ** -4], 1)
    c, *_ = np.linalg.lstsq(X, d, rcond=None)
    r = X @ c - d
    return c, float(np.sqrt(np.mean(r * r)) / SIG)


def main():
    out = sys.argv[1]
    rng = np.random.default_rng(1103)
    res = {}
    for rho in (0.35, 0.5):
        c, rms = fixed_rho_fit(rho, K, DC)
        if MUT == "zeroB":
            c = c.copy(); c[2] = 0.0
        truth = lambda k: c[0] + c[1] * rho ** (k / 8.0) + c[2] * (k / 100.0) ** -4
        print(f"rho {rho}: fit to the real zeros D_c* {c[0]:.5f}, a {c[1]:.4g}, b {c[2]:.4g} (rms/sigma {rms:.2f}); "
              f"RK part at k_c 135: {c[2] * 1.35 ** -4:+.5f}, truncation part: {c[1] * rho ** (135 / 8):+.6f}")
        base = truth(K.astype(float))
        vals = {"geo": [], "last": [], "geo+rk4": []}
        for _ in range(400):
            d = base + SIG * rng.standard_normal(len(K))
            vals["geo"].append(fit("geo", K, d, SIG)["r_inf"])
            vals["geo+rk4"].append(fit("geo+rk4", K, d, SIG)["r_inf"])
            vals["last"].append(d[-1])
        row = dict(rho=rho, Dc_true=float(c[0]), a=float(c[1]), b=float(c[2]), rms_sigma=rms)
        for nm, v in vals.items():
            v = np.array(v)
            row[nm] = dict(median=float(np.median(v)), spread=float(np.std(v)), bias=float(np.median(v) - c[0]))
            print(f"   {nm:8s}: median {np.median(v):.5f}, MC spread {np.std(v):.5f}, bias against D_c* "
                  f"{np.median(v) - c[0]:+.5f}")
        res[str(rho)] = row
    ok_tv = all(abs(r["geo+rk4"]["bias"]) <= 2 * max(r["geo+rk4"]["spread"], 1e-9) for r in res.values())
    geo_biased = any(abs(r["geo"]["bias"]) > 2 * r["geo"]["spread"] for r in res.values())
    print(f"\ntwo-variable fit within 2 spreads of the truth: {'pass' if ok_tv else 'FAIL'}; k_c-only geo biased "
          f"(over 2 spreads) in at least one case: {geo_biased}")
    # the real zeros with each method
    real = {}
    for nm in ("geo", "geo+rk4", "rk4", "alg"):
        for lo in (0, 2):
            f = fit(nm, K[lo:], DC[lo:], SIG)
            real[f"{nm}_from{K[lo]}"] = f
            print(f"real zeros, {nm:8s} k_c >= {K[lo]}: D_inf {f['r_inf']:.5f} chi2/dof {f['chi2_dof']:.2f} theta "
                  + " ".join(f"{v:.3g}" for v in f["theta"][1:]))
    res["real"] = real
    with open(out, "w") as fh:
        json.dump(res, fh)
    sys.exit(0 if ok_tv else 1)


if __name__ == "__main__":
    main()
