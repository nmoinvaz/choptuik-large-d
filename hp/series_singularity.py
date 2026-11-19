#!/usr/bin/env python3
"""The nearest singularity of Delta(D) from its Taylor coefficients in D at a low-D base (#997; anm_lowd.py
--save_series), by the ratio method with Domb and Sykes' 1/k extrapolation (Domb and Sykes, Proc. R. Soc. A 240
(1957) 214).

For Delta(D) ~ C (1 - h/h*)^mu near a dominant singularity at D* = D0 + h*, the ratios r_k = Delta_k / Delta_{k-1}
of the Taylor coefficients in h = D - D0 tend to (1/h*)(1 - (1 + mu)/k). A least-squares line r_k = a + b/k over a
window k = K1..K2 gives h* = 1/a and mu = -b/a - 1. Real alternating ratios put D* on the real axis below D0.

    python3 series_singularity.py SERIES.npz K1 K2 [SERIES2.npz K1 K2 ...]    D*, mu and the fit's residual per window
    python3 series_singularity.py --planted     plants through the same fit, criteria fixed before the real data:
        P1: D* = 3.000, mu = 0.7, with a regular part and a confluent term, Taylor at D0 3.15, window 5..9:
            |D* - 3.000| <= 0.01;
        P2: the same at D0 3.0475, window 3..6: |D* - 3.000| <= 0.01;
        P3 (control): D* = 3.0431 (the sonic closing), at D0 3.0475, window 3..6: |D* - 3.0431| <= 0.002, which
            shows the fit can see a singularity that close;
        --mutate: P1 read with the ratios' sign flipped, which must not return D* within 0.01 of 3.000 (exit 1).
Exit 0 when every check passes (with --mutate: when the mutated check fails, as it must).
"""
import sys

import mpmath as mp
import numpy as np


def coeffs(path):
    d = np.load(path)
    return float(d["D0"]), np.asarray(d["delta_hi"], dtype=float) + np.asarray(d["delta_lo"], dtype=float)


def fit(dk, k1, k2, sign=1.0):
    """least-squares r_k = a + b/k over k = k1..k2; returns h*, mu and the rms residual of the line"""
    ks = np.arange(k1, k2 + 1)
    r = sign * dk[ks] / dk[ks - 1]
    A = np.c_[np.ones_like(ks, dtype=float), 1.0 / ks]
    (a, b), *_ = np.linalg.lstsq(A, r, rcond=None)
    res = float(np.sqrt(np.mean((A @ [a, b] - r) ** 2)))
    return 1.0 / a, -b / a - 1.0, res


def planted_series(Dstar, mu, D0, N=12):
    """Taylor coefficients at D0 of 1.6 + 0.9 (D - Dstar)^mu (1 + 0.8 (D - Dstar)) + 0.3 (D - Dstar)"""
    mp.mp.dps = 40
    f = lambda D: 1.6 + 0.9 * (D - Dstar) ** mu * (1 + 0.8 * (D - Dstar)) + 0.3 * (D - Dstar)
    return np.array([float(c) for c in mp.taylor(f, mp.mpf(D0), N)])


def planted(mutate):
    checks = []
    for name, Ds, mu, D0, k1, k2, tol in (("P1", 3.0, 0.7, 3.15, 5, 9, 0.01), ("P2", 3.0, 0.7, 3.0475, 3, 6, 0.01),
                                         ("P3", 3.0431, 0.7, 3.0475, 3, 6, 0.002)):
        dk = planted_series(Ds, mu, D0)
        sign = -1.0 if (mutate and name == "P1") else 1.0
        h, m, res = fit(dk, k1, k2, sign)
        ok = abs(D0 + h - Ds) <= tol
        print(f"{name}: planted D* {Ds}, mu {mu}, D0 {D0}, window {k1}..{k2}: D* {D0 + h:.4f}, mu {m:.3f}, "
              f"rms {res:.2e}: {'PASS' if ok else 'FAIL'}{' (mutated)' if sign < 0 else ''}")
        checks.append((name, ok))
    if mutate:
        return 0 if not dict(checks)["P1"] else 1
    return 0 if all(ok for _, ok in checks) else 1


def main(argv):
    if "--planted" in argv:
        return planted("--mutate" in argv)
    i = 0
    while i < len(argv):
        path, k1, k2 = argv[i], int(argv[i + 1]), int(argv[i + 2])
        D0, dk = coeffs(path)
        h, m, res = fit(dk, k1, k2)
        r = dk[1:] / dk[:-1]
        print(f"{path}: D0 {D0}, ratios k = 1..12: " + " ".join(f"{x:.3g}" for x in r[:12]))
        print(f"  window {k1}..{k2}: D* {D0 + h:.4f} (h* {h:+.4f}), mu {m:.3f}, rms {res:.2e}")
        i += 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
