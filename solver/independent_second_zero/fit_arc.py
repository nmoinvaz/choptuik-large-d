"""Fit the data's sigma_0..sigma_22 at w = 1.09, 1.10, 1.11 with the arc family of arc_family.py (issue #214).

    python3 fit_arc.py <dir with I82-win2's sigF_w1.09/1.10/1.11_K45_J1500_p1280.json> [--out arc_fit.json]

Model.  c_n = n! sigma_n = int_0^1 gamma(s)^n rho(s) ds / int_0^1 rho(s) ds, the moments of a density on an arc,
    gamma(s) = za + s (zp - za) + kap s (1 - s),   rho(s) = s^{al-1} (1 - s)^{be-1} (e0 + e1 s),
with every parameter linear in w: p(w) = pA + pB (w - 1.10)/0.01.  The moments are exact sums of Beta integrals,
int_0^1 s^{al-1+k} (1-s)^{be-1} ds / B(al, be) = (al)_k / (al+be)_k.
Fit.  Variable projection (Golub & Pereyra, SIAM J. Numer. Anal. 10, 413 (1973)): (e0, e1)(w) by weighted linear
least squares, weights 1/|c_n| (row n = 0 weighted 1e4, so that c_0 = 1); (za, zp, kap, al, be)(w) by
Levenberg-Marquardt (scipy).  Starts: 30 random single-node starts at w = 1.10 (numpy seed 5), the best continued to
1.09 and 1.11, then the joint fit.  Double precision; arc_family.py evaluates the fitted family to any precision.
Writes the parameters and |log(c_model/c_data)| per n and node.  No zero of S is computed here.
"""
import json
import os
import sys
from math import factorial

import mpmath as mp
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
WS = ["1.09", "1.10", "1.11"]
TS = [-1.0, 0.0, 1.0]
N = 23
W0 = 1e4


def load_c(D, w):
    d = json.load(open(os.path.join(D, f"sigF_w{w}_K45_J1500_p1280.json")))
    with mp.workdps(40):
        return np.array([complex(mp.mpc(mp.mpf(a), mp.mpf(b)) * factorial(n)) for n, (a, b) in enumerate(d["sigma_str"])])


def cp(p, i):
    return complex(p[2 * i], p[2 * i + 1])


def design(th, M=1):
    """A[n, m] = int gamma^n s^{al-1+m} (1-s)^{be-1} ds / B(al, be), n < N, m <= M"""
    za, zp, kap, al, be = [cp(th, i) for i in range(5)]
    path = np.array([za, zp - za + kap, -kap])
    K = 2 * (N - 1) + M + 2
    pr = np.ones(K, complex)
    for k in range(1, K):
        pr[k] = pr[k - 1] * (al + k - 1) / (al + be + k - 1)
    A = np.zeros((N, M + 1), complex)
    P = np.array([1.0 + 0j])
    for n in range(N):
        for m in range(M + 1):
            A[n, m] = np.sum(P * pr[m:m + len(P)])
        P = np.convolve(P, path)
    return A


def weights(c):
    W = 1 / np.abs(c)
    W[0] *= W0
    return W


def node_res(th, c):
    A = design(th)
    W = weights(c)
    e, *_ = np.linalg.lstsq(A * W[:, None], c * W, rcond=None)
    r = (A @ e - c) * W
    return np.concatenate([r.real, r.imag])


def joint(p, C, ret=False):
    thA, thB = p[:10], p[10:]
    rows, rhs, As = [], [], []
    for w, t in zip(WS, TS):
        A = design(thA + t * thB)
        As.append(A)
        W = weights(C[w])
        rows.append(np.hstack([A, t * A]) * W[:, None])
        rhs.append(C[w] * W)
    big, y = np.vstack(rows), np.concatenate(rhs)
    e, *_ = np.linalg.lstsq(big, y, rcond=None)
    if ret:
        return e, As
    r = big @ e - y
    return np.concatenate([r.real, r.imag])


def fit(D, nstart=30, seed=5):
    C = {w: load_c(D, w) for w in WS}
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(nstart):
        z = [complex(rng.uniform(0.5, 1.1), rng.uniform(-0.2, 0.3)), complex(rng.uniform(1.4, 1.8), rng.uniform(0.9, 1.3)),
             complex(rng.normal(0, 0.6), rng.normal(0, 0.6)), complex(rng.uniform(0.5, 4), rng.normal(0, 2)),
             complex(rng.uniform(1, 9), rng.normal(0, 3))]
        st = np.array([v for x in z for v in (x.real, x.imag)])
        try:
            r = least_squares(node_res, st, args=(C["1.10"],), method="lm", xtol=1e-15, ftol=1e-15, max_nfev=4000)
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(r.cost) and (best is None or r.cost < best.cost):
            best = r
    per = {"1.10": best.x}
    for w in ["1.09", "1.11"]:
        per[w] = least_squares(node_res, best.x, args=(C[w],), method="lm", xtol=1e-15, ftol=1e-15, max_nfev=4000).x
    p0 = np.concatenate([per["1.10"], (per["1.11"] - per["1.09"]) / 2])
    r = least_squares(joint, p0, args=(C,), method="lm", xtol=1e-15, ftol=1e-15, max_nfev=20000)
    e, As = joint(r.x, C, ret=True)
    names = ["za", "zp", "kap", "al", "be"]
    out = {"model": "arc: gamma(s) = za + s (zp - za) + kap s (1 - s), rho(s) = s^(al-1) (1-s)^(be-1) (e0 + e1 s); "
                    "p(w) = pA + pB (w - 1.10)/0.01",
           "A": {k: [r.x[2 * i], r.x[2 * i + 1]] for i, k in enumerate(names)},
           "B": {k: [r.x[10 + 2 * i], r.x[10 + 2 * i + 1]] for i, k in enumerate(names)},
           "cost": float(r.cost), "abs_log_ratio": {}, "max_abs_log_ratio_n1_22": {}}
    out["A"]["e0"], out["A"]["e1"] = [[x.real, x.imag] for x in e[:2]]
    out["B"]["e0"], out["B"]["e1"] = [[x.real, x.imag] for x in e[2:]]
    for w, t, A in zip(WS, TS, As):
        lr = np.abs(np.log(A @ (e[:2] + t * e[2:]) / C[w]))
        out["abs_log_ratio"][w] = [float(x) for x in lr]
        out["max_abs_log_ratio_n1_22"][w] = float(lr[1:].max())
    return out


if __name__ == "__main__":
    argv = sys.argv[1:]
    outp = os.path.join(HERE, "arc_fit.json")
    if "--out" in argv:
        i = argv.index("--out")
        outp = argv[i + 1]
        del argv[i:i + 2]
    res = fit(argv[0])
    for w in WS:
        print(f"w {w}: max |log(c_model/c_data)|, n = 1..22: {res['max_abs_log_ratio_n1_22'][w]:.2e}; n = 0: "
              f"{res['abs_log_ratio'][w][0]:.1e}; per n " + " ".join(f"{x:.1e}" for x in res["abs_log_ratio"][w][1:]))
    for k in res["A"]:
        print(f"  {k}: {complex(*res['A'][k]):.6f} + {complex(*res['B'][k]):.6f} (w - 1.10)/0.01")
    json.dump(res, open(outp, "w"), indent=1)
