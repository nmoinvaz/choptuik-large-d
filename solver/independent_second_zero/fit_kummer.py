"""Fit sigma_1..sigma_22 of the data with e^{za x} 1F1(a; b; (zp - za) x), for the planted family's parameters only.

    python3 fit_kummer.py <dir with I82-win2's sigF_w1.09/1.10/1.11_K45_J1500_p1280.json> [--out kummer_fit.json]

Least squares (Levenberg-Marquardt, scipy) on log(sigma_model/sigma_data), n = 1..22.  Writes za, zp, a, b per node
and the per-n |log ratio|.  No zero of S is computed here.
"""
import json
import os
import sys

import mpmath as mp
import numpy as np
from scipy.optimize import least_squares

mp.mp.dps = 30
HERE = os.path.dirname(os.path.abspath(__file__))


def model_sigma(za, dz, a, b, N=23):
    k = [mp.rf(a, m) * dz ** m / (mp.rf(b, m) * mp.factorial(m)) for m in range(N)]
    e = [za ** j / mp.factorial(j) for j in range(N)]
    return [mp.fsum(e[n - m] * k[m] for m in range(n + 1)) for n in range(N)]


def unpack(p):
    return tuple(mp.mpc(p[2 * i], p[2 * i + 1]) for i in range(4))


def fit_node(path):
    d = json.load(open(path))
    s = [mp.mpc(mp.mpf(x), mp.mpf(y)) for x, y in d["sigma_str"]]

    def res(p):
        m = model_sigma(*unpack(p))
        out = []
        for n in range(1, 23):
            r = mp.log(m[n] / s[n])
            out += [float(r.real), float(r.imag)]
        return np.array(out)

    best = None
    for start in [[0.68, 0, 1.0, 1.2, 1, 0, 8, -3], [0.8, 0.1, 0.8, 1.1, 1.5, -0.5, 8, -2]]:
        r = least_squares(res, start, method="lm", xtol=1e-14, ftol=1e-14, max_nfev=4000)
        if best is None or r.cost < best.cost:
            best = r
    za, dz, a, b = unpack(best.x)
    rr = res(best.x)
    return {"za": [float(za.real), float(za.imag)], "zp": [float((za + dz).real), float((za + dz).imag)],
            "a": [float(a.real), float(a.imag)], "b": [float(b.real), float(b.imag)],
            "abs_log_ratio": [float(np.hypot(rr[2 * i], rr[2 * i + 1])) for i in range(22)], "cost": float(best.cost)}


if __name__ == "__main__":
    argv = sys.argv[1:]
    out = os.path.join(HERE, "kummer_fit.json")
    if "--out" in argv:
        i = argv.index("--out")
        out = argv[i + 1]
        del argv[i:i + 2]
    D = argv[0]
    res = {}
    for w in ["1.09", "1.10", "1.11"]:
        res[w] = fit_node(os.path.join(D, f"sigF_w{w}_K45_J1500_p1280.json"))
        print(w, {k: v for k, v in res[w].items() if k != "abs_log_ratio"},
              "max |log ratio|", f"{max(res[w]['abs_log_ratio']):.2e}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
