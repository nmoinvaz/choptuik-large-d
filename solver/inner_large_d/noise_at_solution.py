"""Rounding amplification of the horizon sum at x* at stored solutions of solve_cont.py, against the Moebius parameter a
and the orders that carry the noise (#1090). Relative noise eta on U, V after every order k in [klo, khi]; printed:
the change of max|U(x*)| over eta and over max|U(x*)| (two draws, the larger), and max_k |U_k| / max|U(x*)|.

Usage: python3 noise_at_solution.py SOL.json [SOL2.json ...] [--as=0,1,2,3] [--bands=all,low,mid,high]
  low: k < sigma_R/2; mid: sigma_R/2 <= k < 1.5 sigma_R; high: k >= 1.5 sigma_R (sigma_R ~ n/2 + 2.7)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mobius_s as ms  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
files = [a for a in sys.argv[1:] if not a.startswith("--")]
As = [float(v) for v in opts.get("as", "0,1,2,3").split(",")]
bands = opts.get("bands", "all,low,mid,high").split(",")
eta = 1e-13
for fn in files:
    r = json.load(open(fn))
    D, M = r["D"], r["M"]
    y = np.concatenate([r["p1"], r["c0"], r["Q0"], [r["Delta"]]])
    n = D - 2
    sR = n / 2 + 2.7
    for a in As:
        f = ms.FormS(D, M, a=a, kb=150)
        om = 2 * np.pi / y[3 * M]
        with np.errstate(all="ignore"):
            base = f.horizon(y[2 * M:3 * M], om, keep=True)
        u = np.max(np.abs(base["U"]))
        big = float(np.max(np.abs(base["series"]["U"]))) / u
        row = []
        for b in bands:
            lo, hi = {"all": (0, 10 ** 9), "low": (0, sR / 2), "mid": (sR / 2, 1.5 * sR), "high": (1.5 * sR, 10 ** 9)}[b]
            rng = np.random.default_rng(7)
            worst = 0.0
            for _ in range(2):
                orig = f.horizon.__func__
                g = ms.FormS(D, M, a=a, kb=150)
                g.noise = (["U", "V"], 0.0, rng)
                # band-limited noise: wrap the rng so that orders outside [lo, hi) get zero noise
                class R:
                    k = -1
                    def standard_normal(self, m):
                        R.k += 1
                        kk = R.k // 2  # two arrays (U, V) per order
                        return rng.standard_normal(m) * (1.0 if lo <= kk < hi else 0.0)
                R.k = -1
                g.noise = (["U", "V"], eta, R())
                with np.errstate(all="ignore"):
                    d = np.max(np.abs(g.horizon(y[2 * M:3 * M], om)["U"] - base["U"]))
                worst = max(worst, d / eta / u)
            row.append("%s %.1e" % (b, worst))
        print("D %7g M %d a %g: max|U*| %.4e  max_k|U_k|/|U*| %.1e  amplification %s" % (D, M, a, u, big, "  ".join(row)), flush=True)
