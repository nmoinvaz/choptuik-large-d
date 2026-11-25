"""Rounding amplification of form S's horizon sum at x* against D, in the affine variable (a = 0, Track B's w) and the
Moebius variable (a > 0), at the seeds of Track B's seed_precision.py (#1090).

Relative noise eta is put on the stored horizon coefficients (U, V; or c) after every order; printed is the change of
the sum of U at x* over eta and over max|U(x*)|, the larger of two draws. The seeds are Track B's D-100 form-S solution
rescaled to each D (p1, Q0 by sqrt(n0/n), c0 - 1 by n0/n, Delta from the 1/D fit), as in
solver/factored/seed_precision.py, so they are not solutions; the noise test measures the recurrence's linear response
there. PASS when the Moebius amplification stays below 1e3 at every D (no exponential growth); the control
--mutate=affine runs the same criterion on a = 0, which must FAIL.

Usage: python3 noise_variable.py SOL.json [--Ds=100,150,200,245,400] [--a=1] [--eta=1e-14] [--mutate=affine]
  SOL.json: Sdd_D100_M8.json of the raw folder I1088-mac/.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mobius_s as ms  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
fn = [a for a in sys.argv[1:] if not a.startswith("--")][0]
Ds = [float(v) for v in opts.get("Ds", "100,150,200,245,400").split(",")]
a_m = float(opts.get("a", 1.0))
eta = float(opts.get("eta", 1e-14))
mut = opts.get("mutate", "")
js = json.load(open(fn))
M, D0 = js["M"], js["D"]
y0 = np.concatenate([js["p1"], js["c0"], js["Q0"], [js["Delta"]]])
y0 = y0 + np.concatenate([js["p1_lo"], js["c0_lo"], js["Q0_lo"], [js["Delta_lo"]]])


def seed(D):
    y = y0.copy()
    n, n0 = D - 2, D0 - 2
    if abs(D - D0) > 1e-12:
        s = np.sqrt(n0 / n)
        y[0:M] *= s
        y[2 * M:3 * M] *= s
        y[M:2 * M] = 1 + (y[M:2 * M] - 1) * n0 / n
        y[3 * M] = 1.6439 + 10.37 / D - 49.17 / D ** 2
    return y[2 * M:3 * M], 2 * np.pi / y[3 * M]


def amp(f, Q0, om, names):
    with np.errstate(all="ignore"):
        base = f.horizon(Q0, om)["U"]
        f.noise = (names, eta, np.random.default_rng(7))
        d = max(np.max(np.abs(f.horizon(Q0, om)["U"] - base)) for _ in range(2))
        f.noise = None
    return d / eta / np.max(np.abs(base)), float(np.max(np.abs(base)))


print("%8s %6s | %-34s | %-34s" % ("D", "ks", "a = 0: amp(U,V)  amp(c)  max|U*|", "a = %g: amp(U,V)  amp(c)  max|U*|" % a_m))
worst = 0.0
for D in Ds:
    Q0, om = seed(D)
    t0 = time.time()
    f0 = ms.FormS(D, M, a=0.0, kb=150)
    f1 = ms.FormS(D, M, a=a_m, kb=150)
    u0, s0 = amp(f0, Q0, om, ["U", "V"])
    c0, _ = amp(f0, Q0, om, ["c"])
    u1, s1 = amp(f1, Q0, om, ["U", "V"])
    c1, _ = amp(f1, Q0, om, ["c"])
    w = max(u0, c0) if mut == "affine" else max(u1, c1)
    worst = max(worst, w if np.isfinite(w) else np.inf)
    print("%8.2f %6d | %9.1e %9.1e %12.4e | %9.1e %9.1e %12.4e  (%.0f s)" % (D, f0.ks, u0, c0, s0, u1, c1, s1, time.time() - t0), flush=True)
ok = worst < 1e3
print("largest amplification of the %s variable: %.1e  %s" % ("affine" if mut == "affine" else "Moebius", worst, "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
