"""Cost of one form-S residual against D, Moebius horizon variable, banded (kb = 150): float64 and double-double,
one evaluation each at the seed of noise_variable.py (#1090, check 4). Also the float64 noise amplification there.

Usage: python3 cost_vs_D.py SOL.json [--D64=100,400,1000,2000,5000,10000] [--Ddd=100,400,1000,2000] [--M=8]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddnp import DD  # noqa: E402
import mobius_s as ms  # noqa: E402
import mobius_sdd as msd  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
fn = [a for a in sys.argv[1:] if not a.startswith("--")][0]
js = json.load(open(fn))
M0, D0 = js["M"], js["D"]
M = int(opts.get("M", M0))
y0 = np.concatenate([js["p1"], js["c0"], js["Q0"], [js["Delta"]]])
if M != M0:
    y0 = np.concatenate([ms.resample_odd(np.array(js["p1"]), M), ms.resample_even(np.array(js["c0"]), M),
                         ms.resample_odd(np.array(js["Q0"]), M), [js["Delta"]]])


def seed(D):
    y = y0.copy()
    n, n0 = D - 2, D0 - 2
    s = np.sqrt(n0 / n)
    y[0:M] *= s
    y[2 * M:3 * M] *= s
    y[M:2 * M] = 1 + (y[M:2 * M] - 1) * n0 / n
    y[3 * M] = 1.6439 + 10.37 / D - 49.17 / D ** 2
    return y


print("%8s %6s %12s %12s %12s" % ("D", "ks", "f64 s", "dd s", "f64 |U*|"))
D64 = [float(v) for v in opts.get("D64", "100,400,1000,2000,5000,10000").split(",")]
Ddd = [float(v) for v in opts.get("Ddd", "100,400,1000,2000").split(",")]
for D in sorted(set(D64) | set(Ddd)):
    y = seed(D)
    f = ms.FormS(D, M, kb=150)
    t64 = tdd = float("nan")
    u = float("nan")
    if D in D64:
        t0 = time.time()
        with np.errstate(all="ignore"):
            f.residual(y)
        t64 = time.time() - t0
        u = float(np.max(np.abs(f.last[1]["U"])))
    if D in Ddd:
        g = msd.FormSDD(D, M, kb=150)
        t0 = time.time()
        g.residual(DD(y))
        tdd = time.time() - t0
    print("%8g %6d %12.2f %12.2f %12.4e" % (D, f.ks, t64, tdd, u), flush=True)
