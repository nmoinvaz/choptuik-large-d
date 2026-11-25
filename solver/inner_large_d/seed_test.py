"""The horizon series' numerics at large D without a solve: a stored solution rescaled to each D (p1, Q0 by
sqrt(n0/n), c0 - 1 by n0/n, Delta from Track C at that D), then per Moebius parameter a: max|U(x*)| in float64, its
float64 error against double-double (where --Ddd lists the D), and the amplification of relative noise on U, V
(#1090). The seeds are not solutions; this measures the recurrence's response there, as noise_variable.py does.

Usage: python3 seed_test.py SOL.json TRACKC.json [--Ds=1000,2000,5000,10000] [--as=1,8] [--Ddd=1000,2000] [--a_rule=400]
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
fn, tcf = [a for a in sys.argv[1:] if not a.startswith("--")][:2]
r = json.load(open(fn))
tc = json.load(open(tcf))["values"]
D0, M = r["D"], r["M"]
y0 = np.concatenate([r["p1"], r["c0"], r["Q0"], [r["Delta"]]])
As = [float(v) for v in opts.get("as", "1,8").split(",")]
arule = float(opts.get("a_rule", 0))  # with --a_rule=N the last a is max(a, n/N)
Ddd = [float(v) for v in opts.get("Ddd", "1000,2000").split(",") if v]
print("seed from %s (D %g, M %d)" % (os.path.basename(fn), D0, M))
for D in [float(v) for v in opts.get("Ds", "1000,2000,5000,10000").split(",")]:
    n, n0 = D - 2, D0 - 2
    y = y0.copy()
    s = np.sqrt(n0 / n)
    y[:M] *= s
    y[2 * M:3 * M] *= s
    y[M:2 * M] = 1 + (y[M:2 * M] - 1) * n0 / n
    y[3 * M] = float(tc["%g" % D]["Delta"])
    om = 2 * np.pi / y[3 * M]
    row = []
    for ia, a in enumerate(As):
        if arule and ia == len(As) - 1:
            a = max(a, n / arule)
        t0 = time.time()
        f = ms.FormS(D, M, a=a, kb=150)
        with np.errstate(all="ignore"):
            base = f.horizon(y[2 * M:3 * M], om)["U"]
            f.noise = (["U", "V"], 1e-13, np.random.default_rng(7))
            dn = max(np.max(np.abs(f.horizon(y[2 * M:3 * M], om)["U"] - base)) for _ in range(2))
            f.noise = None
        u = float(np.max(np.abs(base)))
        amp = dn / 1e-13 / u if u > 0 else float("nan")
        err = ""
        if D in Ddd:
            g = msd.FormSDD(D, M, a=a, kb=150)
            yd = DD(y)
            hd = g.horizon(yd[2 * M:3 * M], (g.pi * 2.0) / yd[3 * M])
            ud = float(np.max(np.abs(hd["U"].f())))
            err = " err vs dd %.1e (dd |U*| %.4e)" % (float(np.max(np.abs(base - hd["U"].f()))) / ud, ud)
        row.append("a %g: |U*| %.4e amp %.1e%s (%.0f s)" % (a, u, amp, err, time.time() - t0))
    print("D %7g ks %6d  %s" % (D, ms.FormS(D, M, a=1.0).ks, "   ".join(row)), flush=True)
