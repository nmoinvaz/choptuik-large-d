"""Where the affine horizon series loses its digits: float64 against double-double, order by order (#1090).

At a seed (Track B's D-100 form-S solution rescaled to D, as in noise_variable.py) the horizon recurrence is run in
float64 (mobius_s.py) and in double-double (mobius_sdd.py), in the affine variable (a = 0, Track B's w) and the Moebius
variable (a = 1), both in units where x* is at 1. Printed per order k: |U_k| (dd), the float64 error of U_k, absolute
and relative to |U_k|, for both variables; then the error of the sum at x*.

Usage: python3 affine_orders.py SOL.json [--Ds=100,200,245]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddnp import DD  # noqa: E402
import ddnp as d  # noqa: E402
import mobius_s as ms  # noqa: E402
import mobius_sdd as msd  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
fn = [a for a in sys.argv[1:] if not a.startswith("--")][0]
js = json.load(open(fn))
M, D0 = js["M"], js["D"]
y0h = np.concatenate([js["p1"], js["c0"], js["Q0"], [js["Delta"]]])
y0l = np.concatenate([js["p1_lo"], js["c0_lo"], js["Q0_lo"], [js["Delta_lo"]]])
for D in [float(v) for v in opts.get("Ds", "100,200,245").split(",")]:
    n, n0 = D - 2, D0 - 2
    if abs(D - D0) < 1e-12:
        y = DD(y0h, y0l)
    else:
        yh = y0h.copy()
        s = np.sqrt(n0 / n)
        yh[0:M] *= s
        yh[2 * M:3 * M] *= s
        yh[M:2 * M] = 1 + (yh[M:2 * M] - 1) * n0 / n
        yh[3 * M] = 1.6439 + 10.37 / D - 49.17 / D ** 2
        y = DD(yh)
    res = {}
    for a in (0.0, 1.0):
        f = ms.FormS(D, M, a=a, kb=150)
        g = msd.FormSDD(D, M, a=int(a), kb=150)
        with np.errstate(all="ignore"):
            h64 = f.horizon(y.f()[2 * M:3 * M], 2 * np.pi / y.f()[3 * M], keep=True)
        hd = g.horizon(y[2 * M:3 * M], (g.pi * 2.0) / y[3 * M], keep=True)
        res[a] = (h64, hd, f.ks)
    ks = res[0.0][2]
    print("D %.2f  n %.0f  ks %d: |U_k| (dd) and the float64 error, absolute (relative), affine | Moebius" % (D, n, ks))
    ks_list = sorted(set(list(range(0, int(0.6 * n), max(1, int(n / 25)))) + list(range(int(0.6 * n), ks + 1, max(1, int(n / 8))))))
    for k in ks_list:
        row = []
        for a in (0.0, 1.0):
            h64, hd, _ = res[a]
            ud = hd["series"]["U"][k].f()
            err = np.max(np.abs(h64["series"]["U"][k] - ud))
            row.append("%.1e %.1e (%.1e)" % (np.max(np.abs(ud)), err, err / max(np.max(np.abs(ud)), 1e-300)))
        print("  k %4d  %s  |  %s" % (k, row[0], row[1]))
    for a in (0.0, 1.0):
        h64, hd, _ = res[a]
        e = np.max(np.abs(h64["U"] - hd["U"].f())) / np.max(np.abs(hd["U"].f()))
        print("  a = %g: sum at x*, max|U*| (dd) %.4e, float64 error relative %.1e" % (a, np.max(np.abs(hd["U"].f())), e))
