"""The float64 error of the horizon sums at x* at stored solutions, against double-double, for several horizon
variables (a = 0 is Track B's affine one) (#1090). Printed per solution and a: max|U(x*)| in dd and the relative
float64 error of U, V and c at x*.

Usage: python3 f64_vs_dd.py SOL.json [SOL2.json ...] [--as=0,1,8]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddnp import DD  # noqa: E402
import mobius_s as ms  # noqa: E402
import mobius_sdd as msd  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
files = [a for a in sys.argv[1:] if not a.startswith("--")]
As = [float(v) for v in opts.get("as", "0,1,8").split(",")]
for fn in files:
    r = json.load(open(fn))
    D, M = r["D"], r["M"]
    y = np.concatenate([r["p1"], r["c0"], r["Q0"], [r["Delta"]]])
    row = []
    for a in As:
        f = ms.FormS(D, M, a=a, kb=150)
        g = msd.FormSDD(D, M, a=a, kb=150)
        with np.errstate(all="ignore"):
            h64 = f.horizon(y[2 * M:3 * M], 2 * np.pi / y[3 * M])
        yd = DD(y)
        hd = g.horizon(yd[2 * M:3 * M], (g.pi * 2.0) / yd[3 * M])
        err = max(float(np.max(np.abs(h64[k] - hd[k].f())) / np.max(np.abs(hd[k].f()))) for k in ("U", "V", "c"))
        row.append("a %g: |U*| %.4e err %.1e" % (a, float(np.max(np.abs(hd["U"].f()))), err))
    print("D %7g M %d  %s" % (D, M, "   ".join(row)), flush=True)
