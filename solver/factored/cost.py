"""Cost of one residual evaluation of form S against D, full and banded horizon recurrence (#1088).

The D-100 solution (M 8) rescaled to each D (p1, m_h by sqrt(n0/n), c0 - 1 by n0/n) is evaluated once in float64
(fseries_s.py) and once in double-double (fseries_sdd.py), with the full recurrence (ks^2 work) where affordable and
the banded one (kb = 150, ks kb work). Printed: ks, seconds, the horizon tail and, where both run, the largest
difference between the banded and the full residual. The control --mutate=kb10 bands at 10 orders, inside the slow
part, and must show a difference far above rounding.

Usage: python3 cost.py SOL.json [--Ds=100,245,400,1000,2000,10000] [--dd_max=2000] [--full_max=1000] [--kb=150]
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ddnp import DD  # noqa: E402
from fseries_s import FactoredS  # noqa: E402
from fseries_sdd import FactoredSDD  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
fn = [a for a in sys.argv[1:] if not a.startswith("--")][0]
Ds = [float(v) for v in opts.get("Ds", "100,245,400,1000,2000,10000").split(",")]
dd_max = float(opts.get("dd_max", 2000))
full_max = float(opts.get("full_max", 1000))
kb = 10 if opts.get("mutate") == "kb10" else int(opts.get("kb", 150))
js = json.load(open(fn))
M = js["M"]
D0 = js["D"]
y0 = np.concatenate([js["p1"], js["c0"], js["Q0"], [js["Delta"]]])
print("%8s %6s %10s %10s %10s %10s %12s %12s" % ("D", "ks", "f64 full", "f64 band", "dd band", "dd full", "|band-full|", "horizon tail"))
worst = 0.0
for D in Ds:
    n, n0 = D - 2, D0 - 2
    y = y0.copy()
    s = np.sqrt(n0 / n)
    y[0:M] *= s
    y[2 * M:3 * M] *= s
    y[M:2 * M] = 1 + (y[M:2 * M] - 1) * n0 / n
    y[3 * M] = 1.6439 + 10.37 / D - 49.17 / D**2
    row = {}
    fb = FactoredS(D, M, kb=kb)
    t = time.time()
    with np.errstate(all="ignore"):
        Rb = fb.residual(y)
    row["f64b"] = time.time() - t
    tail = fb.last[1]["tail"]
    Rf = None
    if D <= full_max:
        ff = FactoredS(D, M)
        t = time.time()
        with np.errstate(all="ignore"):
            Rf = ff.residual(y)
        row["f64f"] = time.time() - t
    if D <= dd_max:
        g = FactoredSDD(D, M, kb=kb)
        t = time.time()
        g.residual(DD(y))
        row["ddb"] = time.time() - t
        if D <= min(dd_max, 400):
            g2 = FactoredSDD(D, M)
            t = time.time()
            g2.residual(DD(y))
            row["ddf"] = time.time() - t
    diff = float(np.max(np.abs(Rb - Rf))) if Rf is not None else float("nan")
    if Rf is not None:
        worst = max(worst, diff / max(1e-300, float(np.max(np.abs(Rf)))))
    print("%8.1f %6d %10s %10.2f %10s %10s %12.2e %12.2e" % (D, fb.ks, "%.2f" % row["f64f"] if "f64f" in row else "-",
                                                         row["f64b"], "%.2f" % row["ddb"] if "ddb" in row else "-",
                                                         "%.2f" % row["ddf"] if "ddf" in row else "-", diff, tail), flush=True)
print("largest |banded - full| / |residual|: %.2e" % worst)
ok = worst < 1e-10
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
