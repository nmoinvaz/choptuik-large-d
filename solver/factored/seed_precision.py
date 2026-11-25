"""How much precision form S's horizon series needs off the solution, against D (#1088).

The D-100 solution (M 8, dd) rescaled to each D (p1, m_h by sqrt(n0/n), c0 - 1 by n0/n, Delta from the 1/D fit) is a
jump's seed, not a solution there. Its horizon series is summed at x* in float64 (fseries_s.py) and in double-double
(fseries_sdd.py), banded at kb = 150 (identical to the full recurrence at a solution; cost.py). Printed: the two sums of
U, their relative difference, and the dd sum's size against U at the SSH. A difference near 1 means the float64 sum
is rounding; growth of the dd sum with D means the seed's shadow itself grows.
PASS (the measurement's sanity) when the difference at the solution itself (the first D, equal to the source's) is
below 1e-12; the control --mutate=sigma (the SSH exponent without g, in both precisions) must keep that PASS while
moving the sums.

Usage: python3 seed_precision.py SOL.json [--Ds=100,150,200,245,300,400]
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ddnp import DD  # noqa: E402
from fseries_s import FactoredS  # noqa: E402
from fseries_sdd import FactoredSDD  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
fn = [a for a in sys.argv[1:] if not a.startswith("--")][0]
Ds = [float(v) for v in opts.get("Ds", "100,150,200,245,300,400").split(",")]
mut = opts.get("mutate", "")
js = json.load(open(fn))
M, D0 = js["M"], js["D"]
y0h = np.concatenate([js["p1"], js["c0"], js["Q0"], [js["Delta"]]])
y0l = np.concatenate([js.get("p1_lo", [0] * M), js.get("c0_lo", [0] * M), js.get("Q0_lo", [0] * M), [js.get("Delta_lo", 0.0)]])
print("%8s %14s %14s %12s %12s" % ("D", "|U(x*)| f64", "|U(x*)| dd", "rel diff", "dd / |U_0|"))
first = None
for D in Ds:
    n, n0 = D - 2, D0 - 2
    if abs(D - D0) < 1e-12:
        y = DD(y0h, y0l)
    else:
        yh = y0h.copy()
        s = np.sqrt(n0 / n)
        yh[0:M] *= s
        yh[2 * M:3 * M] *= s
        yh[M:2 * M] = 1 + (yh[M:2 * M] - 1) * n0 / n
        yh[3 * M] = 1.6439 + 10.37 / D - 49.17 / D**2
        y = DD(yh)
    f = FactoredS(D, M, kb=150, mutate=mut)
    with np.errstate(all="ignore"):
        h64 = f.horizon(y.f()[2 * M:3 * M], 2 * np.pi / y.f()[3 * M])
    g = FactoredSDD(D, M, kb=150, mutate=mut)
    hd = g.horizon(y[2 * M:3 * M], (g.pi * 2.0) / y[3 * M])
    u64 = h64["U"]
    udd = hd["U"].f()
    rel = float(np.max(np.abs(u64 - udd)) / max(1e-300, np.max(np.abs(udd))))
    U0 = float(np.max(np.abs(h64["series"][0][0])))
    print("%8.1f %14.3e %14.3e %12.2e %12.2e" % (D, np.max(np.abs(u64)), np.max(np.abs(udd)), rel, np.max(np.abs(udd)) / U0), flush=True)
    Us = h64["series"][0]
    orders = [100, int(n / 2), int(3 * n / 4), int(n), min(int(1.25 * n), f.ks), f.ks]
    print("         float64 |U_k| at orders %s: %s" % (orders, " ".join("%.0e" % np.max(np.abs(Us[k])) for k in orders)))
    if first is None:
        first = rel
ok = first is not None and first < 1e-12
print("difference at the first D: %.2e" % first)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
