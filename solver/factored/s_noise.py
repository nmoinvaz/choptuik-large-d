"""Rounding amplification of form S's residual at its solutions, against D (#1088).

Every series coefficient U_k, V_k (centre and horizon) is multiplied by (1 + eta r), r standard normal, in float64, and
the change of the residual is divided by eta. At a solution the series carry no shadow, so the amplification measures
how the layout itself treats rounding; form A's a-datum column grows like 2^(n/2) (anchor_noise.py).
PASS when the amplification at the largest D is within a factor 100 of the smallest D's (no exponential growth over
the range). The control --mutate=rough puts absolute noise of the order-0 term's size on every horizon order, the
rough perturbation that the scalar's horizon map amplifies by about 2^(n/2) near the order n/2: it must FAIL.

Usage: python3 s_noise.py SOL1.json [SOL2.json ...] [--eta=1e-12] [--reps=4] [--mutate=rough]
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fseries_s import FactoredS  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
files = [a for a in sys.argv[1:] if not a.startswith("--")]
eta = float(opts.get("eta", 1e-12))
reps = int(opts.get("reps", 4))
mut = opts.get("mutate", "")
rows = []
print("%9s %4s %5s %10s %12s %12s %12s" % ("D", "M", "ks", "max|F|", "amp centre", "amp horizon", "amp both"))
for fn in sorted(files, key=lambda f: json.load(open(f))["D"]):
    js = json.load(open(fn))
    D, M = js["D"], js["M"]
    y = np.concatenate([js["p1"], js["c0"], js["Q0"], [js["Delta"]]])
    if "p1_lo" in js:
        y = y + np.concatenate([js["p1_lo"], js["c0_lo"], js["Q0_lo"], [js["Delta_lo"]]])
    f = FactoredS(D, M)
    f.noise_abs = mut == "rough"
    R0 = f.residual(y)
    amps = []
    for where in ("centre", "horizon", "both"):
        f.noise = eta
        f.noise_where = where
        dr = max(np.max(np.abs(f.residual(y) - R0)) for _ in range(reps))
        f.noise = 0.0
        amps.append(dr / eta)
    rows.append((D, amps))
    print("%9.3f %4d %5d %10.2e %12.3e %12.3e %12.3e" % (D, M, f.ks, np.max(np.abs(R0)), amps[0], amps[1], amps[2]), flush=True)
if len(rows) >= 2:
    ratio = rows[-1][1][2] / rows[0][1][2]
    print("amplification at D %.1f over D %.1f: %.2e" % (rows[-1][0], rows[0][0], ratio))
    ok = ratio < 100
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
