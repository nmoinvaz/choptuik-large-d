"""mobius_sdd.py against the element-wise mpmath reference (mobius_mp.py, 40 digits) and against float64, at a stored
form-S solution: horizon sums at x* within 1e-29 of the reference, the residual at the stored solution below 1e-19
(it solves Track B's dd problem, whose c recurrence rounds lam/(2(k+1)) to float64: printed for information), and the
forward-mode Jacobian within 1e-12 of the float64 complex-step one. Exit status 1 on failure.

Usage: python3 check_mobius_dd.py FACTORED_DIR SOL.json [--kb=150]
"""
import json
import sys
import time

import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
kb = int(opts["kb"]) if "kb" in opts else None
sys.path.insert(0, args[0])
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fseries_sdd import FactoredSDD  # noqa: E402
import ddnp as d  # noqa: E402
from ddnp import DD  # noqa: E402
import mobius_s  # noqa: E402
import mobius_sdd  # noqa: E402
import mpmath as mp  # noqa: E402
from mobius_mp import make  # noqa: E402

js = json.load(open(args[1]))
M, D = js["M"], js["D"]
yh = np.concatenate([js["p1"], js["c0"], js["Q0"], [js["Delta"]]])
yl = np.concatenate([js["p1_lo"], js["c0_lo"], js["Q0_lo"], [js["Delta_lo"]]])
y = DD(yh, yl)
t0 = time.time()
ref = FactoredSDD(D, M, kb=kb)
om = (ref.pi * 2.0) / y[3 * M]
hr = ref.horizon(y[2 * M:3 * M], om)
f = mobius_sdd.FormSDD(D, M, a=1, ks=ref.ks, kb=kb)
ho = f.horizon(y[2 * M:3 * M], om)
dh = max(float(np.max(np.abs((ho[k] - hr[k]).f()))) / float(np.max(np.abs(hr[k].f()))) for k in ("U", "V", "c"))
mp.mp.dps = 40
Q0m = np.array([mp.mpf(float(u)) + mp.mpf(float(v)) for u, v in zip(yh[2 * M:3 * M], yl[2 * M:3 * M])], dtype=object)
omm = 2 * mp.pi / (mp.mpf(float(yh[3 * M])) + mp.mpf(float(yl[3 * M])))
o = make(D, M, 1, backend="mp", ks=ref.ks)
hm = o.horizon(Q0m, omm)
dm = 0.0
for k in ("U", "V", "c"):
    mine = [mp.mpf(float(u)) + mp.mpf(float(v)) for u, v in zip(ho[k].h, ho[k].l)]
    scale = max(abs(v) for v in hm[k])
    dm = max(dm, float(max(abs(mine[i] - hm[k][i]) for i in range(M)) / scale))
print("dd horizon sums vs the mpmath reference (40 digits): %.2e relative" % dm)
R = f.residual(y)
nR = float(np.max(np.abs(R.f())))
print("information: dd horizon sums (a = 1) vs Track B's dd (a = 0, its float64 constant): %.2e relative; max|F| at the stored dd solution %.2e (%.0f s)"
      % (dh, nR, time.time() - t0))
J = f.jacobian(y)
g = mobius_s.FormS(D, M, a=1.0, ks=ref.ks, kb=kb)
J64 = g.jacobian(yh + yl)
dJ = float(np.max(np.abs(J.f() - J64)) / np.max(np.abs(J64)))
print("forward-mode dd Jacobian vs float64 complex step: %.2e relative (%.0f s)" % (dJ, time.time() - t0))
ok = dm < 1e-29 and nR < 1e-19 and dJ < 1e-12
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
