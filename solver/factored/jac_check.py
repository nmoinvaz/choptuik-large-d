"""The exact double-double Jacobian of form S against two others, and the banded recurrence against the full one (#1088).

At a solution (a json of solve_dd.py): the dual-number Jacobian (fseries_sdd.py), the float64 complex-step one
(fseries_s.py) and central differences in double-double at h = 2^-36 and 2^-26 (relative). Printed: the largest
column-wise relative differences. Then the banded dd recurrence (kb) against the full one: residual and the Delta column.
PASS when dual and complex step agree to 1e-8 per column and banded and full to 1e-25 relative; the control
--mutate=kb10 bands inside the slow part (kb = 10) and must FAIL.

Usage: python3 jac_check.py SOL.json [--kb=60] [--mutate=kb10]
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import ddnp as d  # noqa: E402
from ddnp import DD  # noqa: E402
from fseries_s import FactoredS  # noqa: E402
from fseries_sdd import FactoredSDD  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
fn = [a for a in sys.argv[1:] if not a.startswith("--")][0]
kb = 10 if opts.get("mutate") == "kb10" else int(opts.get("kb", 60))
js = json.load(open(fn))
M, D = js["M"], js["D"]
y = DD(np.concatenate([js["p1"], js["c0"], js["Q0"], [js["Delta"]]]),
       np.concatenate([js["p1_lo"], js["c0_lo"], js["Q0_lo"], [js["Delta_lo"]]]))
f = FactoredSDD(D, M)
Jd = f.jacobian(y).f()
Jc = FactoredS(D, M).jacobian(y.f())
L = Jd.shape[1]
cs = max(np.linalg.norm(Jd[:, i] - Jc[:, i]) / np.linalg.norm(Jd[:, i]) for i in range(L))
print("D %.4f M %d: dual against complex step, largest column difference %.2e" % (D, M, cs))
for hrel in (2.0**-36, 2.0**-26):
    Jf = f.jacobian_fd(y, hrel=hrel).f()
    fd = max(np.linalg.norm(Jd[:, i] - Jf[:, i]) / np.linalg.norm(Jd[:, i]) for i in range(L))
    print("   dual against central differences at h %.1e (relative): %.2e" % (hrel, fd))
R0 = f.residual(y)
fb = FactoredSDD(D, M, kb=kb)
R1 = fb.residual(y)
e = np.zeros(L)
e[-1] = 1.0
c0 = f.residual(d.Dual(y, DD(e))).t
c1 = fb.residual(d.Dual(y, DD(e))).t
rb = float(np.max(np.abs((R1 - R0).f())) / max(1e-300, np.max(np.abs(c0.f()))))
cb = float(np.linalg.norm((c1 - c0).f()) / np.linalg.norm(c0.f()))
print("banded (kb %d, ks %d) against full: residual %.2e (relative to the Delta column), Delta column %.2e" % (kb, f.ks, rb, cb))
ok = cs < 1e-8 and rb < 1e-25 and cb < 1e-25
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
