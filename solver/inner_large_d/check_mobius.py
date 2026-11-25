"""mobius_s.py against Track B's form-S series (solver/factored/fseries_s.py on wip/large-d-factored) at a stored
form-S solution: the horizon and centre sums at x* must agree to 1e-13 relative for a = 0 (the same recurrence in w)
and for a = 1 (a different recurrence, the same function). Exit status 1 on failure.

Usage: python3 check_mobius.py FACTORED_DIR SOL.json [--mutate=Lk]
  FACTORED_DIR: a checkout of solver/factored/ from wip/large-d-factored; SOL.json: e.g. Sdd_D100_M8.json of the raw
  folder I1088-mac/. --mutate=Lk drops the factor 2 on c_1 in the resonant operator: the check must FAIL.
"""
import json
import sys

import numpy as np

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
sys.path.insert(0, args[0])
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fseries_s import FactoredS  # noqa: E402
import mobius_s  # noqa: E402

js = json.load(open(args[1]))
M, D = js["M"], js["D"]
y = np.concatenate([js["p1"], js["c0"], js["Q0"], [js["Delta"]]])
y = y + np.concatenate([js.get("p1_lo", [0] * M), js.get("c0_lo", [0] * M), js.get("Q0_lo", [0] * M), [js.get("Delta_lo", 0)]])
ref = FactoredS(D, M)
om = 2 * np.pi / y[3 * M]
hr = ref.horizon(y[2 * M:3 * M], om)
cr = ref.centre(y[:M], y[M:2 * M], om)
ok = True
for a in (0.0, 1.0):
    f = mobius_s.FormS(D, M, a=a, ks=ref.ks, kb=None)
    if opts.get("mutate") == "Lk":
        f.h = f.h.copy()
        f.h[1] = f.h[1] + 0.1 * f.ts  # a wrong coefficient of h: changes both L_k and the recurrence
    ho = f.horizon(y[2 * M:3 * M], om)
    ce = f.centre(y[:M], y[M:2 * M], om)
    dh = max(np.max(np.abs(ho[k] - hr[k])) / np.max(np.abs(hr[k])) for k in ("U", "V", "c"))
    dc = max(np.max(np.abs(ce[k] - cr[k])) / np.max(np.abs(cr[k])) for k in ("U", "V", "c"))
    R = f.residual(y)
    good = dh < 1e-13 and dc < 1e-13
    ok &= good
    print("a %g: horizon sums vs Track B %.2e, centre %.2e, max|F| at the stored solution %.2e  %s"
          % (a, dh, dc, np.max(np.abs(R)), "PASS" if good else "FAIL"))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
