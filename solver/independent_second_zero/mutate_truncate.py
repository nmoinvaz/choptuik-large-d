"""Mutation control for P1 (issue #203): the same zero search and crossing on the planted family, with the continuation
removed (c~_n = 0 for n > 22, the plain partial sum of sigma_n x^n).  It must fail P1's criterion; exits 0 when it does.

    python3 mutate_truncate.py [seed]
"""
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lapl  # noqa: E402
import planted  # noqa: E402
from second_zero import WS, quad_root  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
fam = planted.Family(1.095)
ims = []
for j, w in enumerate(WS):
    mp.mp.dps = 50
    sig = planted.noisy(fam.sigma(mp.mpf(w)), seed + 1000 * j)
    mp.mp.dps = lapl.DPS
    ct = lapl.laplace_coeffs(sig)                      # the 23 data coefficients only
    x1, info = lapl.find_zero(ct, 13.5)
    xt = fam.x1(mp.mpf(w))
    print(f"w {w}: partial-sum zero {mp.nstr(x1, 10)} (converged {info['converged']}), true {mp.nstr(xt, 10)}")
    ims.append(float(x1.imag) if info["converged"] and abs(x1 - 13.5) < 1.5 else float("nan"))
rts = quad_root([float(w) for w in WS], ims) if all(i == i for i in ims) else []
ok_p1 = bool(rts) and abs(rts[0] - 1.095) <= 0.005
print(f"crossing {rts[0] if rts else None}: P1 criterion {'passes (the mutation did not fire)' if ok_p1 else 'fails (fired)'}")
sys.exit(1 if ok_p1 else 0)
