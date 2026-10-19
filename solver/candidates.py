"""Candidate closed forms for Delta_inf: search the window, then force each as c0 in 12 fits (README table).

    python3 ../candidates.py            (from solver/data)
"""
import itertools, math
import mpmath as mp
LO, HI = 1.64388, 1.64403
B = {"pi": math.pi, "e": math.e, "sqrt2": 2 ** .5, "sqrt3": 3 ** .5, "ln2": math.log(2), "ln3": math.log(3),
     "gammaE": 0.5772156649015329, "zeta3": 1.2020569031595942, "phi": (1 + 5 ** .5) / 2, "Catalan": 0.915965594177219,
     "j01": float(mp.besseljzero(0, 1)), "j11": float(mp.besseljzero(1, 1)), "j21": float(mp.besseljzero(2, 1)),
     "j02": float(mp.besseljzero(0, 2)), "j12": float(mp.besseljzero(1, 2)), "y01": float(mp.besselyzero(0, 1)),
     "j32": float(mp.besseljzero(1.5, 1)), "j52": float(mp.besseljzero(2.5, 1))}
P = [-2, -1, -0.5, 0.5, 1, 2]
hits = {}
for i in range(1, 13):
    for j in range(1, 13):
        if math.gcd(i, j) != 1:
            continue
        r = i / j
        for (n1, b1), p1 in itertools.product(B.items(), P):
            v = r * b1 ** p1
            if LO <= v <= HI:
                hits[f"{i}/{j} * {n1}^{p1}"] = v
            for (n2, b2), p2 in itertools.product(B.items(), P):
                if n2 <= n1:
                    continue
                w = v * b2 ** p2
                if LO <= w <= HI:
                    hits[f"{i}/{j} * {n1}^{p1} * {n2}^{p2}"] = w
        for (n1, b1) in B.items():                         # 2 pi / (r b) and r * ln(b)
            for v, name in ((2 * math.pi / (r * b1), f"2pi/({i}/{j} * {n1})"), (r * math.log(b1), f"{i}/{j} * ln({n1})")):
                if LO <= v <= HI:
                    hits[name] = v
print(len(hits), "expressions in", LO, HI)
for k, v in sorted(hits.items(), key=lambda kv: len(kv[0])):
    print(f"{v:.7f}  {k}")

import numpy as np  # noqa: E402
from final_fit import step_converged, weighted_series  # noqa: E402

D, y, s, g = step_converged()
C = {"free fit": None, "pi^2/6": math.pi ** 2 / 6, "sqrt(e)": math.e ** .5, "2pi/j11": 2 * math.pi / B["j11"],
     "golden ratio": B["phi"], **hits}
fams = [(lo, xf, m) for lo in (10, 12) for xf in (lambda d: 1 / d, lambda d: 1 / (d - 1), lambda d: 1 / (d - 2)) for m in (9, 10)]
hi = D > 30.2
print(f"\n{'candidate':40s} {'value':>10s}  chi2/dof best .. worst   held-out rms")
for name, v in C.items():
    chis, offs = [], []
    for lo, xf, m in fams:
        k = (D >= lo) & (D <= 30.2)
        c, chi = weighted_series(xf(D[k]), y[k], 1 / s[k], m, v)
        chis.append(chi)
        offs.append(y[hi] - np.polyval(c[::-1], xf(D[hi])))
    print(f"{name:40s} {v if v else float('nan'):10.7f}  {min(chis):8.2f} .. {max(chis):9.2f}   "
          f"{np.sqrt(np.mean(np.array(offs) ** 2)):.1e}")
