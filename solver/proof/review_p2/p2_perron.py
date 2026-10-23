#!/usr/bin/env python3
"""P2 review: the contraction constants of Lemma LC (theta = 0.7355, b8m_lc_lip.json, 40 x 40) and Lemma J
(theta = 0.176, Gamma of b8m_lcj.json, 4 x 4; theta0 = 5.8e-5, Gamma0) recomputed in exact rational arithmetic
(fractions.Fraction), independently of b8m_lc.contraction_check / b8m_lcj.perron.

Method: a positive weight vector v is found by power iteration on (M + tiny J) in floats (any v > 0 is admissible);
then theta = max_i (M v)_i / v_i is evaluated exactly with M's entries rounded up (each saved float is an arb upper
end rounded to double; it is inflated by a relative 1e-12, which exceeds one double ulp). theta < 1 proves that the
map with Lipschitz matrix <= M is a contraction in the weighted max-norm max_i |.|_i / v_i. Also printed: the
spectral radius (floats) for comparison, and a lower bound on it (Collatz-Wielandt: min_i (Mv)_i / v_i).

    python3 p2_perron.py [--mutate=scale]    scale: every entry multiplied by 1/0.70 (spectral radius just above 1
                                              for LC): the LC contraction must then fail
"""
import json
import sys
from fractions import Fraction

import numpy as np

from p2_common import PROOF

OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
INFL = Fraction(1) + Fraction(1, 10 ** 12)


def up(x):
    """exact rational upper bound of a saved float upper bound"""
    f = Fraction(float(x))
    return f * INFL if f > 0 else Fraction(0) if f == 0 else f / INFL


def perron(Mf, names, label):
    n = len(names)
    A = np.array([[float(Mf[i][j]) for j in range(n)] for i in range(n)])
    rho = max(abs(np.linalg.eigvals(A)))
    v = np.ones(n)
    B = A + 1e-14 * np.ones((n, n))
    for _ in range(5000):
        w = B @ v
        v = w / w.max()
    v = np.maximum(v, 1e-30)
    V = [Fraction(float(x)) for x in v]
    num = [sum((Mf[i][j] * V[j] for j in range(n)), Fraction(0)) for i in range(n)]
    ratios = [num[i] / V[i] for i in range(n)]
    theta = max(ratios)
    cw_lo = min(ratios)
    imax = max(range(n), key=lambda i: ratios[i])
    print(f"{label}: n = {n}; spectral radius (float) {rho:.6f}; exact theta = max (Mv)_i/v_i = {float(theta):.6f} "
          f"(row {names[imax]}); Collatz-Wielandt lower bound min (Mv)_i/v_i = {float(cw_lo):.3e}")
    return theta, rho


def lc():
    d = json.load(open(f"{PROOF}/b8m_lc_lip.json"))
    names = set()
    ent = {}
    for key, val in d["M"].items():
        r, c = key.split("<-")
        names.add(r)
        names.add(c)
        ent[(r, c)] = up(val)
    names = sorted(names)
    Mf = [[ent.get((r, c), Fraction(0)) for c in names] for r in names]
    if MUT == "scale":
        Mf = [[x * Fraction(100, 70) for x in row] for row in Mf]
    # structural facts used in the review: which rows feed back into the loop
    return perron(Mf, names, "Lemma LC Lipschitz matrix (b8m_lc_lip.json)"), names, Mf


def lcj():
    d = json.load(open(f"{PROOF}/b8m_lcj.json"))
    out = []
    for key, lab in (("Gamma", "Lemma J Gamma (weighted norm)"), ("Gamma0", "Lemma J Gamma0 (a priori norm on (1, 1+d/2])")):
        G = d[key]
        names = list(G.keys())
        Mf = [[up(G[i][j]) for j in names] for i in names]
        if MUT == "scale":
            Mf = [[x * Fraction(100, 70) for x in row] for row in Mf]
        out.append(perron(Mf, names, lab))
    return out


def main():
    (thLC, rhoLC), names, Mf = lc()
    (thJ, rhoJ), (thJ0, rhoJ0) = lcj()
    claims = [("LC theta <= 0.7356", thLC <= Fraction(7356, 10000)), ("J theta <= 0.1760", thJ <= Fraction(1760, 10000)),
              ("J theta0 <= 5.84e-5", thJ0 <= Fraction(584, 10 ** 7))]
    contraction = thLC < 1 and thJ < 1 and thJ0 < 1
    for c, ok in claims:
        print(f"  claim {c}: {ok}")
    print(f"all three contractions (< 1): {contraction}" + (f"   [MUTATION {MUT}]" if MUT else ""))
    if MUT:
        print(f"MUTATION {MUT}: the LC contraction must fail: {'caught' if not thLC < 1 else 'NOT caught'}")
        return 0 if not thLC < 1 else 3
    return 0 if contraction and all(ok for _, ok in claims) else 1


if __name__ == "__main__":
    sys.exit(main())
