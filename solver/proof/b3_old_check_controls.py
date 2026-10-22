"""Mutation controls for the numerical check of H-a1/H-a2 in solver/proof/outer_borel.py (proof/B3_borel.md, Section 6).

outer_borel.py reads the nearest Borel singularities as the poles of a diagonal Pade approximant of the Borel transform
of 36 double-precision coefficients and drops "Froissart doublets", poles with a Pade zero within relative distance
1e-4. Its claim (C_selection.md Section 8): no genuine singularity within 0.48 pi of the positive real axis. A check that
has never been seen to fail is untested (AGENTS.md), so this script feeds that exact procedure (function
borel_pade_poles below, copied from outer_borel.py with only the print replaced by a return value) series with known
singularities:

  P0  the planted conjugate pair a, conj(a) at a = ln(0.8) - i pi/2 (|arg| 0.545 pi), exponent g = 0.5 + 3.82 i
      (u-mode-like), nothing on R+                                    -> the check must pass
  P1  P0 plus a singularity on R+ at 1.2 (inside the pair's modulus 1.59) with weight 0.05   -> must fire
  P2  P0 plus a singularity on R+ at 2.0 (outside it) with weight 0.05  -> does NOT fire: the procedure cannot see an
      R+ singularity behind the nearest pair (a limitation, recorded as the expected outcome)
  P3  a u-mode pair at x = 0.95: a = Phi(0.95) - i pi/2 (|arg| 0.30 pi), the true situation of the M = 1 series at
      x = 0.95 (Section 5) -> the stated criterion (|arg| >= 0.48 pi) must fire, although the 1-sum is not obstructed
each in exact arithmetic and with relative noise 1e-13 (double precision after the recursion's cancellations) added to
the coefficients. It also runs the same procedure on the ball-arithmetic M = 1 series (b3_outer_m1.py), truncated to
36 terms and rounded to double, at x = 0.95, where the nearest genuine singularity has |arg| = 0.31 pi.

    python3 b3_old_check_controls.py [b3_outer_m1_lc_K120.json]
Exits 0 when every control behaves as stated in its line (P2's "no fire" is the documented blindness; the old check's
verdicts on the real series are reported, not tested).
"""
import json
import os
import sys

import mpmath as mp
import numpy as np


def borel_pade_poles(c, nmax=4):
    """copied from solver/proof/outer_borel.py (Borel-Pade, doublet filter 1e-4); returns the kept poles."""
    mp.mp.dps = 30
    b = [mp.mpc(complex(ck)) / mp.factorial(k + 1) for k, ck in enumerate(c)]   # c[0] is order 1
    sc = abs(b[len(b) // 2]) ** (-1.0 / (len(b) // 2)) if abs(b[len(b) // 2]) > 0 else 1
    bs = [bk * mp.mpf(sc) ** k for k, bk in enumerate(b)]                      # rescale zeta -> zeta * sc
    L = (len(bs) - 1) // 2
    p, q = mp.pade(bs, L, len(bs) - 1 - L)
    roots = mp.polyroots(q[::-1], maxsteps=400, extraprec=300) if len(q) > 1 else []
    zeros = mp.polyroots(p[::-1], maxsteps=400, extraprec=300) if len(p) > 1 else []
    good = [r for r in roots if not zeros or min(abs(r - z) for z in zeros) > 1e-4 * max(1, abs(r))]
    nd = len(roots) - len(good)
    good = sorted([complex(r) * float(sc) for r in good], key=abs)[:nmax]
    return good, nd


def planted(sings, K=36, noise=0.0, seed=1):
    """c_{k+1} = k! b_k, B(zeta) = sum K_i (1 - zeta/a_i)^(-g_i); real part taken if the set is conjugation-symmetric"""
    mp.mp.dps = 60
    c = [sum(Ki * mp.rf(gi, k) / ai ** k for Ki, ai, gi in sings) for k in range(K)]
    c = [complex(z) for z in c]
    if noise:
        rng = np.random.default_rng(seed)
        c = [z * (1 + noise * complex(rng.standard_normal(), rng.standard_normal())) for z in c]
    return c


def verdict(poles, crit=0.48):
    """the old criterion: does a kept pole have |arg| < crit pi?"""
    return any(abs(np.angle(r)) < crit * np.pi for r in poles)


def main():
    a = complex(np.log(0.8), -np.pi / 2)
    g = complex(0.5, 3.82)
    pair = [(1, a, g), (1, a.conjugate(), g.conjugate())]
    phi95 = np.log(0.95) - 0.5 * np.log(1 - 0.95 ** 2)
    a95 = complex(phi95, -np.pi / 2)
    cases = [
        ("P0 pair only, nothing on R+", pair, False),
        ("P1 pair + R+ singularity at 1.2 (weight 0.05)", pair + [(0.05, 1.2, 0.7)], True),
        ("P2 pair + R+ singularity at 2.0 (weight 0.05)", pair + [(0.05, 2.0, 0.7)], False),
        ("P3 u-mode pair at x = 0.95 (|arg| 0.30 pi)", [(1, a95, g), (1, a95.conjugate(), g.conjugate())], True),
    ]
    ok = True
    for label, sings, expect_fire in cases:
        for noise in (0.0, 1e-13):
            poles, nd = borel_pade_poles(planted([(Ki, mp.mpc(ai), mp.mpc(gi)) for Ki, ai, gi in sings], noise=noise))
            fired = verdict(poles)
            near = ", ".join(f"|a| {abs(r):.3f} arg {np.angle(r) / np.pi:+.3f}pi" for r in poles)
            tag = "as expected" if fired == expect_fire else "NOT as expected"
            print(f"{label:48s} noise {noise:.0e}: fires {str(fired):5s} ({tag}); {nd} doublets dropped; kept: {near}")
            if noise == 0.0 and fired != expect_fire:
                ok = False
    fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                             "b3_outer_m1_lc_K120.json")
    if os.path.exists(fn):
        d = json.load(open(fn))
        for xs in ("0.95", "0.8"):
            key = [k for k in d["series"] if abs(float(k) - float(xs)) < 1e-12][0]
            c = [complex(float(re), float(im)) for re, im, r in d["series"][key]["B"][:36]]
            poles, nd = borel_pade_poles(c)
            near = ", ".join(f"|a| {abs(r):.3f} arg {np.angle(r) / np.pi:+.3f}pi" for r in poles)
            print(f"real M = 1 series B_k+1 at x = {xs} (ball data, 36 terms, double): fires {verdict(poles)}; "
                  f"{nd} doublets dropped; kept: {near}")
    sys.exit(0 if ok else 1)


main()
