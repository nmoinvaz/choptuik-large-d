#!/usr/bin/env python3
"""Review check R2 (agent RV): the kernel and inverse-operator bounds of fixed_point.py against brute force.

  * |exp(1/(2t) - 1/(2z))| = exp(-a (1/|t| - 1/|z|)), a = -cos(theta)/2, for t, z on the ray arg = theta, |t| <= |z|:
    checked at random points on the central rays theta = +-3pi/4 and on the edge rays of the --sector=0.2 option, with
    the code's sector rate a = (sqrt 2/4)(cos 2d - sin 2d) a lower bound of -cos(theta)/2 over the whole sector;
  * the fast-operator bound int_0^r exp(-a (1/rho - 1/r)) rho^m drho <= r^(m+2)/a (m >= -2) and the slow one
    int_0^r rho^m drho = r^(m+1)/(m+1), by mpmath quadrature, for the exponents the lemma uses (m from -2 to 60) and
    r up to R = 0.01;
  * fixed_point.intbound reproduces the weighted bound sup_{r <= R} r^-k |int| for majorants K t^q (spot checks).

    python3 rv_kernel_bounds.py
"""
import os
import random
import sys

import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fixed_point as FP  # noqa: E402
from flint import arb  # noqa: E402

mp.mp.dps = 30


def kernel_identity(n=2000, seed=3):
    rng = random.Random(seed)
    worst = mp.mpf(0)
    d = mp.mpf("0.2")
    a_sector = mp.sqrt(2) / 4 * (mp.cos(2 * d) - mp.sin(2 * d))
    sector_ok = True
    for _ in range(n):
        th = rng.choice([1, -1]) * (3 * mp.pi / 4 + rng.uniform(-0.4, 0.4))
        r = mp.mpf(rng.uniform(1e-4, 0.01))
        rho = r * mp.mpf(rng.uniform(1e-3, 1))
        z, t = r * mp.expj(th), rho * mp.expj(th)
        lhs = abs(mp.exp(1 / (2 * t) - 1 / (2 * z)))
        a = -mp.cos(th) / 2
        rhs = mp.exp(-a * (1 / rho - 1 / r))
        worst = max(worst, abs(mp.log(lhs) - mp.log(rhs)))
        sector_ok &= a >= a_sector - mp.mpf(10) ** -25 and lhs <= 1
    return worst, sector_ok, a_sector


def op_bounds():
    rows = []
    ok = True
    for a in (mp.sqrt(2) / 4, mp.mpf("0.188")):
        for r in (mp.mpf("0.01"), mp.mpf("0.003")):
            for m in (-2, -1, 0, 1, 5, 20, 39, 40, 60):
                f = lambda x: mp.exp(-a * (1 / x - 1 / r)) * x ** m
                I = mp.quad(f, [0, r / 100, r / 10, r / 2, r])
                b = r ** (m + 2) / a
                ok &= I <= b * (1 + mp.mpf(10) ** -20)      # m = -2 is an equality
                rows.append((float(a), float(r), m, float(I / b)))
            for m in (0, 5, 40, 60):
                I = mp.quad(lambda x: x ** m, mp.linspace(0, r, 41))
                ok &= abs(I - r ** (m + 1) / (m + 1)) < mp.mpf(10) ** -10 * r ** (m + 1)   # exact calculus; quadrature to 1e-13
    return ok, max(x[3] for x in rows), min(x[3] for x in rows)


def intbound_spot():
    """intbound(K t^q, kind, k) against the brute-force sup over a grid of r <= R of r^-k |int_0^r ...|"""
    from flint import ctx
    ctx.prec = 128
    FP.Maj.R = arb("0.01")
    ok = True
    a = mp.sqrt(2) / 4
    for q, kind, k in ((39, "slow", 40), (52, "slow", 40), (38, "fast", 40), (51, "fast", 40), (40, "slow", 41)):
        b = FP.intbound(FP.Maj(arb(1), q), kind, k)
        sup = mp.mpf(0)
        for r in [mp.mpf("0.01") * (j + 1) / 40 for j in range(40)]:
            if kind == "slow":
                I = r ** (q + 1) / (q + 1)
            else:
                I = mp.quad(lambda x: mp.exp(-a * (1 / x - 1 / r)) * x ** q, [0, r / 10, r])
            sup = max(sup, I / r ** k)
        ok &= sup <= mp.mpf(b.upper().str(30, radius=False)) * (1 + mp.mpf(10) ** -20)   # attained at r = R
    return ok


if __name__ == "__main__":
    w, sok, asec = kernel_identity()
    print(f"kernel |exp(1/(2t) - 1/(2z))| = exp(-a(1/|t| - 1/|z|)) on rays |arg z -+ 3pi/4| <= 0.4: max log error "
          f"{float(w):.1e}; a >= sector rate {float(asec):.6f} and |kernel| <= 1 everywhere: {sok}")
    ok, hi, lo = op_bounds()
    print(f"operator bounds by quadrature (a = sqrt2/4 and 0.188; r = 0.01, 0.003; m = -2..60): all hold: {ok}; "
          f"ratio integral / bound in [{lo:.3g}, {hi:.3g}]")
    ib = intbound_spot()
    print(f"fixed_point.intbound dominates the brute-force weighted sup (5 spot checks): {ib}")
    print("VERDICT:", "kernel and operator bounds confirmed" if (sok and ok and ib and w < 1e-20) else "PROBLEM")
