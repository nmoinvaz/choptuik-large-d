#!/usr/bin/env python3
"""D = 3 (n = 1) checks for notes/LOW_D.md: what the field equations become at n = 1, and why the D -> 3 limit of the
DSS family cannot be a regular solution.

1. Code check: in DSS.rhs (solver/dss.py) at n = 1, c_x = 0 for every a, so c = c(tau, 1) = 1 (the gauge), and the
   p, m right-hand sides do not depend on a: p and m obey the free (flat 2+1) wave equation. Checked on random states;
   at n = 1.01 the dependence on a must appear (control).
2. The n = 1 case of Proposition 2's argument (README), at 40 digits with mpmath: for one harmonic e^{s tau},
   s = i mu omega, the solution regular at the centre is p - m = x 2F1((1+s)/2, (2+s)/2; 1; x^2), p + m =
   [x(1 - x^2)(p - m)' - (1 + s x^2)(p - m)] / (s x); it satisfies both first-order equations; Gauss's connection formula
   at kappa = 1 holds numerically near the light cone, with the coefficient of (1 - x^2)^(-1/2 - s)
   C(1, s) = Gamma(s + 1/2) / (Gamma((1+s)/2) Gamma(1 + s/2)), nonzero for every omega != 0 and odd mu. Control: a wrong
   kappa makes the first-order residual large.
Exits non-zero if a check fails.

    python3 lowD_n1_check.py
"""
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss import DSS  # noqa: E402

ok = True


def code_check(n):
    S = DSS(n=n, N=32)
    rng = np.random.default_rng(1)
    x = 0.37
    p, m = rng.standard_normal((2, 1, 32))
    c = np.ones((1, 32))
    om = np.array([1.8])
    a1, a2 = 1 + 0.3 * rng.random((1, 32)), 1 + 0.9 * rng.random((1, 32))
    r1 = S.rhs(x, np.stack([p, m, a1, c]), om)
    r2 = S.rhs(x, np.stack([p, m, a2, c]), om)
    return float(np.abs(r1[0] - r2[0]).max() + np.abs(r1[1] - r2[1]).max()), float(np.abs(r1[3]).max())


d1, cx1 = code_check(1.0)
d2, cx2 = code_check(1.01)
print(f"1. n = 1: p, m right-hand sides change by {d1:.1e} when a changes; c_x = {cx1:.1e}")
print(f"   control n = 1.01: change {d2:.1e}, c_x up to {cx2:.1e}  ({'fires' if d2 > 1e-6 else 'DID NOT FIRE'})")
ok &= d1 == 0.0 and cx1 == 0.0 and d2 > 1e-6

mp.mp.dps = 40


def first_order_residual(n, s, kappa, x):
    al, be = (1 + s) / 2, (2 + s) / 2
    y = lambda t: t * mp.hyp2f1(al, be, kappa, t * t)
    yp = lambda t: mp.diff(y, t)
    S = lambda t: (t * (1 - t * t) * yp(t) - (1 + s * t * t) * y(t)) / (t * (s - n + 1))
    Sp = mp.diff(S, x)
    # sum:  x y' - y = x^2 S' + s x S ;  difference: (n - 1) S = x^2 y' - x S' + s x y
    e1 = x * yp(x) - y(x) - x * x * Sp - s * x * S(x)
    e2 = (n - 1) * S(x) - (x * x * yp(x) - x * Sp + s * x * y(x))
    return max(abs(e1), abs(e2)) / max(abs(y(x)), 1)


worst = mp.mpf(0)
for mu in (1, 3, 5):
    for om in (1.0, 1.82, 2.5, 3.82):
        s = 1j * mu * mp.mpf(om)
        for x in (mp.mpf("0.2"), mp.mpf("0.5"), mp.mpf("0.8")):
            worst = max(worst, first_order_residual(1, s, mp.mpf(1), x))
ctrl = first_order_residual(1, 1j * mp.mpf(1.82), mp.mpf("1.05"), mp.mpf("0.5"))
print(f"2. n = 1, kappa = 1: first-order residual of the hypergeometric solution <= {mp.nstr(worst, 3)} "
      f"(mu = 1, 3, 5; omega 1-3.82; x = 0.2, 0.5, 0.8); control kappa = 1.05: {mp.nstr(ctrl, 3)} "
      f"({'fires' if ctrl > 1e-10 else 'DID NOT FIRE'})")
ok &= worst < 1e-25 and ctrl > 1e-10

# connection formula at kappa = 1, z near 1
worst_c = mp.mpf(0)
cmin = mp.inf
for mu in (1, 3, 5):
    for om in (1.0, 1.82, 2.5, 3.82):
        s = 1j * mu * mp.mpf(om)
        al, be, ka = (1 + s) / 2, (2 + s) / 2, mp.mpf(1)
        C = mp.gamma(ka) * mp.gamma(al + be - ka) / (mp.gamma(al) * mp.gamma(be))
        A = mp.gamma(ka) * mp.gamma(ka - al - be) / (mp.gamma(ka - al) * mp.gamma(ka - be))
        cmin = min(cmin, abs(C))
        for z in (mp.mpf("0.9"), mp.mpf("0.99")):
            lhs = mp.hyp2f1(al, be, ka, z)
            rhs = A * mp.hyp2f1(al, be, al + be - ka + 1, 1 - z) + (1 - z) ** (ka - al - be) * C * mp.hyp2f1(ka - al, ka - be, ka - al - be + 1, 1 - z)
            worst_c = max(worst_c, abs(lhs - rhs) / abs(lhs))
        if mu == 1 and om == 1.82:
            print(f"   C(1, s) at mu = 1, omega = 1.82: {mp.nstr(C, 8)} (|C| = {mp.nstr(abs(C), 6)})")
print(f"3. connection formula at kappa = 1: relative error <= {mp.nstr(worst_c, 3)}; min |C(1, s)| over the grid "
      f"{mp.nstr(cmin, 4)} (nonzero)")
ok &= worst_c < 1e-25 and cmin > 0
print("all checks pass" if ok else "A CHECK FAILED")
sys.exit(0 if ok else 1)
