#!/usr/bin/env python3
"""The inner heat equation of Proposition 5 is Rozali and Way's leading-order large-D system.

Rozali & Way (arXiv:1807.10283, JHEP 11 (2018) 106), Sec. 2.2, flat space: with r = O(1), tau = sqrt(D) t and
1 - delta_0 the lapse factor, the leading-order equations for the scalar phi_0 are
    d_r phi = r (1-delta)^(-1/2) d_tau[(1-delta)^(-1/2) d_tau phi],     d_r delta = -2 r (d_tau phi)^2 .
With s = r^2/2, c = (1-delta)^(1/2) they read
    phi_s = c^-1 d_tau(c^-1 phi_tau),     c_s = phi_tau^2 / c .
This script checks that Y = 2 phi_tau / c, S = Y_tau / c then satisfy, identically, the inner equations of the README
(Proposition 5 (i); GAMMA.md section 3):
    Y_t = c S,   S_t = c (Y_s + Y^3/4),   c_s = c Y^2 / 4      (t = tau).
Their (r, tau) are our (rho, t) = (r/sqrt(n), t) at leading order (the self-similar scaling relates the two choices).

    python3 rw_map.py
"""
import sympy as sp

s, t = sp.symbols("s t")
phi = sp.Function("phi")(s, t)
c = sp.Function("c")(s, t)
# Rozali-Way, solved for the s-derivatives
phi_s = (sp.diff(phi, t) / c).diff(t) / c
c_s = sp.diff(phi, t) ** 2 / c


def reduce(e):
    """replace every s-derivative of phi and c by the Rozali-Way right-hand sides (and their t-derivatives)"""
    for _ in range(4):
        e = e.subs({sp.Derivative(phi, s, t): sp.diff(phi_s, t), sp.Derivative(phi, t, s): sp.diff(phi_s, t),
                    sp.Derivative(c, s, t): sp.diff(c_s, t), sp.Derivative(c, t, s): sp.diff(c_s, t)})
        e = e.subs({sp.Derivative(phi, s): phi_s, sp.Derivative(c, s): c_s})
    return sp.simplify(e)


Y = 2 * sp.diff(phi, t) / c
S = sp.diff(Y, t) / c
eqs = {
    "Y_t - c S": sp.diff(Y, t) - c * S,
    "c_s - c Y^2/4": sp.diff(c, s) - c * Y ** 2 / 4,
    "S_t - c (Y_s + Y^3/4)": sp.diff(S, t) - c * (sp.diff(Y, s) + Y ** 3 / 4),
}
for name, e in eqs.items():
    print(f"{name:24s} = {reduce(e)}")
