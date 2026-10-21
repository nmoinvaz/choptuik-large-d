#!/usr/bin/env python3
"""Lemma B2(M) of proof/C_selection.md: the order-n part of the self-similar system, linearized on the slow manifold
(b = 1, m = -p), has eigenvalues c^2/(x(x^2 - c^2)) (the p + m mode; = -Phi'(x) at c = 1), -1/x (the b - 1 mode) and 0, 0
(p - m and c, slow). Theta-derivative terms are O(1) and drop out.   python3 fast_eigenvalues.py"""
import sympy as sp

x, n = sp.symbols('x n', positive=True)
p, m, b, c = sp.symbols('p m b c')
G = (n - 1) * c * (b - 1)
rhs = sp.Matrix([(G * p - c * p + n * c * (p + m) / 2) / (x * (x - c)),
                 (-G * m + c * m - n * c * (p + m) / 2) / (x * (x + c)),
                 (b * (p**2 + m**2) - (n - 1) * b * (b - 1)) / x,
                 ((n - 1) * c * (b - 1)) / x])
J = rhs.jacobian(sp.Matrix([p, m, b, c]))
Jn = (J / n).subs({b: 1, m: -p}).applyfunc(lambda e: sp.limit(e, n, sp.oo))
print("leading Jacobian / n on the slow manifold:")
sp.pprint(sp.simplify(Jn))
print("eigenvalues:", {sp.simplify(k): v for k, v in Jn.eigenvals().items()})
Phi = sp.log(x) - sp.log(1 - x**2) / 2
print("lambda_u(c=1) + Phi'(x) =", sp.simplify((c**2 / (x * (x**2 - c**2))).subs(c, 1) + sp.diff(Phi, x)))
