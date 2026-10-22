#!/usr/bin/env python3
"""The singular CSS solution U = pi/2 (the self-similar global-monopole cone) and its linear modes (sympy + numbers).

Background: U = pi/2, c = 1, a^2 = a_e^2 = (n-1)/(n-1-2 eta) (constant; conical deficit 2 eta/(n-1)); it solves the
CSS system of sigma_css.py exactly.  Linearised (sigma_css.lin_res with U' = 0): the metric perturbations decouple
(al = 0 by regularity, ga = 0 by the gauge) and
    (1 - x^2) v'' + [n/x - 2(lam+1) x] v' - lam(lam+1) v + n a_e^2 v / x^2 = 0 .
v = x^beta with lam = -beta is an exact solution, analytic at the SSH, for either root of
    beta^2 + (n-1) beta + n a_e^2 = 0 .
The roots are complex (so the excited CSS solutions f_k accumulate at U = pi/2, as in flat space for d <= 6) iff
    (n-1)^2 < 4 n a_e^2   <=>   n^2 - (6 + 2 eta) n + (1 + 2 eta) < 0 ,
and then Re lam = (n-1)/2.  At the edge the double root gives lam = (n_c - 1)/2.

    python3 sigma_equator.py
"""
import math
import sympy as sp
import numpy as np

x, n, eta, lam, beta = sp.symbols('x n eta lam beta', positive=True)
ae2 = (n - 1) / (n - 1 - 2 * eta)
v = x ** beta
ode = (1 - x ** 2) * sp.diff(v, x, 2) + (n / x - 2 * (lam + 1) * x) * sp.diff(v, x) - lam * (lam + 1) * v + n * ae2 * v / x ** 2
res = sp.simplify((ode / x ** (beta - 2)).subs(lam, -beta))
ind = sp.expand(beta ** 2 + (n - 1) * beta + n * ae2)
print("[%s] v = x^beta, lam = -beta: residual / x^(beta-2) = beta^2 + (n-1) beta + n a_e^2 (the indicial polynomial)"
      % ("ok" if sp.simplify(res - ind) == 0 else "FAIL"))
# check that the equator equation is lin_res of sigma_css at U = pi/2, C = 1, a^2 = a_e^2
from sigma_css import lin_res
xx, ll, nn, ee = sp.symbols('xx ll nn ee', positive=True)
V = sp.Function('V')(xx)
A2 = (nn - 1) / (nn - 1 - 2 * ee)
RW, RH, RS, RM = lin_res(xx, sp.pi / 2, 0, 0, 1, 0, A2, V, sp.diff(V, xx), sp.diff(V, xx, 2), 0, 0, 0, 0, ll, nn, ee)
eq = (1 - xx ** 2) * sp.diff(V, xx, 2) + (nn / xx - 2 * (ll + 1) * xx) * sp.diff(V, xx) - ll * (ll + 1) * V + nn * A2 * V / xx ** 2
print("[%s] lin_res(U = pi/2) = -x^2 * (equator equation)" % ("ok" if sp.simplify(RW + xx ** 2 * eq) == 0 else "FAIL"))
# the background itself: bg_res at U = pi/2, C = 1, a^2 = a_e^2
from sigma_css import bg_res
RWb, RSb, a2b = bg_res(xx, sp.pi / 2, 0, 0, 1, 0, nn, ee)
print("[%s] U = pi/2, c = 1 solves the CSS system, a^2 = (n-1)/(n-1-2eta)" % ("ok" if sp.simplify(RWb) == 0 and sp.simplify(RSb) == 0
                                                                           and sp.simplify(a2b - A2) == 0 else "FAIL"))
print()
print("Edge of the oscillatory regime, n_c(eta), D_c = n_c + 2, and lam = (n_c - 1)/2 there:")
for e in (0.0, 0.07, 0.1, 0.14, 0.17, 0.3, 0.5, 1.0, 2.0):
    nc = ((6 + 2 * e) + math.sqrt((6 + 2 * e) ** 2 - 4 * (1 + 2 * e))) / 2
    print("   eta = %-5g  D_c = %.4f   lam = %.4f   gamma = %.4f" % (e, nc + 2, (nc - 1) / 2, 2 / (nc - 1)))
print("Same at fixed eta_hat = eta/(n-1):  (1-2eh) n^2 - (6-4eh) n + (1-2eh) = 0")
for eh in (0.0, 0.07, 0.1, 0.14, 0.2, 0.3, 0.4, 0.45):
    a, b, c = 1 - 2 * eh, -(6 - 4 * eh), 1 - 2 * eh
    nc = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    print("   eta_hat = %-5g D_c = %.4f   lam = %.4f   gamma = %.4f" % (eh, nc + 2, (nc - 1) / 2, 2 / (nc - 1)))
