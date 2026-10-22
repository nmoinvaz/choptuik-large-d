#!/usr/bin/env python3
"""Sympy checks of the remarks marked "(remark)" in matter/LITERATURE.md: the l-mode as a radial field in D + 2l, linearized
Yang-Mills as a radial field in D + 2, and the large-d limits of the flat-space blowup profiles (wave maps, Yang-Mills) as
exact solutions of n-free inner equations.

    python3 lit_remarks_check.py
"""
import sympy as sp
r, s, n, l, d = sp.symbols('r s n ell d', positive=True)
T = sp.Symbol('T')  # time (T < 0 before blowup)
D = sp.Symbol('D', positive=True)
g = sp.Function('g')

# 1. Flat space: l-mode with n-sphere == radial with n+2l
f = r**l * g(r)
lap = sp.diff(f, r, 2) + n/r*sp.diff(f, r) - l*(l+n-1)/r**2*f
rad = r**l*(sp.diff(g(r), r, 2) + (n+2*l)/r*sp.diff(g(r), r))
print("l-mode == radial in n+2l:", sp.simplify(lap - rad) == 0)

# linearized magnetic YM in D dims, delta = r^2 g
dl = r**2*g(r)
op = sp.diff(dl, r, 2) + (D-4)/r*sp.diff(dl, r) - 2*(D-3)/r**2*dl
rad2 = r**2*(sp.diff(g(r), r, 2) + D/r*sp.diff(g(r), r))
print("lin YM == radial with D/r:", sp.simplify(op - rad2) == 0)

# 2. Wave maps in d space dims, exact solution u = 2 atan(r/(sqrt(d-2)(-T)))
u = 2*sp.atan(r/(sp.sqrt(d-2)*(-T)))
res = sp.diff(u, T, 2) - (sp.diff(u, r, 2) + (d-1)/r*sp.diff(u, r) - (d-1)*sp.sin(2*u)/(2*r**2))
for dv in [3, 7, 20]:
    print("WM exact residual d=%d:" % dv, sp.N(res.subs({d: dv, r: sp.Rational(3, 2), T: -sp.Rational(1, 3)})))
# inner equation, s = r^2/(2n), n = d-1:  u_tt = u_s - sin(2u)/(4s)
u0 = 2*sp.atan(sp.sqrt(2*s)/(-T))
res0 = sp.diff(u0, T, 2) - (sp.diff(u0, s) - sp.sin(2*u0)/(4*s))
print("WM inner residual:", sp.simplify(sp.expand_trig(res0)))

# 3. YM in d space dims, Glogic/Biernat-Bizon profile
alpha = 2*(1 + sp.sqrt((d-4)/(3*(d-2))))
beta = sp.Rational(1, 3)*(2*d - 8 + sp.sqrt(3*(d-2)*(d-4)))
rho = r/(-T)
w = 1 - r**2*alpha/(rho**2 + beta)/T**2
resY = sp.diff(w, T, 2) - (sp.diff(w, r, 2) + (d-3)/r*sp.diff(w, r) + (d-2)/r**2*w*(1 - w**2))
for dv in [5, 6, 7, 11, 20]:
    print("YM exact residual d=%d:" % dv, sp.N(resY.subs({d: dv, r: sp.Rational(7, 5), T: -sp.Rational(2, 3)})))
# inner: s = r^2/(2n'), n' = d-3: w_tt = w_s + w(1-w^2)/(2s)
a_inf = 2*(1 + 1/sp.sqrt(3))
c_inf = (2 + sp.sqrt(3))/3
w0 = 1 - a_inf*2*s/(2*s + c_inf*T**2)
res0Y = sp.diff(w0, T, 2) - (sp.diff(w0, s) + w0*(1 - w0**2)/(2*s))
print("YM inner residual:", sp.simplify(res0Y))
print("alpha(inf), beta/d(inf):", sp.N(sp.limit(alpha, d, sp.oo)), sp.N(sp.limit(beta/d, d, sp.oo)))
