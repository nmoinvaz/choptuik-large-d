"""Independent derivation of the self-similar isothermal Euler-Poisson ODEs (G = 1), compared with GHJ (1.9)-(1.10).
rho = beta*R(y)/t^2, u = sqrt(k)*U(y), y = r/(sqrt(k)*(-t)), m = 4*pi*beta*k^(3/2)*(-t)*M(y), M' = y^2 R."""
import sympy as sp
y, k, tt, beta = sp.symbols('y k t beta', positive=True)  # tt = -t > 0
R, U, M = sp.Function('R')(y), sp.Function('U')(y), sp.Function('M')(y)
r = sp.Symbol('r', positive=True)

Y = r / (sp.sqrt(k) * tt)
rho = beta * R.subs(y, Y) / tt**2
u = sp.sqrt(k) * U.subs(y, Y)
m = 4 * sp.pi * beta * k**sp.Rational(3, 2) * tt * M.subs(y, Y)
# d/dt = -d/dtt
cont = -sp.diff(rho, tt) + sp.diff(rho * u, r) + 2 * rho * u / r
mom = -sp.diff(u, tt) + u * sp.diff(u, r) + k * sp.diff(rho, r) / rho + m / r**2
cont = sp.simplify((cont * tt**3 / beta).subs(r, y * sp.sqrt(k) * tt))
mom = sp.simplify((mom * tt / sp.sqrt(k)).subs(r, y * sp.sqrt(k) * tt))
print('cont:', sp.simplify(cont)); print('mom :', sp.simplify(mom))
# mass from continuity: d/dy[y^2 R (y + U)] = 3 y^2 R ... check which combination gives M
Rp, Up = sp.symbols('Rp Up'); Rs, Us, Ms = sp.symbols('Rs Us Ms')
c2 = cont.subs({sp.Derivative(R, y): Rp, sp.Derivative(U, y): Up}).subs({R: Rs, U: Us})
m2 = mom.subs({sp.Derivative(R, y): Rp, sp.Derivative(U, y): Up}).subs({R: Rs, U: Us, M: Ms})
# self-similar mass identity: M = y^2 R (y + U) / 3 ? test: dM/dy = y^2 R must follow from continuity
Mguess = y**2 * Rs * (y + Us)
dM = sp.diff(Mguess, y) + sp.diff(Mguess, Rs) * Rp + sp.diff(Mguess, Us) * Up
lam = sp.symbols('lam')
sol_l = sp.solve(sp.expand(dM - lam * y**2 * Rs - sp.Symbol('mu') * c2 * y**2), [lam, sp.Symbol('mu')], dict=True)
print('mass identity: d/dy[y^2 R (y+U)] =', sol_l)
