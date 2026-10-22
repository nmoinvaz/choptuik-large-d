#!/usr/bin/env python3
"""The D = infinity inner problem of the HE kappa-model as an autonomous PDE in s = r^2/(2n) and t (sympy check).

    python3 css_inner_pde.py

Leading order at r ~ sqrt(n), F = O(1), a = 1 + O(1/n), alpha = c = O(1) (matter/CSS.md section 8): with
d_s = (n/r) d_r, box F = -c^-1 d_t(c^-1 F_t) + F_s + O(1/n) and (grad F)^2 = -F_t^2/c^2 + O(1/n), and the
Hamiltonian + slicing constraints give alpha_r/alpha = (2r/n) h |Pi|^2 + O(1/n^2).  Hence
    F_s = c^-1 d_t(c^-1 F_t) + (2 kappa Fbar/rho) c^-2 F_t^2,      c_s = 2 h |F_t|^2 / c,     rho = 1 - kappa |F|^2, h = rho^-2,
which contains no explicit s or t: s-translation (the lambda = 2 mode of Proposition 5), t-translation, the scaling
(t, s) -> (a t, a^2 s), time reparametrization (with c -> T' c), the U(1) phase and, for kappa > 0, SL(2,R).
This script checks that the CSS ansatz F = (-t)^{i omega} f(xi), c = b(xi), xi = sqrt(2 s)/(-t) = x/sqrt(n)
reduces the PDE exactly to the inner ODE of css_inner.py (which is itself checked there as the n -> infinity limit
of the finite-n ODE).
"""
import sympy as sp

s, T = sp.symbols('s T', positive=True)          # T = -t > 0
om, kap = sp.symbols('omega kappa', positive=True)
xi = sp.Symbol('xi', positive=True)
f, g, b = sp.Function('f'), sp.Function('g'), sp.Function('b')
XI = sp.sqrt(2 * s) / T
F = T ** (sp.I * om) * f(XI)
G = T ** (-sp.I * om) * g(XI)                     # Fbar (g = continuation of fbar)
C = b(XI)
Dt = lambda e: -sp.diff(e, T)                      # d/dt = -d/dT
rho = 1 - kap * F * G
h = 1 / rho ** 2
E1 = sp.diff(F, s) - (Dt(Dt(F) / C) / C + 2 * kap * G / rho * Dt(F) ** 2 / C ** 2)
E2 = sp.diff(C, s) - 2 * h * Dt(F) * Dt(G) / C


def to_xi(e):
    e = e.subs(s, xi ** 2 * T ** 2 / 2)
    return sp.simplify(sp.powsimp(sp.expand(e), force=True))


# inner ODE of css_inner.py (xi^2 f'' = ...,  xi b' = U b)
fx, gx, bx = f(xi), g(xi), b(xi)
q, qg = sp.diff(fx, xi), sp.diff(gx, xi)
rh = 1 - kap * fx * gx
U = 2 * xi ** 2 / rh ** 2 * (sp.I * om * fx - xi * q) * (-sp.I * om * gx - xi * qg) / bx ** 2
ode_f = xi ** 2 * sp.diff(fx, xi, 2) - (q * (bx ** 2 / xi - xi * (2 - 2 * sp.I * om - U - 4 * sp.I * kap * om * fx * gx / rh))
                                        + fx * (om ** 2 + sp.I * om * (1 - U)) - 2 * kap * gx / rh * (xi ** 2 * q ** 2 - om ** 2 * fx ** 2))
ode_b = xi * sp.diff(bx, xi) - U * bx

# eliminate b' with the b-equation (xi b' = U b) and compare numerically at random points
bp = sp.Symbol('bp'); fpp = sp.Symbol('fpp')
r1 = E1.subs(s, xi ** 2 * T ** 2 / 2).doit()
r2 = E2.subs(s, xi ** 2 * T ** 2 / 2).doit()
import random
random.seed(1)
def numeric(expr, vals):
    e = expr
    for k_, v_ in vals.items():
        e = e.subs(k_, v_)
    return complex(sp.N(e))
ok = True
for trial in range(3):
    vals = {}
    xv, Tv, ov, kv = [sp.Float(random.uniform(0.2, 1.5)) for _ in range(4)]
    if trial == 0:
        kv = sp.Float(0)
    fv, gv, qv, qgv, bv, fppv, gppv = [sp.Float(random.uniform(-1, 1)) + sp.I * sp.Float(random.uniform(-1, 1)) for _ in range(7)]
    bv = sp.Float(random.uniform(0.8, 1.6))
    Uv = (2 * xv ** 2 / (1 - kv * fv * gv) ** 2 * (sp.I * ov * fv - xv * qv) * (-sp.I * ov * gv - xv * qgv) / bv ** 2)
    bpv = Uv * bv / xv                                      # b' from the b-equation
    rep = {sp.Derivative(f(xi), (xi, 2)): fppv, sp.Derivative(g(xi), (xi, 2)): gppv,
           sp.Derivative(f(xi), xi): qv, sp.Derivative(g(xi), xi): qgv, sp.Derivative(b(xi), xi): bpv,
           f(xi): fv, g(xi): gv, b(xi): bv}
    def ev(e):
        e = e.subs(rep).subs({xi: xv, T: Tv, om: ov, kap: kv})
        return complex(sp.N(e))
    v1 = ev(r1); v2 = ev(r2.subs(rep).subs(sp.Derivative(b(xi), xi), bpv))
    vf = ev(ode_f)
    # expected: E1 = -T^(i om - 2) * ode_f / b^2
    fac = complex(sp.N((-Tv ** (sp.I * ov - 2) / bv ** 2)))
    print("trial %d: E1 / (-T^(i om-2) ODE_f / b^2) = %s ;  E2 (b' from the b-eq.) = %.1e" % (trial, v1 / (fac * vf), abs(v2)))
    ok &= abs(v1 / (fac * vf) - 1) < 1e-10 and abs(v2) < 1e-10
print("inner PDE reduces to the inner ODE:", ok)
