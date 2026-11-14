#!/usr/bin/env python3
"""sympy check of css_inner_pert.prhs: the linearized inner ODE against the inner PDE in (s, t) (issue #376).

    python3 css_inner_pert_eqs.py [--mutate=hamsign|nometric]     exit 0 iff every residual vanishes (and, with
                                                                  --mutate, exit 0 iff the mutated rhs is rejected)

The D = infinity inner PDE of matter/CSS.md section 8 (css_inner_pde.py),
    E1 = F_s - [c^-1 d_t(c^-1 F_t) + (2 kappa Fbar/rho) c^-2 F_t^2],   E2 = c_s - 2 h F_t Fbar_t / c,
and E1b (F <-> Fbar), is written for F = T^{i om} (f(xi) + eps T^-lam A(xi)), Fbar = T^{-i om} (g + eps T^-lam B),
c = b + eps T^-lam beta, T = -t, xi = sqrt(2 s)/T, i.e. e^{lam tau} with tau = -ln T.  The eps-derivative at eps = 0
is evaluated at random complex points, with f'', g'', b' and A'', B'', beta' taken from css_inner_pert.prhs: every
residual must vanish.  This shares with css_inner_pert.py only the PDE; the (tau, xi) form, the X operators and the
hand linearization in prhs are what is tested.
"""
import os
import random
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import css_inner_pert as cip  # noqa: E402

mutate = None
for a in sys.argv[1:]:
    if a.startswith("--mutate="):
        mutate = a.split("=", 1)[1]

s, T = sp.symbols('s T', positive=True)
om, kap, eps = sp.symbols('omega kappa epsilon', positive=True)
lam = sp.Symbol('lam')
xi = sp.Symbol('xi', positive=True)
f, g, b, A, B, be = [sp.Function(nm) for nm in ('f', 'g', 'b', 'A', 'B', 'beta')]
XI = sp.sqrt(2 * s) / T
F = T ** (sp.I * om) * (f(XI) + eps * T ** (-lam) * A(XI))
G = T ** (-sp.I * om) * (g(XI) + eps * T ** (-lam) * B(XI))
C = b(XI) + eps * T ** (-lam) * be(XI)
Dt = lambda e: -sp.diff(e, T)
rho = 1 - kap * F * G
h = 1 / rho ** 2
E1 = sp.diff(F, s) - (Dt(Dt(F) / C) / C + 2 * kap * G / rho * Dt(F) ** 2 / C ** 2)
E1b = sp.diff(G, s) - (Dt(Dt(G) / C) / C + 2 * kap * F / rho * Dt(G) ** 2 / C ** 2)
E2 = sp.diff(C, s) - 2 * h * Dt(F) * Dt(G) / C
lin = [sp.diff(E, eps).subs(eps, 0) for E in (E1, E1b, E2)]
lin = [e.subs(s, xi ** 2 * T ** 2 / 2).doit() for e in lin]
bgr = [E.subs(eps, 0).subs(s, xi ** 2 * T ** 2 / 2).doit() for E in (E1, E1b, E2)]

random.seed(7)
ok = True
worst = 0.0
for trial in range(4):
    cr = lambda: complex(random.uniform(-1, 1), random.uniform(-1, 1))
    xv, Tv, ov = random.uniform(0.2, 0.9), random.uniform(0.5, 1.5), random.uniform(1, 4)
    kv = 0.0 if trial == 0 else random.uniform(0.1, 1.0)
    lv = complex(random.uniform(-1, 3), random.uniform(-4, 8))
    fv, gv, qv, qgv = cr() * 0.5, cr() * 0.5, cr(), cr()
    bv = complex(random.uniform(0.9, 1.4), 0.1 * random.uniform(-1, 1))
    pv = [cr() for _ in range(10)]          # (A, A', B, B', beta) for two perturbations
    Y = np.array([fv, gv, qv, qgv, bv] + pv, dtype=complex)
    d = cip.prhs(xv, Y, kv, ov, lv, 2, mutate)
    for k in range(2):
        Av, Apv, Bv, Bpv, bev = pv[5 * k:5 * k + 5]
        App, Bpp, bep = d[5 + 5 * k + 1], d[5 + 5 * k + 3], d[5 + 5 * k + 4]
        rep = {sp.Derivative(f(xi), (xi, 2)): d[2], sp.Derivative(g(xi), (xi, 2)): d[3],
               sp.Derivative(f(xi), xi): qv, sp.Derivative(g(xi), xi): qgv, sp.Derivative(b(xi), xi): d[4],
               sp.Derivative(A(xi), (xi, 2)): App, sp.Derivative(B(xi), (xi, 2)): Bpp,
               sp.Derivative(A(xi), xi): Apv, sp.Derivative(B(xi), xi): Bpv, sp.Derivative(be(xi), xi): bep,
               f(xi): fv, g(xi): gv, b(xi): bv, A(xi): Av, B(xi): Bv, be(xi): bev}
        num = {xi: xv, T: Tv, om: ov, kap: kv, lam: lv}

        def ev(e):
            return complex(sp.N(e.subs(rep).subs(num)))
        # scale: the largest single term of the eps part, so that the residual is relative
        r = [ev(e) for e in lin]
        sc = [max(abs(complex(sp.N(t.subs(rep).subs(num)))) for t in sp.Add.make_args(sp.expand(e))) for e in lin]
        rb = [abs(ev(e)) for e in bgr] if k == 0 else [0, 0, 0]
        rel = [abs(x) / y for x, y in zip(r, sc)]
        worst = max(worst, max(rel), max(rb))
        print("trial %d pert %d (kappa %.2f, lam %.2f%+.2fi): background |E1|,|E1b|,|E2| = %.1e %.1e %.1e;"
              "  linear parts relative to their largest term: %.1e %.1e %.1e"
              % (trial, k, kv, lv.real, lv.imag, rb[0], rb[1], rb[2], rel[0], rel[1], rel[2]), flush=True)
passed = worst < 1e-10
print("worst residual %.1e: %s" % (worst, "the linearized rhs is the eps-derivative of the inner PDE" if passed
                                    else "MISMATCH"))
if mutate:
    print("control --mutate=%s: %s" % (mutate, "fires (rejected)" if not passed else "DOES NOT FIRE"))
    sys.exit(0 if not passed else 1)
sys.exit(0 if passed else 1)
