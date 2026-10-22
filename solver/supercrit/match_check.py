#!/usr/bin/env python3
"""Matching check: the most singular terms of the outer 1/n series (expand.py, CSS) at z -> 0 must equal the large-theta
expansion of the layer (layer.py), order by order.

Layer (universal, CSS): Y1' = C1 (C1 Y1 - sqrt2), C1' = C1 Y1^2/2, with
    Y1 = theta^{-1/2} (1 + a1/theta^2 + a2/theta^4 + ...),   C1 = sqrt(2 theta) (1 + b1/theta^2 + b2/theta^4 + ...),
whose coefficients this script derives from the layer ODE alone. The outer series in the layer limit, with
z = 2 theta/(kappa sqrt n), kappa = f beta, is
    Yhat/beta = 1 + eps Y_1/beta + eps^2 Y_2/beta + ...,   Chat/f = 1 + eps C_1/f + ...,
and the leading z^{-2k} term of Y_k/beta (C_k/f) must equal a_k (b_k) times (2/(kappa z))^{2k} n^{-k}... i.e.
    [z^{-2k}] Y_k / beta = a_k (2/kappa)^{2k},     [z^{-2k}] C_k / f = b_k (2/kappa)^{2k},
with no ln z multiplying z^{-2k}. The two sides share no code (expand.py vs the layer's own asymptotic series).

    python3 match_check.py
    python3 match_check.py --mutate=layer     # layer with C1' = C1 Y1^2/4: must FAIL
"""
import pickle
import sys
import sympy as sp

mut = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
th = sp.Symbol("theta", positive=True)
K = 2
a = sp.symbols(f"a1:{K + 1}")
bco = sp.symbols(f"b1:{K + 1}")
Y1 = th ** sp.Rational(-1, 2) * (1 + sum(a[k] / th ** (2 * k + 2) for k in range(K)))
C1 = sp.sqrt(2 * th) * (1 + sum(bco[k] / th ** (2 * k + 2) for k in range(K)))
e1 = sp.diff(Y1, th) - C1 * (C1 * Y1 - sp.sqrt(2))
e2 = sp.diff(C1, th) - C1 * Y1 ** 2 / (4 if mut == "layer" else 2)
u = sp.Symbol("u", positive=True)       # u = 1/theta
r1 = sp.expand(sp.simplify(e1 * th ** sp.Rational(3, 2)).subs(th, 1 / u))
r2 = sp.expand(sp.simplify(e2 * th ** sp.Rational(1, 2)).subs(th, 1 / u))
eqs = []
for p in range(1, 2 * K + 2):
    eqs += [r1.coeff(u, p), r2.coeff(u, p)]
eqs = [q for q in eqs if q != 0]
sol = sp.solve(eqs[:2 * K], list(a) + list(bco), dict=True)
print("layer asymptotic coefficients:", sol)
sol = sol[0]

with open("logs/expand_K2_zero_css.pkl", "rb") as fh:
    E = pickle.load(fh)
z, eps, Bd, Fd, Ad, Cd, Lz, L1 = E["syms"]
beta, f = Bd[0], Fd[0]
kap = sp.Symbol("kappa", positive=True)
ok = True
for name, ser, norm, co in (("Y", E["Y"], beta, a), ("C", E["C"], f, bco)):
    for k in range(1, K + 1):
        ex = sp.expand(ser[k] / norm)
        P = sp.Poly(ex, Lz)
        lead_nolog = sp.expand(P.coeff_monomial(1) * z ** (2 * k))
        lead = sp.limit(lead_nolog, z, 0)
        loglead = [sp.limit(sp.expand(P.coeff_monomial(Lz ** j) * z ** (2 * k)), z, 0) for j in range(1, 3)]
        pred = sp.simplify(sol[co[k - 1]] * (2 / kap) ** (2 * k)).subs(kap, beta * f)
        diff = sp.simplify(lead - pred)
        print(f"{name}_{k}: [z^-{2 * k}] outer = {sp.factor(lead)}; layer predicts {sp.factor(pred)}; "
              f"difference {diff}; ln z terms at z^-{2 * k}: {loglead}")
        ok &= diff == 0 and all(v == 0 for v in loglead)
print("PASS" if ok else "FAIL", f"(mutation: {mut})" if mut else "")
sys.exit(0 if ok else 1)
