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

For general beta(tau) the layer is quasi-static and kappa becomes kappa~ = f beta^2/(beta + beta') (the tau-derivatives
are n^{-1/2} smaller in the layer, so in the outer series they reach only z^{-(2k-1)}); the same identity must hold
with kappa -> kappa~, which tests the quasi-static layer reduction.

    python3 match_check.py                                        # CSS through NNLO
    python3 match_check.py logs/expand_K1_free.pkl                # general beta(tau), f(tau), NLO
    python3 match_check.py logs/expand_K2_zero_fconst.pkl         # general beta(tau), gauge f = const, NNLO
    python3 match_check.py --mutate=layer     # layer with C1' = C1 Y1^2/4: must FAIL
"""
import pickle
import sys
import sympy as sp

mut = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
args = [x for x in sys.argv[1:] if not x.startswith("--")]
path = args[0] if args else "logs/expand_K2_zero_css.pkl"
with open(path, "rb") as fh:
    E = pickle.load(fh)
th = sp.Symbol("theta", positive=True)
K = E["K"]
a = sp.symbols(f"a1:{K + 1}")
bco = sp.symbols(f"b1:{K + 1}")
Y1 = th ** sp.Rational(-1, 2) * (1 + sum(a[k] / th ** (2 * k + 2) for k in range(K)))
C1 = sp.sqrt(2 * th) * (1 + sum(bco[k] / th ** (2 * k + 2) for k in range(K)))
e1 = sp.diff(Y1, th) - C1 * (C1 * Y1 - sp.sqrt(2))
e2 = sp.diff(C1, th) - C1 * Y1 ** 2 / (4 if mut == "layer" else 2)
u = sp.Symbol("u", positive=True)       # u = 1/theta
r1 = sp.expand(sp.simplify(e1 * th ** sp.Rational(3, 2)).subs(th, 1 / u))
r2 = sp.expand(sp.simplify(e2 * th ** sp.Rational(1, 2)).subs(th, 1 / u))
# theta^{3/2} e1 and theta^{1/2} e2 are polynomials in u = 1/theta with even powers only; the coefficients
# (e1: u^0, u^2, ...; e2: u^2, u^4, ...) determine (a_k, b_k) order by order (e2 at u^0 holds identically)
assert sp.simplify(r2.coeff(u, 0)) == 0 or mut
eqs = [r1.coeff(u, 2 * j) for j in range(K)] + [r2.coeff(u, 2 * j + 2) for j in range(K)]
sol = sp.solve(eqs, list(a) + list(bco), dict=True)
print("layer asymptotic coefficients:", sol)
assert len(sol) == 1, "the order-by-order system is linear: one solution"
sol = sol[0]

z, eps, Bd, Fd, Ad, Cd, Lz, L1 = E["syms"]
beta, f = Bd[0], Fd[0]
zero = {q: 0 for k in Ad for q in Ad[k]}
zero.update({q: 0 for k in Cd for q in Cd[k]})
css = not any(sp.sympify(E[nm][k]).has(Bd[1]) for nm in "YC" for k in range(1, K + 1))
ktilde = f * beta if css else f * beta ** 2 / (beta + Bd[1])
print(f"{path}: K = {K}, {'CSS' if css else 'general beta(tau)'}; kappa~ = {ktilde}")
kap = sp.Symbol("kappa", positive=True)
ok = True
for name, ser, norm, co in (("Y", E["Y"], beta, a), ("C", E["C"], f, bco)):
    for k in range(1, K + 1):
        ex = sp.expand(sp.sympify(ser[k]).subs(zero) / norm)
        P = sp.Poly(ex, Lz)
        lead_nolog = sp.expand(P.coeff_monomial(1) * z ** (2 * k))
        lead = sp.limit(lead_nolog, z, 0)
        loglead = [sp.limit(sp.expand(P.coeff_monomial(Lz ** j) * z ** (2 * k)), z, 0) for j in range(1, 3)]
        pred = sp.simplify(sol[co[k - 1]] * (2 / kap) ** (2 * k)).subs(kap, ktilde)
        diff = sp.simplify(lead - pred)
        print(f"{name}_{k}: [z^-{2 * k}] outer = {sp.factor(lead)}; layer predicts {sp.factor(pred)}; "
              f"difference {diff}; ln z terms at z^-{2 * k}: {loglead}")
        ok &= diff == 0 and all(v == 0 for v in loglead)
print("PASS" if ok else "FAIL", f"(mutation: {mut})" if mut else "")
sys.exit(0 if ok else 1)
