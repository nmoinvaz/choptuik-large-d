#!/usr/bin/env python3
"""Print the NLO coefficients of the supercritical branch (expand.py's free run, A_1 = B_1 = 0) factored, by power of
ln z (notes/SUPERCRITICAL.md, section 2.2).

    python3 nlo_factor.py [logs/expand_K1_free.pkl]
"""
import pickle
import sys
import sympy as sp

E = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "logs/expand_K1_free.pkl", "rb"))
z, eps, Bd, Fd, Ad, Cd, Lz, L1 = E["syms"]
sub = {a: 0 for k in Ad for a in Ad[k]}
sub.update({a: 0 for k in Cd for a in Cd[k]})
print("b0, b1, b2, ... = beta, beta', beta'', ...; f0, f1, ... = f, f', ...; Lz = ln z")
for nm, idx in (("Y", 1), ("C", 1), ("U", 1), ("S", 2)):
    e = sp.expand(E[nm][idx].subs(sub))
    P = sp.Poly(e, Lz)
    for (a,), cf in P.terms():
        print(f"{nm}_{idx}, Lz^{a} coefficient:", sp.factor(sp.cancel(cf)))
