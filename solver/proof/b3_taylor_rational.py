"""Symbolic check of Lemma B3.3 (proof/B3_borel.md): in the centre gauge c(0) = 1, the Taylor coefficients at x = 0 of
the centre-regular solution of the M = 1 model are rational in n, with denominators made only of factors n + k, k >= 0
(poles at n = 0, -1, -2, ...; the pole at n = 0 is the 1/n of p^2 + m^2 = (P^2 + Mm^2)/n, a power of eps).

M = 1: p = P/sqrt(n), m = Mm/sqrt(n) carry the harmonic +1 (coefficients P_j, M_j), the harmonic -1 is the conjugate
(F -> Fc, i -> -i); b, c are theta-independent. Equations (C_selection.md Section 1, harmonic +1 / harmonic 0):
    x c'        = (n-1) c (b-1)
    x b'        = 2 b (P Pc + M Mc)/n - (n-1) b (b-1)
    x (x-c) P'  = -i w x P + (n-1) c (b-1) P - c P + (n/2) c (P + Mm)
    x (x+c) Mm' = -i w x Mm - (n-1) c (b-1) Mm + c Mm - (n/2) c (P + Mm)
with P_1 = F, M_1 = -F (P - Mm = 2 F x + ...), c(0) = b(0) = 1; P and Mm carry every power x^j, j >= 1 (P - Mm odd,
P + Mm even in x), b and c the even powers. Solved order by order to x^ORDER with sympy, then every
coefficient's denominator is factored. Exits non-zero if a factor other than n + k (k >= 0 an integer) appears, or if
the substitution into the equations leaves a residual. --mutate=eq replaces the factor n/2 of the m equation by n/3 in
the solved system but not in the residual test, so the residual test must fail.

    python3 b3_taylor_rational.py [ORDER=7] [--mutate=eq]
"""
import sys

import sympy as sp

ORDER = int([a for a in sys.argv[1:] if not a.startswith("--")][0]) if [a for a in sys.argv[1:] if not a.startswith("--")] else 7
MUT = [a.split("=")[1] for a in sys.argv[1:] if a.startswith("--mutate=")]
MUT = MUT[0] if MUT else ""
n, w, x = sp.symbols("n w x")
F, Fc = sp.symbols("F Fc")
I = sp.I


def conj(e):
    return e.subs({F: Fc, Fc: F, I: -I}, simultaneous=True)


Pj = {1: F}
Mj = {1: -F}
bj = {0: sp.Integer(1)}
cj = {0: sp.Integer(1)}
odd = [j for j in range(2, ORDER + 1)]          # P, Mm carry every power >= 2 (only P - Mm is odd in x)
even = [j for j in range(2, ORDER + 1, 2)]
uP = {j: sp.Symbol(f"P{j}") for j in odd}
uM = {j: sp.Symbol(f"M{j}") for j in odd}
ub = {j: sp.Symbol(f"b{j}") for j in even}
uc = {j: sp.Symbol(f"c{j}") for j in even}
P = F * x + sum(uP[j] * x ** j for j in odd)
Mm = -F * x + sum(uM[j] * x ** j for j in odd)
b = 1 + sum(ub[j] * x ** j for j in even)
c = 1 + sum(uc[j] * x ** j for j in even)
fac = sp.Rational(1, 2) if MUT != "eq" else sp.Rational(1, 3)
Pc = conj(F) * x + sum(sp.Symbol(f"P{j}c") * x ** j for j in odd)
Mc = -conj(F) * x + sum(sp.Symbol(f"M{j}c") * x ** j for j in odd)
def eqs(f):
    return [x * sp.diff(c, x) - (n - 1) * c * (b - 1),
            x * sp.diff(b, x) - 2 * b * (P * Pc + Mm * Mc) / n + (n - 1) * b * (b - 1),
            x * (x - c) * sp.diff(P, x) - (-I * w * x * P + (n - 1) * c * (b - 1) * P - c * P + n / 2 * c * (P + Mm)),
            x * (x + c) * sp.diff(Mm, x) - (-I * w * x * Mm - (n - 1) * c * (b - 1) * Mm + c * Mm - f * n * c * (P + Mm))]


E = eqs(fac)                      # the system solved (mutated under --mutate=eq)
ETRUE = eqs(sp.Rational(1, 2))    # the system the residual is checked against
sol = {}
for j in range(2, ORDER + 1):
    eqs = [sp.expand(sp.series(e.subs(sol), x, 0, j + 1).removeO()).coeff(x, j) for e in E]
    unk = [uP[j], uM[j]] + ([uc[j], ub[j]] if j in even else [])
    # the conjugate unknowns of lower order are known once their own order is solved
    s = sp.solve([e for e in eqs if e != 0], unk, dict=True)
    if not s:
        print(f"order {j}: no solution (inconsistent equations)")
        sys.exit(1)
    s = {k: sp.factor(sp.simplify(v)) for k, v in s[0].items()}
    sol.update(s)
    if True:
        sol[sp.Symbol(f"P{j}c")] = conj(s[uP[j]])
        sol[sp.Symbol(f"M{j}c")] = conj(s[uM[j]])
bad = False
for k in sorted(sol, key=str):
    if str(k).endswith("c") and str(k)[0] in "PM":
        continue
    v = sol[k]
    den = sp.denom(sp.together(v))
    factors = [f for f, _ in sp.factor_list(den)[1]]
    ok = all(sp.Poly(f, n).degree() == 1 and f.free_symbols <= {n} and sp.Poly(f, n).LC() == 1 for f in factors)
    roots = [sp.solve(f, n)[0] for f in factors]
    print(f"{str(k):4s}: denominator factors {factors} -> poles at n = {roots}" + ("" if ok else "   <-- NOT of the form n + const"))
    bad |= not ok or any(r > 0 for r in roots)
# residual check: substitute and expand to ORDER
resid = [sp.simplify(sp.expand(sp.series(e.subs(sol), x, 0, ORDER + 1).removeO())) for e in ETRUE]
nonzero = [r for r in resid if r != 0]
print("residual of the equations through x^%d: %s" % (ORDER, "zero" if not nonzero else "NONZERO"))
bad |= bool(nonzero)
print("PASS: every denominator is a product of factors (n + k), k >= 0 (poles at n = 0, -1, -2, ...)" if not bad else "FAIL")
sys.exit(1 if bad else 0)
