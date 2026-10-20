"""Positivity of b - 1 (equivalently f = c <= 1) near a zero of F in the formal 1/n family (Proposition 1, K_k = 0,
omega_k = 0 for k >= 1).  b - 1 = eps^2 B2 + eps^3 B3 + eps^4 B4 + ...; near a zero theta0 of F put theta = theta0 + eps*eta.
At O(eps^4) the window terms are  2 x^2 F'^2 eta^2 + b31(x) eta + b40(x)  (B2 = 2x^2F^2, B3 linear in eta, B4 at the
zero). Exact solutions have b >= 1 (theorem), so b31^2 <= 8 x^2 F'^2 b40 is necessary; Ecker-Ecker-Grumiller's Eq. (20)
is the stronger statement b31 = 0."""
import os
import pickle
import sympy as sp
from sympy import srepr, sympify
d = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "expand_b.pkl"), "rb"))
x, th, eps, eta = sp.symbols("x theta epsilon eta")
w0 = sp.Symbol("w0")
F = sp.Function("F")(th)
E = {k: sp.sympify(eval(v, sp.__dict__.copy() | {"Symbol": sp.Symbol, "Function": sp.Function})) if False else sp.parse_expr(v) if False else None for k, v in d.items()}
from sympy.parsing.sympy_parser import parse_expr
ns = {}
exec("from sympy import *", ns)
E = {k: eval(v, ns) for k, v in d.items()}
# symbols in E: x, theta, w0, F(theta)
X, TH = sp.Symbol("x"), sp.Symbol("theta")
Ff = sp.Function("F")
f1, f2, f3, f4, f5, f6 = sp.symbols("f1:7")      # F^(j) at the zero
# Taylor of F around theta0: F(theta0 + delta) = sum f_j delta^j / j!, f0 = 0
dl = sp.Symbol("delta")
taylor = sum(fj * dl ** (j + 1) / sp.factorial(j + 1) for j, fj in enumerate((f1, f2, f3, f4, f5, f6)))
def at_zero(expr, order):
    """expr(F, F', ...) -> series in delta to the given order"""
    e = expr
    for j in range(8, -1, -1):
        e = e.subs(sp.Derivative(Ff(TH), (TH, j)) if j > 0 else Ff(TH), sp.diff(taylor, dl, j))
    e = e.subs(TH, dl)
    return sp.expand(sp.series(e, dl, 0, order + 1).removeO())
B2 = at_zero(E["B2"], 2); B3 = at_zero(E["B3"], 1); B4 = at_zero(E["B4"], 0)
print("B2 near zero:", sp.factor(B2))
b30 = B3.coeff(dl, 0); b31 = sp.factor(B3.coeff(dl, 1)); b40 = sp.factor(B4.coeff(dl, 0))
print("B3: b30 =", sp.simplify(b30), "  b31 =", b31)
print("B4: b40 =", b40)
disc = sp.factor(sp.expand(8 * X**2 * f1**2 * b40 - b31**2))
print("8 x^2 F'^2 b40 - b31^2 =", disc)
pickle.dump({"b31": sp.srepr(b31), "b40": sp.srepr(b40), "disc": sp.srepr(disc)}, open("window.pkl", "wb"))
