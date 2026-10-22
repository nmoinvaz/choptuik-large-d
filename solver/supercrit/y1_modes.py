#!/usr/bin/env python3
"""The exact CSS solution Y = 1 that both branches approach at large x, and its perturbation exponents at finite n.

For every n and every kappa > 0,
    p = -m = 1/sqrt2  (y = p - m = sqrt2, p + m = 0),   b = n/(n-1)  ((n-1)(b-1) = 1),   c = kappa x
solves the exact equations of hp/src/model.rs (psi = -sqrt(n) kappa ln(-t): a scalar field linear in ln(-t) in a
metric with constant g_rr; at n = 2, b = 2 is the (grad r)^2 = 1/2 of Brady's stationary solution, PRD 51, 4168
(1995), Eq. (2.24)/(3.1); see notes/SUPERCRITICAL.md). Perturbations proportional to x^lambda satisfy
    lambda (lambda + n) [ (1 - kappa^2) lambda^2 - kappa^2 n lambda - 2 kappa^2 n ] = 0     (derived below),
whose decaying slow root is lambda_- = -2 + lambda_1/n + lambda_2/n^2 + ...
The large-x logarithms of expand.py's CSS series (Y_1 ~ v log v, Y_2 ~ v log^2 v, ...) must be the expansion of
x^lambda_-: the script extracts lambda_1 from the NLO and lambda_2 (and the log^2 coefficient) from the NNLO series
and compares them with the exact root. These share no code with expand.py beyond the field equations.

    python3 y1_modes.py
    python3 y1_modes.py --mutate=y1     # Y_1 scaled by 1.01 before the comparison: must FAIL
"""
import pickle
import sys
import sympy as sp

mut = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
x, n, kap, lam = sp.symbols("x n kappa lambda", positive=True)
ok = True

# --- exact solution
p, m, b, c = 1 / sp.sqrt(2), -1 / sp.sqrt(2), n / (n - 1), kap * x
G = (n - 1) * c * (b - 1)
E = [x * sp.diff(c, x) - G,
     x * sp.diff(b, x) - b * (p ** 2 + m ** 2) + (n - 1) * b * (b - 1),
     x * (x - c) * sp.diff(p, x) - (G * p - c * p + n / 2 * c * (p + m)),
     x * (x + c) * sp.diff(m, x) - (-G * m + c * m - n / 2 * c * (p + m))]
res = [sp.simplify(e) for e in E]
print("exact solution residuals E1..E4:", res)
ok &= all(r == 0 for r in res)

# --- linearization, perturbations ~ x^lambda (CSS: no tau dependence)
dy, ds, du, dc = sp.symbols("dy ds du dc")
t = sp.Symbol("t")      # bookkeeping parameter
yv = sp.sqrt(2) + t * dy * x ** lam
sv = t * ds * x ** lam
uv = 1 + t * du * x ** lam
cv = kap * x * (1 + t * dc * x ** lam)
# sum/difference form (eqmap.py)
P = (yv ** 2 + sv ** 2) / 2
eqs = [x * sp.diff(cv, x) - cv * uv,
       x * sp.diff(uv, x) - (n - 1 + uv) * (P - uv),
       x ** 2 * sp.diff(yv, x) - x * cv * sp.diff(sv, x) - cv * (n - 1 + uv) * sv,
       x ** 2 * sp.diff(sv, x) - x * cv * sp.diff(yv, x) - cv * (uv - 1) * yv]
lin = [sp.expand(sp.simplify(sp.diff(e, t).subs(t, 0) / x ** lam)) for e in eqs]
Mx = sp.Matrix([[sp.diff(l, v) for v in (dy, ds, du, dc)] for l in lin])
det = sp.factor(sp.simplify(Mx.det()))
print("det of the linearized system:", det)   # each row is homogeneous in x (scaling symmetry), so x factors out
poly = (1 - kap ** 2) * lam ** 2 - kap ** 2 * n * lam - 2 * kap ** 2 * n
quot = sp.cancel(det / (lam * (lam + n) * poly))
print("det / [lambda (lambda + n) ((1-kappa^2) lambda^2 - kappa^2 n lambda - 2 kappa^2 n)] =", quot)
ok &= quot.free_symbols <= {x, kap} and quot != 0
# slow decaying root expanded in 1/n
e_ = sp.Symbol("e", positive=True)
# eps * poly = (1-kappa^2) lambda^2 eps - kappa^2 lambda - 2 kappa^2; lambda = -2 + l1 eps + l2 eps^2, order by order
L1s, L2s = sp.symbols("l1 l2")
pe = sp.expand(((1 - kap ** 2) * lam ** 2 * e_ - kap ** 2 * lam - 2 * kap ** 2).subs(lam, -2 + L1s * e_ + L2s * e_ ** 2))
assert sp.simplify(pe.coeff(e_, 0)) == 0
l1 = sp.solve(pe.coeff(e_, 1), L1s)[0]
l2 = sp.solve(pe.coeff(e_, 2).subs(L1s, l1), L2s)[0]
print(f"lambda_- = -2 + ({sp.factor(l1)})/n + ({sp.factor(l2)})/n^2 + ...")

# --- compare with the large-x logs of the CSS series
with open("logs/expand_K2_zero_css.pkl", "rb") as fh:
    S = pickle.load(fh)
z, eps, Bd, Fd, Ad, Cd, Lz, L1 = S["syms"]
bb, ff = sp.Rational(13, 10), sp.Rational(9, 10)
kv = bb * ff
X = sp.Symbol("X", positive=True)      # X = 1/x^2
sub = {Bd[0]: bb, Fd[0]: ff}
Y1 = S["Y"][1] * (sp.Rational(101, 100) if mut == "y1" else 1)
yser = [sp.sympify(S["Y"][0]), Y1, S["Y"][2]]
coef = {}
LX = sp.Symbol("LX")                   # log X, kept as a symbol (coefficients are polynomials in it)
for k in range(3):
    e = (sp.sqrt(2) * yser[k] / sp.sqrt(bb ** 2 - X)).subs(sub)
    e = e.subs({Lz: sp.log(bb ** 2 - X) - LX, L1: 2 * sp.log(bb) - LX}).subs(z, bb ** 2 / X - 1)
    sr = sp.expand(sp.series(sp.together(e), X, 0, 2).removeO())
    c1 = sp.expand(sr.coeff(X, 1))
    coef[k] = {j: sp.nsimplify(sp.simplify(c1.coeff(LX, j))) for j in range(3)}
a0 = coef[0][0]
A1, a1 = coef[1][0], coef[1][1]
c22, c21 = coef[2][2], coef[2][1]
lam1_series = sp.nsimplify(-2 * a1 / a0)
lam1_exact = l1.subs(kap, kv)
lam2_series = sp.nsimplify(-2 * (c21 + lam1_series / 2 * A1) / a0)
lam2_exact = sp.nsimplify(l2.subs(kap, kv))
print(f"kappa_f = f beta = {kv}:  y - sqrt2 = X [a0 + eps (A1 + a1 log X) + eps^2 (... + c21 log X + c22 log^2 X)], X = 1/x^2")
print(f"   lambda_1: from the NLO series {lam1_series} = {float(lam1_series):.10f}; exact root {float(lam1_exact):.10f}")
print(f"   log^2 coefficient c22/a0 = {float(c22 / a0):.10f}; lambda_1^2/8 = {float(lam1_exact ** 2 / 8):.10f}")
print(f"   lambda_2: from the NNLO series {float(lam2_series):.10f}; exact root {float(lam2_exact):.10f}")
chk = [abs(float(lam1_series - lam1_exact)), abs(float(c22 / a0 - lam1_exact ** 2 / 8)), abs(float(lam2_series - lam2_exact))]
ok &= all(v < 1e-12 for v in chk)
print("PASS" if ok else "FAIL", f"(mutation: {mut})" if mut else "")
sys.exit(0 if ok else 1)
