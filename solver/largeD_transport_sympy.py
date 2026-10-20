"""derive the fast-mode transport from the exact equations E1-E4, with b, c perturbed too.
Background (Prop 1): p0 = e^(1/2) x F, m0 = -p0 (+ O(e^(3/2))), b0 = 1 + e^2 2x^2F^2, c0 = 1 - e F^2 (1-x^2).
Fast perturbation: (dp, dm, db, dc) = exp(-Phi/e) * (A + e a1, Bm + e b1, e^(3/2) be, e^(3/2) ga), n = 1/e.
"""
import sympy as sp
x, th, e = sp.symbols("x theta epsilon", positive=True)
w = sp.symbols("omega")
F = sp.Function("F")(th)
Phi = sp.Function("Phi")(x)
A, Bm, a1, b1, be, ga = [sp.Function(nm)(x, th) for nm in ("A", "B", "a1", "b1", "beta", "gamma")]
n = 1 / e
h = sp.sqrt(e)
p0 = h * x * F; m0 = -p0
b0 = 1 + e**2 * 2 * x**2 * F**2
c0 = 1 - e * F**2 * (1 - x**2)
E = sp.exp(-Phi / e)
dp = E * (A + e * a1); dm = E * (Bm + e * b1); db = E * h**3 * be; dc = E * h**3 * ga
def lin(expr_fn):
    t = sp.symbols("t")
    full = expr_fn(p0 + t * dp, m0 + t * dm, b0 + t * db, c0 + t * dc)
    return sp.diff(full, t).subs(t, 0)
Dx = lambda f: sp.diff(f, x); Dt = lambda f: sp.diff(f, th)
E1 = lambda p, m, b, c: x * Dx(c) - (n - 1) * c * (b - 1)
E2 = lambda p, m, b, c: x * Dx(b) - b * (p**2 + m**2) + (n - 1) * b * (b - 1)
def E3(p, m, b, c):
    G = (n - 1) * c * (b - 1)
    return x * (x - c) * Dx(p) + w * x * Dt(p) - G * p + c * p - n / 2 * c * (p + m)
def E4(p, m, b, c):
    G = (n - 1) * c * (b - 1)
    return x * (x + c) * Dx(m) + w * x * Dt(m) + G * m - c * m + n / 2 * c * (p + m)
res = {}
for name, fn in (("E1", E1), ("E2", E2), ("E3", E3), ("E4", E4)):
    L = sp.expand(sp.simplify(lin(fn) / E))
    res[name] = L
def order(expr, k):          # coefficient of e^k (half-integer powers allowed)
    s = sp.series(expr, e, 0, k + sp.Rational(3, 2)).removeO()
    return sp.simplify(sp.expand(s).coeff(e, k)) if k == int(k) else sp.simplify(sp.expand(s).coeff(sp.sqrt(e), int(2 * k)))
# leading orders of the b, c perturbation equations: E2 at e^(1/2) ... print the lowest few orders
for name in ("E2", "E1"):
    s = sp.expand(sp.series(res[name] * e, e, 0, 2).removeO())
    print(name, "lowest orders (times e):", sp.collect(s, sp.sqrt(e)))

# ---- p, m equations: orders e^0 and e^1 of e*E3, e*E4
q = 1 / (x * (1 - x**2))
sub = {sp.Derivative(Phi, x): q}
o = {}
for name in ("E3", "E4"):
    s = sp.expand(sp.series(res[name] * e, e, 0, 2).removeO())
    s = s.subs(sub)
    o[name] = [sp.simplify(s.coeff(e, 0)), sp.simplify(s.coeff(e, 1)), sp.simplify(s.coeff(sp.sqrt(e), 1)), sp.simplify(s.coeff(sp.sqrt(e), 3))]
print("leading E3:", o["E3"][0]); print("leading E4:", o["E4"][0])
print("half-integer orders E3 (e^1/2, e^3/2):", o["E3"][2], "|", o["E3"][3])
print("half-integer orders E4 (e^1/2, e^3/2):", o["E4"][2], "|", o["E4"][3])
Bsol = A * (1 - x) / (1 + x)
lead = [sp.simplify(o[k][0].subs(Bm, Bsol).doit()) for k in ("E3", "E4")]
print("leading with B = A(1-x)/(1+x):", lead)
# solvability of the e^1 order: left null vector (1+x, -(1-x)) applied to the order-1 equations
O1 = [o[k][1].subs(Bm, Bsol).doit() for k in ("E3", "E4")]
solv = sp.simplify((1 + x) * O1[0] - (1 - x) * O1[1])
print("a1, b1 drop out:", not solv.has(a1) and not solv.has(b1))
target = x * (1 - x**2) * sp.diff(A, x) - 2 * w * x**2 * sp.diff(A, th) - (1 + x - 2 * x**2 * F**2) * A
ratio = sp.simplify(solv / target)
print("solvability / transport target =", ratio)
