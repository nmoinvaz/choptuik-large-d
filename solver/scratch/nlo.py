"""Large-D expansion, eps = 1/(D-2).  B(tau) := Q(tau, x=0) exactly (centre value of Q = c q / x).
LO:  W0 = c0^2 = 1 - 2 B^2 (1 - x^2),  q0 = x B / c0,  s0 = (x/c0)(q0_tau + x q0_x),  A0 = 2 q0^2.
NLO: x Q1_x = s0_tau + x s0_x,  Q1(tau, 0) = 0.
At the horizon edge x = 1 (c = 1 exactly):  q(tau,1) = Q(tau,1)  and, exactly from E4,
     (1 - eps) A(1) = 2 Q(1)^2 + 2 eps^2 s(1)^2 - eps A_x(1) / (1 + eps A(1)).
Ecker-Ecker-Grumiller's f - 1 ~ -(1-x)(1-eps)A(1); their NLO term is eps*beta(beta''+3beta') with beta^2 = 2B^2/(1-2B^2)."""
import sympy as sp
t, x = sp.symbols('tau x', positive=True)
b0, b1, b2, b3 = sp.symbols('b0 b1 b2 b3', real=True)
B = sp.Function('B')(t)
W0 = 1 - 2 * B**2 * (1 - x**2); c0 = sp.sqrt(W0)
q0 = x * B / c0
s0 = (x / c0) * (sp.diff(q0, t) + x * sp.diff(q0, x))
A0 = 2 * q0**2
integrand = (sp.diff(s0, t) + x * sp.diff(s0, x)) / x
rep = {sp.Derivative(B, (t, 3)): b3, sp.Derivative(B, (t, 2)): b2, sp.Derivative(B, t): b1, B: b0}
def ev(e): return sp.simplify(e.subs(rep))
I = ev(integrand)
print("integrand of Q1 (x-derivative of Q1 times 1):")
print("  ", sp.simplify(I))
# small-B check first (series in b0 to first order), then exact if possible
I_small = sp.series(I, b0, 0, 2).removeO()
Q1_small = sp.integrate(sp.expand(I_small), (x, 0, 1))
print("Q1(tau,1) to first order in B:", sp.simplify(Q1_small))
A0x1 = ev(sp.diff(A0, x).subs(x, 1))
s01 = ev(s0.subs(x, 1))
print("A0_x at x=1:", sp.factor(A0x1), "    s0 at x=1:", sp.factor(s01))
# NLO horizon-edge combination  (1-eps)A(1) = 2B^2 + eps*(4 B Q1(1) - A0_x(1)) + ...
nlo_small = sp.series(sp.expand(4 * b0 * Q1_small - A0x1), b0, 0, 3).removeO()
print("NLO term of (1-eps)A(1), small B:", sp.factor(nlo_small))
print("EEG form eps*beta(beta''+3beta') with beta ~ sqrt(2) B:  2 B (B''+3B') =", sp.expand(2 * b0 * (b2 + 3 * b1)))
# exact-in-B integral, numerically, to see the general structure
import mpmath as mp
f = sp.lambdify((x, b0, b1, b2), I, "mpmath")
for vals in ((0.3, 1.0, 0.5), (0.6, -0.7, 2.0)):
    num = mp.quad(lambda xx: f(xx, *vals), [0, 1])
    print(f"Q1(1) exact quadrature at (B,B',B'')={vals}: {float(num):.6f};  small-B formula: {float(Q1_small.subs({b0: vals[0], b1: vals[1], b2: vals[2]})):.6f}")
