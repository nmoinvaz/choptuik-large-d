"""Second order at a zero crossing.  Expand the exact horizon-edge identity
   (1-eps) A(1) = 2 Q(1)^2 + 2 eps^2 s(1)^2 - eps A_x(1)/(1 + eps A(1))
through eps^2 near tau_z where B(tau_z) = 0, writing B = b1*d + b2*d^2/2 + b3*d^3/6 (d = tau - tau_z).
Ingredients: Q(1) = B + eps Q1(1) + eps^2 Q2(1); A0 = 2 q0^2; A1 = A0 + 4 q0 q1 - x A0_x (both vanish identically in x
when B = 0, since q0 = x B/c0); s(1) = s0(1) + O(eps)."""
import sympy as sp
d, e = sp.symbols('delta epsilon', real=True)
b1, b2, b3, k1, k2, Q2 = sp.symbols('b1 b2 b3 k1 k2 Q2', real=True)
B = b1 * d + b2 * d**2 / 2 + b3 * d**3 / 6
Bp = b1 + b2 * d + b3 * d**2 / 2; Bpp = b2 + b3 * d
Q1 = 2 * B * Bp**2 + B + sp.Rational(3, 2) * Bp + Bpp / 2      # small-B form of Q1(tau,1), exact in Q1's B-linear part
A0x1 = -4 * B**2 * (2 * B**2 - 1)
s01 = -2 * B**3 + B + Bp
# A1_x(1) vanishes identically when B = 0 -> write it as B * k1 + O(B^2) with k1 unknown; Q2 enters as 4 B Q2
expr = 2 * (B + e * Q1 + e**2 * Q2)**2 + 2 * e**2 * s01**2 - e * (A0x1 + e * (k1 * B))
ser = sp.expand(sp.series(expr, e, 0, 3).removeO())
at_zero = sp.simplify(ser.subs(d, 0))
print("(1-eps)A(1) at the zero crossing, through eps^2:", sp.factor(at_zero))
print("   = 2 eps^2 [ Q1^2 + B'^2 ]  with Q1 = (B''+3B')/2:", sp.simplify(at_zero - 2 * e**2 * ((b2 + 3 * b1)**2 / 4 + b1**2)) == 0)
# minimum near the zero: keep terms up to eps^2, delta^2 and eps*delta
trunc = sum(ser.coeff(e, i) * e**i for i in range(3))
quad = sp.expand(sp.series(trunc, d, 0, 3).removeO())
# treat as quadratic in d: a d^2 + b d + c
a_ = quad.coeff(d, 2).subs(e, e); b_ = quad.coeff(d, 1); c_ = quad.coeff(d, 0)
a0 = sp.series(a_, e, 0, 1).removeO(); b0_ = sp.series(b_, e, 0, 2).removeO(); c0_ = sp.series(c_, e, 0, 3).removeO()
mn = sp.simplify(sp.series(c0_ - b0_**2 / (4 * a0), e, 0, 3).removeO())
print("leading behaviour near the zero:  a d^2 + b d + c  with a =", sp.factor(a0), ", b =", sp.factor(b0_), ", c =", sp.factor(c0_))
print("minimum over d (through eps^2):", sp.factor(mn))
trunc1 = ser.coeff(e, 0) + e * ser.coeff(e, 1)
q1 = sp.expand(sp.series(trunc1, d, 0, 3).removeO())
a1_, b1_, c1_ = q1.coeff(d, 2), q1.coeff(d, 1), q1.coeff(d, 0)
a10 = sp.series(a1_, e, 0, 1).removeO()
print("NLO-only truncation (EEG's level): a d^2 + b d + c with b =", sp.factor(b1_), ", c =", sp.factor(c1_),
      "; minimum =", sp.factor(sp.series(c1_ - b1_**2 / (4 * a10), e, 0, 3).removeO()))
