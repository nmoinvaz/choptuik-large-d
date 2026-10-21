"""Rigorous check (arb) of the linear Stokes normalization R_1 = 1 used by lateral_np.py.
Linear (c = 1) harmonic mu: Dd = 2F_mu xi 2F0(a,b;;2 xi^2), a = (1+s)/2, b = (2+s)/2, s = i mu w.
Lateral sums: Dd(xi) = 2F_mu xi x^a U(a, 1/2, x), x = -1/(2 xi^2), continued from arg x < pi (y+) or > pi (y-)."""
from flint import acb, arb, ctx
ctx.prec = 256
I = acb(0, 1)
PI = arb.pi()


def U_upper(a, b, X):
    """U(a,b,z) at z = -X (X > 0), value continuous from above (arg z -> pi-)"""
    return acb(-X).hypgeom_u(a, b)


def U_lower(a, b, X):
    """principal U at z = -X approached from below (arg z -> -pi+): conj(U(conj a, conj b, -X from above))"""
    return acb(-X).hypgeom_u(a.conjugate(), b.conjugate()).conjugate()


def jump_exact(w, xi1, mu=1, Fmu=acb(0, -0.5)):
    s = I * mu * w
    a, b = (1 + s) / 2, (2 + s) / 2
    c = acb(0.5)                       # 1 + a - b
    X = 1 / (2 * xi1 * xi1)
    # y+ (Borel sum from above): principal U on the upper side of its cut, arg x = pi-, with x^a = X^a e^{+i pi a};
    # y- (from below): arg x = -pi+, x^a = X^a e^{-i pi a}. Both are asymptotic to the same formal series.
    Xa = acb(X) ** a
    Dp = 2 * Fmu * xi1 * Xa * (I * PI * a).exp() * U_upper(a, c, X)
    Dm = 2 * Fmu * xi1 * Xa * (-I * PI * a).exp() * U_lower(a, c, X)
    # normalization of lateral_np: 2 pi i K'(s) f(xi1) 2F_mu, f = xi^{-2s} e^{-1/(2 xi^2)} 2F0(1-a, 1-b;; -2 xi^2)
    Kp = acb(2) ** (-s - acb(0.5)) / (a.gamma() * b.gamma())
    a2, b2 = 1 - a, 1 - b
    Y = acb(X)                         # 2F0(a2,b2;;-1/Y) = Y^{a2} U(a2, 1+a2-b2, Y), Y = 1/(2 xi^2) > 0
    f = acb(xi1) ** (-2 * s) * (-acb(X)).exp() * Y ** a2 * Y.hypgeom_u(a2, 1 + a2 - b2)
    ref = 2 * PI * I * Kp * f * 2 * Fmu
    return Dp, Dm, Dp - Dm, ref


if __name__ == "__main__":
    W0 = 2 * PI / arb("1.64389375")
    for xi1 in (arb("0.2"), arb("0.3")):
        for w in (W0, W0 * arb("0.777")):
            Dp, Dm, J, ref = jump_exact(w, xi1)
            print(f"xi1 {xi1.str(3)} w {w.str(8)}: J/ref = {(J / ref).str(25)}")
