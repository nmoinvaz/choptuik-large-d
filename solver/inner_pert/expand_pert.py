"""Growing modes in the 1/n expansion: linearize the formal solution of Proposition 1 (solver/largeD_expand.py) with
perturbations ~ e^{lambda tau}, every tau-derivative of a perturbation becoming lambda + omega d_theta, and run the
same order-by-order construction. Question: does a solvability condition (an x^1 term in the D_k equation, or an x^0
term in the c equation) ever constrain lambda or the perturbation profile f(theta)?

Fields are dual numbers (background, perturbation) with eta^2 = 0. Perturbation data: dD_0 = 2 x f(theta), free dK_k,
lambda = sum eps^k lam_k.  Prints the perturbation's first orders and any CONDITION found.

    python3 expand_pert.py [NORD]
"""
import sys
import sympy as sp

x, th = sp.symbols("x theta")
NORD = int(sys.argv[1]) if len(sys.argv) > 1 else 2
F = sp.Function("F")(th)
f = sp.Function("f")(th)
w = sp.symbols(f"w0:{NORD + 3}")
lam = sp.symbols(f"lam0:{NORD + 3}")
Kf = [sp.Function(f"K{k}")(th) for k in range(NORD + 3)]
dKf = [sp.Function(f"dK{k}")(th) for k in range(NORD + 3)]
N = NORD + 3


class Du:
    """a + eta b, eta^2 = 0"""
    def __init__(self, a, b=0):
        self.a, self.b = sp.sympify(a), sp.sympify(b)

    def __add__(s, o):
        o = o if isinstance(o, Du) else Du(o)
        return Du(s.a + o.a, s.b + o.b)
    __radd__ = __add__

    def __neg__(s):
        return Du(-s.a, -s.b)

    def __sub__(s, o):
        return s + (-(o if isinstance(o, Du) else Du(o)))

    def __mul__(s, o):
        o = o if isinstance(o, Du) else Du(o)
        return Du(s.a * o.a, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__

    def ex(s):
        return Du(sp.expand(s.a), sp.expand(s.b))

    def dx(s):
        return Du(sp.diff(s.a, x), sp.diff(s.b, x))


Z = Du(0)


def smul(a, b):
    return [sum((a[i] * b[k - i] for i in range(k + 1) if i < len(a) and k - i < len(b)), Z).ex() for k in range(N)]


def sadd(*ss):
    out = [Z] * N
    for s in ss:
        for k in range(min(N, len(s))):
            out[k] = out[k] + s[k]
    return [v.ex() for v in out]


def sscale(a, c):
    return [(c * v).ex() for v in a]


def sshift(a, k):
    return ([Z] * k + list(a))[:N]


def sdx(a):
    return [v.dx() for v in a]


def tau_d(a):
    """the series of d_tau acting: background omega d_theta a, perturbation (lambda + omega d_theta) b"""
    out = []
    for k in range(N):
        A = sum((w[i] * sp.diff(a[k - i].a, th) for i in range(k + 1)), sp.Integer(0))
        B = sum((w[i] * sp.diff(a[k - i].b, th) + lam[i] * a[k - i].b for i in range(k + 1)), sp.Integer(0))
        out.append(Du(sp.expand(A), sp.expand(B)))
    return out


def sinv(a):
    out = [Du(sp.Integer(1) / a[0].a, -a[0].b / a[0].a ** 2)]
    for k in range(1, N):
        acc = sum((a[i] * out[k - i] for i in range(1, k + 1)), Z)
        out.append((acc * out[0] * Du(-1)).ex())
    return out


S = [Z] * N
D = [Du(2 * x * F, 2 * x * f)] + [Z] * (N - 1)
G = [Z] * N
Bs = [Z] * N
one_m_eps = [Du(1), Du(-1)]
conds = []


def cser():
    return [Du(1)] + [-G[k] for k in range(1, N)]


def pm():
    P = [((S[i] + D[i]) * Du(sp.Rational(1, 2))).ex() for i in range(N)]
    M = [((S[i] - D[i]) * Du(sp.Rational(1, 2))).ex() for i in range(N)]
    return P, M


for k in range(1, NORD + 1):
    P, M = pm()
    q2 = sadd(smul(P, P), smul(M, M))
    onepB = sadd([Du(1)], Bs)
    rest = sadd(sshift(sscale(sdx(Bs), x), 1), sscale(sshift(smul(onepB, q2), 2), -1), smul(one_m_eps, smul(onepB, Bs)))
    Bs[k + 1] = (-rest[k + 1]).ex()
    c = cser()
    rhs = smul(one_m_eps, smul(c, Bs))[k + 1]
    for part, val in (("bg", rhs.a), ("pert", rhs.b)):
        r0 = sp.simplify(val.subs(x, 0))
        if r0 != 0:
            conds.append((f"E1 order {k + 1} {part}: x^0 term", r0))
    Ga = sp.expand(sp.integrate(sp.expand(rhs.a / x), (x, 1, x)))
    Gb = sp.expand(sp.integrate(sp.expand(rhs.b / x), (x, 1, x)))
    G[k] = Du(Ga, Gb)
    c = cser()
    num = sadd(sscale(sdx(D), x ** 2), sscale(smul(c, sdx(S)), -x), sscale(tau_d(D), x))
    den = smul(c, smul(one_m_eps, onepB))
    Snew = sshift(smul(num, sinv(den)), 1)
    S[k] = Snew[k].ex()
    # D_k: x c D_x - c D = x^2 S_x + x tau_d(S) - ((1-eps)/eps) c B D, at order k
    Dk_a, Dk_b = sp.Function("Dka")(x, th), sp.Function("Dkb")(x, th)
    Dtmp = D[:k] + [Du(Dk_a, Dk_b)] + [Z] * (N - k - 1)
    lhs = sadd(sscale(smul(c, sdx(Dtmp)), x), sscale(smul(c, Dtmp), -1))
    cBD = smul(c, smul(Bs, Dtmp))
    rhs = sadd(sscale(sdx(S), x ** 2), sscale(tau_d(S), x), sscale([cBD[i + 1] if i + 1 < N else Z for i in range(N)], -1), cBD)
    Rk = (rhs[k] - (lhs[k] - Du(x * sp.diff(Dk_a, x) - Dk_a, x * sp.diff(Dk_b, x) - Dk_b))).ex()
    assert not Rk.a.has(Dk_a) and not Rk.b.has(Dk_b) and not Rk.b.has(Dk_a), "D_k leaked into its own forcing"
    newD = []
    for part, val, K in (("bg", Rk.a, Kf[k]), ("pert", Rk.b, dKf[k])):
        r1 = sp.Poly(val, x).coeff_monomial(x)
        if sp.simplify(r1) != 0:
            conds.append((f"sum order {k} {part}: x^1 term (solvability)", sp.simplify(r1)))
        integrand = sp.expand((val - r1 * x) / x ** 2)
        newD.append(sp.expand(x * sp.integrate(integrand, x) + x * K))
    D[k] = Du(newD[0], newD[1])
    print(f"order {k}: dB_{k + 1} = {sp.factor(Bs[k + 1].b)}", flush=True)
    print(f"   dG_{k} = {sp.factor(G[k].b)}", flush=True)
    print(f"   dS_{k} = {sp.collect(S[k].b, x)}", flush=True)
    print(f"   dD_{k} = {sp.collect(D[k].b, x)}", flush=True)
    print(f"   (background D_{k} = {sp.collect(D[k].a, x)})", flush=True)
print("\nconditions found:", len(conds))
for name, cnd in conds:
    print("CONDITION", name, ":\n   ", cnd)
