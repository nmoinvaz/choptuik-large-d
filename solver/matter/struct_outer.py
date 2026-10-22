"""(c) The outer problem (x = r/(-t) = O(1)) at large n, with sympy.

    nice -n 10 python3 struct_outer.py [N]          (N = number of orders, default 3; about 1-3 minutes)

1. The self-similar second-order field equation, from the covariant one (kappa = 2n, K = k0 constant, V = v(phi)/r^2):
       (1 + d_tau + x d_x)[(phi_tau + x phi_x)/c] = x^-n d_x(x^n c phi_x) - c b v'(phi) / (k0 x^2),
       x b_x / b = k0 (p^2 + m^2) + 4 b v + (n-1)(1-b),   x c_x / c = k0 (p^2 + m^2) - x b_x / b,
       p^2 + m^2 = 2 x^2 phi_x^2 + 2 x^2 (phi_tau + x phi_x)^2 / c^2.
2. Flat space: the l-field is the s-wave field in n + 2l dimensions (phi = r^l psi), and the light-cone exponent of the
   self-similar flat equation, phi ~ (1 - x)^{n/2 - sigma} for phi ~ e^{sigma tau}, does not depend on L: a potential
   term has no derivative and cannot enter the indicial equation at the characteristic x = 1, however large it is.
   Hence Proposition 2 carries over: the light-cone coefficient is C(n + 2l, sigma + l), finite and non-zero.
3. The formal outer expansion (the analogue of Proposition 1), eps = 1/n, theta = omega tau:
       multiplet, l fixed:  phi = eps^{(l+1)/2} x^l [G(theta) + sum_k eps^k Psi_k(theta, x)],
       wave map, eta fixed: u   = eps^{1/2}     x   [G(theta) + sum_k eps^k Psi_k(theta, x)],
   b = 1 + O(eps^{l+1}), c = 1 - O(eps^{l+1}) (c(theta, 1) = 1). Order by order every equation is solved by even
   polynomials in x, with a free function K_k(theta) (the x^0 term of Psi_k), for ANY G, omega_0, omega_1, ...: the
   only possible obstruction, an x^0 term in the source of x d_x (it would give log x), is checked to be absent.
   Gravity is weaker than for the scalar: c - 1 = O(eps^{l+1}) at x = O(1), so n c_x(1) = O(eps^l) and the gravitational
   shift of the light-cone exponent (3.7 for l = 0) vanishes at D = infinity for l >= 1.
4. l = lambda n: the flat problem is WKB at x = O(1). phi = A e^{n S(x)}, P = x S': (1 - x^2) P^2 + P - lambda (1 + lambda) = 0
   (omega-independent; time derivatives first enter the O(n) transport equation). The regular branch (P(0) = lambda) is
   analytic through the light cone (P(1) = lambda(1 + lambda)) and meets the other branch at the turning point
   x_t^2 = 1 + 1/(4 lambda (1 + lambda)) > 1; x_t ~ sqrt(n / (4 l)) as lambda -> 0 (the inner region of fixed l).
"""
import sys
import sympy as sp

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ALL = []


def check(label, expr, simp=sp.simplify):
    z = simp(expr)
    ok = (z == 0)
    ALL.append(ok)
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + ("" if ok else f": residual {z}"))


# ---------------------------------------------------------------- 1. self-similar field equation
print("1. Self-similar second-order field equation from the covariant one")
tau, x = sp.symbols("tau x", positive=True)
n, k0 = sp.symbols("n k0", positive=True)
# jet variables: F (the field), C (= c), B (= b) and their tau-, x-derivatives up to second order
JN = {}
for nm in ("F", "C", "B"):
    for i in range(3):
        for j in range(3 - i):
            JN[(nm, i, j)] = sp.Symbol(f"{nm}_{'t' * i}{'x' * j}" if i + j else nm)


def Dj(e, var):
    """total tau- (var = 0) or x-derivative (var = 1) of an expression in the jet variables"""
    out = sp.diff(e, tau if var == 0 else x)
    for (nm, i, j), sym in JN.items():
        nxt = (nm, i + 1, j) if var == 0 else (nm, i, j + 1)
        if nxt in JN:
            out += sp.diff(e, sym) * JN[nxt]
        elif sp.diff(e, sym) != 0:
            raise RuntimeError("jet order exceeded")
    return out


def d_t(e):          # d/dt = e^tau (d_tau + x d_x)
    return sp.exp(tau) * (Dj(e, 0) + x * Dj(e, 1))


def d_r(e):          # d/dr = e^tau d_x
    return sp.exp(tau) * Dj(e, 1)


F0, C0, B0 = JN[("F", 0, 0)], JN[("C", 0, 0)], JN[("B", 0, 0)]
vp = sp.Function("vprime")(F0)            # v'(phi), a generic function of the field
rr = x * sp.exp(-tau)
Pi = d_t(F0) / C0                          # Pi = a phi_t / alpha = phi_t / c
Phi = d_r(F0)
cov = d_t(k0 * Pi) - (d_r(rr ** n * C0 * k0 * Phi) / rr ** n - C0 * B0 * vp / rr ** 2)
inner_ = (JN[("F", 1, 0)] + x * JN[("F", 0, 1)]) / C0
ss = (inner_ + Dj(inner_, 0) + x * Dj(inner_, 1)) - (Dj(x ** n * C0 * JN[("F", 0, 1)], 1) / x ** n
                                                     - C0 * B0 * vp / (k0 * x ** 2))
check("covariant Pi_t equation e^{-2 tau}/k0 = the self-similar operator above",
      sp.simplify(sp.expand(cov * sp.exp(-2 * tau) / k0 - ss)))

# ---------------------------------------------------------------- 2. flat: l <-> n + 2 l, light-cone exponent
print("\n2. Flat space")
r = sp.Symbol("r", positive=True)
l, sig = sp.symbols("l sigma", positive=True)
psi = sp.Function("psi")(r)
ph = r ** l * psi
check("phi_rr + n phi_r/r - l(l+n-1) phi/r^2 = r^l (psi_rr + (n+2l) psi_r/r)",
      sp.diff(ph, r, 2) + n * sp.diff(ph, r) / r - l * (l + n - 1) * ph / r ** 2
      - r ** l * (sp.diff(psi, r, 2) + (n + 2 * l) * sp.diff(psi, r) / r))
f = sp.Function("f")(x)
L = sp.Symbol("L")
flat = x ** 2 * f.diff(x, 2) + (2 * sig + 2) * x * f.diff(x) + sig * (sig + 1) * f \
    - (f.diff(x, 2) + n * f.diff(x) / x - L * f / x ** 2)        # (1 + d_tau + x d_x)(d_tau + x d_x) phi - flat RHS
rho_ = sp.Symbol("rho")
fl = sp.expand(flat)
P2, P1 = fl.coeff(f.diff(x, 2)), fl.coeff(f.diff(x))
P0 = sp.expand(fl - P2 * f.diff(x, 2) - P1 * f.diff(x)).coeff(f)
p0 = sp.limit((x - 1) * P1 / P2, x, 1)
q0 = sp.limit((x - 1) ** 2 * P0 / P2, x, 1)
ind = sp.expand(rho_ * (rho_ - 1) + p0 * rho_ + q0)          # Frobenius at the regular singular point x = 1
print("   indicial polynomial at x = 1 (phi ~ (x-1)^rho):", sp.factor(ind), "  (q0 =", q0, ")")
check("exponents at the light cone: rho = 0 and rho = n/2 - sigma, independent of L",
      0 if set(sp.solve(ind, rho_)) == {0, n / 2 - sig} else 1)
check("indicial polynomial has no L", sp.diff(ind, L))

# ---------------------------------------------------------------- 3. formal outer expansion
print("\n3. Formal outer expansion (eps = 1/n = d^2; G and the K_k arbitrary functions of theta, omega_k arbitrary)")
JMAX = 16
gj = sp.symbols(f"g0:{JMAX}")                       # G, G', G'', ...  (theta-derivatives as jet symbols)
NEXT = {gj[i]: gj[i + 1] for i in range(JMAX - 1)}
Kj = {}


def Kfun(k):
    syms = sp.symbols(f"K{k}_0:{JMAX}")
    for i in range(JMAX - 1):
        NEXT[syms[i]] = syms[i + 1]
    return syms


def dth(e):
    e = sp.expand(e)
    return sp.expand(sum(sp.diff(e, s_) * nx for s_, nx in NEXT.items() if e.has(s_)))


def smul(a, b, M):
    out = [sp.Integer(0)] * (M + 1)
    for i, ai in enumerate(a[:M + 1]):
        if ai == 0:
            continue
        for j, bj in enumerate(b[:M + 1 - i]):
            if bj != 0:
                out[i + j] += ai * bj
    return [sp.expand(o) for o in out]


def sadd(*args, M):
    out = [sp.Integer(0)] * (M + 1)
    for a_ in args:
        for i, ai in enumerate(a_[:M + 1]):
            out[i] += ai
    return [sp.expand(o) for o in out]


def sscale(a, c_):
    return [sp.expand(c_ * ai) for ai in a]


def sshift(a, k, M):
    return ([sp.Integer(0)] * k + list(a))[:M + 1] + [sp.Integer(0)] * max(0, M + 1 - len(a) - k)


def sinv(a, M):
    """1/a for a series with a[0] = 1"""
    out = [sp.Integer(1)] + [sp.Integer(0)] * M
    for k in range(1, M + 1):
        out[k] = sp.expand(-sum(a[i] * out[k - i] for i in range(1, k + 1) if i < len(a)))
    return out


def sdx(a):
    return [sp.expand(sp.diff(ai, x)) for ai in a]


def sdth(a):
    return [dth(ai) for ai in a]


def spow(a, p, M):
    out = [sp.Integer(1)] + [sp.Integer(0)] * M
    for _ in range(p):
        out = smul(out, a, M)
    return out


def outer(model, l, N, eta=None):
    """Order-by-order solution to relative order eps^N. model 'mult' (l-multiplet) or 'wm' (wave map, l = 1)."""
    M = 2 * N
    beta2 = l + 1 if model == "mult" else 1          # field = d^beta2 x^l Psi
    om = [sp.Integer(0)] * (M + 1)
    for k in range(N + 1):
        om[2 * k] = sp.Symbol(f"w{k}")
    Psi = [gj[0]] + [sp.Integer(0)] * M
    Bs = [sp.Integer(0)] * (M + 1)                   # b - 1
    Cs = [sp.Integer(0)] * (M + 1)                   # c - 1
    one = [sp.Integer(1)] + [sp.Integer(0)] * M
    eps_ = sshift(one, 2, M)
    ok_all = True
    for k in range(1, N + 1):
        # ---- metric, iterated to self-consistency at the current Psi (orders < k)
        ph = [sp.expand(x ** l * p_) for p_ in Psi]
        phx, pht = sdx(ph), sdth(ph)
        for _ in range(N + 1):
            cser = sadd(one, Cs, M=M)
            cinv = sinv(cser, M)
            vel = sadd(smul(om, pht, M), sscale(phx, x), M=M)
            Q = sadd(sscale(smul(phx, phx, M), 2 * x ** 2), sscale(smul(smul(vel, vel, M), smul(cinv, cinv, M), M), 2 * x ** 2), M=M)
            if model == "mult":
                kPM = sshift(Q, 2 * beta2, M)                                   # k0 (p^2 + m^2)
                e4v = sshift(smul([2 * l, 0, 2 * l * (l - 1)], smul(ph, ph, M), M), 2 * beta2, M)   # eps 4 v = 2 eps L phi^2
            else:
                kPM = sscale(sshift(Q, 2 * beta2 + 2, M), eta)                  # (eta eps) (p^2 + m^2)
                u2 = smul(ph, ph, M)
                s2 = sadd(sshift(u2, 2, M), sscale(sshift(spow(u2, 2, M), 4, M), -sp.Rational(1, 3)),
                          sscale(sshift(spow(u2, 3, M), 6, M), sp.Rational(2, 45)), M=M)   # sin^2(d phi), d-series
                e4v = sscale(sshift(s2, 2, M), 2 * eta)                         # eps 2 eta sin^2 u
            bser = sadd(one, Bs, M=M)
            xbx_b = smul(sscale(sdx(Bs), x), sinv(bser, M), M)
            # (1 - eps) beta = eps k0 (p^2+m^2) + (1 + beta) eps 4 v - eps x beta_x / (1 + beta)
            rhs = sadd(sshift(kPM, 2, M), smul(bser, e4v, M), sscale(sshift(xbx_b, 2, M), -1), M=M)
            Bs = smul(rhs, sinv(sadd(one, sscale(eps_, -1), M=M), M), M)
            bser = sadd(one, Bs, M=M)
            xbx_b = smul(sscale(sdx(Bs), x), sinv(bser, M), M)
            src = smul(cser, sadd(kPM, sscale(xbx_b, -1), M=M), M)              # x c_x = c (k0 (p^2+m^2) - x b_x/b)
            for j, sj in enumerate(src):
                if sp.expand(sj).subs(x, 0) != 0:
                    print(f"   order d^{j}: OBSTRUCTION in the c equation, x^0 term {sp.expand(sj).subs(x, 0)}")
                    ok_all = False
            Cs = [sp.expand(sp.integrate(sp.expand(sj / x), (x, 1, x))) for sj in src]
        # ---- field equation, divided by d^beta2:
        # E = d^2 x^2 [I + w I_th + x I_x - (c ph_x)_x] - c x ph_x + c b vp,  I = (w ph_th + x ph_x)/c
        def field(Psi_):
            ph_ = [sp.expand(x ** l * p_) for p_ in Psi_]
            phx_, pht_ = sdx(ph_), sdth(ph_)
            cser_ = sadd(one, Cs, M=M)
            I_ = smul(sadd(smul(om, pht_, M), sscale(phx_, x), M=M), sinv(cser_, M), M)
            t1 = sadd(I_, smul(om, sdth(I_), M), sscale(sdx(I_), x), sscale(sdx(smul(cser_, phx_, M)), -1), M=M)
            t1 = sshift(sscale(t1, x ** 2), 2, M)
            t2 = sscale(smul(cser_, phx_, M), -x)
            if model == "mult":
                vp = smul([l, 0, l * (l - 1)], ph_, M)                           # eps L phi (divided by d^beta2)
            else:
                vp = sadd(ph_, sscale(sshift(spow(ph_, 3, M), 2, M), -sp.Rational(2, 3)),
                          sscale(sshift(spow(ph_, 5, M), 4, M), sp.Rational(2, 15)), M=M)   # sin(2u)/2 / d
            t3 = smul(smul(cser_, sadd(one, Bs, M=M), M), vp, M)
            return sadd(t1, t2, t3, M=M)
        Rk = sp.expand(field(Psi)[2 * k])                 # the order-d^{2k} equation with Psi_k = 0
        Kk = Kfun(k)
        if sp.expand(Rk / x ** l).subs(x, 0) != 0:
            print(f"   order {k}: OBSTRUCTION, x^0 term {sp.expand(Rk / x ** l).subs(x, 0)}")
            ok_all = False
        Pk = sp.expand(sp.integrate(sp.expand(Rk / x ** (l + 1)), x)) + Kk[0]
        degs = sp.Poly(Pk, x).monoms()
        odd = [m_[0] for m_ in degs if m_[0] % 2]
        Psi[2 * k] = Pk
        res = sp.expand(field(Psi)[2 * k])
        good = (res == 0) and not odd
        ok_all = ok_all and good
        print(f"   order {k}: no x^0 obstruction; Psi_{k} even polynomial of degree {sp.degree(Pk, x)} + K_{k}(theta);"
              f" equation satisfied: {res == 0}")
    ALL.append(ok_all)
    return Psi, Bs, Cs


for (model, l_, N_, eta_) in (("mult", 1, max(N, 3), None), ("mult", 2, max(N, 4), None),
                              ("wm", 1, max(N, 3), sp.Symbol("eta", positive=True))):
    print(f"\n   {'multiplet l = ' + str(l_) if model == 'mult' else 'wave map (eta symbolic)'}, to relative order eps^{N_}:")
    Psi, Bs, Cs = outer(model, l_, N_, eta_)
    print("   Psi_1 =", sp.factor(Psi[2]))
    beta2 = l_ + 1 if model == "mult" else 1
    lowB = min(j for j in range(len(Bs)) if Bs[j] != 0)
    lowC = min(j for j in range(len(Cs)) if Cs[j] != 0)
    # absolute order of the metric functions in eps: index j in d (they are not divided by d^beta2)
    print(f"   b - 1 = eps^{sp.Rational(lowB, 2)} [{sp.factor(Bs[lowB])}] + ...")
    print(f"   c - 1 = eps^{sp.Rational(lowC, 2)} [{sp.factor(Cs[lowC])}] + ...")
    check(f"gravity at x = O(1): b - 1 and c - 1 start at eps^{l_ + 1}", (lowB - 2 * (l_ + 1)) ** 2 + (lowC - 2 * (l_ + 1)) ** 2)
    print(f"   so n c_x(1) = O(eps^{l_}): no gravitational shift of the light-cone exponent at D = infinity")

# ---------------------------------------------------------------- 4. l = lambda n: eikonal
print("\n4. l = lambda n: eikonal of the flat self-similar equation")
lam, nn = sp.symbols("lambda n", positive=True)
S = sp.Function("S")(x)
A = sp.Function("A")(tau, x)
ph_ = A * sp.exp(nn * S)
Lq = lam * nn * (lam * nn + nn - 1)
D1 = sp.diff(ph_, tau) + x * sp.diff(ph_, x)
lhs = D1 + sp.diff(D1, tau) + x * sp.diff(D1, x)
rhs = sp.diff(x ** nn * sp.diff(ph_, x), x) / x ** nn - Lq * ph_ / x ** 2
E = sp.expand(sp.simplify((lhs - rhs) * x ** 2 / ph_))
Pq = sp.Symbol("P")
eik = sp.expand(E.coeff(nn, 2)).subs(sp.diff(S, x), Pq / x)
print("   O(n^2):", sp.factor(eik))
check("eikonal: (1 - x^2) P^2 + P - lambda(1 + lambda) = 0", sp.expand(eik + ((1 - x ** 2) * Pq ** 2 + Pq - lam * (1 + lam))))
tr = sp.expand(E.coeff(nn, 1))
print("   O(n): contains d_tau A (transport):", tr.has(sp.Derivative(A, tau)))
check("the eikonal has no omega/tau dependence", sp.diff(eik, tau))
Preg = (-1 + sp.sqrt(1 + 4 * (1 - x ** 2) * lam * (1 + lam))) / (2 * (1 - x ** 2))
check("regular branch P(0) = lambda", sp.factor(sp.limit(Preg, x, 0) - lam).subs(sp.sqrt(4 * lam ** 2 + 4 * lam + 1), 2 * lam + 1))
check("regular branch is analytic at the light cone, P(1) = lambda (1 + lambda)", sp.limit(Preg, x, 1) - lam * (1 + lam))
xt2 = 1 + 1 / (4 * lam * (1 + lam))
check("turning point (double root in P) at x_t^2 = 1 + 1/(4 lambda (1 + lambda))",
      sp.discriminant((1 - x ** 2) * Pq ** 2 + Pq - lam * (1 + lam), Pq).subs(x ** 2, xt2))
ll = sp.Symbol("l", positive=True)
print("   x_t as lambda = l/n -> 0:", sp.series(sp.sqrt(xt2.subs(lam, ll / nn)), nn, sp.oo, 1).removeO(),
      " (the fixed-l inner region x ~ sqrt(n))")
# growth of |phi| = e^{n S} along the regular branch: S' = P/x > 0, so gravity (proportional to phi^2) peaks at x_t
import mpmath as mp
for lv in (0.25, 1.0, 4.0):
    Pn = sp.lambdify(x, Preg.subs(lam, lv), "mpmath")
    xt = mp.sqrt(1 + 1 / (4 * lv * (1 + lv)))
    dS = mp.re(mp.quad(lambda z: Pn(z) / z, [1, xt]))
    print(f"   lambda = {lv}: x_t = {mp.nstr(xt, 6)},  S(x_t) - S(1) = {mp.nstr(dS, 6)}"
          f"  -> gravity at the light cone suppressed by e^(-2 n * {mp.nstr(dS, 4)}) relative to x_t")

print("\nall ok" if all(ALL) else "\nSOME CHECKS FAILED")
