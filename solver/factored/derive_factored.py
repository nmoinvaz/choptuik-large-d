"""Symbolic and numerical checks of the factored large-D formulation (#1088, notes/LARGE_D_FACTORED.md).

The equations are those of hp/src/model.rs (n = D - 2, d/dtau = omega d/dtheta, horizon at x = 1 where c = 1):

  E1 = x c_x - (n-1) c (a^2 - 1)
  E2 = 2 x a_x - a (p^2 + m^2) - (n-1) a (1 - a^2)
  E3 = x (x - c) p_x + omega x p_th - G p + c p - (n/2) c (p + m)
  E4 = x (x + c) m_x + omega x m_th + G m - c m + (n/2) c (p + m),     G = (n-1) c (a^2 - 1)

Checks, each printed with PASS/FAIL (exit status 1 on any FAIL):
  1. parity variables: with U = p + m, V = (p - m)/x, z = x^2, (E3 + E4)/x and E3 - E4 are
       A = 2z U_z - 2cz V_z + omega U_th - G V
       B = 2z^2 V_z - 2cz U_z + omega z V_th + z V - (G + (n-1)c) U          (all polynomial in z)
     and E1, E2 become 2z c_z - G and 4z a_z - a q - (n-1) a (1 - a^2), q = (U^2 + z V^2)/2
  2. the singular combinations S1 = c A + B and S2 = z A + c B carry the factor 2z(z - c^2) on V_z and U_z alone
  3. beta = (n-1)(1 - 1/a^2) obeys x beta_x = (n-1) q - (n-1+q) beta, and G = c beta / (1 - beta/(n-1))
  4. horizon series in w = 1 - z (c(w=0) = 1, constant g = G(0), D -> i lam), P = U + V, Q = U - V: order 0 of S1
     and of S2 is (i lam - (n/2 - 1 + g)) P0 - (n/2) Q0; at order k >= 1, S1 - S2 carries -2k(1-g) Q_k and no P_k,
     and S2 carries (i lam + k(1-g) - (n/2 - 1 + g)) P_k + (k(1-g) - n/2) Q_k
  5. centre series in z: the explicit order-k formulas of fseries.py (a_k, c_k, U_k, V_k) agree with a direct
     series solution of A, B, E1, E2 to order 4 (theta-independent fields, D -> i lam)
  6. digits lost to cancellation when the free-field horizon-regular solution 2F1(a, b; s + 2 - n/2; w), s = i omega,
     is summed at x = 0.7 as a series in w = 1 - x^2 and as a series in u = 1 - x (hp's variable), n = 40 to 320:
     PASS when the two sums agree (to 1e-8) and neither loses a digit (a measurement: the u-variable does not
     cancel either, so w is chosen for the parity, not for cancellation)
  7. the fast mode's modulus at the matching point: |F| = x^-n (1 - x^2)^(n(1+K)/2) at x = 0.7 grows like
     e^(0.020 n), at x = 1/sqrt(2) it is (1/2)^(nK/2) = 2^(-kappa/2) for K = kappa/n, independent of n

Usage: python3 derive_factored.py [--mutate=N]   (N = 1..7 breaks check N on purpose; that check must FAIL)
"""
import sys

import mpmath as mp
import sympy as sp

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
MUT = int(opts.get("mutate", 0))
results = []


def report(k, name, ok):
    results.append(ok)
    print("%d. %-78s %s" % (k, name, "PASS" if ok else "FAIL"), flush=True)


x, z, n, om, th = sp.symbols("x z n omega theta", positive=True)
Uf = sp.Function("U")
Vf = sp.Function("V")
af = sp.Function("a")
cf = sp.Function("c")

# 1. parity variables
U = Uf(x**2, th)
V = Vf(x**2, th)
a = af(x**2, th)
c = cf(x**2, th)
p = (U + x * V) / 2
m = (U - x * V) / 2
G = (n - 1) * c * (a**2 - 1)
E1 = x * sp.diff(c, x) - (n - 1) * c * (a**2 - 1)
E2 = 2 * x * sp.diff(a, x) - a * (p**2 + m**2) - (n - 1) * a * (1 - a**2)
E3 = x * (x - c) * sp.diff(p, x) + om * x * sp.diff(p, th) - G * p + c * p - (n / 2) * c * (p + m)
E4 = x * (x + c) * sp.diff(m, x) + om * x * sp.diff(m, th) + G * m - c * m + (n / 2) * c * (p + m)
Uz, Vz, az, cz_, Ut, Vt = sp.symbols("Uz Vz az cz Ut Vt")
Us, Vs, As, Cs = sp.symbols("Us Vs As Cs")


def zform(e):
    # explicit chain rule: d/dx f(x^2) = 2x f'(x^2)
    e = e.doit()
    out = e
    for f, sym, dsym, tsym in ((Uf, Us, Uz, Ut), (Vf, Vs, Vz, Vt), (af, As, az, None), (cf, Cs, cz_, None)):
        F = f(x**2, th)
        derivs = [d for d in out.atoms(sp.Subs)] + [d for d in out.atoms(sp.Derivative)]
        for d in derivs:
            if isinstance(d, sp.Subs) and d.expr.func == sp.Derivative and d.expr.expr.func == f:
                out = out.subs(d, dsym)
            elif isinstance(d, sp.Derivative) and d.expr == F and d.variables == (th,):
                out = out.subs(d, tsym)
        out = out.subs(F, sym)
    return sp.expand(out.subs(x, sp.sqrt(z)))


Gz = (n - 1) * Cs * (As**2 - 1)
qz = (Us**2 + z * Vs**2) / 2
A_ = 2 * z * Uz - 2 * Cs * z * Vz + om * Ut - Gz * Vs
B_ = 2 * z**2 * Vz - 2 * Cs * z * Uz + om * z * Vt + z * Vs - (Gz + (n - 1) * Cs) * Us
if MUT == 1:
    B_ = B_ + z * Vs
lhsA = zform(sp.expand((E3 + E4) / x))
lhsB = zform(sp.expand(E3 - E4))
ok1 = sp.simplify(lhsA - sp.expand(A_)) == 0 and sp.simplify(lhsB - sp.expand(B_)) == 0
ok1 = ok1 and sp.simplify(zform(E1) - (2 * z * cz_ - Gz)) == 0
ok1 = ok1 and sp.simplify(zform(E2) - sp.expand(4 * z * az - As * qz - (n - 1) * As * (1 - As**2))) == 0
report(1, "parity variables U, V in z = x^2: A, B, E1, E2 polynomial in z", ok1)

# 2. singular combinations
S1 = sp.expand(Cs * A_ + B_)
S2 = sp.expand(z * A_ + Cs * B_)
f2 = 2 * z * (z - Cs**2)
if MUT == 2:
    f2 = 2 * z * (z - Cs)
ok2 = sp.simplify(S1.coeff(Vz) - f2) == 0 and sp.simplify(S1.coeff(Uz)) == 0
ok2 = ok2 and sp.simplify(S2.coeff(Uz) - f2) == 0 and sp.simplify(S2.coeff(Vz)) == 0
report(2, "S1 = cA + B, S2 = zA + cB: 2z(z - c^2) on V_z, U_z alone", ok2)

# 3. beta and G
X = sp.symbols("X", positive=True)
af1 = sp.Function("a1")(X)
qq = sp.Function("q")(X)
nn = sp.symbols("n", positive=True)
ax_rhs = af1 * (qq + (nn - 1) * (1 - af1**2)) / (2 * X)  # 2x a_x = a q + (n-1) a (1 - a^2)
beta = (nn - 1) * (1 - 1 / af1**2)
lhs3 = X * sp.diff(beta, X).subs(sp.Derivative(af1, X), ax_rhs)
rhs3 = (nn - 1) * qq - (nn - 1 + qq) * beta
if MUT == 3:
    rhs3 = rhs3 + qq * beta
cc = sp.symbols("cc")
G3 = (nn - 1) * cc * (af1**2 - 1)
ok3 = sp.simplify(lhs3 - rhs3) == 0 and sp.simplify(G3 - cc * beta / (1 - beta / (nn - 1))) == 0
report(3, "x beta_x = (n-1) q - (n-1+q) beta, G = c beta / (1 - beta/(n-1))", ok3)

# 4. horizon recurrence in w = 1 - z for one theta-mode (D -> i lam), constant g, c = 1 + c1 w + c2 w^2, G = g + ...
w = sp.symbols("w")
lam, g = sp.symbols("lam g")
Kh = 3
Uc = sp.symbols("U0:%d" % (Kh + 1))
Vc = sp.symbols("V0:%d" % (Kh + 1))
gc = sp.symbols("g1:%d" % (Kh + 1))
Gw = g + sum(gc[j - 1] * w**j for j in range(1, Kh + 1))
# c from 2 z c_z = G with c(0) = 1: -2 (1 - w) c_w = G
cw = sp.Integer(1)
cs_ = sp.symbols("c1:%d" % (Kh + 2))
cw = 1 + sum(cs_[j - 1] * w**j for j in range(1, Kh + 2))
ceq = sp.expand(-2 * (1 - w) * sp.diff(cw, w) - Gw)
csol = sp.solve([ceq.coeff(w, j) for j in range(0, Kh + 1)], cs_[: Kh + 1], dict=True)[0]
cw = cw.subs(csol).subs(cs_[Kh + 1 - 1], 0)
Uw = sum(Uc[j] * w**j for j in range(Kh + 1))
Vw = sum(Vc[j] * w**j for j in range(Kh + 1))
zw = 1 - w
# z-derivative = -d/dw; theta derivative -> i lam / omega, and omega d/dtheta -> i lam
dz = lambda f: -sp.diff(f, w)  # noqa: E731
Aw = 2 * zw * dz(Uw) - 2 * cw * zw * dz(Vw) + sp.I * lam * Uw - Gw * Vw
Bw = 2 * zw**2 * dz(Vw) - 2 * cw * zw * dz(Uw) + sp.I * lam * zw * Vw + zw * Vw - (Gw + (n - 1) * cw) * Uw
S1w = sp.expand(cw * Aw + Bw)
S2w = sp.expand(zw * Aw + cw * Bw)
o0 = sp.expand(S2w.coeff(w, 0))
P0, Q0 = sp.symbols("P0 Q0")
o0 = sp.expand(o0.subs({Uc[0]: (P0 + Q0) / 2, Vc[0]: (P0 - Q0) / 2}))
pred0 = sp.expand((sp.I * lam - (n / 2 - 1 + g)) * P0 - (n / 2) * Q0)
if MUT == 4:
    pred0 = pred0 + P0
ok4 = sp.simplify(o0 - pred0) == 0 and sp.simplify(sp.expand(S1w.coeff(w, 0)).subs(
    {Uc[0]: (P0 + Q0) / 2, Vc[0]: (P0 - Q0) / 2}) - pred0) == 0
ok4k = True
for k in range(1, Kh):
    Pk, Qk = sp.symbols("Pk Qk")
    e1 = sp.expand(S1w.coeff(w, k)).subs({Uc[k]: (Pk + Qk) / 2, Vc[k]: (Pk - Qk) / 2})
    e2 = sp.expand(S2w.coeff(w, k)).subs({Uc[k]: (Pk + Qk) / 2, Vc[k]: (Pk - Qk) / 2})
    d12 = sp.expand(e1 - e2)
    # coefficient of Qk in e1 - e2 is -2k(1-g); no Pk
    okq = sp.simplify(d12.coeff(Qk) + 2 * k * (1 - g)) == 0 and sp.simplify(d12.coeff(Pk)) == 0
    okp = sp.simplify(sp.expand(e2).coeff(Pk) - (sp.I * lam + k * (1 - g) - (n / 2 - 1 + g))) == 0
    okqk = sp.simplify(sp.expand(e2).coeff(Qk) - (k * (1 - g) - n / 2)) == 0
    ok4k = ok4k and okq and okp and okqk
report(4, "horizon: order 0 (i lam - (n/2-1+g)) P0 - (n/2) Q0 = 0; order-k operators", ok4 and ok4k)

# 5. centre series explicit formulas against a direct series solve (theta-independent, D -> i lam)
Kc = 4
c0s, V0s = sp.symbols("c0 V0")
nv = sp.Rational(7, 1)
lamv = sp.Rational(3, 2)


def centre_explicit(nv, lamv, c0v, V0v, K, mut=False):
    Uk = [sp.Integer(0)] * (K + 1)
    Vk = [sp.Integer(0)] * (K + 1)
    ak = [sp.Integer(0)] * (K + 1)
    ck = [sp.Integer(0)] * (K + 1)
    Gk = [sp.Integer(0)] * (K + 1)
    a2 = [sp.Integer(0)] * (K + 1)
    ak[0] = sp.Integer(1)
    ck[0] = c0v
    Vk[0] = V0v
    a2[0] = sp.Integer(1)
    D = sp.I * lamv
    for k in range(1, K + 1):
        U2 = sum(Uk[j] * Uk[k - j] for j in range(1, k))
        V2 = sum(Vk[j] * Vk[k - 1 - j] for j in range(0, k))
        qk = [None] * (k + 1)
        for mm in range(1, k + 1):
            qk[mm] = (sum(Uk[j] * Uk[mm - j] for j in range(1, mm)) + sum(Vk[j] * Vk[mm - 1 - j] for j in range(0, mm))) / 2
        aq = sum(ak[j] * qk[k - j] for j in range(0, k))
        S2_ = sum(ak[j] * ak[k - j] for j in range(1, k))
        S3_ = sum(ak[j] * a2[k - j] for j in range(1, k))
        ak[k] = (aq - (nv - 1) * (S2_ + S3_)) / (4 * k + 2 * (nv - 1))
        a2[k] = 2 * ak[k] + S2_
        ck[k] = (nv - 1) * sum(ck[j] * a2[k - j] for j in range(0, k)) / (2 * k)
        Gk[k] = 2 * k * ck[k]
        num = 2 * (k - 1) * Vk[k - 1] + D * Vk[k - 1] + Vk[k - 1]
        num -= sum((2 * (k - j) * ck[j] + Gk[j] + (nv - 1) * ck[j]) * Uk[k - j] for j in range(1, k))
        Uk[k] = num / (ck[0] * (2 * k + nv - 1 + (1 if mut else 0)))
        num = 2 * k * Uk[k] + D * Uk[k] - sum(2 * ck[j] * (k - j) * Vk[k - j] for j in range(1, k))
        num -= sum(Gk[j] * Vk[k - j] for j in range(1, k + 1))
        Vk[k] = num / (2 * ck[0] * k)
    return Uk, Vk, ak, ck


c0v = sp.Rational(9, 10)
V0v = sp.Rational(1, 3)
Uk, Vk, ak, ck = centre_explicit(nv, lamv, c0v, V0v, Kc, mut=(MUT == 5))
Zs = sp.symbols("zz")
Us_ = sum(Uk[j] * Zs**j for j in range(Kc + 1))
Vs_ = sum(Vk[j] * Zs**j for j in range(Kc + 1))
as_ = sum(ak[j] * Zs**j for j in range(Kc + 1))
cs2 = sum(ck[j] * Zs**j for j in range(Kc + 1))
Gs = (nv - 1) * cs2 * (as_**2 - 1)
qs = (Us_**2 + Zs * Vs_**2) / 2
DD = sp.I * lamv
rA = 2 * Zs * sp.diff(Us_, Zs) - 2 * cs2 * Zs * sp.diff(Vs_, Zs) + DD * Us_ - Gs * Vs_
rB = 2 * Zs**2 * sp.diff(Vs_, Zs) - 2 * cs2 * Zs * sp.diff(Us_, Zs) + DD * Zs * Vs_ + Zs * Vs_ - (Gs + (nv - 1) * cs2) * Us_
rC = 2 * Zs * sp.diff(cs2, Zs) - Gs
rD = 4 * Zs * sp.diff(as_, Zs) - as_ * qs - (nv - 1) * as_ * (1 - as_**2)
ok5 = True
for r, top in ((rA, Kc), (rB, Kc), (rC, Kc), (rD, Kc)):
    rr = sp.expand(r)
    for j in range(0, top + 1):
        if sp.simplify(rr.coeff(Zs, j)) != 0:
            ok5 = False
report(5, "centre series: explicit a_k, c_k, U_k, V_k solve A, B, E1, E2 to order 4", ok5)

# 6. cancellation in the free-field horizon-regular solution at the matching point, u against w
mp.mp.dps = 60
om_v = mp.mpf(2) * mp.pi / mp.mpf("1.6696580641")  # omega near D 400; the j = 1 harmonic


def loss_digits(nv, uK, wK, mut=False):
    s = mp.mpc(0, om_v)
    A1 = (1 + s) / 2
    B1 = (2 + s) / 2
    C1 = s + 2 - mp.mpf(nv) / 2
    N = int(3 * nv + 200)
    t = [mp.mpc(1)]
    for k in range(N):
        t.append(t[-1] * (A1 + k) * (B1 + k) / ((C1 + k) * (k + 1)))
    # w-sum
    sw = mp.fsum(t[k] * wK**k for k in range(N + 1))
    aw = mp.fsum(abs(t[k]) * wK**k for k in range(N + 1))
    # u-coefficients of f(w(u)), w = 2u - u^2 = u (2 - u): coefficient of u^m is sum_k t_k C(k, m - k) 2^(2k - m) (-1)^(m - k)
    Nu = N
    cu = [mp.mpc(0)] * (Nu + 1)
    for k in range(N + 1):
        if t[k] == 0:
            continue
        # (2u - u^2)^k = u^k sum_i C(k, i) 2^(k-i) (-u)^i
        b = mp.mpf(2) ** k
        for i in range(0, k + 1):
            mm = k + i
            if mm > Nu:
                break
            cu[mm] += t[k] * b * ((-1) ** i if not mut else 1)
            b = b * (k - i) / (i + 1) / 2
    su = mp.fsum(cu[mm] * uK**mm for mm in range(Nu + 1))
    au = mp.fsum(abs(cu[mm]) * uK**mm for mm in range(Nu + 1))
    return float(mp.log10(aw / abs(sw))), float(mp.log10(au / abs(su))), complex(sw), complex(su)


uK = mp.mpf("0.3")
wK = 1 - (1 - uK) ** 2
lw, lu, same = [], [], []
for nv6 in (40, 80, 160, 320):
    a_, b_, swv, suv = loss_digits(nv6, uK, wK, mut=(MUT == 6))
    lw.append(a_)
    lu.append(b_)
    print("   n %4d: digits lost, w-sum %.2f, u-sum %.2f (same function: %.1e apart)" % (nv6, a_, b_, abs(swv - suv) / abs(swv)))
    same.append(abs(swv - suv) / abs(swv))
ok6 = max(lw) < 1.0 and max(lu) < 1.0 and max(same) < 1e-8
report(6, "free field at x 0.7: w-sum = u-sum, neither loses a digit to cancellation", ok6)

# 7. the fast mode's modulus at the matching point
kap = mp.mpf("7.7")
ok7 = True
for nv7 in (100, 1000, 10000):
    K7 = kap / nv7
    f07 = mp.mpf(nv7) * (-mp.log(mp.mpf("0.7")) + (1 + K7) / 2 * mp.log(1 - mp.mpf("0.49")))
    xs = 1 / mp.sqrt(2) if MUT != 7 else mp.mpf("0.7")
    fs = mp.mpf(nv7) * (-mp.log(xs) + (1 + K7) / 2 * mp.log(1 - xs**2))
    print("   n %5d: ln|F| at 0.7 = %8.3f, at 1/sqrt2 = %8.4f (kappa/2 ln 1/2 = %.4f)" % (nv7, f07, fs, -kap / 2 * mp.log(2)))
    ok7 = ok7 and abs(fs + kap / 2 * mp.log(2)) < 1e-30
report(7, "ln|F(1/sqrt2)| = -(kappa/2) ln 2, independent of n; at 0.7 it grows like 0.020 n", ok7)

print("ALL PASS" if all(results) else "SOME FAIL")
sys.exit(0 if all(results) else 1)
