"""The factored horizon segment (notes/FACTORED_HORIZON.md, #491): the mass mode factored at the last node, checked
symbolically against the unfactored equations, and a linear model of the fast mode's shadow in the truncated horizon
series.

Run from the repository root:
    python3 solver/factored_horizon_sympy.py                  # all checks; exit 0 when every one passes
    python3 solver/factored_horizon_sympy.py --mutate=sign    # the control: the transport's sign flipped, exit 1
    python3 solver/factored_horizon_sympy.py check            # the symbolic and exact checks only (S1-S7)
    python3 solver/factored_horizon_sympy.py model            # the shadow model only (M1-M4)
    python3 solver/factored_horizon_sympy.py model --mutate=ks    # the model's control: no truncation, exit 1

The checks (each prints PASS or FAIL):
  S1  the variables B = 1/a^2, C = c a^2: from hp's a and c equations (hp/src/model.rs), x C_x = C q and
      x B_x = (n-1)(1 - B) - q B exactly, with q = p^2 + m^2, G = (n-1) C (1 - B), c = C B.
  S2  the mass mode chi = (x/x_K)^-(n-1) exp(-int_{x_K}^x q/t dt) solves B's homogeneous equation, chi(x_K) = 1,
      B_s + mu chi solves B's equation when B_s does, and chi(x_K)/chi(1) is the gain.
  S3  the series in u = 1 - x at low order: hp's forward recurrences for a and c against those of B and C (symbolic,
      orders 0-4); the split B = B_sl + gamma chi_hat (B_sl from the truncated system with b_{K+1} = 0, chi_hat the
      mass mode normalized to sum 1 at u0) against the forward recurrence (exact rationals, n = 37, K = 10); and the
      rounding of the two at n = 240, K = 130 in 30-digit arithmetic against 200 digits.
  S4  the transport of the mass mode's amplitude, by Proposition 3's method from the exact equations E1-E4 with
      Proposition 1's background (as solver/largeD_transport_sympy.py): delta b = x^(1-n) (beta0 + eps beta1), with
      delta p = delta m and delta c = -delta b at leading order and x beta' = -2 eps x^2 F^2 beta + O(eps^2).
  S5  the gain before and after the factoring, from S2 and S4: x_K^-(n-1) exp(eps F^2 (1 - x_K^2) + O(eps^2)), and 1.
  S6  affine invariance: Newton's step under z = T y and F -> S F, and the invariance of e_Delta^T J^-1 under any
      change of the unknowns that keeps Delta; a nonlinear change agrees in the first step to first order only.
  S7  the truncated matching of the linear model: y_reg = A y_h + C y_f, y_h^(K) = (y_reg - f C y_f)/A, so the
      matching determinant is -(f C/A) det(y_reg, y_f) and its grip on omega is f times the continuum's.
The model (numerical evidence; its tolerances were set after a prototype run, so they describe, not test):
  M1  the linear model's horizon-regular series (Proposition 2's equation in u) against 2F1 at 1 - x^2.
  M2  the peak order of its transient against hp/data/horizon_radius.csv (k_peak, #693), at n_lin = 2 (Re s + 1).
  M3  f = W_K / W_inf, the share of the continuum's matching determinant that a series truncated at K keeps, at
      u0 = 0.3, and the smallest K with |ln f| < 0.01.
  M4  the rates in D of ln|W_130| for mu = 1 and 3 against the weak sigma of raw I488-mac (994f6f21, D237_v2.json and
      D242_kf3.json), and the shift for the last node 0.70 -> 0.72 against #537's raw shift of ln sigma_3, -8.96
      (notes/LARGE_D_WALL.md).
--mutate=sign flips the sign of the mass mode's transport (q -> -q in chi and in the target of S4): S2, S3, S4 and S5
must fail. --mutate=ks evaluates the model without truncation (K = 600): M4 must fail.
"""
import csv
import os
import sys

import mpmath as mp
import sympy as sp

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
MODE = ARGS[0] if ARGS else "all"
MUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), "")
SIGN = -1 if MUT == "sign" else 1
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = []


def report(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{name}: {'PASS' if ok else 'FAIL'}{'  ' + detail if detail else ''}")
    sys.stdout.flush()


def is_zero(e):
    return sp.simplify(sp.expand(e)) == 0


# ------------------------------------------------------------------------------------------------------------------
def s1():
    x, n = sp.symbols("x n", positive=True)
    a, c, q = (sp.Function(f)(x) for f in ("a", "c", "q"))
    # hp/src/model.rs: x c_x = (n-1)(c a^2 - c), 2 x a_x = a (p^2 + m^2) + (n-1)(a - a a^2), q = p^2 + m^2
    cx = (n - 1) * (c * a**2 - c) / x
    ax = (a * q + (n - 1) * (a - a**3)) / (2 * x)
    sub = {sp.Derivative(c, x): cx, sp.Derivative(a, x): ax}
    B, C = 1 / a**2, c * a**2
    eqC = (x * sp.diff(C, x)).subs(sub) - C * q
    eqB = (x * sp.diff(B, x)).subs(sub) - ((n - 1) * (1 - B) - q * B)
    G = (n - 1) * (c * a**2 - c)
    report("S1 x C_x = C q", is_zero(eqC))
    report("S1 x B_x = (n-1)(1-B) - q B", is_zero(eqB))
    report("S1 G = (n-1) C (1 - B), c = C B", is_zero(G - (n - 1) * C * (1 - B)) and is_zero(c - C * B))


# ------------------------------------------------------------------------------------------------------------------
def s2():
    x, t, xK, n, mu = sp.symbols("x t x_K n mu", positive=True)
    q = sp.Function("q")
    Q = sp.Integral(q(t) / t, (t, xK, x))
    chi = (x / xK) ** (-(n - 1)) * sp.exp(-SIGN * Q)
    hom = x * sp.diff(chi, x) + (n - 1 + q(x)) * chi
    report("S2 chi solves x B' + (n-1+q) B = 0", is_zero(sp.simplify(hom / chi)))
    report("S2 chi(x_K) = 1", sp.simplify(chi.subs(x, xK).doit()) == 1)
    Bs = sp.Function("B_s")(x)
    full = lambda B: x * sp.diff(B, x) - ((n - 1) * (1 - B) - q(x) * B)
    lin = sp.simplify((full(Bs + mu * chi) - full(Bs)) / chi)
    report("S2 B_s + mu chi solves B's equation when B_s does", is_zero(lin))
    gain = sp.simplify((chi.subs(x, xK) / chi.subs(x, 1)).doit())
    target = xK ** (-(n - 1)) * sp.exp(sp.Integral(q(t) / t, (t, xK, 1)))
    report("S2 gain chi(x_K)/chi(1) = x_K^-(n-1) exp(int_{x_K}^1 q/t dt)", is_zero(sp.simplify(gain / target.doit()) - 1))


# ------------------------------------------------------------------------------------------------------------------
def conv(a, b, k):
    return sum(a[i] * b[k - i] for i in range(k + 1))


def s3_symbolic(K=4):
    """hp's horizon recurrences in u (model.rs, unscaled): a_{k+1} = [k a_k - (a q2 + (n-1)(a - a^3))_k / 2]/(k+1),
    c_{k+1} = [k c_k - (n-1)(c (a^2 - 1))_k]/(k+1), with c_0 = 1, a_0 = a_h; against B's and C's recurrences
    b_{k+1} = [(n-1+k) b_k + (q b)_k - (n-1) delta_k0]/(k+1), C_{k+1} = [k C_k - (q C)_k]/(k+1)."""
    n, ah = sp.symbols("n a_h", positive=True)
    qs = sp.symbols(f"q0:{K + 1}")
    a, c = [ah], [sp.Integer(1)]
    for k in range(K):
        a2 = [conv(a, a, j) for j in range(k + 1)]
        a3k = conv(a, a2, k)
        a.append(sp.expand((k * a[k] - (conv(a, list(qs), k) + (n - 1) * (a[k] - a3k)) / 2) / (k + 1)))
        ca1 = conv(c, a2, k) - c[k]
        c.append(sp.expand((k * c[k] - (n - 1) * ca1) / (k + 1)))
    b, C = [1 / ah**2], [ah**2]
    for k in range(K):
        b.append(sp.expand(((n - 1 + k) * b[k] + SIGN * conv(list(qs), b, k) - (n - 1) * (1 if k == 0 else 0)) / (k + 1)))
        C.append(sp.expand((k * C[k] - conv(list(qs), C, k)) / (k + 1)))
    u = sp.symbols("u")
    A = sum(a[k] * u**k for k in range(K + 1))
    cser = sum(c[k] * u**k for k in range(K + 1))
    Bser = sp.series(1 / A**2, u, 0, K + 1).removeO()
    Cser = sp.series(cser * A**2, u, 0, K + 1).removeO()
    okB = all(is_zero(Bser.coeff(u, k) - b[k]) for k in range(K + 1))
    okC = all(is_zero(Cser.coeff(u, k) - C[k]) for k in range(K + 1))
    report(f"S3 hp's a-series squared and inverted equals B's series to order {K}", okB)
    report(f"S3 hp's c-series times a^2 equals C's series to order {K}", okC)


def bsl_system(n, q, K, dtype):
    """the minimal (slaved) particular solution: rows k = 0..K of (k+1) b_{k+1} - (n-1+k) b_k - (q b)_k =
    -(n-1) delta_k0 with b_{K+1} = 0, solved as one (K+1) x (K+1) system (diagonally dominant by rows)"""
    M = [[dtype(0)] * (K + 1) for _ in range(K + 1)]
    r = [dtype(0)] * (K + 1)
    for k in range(K + 1):
        M[k][k] -= (n - 1 + k)
        if k < K:
            M[k][k + 1] += k + 1
        for j in range(k + 1):
            M[k][k - j] -= q[j]
    r[0] = -(n - 1)
    return M, r


def chi_series(n, q, K, sign=1):
    """the mass mode's coefficients from chi_0 = 1: (k+1) chi_{k+1} = (n-1+k) chi_k + sign (q chi)_k"""
    ch = [q[0] * 0 + 1]
    for k in range(K):
        ch.append(((n - 1 + k) * ch[k] + sign * conv(q, ch, k)) / (k + 1))
    return ch


def s3_exact(n=37, K=10, u0=sp.Rational(3, 10)):
    import random
    rnd = random.Random(491)
    q = [sp.Rational(rnd.randint(1, 60), 1000 * (j + 1)) for j in range(K + 1)]
    M, r = bsl_system(sp.Integer(n), q, K, sp.Integer)
    bsl = list(sp.Matrix(M).LUsolve(sp.Matrix(r)))
    ch = chi_series(sp.Integer(n), q, K, SIGN)
    S = sum(ch[k] * u0**k for k in range(K + 1))
    chat = [v / S for v in ch]
    BK = sp.Rational(1001, 1000)                        # the factored unknown: B at the last node
    gam = BK - sum(bsl[k] * u0**k for k in range(K + 1))
    b = [bsl[k] + gam * chat[k] for k in range(K + 1)]
    # the unfactored forward recurrence from the factored b_0 (q fixed: the coupling to p, m is not part of S3)
    fwd = [b[0]]
    for k in range(K):
        fwd.append(((n - 1 + k) * fwd[k] + conv(q, fwd, k) - (n - 1) * (1 if k == 0 else 0)) / (k + 1))
    report(f"S3 the split B_sl + gamma chi_hat is the forward recurrence's series (n {n}, K {K}, exact)",
           all(fwd[k] == b[k] for k in range(K + 1)))
    report("S3 the split's sum at u0 is the factored unknown B(x_K)", sum(b[k] * u0**k for k in range(K + 1)) == BK)
    # the closed form: chi(u) / chi(0) = (1-u)^-(n-1) exp(int_0^u q(s)/(1-s) ds), q(u) = sum q_j u^j
    u, s = sp.symbols("u s")
    qu = sum(q[j] * u**j for j in range(K + 1))
    Qu = sp.integrate(sp.series(qu / (1 - u), u, 0, K + 1).removeO(), u)
    cf = sp.series((1 - u) ** (-(n - 1)) * sp.exp(Qu), u, 0, K + 1).removeO()
    report("S3 the mass mode's series equals (1-u)^-(n-1) exp(int_0^u q/(1-s) ds)",
           all(sp.simplify(cf.coeff(u, k) - ch[k]) == 0 for k in range(K + 1)))


def s3_rounding(n=240, K=130, u0="0.3", lo=30, hi=200):
    """the sum at u0 of B's series in lo digits, by the forward recurrence from b_0 and by the split, against hi
    digits; q = 2 eps (1-u)^2 F^2 (Proposition 1's q at x = 1 - u), F^2 = 3.7 (README's <F^2>, D -> infinity)"""
    def run(dps, split):
        with mp.workdps(dps):
            eps = mp.mpf(1) / n
            F2 = mp.mpf("3.7")
            q = [2 * eps * F2, -4 * eps * F2, 2 * eps * F2] + [mp.mpf(0)] * (K - 2)
            u = mp.mpf(u0)
            M, r = bsl_system(mp.mpf(n), q, K, mp.mpf)
            bsl = mp.lu_solve(mp.matrix(M), mp.matrix(r))
            ch = chi_series(mp.mpf(n), q, K)
            S = mp.fsum(ch[k] * u**k for k in range(K + 1))
            BK = mp.mpf("1.001")
            gam = BK - mp.fsum(bsl[k] * u**k for k in range(K + 1))
            b = [bsl[k] + gam * ch[k] / S for k in range(K + 1)]
            if not split:                                   # forward from the factored b_0, rounded to dps
                b0 = +b[0]
                b = [b0]
                for k in range(K):
                    b.append(((n - 1 + k) * b[k] + conv(q, b, k) - (n - 1) * (1 if k == 0 else 0)) / (k + 1))
            return mp.fsum(b[k] * u**k for k in range(K + 1)), BK
    with mp.workdps(hi):
        ref, BK = run(hi, True)
        e_split = abs(run(lo, True)[0] - BK)
        e_fwd = abs(run(lo, False)[0] - BK)
        G = (1 - mp.mpf(u0)) ** (-(n - 1))
    detail = f"(G = {mp.nstr(G, 3)}; error of the sum at u0 in {lo} digits: forward {mp.nstr(e_fwd, 3)}, split {mp.nstr(e_split, 3)})"
    report("S3 rounding: the split's sum is good to the working precision, the forward one is not",
           e_split < mp.mpf(10) ** (-(lo - 4)) and e_fwd > G * mp.mpf(10) ** (-lo - 2), detail)


# ------------------------------------------------------------------------------------------------------------------
def s4():
    """Proposition 3's method for the mass mode: h = sqrt(eps), n = 1/h^2; background p0 = h x F, m0 = -p0 (+O(h^3),
    which enters the transport at O(h^4) only), b0 = 1 + 2 h^4 x^2 F^2, c0 = 1 - h^2 F^2 (1 - x^2) (Proposition 1, as
    in largeD_transport_sympy.py). Perturbation E = x^(1-n): db = E (beta0 + h^2 beta1), dc = E (g0 + h^2 g1),
    dp = E h (P0 + h^2 P1), dm = E h (Q0 + h^2 Q1)."""
    x, th, h = sp.symbols("x theta h", positive=True)
    w = sp.symbols("omega")
    F = sp.Function("F")(th)
    n = 1 / h**2
    fn = lambda s: sp.Function(s)(x, th)
    b0_, b1_, g0_, g1_, P0, P1, Q0, Q1 = (fn(s) for s in ("beta0", "beta1", "g0", "g1", "P0", "P1", "Q0", "Q1"))
    p0 = h * x * F
    m0 = -p0
    bb = 1 + 2 * h**4 * x**2 * F**2
    cb = 1 - h**2 * F**2 * (1 - x**2)
    lnE = (1 - n) * sp.log(x)
    Ef = sp.exp(lnE)
    db = Ef * (b0_ + h**2 * b1_)
    dc = Ef * (g0_ + h**2 * g1_)
    dp = Ef * h * (P0 + h**2 * P1)
    dm = Ef * h * (Q0 + h**2 * Q1)
    Dx = lambda f: sp.diff(f, x)
    Dt = lambda f: sp.diff(f, th)
    E1 = lambda p, m, b, c: x * Dx(c) - (n - 1) * c * (b - 1)
    E2 = lambda p, m, b, c: x * Dx(b) - b * (p**2 + m**2) + (n - 1) * b * (b - 1)
    def E3(p, m, b, c):
        G = (n - 1) * c * (b - 1)
        return x * (x - c) * Dx(p) + w * x * Dt(p) - G * p + c * p - n / 2 * c * (p + m)
    def E4(p, m, b, c):
        G = (n - 1) * c * (b - 1)
        return x * (x + c) * Dx(m) + w * x * Dt(m) + G * m - c * m + n / 2 * c * (p + m)
    tt = sp.symbols("tt")
    def lin(fn_):
        full = fn_(p0 + tt * dp, m0 + tt * dm, bb + tt * db, cb + tt * dc)
        return sp.expand(sp.simplify(sp.diff(full, tt).subs(tt, 0) / Ef) * h**2)
    def orders(e, k):
        e = sp.expand(e)
        return sp.simplify(e.coeff(h, k))
    L1, L2, L3, L4 = (lin(f) for f in (E1, E2, E3, E4))
    # leading orders: E1 at h^0 gives g0 = -beta0; E3, E4 at h^1 give P0, Q0
    o1 = orders(L1, 0)
    sol_g = sp.solve(o1, g0_)
    o3, o4 = orders(L3, 1), orders(L4, 1)
    sol = sp.solve([o3, o4], [P0, Q0], dict=True)
    ok_lead = len(sol) == 1 and is_zero(sol[0][P0] - sol[0][Q0]) and sol_g and is_zero(sol_g[0] + b0_)
    report("S4 leading order: delta c = -delta b, delta p = delta m (= -h F x^(1-n) beta0)",
           ok_lead and is_zero(sol[0][P0] + F * b0_), f"P0 = {sol[0][P0] if sol else None}")
    # E2: h^2 order gives x beta0' = 0; h^4 order gives the transport
    e2_2 = orders(L2, 2)
    e2_4 = orders(L2, 4).subs({P0: sol[0][P0], Q0: sol[0][Q0]}).doit()
    target = x * Dx(b1_) + SIGN * 2 * x**2 * F**2 * b0_
    report("S4 O(eps^0): x d(beta0)/dx = 0", is_zero(e2_2 - x * Dx(b0_)))
    report("S4 O(eps): x d(beta1)/dx = -2 x^2 F^2 beta0, i.e. x beta' = -2 eps x^2 F^2 beta + O(eps^2)",
           is_zero(e2_4 - target), f"E2 at h^4: {sp.simplify(e2_4)}")
    # the odd orders of E2 vanish to h^3 (no half-integer powers of eps in the transport)
    report("S4 E2 has no h^1 or h^3 term", is_zero(orders(L2, 1)) and is_zero(orders(L2, 3)))


# ------------------------------------------------------------------------------------------------------------------
def s5():
    x, t, xK, eps, F2 = sp.symbols("x t x_K epsilon F2", positive=True)
    n = 1 / eps
    q = 2 * eps * t**2 * F2                       # Proposition 1's q = p0^2 + m0^2 at O(eps)
    lnE = sp.integrate(q / t, (t, xK, 1))
    G_before = xK ** (-(n - 1)) * sp.exp(SIGN * lnE)
    target = xK ** (-(n - 1)) * sp.exp(eps * F2 * (1 - xK**2))
    report("S5 gain before: x_K^-(n-1) exp(eps F^2 (1 - x_K^2)) at O(eps)", is_zero(sp.simplify(G_before / target) - 1))
    # S4's transport integrated over the segment gives the same factor: beta(1)/beta(x_K) = exp(-eps F^2 (1 - x_K^2))
    beta_ratio = sp.exp(-SIGN * sp.integrate(2 * eps * t * F2, (t, xK, 1)))
    report("S5 S4's transport gives S2's factor", is_zero(sp.simplify(beta_ratio * sp.exp(eps * F2 * (1 - xK**2))) - 1))


# ------------------------------------------------------------------------------------------------------------------
def s6():
    J = sp.Matrix(3, 3, sp.symbols("j0:9"))
    T = sp.Matrix([[sp.Symbol("t11"), sp.Symbol("t12"), sp.Symbol("t13")],
                   [sp.Symbol("t21"), sp.Symbol("t22"), sp.Symbol("t23")], [0, 0, 1]])   # Delta last, kept
    S = sp.Matrix(3, 3, sp.symbols("s0:9"))
    F = sp.Matrix(sp.symbols("f0:3"))
    dz = -J.inv() * F
    dy = -(S * J * T).inv() * (S * F)
    report("S6 Newton's step is affine invariant: T dy = dz", all(is_zero(v) for v in (T * dy - dz)))
    eD = sp.Matrix([[0, 0, 1]])
    report("S6 e_Delta^T (J T)^-1 = e_Delta^T J^-1 when T keeps Delta",
           all(is_zero(v) for v in (eD * (J * T).inv() - eD * J.inv())))
    # a nonlinear change z = y + k y^2 on F(z) = z^2 - 2: the first step agrees to first order, not to second
    z0, k, y = sp.symbols("z0 k y")
    Fz = lambda z: z**2 - 2
    z1 = z0 - Fz(z0) / (2 * z0)
    yy0 = (-1 + sp.sqrt(1 + 4 * k * z0)) / (2 * k)        # the root that tends to z0 as k -> 0
    phi = lambda yv: yv + k * yv**2
    Fy = lambda yv: Fz(phi(yv))
    y1 = yy0 - Fy(yy0) / sp.diff(Fy(y), y).subs(y, yy0)
    diff = sp.series(sp.simplify(phi(y1) - z1).subs(z0, sp.sqrt(2) + sp.Symbol("d")), sp.Symbol("d"), 0, 3).removeO()
    d = sp.Symbol("d")
    report("S6 a nonlinear change: the first iterate differs at second order in the start's error",
           is_zero(diff.coeff(d, 0)) and is_zero(diff.coeff(d, 1)) and not is_zero(diff.coeff(d, 2)),
           f"phi(y1) - z1 = {sp.simplify(diff.coeff(d, 2))} d^2 + O(d^3)")


# ------------------------------------------------------------------------------------------------------------------
def s7():
    A, C, f, Cp = sp.symbols("A C f Cprime")
    yr = sp.Matrix(sp.symbols("r1 r2"))
    yf = sp.Matrix(sp.symbols("g1 g2"))
    yh = (yr - C * yf) / A
    yhK = yh + (1 - f) * (C / A) * yf                  # the truncation keeps the share f of y_h's fast part
    det = lambda u, v: u[0] * v[1] - u[1] * v[0]
    report("S7 the truncated matching determinant is -(f C/A) det(y_reg, y_f)",
           is_zero(det(yr, yhK) + f * C / A * det(yr, yf)))
    report("S7 the continuum's (f = 1) is -(C/A) det(y_reg, y_f)", is_zero(det(yr, yh) + C / A * det(yr, yf)))


# ------------------------------------------------------------------------------------------------------------------
# M: the linear model (Proposition 2's equation) in u = 1 - x
def lin_series(n, s, J):
    """y = p - m: x^2 (1-x^2) y'' + x (n-2-2(s+1) x^2) y' - (n-2+s(s+1) x^2) y = 0 (README, Proposition 2), the
    horizon-regular solution's coefficients in u = 1 - x with y(1) = 1"""
    def pm(a, b):
        out = [mp.mpc(0)] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
        return out
    X = [mp.mpf(1), mp.mpf(-1)]
    X2 = pm(X, X)
    P2 = pm(X2, [1 - X2[0], -X2[1], -X2[2]])
    P1 = pm(X, [-(n - 2) + 2 * (s + 1) * X2[0], 2 * (s + 1) * X2[1], 2 * (s + 1) * X2[2]])
    P0 = [-(n - 2) - s * (s + 1) * X2[0], -s * (s + 1) * X2[1], -s * (s + 1) * X2[2]]
    tau = [mp.mpc(1)]
    for j in range(J):
        lead = P2[1] * (j + 1) * j + P1[0] * (j + 1)
        rest = mp.mpc(0)
        for i in range(2, len(P2)):
            kk = j - i + 2
            if 0 <= kk < len(tau):
                rest += P2[i] * kk * (kk - 1) * tau[kk]
        for i in range(1, len(P1)):
            kk = j - i + 1
            if 0 <= kk < len(tau):
                rest += P1[i] * kk * tau[kk]
        for i in range(len(P0)):
            kk = j - i
            if 0 <= kk < len(tau):
                rest += P0[i] * tau[kk]
        tau.append(-rest / lead)
    return tau


def lin_model(n, s, u0, Ks, J=700):
    tau = lin_series(n, s, J)
    u = mp.mpf(u0)
    xK = 1 - u
    kap = (n + 1) / 2
    al, be = (1 + s) / 2, (2 + s) / 2
    yr = xK * mp.hyp2f1(al, be, kap, xK**2)
    yr1 = mp.hyp2f1(al, be, kap, xK**2) + 2 * xK**2 * al * be / kap * mp.hyp2f1(al + 1, be + 1, kap + 1, xK**2)
    def W(K):
        v = mp.fsum(tau[j] * u**j for j in range(K + 1))
        d = -mp.fsum(j * tau[j] * u ** (j - 1) for j in range(1, K + 1))
        return yr * d - yr1 * v, v
    Winf, vinf = W(J)
    yh_exact = xK * mp.hyp2f1(al, be, al + be - kap + 1, 1 - xK**2)
    t = [abs(c) * u**j for j, c in enumerate(tau)]
    return dict(tau=tau, t=t, Winf=Winf, W={K: W(K)[0] for K in Ks}, yh_rel=abs(vinf - yh_exact) / abs(yh_exact))


def load_states():
    """hp/data/horizon_radius.csv (#693): D, Re s and the transient's peak order of each state, field p"""
    rows = {}
    with open(os.path.join(ROOT, "hp", "data", "horizon_radius.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["harmonic"] == "transient" and r["field"] == "p":
                rows[float(r["D"])] = (float(r["Re_s"]), int(r["k_peak"]))
    return sorted((D, *v) for D, v in rows.items())


def load_delta():
    out = []
    with open(os.path.join(ROOT, "hp", "data", "delta_all.csv")) as fh:
        for r in csv.DictReader(fh):
            out.append((float(r["D"]), float(r["Delta_q0"])))
    return sorted(out)


def interp(tab, D):
    """linear interpolation; past either end, the end's value (delta_all.csv ends at D 245.64, where Delta changes by
    about 2e-4 per unit of D; the model depends on Delta only through the imaginary part of its near-resonance)"""
    if D <= tab[0][0]:
        return tab[0][1]
    if D >= tab[-1][0]:
        return tab[-1][1]
    for (d0, v0), (d1, v1) in zip(tab, tab[1:]):
        if d0 <= D <= d1 and d1 > d0:
            return v0 + (v1 - v0) * (D - d0) / (d1 - d0)
    raise ValueError(D)


def model():
    mp.mp.dps = 300
    states = load_states()
    deltas = load_delta()
    Kt = 600 if MUT == "ks" else 130                 # the chain's ks; the control: no truncation
    res_tab = [(D, rs) for D, rs, _ in states]
    def ns(D, mu):
        rs = interp(res_tab, D)
        om = 2 * mp.pi / mp.mpf(interp(deltas, D))
        return 2 * (mp.mpf(rs) + 1), 1j * mu * om
    # M1, M2
    worst_y, worst_pk = mp.mpf(0), 0.0
    print("M2 D, Re s, measured k_peak (horizon_radius.csv), model peak order (mu 1, 3), ln f at K 130, 160, 200, 250")
    lnf = {}
    for D, rs, kpk in states:
        row = []
        for mu in (1, 3):
            n, s = ns(D, mu)
            r = lin_model(n, s, 0.3, [130, 160, 200, 250])
            worst_y = max(worst_y, r["yh_rel"])
            lt = [mp.log10(v) for v in r["t"]]
            jpk = max(range(int(rs), len(lt)), key=lambda j: lt[j])
            worst_pk = max(worst_pk, abs(jpk - kpk) / kpk)
            f = [float(mp.log(abs(r["W"][K] / r["Winf"]))) for K in (130, 160, 200, 250)]
            lnf[(D, mu)] = f[0]
            row.append(f"mu {mu}: peak {jpk}, ln f {f[0]:.2f} {f[1]:.2f} {f[2]:.2f} {f[3]:.2f}")
        print(f"  D {D:.2f} Re s {rs:.2f} k_peak {kpk}: " + "; ".join(row))
        sys.stdout.flush()
    report("M1 the model's horizon series equals 2F1(al, be; al+be-kap+1; 1-x^2) at x_K", worst_y < mp.mpf(10) ** -100,
           f"(worst relative difference {mp.nstr(worst_y, 3)})")
    report("M2 the transient's peak order: model against horizon_radius.csv within 3%", worst_pk < 0.03,
           f"(worst {100 * worst_pk:.1f}%)")
    # M3: the smallest K with |ln f| < 0.01, at the first and last state and two extrapolations of Re s (linear in D)
    (Da, ra, _), (Db, rb, _) = states[0], states[-1]
    print("M3 smallest K with |ln f| < 0.01 (u0 = 0.3); Re s past D", f"{Db:.1f} extrapolated linearly from the csv")
    for D in (Da, Db, 300.0, 400.0):
        rs = ra + (rb - ra) * (D - Da) / (Db - Da)
        n = 2 * (mp.mpf(rs) + 1)
        om = 2 * mp.pi / mp.mpf(interp(deltas, min(D, deltas[-1][0])))
        J = int(4 * rs) + 300
        out = []
        for mu in (1, 3):
            r = lin_model(n, 1j * mu * om, 0.3, [], J=J)
            tau = r["tau"]
            u = mp.mpf("0.3")
            xK = 1 - u
            kap = (n + 1) / 2
            al, be = (1 + 1j * mu * om) / 2, (2 + 1j * mu * om) / 2
            yr = xK * mp.hyp2f1(al, be, kap, xK**2)
            yr1 = mp.hyp2f1(al, be, kap, xK**2) + 2 * xK**2 * al * be / kap * mp.hyp2f1(al + 1, be + 1, kap + 1, xK**2)
            v = d = mp.mpc(0)
            Kmin = None
            for j in range(J + 1):
                v += tau[j] * u**j
                if j:
                    d -= j * tau[j] * u ** (j - 1)
                if abs(mp.log(abs((yr * d - yr1 * v) / r["Winf"]))) < 0.01 and Kmin is None:
                    Kmin = j
            out.append(f"mu {mu}: K {Kmin} ({Kmin / rs:.2f} Re s)")
        print(f"  D {D:.1f} (Re s {rs:.1f}): " + "; ".join(out))
        sys.stdout.flush()
    # M4: rates against I488-mac's sigma (raw 994f6f21) and the last-node shift against #537
    sig = {237.13125: (3.0107328097098527e-25, 1.4996164881594329e-23),
           242.36419067382812: (2.5513801984886627e-27, 4.3919115051926723e-26)}
    (Dl, (s1l, s3l)), (Dh, (s1h, s3h)) = sorted(sig.items())
    meas = {3: mp.log(s1h / s1l), 1: mp.log(s3h / s3l)}   # sigma_1 is the harmonic-3 pair, sigma_3 mostly Delta
    lw = {}
    for D in (Dl, Dh):
        for mu in (1, 3):
            n, s = ns(D, mu)
            r = lin_model(n, s, 0.3, [Kt])
            lw[(D, mu)] = (mp.log(abs(r["W"][Kt])), mp.log(abs(r["Winf"])))
    ok = True
    for mu, name in ((1, "sigma_3"), (3, "sigma_1")):
        dm = lw[(Dh, mu)][0] - lw[(Dl, mu)][0]
        di = lw[(Dh, mu)][1] - lw[(Dl, mu)][1]
        rel = abs(dm - meas[mu]) / abs(meas[mu])
        ok &= rel < 0.05
        print(f"  M4 D {Dl:.2f} -> {Dh:.2f}: ln {name} changes by {float(meas[mu]):.3f} (I488), the model's ln|W_{Kt}| (mu {mu}) "
              f"by {float(dm):.3f} ({100 * float(rel):.1f}%), its continuum ln|W_inf| by {float(di):.3f}")
    report("M4 the model's rates in D against I488's weak sigma within 5%", ok)
    Phi = lambda x: mp.log(x) - mp.log(1 - x**2) / 2
    shifts = []
    for mu in (1, 3):
        n, s = ns(Dh, mu)
        a = lin_model(n, s, 0.3, [Kt])
        b = lin_model(n, s, 0.28, [Kt])
        rowf = mp.log((1 - mp.mpf("0.72") ** 2) / (1 - mp.mpf("0.7") ** 2))   # (p, m) rows against (y, y'): (1 - x^2)
        shifts.append(float(mp.log(abs(b["W"][Kt])) - mp.log(abs(a["W"][Kt])) + rowf))
        cont = float(mp.log(abs(b["Winf"])) - mp.log(abs(a["Winf"])) + rowf)
    law = float(-(Dh - 2) * (Phi(mp.mpf("0.72")) - Phi(mp.mpf("0.7"))))
    print(f"  M4 last node 0.70 -> 0.72 at D {Dh:.2f}: model {shifts[0]:.2f} (mu 1), {shifts[1]:.2f} (mu 3); continuum "
          f"{cont:.2f}; the fast-mode law -n[Phi(0.72) - Phi(0.70)] = {law:.2f}; #537's raw shift of ln sigma_3: -8.96")
    report("M4 the last-node shift against #537's -8.96 within 5%", all(abs(v + 8.96) / 8.96 < 0.05 for v in shifts))


def main():
    if MODE in ("all", "check"):
        for f in (s1, s2, s3_symbolic, s3_exact, s3_rounding, s4, s5, s6, s7):
            f()
    if MODE in ("all", "model"):
        model()
    bad = [n for n, ok in results if not ok]
    print(f"{len(results) - len(bad)} of {len(results)} checks pass" + (f"; failed: {', '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
