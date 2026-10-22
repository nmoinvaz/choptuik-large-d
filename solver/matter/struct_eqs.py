"""(a) The D-dimensional Einstein-matter equations in polar-areal gauge, for every model of matter/STRUCTURE.md, derived
and checked with sympy.

    nice -n 10 python3 struct_eqs.py            (a few minutes)

1. Einstein tensor of ds^2 = -alpha^2 dt^2 + a^2 dr^2 + r^2 dOmega_n^2: the general-n formulas of struct_common.py
   against brute-force curvature for n = 2, 3, 4 (D = 4, 5, 6).
2. Matter, by brute force from explicit non-spherically-symmetric fields (n = 2 and 3): the stress tensor is
   SO(n+1)-symmetric (T^i_j proportional to delta^i_j on the sphere, T_ti = T_ri = 0) and has the (K, V) form, and the
   covariant field equation is equivalent to the reduced Euler-Lagrange equation:
     - scalar l-multiplet  phi_I = phi(t,r) Y_I, summed over a tight frame of degree-l harmonics:  K = 1, V = L phi^2/(2 r^2),
       L = l (l + n - 1)  (l = 1, 2; after normalizing sum_I Y_I^2 = 1);
     - wave map into S^{n+1}, U = (sin u * nhat, cos u):  K = 1, V = n sin^2 u / (2 r^2)  (times f^2);
     - SO(n+1) Yang-Mills, A = (1 - W)(nhat dnhat^T - dnhat nhat^T):  K = n / r^2, V = n (n-1)(1-W^2)^2 / (4 r^4)  (times 1/e^2).
   The flat YM equation is W_tt = W_rr + (n-2) W_r / r + (n-1) W (1-W^2) / r^2, i.e. (d-3) and (d-2) with d = n+1 space
   dimensions (Bizon-Tabor); a coefficient (d-4) would give -W_r/r in D = 4, against Gundlach's eq. 10.
3. D = 4 forms: the multiplet against Olabarrieta-Ventrella-Choptuik-Unruh (PRD 76, 124014), the YM model against
   Gundlach (PRD 55, 6002, eqs. 10-12), the wave map against the standard hedgehog energy.
4. Self-similar first-order systems (tau = -ln(-t), x = r/(-t); p, m = r(Phi +- Pi), A = a, c = alpha/a, f = the field)
   for the multiplet and the wave map, checked against the covariant equations; for l = 0 they are exactly the
   equations of solver/dss.py, with f decoupled.
5. Yang-Mills has a length scale: kappa / e^2 has dimension length^2 in every D, and in self-similar variables the YM
   source carries an explicit factor e^{2 tau}. The scale-invariant system on small scales is W = 1 - (r / l_YM) U,
   l_YM^2 = kappa / e^2 (Gundlach's W = 1 - r S in D = 4); U obeys the l = 1 multiplet equation up to the metric terms
   -(n-2)(b-1) c U/r^2 + c_r U/r, with a Yang-Mills stress tensor (same rho + P_r structure, different rho - P_r).
"""
import sympy as sp
from sympy.calculus.euler import euler_equations
from struct_common import (t, r, polar_areal, einstein_diag, einstein_formulas, sqrt_det, box, reduced_el, report)

ALL = []


def check(label, expr, simp=sp.simplify):
    ALL.append(report(label, expr, simp))


# ---------------------------------------------------------------- 1. Einstein tensor
print("1. Einstein tensor, general-n formulas against brute force")
for n in (2, 3, 4):
    X, g, al, a, th, nv = polar_areal(n)
    G = einstein_diag(X, g)
    Gtt, Grr, Gtr = einstein_formulas(n, al, a)
    check(f"n = {n}: G^t_t", G[0, 0] / g[0] - Gtt)
    check(f"n = {n}: G^r_r", G[1, 1] / g[1] - Grr)
    check(f"n = {n}: G_tr", G[0, 1] - Gtr)
    check(f"n = {n}: G^th_th isotropic", G[2, 2] / g[2] - G[n + 1, n + 1] / g[n + 1])


# ---------------------------------------------------------------- 2. matter by brute force
def stress_scalar_fields(fields, X, g):
    """sum_I (d_mu f_I d_nu f_I - 1/2 g_mu_nu (d f_I)^2), lower indices."""
    D = len(X)
    T = sp.zeros(D, D)
    for f in fields:
        df = [sp.diff(f, xx) for xx in X]
        sq = sum(df[i] ** 2 / g[i] for i in range(D))
        for i in range(D):
            for j in range(D):
                T[i, j] += df[i] * df[j] - (g[i] * sq / 2 if i == j else 0)
    return T


def check_KV(label, T, g, n, K, V, psi, Pf, al, a):
    """T^t_t = -rho, T^r_r = P_r, T_tr, T_t,th = T_r,th = 0 and isotropy on the sphere, for given K, V."""
    b = a ** 2
    Phi, Pi = sp.diff(Pf, r), a * sp.diff(Pf, t) / al
    Kf, Vf = K.subs(psi, Pf), V.subs(psi, Pf)
    rho = Kf * (Phi ** 2 + Pi ** 2) / (2 * b) + Vf
    Pr = Kf * (Phi ** 2 + Pi ** 2) / (2 * b) - Vf
    check(f"{label}: T^t_t = -rho", T[0, 0] / g[0] + rho)
    check(f"{label}: T^r_r = P_r", T[1, 1] / g[1] - Pr)
    check(f"{label}: T_tr = K psi_t psi_r", T[0, 1] - Kf * sp.diff(Pf, t) * sp.diff(Pf, r))
    off = [T[0, k] for k in range(2, n + 2)] + [T[1, k] for k in range(2, n + 2)]
    off += [T[i, j] for i in range(2, n + 2) for j in range(2, n + 2) if i != j]
    check(f"{label}: T_t,th = T_r,th = T_th,th' = 0", sum(sp.Abs(sp.simplify(o)) for o in off))
    iso = [T[k, k] / g[k] - T[2, 2] / g[2] for k in range(3, n + 2)]
    check(f"{label}: T^th_th isotropic", sum(sp.Abs(sp.simplify(o)) for o in iso))


psi = sp.Symbol("psi")
print("\n2. Matter by brute force (non-symmetric fields summed or composed, full D-dimensional metric)")
for n in (2, 3):
    X, g, al, a, th, nv = polar_areal(n)
    sq = sqrt_det(n, al, a, th)
    w = sq / (al * a * r ** n)
    D = len(X)
    # ---- l-multiplet, l = 1, 2: a tight frame of harmonics with sum Y^2 = 1
    for l in (1, 2):
        if l == 1:
            Ys = list(nv)
        else:
            Ys = [(nv[i] * nv[j] - (1 if i == j else 0) * sp.Rational(1, n + 1)) / sp.sqrt(1 - sp.Rational(1, n + 1))
                  for i in range(n + 1) for j in range(n + 1)]
        L = l * (l + n - 1)
        check(f"n={n} l={l}: sum Y^2 = 1", sum(Y ** 2 for Y in Ys) - 1)
        Vm = L * psi ** 2 / (2 * r ** 2)
        EL, Pf = reduced_el(sp.Integer(1), Vm, psi, al, a, n)
        T = stress_scalar_fields([Pf * Y for Y in Ys], X, g)
        check_KV(f"n={n} l={l} multiplet", T, g, n, sp.Integer(1), Vm, psi, Pf, al, a)
        Y = Ys[0]
        check(f"n={n} l={l} multiplet: sqrt(-g) box(phi Y) + w Y EL = 0 (box phi_I = 0 <=> reduced EL)",
              sq * box(Pf * Y, X, g, sq) + w * Y * EL)
    # ---- wave map into S^{n+1}
    Vw = n * sp.sin(psi) ** 2 / (2 * r ** 2)
    EL, Pf = reduced_el(sp.Integer(1), Vw, psi, al, a, n)
    Usym = [sp.sin(psi) * nv[i] for i in range(n + 1)] + [sp.cos(psi)]
    U = [e.subs(psi, Pf) for e in Usym]
    T = stress_scalar_fields(U, X, g)
    check_KV(f"n={n} wave map", T, g, n, sp.Integer(1), Vw, psi, Pf, al, a)
    dU2 = sum(sum(sp.diff(Ua, X[i]) ** 2 / g[i] for i in range(D)) for Ua in U)
    E = [box(Ua, X, g, sq) + dU2 * Ua for Ua in U]              # harmonic-map equation into the unit sphere
    dUu = [sp.diff(e, psi).subs(psi, Pf) for e in Usym]
    res = sum(sp.Abs(sp.simplify(sq * E[k] + w * EL * dUu[k])) for k in range(len(U)))
    check(f"n={n} wave map: sqrt(-g) (box U + |dU|^2 U) = -w EL d_u U  (all components)", res)
    # ---- SO(n+1) Yang-Mills, magnetic ansatz
    KY, VY = n / r ** 2, n * (n - 1) * (1 - psi ** 2) ** 2 / (4 * r ** 4)
    EL, Pf = reduced_el(KY, VY, psi, al, a, n)
    N1 = n + 1
    A = []
    for xx in X:
        dn = sp.diff(nv, xx)
        A.append((1 - Pf) * (nv * dn.T - dn * nv.T))
    F = [[(sp.diff(A[j], X[i]) - sp.diff(A[i], X[j]) + A[i] * A[j] - A[j] * A[i]).applyfunc(sp.simplify)
          for j in range(D)] for i in range(D)]
    # T_mu nu = sum_{a<b} [F_mu alpha F_nu^alpha - 1/4 g_mu nu F^2] = -1/2 tr(F_mu a F_nu^a) + 1/8 g tr(F F)
    FF = sum(-sp.Rational(1, 2) * (F[i][j] * F[i][j]).trace() / (g[i] * g[j]) for i in range(D) for j in range(D))
    TY = sp.zeros(D, D)
    for i in range(D):
        for j in range(D):
            TY[i, j] = sum(-sp.Rational(1, 2) * (F[i][k] * F[j][k]).trace() / g[k] for k in range(D)) \
                - (g[i] * FF / 4 if i == j else 0)
    check_KV(f"n={n} Yang-Mills", TY, g, n, KY, VY, psi, Pf, al, a)
    # D_mu F^{mu nu} = (1/sqrt g) d_mu (sqrt g F^{mu nu}) + [A_mu, F^{mu nu}]
    nus = list(range(D)) if n == 2 else [0, 2]       # n = 3: one t- and one angular component (cost)
    J = {}
    for nu in nus:
        Jn = sp.zeros(N1, N1)
        for mu in range(D):
            Fup = F[mu][nu] / (g[mu] * g[nu])
            Jn += sp.diff(sq * Fup, X[mu]) / sq + A[mu] * Fup - Fup * A[mu]
        J[nu] = Jn.applyfunc(sp.simplify)
    check(f"n={n} Yang-Mills: D_mu F^(mu nu) = 0 identically for nu = " + ("t, r" if n == 2 else "t"),
          sum(sp.Abs(e) for nu in nus if nu < 2 for e in list(J[nu])))
    ok = True
    for nu in [v for v in nus if v >= 2]:
        M = sp.diff(nv, X[nu]) * nv.T - nv * sp.diff(nv, X[nu]).T
        # g_thth / r^2 = unit-sphere metric; the factor 1/n is the sum over the n sphere directions of |n ^ d_th n|^2 / h
        R = (J[nu] * sq * g[nu] / r ** 2 + w * EL * M / n).applyfunc(sp.simplify)
        ok = ok and all(e == 0 for e in R)
    ALL.append(ok)
    print(f"  [{'ok' if ok else 'FAIL'}] n={n} Yang-Mills: sqrt(-g) h_thth D_mu F^(mu th) = -(1/n) w EL (d_th n n^T - n d_th n^T),"
          " all angular components")

print("\n   flat-space radial equations from the reduced Lagrangians (alpha = a = 1, general n):")
nn = sp.Symbol("n", positive=True)
Wf = sp.Function("W")(t, r)
flat_YM = sp.diff(Wf, t, 2) - (sp.diff(Wf, r, 2) + (nn - 2) * sp.diff(Wf, r) / r + (nn - 1) * Wf * (1 - Wf ** 2) / r ** 2)
Lred = r ** nn * ((nn / r ** 2) * (sp.diff(Wf, t) ** 2 - sp.diff(Wf, r) ** 2) / 2 - nn * (nn - 1) * (1 - Wf ** 2) ** 2 / (4 * r ** 4))
eY = euler_equations(Lred, Wf, [t, r])[0].lhs
check("Yang-Mills: W_tt = W_rr + (n-2) W_r/r + (n-1) W (1-W^2)/r^2   [(d-3), (d-2) with d = n+1]",
      sp.simplify(sp.powsimp(sp.expand(eY / (-nn * r ** (nn - 2)) - flat_YM), force=True)))
uf = sp.Function("u")(t, r)
Lred = r ** nn * ((sp.diff(uf, t) ** 2 - sp.diff(uf, r) ** 2) / 2 - nn * sp.sin(uf) ** 2 / (2 * r ** 2))
eU = euler_equations(Lred, uf, [t, r])[0].lhs
flat_WM = sp.diff(uf, t, 2) - (sp.diff(uf, r, 2) + nn * sp.diff(uf, r) / r - nn * sp.sin(2 * uf) / (2 * r ** 2))
check("wave map: u_tt = u_rr + n u_r/r - n sin(2u)/(2 r^2)   [(d-1) with d = n+1]",
      sp.simplify(sp.powsimp(sp.expand(eU / (-r ** nn) - flat_WM), force=True)))

# ---------------------------------------------------------------- 3. D = 4 literature forms
print("\n3. D = 4 forms (general-n formulas at n = 2)")
al = sp.Function("alpha")(t, r)
a = sp.Function("a")(t, r)
b = a ** 2
kap = sp.Symbol("kappa", positive=True)


def polar_rhs(K, V, P, n, kap):
    Phi, Pi = sp.diff(P, r), a * sp.diff(P, t) / al
    rho = K * (Phi ** 2 + Pi ** 2) / (2 * b) + V
    Pr = K * (Phi ** 2 + Pi ** 2) / (2 * b) - V
    ar = (kap / n) * r * b * rho + (n - 1) * (1 - b) / (2 * r)
    alr = (kap / n) * r * b * Pr + (n - 1) * (b - 1) / (2 * r)
    return ar, alr, Phi, Pi


l = sp.Symbol("l", positive=True)
Pf = sp.Function("psi")(t, r)
Nl = (2 * l + 1) / (4 * sp.pi)                 # sum_m |Y_lm|^2 with the standard normalization
ar, alr, Phi, Pi = polar_rhs(Nl, Nl * l * (l + 1) * Pf ** 2 / (2 * r ** 2), Pf, 2, 8 * sp.pi)
ovcu_a = (2 * l + 1) / 2 * r * (Pi ** 2 + Phi ** 2 + l * (l + 1) * b * Pf ** 2 / r ** 2) - (b - 1) / (2 * r)
ovcu_al = (2 * l + 1) / 2 * r * (Pi ** 2 + Phi ** 2 - l * (l + 1) * b * Pf ** 2 / r ** 2) + (b - 1) / (2 * r)
check("multiplet vs Olabarrieta et al. 2007, a_r/a (G = 1, kappa = 8 pi)", ar - ovcu_a)
check("multiplet vs Olabarrieta et al. 2007, alpha_r/alpha", alr - ovcu_al)
EL, P2 = reduced_el(Nl, Nl * l * (l + 1) * psi ** 2 / (2 * r ** 2), psi, al, a, 2)
Pi2 = a * sp.diff(P2, t) / al
ovcu_w = sp.diff(Pi2, t) - (sp.diff(r ** 2 * al / a * sp.diff(P2, r), r) / r ** 2 - l * (l + 1) * a * al * P2 / r ** 2)
check("multiplet vs Olabarrieta et al. 2007, Pi_t equation", EL / (Nl * r ** 2) - ovcu_w)
e2 = sp.Symbol("e2", positive=True)            # e^2
Wg = sp.Function("W")(t, r)
ar, alr, Phi, Pi = polar_rhs(2 / (e2 * r ** 2), (1 - Wg ** 2) ** 2 / (2 * e2 * r ** 4), Wg, 2, kap)
g11 = sp.Rational(1, 2) * (1 - b) + Phi ** 2 + Pi ** 2 + b * (1 - Wg ** 2) ** 2 / (2 * r ** 2)
g12 = sp.Rational(1, 2) * (b - 1) + Phi ** 2 + Pi ** 2 - b * (1 - Wg ** 2) ** 2 / (2 * r ** 2)
check("Yang-Mills vs Gundlach 1997 eq. 11, r a_r/a (units kappa/e^2 = 2)", (r * ar - g11).subs(kap, 2 * e2))
check("Yang-Mills vs Gundlach 1997 eq. 12, r alpha_r/alpha", (r * alr - g12).subs(kap, 2 * e2))
EL, W2 = reduced_el(2 / (e2 * r ** 2), (1 - psi ** 2) ** 2 / (2 * e2 * r ** 4), psi, al, a, 2)
Pi2, Phi2 = a * sp.diff(W2, t) / al, sp.diff(W2, r)
g10 = r * sp.diff(Pi2, t) - (r * sp.diff(al / a * Phi2, r) + a * al * W2 * (1 - W2 ** 2) / r)
check("Yang-Mills vs Gundlach 1997 eq. 10, r Pi_t = r (c Phi)_r + a alpha W (1-W^2)/r", EL * e2 / 2 - g10 / r)
print("   (wave map, D = 4: V = n sin^2 u/(2 r^2) = sin^2 u / r^2, the standard hedgehog energy; checked in 2.)")

# ---------------------------------------------------------------- 4. self-similar first-order systems
print("\n4. Self-similar first-order systems (kappa = 2n, the solver's normalization)")
tau, x = sp.symbols("tau x", positive=True)
n = sp.Symbol("n", positive=True)
names = ["p", "m", "f", "A", "c"]
J0 = {k: sp.Symbol(k) for k in names}
Jx = {k: sp.Symbol(k + "_x") for k in names}
Jt = {k: sp.Symbol(k + "_tau") for k in names}


def Dx(e):
    return sp.diff(e, x) + sum(sp.diff(e, J0[k]) * Jx[k] for k in names)


def Dtau(e):
    return sp.diff(e, tau) + sum(sp.diff(e, J0[k]) * Jt[k] for k in names)


def d_r(e):
    return sp.exp(tau) * Dx(e)


def d_t(e):
    return sp.exp(tau) * (Dtau(e) + x * Dx(e))


p, m, f, A, c = (J0[k] for k in names)
rr = x * sp.exp(-tau)
bb = A ** 2
Phi = (p + m) / (2 * rr)
Pi = (p - m) / (2 * rr)
alpha = c * A
rv = sp.Symbol("rv", positive=True)


def ss_check(label, K, Vexpr, claimed):
    """Covariant equations (kappa = 2n, K constant, V(r, f)) with the claimed derivatives substituted."""
    V = Vexpr.subs(rv, rr)
    Vf = sp.diff(Vexpr, f).subs(rv, rr)
    rho = K * (Phi ** 2 + Pi ** 2) / (2 * bb) + V
    Pr = K * (Phi ** 2 + Pi ** 2) / (2 * bb) - V
    eqs = {
        "Hamiltonian": d_r(A) / A - (2 * rr * bb * rho + (n - 1) * (1 - bb) / (2 * rr)),
        "slicing": d_r(alpha) / alpha - (2 * rr * bb * Pr + (n - 1) * (bb - 1) / (2 * rr)),
        "Phi_t = (c Pi)_r": d_t(Phi) - d_r(c * Pi),
        "Pi_t equation": d_t(K * Pi) - (d_r(rr ** n * c * K * Phi) / rr ** n - c * bb * Vf),
        "f_t = c Pi": d_t(f) - c * Pi,
        "f_r = Phi": d_r(f) - Phi,
    }
    subs = {Jx["A"]: claimed["A_x"], Jx["c"]: claimed["c_x"], Jx["f"]: claimed["f_x"], Jt["f"]: claimed["f_tau"]}
    subs[Jt["p"]] = claimed["p_tau"] + (c - x) * Jx["p"]
    subs[Jt["m"]] = claimed["m_tau"] - (x + c) * Jx["m"]
    for k_, e_ in eqs.items():
        check(f"{label}: {k_}", sp.simplify(sp.expand(e_.subs(subs))))


# l-multiplet (K = 1, V = L f^2 / (2 r^2)); f = phi
L = sp.Symbol("L", nonnegative=True)
cx_mult = c * ((n - 1) * (A ** 2 - 1) - 2 * L * A ** 2 * f ** 2) / x
claimed = {
    "A_x": A * ((p ** 2 + m ** 2) + 2 * L * A ** 2 * f ** 2 + (n - 1) * (1 - A ** 2)) / (2 * x),
    "c_x": cx_mult,
    "p_tau": cx_mult * p - c * p / x + n * c * (p + m) / (2 * x) - c * A ** 2 * L * f / x,   # = p_tau + (x - c) p_x
    "m_tau": -cx_mult * m + c * m / x - n * c * (p + m) / (2 * x) + c * A ** 2 * L * f / x,  # = m_tau + (x + c) m_x
    "f_x": (p + m) / (2 * x),
    "f_tau": c * (p - m) / (2 * x) - (p + m) / 2,
}
ss_check("multiplet", sp.Integer(1), L * f ** 2 / (2 * rv ** 2), claimed)
solver = {
    "A_x": A * ((p ** 2 + m ** 2) + (n - 1) * (1 - A ** 2)) / (2 * x),
    "c_x": c * (n - 1) * (A ** 2 - 1) / x,
}
solver["p_tau"] = solver["c_x"] * p - c * p / x + n * c * (p + m) / (2 * x)
solver["m_tau"] = -solver["c_x"] * m + c * m / x - n * c * (p + m) / (2 * x)
for k_ in solver:
    check(f"multiplet at L = 0 is solver/dss.py: {k_}", (claimed[k_] - solver[k_]).subs(L, 0))
check("multiplet at L = 0: f absent from the p, m, a, c equations",
      sum(sp.diff(claimed[k_].subs(L, 0), f) for k_ in solver))
# wave map with coupling eta = kappa f_pi^2 / 2 (Husa et al.'s eta = 4 pi G f_pi^2 in D = 4): K = eta/n, V = eta sin^2 u/(2 r^2)
eta = sp.Symbol("eta", positive=True)
cx_wm = c * ((n - 1) * (A ** 2 - 1) - 2 * eta * A ** 2 * sp.sin(f) ** 2) / x
claimed = {
    "A_x": A * (eta * (p ** 2 + m ** 2) / n + 2 * eta * A ** 2 * sp.sin(f) ** 2 + (n - 1) * (1 - A ** 2)) / (2 * x),
    "c_x": cx_wm,
    "p_tau": cx_wm * p - c * p / x + n * c * (p + m) / (2 * x) - c * A ** 2 * n * sp.sin(2 * f) / (2 * x),
    "m_tau": -cx_wm * m + c * m / x - n * c * (p + m) / (2 * x) + c * A ** 2 * n * sp.sin(2 * f) / (2 * x),
    "f_x": (p + m) / (2 * x),
    "f_tau": c * (p - m) / (2 * x) - (p + m) / 2,
}
ss_check("wave map", eta / n, eta * sp.sin(f) ** 2 / (2 * rv ** 2), claimed)

# ---------------------------------------------------------------- 5. Yang-Mills: the scale and the U-system
print("\n5. Yang-Mills: the length scale and the scale-invariant small-scale system")
lY = sp.Symbol("l_YM", positive=True)
Wv, Wx, Wtau = sp.symbols("W W_x W_tau")
src = (lY ** 2 / n) * rr ** 2 * ((n / rr ** 2) * ((sp.exp(tau) * Wx) ** 2
                                                 + (A * sp.exp(tau) * (Wtau + x * Wx) / alpha) ** 2) / (2 * bb)
                                + n * (n - 1) * (1 - Wv ** 2) ** 2 / (4 * rr ** 4))
print("   (kappa/n) r^2 b rho_YM / b, self-similar variables:", sp.factor(sp.simplify(sp.expand(src))))
check("the YM source in x a_x/a is l_YM^2 e^{2 tau} x (tau-free): no exactly self-similar form",
      sp.simplify(sp.diff(src, tau) - 2 * src))
n_ = sp.Symbol("n", positive=True)
al = sp.Function("alpha")(t, r)
a = sp.Function("a")(t, r)
b = a ** 2
cc = al / a
Uf = sp.Function("U")(t, r)
eps = sp.Symbol("epsilon", positive=True)        # 1 / l_YM
W = 1 - eps * r * Uf
wd = sp.Symbol("wd")
K_YM, V_YM = n_ / r ** 2, n_ * (n_ - 1) * (1 - W ** 2) ** 2 / (4 * r ** 4)
PiW, PhiW = a * sp.diff(W, t) / al, sp.diff(W, r)
VW = (n_ * (n_ - 1) * sp.diff((1 - wd ** 2) ** 2, wd) / (4 * r ** 4)).subs(wd, W)
EL_W = sp.diff(r ** n_ * K_YM * PiW, t) - sp.diff(r ** n_ * cc * K_YM * PhiW, r) + r ** n_ * cc * b * VW
EL_U = sp.expand(EL_W).coeff(eps, 1)                                        # linear in U
PiU, PhiU = a * sp.diff(Uf, t) / al, sp.diff(Uf, r)
mult_l1 = sp.diff(PiU, t) - (sp.diff(r ** n_ * cc * PhiU, r) / r ** n_ - cc * b * n_ * Uf / r ** 2)   # l = 1 multiplet
extra = -(n_ - 2) * (b - 1) * cc * Uf / r ** 2 + sp.diff(cc, r) * Uf / r
check("U-system field equation = l = 1 multiplet equation (L = n) + [-(n-2)(b-1) c U/r^2 + c_r U/r]",
      EL_U / (-n_ * r ** (n_ - 1)) - (mult_l1 - extra))
check("EL_W has no epsilon^0 term (W = 1 is vacuum)", sp.expand(EL_W).coeff(eps, 0))
rhoYM = sp.expand((K_YM * (PhiW ** 2 + PiW ** 2) / (2 * b) + V_YM) / eps ** 2)
PrYM = sp.expand((K_YM * (PhiW ** 2 + PiW ** 2) / (2 * b) - V_YM) / eps ** 2)
rhoU, PrU = rhoYM.coeff(eps, 0), PrYM.coeff(eps, 0)
check("kappa rho_YM has no negative powers of epsilon = r/l_YM-suppressed corrections only",
      sum(rhoYM.coeff(eps, -k) for k in (1, 2)))
check("kappa (rho + P_r)_U = n (U_t^2 / alpha^2 + (U + r U_r)^2 / (r^2 a^2)): kinetic only, as for every model",
      rhoU + PrU - n_ * (sp.diff(Uf, t) ** 2 / al ** 2 + (Uf + r * sp.diff(Uf, r)) ** 2 / (r ** 2 * b)))
check("kappa (rho - P_r)_U = 2 n (n-1) U^2 / r^2  (l = 1 multiplet, U = sqrt2 U': n^2 U'^2 / r^2 = n^2 U^2/(2 r^2))",
      rhoU - PrU - 2 * n_ * (n_ - 1) * Uf ** 2 / r ** 2)

print("\nall ok" if all(ALL) else "\nSOME CHECKS FAILED")
