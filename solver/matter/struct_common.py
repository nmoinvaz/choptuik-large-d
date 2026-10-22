"""Shared helpers for the struct_*.py scripts (matter/STRUCTURE.md).

Metric (polar-areal), D = n + 2:  ds^2 = -alpha^2 dt^2 + a^2 dr^2 + r^2 dOmega_n^2,  b = a^2,  c = alpha / a.

Every matter model here is a single field psi(t, r) whose SO(n+1)-symmetric stress tensor has the form
    T_AB = K d_A psi d_B psi + g_AB Lag,   Lag = -1/2 K g^CD d_C psi d_D psi - V        (A, B in {t, r})
with K = K(r, psi) and V = V(r, psi) (the r-dependence comes from the sphere radius r). With Phi = psi_r, Pi = a psi_t / alpha:
    rho = -T^t_t = K (Phi^2 + Pi^2) / (2 b) + V,    P_r = T^r_r = K (Phi^2 + Pi^2) / (2 b) - V,   T_tr = K psi_t psi_r .
Einstein's equations G_ab = kappa T_ab in polar-areal gauge (verified by brute force in struct_eqs.py):
    a_r / a     = (kappa / n) r b rho + (n - 1)(1 - b) / (2 r)            (Hamiltonian constraint, from G^t_t)
    alpha_r/alpha = (kappa / n) r b P_r + (n - 1)(b - 1) / (2 r)          (slicing condition, from G^r_r)
    a_t / a     = (kappa / n) r T_tr                                       (momentum constraint, from G_tr)
and the field equation, from the reduced action r^n alpha a Lag (symmetric criticality; verified by brute force):
    d_t (r^n K Pi) = d_r (r^n c K Phi) + r^n c [ K_psi (Pi^2 - Phi^2) / 2 - b V_psi ],   psi_t = c Pi .
"""
import sympy as sp

t, r = sp.symbols("t r", real=True)


def sphere(n):
    """Coordinates th_1..th_n on S^n, the round metric (diagonal entries) and the unit embedding vector in R^{n+1}."""
    th = sp.symbols(f"th1:{n + 1}", real=True)
    h = []
    pref = sp.Integer(1)
    for i in range(n):
        h.append(pref)
        pref = pref * sp.sin(th[i]) ** 2
    nv = []
    pref = sp.Integer(1)
    for i in range(n):
        nv.append(pref * sp.cos(th[i]))
        pref = pref * sp.sin(th[i])
    nv.append(pref)
    return th, h, sp.Matrix(nv)


def polar_areal(n):
    """Coordinates, diagonal metric entries and the functions alpha, a for D = n + 2."""
    al = sp.Function("alpha")(t, r)
    a = sp.Function("a")(t, r)
    th, h, nv = sphere(n)
    X = [t, r] + list(th)
    g = [-al ** 2, a ** 2] + [r ** 2 * hi for hi in h]
    return X, g, al, a, th, nv


def einstein_diag(X, g):
    """Einstein tensor G_{ab} (lower indices) of a diagonal metric, by brute force."""
    D = len(X)
    gi = [1 / gg for gg in g]
    Gam = [[[sp.Integer(0)] * D for _ in range(D)] for _ in range(D)]
    for a_ in range(D):
        for b_ in range(D):
            for c_ in range(D):
                val = 0
                if a_ == c_:
                    val += sp.diff(g[a_], X[b_])
                if a_ == b_:
                    val += sp.diff(g[a_], X[c_])
                if b_ == c_:
                    val -= sp.diff(g[b_], X[a_])
                Gam[a_][b_][c_] = sp.simplify(gi[a_] * val / 2) if val != 0 else sp.Integer(0)
    Ric = sp.zeros(D, D)
    for b_ in range(D):
        for d_ in range(b_, D):
            s = 0
            for a_ in range(D):
                s += sp.diff(Gam[a_][b_][d_], X[a_]) - sp.diff(Gam[a_][b_][a_], X[d_])
                for e_ in range(D):
                    s += Gam[a_][a_][e_] * Gam[e_][b_][d_] - Gam[a_][d_][e_] * Gam[e_][b_][a_]
            Ric[b_, d_] = Ric[d_, b_] = sp.simplify(s)
    R = sp.simplify(sum(gi[i] * Ric[i, i] for i in range(D)))
    G = sp.zeros(D, D)
    for i in range(D):
        for j in range(D):
            G[i, j] = Ric[i, j] - (g[i] * R / 2 if i == j else 0)
    return G


def einstein_formulas(n, al, a):
    """Claimed general-n polar-areal components: G^t_t, G^r_r, G_tr."""
    b = a ** 2
    Gtt_mixed = -(n / (2 * r ** 2 * b)) * (2 * r * sp.diff(a, r) / a + (n - 1) * (b - 1))
    Grr_mixed = (n / (2 * r ** 2 * b)) * (2 * r * sp.diff(al, r) / al - (n - 1) * (b - 1))
    Gtr = n * sp.diff(a, t) / (r * a)
    return Gtt_mixed, Grr_mixed, Gtr


def sqrt_det(n, al, a, th):
    """sqrt(-det g) for the polar-areal metric, written without square roots of sin^2."""
    w = sp.Integer(1)
    for i in range(n):
        w = w * sp.sin(th[i]) ** (n - 1 - i)
    return al * a * r ** n * w


def box(f, X, g, sq):
    """Covariant wave operator for a diagonal metric; sq = sqrt(-det g)."""
    return sum(sp.diff(sq / g[i] * sp.diff(f, X[i]), X[i]) for i in range(len(X))) / sq


def reduced_el(K, V, psi, al, a, n):
    """Euler-Lagrange expression of the reduced action r^n alpha a [K psi_t^2/(2 alpha^2) - K psi_r^2/(2 a^2) - V]
    (zero on solutions). K, V are expressions in r and the symbol `psi` (replaced by the function)."""
    P = sp.Function("psi")(t, r)
    Kf = K.subs(psi, P)
    Vf = V.subs(psi, P)
    Lred = r ** n * al * a * (Kf * sp.diff(P, t) ** 2 / (2 * al ** 2) - Kf * sp.diff(P, r) ** 2 / (2 * a ** 2) - Vf)
    pt, pr, pp = sp.symbols("pt pr pp")
    Ls = Lred.subs({sp.diff(P, t): pt, sp.diff(P, r): pr}).subs(P, pp)
    dLt = sp.diff(Ls, pt).subs({pt: sp.diff(P, t), pr: sp.diff(P, r), pp: P})
    dLr = sp.diff(Ls, pr).subs({pt: sp.diff(P, t), pr: sp.diff(P, r), pp: P})
    dLp = sp.diff(Ls, pp).subs({pt: sp.diff(P, t), pr: sp.diff(P, r), pp: P})
    return sp.diff(dLt, t) + sp.diff(dLr, r) - dLp, P


def report(label, expr, simp=sp.simplify):
    """Print a residual and return True if it simplifies to zero."""
    z = simp(expr)
    ok = (z == 0)
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + ("" if ok else f": residual {z}"))
    return ok
