"""Field equations of the massless scalar l-multiplet in D = n + 2 (sympy checks).

Model: phi_m = psi(t, r) Y_m(angles), m = 1..N_l, the Y_m real degree-l harmonics on S^n normalised so that
sum_m Y_m^2 = 1 (then sum_m dY_m.dY_m = L / r^2 * (angular metric) / n with L = l (l + n - 1)), and
G_ab = 2n sum_m (d_a phi_m d_b phi_m - 1/2 g_ab (d phi_m)^2), the normalisation of solver/dss.py (l = 0 is the single
field Y = 1). Metric ds^2 = -alpha^2 dt^2 + a^2 dr^2 + r^2 dOmega_n^2, Phi = psi_r, Pi = (a/alpha) psi_t, c = alpha/a.

Checks (each prints OK or raises):
  1. Explicit Einstein tensor and multiplet stress tensor for (n, l) = (2, 1), (2, 2), (3, 1), (4, 1), (3, 2):
     tt and rr components give
        a_r/a     = r (Phi^2 + Pi^2) + L a^2 psi^2 / r + (n-1)(1 - a^2)/(2r)
        alpha_r/a = r (Phi^2 + Pi^2) - L a^2 psi^2 / r + (n-1)(a^2 - 1)/(2r)
     tr gives a_t = 2 r alpha Phi Pi (unchanged by l), mixed (t|r, angle) and off-diagonal angular components of the
     summed stress tensor vanish, and box phi_m = 0 is
        d_t Pi = r^-n d_r (r^n c Phi) - L alpha a psi / r^2,   L = l (l + n - 1).
  2. Self-similar form (tau = -ln(-t), x = r/(-t), p = r(Phi + Pi), m = r(Phi - Pi), psi):
        a_x   = a [p^2 + m^2 + 2 L a^2 psi^2 + (n-1)(1 - a^2)] / (2x)
        c_x   = c [(n-1)(a^2 - 1) - 2 L a^2 psi^2] / x
        psi_x = (p + m) / (2x)
        p_tau + (x - c) p_x =  c_x p - c p/x + n c (p + m)/(2x) - L c a^2 psi / x
        m_tau + (x + c) m_x = -c_x m + c m/x - n c (p + m)/(2x) + L c a^2 psi / x
     derived from the (t, r) equations; at L = 0 they are solver/dss.py's equations.
  3. D = 4 (n = 2): identical to Olabarrieta, Ventrella, Choptuik & Unruh, PRD 76, 124014 (2007), Eqs. (15)-(18), with
     psi_here = sqrt((2l + 1)/2) psi_paper.
  4. Centre series: with p, m, psi ~ x^l the order-k system for (P_k, M_k) has determinant
     -c0^2 (k - 1)(k - l)(k + l + n - 1)/k; at k = l the free datum is psi ~ q(tau) x^l, P_l = M_l = l q.
  5. Horizon (x = 1, c = 1): p(tau, 1) obeys p_tau + (1 - c_x(1) - n/2) p = (n/2) m - L a^2 psi; the constraint
     psi_tau + (p + m)/2 - c (p - m)/(2x) is x-independent, and equals psi_tau + m at x = 1.
"""
import itertools
import sympy as sp

t, r = sp.symbols("t r", real=True)


def einstein_diag(g, X):
    """Einstein tensor G_ab (lower indices) of a diagonal metric g (list of diagonal entries) in coordinates X."""
    D = len(X)
    ginv = [1 / gi for gi in g]
    Gam = {}
    for a, b, c in itertools.product(range(D), repeat=3):
        # Gamma^a_{bc} = 1/2 g^aa (d_b g_ac + d_c g_ab - d_a g_bc), diagonal metric
        val = 0
        if a == c:
            val += sp.diff(g[a], X[b])
        if a == b:
            val += sp.diff(g[a], X[c])
        if b == c:
            val -= sp.diff(g[b], X[a])
        Gam[a, b, c] = sp.simplify(ginv[a] * val / 2) if val != 0 else 0
    Ric = sp.zeros(D, D)
    for b, c in itertools.product(range(D), repeat=2):
        s = 0
        for a in range(D):
            s += sp.diff(Gam[a, b, c], X[a]) - sp.diff(Gam[a, b, a], X[c])
            for d in range(D):
                s += Gam[a, a, d] * Gam[d, b, c] - Gam[a, c, d] * Gam[d, b, a]
        Ric[b, c] = s
    R = sum(ginv[a] * Ric[a, a] for a in range(D))
    G = sp.zeros(D, D)
    for b, c in itertools.product(range(D), repeat=2):
        G[b, c] = Ric[b, c] - (g[b] * R / 2 if b == c else 0)
    return G


def harmonics(n, l, th):
    """Real degree-l harmonics on S^n (angles th), normalised to sum Y^2 = 1, from the embedding unit vector w."""
    w = []
    s = 1
    for i in range(n):
        w.append(s * sp.cos(th[i]))
        s = s * sp.sin(th[i])
    w.append(s)                                   # n + 1 components, sum w_i^2 = 1
    if l == 1:
        return w
    if l == 2:
        # traceless symmetric w_i w_j - delta_ij/(n+1), orthonormal combination with sum of squares 1:
        # sum_{i<j} (w_i w_j)^2 * 2 + sum_i (w_i^2 - 1/(n+1))^2 = 1 - 1/(n+1) = n/(n+1)
        Y = [sp.sqrt(2) * w[i] * w[j] for i in range(n + 1) for j in range(i + 1, n + 1)]
        # diagonal part: n traceless combinations; use the n+1 entries d_i = w_i^2 - 1/(n+1) (sum d_i = 0) projected
        # with an orthonormal basis of the hyperplane sum = 0 -- equivalently sum_i d_i^2 (Parseval)
        d = [w[i] ** 2 - sp.Rational(1, n + 1) for i in range(n + 1)]
        # Gram-Schmidt basis of {v : sum v = 0} in R^(n+1)
        basis = []
        for k in range(1, n + 1):
            v = [sp.Integer(0)] * (n + 1)
            for i in range(k):
                v[i] = sp.Integer(1)
            v[k] = -sp.Integer(k)
            nv = sp.sqrt(sum(vi ** 2 for vi in v))
            basis.append([vi / nv for vi in v])
        Y += [sum(b[i] * d[i] for i in range(n + 1)) for b in basis]
        scale = sp.sqrt(sp.Rational(n + 1, n))
        return [scale * y for y in Y]
    raise ValueError


def check_explicit(n, l):
    th = sp.symbols(f"th1:{n + 1}", real=True)
    X = [t, r] + list(th)
    al = sp.Function("alpha")(t, r)
    a = sp.Function("a")(t, r)
    psi = sp.Function("psi")(t, r)
    g = [-al ** 2, a ** 2]
    s = r ** 2
    for i in range(n):
        g.append(s)
        s = s * sp.sin(th[i]) ** 2
    Y = harmonics(n, l, th)
    assert sp.simplify(sum(y ** 2 for y in Y) - 1) == 0
    L = l * (l + n - 1)
    D = n + 2
    ginv = [1 / gi for gi in g]
    sqrtg = al * a * r ** n                       # sqrt(-g) on 0 < th_i < pi: alpha a r^n prod sin(th_i)^(n-1-i)
    for i in range(n - 1):
        sqrtg = sqrtg * sp.sin(th[i]) ** (n - 1 - i)
    # laplacian of each harmonic on the unit sphere: Delta Y = -L Y
    for y in Y:
        phi = psi * y
        box = sum(sp.diff(sqrtg * ginv[i] * sp.diff(phi, X[i]), X[i]) for i in range(D)) / sqrtg
        Phi, Pi = sp.diff(psi, r), a / al * sp.diff(psi, t)
        c = al / a
        # claimed: box phi = (Y / (alpha a)) [ -d_t Pi + r^-n d_r (r^n c Phi) - L alpha a psi / r^2 ]
        claim = y / (al * a) * (-sp.diff(Pi, t) + sp.diff(r ** n * c * Phi, r) / r ** n - L * al * a * psi / r ** 2)
        assert sp.simplify(box - claim) == 0, ("wave", n, l)
    # stress tensor summed over the multiplet
    T = sp.zeros(D, D)
    for y in Y:
        phi = psi * y
        dphi = [sp.diff(phi, xi) for xi in X]
        dd = sum(ginv[i] * dphi[i] ** 2 for i in range(D))
        for i, j in itertools.product(range(D), repeat=2):
            T[i, j] += dphi[i] * dphi[j] - (g[i] * dd / 2 if i == j else 0)
    T = T.applyfunc(sp.simplify)
    for i, j in itertools.product(range(D), repeat=2):
        if i != j and not (i < 2 and j < 2):
            assert sp.simplify(T[i, j]) == 0, ("offdiag", i, j)
    G = einstein_diag(g, X)
    Phi, Pi = sp.diff(psi, r), a / al * sp.diff(psi, t)
    a_r = a * (r * (Phi ** 2 + Pi ** 2) + L * a ** 2 * psi ** 2 / r + (n - 1) * (1 - a ** 2) / (2 * r))
    al_r = al * (r * (Phi ** 2 + Pi ** 2) - L * a ** 2 * psi ** 2 / r + (n - 1) * (a ** 2 - 1) / (2 * r))
    a_t = 2 * r * al * Phi * Pi
    sub = {sp.Derivative(a, r): a_r, sp.Derivative(al, r): al_r, sp.Derivative(a, t): a_t}
    for (i, j) in ((0, 0), (1, 1), (0, 1)):
        E = sp.simplify((G[i, j] - 2 * n * T[i, j]).subs(sub))
        assert E == 0, ("einstein", n, l, i, j, E)
    print(f"  explicit n = {n}, l = {l} (L = {L}, {len(Y)} harmonics): wave eq, G_tt, G_rr, G_tr, T off-diagonal OK",
          flush=True)


def check_selfsimilar():
    n, L = sp.symbols("n L", positive=True)
    tau, x = sp.symbols("tau x", real=True)
    P, M, A, C, S = [sp.Function(f)(tau, x) for f in ("p", "m", "a", "c", "psi")]
    # the (t, r) equations written for fields of (tau, x): t < 0, tau = -ln(-t), x = r/(-t)
    # d/dt|_r = e^tau (d_tau + x d_x), d/dr|_t = e^tau d_x, r = x e^-tau
    et = sp.exp(tau)
    rr = x / et
    Dt = lambda f: et * (sp.diff(f, tau) + x * sp.diff(f, x))
    Dr = lambda f: et * sp.diff(f, x)
    Phi = (P + M) / (2 * rr)
    Pi = (P - M) / (2 * rr)
    al = C * A
    # claimed self-similar equations
    cx = C * ((n - 1) * (A ** 2 - 1) - 2 * L * A ** 2 * S ** 2) / x
    ax = A * (P ** 2 + M ** 2 + 2 * L * A ** 2 * S ** 2 + (n - 1) * (1 - A ** 2)) / (2 * x)
    sx = (P + M) / (2 * x)
    px = (-sp.diff(P, tau) + cx * P - C * P / x + n * C * (P + M) / (2 * x) - L * C * A ** 2 * S / x) / (x - C)
    mx = (-sp.diff(M, tau) - cx * M + C * M / x - n * C * (P + M) / (2 * x) + L * C * A ** 2 * S / x) / (x + C)
    sub = {sp.Derivative(P, x): px, sp.Derivative(M, x): mx, sp.Derivative(A, x): ax,
           sp.Derivative(C, x): cx, sp.Derivative(S, x): sx}
    E = []
    E.append(Dr(A) / A - (rr * (Phi ** 2 + Pi ** 2) + L * A ** 2 * S ** 2 / rr + (n - 1) * (1 - A ** 2) / (2 * rr)))
    E.append(Dr(al) / al - (rr * (Phi ** 2 + Pi ** 2) - L * A ** 2 * S ** 2 / rr + (n - 1) * (A ** 2 - 1) / (2 * rr)))
    E.append(Dr(S) - Phi)                                             # Phi = psi_r
    E.append(Dt(Phi) - Dr(C * Pi))                                    # d_t Phi = d_r (c Pi)
    E.append(Dt(Pi) - Dr(rr ** n * C * Phi) / rr ** n + L * al * A * S / rr ** 2)
    for i, e in enumerate(E):
        z = sp.simplify(sp.expand(e.subs(sub)))
        assert z == 0, ("selfsimilar", i, z)
    # the constraint psi_t = c Pi, in self-similar form K = psi_tau + (p+m)/2 - c (p-m)/(2x), is x-independent:
    K = sp.diff(S, tau) + (P + M) / 2 - C * (P - M) / (2 * x)
    Kx = sp.diff(K, x).subs(sub)
    Kx = Kx.subs(sp.Derivative(S, tau, x), sp.diff(sx, tau)).subs(sp.Derivative(S, x, tau), sp.diff(sx, tau))
    assert sp.simplify(Kx) == 0, ("constraint", sp.simplify(Kx))
    # l = 0 limit equals solver/dss.py
    dss = dict(ax=A * ((P ** 2 + M ** 2) + (n - 1) * (1 - A ** 2)) / (2 * x), cx=C * (n - 1) * (A ** 2 - 1) / x)
    assert sp.simplify(ax.subs(L, 0) - dss["ax"]) == 0 and sp.simplify(cx.subs(L, 0) - dss["cx"]) == 0
    cx0 = cx.subs(L, 0)
    assert sp.simplify(px.subs(L, 0) - (-sp.diff(P, tau) + cx0 * P - C * P / x + n * C * (P + M) / (2 * x)) / (x - C)) == 0
    assert sp.simplify(mx.subs(L, 0) - (-sp.diff(M, tau) - cx0 * M + C * M / x - n * C * (P + M) / (2 * x)) / (x + C)) == 0
    print("  self-similar equations follow from the (t, r) equations; psi_t = c Pi constraint is x-independent; "
          "L = 0 gives solver/dss.py: OK", flush=True)
    # horizon: x = 1, c = 1 -> regular p obeys p_tau + (1 - c_x - n/2) p = n m / 2 - L a^2 psi
    num = (-sp.diff(P, tau) + cx * P - C * P / x + n * C * (P + M) / (2 * x) - L * C * A ** 2 * S / x)
    h = num.subs({x: 1}).subs(C.subs(x, 1), 1)
    cx1 = ((n - 1) * (A ** 2 - 1) - 2 * L * A ** 2 * S ** 2).subs(x, 1)
    claim = -(sp.diff(P, tau).subs(x, 1) + (1 - cx1 - n / 2) * P.subs(x, 1) - n * M.subs(x, 1) / 2
              + L * A.subs(x, 1) ** 2 * S.subs(x, 1))
    assert sp.simplify(h - claim) == 0
    print("  horizon regularity condition: OK", flush=True)


def check_paper():
    """n = 2 against Olabarrieta et al. 2007, Eqs. (15)-(18) (G = c = 1):
       a_r/a = (2l+1)/2 r [Pi^2 + Phi^2 + l(l+1) a^2 psi^2 / r^2] - (a^2 - 1)/(2r)
       al_r/al = (2l+1)/2 r [Pi^2 + Phi^2 - l(l+1) a^2 psi^2 / r^2] + (a^2 - 1)/(2r)
       d_t Phi = d_r (al Pi / a),  d_t Pi = r^-2 d_r (r^2 al Phi / a) - l(l+1) a al psi / r^2."""
    l = sp.symbols("l", positive=True)
    a, al, Phi_p, Pi_p, psi_p = sp.symbols("a alpha Phi_p Pi_p psi_p")
    k = sp.sqrt((2 * l + 1) / 2)
    n = 2
    L = l * (l + n - 1)
    Phi, Pi, psi = k * Phi_p, k * Pi_p, k * psi_p
    ours_a = r * (Phi ** 2 + Pi ** 2) + L * a ** 2 * psi ** 2 / r + (n - 1) * (1 - a ** 2) / (2 * r)
    ours_al = r * (Phi ** 2 + Pi ** 2) - L * a ** 2 * psi ** 2 / r + (n - 1) * (a ** 2 - 1) / (2 * r)
    pap_a = (2 * l + 1) / 2 * r * (Pi_p ** 2 + Phi_p ** 2 + l * (l + 1) * a ** 2 * psi_p ** 2 / r ** 2) - (a ** 2 - 1) / (2 * r)
    pap_al = (2 * l + 1) / 2 * r * (Pi_p ** 2 + Phi_p ** 2 - l * (l + 1) * a ** 2 * psi_p ** 2 / r ** 2) + (a ** 2 - 1) / (2 * r)
    assert sp.simplify(ours_a - pap_a) == 0 and sp.simplify(ours_al - pap_al) == 0
    # wave equation: linear in psi with the same centrifugal coefficient L = l(l+1) at n = 2
    assert sp.simplify(L - l * (l + 1)) == 0
    print("  n = 2: identical to Olabarrieta et al. (2007) Eqs. (15)-(18) with psi = sqrt((2l+1)/2) psi_paper: OK",
          flush=True)


def check_centre():
    """Order-k centre system for (P_k, M_k) after eliminating Psi_k = (P_k + M_k)/(2k)."""
    k, l, n, c0 = sp.symbols("k l n c0", positive=True)
    L = l * (l + n - 1)
    P, M = sp.symbols("P M")
    Psi = (P + M) / (2 * k)
    # unknown-dependent parts of (x * p-equation) and (x * m-equation) at order k (moved to the left side):
    ep = c0 * ((1 - k) * P - n / 2 * (P + M) + L * Psi)
    em = c0 * ((k - 1) * M + n / 2 * (P + M) - L * Psi)
    Mx = sp.Matrix([[sp.diff(ep, P), sp.diff(ep, M)], [sp.diff(em, P), sp.diff(em, M)]])
    det = sp.factor(sp.simplify(Mx.det()))
    assert sp.simplify(det + c0 ** 2 * (k - 1) * (k - l) * (k + l + n - 1) / k) == 0, det
    # at k = l the null vector is P = M (any l): check
    Ml = Mx.subs(k, l)
    assert sp.simplify(Ml * sp.Matrix([1, 1])) == sp.zeros(2, 1)
    print(f"  centre series: det = {det}; at k = l the null vector is P_l = M_l: OK", flush=True)


def check_chi_form():
    """psi = r^l chi: (i) flat space, the l-mode operator in D = n + 2 acting on r^l g is r^l times the radial
    operator in n + 2l + 2 dimensions; (ii) the curved-space evolution and constraint equations of ell_collapse.py."""
    n, l = sp.symbols("n l", positive=True)
    L = l * (l + n - 1)
    g = sp.Function("g")(r)
    lhs = sp.diff(r ** n * sp.diff(r ** l * g, r), r) / r ** n - L * r ** (l - 2) * g
    rhs = r ** l * (sp.diff(g, r, 2) + (n + 2 * l) * sp.diff(g, r) / r)
    assert sp.simplify(sp.expand(lhs - rhs)) == 0
    a = sp.Function("a")(r); c = sp.Function("c")(r); chi = sp.Function("chi")(r); Pt = sp.Symbol("Pt")
    psi = r ** l * chi
    Phi = sp.diff(psi, r)
    X = sp.diff(chi, r)
    # c_r from the two constraints (c = alpha/a): c_r/c = (n-1)(a^2-1)/r - 2 L a^2 psi^2 / r
    cr = c * ((n - 1) * (a ** 2 - 1) / r - 2 * L * a ** 2 * psi ** 2 / r)
    # Pi_t = r^-n (r^n c Phi)_r - L c a^2 psi / r^2;  claimed Pt_t = Pi_t / r^l:
    pit = (sp.diff(r ** n * c * Phi, r) / r ** n - L * c * a ** 2 * psi / r ** 2) / r ** l
    Ne = n + 2 * l
    claim = sp.diff(r ** Ne * c * X, r) / r ** Ne - c * chi * (l ** 2 * (a ** 2 - 1) / r ** 2
                                                              + 2 * l * L * r ** (2 * l - 2) * a ** 2 * chi ** 2)
    diff = (pit - claim).subs(sp.Derivative(c, r), cr)
    assert sp.simplify(sp.expand(diff)) == 0
    # constraint for w = r^(n-1) (1 - 1/a^2), with a_r from the Hamiltonian constraint and Pi = r^l Pt
    Pi = r ** l * Pt
    ar = a * (r * (Phi ** 2 + Pi ** 2) + L * a ** 2 * psi ** 2 / r + (n - 1) * (1 - a ** 2) / (2 * r))
    w = r ** (n - 1) * (1 - 1 / a ** 2)
    K = r ** (2 * l - 1) * ((l * chi + r * X) ** 2 + r ** 2 * Pt ** 2)
    e = (sp.diff(w, r) + 2 * K * w - 2 * r ** (n - 1) * K - 2 * L * r ** (n + 2 * l - 2) * chi ** 2)
    assert sp.simplify(sp.expand(e.subs(sp.Derivative(a, r), ar))) == 0
    print("  r^l reduction (flat: radial operator in n + 2l + 2 dimensions) and ell_collapse.py chi-form "
          "equations: OK", flush=True)


if __name__ == "__main__":
    import sys
    check_chi_form()
    if "--chi" in sys.argv:
        sys.exit(0)
    check_selfsimilar()
    check_paper()
    check_centre()
    cases = [(2, 1), (2, 2), (3, 1)] if "--quick" in sys.argv else [(2, 1), (2, 2), (3, 1), (3, 2), (4, 1)]
    for n_, l_ in cases:
        check_explicit(n_, l_)
    print("all checks passed")
