#!/usr/bin/env python3
"""Self-gravitating sigma model (hedgehog wave map) in D = n + 2 dimensions: derivation and checks (sympy).

Model.  S = int d^D x sqrt(-g) [ R/(16 pi G) - (f^2/2) g^{ab} G_AB(X) d_a X^A d_b X^B ],  target S^{n+1} = S^{D-1}
(round, unit radius), hedgehog X = (cos u, sin u * n^)(t, r, angles), n^ the unit normal of the spatial S^n.  The
pull-back of the target metric is h_ab = u_a u_b + sin^2 u gamma_ab (gamma = unit S^n metric), so the Lagrangian is
-(f^2/2)[(grad u)^2 + n sin^2 u / r^2] and T_ab = f^2 (h_ab - g_ab h/2).  G has dimension length^(D-2) and f^2 the
dimension of 1/G (both multiply two derivatives in the action), so eta = 4 pi G f^2 is dimensionless in every D.

Metric (polar-areal, as solver/dss.py):  ds^2 = -alpha^2 dt^2 + a^2 dr^2 + r^2 dOmega_n^2,  c = alpha/a.
With Phi = u_r, Pi = u_t / c  (= (a/alpha) u_t):

  (H)  a_r/a      = (eta r/n)(Phi^2+Pi^2) + eta a^2 sin^2u / r + (n-1)(1-a^2)/(2r)
  (S)  alpha_r/alpha = (eta r/n)(Phi^2+Pi^2) - eta a^2 sin^2u / r + (n-1)(a^2-1)/(2r)
       => c_r/c = (n-1)(a^2-1)/r - 2 eta a^2 sin^2u / r
  (M)  a_t/a      = (2 eta r/n) u_t u_r
  (W)  Phi_t = (c Pi)_r,   Pi_t = r^-n (r^n c Phi)_r - n alpha a sin(2u)/(2 r^2)

Self-similar variables (dss.py): tau = -ln(-t), x = r/(-t), SSH at x = 1 (gauge c(tau,1) = 1).  With Q = u_tau + x u_x
(so r Phi = x u_x, r Pi = x Q / c) and q = Q/c:

  (W)  q + (d_tau + x d_x) q - x^-n d_x(x^n c u_x) + n c a^2 sin(2u)/(2x^2) = 0
  (H)  x a_x/a = (eta/n)(x^2 u_x^2 + x^2 q^2) + eta a^2 sin^2u + (n-1)(1-a^2)/2
  (S)  x c_x/c = (n-1)(a^2-1) - 2 eta a^2 sin^2u
  (M)  (d_tau + x d_x) ln a = (2 eta/n) x u_x Q

In dss.py's first-order form, with the field normalised as dss.py's scalar, psi = sqrt(eta/n) u
(so that G_ab = 2n (d psi d psi - ...) for the kinetic part), p = sqrt(eta/n) r(Phi+Pi), m = sqrt(eta/n) r(Phi-Pi):

  a_x = a[(p^2+m^2) + (n-1)(1-a^2) + 2 eta a^2 sin^2u]/(2x)
  c_x = c[(n-1)(a^2-1) - 2 eta a^2 sin^2u]/x
  p_tau + (x-c) p_x =  c_x p - c p/x + n c (p+m)/(2x) - sqrt(eta n) c a^2 sin(2u)/(2x)
  m_tau + (x+c) m_x = -c_x m + c m/x - n c (p+m)/(2x) + sqrt(eta n) c a^2 sin(2u)/(2x)
  u_tau = sqrt(n/eta) [ c (p-m)/(2x) - (p+m)/2 ]      (and x u_x = sqrt(n/eta)(p+m)/2)

CSS (u = U(x), a(x), c(x)): (H) - (M) is algebraic,
  a^2 = [ (n-1)/2 - (eta/n) x^2 U'^2 (1 - x^2/c^2) ] / [ (n-1)/2 - eta sin^2 U ],
and at the SSH (x = c = 1): a^2 = (n-1)/(n-1-2 eta sin^2U), c'(1) = 0 (all n, eta), and regularity of (W) is
  (n-2) U'(1) = n a(1)^2 sin(2U(1))/2,
so D = 4 forces U(1) = pi/2 (Bizon-Wasserman), while for D > 4 U(1) is a free shooting parameter.

The script checks every claim above with sympy: the Einstein tensor for n = 2, 3, 4 (explicit coordinates on S^n)
against the general-n formulas, the wave equation from the action, the self-similar forms, the D = 4 reduction to
Bizon & Wasserman, PRD 62, 084031 (2000), Eqs. (8)-(17), the flat-space ground state 2 arctan(x/sqrt(n-1)), and the
large-n inner limits (fixed eta and eta = (n-1) eta_hat).

    python3 sigma_eqs.py          (about a minute)
"""
import sympy as sp

ok_all = True


def check(name, expr):
    global ok_all
    e = sp.simplify(expr)
    good = (e == 0)
    ok_all &= bool(good)
    print(("PASS " if good else "FAIL ") + name + ("" if good else "   residual: %s" % e))


t, r = sp.symbols('t r', real=True)
eta, f2, G = sp.symbols('eta f2 G', positive=True)
nn = sp.Symbol('n', positive=True)

# ---------------------------------------------------------------- 1. Einstein tensor and stress tensor, n = 2, 3, 4
def einstein_diag(n):
    th = sp.symbols('th1:%d' % (n + 1), real=True)
    X = [t, r] + list(th)
    al = sp.Function('alpha')(t, r)
    a = sp.Function('a')(t, r)
    g = [-al ** 2, a ** 2]
    w = r ** 2
    gam = []                                    # unit-sphere metric diagonal
    for k in range(n):
        f = sp.Integer(1)
        for j in range(k):
            f *= sp.sin(th[j]) ** 2
        gam.append(f)
        g.append(w * f)
    D = n + 2
    gm = sp.diag(*g)
    gi = sp.diag(*[1 / gg for gg in g])
    # Christoffels
    Gam = [[[0] * D for _ in range(D)] for _ in range(D)]
    for i in range(D):
        for j in range(D):
            for k in range(D):
                Gam[i][j][k] = sp.simplify(sum(gi[i, l] * (sp.diff(gm[l, j], X[k]) + sp.diff(gm[l, k], X[j])
                                                           - sp.diff(gm[j, k], X[l])) for l in range(D)) / 2)
    def Ric(i, j):
        s = 0
        for k in range(D):
            s += sp.diff(Gam[k][i][j], X[k]) - sp.diff(Gam[k][i][k], X[j])
            for l in range(D):
                s += Gam[k][k][l] * Gam[l][i][j] - Gam[k][j][l] * Gam[l][i][k]
        return sp.simplify(s)
    R = {(0, 0): Ric(0, 0), (1, 1): Ric(1, 1), (0, 1): Ric(0, 1), (2, 2): Ric(2, 2)}
    Rs = sp.simplify(gi[0, 0] * R[(0, 0)] + gi[1, 1] * R[(1, 1)] + n * gi[2, 2] * R[(2, 2)])
    Gtt = sp.simplify(gi[0, 0] * R[(0, 0)] - Rs / 2)       # G^t_t
    Grr = sp.simplify(gi[1, 1] * R[(1, 1)] - Rs / 2)       # G^r_r
    Gtr = sp.simplify(R[(0, 1)])                           # G_tr
    Gthth = sp.simplify(gi[2, 2] * R[(2, 2)] - Rs / 2)     # G^th_th
    return al, a, Gtt, Grr, Gtr, Gthth, th


print("== 1. Einstein equations for n = 2, 3, 4")
for n in (2, 3, 4):
    al, a, Gtt, Grr, Gtr, Gthth, th = einstein_diag(n)
    ar, at, alr = sp.diff(a, r), sp.diff(a, t), sp.diff(al, r)
    check("n=%d  G^t_t general-n formula" % n,
          Gtt + n / (2 * r ** 2) * ((n - 1) * (1 - a ** -2) + 2 * r * ar / a ** 3))
    check("n=%d  G^r_r general-n formula" % n,
          Grr + n / (2 * r ** 2) * ((n - 1) * (1 - a ** -2) - 2 * r * alr / (al * a ** 2)))
    check("n=%d  G_tr  general-n formula" % n, Gtr - n * at / (r * a))
    # matter: u(t,r); rho = -T^t_t, T^r_r, T_tr with eta = 4 pi G f^2  (8 pi G T = 2 eta (h - g h/2)/1)
    u = sp.Function('u')(t, r)
    ut, ur = sp.diff(u, t), sp.diff(u, r)
    trh = -ut ** 2 / al ** 2 + ur ** 2 / a ** 2 + n * sp.sin(u) ** 2 / r ** 2   # g^{cd} h_cd
    Ttt_up = f2 * (-ut ** 2 / al ** 2) - f2 * trh / 2          # T^t_t = f2 (h^t_t - h/2)
    Trr_up = f2 * (ur ** 2 / a ** 2) - f2 * trh / 2
    Ttr = f2 * ut * ur
    Tthth_up = f2 * (sp.sin(u) ** 2 / r ** 2) - f2 * trh / 2
    k8 = 2 * eta / f2                                          # 8 pi G = 2 eta / f^2
    Pi = a / al * ut
    Phi = ur
    # (H): G^t_t = 8piG T^t_t  <=>  a_r/a = ...
    Hform = ar / a - ((eta * r / n) * (Phi ** 2 + Pi ** 2) + eta * a ** 2 * sp.sin(u) ** 2 / r + (n - 1) * (1 - a ** 2) / (2 * r))
    check("n=%d  (H) from G^t_t = 8 pi G T^t_t" % n,
          sp.solve(Gtt - k8 * Ttt_up, ar)[0] / a - (ar / a - Hform))
    Sform = alr / al - ((eta * r / n) * (Phi ** 2 + Pi ** 2) - eta * a ** 2 * sp.sin(u) ** 2 / r + (n - 1) * (a ** 2 - 1) / (2 * r))
    check("n=%d  (S) from G^r_r = 8 pi G T^r_r" % n,
          sp.solve(Grr - k8 * Trr_up, alr)[0] / al - (alr / al - Sform))
    check("n=%d  (M) from G_tr = 8 pi G T_tr" % n,
          sp.solve(Gtr - k8 * Ttr, at)[0] / a - (2 * eta * r / n) * ut * ur)
    # The angular equation G^th_th = 8 pi G T^th_th follows from (H), (S), (M), (W) by the contracted Bianchi
    # identity (T_ab comes from a covariant action, so it is conserved when (W) holds); it is not used below.
    sg = al * a * r ** n
    box = (-sp.diff(sg / al ** 2 * ut, t) + sp.diff(sg / a ** 2 * ur, r)) / sg
    c_ = al / a
    lhs = sp.diff(Pi, t) - (sp.diff(r ** n * c_ * Phi, r) / r ** n - n * al * a * sp.sin(2 * u) / (2 * r ** 2))
    check("n=%d  (W) Pi_t form == al a (box u - n sin 2u/(2r^2))" % n,
          sp.expand(lhs + al * a * (box - n * sp.sin(2 * u) / (2 * r ** 2))))

# ---------------------------------------------------------------- 2. self-similar form (general n)
print("== 2. Self-similar variables (general n)")
n = nn
tau, x = sp.symbols('tau x', real=True)
uu = sp.Function('u')(tau, x)
aa = sp.Function('a')(tau, x)
cc = sp.Function('c')(tau, x)
# chain rule: d_t = e^tau (d_tau + x d_x), d_r = e^tau d_x, r = x e^-tau
E = sp.exp(tau)
Dt = lambda F: E * (sp.diff(F, tau) + x * sp.diff(F, x))
Dr = lambda F: E * sp.diff(F, x)
rr_ = x / E
Phi = Dr(uu)
Pi = Dt(uu) / cc
alpha_ = cc * aa
Q = sp.diff(uu, tau) + x * sp.diff(uu, x)
q = Q / cc
Wt = Dt(Pi) - (Dr(rr_ ** n * cc * Phi) / rr_ ** n - n * alpha_ * aa * sp.sin(2 * uu) / (2 * rr_ ** 2))
Wss = q + sp.diff(q, tau) + x * sp.diff(q, x) - x ** (-n) * sp.diff(x ** n * cc * sp.diff(uu, x), x) \
    + n * cc * aa ** 2 * sp.sin(2 * uu) / (2 * x ** 2)
check("(W) in (tau,x) == e^{-2tau} (W) in (t,r)", sp.expand(sp.simplify(Wt * sp.exp(-2 * tau) - Wss)))
Ht = Dr(aa) / aa - ((eta * rr_ / n) * (Phi ** 2 + Pi ** 2) + eta * aa ** 2 * sp.sin(uu) ** 2 / rr_ + (n - 1) * (1 - aa ** 2) / (2 * rr_))
Hss = x * sp.diff(aa, x) / aa - ((eta / n) * (x ** 2 * sp.diff(uu, x) ** 2 + x ** 2 * q ** 2) + eta * aa ** 2 * sp.sin(uu) ** 2
                                 + (n - 1) * (1 - aa ** 2) / 2)
check("(H) in (tau,x)", sp.simplify(Ht * rr_ - Hss))
Mt = Dt(aa) / aa - (2 * eta * rr_ / n) * Dt(uu) * Dr(uu)
Mss = (sp.diff(aa, tau) + x * sp.diff(aa, x)) / aa - (2 * eta / n) * x * sp.diff(uu, x) * Q
check("(M) in (tau,x)", sp.simplify(Mt / E - Mss))
# dss.py first-order form
sq = sp.sqrt(eta / n)
p = sq * (x * sp.diff(uu, x) + x * q)
m = sq * (x * sp.diff(uu, x) - x * q)
cx = cc * ((n - 1) * (aa ** 2 - 1) - 2 * eta * aa ** 2 * sp.sin(uu) ** 2) / x
ax = aa * ((p ** 2 + m ** 2) + (n - 1) * (1 - aa ** 2) + 2 * eta * aa ** 2 * sp.sin(uu) ** 2) / (2 * x)
check("dss-form a_x == (H)", sp.simplify(ax * x / aa - (Hss - x * sp.diff(aa, x) / aa) * (-1)))
# p, m equations: use (W) to eliminate u_tautau, and c_x from (S)
utt = sp.solve(Wss, sp.diff(uu, tau, 2))[0]
def onshell(F):
    F = F.subs(sp.diff(uu, tau, 2), utt)
    F = F.subs(sp.diff(cc, x), cx)
    return F
pe = sp.diff(p, tau) + (x - cc) * sp.diff(p, x) - (cx * p - cc * p / x + n * cc * (p + m) / (2 * x)
                                                   - sp.sqrt(eta * n) * cc * aa ** 2 * sp.sin(2 * uu) / (2 * x))
me = sp.diff(m, tau) + (x + cc) * sp.diff(m, x) - (-cx * m + cc * m / x - n * cc * (p + m) / (2 * x)
                                                   + sp.sqrt(eta * n) * cc * aa ** 2 * sp.sin(2 * uu) / (2 * x))
check("dss-form p equation", sp.simplify(onshell(pe)))
check("dss-form m equation", sp.simplify(onshell(me)))
check("u_tau from p, m", sp.simplify(sp.diff(uu, tau) - sp.sqrt(n / eta) * (cc * (p - m) / (2 * x) - (p + m) / 2)))

# ---------------------------------------------------------------- 3. CSS reduction
print("== 3. CSS reduction")
U = sp.Function('U')(x)
A = sp.Function('A')(x)          # a
C = sp.Function('C')(x)          # c
Up = sp.diff(U, x)
css = {uu: U, aa: A, cc: C}
def tocss(F):
    F = F.subs({sp.diff(uu, tau, 2): 0, sp.diff(uu, tau, x): 0, sp.diff(uu, tau): 0,
                sp.diff(aa, tau): 0, sp.diff(cc, tau): 0})
    return F.subs(css).doit()
Wc = sp.simplify(tocss(Wss))
Hc = tocss(Hss)
Mc = tocss(Mss)
a2alg = ((n - 1) / 2 - (eta / n) * x ** 2 * Up ** 2 * (1 - x ** 2 / C ** 2)) / ((n - 1) / 2 - eta * sp.sin(U) ** 2)
check("(H)-(M) algebraic a^2", sp.simplify(sp.solve(sp.expand(Hc - Mc), A ** 2)[0] - a2alg) if sp.solve(sp.expand(Hc - Mc), A ** 2) else sp.Integer(1))
Wform = (C ** 2 - x ** 2) * sp.diff(U, x, 2) + Up * (C * sp.diff(C, x) + n * C ** 2 / x - 2 * x + x ** 2 * sp.diff(C, x) / C) \
    - C ** 2 * A ** 2 * n * sp.sin(2 * U) / (2 * x ** 2)
check("CSS wave equation: (C^2-x^2)U'' + U'[CC' + nC^2/x - 2x + x^2 C'/C] = n C^2 a^2 sin2U/(2x^2)",
      sp.simplify(Wc * (-C) - Wform))
# SSH: x = C = 1  (plain symbols for the values there)
U1, V1, W1, C1, Cp1, A1 = sp.symbols('U1 V1 W1 C1 Cp1 A1')
pt = lambda F: F.subs({sp.diff(U, x, 2): W1, sp.diff(C, x): Cp1}).subs({Up: V1}).subs({U: U1, C: C1, A: A1})
a2ssh = sp.simplify(pt(a2alg).subs(C1, x).subs(x, 1))
check("SSH: a^2 = (n-1)/(n-1-2 eta sin^2 U)", sp.simplify(a2ssh - (n - 1) / (n - 1 - 2 * eta * sp.sin(U1) ** 2)))
Sc = (n - 1) * (A1 ** 2 - 1) - 2 * eta * A1 ** 2 * sp.sin(U1) ** 2
check("SSH: c'(1) = 0  (slicing at x = c = 1)", sp.simplify(Sc.subs(A1, sp.sqrt(a2ssh))))
Wssh = pt(Wform).subs({C1: 1, Cp1: 0, x: 1}).subs(A1, sp.sqrt(a2ssh))
reg = (n - 2) * V1 - n * a2ssh * sp.sin(2 * U1) / 2
check("SSH regularity (n-2)U'(1) = n a^2 sin(2U)/2", sp.simplify(Wssh - reg))

# D = 4 vs Bizon-Wasserman (alpha_BW = eta, A_BW = 1/a^2, e^{-delta} = c a^2, z = x/c):
print("== 4. D = 4: Bizon & Wasserman (2000), Eqs. (8)-(17)")
n4 = 2
FF = sp.Function('F')(t, r)
ABW = sp.Function('A')(t, r)
dl = sp.Function('delta')(t, r)
a_ = 1 / sp.sqrt(ABW)
al_ = sp.exp(-dl) * sp.sqrt(ABW)
c_ = al_ / a_
Phi_ = sp.diff(FF, r)
Pi_ = sp.diff(FF, t) / c_
# (10) d_t A = -2 alpha r A F_t F_r  vs (M)
Mbw = sp.diff(ABW, t) + 2 * eta * r * ABW * sp.diff(FF, t) * sp.diff(FF, r)
Mours = sp.diff(a_, t) / a_ - (2 * eta * r / n4) * sp.diff(FF, t) * sp.diff(FF, r)
check("BW (10) == (M)", sp.simplify(Mbw - (-2 * ABW) * Mours))
# (12) A_r = (1-A)/r - alpha r (A F_r^2 + A^-1 e^{2 delta} F_t^2 + 2 sin^2F/r^2) vs (H)
Hbw = sp.diff(ABW, r) - ((1 - ABW) / r - eta * r * (ABW * sp.diff(FF, r) ** 2 + sp.exp(2 * dl) * sp.diff(FF, t) ** 2 / ABW
                                                   + 2 * sp.sin(FF) ** 2 / r ** 2))
Hours = sp.diff(a_, r) / a_ - ((eta * r / n4) * (Phi_ ** 2 + Pi_ ** 2) + eta * a_ ** 2 * sp.sin(FF) ** 2 / r + (n4 - 1) * (1 - a_ ** 2) / (2 * r))
check("BW (12) == (H)", sp.simplify(Hbw - (-2 * ABW) * Hours))
# (11) delta_r = -alpha r (F_r^2 + A^-2 e^{2delta} F_t^2)  vs (S)-(H) combination: alpha_r/alpha - ...
Sours = sp.diff(al_, r) / al_ - ((eta * r / n4) * (Phi_ ** 2 + Pi_ ** 2) - eta * a_ ** 2 * sp.sin(FF) ** 2 / r + (n4 - 1) * (a_ ** 2 - 1) / (2 * r))
Dbw = sp.diff(dl, r) + eta * r * (sp.diff(FF, r) ** 2 + sp.exp(2 * dl) * sp.diff(FF, t) ** 2 / ABW ** 2)
# alpha_r/alpha = -delta_r + A_r/(2A); so (S) + (H)-relation: delta_r = A_r/(2A) - alpha_r/alpha
check("BW (11) == (H) + (S)", sp.simplify(Dbw - (-(Sours) + (-Hours)) ))
# (8)-(9) wave equation
boxBW = -sp.exp(dl) * sp.diff(sp.exp(dl) / ABW * sp.diff(FF, t), t) + sp.exp(dl) / r ** 2 * sp.diff(r ** 2 * sp.exp(-dl) * ABW * sp.diff(FF, r), r)
Wours = sp.diff(Pi_, t) - (sp.diff(r ** 2 * c_ * Phi_, r) / r ** 2 - n4 * al_ * a_ * sp.sin(2 * FF) / (2 * r ** 2))
check("BW (8)-(9) == (W)", sp.simplify((boxBW - sp.sin(2 * FF) / r ** 2) + Wours / (al_ * a_)))
# CSS: (13)-(17)
rho = x
Fx = sp.Function('F')(x)
Ax = sp.Function('A')(x)
zx = sp.Function('z')(x)
# our C = x/z, a^2 = 1/A
sub4 = {U: Fx, A: 1 / sp.sqrt(Ax), C: x / zx}
alg17 = 1 - Ax - 2 * eta * sp.sin(Fx) ** 2 + eta * Ax * x ** 2 * sp.diff(Fx, x) ** 2 * (1 - zx ** 2)
check("BW (17) == our algebraic a^2 (n=2)", sp.simplify((1 / sp.sqrt(Ax)) ** 2 - a2alg.subs(n, 2).subs(sub4).doit()
                                                       ).subs(Ax, sp.solve(alg17, Ax)[0]))
# (15): rho z' = z (1 + alpha (1 - z^2) rho^2 F'^2), from (S) with a^2 from (17)
Sx = x * sp.diff(C, x) / C - ((n - 1) * (A ** 2 - 1) - 2 * eta * A ** 2 * sp.sin(U) ** 2)
Sx4 = Sx.subs(n, 2).subs(sub4).doit()
zsol = sp.solve(Sx4, sp.diff(zx, x))[0]
claim15 = zx * (1 + eta * (1 - zx ** 2) * x ** 2 * sp.diff(Fx, x) ** 2) / x
check("BW (15) == (S) on the constraint (17)", sp.simplify((zsol - claim15).subs(Ax, sp.solve(alg17, Ax)[0])))
# (13): F'' + 2F'/rho - alpha(1+z^2) rho F'^3 - sin2F/(A rho^2 (1-z^2)) = 0
W4 = Wform.subs(n, 2).subs(sub4).doit()
W4 = W4.subs(sp.diff(zx, x), claim15)
Fpp = sp.solve(W4, sp.diff(Fx, x, 2))[0]
claim13 = -(2 / x) * sp.diff(Fx, x) + eta * (1 + zx ** 2) * x * sp.diff(Fx, x) ** 3 + sp.sin(2 * Fx) / (Ax * x ** 2 * (1 - zx ** 2))
check("BW (13) == (W) on (15), (17)", sp.simplify((Fpp - claim13).subs(Ax, sp.solve(alg17, Ax)[0])))

# ---------------------------------------------------------------- 5. flat space
print("== 5. Flat space (eta = 0, a = c = 1)")
U0 = 2 * sp.atan(x / sp.sqrt(n - 1))
flat = (1 - x ** 2) * sp.diff(U0, x, 2) + (n / x - 2 * x) * sp.diff(U0, x) - n * sp.sin(2 * U0) / (2 * x ** 2)
for nv in (2, 3, sp.Rational(7, 2), 10):
    check("U0 = 2 atan(x/sqrt(n-1)) solves the flat CSS equation, n = %s" % nv,
          sp.simplify(sp.expand_trig(flat.subs(n, nv))))

# ---------------------------------------------------------------- 6. inner limit (large n)
print("== 6. Large-n inner limit, s = r^2/(2n), physical t")
s, T = sp.symbols('s T', positive=True)
eps = sp.Symbol('epsilon', positive=True)   # 1/n
uI = sp.Function('u')(s, T)
cI = sp.Function('c')(s, T)
# r d_r = 2 s d_s; Phi = u_r = (r/n) u_s; with a^2 = 1 + O(1/n) (from (H)), alpha = c a
# (H): 2s d_s ln a = 2 eta s Pi^2 + eta a^2 sin^2u + (n-1)(1-a^2)/2 + eta (r^2/n)Phi^2 -> a^2 - 1 = O(eta/n)
# (S): 2s d_s ln c = (n-1)(a^2-1) - 2 eta a^2 sin^2 u.  Eliminating (n-1)(a^2-1) with (H) at leading order:
# (n-1)(a^2-1) = 2[2 eta s Pi^2 + eta sin^2 u] + O(1/n)  =>  d_s ln c = 2 eta Pi^2 = 2 eta u_t^2/c^2
# exact identity used: (n-1)(1-a^2)/2 = 2 s d_s ln a - 2 eta s(Pi^2 + Phi^2 2s/(n...)) - eta a^2 sin^2u
nsym = 1 / eps
aI = sp.Function('a')(s, T)
PiI = sp.diff(uI, T) / cI
PhiI = sp.sqrt(2 * s * eps) * sp.diff(uI, s)        # (r/n) u_s with r = sqrt(2 n s)
Hin = 2 * s * sp.diff(aI, s) / aI - (eta * 2 * s * (PhiI ** 2 + PiI ** 2) + eta * aI ** 2 * sp.sin(uI) ** 2
                                     + (nsym - 1) * (1 - aI ** 2) / 2)
Sin_ = 2 * s * sp.diff(cI, s) / cI - ((nsym - 1) * (aI ** 2 - 1) - 2 * eta * aI ** 2 * sp.sin(uI) ** 2)
# (S) + 2 (H) eliminates the O(n) term exactly:
comb = sp.expand(Sin_ + 2 * Hin)
claim = 2 * s * sp.diff(cI, s) / cI + 4 * s * sp.diff(aI, s) / aI - 4 * eta * s * (PhiI ** 2 + PiI ** 2)
check("(S) + 2(H) is free of n-sized terms (exact)", sp.simplify(comb - sp.expand(claim)))
print("     => at fixed eta, with a = 1 + O(1/n):   c_s = 2 eta u_t^2 / c   (inner slicing), and the potential's")
print("        gravity cancels at O(1).  Wave equation, inner form:")
# wave: Pi_t = r^-n d_r(r^n c Phi) - n c a^2 sin2u/(2r^2) with r = sqrt(2 n s), d_r = (r/n) d_s
rI = sp.sqrt(2 * s / eps)
dr = lambda F: (rI * eps) * sp.diff(F, s)
lap = (dr(rI ** nsym * cI * dr(uI)) / rI ** nsym)
Win = sp.diff(PiI, T) - (lap - nsym * cI * sp.sin(2 * uI) / (2 * rI ** 2))   # a = 1 at leading order
lead = sp.diff(PiI, T) - cI * (sp.diff(uI, s) - sp.sin(2 * uI) / (4 * s))
check("inner wave equation at eps -> 0: d_t(u_t/c) = c (u_s - sin(2u)/(4s))", sp.simplify(sp.limit(sp.simplify(Win - lead), eps, 0)))
uin = 2 * sp.atan(sp.sqrt(2 * s) / (-T))
check("flat inner ground state 2 atan(sqrt(2s)/(-t)) solves u_tt = u_s - sin2u/(4s)",
      sp.simplify(sp.expand_trig(sp.diff(uin, T, 2) - sp.diff(uin, s) + sp.sin(2 * uin) / (4 * s))))
# scaled coupling eta = (n-1) etah: then a^2 = 1/(1 - 2 etah sin^2 u) + O(1/n) at leading order (conical deficit)
etah = sp.Symbol('etahat', positive=True)
lead_a2 = sp.solve(sp.expand(eps * Hin.subs(eta, (nsym - 1) * etah)).subs(eps, 0), aI ** 2)
print("     eta = (n-1) eta_hat:  leading-order a^2 =", lead_a2)
print()
print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
