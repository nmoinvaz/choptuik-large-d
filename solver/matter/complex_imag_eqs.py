#!/usr/bin/env python3
"""sympy: the imaginary sector of a complex scalar on the real DSS solution, from the covariant equations
(matter/COMPLEX_CRITICAL.md section 3).

    python3 complex_imag_eqs.py            all checks; exit 1 if one fails
    python3 complex_imag_eqs.py --mutate   the control: the p equation's (n/2) c (p + m) term with n/2 -> n, which
                                           check 3 must reject (exit 1 when it does, 2 if it does not)

Checks (each prints its residual, which must simplify to 0):
 1. Box psi from the metric ds^2 = -alpha^2 dt^2 + a^2 dr^2 + r^2 dOmega_n^2 with explicit sphere coordinates
    (D = 4, 5) equals the general-n form (1/(alpha a r^n)) [-(a r^n psi_t/alpha)_t + (alpha r^n psi_r/a)_r].
 2. For phi = phi_R + i eps psi the stress tensor T_ab = Re(d_a phi d_b phibar) - 1/2 g_ab |d phi|^2 has no term of
    order eps: T = T[phi_R] + eps^2 T[psi].  So at linear order psi does not act on the metric or on phi_R, and it
    obeys Box psi = 0 on the frozen DSS metric (the field equation Box phi = 0 is linear).
 3. With c = alpha/a, Phi = psi_r, Pi = psi_t/c, p = r (Phi + Pi), m = r (Phi - Pi), tau = -ln(-t), x = r/(-t), the
    equations of hp/src/model.rs
        x (x - c) p_x = -x p_tau + G p - c p + (n/2) c (p + m),
        x (x + c) m_x = -x m_tau - G m + c m - (n/2) c (p + m),      G = x c_x,
    are each a multiple of (alpha a r^n) Box psi, with no other term (no a, no Einstein equation used); on the DSS
    background G = x c_x = (n - 1) c (a^2 - 1) by the slicing equation.
 4. Modes psi ~ e^{lam tau} (periodic): d_tau -> d_tau + lam in both equations.  They are homogeneous of degree one in
    (p, m), and with the background's own (p, m) at lam = 0 they are the background's p, m equations: psi = phi_R,
    the U(1) phase rotation, is an exact lam = 0 mode.
 5. The series used by complex_imag.py: at the centre x^k (P_k, M_k) obeys a pointwise 2 x 2 system whose determinant
    is -c0^2 (k - 1)(k - 1 + n) (free data P_1 = -M_1); at the past light cone x = 1 - u, u^k P_k obeys
    (d_tau + lam + (1 + c1)(k + 1) - n/2) P_k = rhs, with c = 1 + c1 u + ..., and the m equation gives M_{k+1}
    explicitly (factor -2 (k + 1)).
"""
import sys

import sympy as sp

MUT = "--mutate" in sys.argv
fails = []


def report(name, expr):
    e = sp.simplify(expr)
    ok = e == 0
    print("%-70s %s" % (name, "0" if ok else "NONZERO: %s" % e))
    if not ok:
        fails.append(name)


t, r = sp.symbols('t r')
al = sp.Function('alpha')(t, r)
a = sp.Function('a')(t, r)
psi = sp.Function('psi')(t, r)


def box_general(n):
    return (-sp.diff(a * r ** n * sp.diff(psi, t) / al, t) + sp.diff(al * r ** n * sp.diff(psi, r) / a, r)) / (al * a * r ** n)


# 1. explicit sphere coordinates
for D in (4, 5):
    n = D - 2
    th = sp.symbols('th1:%d' % (n + 1))
    coords = [t, r] + list(th)
    diag = [-al ** 2, a ** 2]
    w = r ** 2
    for i in range(n):
        diag.append(w)
        w = w * sp.sin(th[i]) ** 2
    g = sp.diag(*diag)
    ginv = g.inv()
    sg = sp.sqrt(-g.det())
    sg = sp.simplify(sg)
    bx = sum(sp.diff(sg * ginv[i, i] * sp.diff(psi, coords[i]), coords[i]) for i in range(D)) / sg
    report("1. Box psi, explicit sphere (D = %d) - general-n form" % D, sp.simplify(bx - box_general(n)))

# 2. stress tensor of phi = phi_R + i eps psi
eps = sp.symbols('eps')
dR = sp.symbols('dR0:3')     # components of d phi_R
dI = sp.symbols('dI0:3')     # components of d psi
gm = sp.diag(-1, 1, 1)       # any metric (signature irrelevant for the algebra)
gi = gm.inv()
dphi = [dR[i] + sp.I * eps * dI[i] for i in range(3)]
dphib = [dR[i] - sp.I * eps * dI[i] for i in range(3)]
norm = sum(gi[i, j] * dphi[i] * dphib[j] for i in range(3) for j in range(3))
for i in range(3):
    for j in range(3):
        T = (dphi[i] * dphib[j] + dphib[i] * dphi[j]) / 2 - gm[i, j] * norm / 2
        report("2. T_%d%d: coefficient of eps" % (i, j), sp.expand(T).coeff(eps, 1))

# 3. the p, m equations from Box psi
nn = sp.symbols('n', positive=True)
tau, x = sp.symbols('tau x')
Psi = sp.Function('Psi')(tau, x)
cf = sp.Function('c')(tau, x)
# psi(t, r) = Psi(tau, x), tau = -ln(-t), x = r/(-t): d_t = (d_tau + x d_x)/(-t), d_r = d_x/(-t); write T = -t > 0
Tm = sp.exp(-tau)            # T = -t = e^{-tau}; d_tau acts on it
Psi_t = (sp.diff(Psi, tau) + x * sp.diff(Psi, x)) / Tm
Psi_r = sp.diff(Psi, x) / Tm
Phi = Psi_r
Pi = Psi_t / cf
p = x * Tm * (Phi + Pi)
m = x * Tm * (Phi - Pi)
G = x * sp.diff(cf, x)
half = nn if MUT else nn / 2
Ep = x * (x - cf) * sp.diff(p, x) + x * sp.diff(p, tau) - (G * p - cf * p + half * cf * (p + m))
Em = x * (x + cf) * sp.diff(m, x) + x * sp.diff(m, tau) - (-G * m + cf * m - nn / 2 * cf * (p + m))
# (alpha a r^n) Box psi in self-similar variables: (r^n c Phi)_r - (r^n Pi)_t, with alpha/a = c
W = (sp.diff((x * Tm) ** nn * cf * Phi, x) / Tm
     - ((sp.diff((x * Tm) ** nn * Pi, tau) + x * sp.diff((x * Tm) ** nn * Pi, x)) / Tm))
W = sp.simplify(W / (x * Tm) ** nn * Tm ** 2)       # dimensionless, times T^2 / r^n
for name, E in (("p", Ep), ("m", Em)):
    ratio = sp.simplify(E / W)
    free = ratio.free_symbols | {f for f in ratio.atoms(sp.Derivative)}
    has_psi = ratio.has(Psi)
    print("   %s equation / Box psi = %s" % (name, ratio))
    report("3. %s equation is a multiple of Box psi (no Psi left in the ratio)" % name, 1 if has_psi else 0)

# the Box psi expression itself, written out, has no a: only c and its derivatives
report("3. Box psi in (tau, x) contains only c (not a)", 1 if W.has(sp.Function('a')) else 0)

# 4. modes and the phase rotation
lam = sp.symbols('lam')
Pf, Mf = sp.Function('P')(tau, x), sp.Function('Mm')(tau, x)
Gs, cs, = sp.symbols('G c')
eqp = lambda P, M_, l: -x * (sp.diff(P, tau) + l * P) + Gs * P - cs * P + nn / 2 * cs * (P + M_) - x * (x - cs) * sp.diff(P, x)
eqm = lambda P, M_, l: -x * (sp.diff(M_, tau) + l * M_) - Gs * M_ + cs * M_ - nn / 2 * cs * (P + M_) - x * (x + cs) * sp.diff(M_, x)
psi_mode = sp.exp(lam * tau) * Pf
lhs = -x * sp.diff(psi_mode, tau) + Gs * psi_mode - cs * psi_mode + nn / 2 * cs * (psi_mode + sp.exp(lam * tau) * Mf) \
    - x * (x - cs) * sp.diff(psi_mode, x)
report("4. e^{lam tau} P in the p equation = e^{lam tau} x (d_tau + lam) form", sp.simplify(lhs / sp.exp(lam * tau) - eqp(Pf, Mf, lam)))
k1 = sp.symbols('k1')
report("4. homogeneous of degree one in (p, m)", sp.simplify(eqp(k1 * Pf, k1 * Mf, lam) - k1 * eqp(Pf, Mf, lam))
       + sp.simplify(eqm(k1 * Pf, k1 * Mf, lam) - k1 * eqm(Pf, Mf, lam)))
# background equations of model.rs (p, m rows), with the background's p, m:
pb, mb = sp.Function('p')(tau, x), sp.Function('m')(tau, x)
bgp = -x * sp.diff(pb, tau) + Gs * pb - cs * pb + nn / 2 * cs * (pb + mb) - x * (x - cs) * sp.diff(pb, x)
bgm = -x * sp.diff(mb, tau) - Gs * mb + cs * mb - nn / 2 * cs * (pb + mb) - x * (x + cs) * sp.diff(mb, x)
report("4. frozen equations at lam = 0 with (P, M) = background (p, m) = model.rs rows",
       sp.simplify(eqp(pb, mb, 0) - bgp) + sp.simplify(eqm(pb, mb, 0) - bgm))

# 5. series: indicial structure at the centre and at the light cone (coefficients constant in tau for the check)
k = sp.symbols('k', integer=True, positive=True)
c0, c1, Pk, Mk = sp.symbols('c0 c1 P_k M_k')
# centre: c = c0 + O(x^2), G = O(x^2); leading x^k terms of the two equations
Pc, Mc = Pk * x ** k, Mk * x ** k
lp = sp.expand(eqp(Pc, Mc, lam).subs({Gs: 0, cs: c0}))
lm = sp.expand(eqm(Pc, Mc, lam).subs({Gs: 0, cs: c0}))
cp_ = sp.Poly(sp.expand(lp / x ** k), x).coeff_monomial(1)
cm_ = sp.Poly(sp.expand(lm / x ** k), x).coeff_monomial(1)
Mat = sp.Matrix([[sp.diff(cp_, Pk), sp.diff(cp_, Mk)], [sp.diff(cm_, Pk), sp.diff(cm_, Mk)]])
report("5. centre: det of the order-k matrix + c0^2 (k-1)(k-1+n)", sp.factor(Mat.det() + c0 ** 2 * (k - 1) * (k - 1 + nn)))
report("5. centre: k = 1 null vector is P_1 = -M_1", sp.simplify((Mat.subs(k, 1) * sp.Matrix([1, -1]))[0])
       + sp.simplify((Mat.subs(k, 1) * sp.Matrix([1, -1]))[1]))
# light cone: x = 1 - u, c = 1 + c1 u, G = x c_x = -(1 - u) c1; P = P_k u^k, M = M_k u^k (lam stands for d_tau + lam)
u = sp.symbols('u')
cu = 1 + c1 * u
Gu = -(1 - u) * c1
Pu, Mu = Pk * u ** k, Mk * u ** k
xp = 1 - u
ep_ = -xp * lam * Pu + Gu * Pu - cu * Pu + nn / 2 * cu * (Pu + Mu) + xp * (xp - cu) * sp.diff(Pu, u)   # p_x = -p_u
em_ = -xp * lam * Mu - Gu * Mu + cu * Mu - nn / 2 * cu * (Pu + Mu) + xp * (xp + cu) * sp.diff(Mu, u)
lead_p = sp.limit(sp.expand(ep_ / u ** k), u, 0)
report("5. light cone: u^k coefficient of the p equation = -(lam + (1+c1)(k+1) - n/2) P_k + (n/2) M_k",
       sp.expand(lead_p - (-(lam + (1 + c1) * (k + 1) - nn / 2) * Pk + nn / 2 * Mk)))
lead_m = sp.limit(sp.expand(em_ / u ** (k - 1)), u, 0)
report("5. light cone: u^(k-1) coefficient of the m equation = 2 k M_k (so M_k is explicit)", sp.expand(lead_m - 2 * k * Mk))

print("FAILED: %s" % fails if fails else "all checks pass")
if MUT:
    sys.exit(1 if fails else 2)
sys.exit(1 if fails else 0)
