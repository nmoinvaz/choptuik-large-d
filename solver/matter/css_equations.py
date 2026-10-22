#!/usr/bin/env python3
"""Sympy derivation and checks of the D-dimensional equations used in css_model.py (matter/CSS.md, section 1).

    python3 css_equations.py          (about a minute)

A. Curvature of ds^2 = -alpha^2 dt^2 + a^2 dr^2 + r^2 dOmega_n^2 computed from the metric with explicit sphere
   coordinates in D = 4, 5, 6, against the general-n formulas
       G^t_t = -(n/(2 r^2 a^2)) [(n-1)(a^2-1) + 2 r a_r/a],   G^r_r = (n/(2 r^2 a^2)) [-(n-1)(a^2-1) + 2 r alpha_r/alpha],
       G_tr = n a_t/(r a),   box F = (alpha a r^n)^-1 [-(a r^n F_t/alpha)_t + (alpha r^n F_r/a)_r].
B. The matter equation as the Euler-Lagrange equation of S = int sqrt(-g) [R - 2 h |grad F|^2], h = (1-kappa F Fbar)^-2
   (F, Fbar independent), and the constraints (Hamiltonian, slicing, momentum) from G_ab = T_ab with
   T_ab = h (F_a Fbar_b + Fbar_a F_b) - g_ab h grad F . grad Fbar  (equivalently R_ab = h (F_a Fbar_b + Fbar_a F_b), HE97 (8a)).
C. css_model.pde_terms (the equations the numerics use, in tau = -ln(-t), w = (x/X)^2) against the covariant
   equations of A-B: every residual is a nonzero multiple of a covariant equation.
D. The CSS ODEs against published D = 4 forms: HE95a eqs (17a-d) (kappa = 0, with F = sqrt(4 pi) phi for their
   8 pi normalization), HE97 eqs (22a-c) (D = 4, any kappa), and the first equation (12) of Alvarez-Gaume & Hatefi
   (any d, kappa = 1); and the map of Hamade-Horne-Stewart's tau = i(1-F)/(1+F) to the HE97 kappa = 1 normalization.
"""
import sys, os
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from css_model import pde_terms

ok_all = True


def report(name, expr):
    global ok_all
    z = sp.simplify(expr) == 0
    ok_all &= z
    print("  %-70s %s" % (name, "0" if z else "NONZERO: %s" % sp.simplify(expr)))


# ---------------------------------------------------------------------------------------------------------------
print("A. curvature of the warped metric, explicit sphere coordinates vs general-n formulas")
t, r = sp.symbols('t r', real=True)
al = sp.Function('alpha')(t, r)
aa = sp.Function('a')(t, r)
Ff = sp.Function('F')(t, r)


def einstein_diag(coords, g):
    D = len(coords)
    ginv = [1 / g[i] for i in range(D)]
    Gam = [[[0] * D for _ in range(D)] for _ in range(D)]
    for a_ in range(D):
        for b_ in range(D):
            for c_ in range(D):
                val = 0
                if a_ == b_:
                    val += sp.diff(g[a_], coords[c_])
                if a_ == c_:
                    val += sp.diff(g[a_], coords[b_])
                if b_ == c_:
                    val -= sp.diff(g[b_], coords[a_])
                Gam[a_][b_][c_] = sp.simplify(ginv[a_] * val / 2)
    Ric = sp.zeros(D, D)
    for b_ in range(D):
        for c_ in range(b_, D):
            val = 0
            for a_ in range(D):
                val += sp.diff(Gam[a_][b_][c_], coords[a_]) - sp.diff(Gam[a_][b_][a_], coords[c_])
                for d_ in range(D):
                    val += Gam[a_][a_][d_] * Gam[d_][b_][c_] - Gam[a_][c_][d_] * Gam[d_][b_][a_]
            Ric[b_, c_] = Ric[c_, b_] = sp.simplify(val)
    Rs = sp.simplify(sum(ginv[i] * Ric[i, i] for i in range(D)))
    return Ric, Rs, ginv


for nv in (2, 3, 4):
    th = sp.symbols('th1:%d' % (nv + 1), real=True)
    coords = [t, r] + list(th)
    g = [-al ** 2, aa ** 2]
    pref = r ** 2
    for k in range(nv):
        g.append(pref)
        pref = pref * sp.sin(th[k]) ** 2
    Ric, Rs, ginv = einstein_diag(coords, g)
    Gtt = ginv[0] * (Ric[0, 0] - g[0] * Rs / 2)        # G^t_t
    Grr = ginv[1] * (Ric[1, 1] - g[1] * Rs / 2)
    Gtr = Ric[0, 1]
    ar, alr, at = sp.diff(aa, r), sp.diff(al, r), sp.diff(aa, t)
    report("D=%d  G^t_t" % (nv + 2), Gtt + nv / (2 * r ** 2 * aa ** 2) * ((nv - 1) * (aa ** 2 - 1) + 2 * r * ar / aa))
    report("D=%d  G^r_r" % (nv + 2), Grr - nv / (2 * r ** 2 * aa ** 2) * (-(nv - 1) * (aa ** 2 - 1) + 2 * r * alr / al))
    report("D=%d  G_tr" % (nv + 2), Gtr - nv * at / (r * aa))
    # scalar wave operator from sqrt(-g) g^ab
    sq = al * aa * r ** nv     # angular factor cancels
    box = (sp.diff(sq * ginv[0] * sp.diff(Ff, t), t) + sp.diff(sq * ginv[1] * sp.diff(Ff, r), r)) / sq
    box_n = (-sp.diff(aa * r ** nv * sp.diff(Ff, t) / al, t) + sp.diff(al * r ** nv * sp.diff(Ff, r) / aa, r)) / (al * aa * r ** nv)
    report("D=%d  box F" % (nv + 2), box - box_n)

# ---------------------------------------------------------------------------------------------------------------
print("B. matter equation (Euler-Lagrange) and constraints, general n")
n, kap, om = sp.symbols('n kappa omega', positive=True)
Gf = sp.Function('G')(t, r)        # stands for Fbar
h = (1 - kap * Ff * Gf) ** -2
Ft, Fr, Gt, Gr = sp.diff(Ff, t), sp.diff(Ff, r), sp.diff(Gf, t), sp.diff(Gf, r)
L = al * aa * r ** n * (-2 * h * (-Ft * Gt / al ** 2 + Fr * Gr / aa ** 2))
EL_G = sp.diff(sp.diff(L, Gt), t) + sp.diff(sp.diff(L, Gr), r) - sp.diff(L, Gf)   # = 0
box_F = (-sp.diff(aa * r ** n * Ft / al, t) + sp.diff(al * r ** n * Fr / aa, r)) / (al * aa * r ** n)
gradFF = -Ft ** 2 / al ** 2 + Fr ** 2 / aa ** 2
matter = box_F + 2 * kap * Gf / (1 - kap * Ff * Gf) * gradFF       # HE97 (8b): box F = -2 kappa Fbar (grad F)^2/(1-kappa|F|^2)
report("E-L(Fbar) = -2 h alpha a r^n [box F + 2 kappa Fbar (grad F)^2/rho]", EL_G + 2 * h * al * aa * r ** n * matter)
gradFG = -Ft * Gt / al ** 2 + Fr * Gr / aa ** 2
Ttt = -(1 / al ** 2) * (h * 2 * Ft * Gt) - h * gradFG               # T^t_t = g^tt T_tt
Trr = (1 / aa ** 2) * (h * 2 * Fr * Gr) - h * gradFG
Ttr = h * (Ft * Gr + Gt * Fr)
Gtt_n = -n / (2 * r ** 2 * aa ** 2) * ((n - 1) * (aa ** 2 - 1) + 2 * r * sp.diff(aa, r) / aa)
Grr_n = n / (2 * r ** 2 * aa ** 2) * (-(n - 1) * (aa ** 2 - 1) + 2 * r * sp.diff(al, r) / al)
Gtr_n = n * sp.diff(aa, t) / (r * aa)
Pi_PiB = (aa / al) ** 2 * Ft * Gt          # |Pi|^2,  Pi = (a/alpha) F_t
Phi_PhiB = Fr * Gr
ham = sp.diff(aa, r) / aa - (r / n * h * (Pi_PiB + Phi_PhiB) - (n - 1) * (aa ** 2 - 1) / (2 * r))
report("G^t_t - T^t_t = -(n/(r a^2)) * Hamiltonian", (Gtt_n - Ttt) + n / (r * aa ** 2) * ham)
sli = sp.diff(al, r) / al - sp.diff(aa, r) / aa - (n - 1) * (aa ** 2 - 1) / r
report("(G^r_r - T^r_r) + (G^t_t - T^t_t) = (n/(r a^2)) * slicing", (Grr_n - Trr) + (Gtt_n - Ttt) - n / (r * aa ** 2) * sli)
report("G_tr - T_tr  (momentum: n a_t/(r a) = h (F_t Fbar_r + Fbar_t F_r))", Gtr_n - Ttr - (n * sp.diff(aa, t) / (r * aa) - h * (Ft * Gr + Gt * Fr)))

# ---------------------------------------------------------------------------------------------------------------
print("C. css_model.pde_terms against the covariant equations, in tau = -ln(-t), w = (x/X)^2")
T, W, X = sp.symbols('T W X', positive=True)
sR, sI, pR, pI, vv, bb = [sp.Function(nm)(T, W) for nm in ('sR', 'sI', 'pR', 'pI', 'v', 'b')]
xx = X * sp.sqrt(W)
rr = xx * sp.exp(-T)
Dt = lambda f: sp.exp(T) * (sp.diff(f, T) + 2 * W * sp.diff(f, W))      # d/dt at fixed r
Dr = lambda f: sp.exp(T) * 2 * sp.sqrt(W) / X * sp.diff(f, W)           # d/dr at fixed t
A_ = sp.sqrt(1 + W * vv)                  # a
AL = bb * A_                              # alpha
psi = sR + sp.I * sI
psib = sR - sp.I * sI
P = pR + sp.I * pI
Pb = pR - sp.I * pI
F_ = sp.exp(-sp.I * om * T) * psi
G_ = sp.exp(sp.I * om * T) * psib
rho_ = 1 - kap * psi * psib
h_ = 1 / rho_ ** 2
terms = pde_terms(n, kap, om, X, W, sR, sI, pR, pI, vv, bb,
                  sp.diff(sR, W), sp.diff(sI, W), sp.diff(pR, W), sp.diff(pI, W), sp.diff(vv, W), sp.diff(bb, W),
                  sp.diff(sR, W, 2), sp.diff(sI, W, 2))
dsR, dsI, dpR, dpI, hamM, sliM, vtauM = terms
# (1) definition of P:  Pi = (a/alpha) F_t = e^{tau} e^{-i om tau} P  <=>  psi_tau = model
rep_psi = {sp.Derivative(sR, T): dsR, sp.Derivative(sI, T): dsI}
Pi_cov = (A_ / AL) * Dt(F_)
report("(a/alpha) F_t - e^{tau - i om tau} P, with psi_tau from model", sp.expand(Pi_cov.subs(rep_psi) - sp.exp(T) * sp.exp(-sp.I * om * T) * P))
# (2) wave equation: d_t(r^n Pi) = d_r(b r^n F_r) + b r^n (2 kappa Fbar/rho)(F_r^2 - Pi^2)
Pi_ = sp.exp(T) * sp.exp(-sp.I * om * T) * P
wave = -Dt(rr ** n * Pi_) + Dr(bb * rr ** n * Dr(F_)) + bb * rr ** n * 2 * kap * G_ / rho_ * (Dr(F_) ** 2 - Pi_ ** 2)
target = -rr ** n * sp.exp(2 * T) * sp.exp(-sp.I * om * T) * ((sp.diff(pR, T) + sp.I * sp.diff(pI, T)) - (dpR + sp.I * dpI))
chk = sp.simplify(sp.expand((wave - target) / (rr ** n * sp.exp(2 * T) * sp.exp(-sp.I * om * T))))
report("wave eq.  == -r^n e^{2tau-i om tau} (P_tau - model)", chk)
# (3) Hamiltonian, slicing, momentum (as functions of tau, w)
Pi2 = Pi_ * sp.exp(T) * sp.exp(sp.I * om * T) * Pb                 # |Pi|^2 = e^{2tau}|P|^2
Phi2 = Dr(F_) * Dr(G_)
ham_cov = Dr(A_) / A_ - (rr / n * h_ * (Pi2 + Phi2) - (n - 1) * (A_ ** 2 - 1) / (2 * rr))
# expected: ham_cov = (e^{tau} X^-2 ... ) * hamM ; compute ratio
report("Hamiltonian: covariant - (e^tau sqrt(w)/(X (1+wv))) * model", sp.simplify(ham_cov - sp.exp(T) * sp.sqrt(W) / (X * (1 + W * vv)) * hamM))
sli_cov = Dr(AL) / AL - Dr(A_) / A_ - (n - 1) * (A_ ** 2 - 1) / rr
report("slicing: covariant - (2 e^tau sqrt(w)/(X b)) * model", sp.simplify(sli_cov - 2 * sp.exp(T) * sp.sqrt(W) / (X * bb) * sliM))
mom_cov = n * Dt(A_) / (rr * A_) - h_ * (Dt(F_) * Dr(G_) + Dt(G_) * Dr(F_))
mom_cov = mom_cov.subs(rep_psi)
tgt_m = sp.diff(vv, T) - vtauM
report("momentum: covariant - n e^{2tau} sqrt(w)/(2 X (1+wv)) (v_tau - model)",
       sp.simplify(sp.expand(mom_cov - n * sp.exp(2 * T) * sp.sqrt(W) / (2 * X * (1 + W * vv)) * tgt_m)))

# ---------------------------------------------------------------------------------------------------------------
print("D. CSS ODEs vs published D = 4 forms")
x = sp.symbols('x', positive=True)
f, fb, b, u = [sp.Function(nm)(x) for nm in ('f', 'fb', 'b', 'u')]
q, qb = sp.diff(f, x), sp.diff(fb, x)
# steady model in x (X -> x/sqrt(w) is irrelevant: use x directly).  P = -p, p = (i om f - x f')/b
p = (sp.I * om * f - x * q) / b
pb = (-sp.I * om * fb - x * qb) / b
rho = 1 - kap * f * fb
hh = 1 / rho ** 2
# steady wave eq. in x:  0 = (i om - 1) P - x P_x + x^-n (x^n b f_x)_x + b (2 kappa fbar/rho)(f_x^2 - P^2), P = -p
Px = -p
wave_x = (sp.I * om - 1) * Px - x * sp.diff(Px, x) + sp.diff(x ** n * b * q, x) / x ** n + b * 2 * kap * fb / rho * (q ** 2 - Px ** 2)
bprime = (n - 1) * b * u / x                                        # slicing
u_alg = 2 * x * hh / (n * (n - 1)) * (x * (p * pb + q * qb) + b * (p * qb + pb * q))   # Ham + momentum combined
u_ham = (1 + u) * (2 * x / n * hh * (p * pb + q * qb) - (n - 1) * u / x)
u_mom = -(4 / n) * (1 + u) * b * hh * (p * qb + pb * q) / 2
report("algebraic u = combination of Hamiltonian and momentum", sp.simplify((u_ham - u_mom) - (1 + u) * (n - 1) / x * (u_alg - u)))
# second-order form (cf. HE97 22b-c):
Delta = b ** 2 - x ** 2
f2 = sp.diff(f, x, 2)
rhs_he = q * (-2 * sp.I * om * x + 2 * x - (n - 1) * u * x - b ** 2 / x * (n + (n - 1) * u) - 4 * sp.I * kap * om * x * f * fb / rho) \
    - f * (om ** 2 + sp.I * om * (1 - (n - 1) * u)) - 2 * kap * fb / rho * (Delta * q ** 2 + om ** 2 * f ** 2)
expr = sp.expand(wave_x * b).subs(sp.Derivative(b, x), bprime)
report("b*wave = Delta f'' - rhs (general n)", sp.simplify(sp.expand(expr - (Delta * f2 - rhs_he))))
rhs_he97 = q * (-2 * sp.I * om * x - x * (u - 2) - b ** 2 / x * (u + 2) - 4 * sp.I * kap * om * x * f * fb / rho) \
    - f * (om ** 2 + sp.I * om * (1 - u)) - 2 * kap / rho * fb * (Delta * q ** 2 + om ** 2 * f ** 2)
report("n = 2: HE97 (22b-c) as printed", sp.simplify(rhs_he.subs(n, 2) - rhs_he97))
report("n = 2: HE97 (22a) b' = b u/x", bprime.subs(n, 2) - b * u / x)
# HE95a (17a-d), kappa = 0, n = 2, their phi = F/sqrt(4 pi): q,p scale the same way
# Express q', p' from the model (kappa = 0, n = 2): p' from wave eq, q' from x q = i om f - b p
pS = sp.Function('pS')(x)
fS = sp.Function('fS')(x)
qS = sp.Function('qS')(x)
# relations: q = f', p = (i om f - x q)/b  ->  b p = i om f - x q  (differentiate)
bp_rel = sp.diff(b * pS, x) - (sp.I * om * qS - qS - x * sp.diff(qS, x))
# wave eq. with P = -p (kappa = 0): 0 = -(i om - 1) p + x p' + b q' + (n b/x + b') q
wave_k0 = -(sp.I * om - 1) * pS + x * sp.diff(pS, x) + b * sp.diff(qS, x) + (n * b / x + sp.diff(b, x)) * qS
sol = sp.solve([bp_rel.subs(sp.Derivative(b, x), bprime), wave_k0.subs(sp.Derivative(b, x), bprime)],
               [sp.diff(qS, x), sp.diff(pS, x)], dict=True)[0]
bpl, bmi = sp.I * om + u + 1, sp.I * om + u - 1
Dl = b ** 2 - x ** 2
q17 = -(u + 2) / x * qS + (-x * bpl * qS + b * bmi * pS) / Dl
p17 = -u / x * pS + (b * bpl * qS - x * bmi * pS) / Dl
report("n = 2, kappa = 0: HE95a (17a) q'", sp.simplify(sol[sp.diff(qS, x)].subs(n, 2) - q17))
report("n = 2, kappa = 0: HE95a (17a) p'", sp.simplify(sol[sp.diff(pS, x)].subs(n, 2) - p17))
# 17c-d with 4 pi |phi|^2 = |F|^2:
report("n = 2, kappa = 0: HE95a (17c) u' = (u+1)[x(|q|^2+|p|^2) - u/x]", sp.simplify(u_ham.subs({n: 2, kap: 0}) - (1 + u) * (x * (p * pb + q * qb) - u / x)))
report("n = 2, kappa = 0: HE95a (17d) u' = -2(u+1) b Re(qbar p)", sp.simplify(u_mom.subs({n: 2, kap: 0}) + (1 + u) * b * (p * qb + pb * q)))
# Alvarez-Gaume & Hatefi eq. (12), first line (d = n + 2, kappa = 1)
bprime_alg = ((n - 1) * b * u_alg / x).subs(kap, 1)
fq = sp.diff(f, x); fbq = sp.diff(fb, x)
agh = -(2 * x * (b ** 2 - x ** 2) / (n * b * (-1 + f * fb) ** 2) * fq * fbq
        - 2 * sp.I * om * (b ** 2 - x ** 2) / (n * b * (-1 + f * fb) ** 2) * (f * fbq - fb * fq)
        - 2 * om ** 2 * x * f * fb / (n * b * (-1 + f * fb) ** 2))
report("AGH12 (12) first line: b' = (n-1) b u_alg / x", sp.simplify(bprime_alg - agh))
# Hamade-Horne-Stewart: tau = i(1-F)/(1+F), L = R - (1/2)|d tau|^2/(Im tau)^2  ->  R - 2|dF|^2/(1-|F|^2)^2
Fs, Fbs, dF, dFb = sp.symbols('F Fb dF dFb')
tau_ = sp.I * (1 - Fs) / (1 + Fs)
taub = -sp.I * (1 - Fbs) / (1 + Fbs)
imt = (tau_ - taub) / (2 * sp.I)
dtau = sp.diff(tau_, Fs) * dF
dtaub = sp.diff(taub, Fbs) * dFb
report("HHS: (1/2)|dtau|^2/(Im tau)^2 = 2|dF|^2/(1-|F|^2)^2", sp.simplify(dtau * dtaub / (2 * imt ** 2) - 2 * dF * dFb / (1 - Fs * Fbs) ** 2))

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
