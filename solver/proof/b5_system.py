#!/usr/bin/env python3
"""Lemma B5(1) (proof/B5_inner.md): the exact M-harmonic model in inner variables, in the form the fixed point uses.

The M-harmonic model (proof/B5_inner.md, section 1): the outer equations of proof/C_selection.md section 1 with p, m
on the odd harmonics |mu| <= M and b, c on the even harmonics |nu| <= M + 1 (the truncation of the certified inner
problem, solver/inner_stokes/lateral_np.py), every binary product projected onto the result's harmonic set, G p
computed as [[c(b-1)] p] and b(p^2 + m^2) as [b [p^2 + m^2]]:
    x c_x = (n-1)[c(b-1)],   x b_x = [b[p^2+m^2]] - (n-1)[b(b-1)],
    x^2 p_x - x[c p_x] = -w x p_th + (n-1)[[c(b-1)] p] - [c p] + n/2 [c(p+m)],
    x^2 m_x + x[c m_x] = -w x m_th - (n-1)[[c(b-1)] m] + [c m] - n/2 [c(p+m)].
Inner variables (C_selection.md section 9): x = sqrt(n) xi, y = p - m, sigma = sqrt(n)(p + m), b = 1 + eps eta,
eps = 1/n; then z = xi^2, y = xi P, sigma = z Q, eta = z Theta, D = 1 + s, D2 = 2 + s (s = i mu w, diagonal),
Cop X = [C X], Bop X = [[C Theta] X]. The claim checked here: the model is equivalent to
    2 z P' = Phi,
    (I - (eps/z) Cop^2) Phi = Eop Q - D P + eps(1-eps) Cop Bop P,     Eop = Cop (I - eps D) + eps(1-eps) z Bop,
    2 z^2 Q' = Cop Phi + A2,   A2 = -z D2 Q + (1-eps) z Bop P,
    C' = (1-eps) [C Theta]/2,
    2 eps z Theta' = h,   h = E_eps - (1+eps) Theta + eps z ([Theta E_eps] - (1-eps)[Theta Theta]),
    E_eps = ([P P] + eps z [Q Q])/2,
(' = d/dz), and at eps = 0 (Theta = [P P]/2) to the inner system of fixed_point.py:
    P' = (V + [(C-1)Q])/(2z), V' = V/(2z^2) + ([(C-1)Q] + [(C-1)W])/(2z^2) + (-(2+s)Q - (1+s)W + [C E P])/(2z),
    C' = [C E]/2,  V = Q - (1+s) P,  W = V + [(C-1)Q] (= Phi),  E = [P P]/2.

Checks (random complex points, mpmath at 60 digits; each must be ~1e-50, and each mutation must fail):
    python3 b5_system.py                    model <-> outer equations; eps = 0 <-> fixed_point's system
    python3 b5_system.py --mutate=epsCB     drops the eps(1-eps) Cop Bop P term: the model check must fail
    python3 b5_system.py --mutate=theta     (1+eps) -> (1-eps) in h: the model check must fail
    python3 b5_system.py --mutate=t5        drops the eps(1-eps)(Cop' Bop + Cop Bop')P term of Phi': the Phi' check
                                            and the eps -> 0 limit check must fail
Further checks: Phi' of f_eps (state (P, Phi, C, vt), vt = Theta - [P P]/2) equals the total derivative of the Phi
relation along the flow; and f_eps -> f_0 as eps -> 0 with vt held arbitrary, i.e. P' and Phi' of the limit do not
depend on vt (the O(1) vt-terms cancel; proof/B5_inner.md section 2).
The vector fields f_eps and f_0 are used by b5_numeric.py.
"""
import sys
import random

import mpmath as mp
import sympy as sp

M = 1
ODD = [mu for mu in range(-M, M + 1) if mu % 2]
EVEN = [nu for nu in range(-(M + 1), M + 2) if nu % 2 == 0]
MUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)


def proj(A, B, keep, zero=0):
    """[A B] on the harmonic set keep (A, B dicts harmonic -> value)"""
    out = {k: zero for k in keep}
    for i, a in A.items():
        for j, b in B.items():
            if i + j in out:
                out[i + j] = out[i + j] + a * b
    return out


# ----------------------------------------------------------------------------------------------------------------------
# the model in the (P, Q, C, Theta) form: derivatives from the state (numeric, mpmath or python complex)
# ----------------------------------------------------------------------------------------------------------------------
class Model:
    """the exact M-model in inner variables; state (P, Phi, C, Theta) or (P, Q, C, Theta); eps may be 0 (inner limit:
    then Theta is not an ODE variable, h/(2 eps z) is replaced by the limit d/dz [P P]/2 when theta_inner=True)"""

    def __init__(self, w, eps, mp_=mp, mutate=None):
        self.w, self.eps, self.mp, self.mut = w, eps, mp_, mutate
        self.s = {mu: 1j * mu * w for mu in ODD}

    # linear operators on odd vectors as matrices (lists of lists) in the order ODD
    def opC(self, C):
        return [[C.get(a - b, 0) for b in ODD] for a in ODD]

    def opB(self, C, Th):
        CT = proj(C, Th, EVEN)
        return [[CT.get(a - b, 0) for b in ODD] for a in ODD]

    @staticmethod
    def mv(Mx, v):
        return [sum(Mx[i][j] * v[j] for j in range(len(v))) for i in range(len(Mx))]

    @staticmethod
    def mm(A, B):
        n = len(A)
        return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

    @staticmethod
    def madd(A, B, a=1, b=1):
        return [[a * A[i][j] + b * B[i][j] for j in range(len(A))] for i in range(len(A))]

    def solve(self, A, v):
        Mx = self.mp.matrix(A)
        return list(self.mp.lu_solve(Mx, self.mp.matrix(v)))

    def ident(self):
        n = len(ODD)
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def parts(self, z, P, C, Th):
        """operators at a state; P, C, Th dicts"""
        e = self.eps
        Cop, Bop = self.opC(C), self.opB(C, Th)
        I_ = self.ident()
        Dg = [[(1 + self.s[a]) if a == b else 0 for b in ODD] for a in ODD]
        D2 = [[(2 + self.s[a]) if a == b else 0 for b in ODD] for a in ODD]
        C2 = self.mm(Cop, Cop)
        Mop = self.madd(I_, C2, 1, -e / z)
        Eop = self.madd(self.mm(Cop, self.madd(I_, Dg, 1, -e)), Bop, 1, e * (1 - e) * z)
        return Cop, Bop, Dg, D2, Mop, Eop

    def Q_of(self, z, P, Phi, C, Th):
        """Q from the Phi relation: Eop Q = Mop Phi + D P - eps(1-eps) Cop Bop P"""
        e = self.eps
        Cop, Bop, Dg, D2, Mop, Eop = self.parts(z, P, C, Th)
        Pv = [P[mu] for mu in ODD]
        Phv = [Phi[mu] for mu in ODD]
        CBP = self.mv(Cop, self.mv(Bop, Pv))
        epsCB = 0 if self.mut == "epsCB" else e * (1 - e)
        rhs = [a + b - epsCB * c for a, b, c in zip(self.mv(Mop, Phv), self.mv(Dg, Pv), CBP)]
        return dict(zip(ODD, self.solve(Eop, rhs)))

    def Phi_of(self, z, P, Q, C, Th):
        """Phi from the state (P, Q, C, Theta)"""
        e = self.eps
        Cop, Bop, Dg, D2, Mop, Eop = self.parts(z, P, C, Th)
        Pv = [P[mu] for mu in ODD]
        Qv = [Q[mu] for mu in ODD]
        CBP = self.mv(Cop, self.mv(Bop, Pv))
        epsCB = 0 if self.mut == "epsCB" else e * (1 - e)
        rhs = [a - b + epsCB * c for a, b, c in zip(self.mv(Eop, Qv), self.mv(Dg, Pv), CBP)]
        return dict(zip(ODD, self.solve(Mop, rhs)))

    def h_of(self, z, P, Q, Th):
        e = self.eps
        PP, QQ = proj(P, P, EVEN), proj(Q, Q, EVEN)
        Ee = {k: (PP[k] + e * z * QQ[k]) / 2 for k in EVEN}
        TE, TT = proj(Th, Ee, EVEN), proj(Th, Th, EVEN)
        one = (1 - e) if self.mut == "theta" else (1 + e)
        return {k: Ee[k] - one * Th[k] + e * z * (TE[k] - (1 - e) * TT[k]) for k in EVEN}

    def rates_PQ(self, z, P, Q, C, Th):
        """(P', Q', C', Theta') from the state (P, Q, C, Theta); eps > 0"""
        e = self.eps
        Phi = self.Phi_of(z, P, Q, C, Th)
        Cop, Bop, Dg, D2, Mop, Eop = self.parts(z, P, C, Th)
        Pv, Qv, Phv = ([X[mu] for mu in ODD] for X in (P, Q, Phi))
        A2 = [-z * a + (1 - e) * z * b for a, b in zip(self.mv(D2, Qv), self.mv(Bop, Pv))]
        dQ = [(a + b) / (2 * z * z) for a, b in zip(self.mv(Cop, Phv), A2)]
        dP = [p / (2 * z) for p in Phv]
        CT = proj(C, Th, EVEN)
        dC = {k: (1 - e) * CT[k] / 2 for k in EVEN}
        h = self.h_of(z, P, Q, Th)
        dTh = {k: h[k] / (2 * e * z) for k in EVEN}
        return dict(zip(ODD, dP)), dict(zip(ODD, dQ)), dC, dTh, Phi

    # ------------------------------------------------------------------------------------------------------------------
    # the state Y = (P, Phi, C, vt), vt = Theta - [P P]/2 (the deviation from the inner slow manifold of the b-mode)
    # ------------------------------------------------------------------------------------------------------------------
    def opX(self, X):
        """X . (odd vector) for an even-harmonic field X, as a matrix"""
        return [[X.get(a - b, 0) for b in ODD] for a in ODD]

    def f_eps(self, z, P, Phi, C, vt):
        """exact rates (P', Phi', C', vt') in the state (P, Phi, C, vt); eps > 0. Phi' from differentiating the Phi
        relation:  Phi' = Mop^-1 [Eop (Cop Phi + A2)/(2z^2) - Mop' Phi + Eop' Q - (D - e1 Cop Bop) Phi/(2z)
        + e1 (Cop' Bop + Cop Bop') P],  e1 = eps(1-eps), with Theta' = h/(2 eps z)"""
        e = self.eps
        e1 = e * (1 - e)
        PP = proj(P, P, EVEN)
        E0 = {k: v / 2 for k, v in PP.items()}
        Th = {k: E0[k] + vt[k] for k in EVEN}
        Q = self.Q_of(z, P, Phi, C, Th)
        Cop, Bop, Dg, D2, Mop, Eop = self.parts(z, P, C, Th)
        I_ = self.ident()
        Pv, Qv, Phv = ([X[mu] for mu in ODD] for X in (P, Q, Phi))
        CT = proj(C, Th, EVEN)
        dC = {k: (1 - e) * CT[k] / 2 for k in EVEN}
        h = self.h_of(z, P, Q, Th)
        dTh = {k: h[k] / (2 * e * z) for k in EVEN}
        dP = {mu: Phi[mu] / (2 * z) for mu in ODD}
        dPP = proj(P, dP, EVEN)
        dvt = {k: dTh[k] - dPP[k] for k in EVEN}           # d/dz [P P]/2 = [P P']
        # operators' derivatives
        dCop = self.opX(dC)
        dCT = {k: proj(dC, Th, EVEN)[k] + proj(C, dTh, EVEN)[k] for k in EVEN}
        dBop = self.opX(dCT)
        C2 = self.mm(Cop, Cop)
        dC2 = self.madd(self.mm(dCop, Cop), self.mm(Cop, dCop))
        dMop = self.madd(C2, dC2, e / (z * z), -e / z)
        dEop = self.madd(self.madd(self.mm(dCop, self.madd(I_, Dg, 1, -e)), Bop, 1, e1), dBop, 1, e1 * z)
        A2 = [-z * a + (1 - e) * z * b for a, b in zip(self.mv(D2, Qv), self.mv(Bop, Pv))]
        t1 = [x / (2 * z * z) for x in self.mv(Eop, [a + b for a, b in zip(self.mv(Cop, Phv), A2)])]
        t2 = [-x for x in self.mv(dMop, Phv)]
        t3 = self.mv(dEop, Qv)
        DmCB = self.madd(Dg, self.mm(Cop, Bop), 1, -e1)
        t4 = [-x / (2 * z) for x in self.mv(DmCB, Phv)]
        t5 = [e1 * x for x in self.mv(self.madd(self.mm(dCop, Bop), self.mm(Cop, dBop)), Pv)]
        if self.mut == "t5":           # MUTATION: drop the eps(1-eps)(Cop' Bop + Cop Bop') P term
            t5 = [0 * x for x in t5]
        K = [a + b + c + d + f for a, b, c, d, f in zip(t1, t2, t3, t4, t5)]
        dPhi = dict(zip(ODD, self.solve(Mop, K)))
        return dP, dPhi, dC, dvt

    def f_0(self, z, P, Phi, C, vt=None):
        """the eps -> 0 limit of f_eps with vt held fixed: P' and Phi' do not depend on vt (proof/B5_inner.md
        section 2), C' = [C (E0 + vt)]/2 does. At vt = 0 this is the inner system in the state (P, Phi, C):
        P' = Phi/(2z), Phi' = Phi/(2z^2) + G0, C' = [C E0]/2, Q0 = Cop^-1 (Phi + D P),
        G0 = (Cop^2 - I) Phi/(2z^2) + Cop(B0 P - D2 Q0)/(2z) + [([C E0]/2) Q0] - D Phi/(2z), B0 = [[C E0] .]"""
        PP = proj(P, P, EVEN)
        E0 = {k: v / 2 for k, v in PP.items()}
        Cop = self.opC(C)
        B0 = self.opB(C, E0)
        Dg = [[(1 + self.s[a]) if a == b else 0 for b in ODD] for a in ODD]
        D2 = [[(2 + self.s[a]) if a == b else 0 for b in ODD] for a in ODD]
        Pv, Phv = [P[mu] for mu in ODD], [Phi[mu] for mu in ODD]
        Q0 = self.solve(Cop, [a + b for a, b in zip(Phv, self.mv(Dg, Pv))])
        CE = proj(C, E0, EVEN)
        dC0 = {k: CE[k] / 2 for k in EVEN}
        Cvt = proj(C, vt, EVEN) if vt is not None else {k: 0 for k in EVEN}
        dC = {k: dC0[k] + Cvt[k] / 2 for k in EVEN}
        C2 = self.mm(Cop, Cop)
        g1 = [(a - b) / (2 * z * z) for a, b in zip(self.mv(C2, Phv), Phv)]
        g2 = [x / (2 * z) for x in self.mv(Cop, [a - b for a, b in zip(self.mv(B0, Pv), self.mv(D2, Q0))])]
        g3 = self.mv(self.opX(dC0), Q0)
        g4 = [-x / (2 * z) for x in self.mv(Dg, Phv)]
        G0 = [a + b + c + d for a, b, c, d in zip(g1, g2, g3, g4)]
        dPhi = {mu: Phi[mu] / (2 * z * z) + G0[i] for i, mu in enumerate(ODD)}
        dP = {mu: Phi[mu] / (2 * z) for mu in ODD}
        return dP, dPhi, dC, dict(zip(ODD, G0))


def check_phi_rate(npts=5, dps=50):
    """Phi' of f_eps equals the total derivative of Phi_of(z, P, Q, C, Theta) along the model flow (mpmath.diff)"""
    mp.mp.dps = dps
    rnd = random.Random(4242)
    worst = mp.mpf(0)
    for _ in range(npts):
        cr = lambda: mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1))
        z = mp.mpc(rnd.uniform(0.005, 0.05), rnd.uniform(-0.02, 0.02))
        eps = mp.mpf(rnd.uniform(1e-4, 1e-3))
        w = mp.mpf(rnd.uniform(1, 4))
        P = {mu: 2 * cr() for mu in ODD}
        Q = {mu: 3 * cr() for mu in ODD}
        C = {k: (1 if k == 0 else 0) + 0.1 * cr() for k in EVEN}
        Th = {k: cr() for k in EVEN}
        mdl = Model(w, eps, mp, MUT)
        dP, dQ, dC, dTh, Phi = mdl.rates_PQ(z, P, Q, C, Th)
        vt = {k: Th[k] - proj(P, P, EVEN)[k] / 2 for k in EVEN}
        _, dPhi, _, _ = mdl.f_eps(z, P, Phi, C, vt)
        for mu in ODD:
            g = lambda d, mu=mu: mdl.Phi_of(z + d, {m: P[m] + d * dP[m] for m in ODD}, {m: Q[m] + d * dQ[m] for m in ODD},
                                           {k: C[k] + d * dC[k] for k in EVEN}, {k: Th[k] + d * dTh[k] for k in EVEN})[mu]
            num = mp.diff(g, 0)
            worst = max(worst, abs(num - dPhi[mu]) / (1 + abs(dPhi[mu])))
    return worst


def check_limit(npts=5, dps=80):
    """f_eps -> f_0 as eps -> 0 at fixed state (P, Phi, C, vt), vt arbitrary (so f_0 does not depend on vt): the
    difference at eps = 1e-30 must be O(eps/|z|^2)"""
    mp.mp.dps = dps
    rnd = random.Random(99)
    worst = mp.mpf(0)
    for _ in range(npts):
        cr = lambda: mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1))
        z = mp.mpc(rnd.uniform(0.005, 0.05), rnd.uniform(-0.02, 0.02))
        w = mp.mpf(rnd.uniform(1, 4))
        P = {mu: 2 * cr() for mu in ODD}
        Phi = {mu: cr() * z for mu in ODD}
        C = {k: (1 if k == 0 else 0) + 0.1 * cr() for k in EVEN}
        vt = {k: cr() for k in EVEN}
        eps = mp.mpf("1e-30")
        a = Model(w, eps, mp, MUT).f_eps(z, P, Phi, C, vt)
        b = Model(w, 0, mp, MUT).f_0(z, P, Phi, C, vt)
        for i in range(3):
            for k in a[i]:
                worst = max(worst, abs(a[i][k] - b[i][k]) * abs(z) ** 2 / eps)
    return worst


# ----------------------------------------------------------------------------------------------------------------------
# symbolic outer equations in inner variables, linear in the derivatives
# ----------------------------------------------------------------------------------------------------------------------
def outer_equations_symbolic():
    """the four outer equations (per harmonic) of the M-model, written in the inner variables (z, P, Q, C, Theta) and
    their z-derivatives; returns sympy expressions and the symbol lists"""
    xi, nu, w = sp.symbols("xi nu omega", positive=True)
    zs = sp.Symbol("z")
    Ps = {mu: sp.Symbol(f"P{mu}") for mu in ODD}
    Qs = {mu: sp.Symbol(f"Q{mu}") for mu in ODD}
    Cs = {nu_: sp.Symbol(f"C{nu_}") for nu_ in EVEN}
    Ts = {nu_: sp.Symbol(f"T{nu_}") for nu_ in EVEN}
    dPs = {mu: sp.Symbol(f"dP{mu}") for mu in ODD}
    dQs = {mu: sp.Symbol(f"dQ{mu}") for mu in ODD}
    dCs = {nu_: sp.Symbol(f"dC{nu_}") for nu_ in EVEN}
    dTs = {nu_: sp.Symbol(f"dT{nu_}") for nu_ in EVEN}
    # fields as functions of xi through z = xi^2: f(xi) with f' = 2 xi f_z
    n = nu ** -2
    x = xi / nu
    # values and xi-derivatives
    y = {mu: xi * Ps[mu] for mu in ODD}
    y_xi = {mu: Ps[mu] + xi * 2 * xi * dPs[mu] for mu in ODD}
    sg = {mu: xi ** 2 * Qs[mu] for mu in ODD}
    sg_xi = {mu: 2 * xi * Qs[mu] + xi ** 2 * 2 * xi * dQs[mu] for mu in ODD}
    eta = {k: xi ** 2 * Ts[k] for k in EVEN}
    eta_xi = {k: 2 * xi * Ts[k] + xi ** 2 * 2 * xi * dTs[k] for k in EVEN}
    c = dict(Cs)
    c_xi = {k: 2 * xi * dCs[k] for k in EVEN}
    # outer fields and x-derivatives (d/dx = nu d/dxi)
    p = {mu: (y[mu] + nu * sg[mu]) / 2 for mu in ODD}
    m = {mu: (nu * sg[mu] - y[mu]) / 2 for mu in ODD}
    p_x = {mu: nu * (y_xi[mu] + nu * sg_xi[mu]) / 2 for mu in ODD}
    m_x = {mu: nu * (nu * sg_xi[mu] - y_xi[mu]) / 2 for mu in ODD}
    bm1 = {k: nu ** 2 * eta[k] for k in EVEN}
    b = {k: (1 if k == 0 else 0) + bm1[k] for k in EVEN}
    b_x = {k: nu * nu ** 2 * eta_xi[k] for k in EVEN}
    c_x = {k: nu * c_xi[k] for k in EVEN}
    G = {k: (n - 1) * v for k, v in proj(c, bm1, EVEN).items()}
    pm2 = {k: proj(p, p, EVEN)[k] + proj(m, m, EVEN)[k] for k in EVEN}
    th = lambda f: {mu: sp.I * mu * v for mu, v in f.items()}
    E_c = {k: x * c_x[k] - (n - 1) * proj(c, bm1, EVEN)[k] for k in EVEN}
    E_b = {k: x * b_x[k] - proj(b, pm2, EVEN)[k] + (n - 1) * proj(b, bm1, EVEN)[k] for k in EVEN}
    pp_ = {mu: p[mu] + m[mu] for mu in ODD}
    cpm = proj(c, pp_, ODD)
    E_p = {mu: x * x * p_x[mu] - x * proj(c, p_x, ODD)[mu] + w * x * th(p)[mu] - proj(G, p, ODD)[mu]
           + proj(c, p, ODD)[mu] - n / 2 * cpm[mu] for mu in ODD}
    E_m = {mu: x * x * m_x[mu] + x * proj(c, m_x, ODD)[mu] + w * x * th(m)[mu] + proj(G, m, ODD)[mu]
           - proj(c, m, ODD)[mu] + n / 2 * cpm[mu] for mu in ODD}
    syms = dict(xi=xi, nu=nu, w=w, z=zs, P=Ps, Q=Qs, C=Cs, T=Ts, dP=dPs, dQ=dQs, dC=dCs, dT=dTs)
    eqs = {("c", k): E_c[k] for k in EVEN}
    eqs.update({("b", k): E_b[k] for k in EVEN})
    eqs.update({("p", mu): E_p[mu] for mu in ODD})
    eqs.update({("m", mu): E_m[mu] for mu in ODD})
    return eqs, syms


def check_model(npts=6, dps=60):
    """random complex points: the (P, Q, C, Theta) form makes every outer equation vanish"""
    mp.mp.dps = dps
    eqs, S = outer_equations_symbolic()
    args = [S["xi"], S["nu"], S["w"]] + [S[k][i] for k in ("P", "Q", "dP", "dQ") for i in ODD] + \
           [S[k][i] for k in ("C", "T", "dC", "dT") for i in EVEN]
    fns = {k: sp.lambdify(args, v, "mpmath") for k, v in eqs.items()}
    rnd = random.Random(12345)
    worst = mp.mpf(0)
    for _ in range(npts):
        cr = lambda: mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1))
        xi = mp.mpc(rnd.uniform(0.05, 0.3), rnd.uniform(-0.1, 0.1))
        nuv = mp.mpf(rnd.uniform(0.05, 0.3))
        eps = nuv ** 2
        w = mp.mpf(rnd.uniform(1, 4))
        z = xi * xi
        P = {mu: 2 * cr() for mu in ODD}
        Q = {mu: 3 * cr() for mu in ODD}
        C = {k: (1 if k == 0 else 0) + 0.2 * cr() for k in EVEN}
        Th = {k: cr() for k in EVEN}
        mdl = Model(w, eps, mp, MUT)
        dP, dQ, dC, dTh, Phi = mdl.rates_PQ(z, P, Q, C, Th)
        vals = [xi, nuv, w] + [P[i] for i in ODD] + [Q[i] for i in ODD] + [dP[i] for i in ODD] + \
               [dQ[i] for i in ODD] + [C[i] for i in EVEN] + [Th[i] for i in EVEN] + [dC[i] for i in EVEN] + \
               [dTh[i] for i in EVEN]
        for k, f in fns.items():
            r = abs(f(*vals))
            # scale: the equations carry factors up to n^1 * |terms| ~ 1/nu^2
            worst = max(worst, r * nuv ** 2)
    return worst


def check_inner_limit(npts=6, dps=40):
    """eps = 0: Phi = Cop Q - D P equals fixed_point's W = V + [(C-1)Q]; Q' from the model equals the fixed_point V'
    + (1+s) P' with Theta = [P P]/2"""
    mp.mp.dps = dps
    rnd = random.Random(777)
    worst = mp.mpf(0)
    for _ in range(npts):
        cr = lambda: mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1))
        z = mp.mpc(rnd.uniform(0.01, 0.1), rnd.uniform(-0.05, 0.05))
        w = mp.mpf(rnd.uniform(1, 4))
        P = {mu: 2 * cr() for mu in ODD}
        Q = {mu: 3 * cr() for mu in ODD}
        C = {k: (1 if k == 0 else 0) + 0.2 * cr() for k in EVEN}
        Th = {k: v / 2 for k, v in proj(P, P, EVEN).items()}
        mdl = Model(w, 0, mp, MUT)
        Phi = mdl.Phi_of(z, P, Q, C, Th)
        Cop, Bop, Dg, D2, Mop, Eop = mdl.parts(z, P, C, Th)
        Pv, Qv, Phv = ([X[mu] for mu in ODD] for X in (P, Q, Phi))
        A2 = [-z * a + z * b for a, b in zip(mdl.mv(D2, Qv), mdl.mv(Bop, Pv))]
        dQ = [(a + b) / (2 * z * z) for a, b in zip(mdl.mv(Cop, Phv), A2)]
        dP = [p / (2 * z) for p in Phv]
        # fixed_point.py's system
        s = {mu: 1j * mu * w for mu in ODD}
        V = {mu: Q[mu] - (1 + s[mu]) * P[mu] for mu in ODD}
        Cm1 = {k: C[k] - (1 if k == 0 else 0) for k in EVEN}
        CQ = proj(Cm1, Q, ODD)
        W = {mu: V[mu] + CQ[mu] for mu in ODD}
        CW = proj(Cm1, W, ODD)
        E = {k: v / 2 for k, v in proj(P, P, EVEN).items()}
        CE = proj(C, E, EVEN)
        CEP = proj(CE, P, ODD)
        fP = {mu: W[mu] / (2 * z) for mu in ODD}
        fV = {mu: V[mu] / (2 * z * z) + (CQ[mu] + CW[mu]) / (2 * z * z)
              + (-(2 + s[mu]) * Q[mu] - (1 + s[mu]) * W[mu] + CEP[mu]) / (2 * z) for mu in ODD}
        for i, mu in enumerate(ODD):
            worst = max(worst, abs(Phi[mu] - W[mu]), abs(dP[i] - fP[mu]),
                        abs(dQ[i] - (fV[mu] + (1 + s[mu]) * fP[mu])) * abs(z) ** 2)
    return worst


if __name__ == "__main__":
    wm = check_model()
    wi = check_inner_limit()
    wr = check_phi_rate()
    wl = check_limit()
    ok_m = wm < mp.mpf("1e-45")
    ok_i = wi < mp.mpf("1e-30")
    ok_r = wr < mp.mpf("1e-30")
    ok_l = wl < 100
    print(f"M = {M}: model (P, Q, C, Theta) form vs the outer equations in inner variables, worst scaled residual "
          f"{mp.nstr(wm, 3)} ({'PASS' if ok_m else 'FAIL'}, threshold 1e-45)")
    print(f"        eps = 0 limit vs fixed_point.py's inner system, worst difference {mp.nstr(wi, 3)} "
          f"({'PASS' if ok_i else 'FAIL'}, threshold 1e-30)")
    print(f"        Phi' formula of f_eps vs the total derivative of the Phi relation along the flow, worst relative "
          f"difference {mp.nstr(wr, 3)} ({'PASS' if ok_r else 'FAIL'}, threshold 1e-30)")
    print(f"        f_eps - f_0 at eps = 1e-30 with vt arbitrary, worst |difference| |z|^2/eps = {mp.nstr(wl, 3)} "
          f"({'PASS' if ok_l else 'FAIL'}: bounded, threshold 100; an O(1) vt-dependence of the limit would give 1e30)")
    allok = ok_m and ok_i and ok_r and ok_l
    if MUT:
        bad = not (ok_m and ok_r and ok_l)
        print(f"MUTATION {MUT}: some check must FAIL: {'caught' if bad else 'NOT caught'}")
        sys.exit(0 if bad else 3)
    sys.exit(0 if allok else 1)
