#!/usr/bin/env python3
"""A second implementation of c2 in the M = 1 model, for the review of L2_MODE.md section 9 (notes/L2_C2_REVIEW.md, #1010).

It imports nothing from first_order_*.py, l2_c2*.py or inner_np.py. What it shares with them is listed in the review
note, section 6: the M-harmonic model (which products are truncated, and where), the slice formula for c2, the rule
that fixes the association of the forcing's products, and scipy's DOP853.

What is its own:
  - the series recursions of the background and of a forced linear perturbation, written from the inner equations
    in inner_np.py's form of the sigma equation (first_order_jet.py uses the (xi d_xi - 1)(c y) form);
  - the linearised right-hand side, written by hand (no jet arithmetic); d/dlambda is a variational equation;
  - the forcing: its polynomials in the t-derivatives of Phi and c come from l2_c2_review_sym.py's own reduction
    of F2 = 2 s Y2 Q, not from data/l2_c2_forcing.json. Along the integration its Euler derivatives are Taylor
    coefficients in eps of the fields at xi e^eps (a Taylor recursion of the model's ODEs at each point), and in the
    series they are the series' own derivatives. l2_c2.py expands them symbolically by the Leibniz rule into a graph;
  - the path: the ray arg xi^2 = pi/2 + pi/5 (l2_c2.py: pi/2 + pi/8);
  - the linear algebra: raw harmonic-1 jumps and a 2 x 2 real solve in profile space. No de-mixing is needed for the
    slice values c2(phi). The de-mixing map is applied only to reproduce the "left singular vector" value.

    python3 l2_c2_review_num.py checks [--mutate=bg|lin|dlam|ser|fc]
                                              planted checks of this code (the zero, finite differences, exact modes,
                                              lambda*, the forcings two ways and in two forms); exit 0 iff all pass.
                                              Mutations, each in the right-hand side or the series only: bg (the
                                              background's gravity term off by 1e-6), lin (a sign in the linearised
                                              y equation), dlam (a term of the d/dlambda equation dropped), ser (the
                                              series' Euler derivative off by 1e-6), fc (the sign of F_c in the
                                              t-derivative forms). Every check fails under at least one
    python3 l2_c2_review_num.py diag          the control forcing's two evaluations by radius (a diagnostic for 9.3)
    python3 l2_c2_review_num.py c2 [--M=1] [--r1=0.2] [--delta=0.6283] [--own-zero] [--mutate=fc] [--tol=1e-6]
                                              c2, lambda_1 and G on the slices phi = e1, e2, f2 (and the left singular
                                              vector at M = 1), compared with data/l2_c2_M{M}_xi0.2_l2-l0-G-sym.json;
                                              exit 0 iff they agree to --tol. M = 3 takes the one-shot solve of the
                                              raw 4 x 4 jump matrix, rows and columns scaled
"""
import json
import os
import sys
import time

import numpy as np
import mpmath as mp
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.signal import convolve2d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OPTS = {a.split('=')[0]: (a.split('=', 1)[1] if '=' in a else True) for a in sys.argv[1:] if a.startswith('--')}
MUT = OPTS.get('--mutate', '')
W_INF = float(2 * mp.pi / mp.mpf('1.64389375'))          # omega_infinity = 2 pi / Delta_infinity (README)
T0 = time.time()


def log(*a):
    print(*a, '[%.0f s]' % (time.time() - T0), flush=True)


class Alg:
    """harmonics -L..L, L = M + 2; odd fields on |mu| <= M, even fields on |mu| <= M + 1; every product projected"""

    def __init__(self, M, L=None, full=False):
        self.M, self.L = M, (M + 2 if L is None else L)
        k = np.arange(-self.L, self.L + 1)
        self.k = k
        lo, le = (self.L, self.L) if full else (M, M + 1)      # full: no projection below L (an untruncated test)
        self.mask = {1: ((np.abs(k) <= lo) & (k % 2 == 1)).astype(float), 0: ((np.abs(k) <= le) & (k % 2 == 0)).astype(float)}
        self.imu = 1j * k
        self.N = 2 * self.L + 1

    def cv(self, a, b, par):
        L = self.L
        return np.convolve(a, b)[L:3 * L + 1] * self.mask[par]

    def cv2(self, a, b, par):
        """Cauchy product of two coefficient lists (orders x harmonics), truncated at the shorter length"""
        K, L = min(a.shape[0], b.shape[0]), self.L
        return convolve2d(a[:K], b[:K])[:K, L:3 * L + 1] * self.mask[par][None, :]


# ---------------------------------------------------------------------------------------------------------------
# series (scaled at xi0: D_j = d_j q^j, y = xi sum d_j xi^2j; S_j: sigma = xi^2 sum s_j xi^2j; G_j: c = sum g_j xi^2j)
# ---------------------------------------------------------------------------------------------------------------
def bg_series(al, w, F, q, J):
    o, e = 1, 0
    D, S, G, E, CE = [2 * F * al.mask[1]], [], [np.where(al.k == 0, 1.0 + 0j, 0)], [np.zeros(al.N, complex)], [np.zeros(al.N, complex)]
    for j in range(J):
        sj = (2 * j + 1 + al.imu * w) * D[j]
        for i in range(1, j + 1):
            sj = sj - al.cv(G[i], S[j - i], o)
        S.append(sj * al.mask[1])
        m = j + 1
        em = sum(al.cv(D[i], D[m - 1 - i], e) for i in range(m)) * q / 2
        E.append(em)
        cm = sum(al.cv(G[i], E[m - i], e) for i in range(m))
        CE.append(cm)
        G.append(cm / (2 * m))
        num = q * (2 * j + 2 + al.imu * w) * S[j]
        for i in range(1, j + 1):
            num = num - 2 * (j + 1 - i) * al.cv(G[i], D[j + 1 - i], o)
        for mm in range(1, j + 2):
            num = num - al.cv(CE[mm], D[j + 1 - mm], o)
        D.append(num * al.mask[1] / (2 * j + 2))
    return dict(D=D, S=S, G=G, E=E, CE=CE)


def pert_series(al, w, lam, bg, q, J, d0, g0, FY=None, FC=None, parent=None):
    """series of a linear perturbation e^{lam tau}: data d0 (xi coefficient of dy) and g0 (dc at 0); forcing series
    FY[m], FC[m] (coefficients of xi^2m, scaled by q^m) in the y and c equations; parent: this is d/dlam of it"""
    o, e = 1, 0
    Lm = al.imu * w + lam
    D, S, G, E, CE = bg['D'], bg['S'], bg['G'], bg['E'], bg['CE']
    z = np.zeros(al.N, complex)
    dD, dS, dG, dE, dCE = [d0 * al.mask[1]], [], [g0 * al.mask[0]], [z.copy()], [z.copy()]
    for j in range(J):
        sj = (2 * j + 1 + Lm) * dD[j]
        if parent is not None:
            sj = sj + parent['D'][j]
        for i in range(1, j + 1):
            sj = sj - al.cv(G[i], dS[j - i], o)
        for i in range(0, j + 1):
            sj = sj - al.cv(dG[i], S[j - i], o)
        if FY is not None:
            sj = sj - FY[j + 1] / q
        dS.append(sj * al.mask[1])
        m = j + 1
        em = sum(al.cv(D[i], dD[m - 1 - i], e) for i in range(m)) * q
        dE.append(em)
        cm = sum(al.cv(dG[i], E[m - i], e) + al.cv(G[i], dE[m - i], e) for i in range(m))
        if FC is not None:
            cm = cm + FC[m]
        dCE.append(cm)
        dG.append(cm / (2 * m))
        num = q * (2 * j + 2 + Lm) * dS[j]
        if parent is not None:
            num = num + q * parent['S'][j]
        num = num - (2 * j + 2) * al.cv(dG[0], D[j + 1], o)
        for i in range(1, j + 1):
            num = num - 2 * (j + 1 - i) * (al.cv(G[i], dD[j + 1 - i], o) + al.cv(dG[i], D[j + 1 - i], o))
        for mm in range(1, j + 2):
            num = num - al.cv(dCE[mm], D[j + 1 - mm], o) - al.cv(CE[mm], dD[j + 1 - mm], o)
        dD.append(num * al.mask[1] / (2 * j + 2))
    return dict(D=dD, S=dS, G=dG)


def optimal(bg):
    mags = [np.abs(d).max() for d in bg['D']]
    jt = int(np.argmin(mags[1:])) + 1
    return jt, mags[jt] / mags[0]


def summed(ser, jt, xi0):
    return xi0 * sum(ser['D'][:jt]), xi0 * xi0 * sum(ser['S'][:jt]), sum(ser['G'][:jt])


# ---------------------------------------------------------------------------------------------------------------
# the forcing: one builder over a small ring interface, with two rings
# ---------------------------------------------------------------------------------------------------------------
class SerRing:
    """power series in xi^2, scaled by q (coefficient lists of length K)"""

    def __init__(self, al, w, bg, q, K):
        self.al, self.w, self.q, self.K = al, w, q, K
        self.Y = (np.array(bg['D'][:K]), 1)
        self.S = (np.array(bg['S'][:K]), 1)
        self.C = (np.array(bg['G'][:K]), 0)
        self.dropped = 0.0

    def lin(self, *terms):
        return (sum(c * a[0] for c, a in terms), terms[0][1][1])

    def mul(self, a, b):
        par = a[1] ^ b[1]
        return (self.al.cv2(a[0], b[0], par), par)

    def dth(self, a):
        return (self.al.imu[None, :] * a[0], a[1])

    def E(self, a):
        sc = 1 + 1e-6 if MUT == 'ser' else 1.0          # mutation: the series' Euler derivative off by 1e-6
        return (sc * 2 * np.arange(a[0].shape[0])[:, None] * a[0], a[1])

    def xi2(self, a, k):
        x = np.zeros_like(a[0])
        K = x.shape[0]
        if k >= 0:
            x[k:] = a[0][:K - k] * self.q ** k
        else:
            m = -k
            x[:K - m] = a[0][m:] * self.q ** k
            self.dropped = max(self.dropped, np.abs(a[0][:m]).max() / (np.abs(a[0][:m + 4]).max() or 1.0))
        return (x, a[1])

    def inv(self, c):
        return (_series_inverse(self.al, c[0]), 0)


class JetRing:
    """Taylor coefficients in eps of the fields at xi e^eps (lists of length N + 1), from the model's ODEs:
    E Yh = [C Sh] - (1 + w d_theta) Yh,  E C = [C Ec], Ec = xi^2 [Yh Yh]/2,  E Sh = E[C Yh]/xi^2 - (2 + w d_theta) Sh"""

    def __init__(self, al, w, xi, Yh, Sh, C, N=8):
        self.al, self.w, self.xi, self.N = al, w, xi, N
        Yj = np.zeros((N + 1, al.N), complex)
        Sj = np.zeros_like(Yj)
        Cj = np.zeros_like(Yj)
        Yj[0], Sj[0], Cj[0] = Yh, Sh, C
        fact = np.cumprod(np.concatenate([[1.0], np.arange(1, N + 1)]))
        ep = 2.0 ** np.arange(N + 1) / fact                       # e^{2 eps}
        em = (-2.0) ** np.arange(N + 1) / fact                    # e^{-2 eps}
        x2 = xi * xi
        for k in range(N):
            # order k of the right-hand sides, from the orders <= k (and k + 1 of C and Yh for Sh)
            cs = sum(al.cv(Cj[i], Sj[k - i], 1) for i in range(k + 1))
            Yj[k + 1] = (cs - Yj[k] - w * al.imu * Yj[k]) / (k + 1)
            yy = [sum(al.cv(Yj[i], Yj[m - i], 0) for i in range(m + 1)) for m in range(k + 1)]
            ec = [0.5 * x2 * sum(ep[m - i] * yy[i] for i in range(m + 1)) for m in range(k + 1)]
            cec = sum(al.cv(Cj[i], ec[k - i], 0) for i in range(k + 1))
            Cj[k + 1] = cec / (k + 1)
            cy = [sum(al.cv(Cj[i], Yj[m - i], 1) for i in range(m + 1)) for m in range(k + 2)]
            ecy = [(m + 1) * cy[m + 1] for m in range(k + 1)]
            t = sum(em[k - i] * ecy[i] for i in range(k + 1)) / x2
            Sj[k + 1] = (t - 2 * Sj[k] - w * al.imu * Sj[k]) / (k + 1)
        self.Y, self.S, self.C = (Yj, 1), (Sj, 1), (Cj, 0)
        self.fact = fact

    def lin(self, *terms):
        return (sum(c * a[0] for c, a in terms), terms[0][1][1])

    def mul(self, a, b):
        par = a[1] ^ b[1]
        return (self.al.cv2(a[0], b[0], par), par)

    def dth(self, a):
        return (self.al.imu[None, :] * a[0], a[1])

    def E(self, a):
        x = np.zeros_like(a[0])
        x[:-1] = np.arange(1, a[0].shape[0])[:, None] * a[0][1:]
        return (x, a[1])

    def xi2(self, a, k):
        n = a[0].shape[0]
        ser = (2.0 * k) ** np.arange(n) / self.fact[:n]
        x = np.zeros_like(a[0])
        for m in range(n):
            x[m] = sum(ser[m - i] * a[0][i] for i in range(m + 1))
        return (x * self.xi ** (2 * k), a[1])

    def inv(self, c):
        return (_series_inverse(self.al, c[0]), 0)


def _series_inverse(al, c):
    """z with [c z] = 1 on the even harmonics, order by order"""
    ev = al.mask[0] > 0
    Mx = np.zeros((al.N, al.N), complex)
    for j in range(al.N):
        ej = np.zeros(al.N, complex)
        ej[j] = 1
        Mx[:, j] = al.cv(c[0], ej, 0)
    Mx = Mx[np.ix_(ev, ev)]
    z = np.zeros_like(c)
    for m in range(c.shape[0]):
        r = np.zeros(al.N, complex)
        if m == 0:
            r[al.L] = 1
        for i in range(1, m + 1):
            r = r - al.cv(c[i], z[m - i], 0)
        z[m, ev] = np.linalg.solve(Mx, r[ev])
    return z


NPD = 8
Psym = sp.symbols('P1:%d' % (NPD + 1))
Csym = sp.symbols('C0:%d' % NPD)


def review_forcings():
    """F/(s Y2) of the l = 2 mode, FIRST_ORDER.md's l = 0 forcing and G = L0[(0, Phi_s)] in the rows (W, E^t_t), as
    rational functions of P_k = d_t^k Phi and C_k = d_t^k c, from l2_c2_review_sym.py's own derivation"""
    argv, sys.argv = sys.argv, [sys.argv[0]]
    import l2_c2_review_sym as RS
    sys.argv = argv
    c, P, s, t = RS.cF, RS.PhF, RS.s, RS.t
    cs, Ps, Pt = sp.diff(c, s), sp.diff(P, s), sp.diff(P, t)
    QW = sp.diff(P, s, 2) - Ps * Pt**2 / c**2
    QE = -(sp.diff(c, s, 2) - cs**2 / c - sp.diff(Ps * Pt / c, t))
    L1 = RS.L0_rows(-cs, sp.Integer(0))
    LG = RS.L0_rows(sp.Integer(0), Ps)
    raw = {'l2': (2 * QW, 2 * QE / c),
           'l0': (-2 * QW + L1[0], (-2 * QE + L1[1]) / c),
           'G': (LG[0], LG[1] / c)}
    out = {}
    for name, (fw, fe) in raw.items():
        rows = {}
        for key, ex in (('W', fw), ('E^t_t', fe)):
            ex = RS.reduce_s(ex)
            rep = {}
            for a in ex.atoms(sp.Derivative):
                kt = sum(1 for v in a.variables if v == t)
                assert all(v == t for v in a.variables)
                rep[a] = Psym[kt - 1] if a.expr == P else Csym[kt]
            ex = ex.xreplace(rep).xreplace({c: Csym[0]})
            assert not ex.has(P) and not ex.has(s)
            rows[key] = sp.cancel(sp.together(ex))
        out[name] = rows
    return out


Ssym = sp.Symbol('s', positive=True)


def monomials(expr):
    """[(coefficient, exponents of P1.., exponents of C0.., exponent of s)] of a rational function with a monomial
    denominator. An explicit s (only the control forcing of l2_c2_forcing.py has one) is T^2 xi^2/2"""
    num, den = sp.fraction(sp.together(expr))
    gens = list(Psym) + list(Csym) + [Ssym]
    dp = sp.Poly(den, *gens)
    assert len(dp.terms()) == 1, 'the denominator is not a monomial'
    dmon, dco = dp.terms()[0]
    out = []
    for mon, co in sp.Poly(sp.expand(num), *gens).terms():
        e = [a - b for a, b in zip(mon, dmon)]
        assert e[2 * NPD] >= 0
        out.append((complex(co / dco), e[:NPD], e[NPD:2 * NPD], e[2 * NPD]))
    return out


def build_converted(R, rows):
    """F_y = sqrt2 xi^2 [C Ft_W], F_c = xi^2 [C Ft_E] in ring R. Phi_t = [C Yh]/sqrt2 with T-weight 1, and
    d_t (p, q) = (p + 1, p q + E q + w d_theta q) at T = 1. Each monomial is a left fold of truncated products in the
    order Phi_t.., Phi_tt.., c_t.., then powers of c or of its truncated inverse: the association of l2_c2.py."""
    w = R.w
    phi = [None, (1, R.lin((1 / np.sqrt(2), R.mul(R.C, R.Y))))]
    cq = [(0, R.C)]

    def dt(pq):
        p, qq = pq
        return (p + 1, R.lin((p, qq), (1, R.E(qq)), (w, R.dth(qq))))
    invC = R.inv(R.C)
    res = []
    for key in ('W', 'E^t_t'):
        acc = None
        for co, eP, eC, es in monomials(rows[key]):
            facs, tp = [], -2 * es
            co = co * 0.5 ** es
            for k in range(NPD):
                if eP[k]:
                    while len(phi) <= k + 1:
                        phi.append(dt(phi[-1]))
                    facs += [phi[k + 1]] * eP[k]
            for k in range(1, NPD):
                assert eC[k] >= 0
                if eC[k]:
                    while len(cq) <= k:
                        cq.append(dt(cq[-1]))
                    facs += [cq[k]] * eC[k]
            facs += [(0, R.C)] * max(eC[0], 0) + [(0, invC)] * max(-eC[0], 0)
            nd = None
            for pp, qq in facs:
                tp += pp
                nd = qq if nd is None else R.mul(nd, qq)
            assert tp == 4, 'T-weight %d, not 4' % tp
            if es:
                nd = R.xi2(nd, es)
            acc = (co * nd[0], nd[1]) if acc is None else (acc[0] + co * nd[0], nd[1])
        res.append(acc)
    fy = R.xi2(R.mul(R.C, res[0]), 1)
    fc = R.xi2(R.mul(R.C, res[1]), 1)
    sg = -1.0 if MUT == 'fc' else 1.0
    return np.sqrt(2) * fy[0], sg * fc[0]


def build_compact(R, which):
    """The same forcings in the heat variables, with s-derivatives instead of t-derivatives (review note, finding 2):
    sym: FIRST_ORDER.md section 4's terms; Gc: L0[(0, Phi_s)]; l2c = -sym + Gc."""
    Y, S, C = R.Y, R.S, R.C
    YY = R.mul(Y, Y)
    EY, ES = R.E(Y), R.E(S)
    sym_y = R.lin((-2, R.mul(C, ES)), (-1, R.xi2(R.mul(C, R.mul(YY, S)), 1)))
    sym_c = R.lin((-1, R.xi2(R.mul(C, R.mul(S, S)), 1)), (2, R.mul(C, R.mul(Y, EY))))
    g_y = R.lin((-2, R.xi2(R.mul(C, R.mul(YY, S)), 1)))
    g_c = R.lin((1, R.mul(C, R.mul(Y, EY))), (0.5, R.xi2(R.mul(C, R.mul(YY, YY)), 1)))
    if which == 'sym':
        return sym_y[0], sym_c[0]
    if which == 'Gc':
        return g_y[0], g_c[0]
    if which == 'l2c':
        return g_y[0] - sym_y[0], g_c[0] - sym_c[0]
    raise ValueError(which)


# ---------------------------------------------------------------------------------------------------------------
# the lateral integration of the background and of linear perturbations
# ---------------------------------------------------------------------------------------------------------------
class Comp:
    def __init__(self, name, lam=0.0, d0=None, g0=None, force=None, parent=None):
        self.name, self.lam, self.d0, self.g0, self.force, self.parent = name, lam, d0, g0, force, parent


class Lateral:
    def __init__(self, M, w, F, comps, delta=np.pi / 5, margin=12.0, rtol=1e-12):
        self.al = Alg(M)
        self.w, self.F, self.comps, self.delta, self.rtol = w, F, comps, delta, rtol
        self.phi = (np.pi / 2 + delta) / 2
        self.r0 = float(np.sqrt(np.sin(delta) / (2 * (M * w * (np.pi / 2 + delta) + margin))))
        self.oi = self.al.mask[1] > 0
        self.ei = self.al.mask[0] > 0
        self.no, self.ne = int(self.oi.sum()), int(self.ei.sum())
        self.nper = 2 * self.no + self.ne
        self.forces = sorted({c.force for c in comps if c.force})

    def start(self):
        al, w = self.al, self.w
        xi0 = self.r0 * np.exp(1j * self.phi)
        q = xi0 * xi0
        J = int(1.2 / (2 * self.r0 ** 2)) + 40
        bg = bg_series(al, w, self.F, q, J + 12)
        jt, terr = optimal({'D': bg['D'][:J + 1]})
        self.info = dict(r0=self.r0, jt=jt, terr=terr)
        fser = {}
        if self.forces:
            R = SerRing(al, w, bg, q, jt + 8)
            for f in self.forces:
                fser[f] = FORCE_BUILDERS[f](R)
            self.info['dropped'] = R.dropped
        sers = {}
        Y = [self.pack(*summed(bg, jt, xi0))]
        for cp in self.comps:
            FY = FC = None
            if cp.force:
                FY, FC = fser[cp.force]
            z = np.zeros(al.N, complex)
            ser = pert_series(al, w, cp.lam, bg, q, jt + 2, z if cp.d0 is None else cp.d0, z if cp.g0 is None else cp.g0,
                              FY, FC, sers.get(cp.parent))
            sers[cp.name] = ser
            Y.append(self.pack(*summed(ser, jt, xi0)))
        return np.concatenate(Y)

    def pack(self, D, S, C):
        return np.concatenate([D[self.oi], S[self.oi], C[self.ei]])

    def unpack(self, v):
        al = self.al
        D = np.zeros(al.N, complex); S = np.zeros(al.N, complex); C = np.zeros(al.N, complex)
        D[self.oi] = v[:self.no]; S[self.oi] = v[self.no:2 * self.no]; C[self.ei] = v[2 * self.no:]
        return D, S, C

    def rhs(self, xi, Yv):
        al, w = self.al, self.w
        cv = al.cv
        x2 = xi * xi
        n = self.nper
        D, S, C = self.unpack(Yv[:n])
        tr = al.imu * w
        dD = (cv(C, S, 1) - xi * tr * D) / x2
        E = cv(D, D, 0) / 2
        cE = cv(C, E, 0)
        gm = 1 + 1e-6 if MUT == 'bg' else 1.0           # mutation: the background's gravity term off by 1e-6
        dS = (cv(C, xi * dD, 1) + gm * cv(cE, D, 1) - cv(C, D, 1) - xi * tr * S) / x2
        out = [self.pack(dD, dS, cE / xi)]
        fpt = {}
        if self.forces:
            R = JetRing(al, w, xi, D / xi, S / x2, C)
            for f in self.forces:
                fpt[f] = FORCE_BUILDERS[f](R)
        st = {}
        for i, cp in enumerate(self.comps):
            pD, pS, pC = self.unpack(Yv[(i + 1) * n:(i + 2) * n])
            st[cp.name] = (pD, pS)
            Lm = tr + cp.lam
            sl = -1.0 if MUT == 'lin' else 1.0          # mutation: a sign in the linearised y equation
            num = sl * cv(pC, S, 1) + cv(C, pS, 1) - xi * Lm * pD
            pcE = cv(pC, E, 0) + cv(C, cv(D, pD, 0), 0)
            if cp.force:
                fy, fc = fpt[cp.force]
                num = num + fy[0]
                pcE = pcE + fc[0]
            if cp.parent:
                num = num - xi * st[cp.parent][0]
            pdD = num / x2
            r2 = (cv(pC, xi * dD, 1) + cv(C, xi * pdD, 1) + cv(pcE, D, 1) + cv(cE, pD, 1) - cv(pC, D, 1) - cv(C, pD, 1)
                  - xi * Lm * pS)
            if cp.parent and MUT != 'dlam':             # mutation: the parent's sigma dropped from d/dlambda
                r2 = r2 - xi * st[cp.parent][1]
            out.append(self.pack(pdD, r2 / x2, pcE / xi))
        return np.concatenate(out)

    def jumps(self, r1):
        """harmonic +1 of y+ - conj(y+ reflected) at xi = r1, for the background and every component"""
        Y = self.start()
        u = np.exp(1j * self.phi)
        sol = solve_ivp(lambda tt, Z: self.rhs(tt * u, Z) * u, (self.r0, r1), Y, method='DOP853', rtol=self.rtol, atol=1e-16)
        assert sol.success, sol.message
        Y = sol.y[:, -1]
        sol = solve_ivp(lambda tt, Z: self.rhs(r1 * np.exp(1j * tt), Z) * 1j * r1 * np.exp(1j * tt), (self.phi, 0.0), Y,
                        method='DOP853', rtol=self.rtol, atol=1e-16)
        assert sol.success, sol.message
        Y = sol.y[:, -1]
        n = self.nper
        out = {}
        for i, name in enumerate(['bg'] + [c.name for c in self.comps]):
            D, S, C = self.unpack(Y[i * n:(i + 1) * n])
            out[name] = D - np.conj(D[::-1])
        return out


FORCE_BUILDERS = {}


def Fprof(al, a, b):
    """profile a sin(theta) + b cos(theta) as harmonics"""
    F = np.zeros(al.N, complex)
    F[al.L + 1] = a / 2j + b / 2
    F[al.L - 1] = -a / 2j + b / 2
    return F


def ri(z):
    return np.array([z.real, z.imag])


def kprime(sv):
    a, b = (1 + sv) / 2, (2 + sv) / 2
    return complex(mp.power(2, -sv - mp.mpf(1) / 2) / (mp.gamma(a) * mp.gamma(b)))


def demix1(x, r1, NTH=64):
    """the de-mixing and normalisation of the M = 1 jump (the formula of inner_np.residual_demixed), a real 2 x 2 map
    on (Re J_1, Im J_1). Shared with the original; used only for the left-singular-vector value."""
    A, w = x[0], W_INF * x[1]
    th = 2 * np.pi * np.arange(NTH) / NTH
    Pi = -(A * A / (4 * w)) * np.sin(2 * th)                  # (1/w) int (F^2 - <F^2>), F = A sin(theta)
    fac = r1 ** (2j * w) * np.exp(1 / (2 * r1 * r1) - A * A * np.log(r1)) / (2 * np.pi * abs(kprime(1j * w)))

    def f(J1):
        Jth = J1 * np.exp(1j * th) - np.conj(J1) * np.exp(-1j * th)
        return ri(np.fft.fft(Jth * np.exp(-Pi))[1] / NTH * fac)
    return np.column_stack([f(1.0), f(1j)])


def bg_jump(x, r1, delta, rtol=1e-12):
    al = Alg(1)
    lat = Lateral(1, W_INF * x[1], Fprof(al, x[0], 0.0), [], delta=delta, rtol=rtol)
    return lat.jumps(r1)['bg'][al.L + 1]


def own_zero(x, r1, delta):
    """Newton on the raw harmonic-1 jump of the M = 1 background, Jacobian by centred differences"""
    x = np.array(x, float)
    for it in range(8):
        r = ri(bg_jump(x, r1, delta))
        Jm = np.zeros((2, 2))
        for p, h in ((0, 1e-5), (1, 1e-6)):
            xp, xm = x.copy(), x.copy()
            xp[p] += h; xm[p] -= h
            Jm[:, p] = (ri(bg_jump(xp, r1, delta)) - ri(bg_jump(xm, r1, delta))) / (2 * h)
        dx = -np.linalg.solve(Jm, r)
        log('   zero it %d: |J| %.2e  step %.2e  x %s' % (it, np.linalg.norm(r), np.abs(dx).max(), x))
        x = x + dx
        if np.abs(dx).max() < 3e-11:
            break
    return x


def Fbg(al, x):
    """F = A sin(theta) + sum_nu (B_nu sin(nu theta) + C_nu cos(nu theta)), x = (A, w/w_inf, B3, C3, ...)"""
    F = Fprof(al, x[0], 0.0)
    for i, nu in enumerate(range(3, al.M + 1, 2)):
        B, C = x[2 + 2 * i], x[3 + 2 * i]
        F[al.L + nu] = B / 2j + C / 2
        F[al.L - nu] = -B / 2j + C / 2
    return F


def basis_prof(al, i):
    """profile direction i: sin(nu theta) for even i, cos(nu theta) for odd i, nu = 1, 3, ..."""
    nu = 1 + 2 * (i // 2)
    F = np.zeros(al.N, complex)
    if i % 2 == 0:
        F[al.L + nu], F[al.L - nu] = 1 / 2j, -1 / 2j
    else:
        F[al.L + nu], F[al.L - nu] = 0.5, 0.5
    return F


def jvec(al, Jd):
    """raw jump on the harmonics nu = 1, 3, ..., M as a real vector (Re, Im pairs)"""
    return np.concatenate([ri(Jd[al.L + nu]) for nu in range(1, al.M + 1, 2)])


def hom_matrix(x, lam, r1, delta, extra=(), M=1, rtol=1e-12):
    """A(lam): columns = raw jumps of the profile directions with dc(0) = 0; and the jumps of `extra`"""
    al = Alg(M)
    nb = M + 1
    comps = [Comp('h%d' % i, lam, 2 * basis_prof(al, i)) for i in range(nb)] + list(extra)
    lat = Lateral(M, W_INF * x[1], Fbg(al, x), comps, delta=delta, rtol=rtol)
    J = lat.jumps(r1)
    A = np.column_stack([jvec(al, J['h%d' % i]) for i in range(nb)])
    return A, {k: jvec(al, v) for k, v in J.items()}, lat


def eq_solve(A, b):
    """A^-1 b with rows and columns scaled to unit norm first"""
    rs = 1 / np.linalg.norm(A, axis=1)
    As = A * rs[:, None]
    cs = 1 / np.linalg.norm(As, axis=0)
    return np.linalg.solve(As * cs[None, :], b * rs) * cs


def stored_x(M=1):
    return np.array(json.load(open(os.path.join(HERE, 'data', 'first_order_hier_M%d_xi0.2.json' % M)))['x'])


def checks():
    ok = True

    def say(label, good, detail):
        nonlocal ok
        ok &= bool(good)
        log(('PASS  ' if good else 'FAIL  ') + label + ':  ' + detail)
    al = Alg(1)
    r1, delta = 0.2, np.pi / 5
    x0 = stored_x()
    # 1. the zero, from the README's five digits
    x = own_zero([2.54215, 0.77699], r1, delta)
    say('1. the M = 1 zero, own Newton on the raw jump, against the stored zero', np.abs(x - x0).max() < 1e-8,
        'x = (%.12f, %.12f), stored (%.12f, %.12f), difference %.1e (< 1e-8)' % (x[0], x[1], x0[0], x0[1], np.abs(x - x0).max()))
    # 2. the linearised code against the nonlinear one: profile sin at lam = 0 is d/dA
    A0, _, _ = hom_matrix(x0, 0.0, r1, delta)
    h = 1e-3                                              # five-point stencil: the jump's noise is 1e-15 absolute
    jb = lambda k: ri(bg_jump(x0 + [k * h, 0], r1, delta))
    fd = (-jb(2) + 8 * jb(1) - 8 * jb(-1) + jb(-2)) / (12 * h)
    rel = np.linalg.norm(A0[:, 0] - fd) / np.linalg.norm(fd)
    say('2. linear perturbation (profile sin, lam = 0) = d(jump)/dA by five-point differences', rel < 1e-6, 'relative %.1e (< 1e-6)' % rel)
    # 3. exact modes: lam = 0 (f = F', a pure cos), lam = 1 (f = F + w F'), control lam = 0.5
    w = W_INF * x0[1]
    sv0 = np.linalg.svd(A0, compute_uv=False)
    n0 = np.linalg.norm(A0 @ np.array([0.0, 1.0])) / np.linalg.norm(A0)
    A1, _, _ = hom_matrix(x0, 1.0, r1, delta)
    f1 = np.array([1.0, w]) / np.hypot(1.0, w)
    n1 = np.linalg.norm(A1 @ f1) / np.linalg.norm(A1)
    Ah, _, _ = hom_matrix(x0, 0.5, r1, delta)
    svh = np.linalg.svd(Ah, compute_uv=False)
    say('3. exact modes: A(0) F\' = 0 and A(1)(F + w F\') = 0; control: A(0.5) is not singular',
        n0 < 1e-6 and n1 < 1e-6 and svh[1] / svh[0] > 1e-3,
        '|A(0) f|/|A| %.1e, |A(1) f|/|A| %.1e (< 1e-6); A(0.5) sv ratio %.1e (> 1e-3); A(0) sv ratio %.1e'
        % (n0, n1, svh[1] / svh[0], sv0[1] / sv0[0]))
    # 4. lambda* of the M = 1 model near 2: secant on det A(lam)
    la, lb = 1.75, 1.78
    fa = np.linalg.det(hom_matrix(x0, la, r1, delta)[0])
    fb = np.linalg.det(hom_matrix(x0, lb, r1, delta)[0])
    for it in range(8):
        lc = lb - fb * (lb - la) / (fb - fa)
        la, fa, lb = lb, fb, lc
        fb = np.linalg.det(hom_matrix(x0, lb, r1, delta)[0])
        if abs(lb - la) < 1e-11:
            break
    say('4. lambda* of the M = 1 model against FIRST_ORDER.md', abs(lb - 1.769122008) < 1e-7,
        'lambda* = %.10f, stored 1.7691220084, difference %.1e (< 1e-7)' % (lb, abs(lb - 1.7691220084)))
    # 5. the variational d/dlambda against centred differences in lambda
    bg = bg_series(al, w, Fprof(al, x0[0], 0.0), 1.0, 3)
    d1, g1 = bg['D'][1], bg['G'][1]
    mode = Comp('m', 2.0, 2 * d1, 2 * g1)
    _, J, _ = hom_matrix(x0, 2.0, r1, delta, [mode, Comp('ml', 2.0, parent='m')])
    hl = 1e-4
    _, Jp, _ = hom_matrix(x0, 2.0 + hl, r1, delta, [Comp('m', 2.0 + hl, 2 * d1, 2 * g1)])
    _, Jm, _ = hom_matrix(x0, 2.0 - hl, r1, delta, [Comp('m', 2.0 - hl, 2 * d1, 2 * g1)])
    fdl = (Jp['m'] - Jm['m']) / (2 * hl)
    rel = np.linalg.norm(J['ml'] - fdl) / np.linalg.norm(fdl)
    say('5. d/dlambda of the mode (variational equation) = centred differences', rel < 1e-6, 'relative %.1e (< 1e-6)' % rel)
    # 6. the forcing: summed series = Taylor-jet value at a point on the ray (two evaluations of one function)
    FORCE_BUILDERS.update(make_builders())
    xi0 = 0.12 * np.exp(1j * (np.pi / 2 + delta) / 2)
    q = xi0 * xi0
    bgs = bg_series(al, w, Fprof(al, x0[0], 0.0), q, 60)
    jt, terr = optimal(bgs)
    Rs = SerRing(al, w, bgs, q, jt + 8)
    D0, S0, C0 = summed(bgs, jt, xi0)
    Rj = JetRing(al, w, xi0, D0 / xi0, S0 / q, C0)
    worst = 0.0
    for name in sorted(FORCE_BUILDERS):
        sy, sc = FORCE_BUILDERS[name](Rs)
        py, pc = FORCE_BUILDERS[name](Rj)
        d = max(np.abs(sy[:jt + 1].sum(0) - py[0]).max() / np.abs(py[0]).max(),
                np.abs(sc[:jt + 1].sum(0) - pc[0]).max() / np.abs(pc[0]).max())
        worst = max(worst, d)
    say('6. forcings: summed series = Taylor-jet value at |xi| = 0.12 (all six)',
        worst < 1e-8, 'max relative %.1e (< 1e-8) (series truncation %.0e)' % (worst, terr))
    # 7. the compact forms against the t-derivative forms, without truncation (L = 40, random fields at a point):
    #    products then associate, so the two forms of each forcing must agree to rounding
    alF = Alg(3, L=40, full=True)
    rng = np.random.default_rng(11)

    def rnd(par, top, amp):
        zz = (rng.normal(size=alF.N) + 1j * rng.normal(size=alF.N)) * amp
        return zz * ((np.abs(alF.k) <= top) & (alF.k % 2 == par)) * np.exp(-0.3 * np.abs(alF.k))
    worst = 0.0
    for trial in range(3):
        Yh, Sh, Cc = rnd(1, 3, 1.0), rnd(1, 3, 1.0), rnd(0, 2, 0.1)
        Cc[alF.L] = 1.0
        Rf = JetRing(alF, w, 0.3 * np.exp(0.4j * (trial + 1)), Yh, Sh, Cc)
        for a, b in (('l2', 'l2c'), ('l0', 'sym'), ('G', 'Gc')):
            ay, ac = FORCE_BUILDERS[a](Rf)
            by, bc = FORCE_BUILDERS[b](Rf)
            worst = max(worst, np.abs(ay[0] - by[0]).max() / np.abs(by[0]).max(), np.abs(ac[0] - bc[0]).max() / np.abs(bc[0]).max())
    say('7. compact forms = t-derivative forms without truncation (l2, l0 against SymForce\'s terms, G)', worst < 1e-9,
        'max relative %.1e (< 1e-9)' % worst)
    return ok


def make_builders():
    rows = review_forcings()
    b = {name: (lambda R, rr=rows[name]: build_converted(R, rr)) for name in rows}
    for name in ('sym', 'Gc', 'l2c'):
        b[name] = (lambda R, nm=name: build_compact(R, nm))
    return b


def c2_run():
    M = int(OPTS.get('--M', 1))
    r1 = float(OPTS.get('--r1', 0.2))
    delta = float(OPTS.get('--delta', np.pi / 5))
    rtol = float(OPTS.get('--rtol', 1e-12))
    al = Alg(M)
    x = stored_x(M)
    if '--own-zero' in OPTS:
        assert M == 1
        x = own_zero([2.54215, 0.77699], r1, delta)
    w = W_INF * x[1]
    FORCE_BUILDERS.update(make_builders())
    names = OPTS.get('--forcing', 'l2,l0,G,sym,l2c,Gc').split(',')
    bg = bg_series(al, w, Fbg(al, x), 1.0, 3)
    d1, g1 = bg['D'][1], bg['G'][1]
    f2 = np.concatenate([[-2 * d1[al.L + nu].imag, 2 * d1[al.L + nu].real] for nu in range(1, M + 1, 2)])
    comps = [Comp('mode', 2.0, 2 * d1, 2 * g1), Comp('lam', 2.0, parent='mode')]
    comps += [Comp(nm, 2.0, force=nm) for nm in names]
    A, J, lat = hom_matrix(x, 2.0, r1, delta, comps, M=M, rtol=rtol)
    log('M %d, x = %s, r1 %.3f, ray arg xi^2 = pi/2 + %.4f, r0 %.4f, series cut at j = %d (%.0e), rtol %.0e'
        % (M, np.array2string(x, precision=13), r1, delta, lat.info['r0'], lat.info['jt'], lat.info['terr'], rtol))
    p = {k: eq_solve(A, v) for k, v in J.items()}
    log('   mode d_s Z0: profile f2 = %s, dc(0) = 2 g1 = %s' % (np.round(f2, 6), np.round(2 * g1[al.mask[0] > 0].real, 6)))
    log('   jump of the mode over |A| |f2|: %.1e; cond(A) %.1e'
        % (np.linalg.norm(J['mode']) / (np.linalg.norm(A) * np.linalg.norm(f2)), np.linalg.cond(A)))
    z = np.zeros(M + 1)
    e1, e2, ff = z.copy(), z.copy(), z.copy()
    e1[0], e2[1] = 1, 1
    ff[:2] = f2[:2] / np.linalg.norm(f2[:2])
    phis = {'e1': e1, 'e2': e2, 'f2': ff}
    res = {}
    for nm in names:
        res[nm] = {k: float(-(ph @ p[nm]) / (ph @ p['lam'])) for k, ph in phis.items()}
    keys = ['e1', 'e2', 'f2']
    if M == 1:
        T = demix1(x, r1)
        Sch = T @ A
        U, sv, Vt = np.linalg.svd(Sch)
        ll = U[:, -1]
        for nm in names:
            res[nm]['ell'] = float(-(ll @ T @ J[nm]) / (ll @ T @ J['lam']))
        keys.append('ell')
    ref = json.load(open(os.path.join(HERE, 'data', 'l2_c2_M%d_xi0.2_l2-l0-G-sym.json' % M)))
    tol = float(OPTS.get('--tol', 1e-6))
    ok = True
    print()
    print('%-5s %-4s %14s %14s %10s' % ('', 'phi', 'this code', 'l2_c2.py', 'rel diff'))
    for nm in names:
        for k in keys:
            mine = res[nm][k]
            if nm in ref:
                theirs = ref[nm][k]
                rel = abs(mine - theirs) / abs(theirs)
                if k != 'ell' or abs(r1 - 0.2) < 1e-12:      # the left-singular-vector value depends on r1
                    ok &= rel < tol
                print('%-5s %-4s %14.8f %14.8f %10.1e' % (nm, k, mine, theirs, rel))
            else:
                print('%-5s %-4s %14.8f %14s' % (nm, k, mine, '-'))
    print()
    if M == 1:
        dS = np.abs(Sch - np.array(ref['Sch2'])).max() / np.abs(ref['Sch2']).max()
        print('Sch(2), de-mixed: max relative difference from l2_c2.py %.1e' % dS)
    df = np.abs(f2 - np.array(ref['f2'])).max() / np.abs(ref['f2']).max()
    print('f2: max relative difference from l2_c2.py %.1e' % df)
    if all(nm in res for nm in ('l2', 'l0', 'G')):
        print('linearity: c2 + lambda_1 - G over phi: %s' % ', '.join('%.1e' % (res['l2'][k] + res['l0'][k] - res['G'][k]) for k in phis))
    if all(nm in res for nm in ('l2c', 'sym', 'Gc')):
        print('compact:   c2c + lambda_1(sym) - Gc over phi: %s' % ', '.join('%.1e' % (res['l2c'][k] + res['sym'][k] - res['Gc'][k]) for k in phis))
    out = dict(M=M, x=x.tolist(), r1=r1, delta=delta, rtol=rtol, info=lat.info, f2=f2.tolist(), res=res, agree=bool(ok), tol=tol,
               A=A.tolist(), J={k: v.tolist() for k, v in J.items()})
    fn = OPTS.get('--out')
    if fn:
        json.dump(out, open(fn, 'w'), indent=1)
        log('wrote ' + fn)
    return ok


def diag():
    """The control forcing of l2_c2_forcing.py (leading background only, read from data/l2_c2_forcing.json) and the
    l = 2 forcing, each evaluated two ways on the stored M = 3 background at four radii on l2_c2.py's ray: the summed
    series (Euler derivatives are exact, no division by xi^2) and the Taylor-jet value at the point (divisions by
    xi^2). L2_MODE.md 9.3 reports that its own two evaluations of the control differ by 1.7e-10 at |xi| = 0.05 and
    attributes it to rounding in nested divisions by xi^2. Not a check: it prints the differences."""
    M = 3
    al = Alg(M)
    x = stored_x(M)
    w = W_INF * x[1]
    d = json.load(open(os.path.join(HERE, 'data', 'l2_c2_forcing.json')))
    P8, C8 = sp.symbols('P1:9'), sp.symbols('C0:8')
    assert P8[:NPD] == Psym and C8[:NPD] == Csym
    rows = {nm: {k: sp.sympify(v) for k, v in d[nm].items()} for nm in ('l2', 'l2_ctrl')}
    for nm in ('l2', 'l2_ctrl'):
        line = []
        for r in (0.04, 0.05, 0.07, 0.1):
            xi0 = r * np.exp(1j * (np.pi / 2 + np.pi / 8) / 2)
            q = xi0 * xi0
            bgs = bg_series(al, w, Fbg(al, x), q, int(1.2 / (2 * r * r)) + 60)
            jt, terr = optimal(bgs)
            Rs = SerRing(al, w, bgs, q, jt + 8)
            D0, S0, C0 = summed(bgs, jt, xi0)
            Rj = JetRing(al, w, xi0, D0 / xi0, S0 / q, C0)
            sy, sc = build_converted(Rs, rows[nm])
            py, pc = build_converted(Rj, rows[nm])
            dd = max(np.abs(sy[:jt + 1].sum(0) - py[0]).max() / np.abs(py[0]).max(),
                     np.abs(sc[:jt + 1].sum(0) - pc[0]).max() / np.abs(pc[0]).max())
            line.append('|xi| %.2f: %.1e (series truncation %.0e)' % (r, dd, terr))
        log('%s: summed series - point value, relative: %s' % (nm, '; '.join(line)))
    return True


if __name__ == '__main__':
    cmd = next((a for a in sys.argv[1:] if not a.startswith('--')), 'c2')
    good = checks() if cmd == 'checks' else diag() if cmd == 'diag' else c2_run()
    print('PASS' if good else 'FAIL')
    sys.exit(0 if good else 1)
