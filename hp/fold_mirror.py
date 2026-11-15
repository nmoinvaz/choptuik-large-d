#!/usr/bin/env python3
"""The fold's mirror branch: the second sheet of Delta(D) at the square-root point D* = 259.9033, from chain_nsr's own
Taylor series in D (#829), by two approximant families that share only the input series.

    cd hp && python3 fold_mirror.py planted RAWDIR [--json=OUT] [--shift=0.01 --usable=..] [--bases=..] [--points=1]
                                                    [--cont=0.3] [--noise=1e-30]
    cd hp && python3 fold_mirror.py real RAWDIR --planted=PLANTED.json [--json=OUT] [--csv=OUT]
    cd hp && python3 fold_mirror.py literal RAWDIR      the issue's method 1 on the truncated series alone (no model)
    cd hp && python3 fold_mirror.py noise RAWDIR [--json=OUT]   the jets' noise, from the forms' residuals
    cd hp && python3 fold_mirror.py posthoc RAWDIR --planted=P.json --real=R.json [--json=OUT]

RAWDIR is a clone of the raw-data repository: the jets are I603-win2/chain_nsr/jet_pin3_D*.npz (25 orders, eight
parts; Delta is the last entry), at the 46 bases from D 258.7829 to 259.8352 registered on #829 (#818 used the first
42, to 259.8184; chain_nsr has since gone on), and the measured Delta is the last entry of each state anm_pin3_D*.npz
(every state in [259.64, D*) present in RAWDIR).

Near D*, Delta = Delta* + sum_k a_k s^k with s = +sqrt(D* - D) on the chain's sheet and s = -sqrt(D* - D) on the
mirror. Each method builds a two-valued approximant from the series of two bases (h = D - D_b at the anchor D_b): the
base and its partner, the base nearest D_b - PAIR. Its polynomials, in x = (D - D*)/W, make the form vanish to all 25
(M2) or 24 (M1) orders of both series, by least squares with each condition scaled to unit largest entry (`forms`;
with --points=1, the anchor's series alone at the lower degrees DA_DEG1, QP_DEG1).
  M1  differential approximant continued around D*. A first-order inhomogeneous approximant Q1 f' + Q0 f + R = 0
      (Guttmann and Joyce), whose zero h* nearest D* is the branch point, with local exponent
      mu = -Q0(h*)/Q1'(h*). Its solution is continued from the anchor along the real axis to h* - rho, once
      counterclockwise around h* on the circle of radius rho, back to h* - rho on the other sheet, then along the real
      axis to each D. Each step re-expands the solution at a new centre, the approximant supplying every order (a
      Taylor step of order KT, of length at most STEP times the distance to the nearest zero of Q1).
      rho = min(RHO, 0.45 d), d the distance from h* to the next zero of Q1, so the circle holds h* alone. The mirror
      is the real part, and |Im| (zero when mu = 1/2) is reported.
  M2  quadratic Pade (Shafer) approximants P f^2 + Q f + R = 0. Their roots (-Q +- sqrt(Q^2 - 4 P R)) / (2 P): the
      root equal to c_0 at the anchor is the chain's, the other the mirror, followed along the real axis, where
      neither the discriminant nor P may vanish between the anchor and D.
The truncated series alone cannot do method 1 as the issue first worded it: a polynomial re-expanded at any centre is
the same polynomial, single-valued, so a loop around D* returns its starting value (`literal` shows it). The orders
past 24 must come from a model, and the differential approximant is that model in M1.
The configuration was fixed on the plant before the real run (#829): matched exactly at one base, both methods reach
about 5e-6 at 259.64 at the base nearest D*; at two bases, matched exactly (degree sum 46), 1e-9 without noise but
1e-2 at noise 1e-30; by least squares at the degrees DEG2 (sums 31-33), about 1e-7 at noise 1e-30.

Delta* and a1..a4. M2: with t = D*_Q - D at the discriminant's zero, E = -Q/(2P) = Delta* + a2 t + a4 t^2 + ... and
sigma sqrt(disc/t)/(2P) = a1 + a3 t + ..., series arithmetic. M1: Frobenius at h*, the particular solution analytic
there (Delta*, a2, a4) and the homogeneous t^mu (1 + q1 t + ...) times the chain's constant K (a1 = K, a3 = K q1),
K matched at t = rho/2 after the real-axis integration.

A base's value for a method is the median over its approximants that are defined on every path (VALID), its spread the
range over them. A grid point's value is the median over the usable bases, its error the larger of the largest
departure of a usable base's value from it and the largest half range over approximants at a usable base: a spread
over bases and approximants, not a bound. The table's value is the mean of M1 and M2, its error the larger of their
errors and half their difference.

planted (criteria registered on #829 before the real series was read by this script): at each base, the coefficients
of F = R(D) + S(D) sqrt(D* - D) + C(D), with the chain's sizes (Delta* 1.6323, a1 0.035, a2 -0.010, a3 0.020, a4
0.010, an entire term 1e-3 Re e^(kappa (D - 259.9033)), kappa 1.019 + 1.607i, and a5 -0.010; from a fit of the
measured Delta, sizes only), the contamination C = CONT a1 [sqrt(260.17 - D) + 2 Re sqrt(259.47 + 1.17i - D)] (#782's
near singularities, CONT 0.3), and relative noise NOISE on every coefficient: the larger of 1e-30 and the jets'
own, 1e-34..8e-34 from the plateau of the two-base least-squares residual of M2's forms on the real series at degrees
12-15 (no mirror evaluated; the jets' sums also meet the next states to 1e-37..5e-33), so 1e-30. Its mirror is R - S sqrt(D* - D) + C. A base is usable when both methods' base values are
within 1e-6 of the plant's mirror at all GRID points. Pass: at least 4 usable bases, each method's grid values within
1e-6 at all GRID points, and its + sheet within 1e-8 of F at the states in [259.64, D*) other than the base.
--shift=0.01 --usable=B1,B2,.. (the control) builds the plant with D* moved by 0.01 and judges it at the unmoved
plant's usable bases against the unmoved plant's mirror: it must fail.
Exit 1 when the verdict is not the expected one.

real: at the usable bases of PLANTED.json. (a) + sheet: both methods' chain sheet meets the measured Delta at the
states in [259.64, D*) other than the base to 1e-8; (b) the two methods' mirrors agree within 1e-6 at every GRID point,
reported with the spread over bases; (c) the table (CSV); (d) Delta*, a1..a4 by both methods, a1 beside #818's
0.0341..0.0352 (hp/data/lam_singularity_posthoc.json).

Methods: differential approximants, Guttmann and Joyce 1972, J. Phys. A 5, L81; quadratic Pade approximants,
Shafer 1974, SIAM J. Numer. Anal. 11, 447 (as cited in solver/inner_amp/pade_chord_arc.py).
"""
import glob
import json
import math
import os
import statistics
import sys

import mpmath as mp
import numpy as np

DPS = 80
DSTAR = mp.mpf("259.9033")        # hp/data/fold_259.csv (#603's kept plan)
D_MIN, D_MAX = 258.78, 259.8353   # the 46 bases registered on #829; #818 had the first 42, to 259.8184
N = 24
GRID = [mp.mpf(25964 + k) / 100 for k in range(27)]      # 259.64 .. 259.90
S_LO = 259.64                     # + sheet check: the states in [S_LO, D*)
TOL_M, TOL_P, NCOUNT = 1e-6, 1e-8, 4
W, PAIR = mp.mpf("0.3"), 0.05     # the forms' variable x = (D - D*)/W; a base's partner is the base nearest D_b - PAIR
DA_DEG1 = [(7, 7, 7), (7, 7, 8), (7, 8, 7), (8, 7, 7), (6, 8, 8), (8, 6, 8), (8, 8, 6)]
QP_DEG1 = [(7, 7, 7), (7, 7, 8), (7, 8, 7), (8, 7, 7), (7, 8, 8), (8, 7, 8), (8, 8, 7)]
DEG2 = [(10, 10, 11), (10, 11, 10), (11, 10, 10), (10, 11, 11), (11, 10, 11), (11, 11, 10), (11, 11, 11)]
DA_DEG2 = QP_DEG2 = DEG2          # least squares over the 48 (DA) or 50 (QP) conditions of the two bases
RHO, STEP, KT, MAXSTEP = mp.mpf("0.12"), mp.mpf("0.3"), 72, 1500
# the plant's sizes
P_DST, P_A1, P_A2, P_A3, P_A4, P_A5 = "1.6323", "0.035", "-0.010", "0.020", "0.010", "-0.010"
P_W, P_KAP, P_D2, P_D3 = "1e-3", (1.019, 1.607), "260.17", (259.47, 1.17)

args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:] if a.startswith("--"))
pos = [a for a in sys.argv[2:] if not a.startswith("--")]


# ---- series and polynomial arithmetic (coefficient lists, lowest first) ----
def smul(a, b, n=None):
    n = len(a) + len(b) - 1 if n is None else n
    return [mp.fsum(a[i] * b[k - i] for i in range(max(0, k - len(b) + 1), min(k, len(a) - 1) + 1)) for k in range(n)]


def sdiv(a, b, n):
    out = []
    for k in range(n):
        s = (a[k] if k < len(a) else 0) - mp.fsum(out[i] * b[k - i] for i in range(max(0, k - len(b) + 1), k))
        out.append(s / b[0])
    return out


def ssqrt(a, n):
    out = [mp.sqrt(a[0])]
    for k in range(1, n):
        s = (a[k] if k < len(a) else 0) - mp.fsum(out[i] * out[k - i] for i in range(1, k))
        out.append(s / (2 * out[0]))
    return out


def peval(p, x):
    v = mp.mpf(0)
    for c in reversed(p):
        v = v * x + c
    return v


def pshift(p, x):
    """coefficients of p(x + u) in u"""
    return [mp.fsum(p[k] * mp.binomial(k, j) * x ** (k - j) for k in range(j, len(p))) for j in range(len(p))]


def proots(p):
    """roots of p (numpy, for locating; the branch point is polished at DPS)"""
    q = [float(mp.re(c)) for c in p]
    while len(q) > 1 and q[-1] == 0:
        q.pop()
    return [complex(r) for r in np.roots(q[::-1])] if len(q) > 1 else []


def polish(p, x0):
    dp = [k * p[k] for k in range(1, len(p))]
    x = mp.mpc(x0)
    for _ in range(60):
        dx = peval(p, x) / peval(dp, x)
        x -= dx
        if abs(dx) < mp.mpf(10) ** (-DPS + 8):
            break
    return x


# ---- data ----
def jets(raw):
    fs = sorted(glob.glob(os.path.join(raw, "I603-win2", "chain_nsr", "jet_pin3_D*.npz")))
    return [f for f in fs if D_MIN - 1e-9 <= float(f.rsplit("_D", 1)[1][:-4]) <= D_MAX]


def delta_series(path):
    z = np.load(path)
    m = json.loads(str(z["meta"]))
    ks = sorted(k for k in z.files if k[0] == "z" and k[1:].isdigit())
    return mp.mpf(m["D_str"]), [mp.fsum(mp.mpf(float(z[k][n][-1])) for k in ks) for n in range(z[ks[0]].shape[0])]


def states(raw):
    out = []
    for p in sorted(glob.glob(os.path.join(raw, "I603-win2", "chain_nsr", "anm_pin3_D*.npz"))):
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        D = mp.mpf(meta["D_str"])
        if D >= S_LO:
            parts = [z[k] for k in ("zh", "zl", "z2", "z3", "z4", "z5", "z6", "z7") if k in z.files]
            out.append((D, mp.fsum(mp.mpf(float(q[-1])) for q in parts)))
    return out


# ---- the plant ----
class Plant:
    def __init__(self, shift=0.0, cont=0.3, noise=1e-30):
        self.Dst = DSTAR + mp.mpf(shift)
        self.cont, self.noise = mp.mpf(cont), noise
        self.R = [mp.mpf(P_DST), mp.mpf(P_A2), mp.mpf(P_A4)]       # in t0 = DSTAR - D
        self.S = [mp.mpf(P_A1), mp.mpf(P_A3), mp.mpf(P_A5)]
        self.kap = mp.mpc(*P_KAP)
        self.D3 = mp.mpc(*P_D3)

    def parts(self, D):
        t0 = DSTAR - D
        R = peval(self.R, t0) + mp.mpf(P_W) * mp.re(mp.exp(self.kap * (D - DSTAR)))
        S = peval(self.S, t0)
        C = self.cont * mp.mpf(P_A1) * (mp.sqrt(mp.mpf(P_D2) - D) + 2 * mp.re(mp.sqrt(self.D3 - D)))
        return R, S * mp.sqrt(self.Dst - D), C

    def value(self, D, sheet=1):
        R, Q, C = self.parts(D)
        return R + sheet * Q + C

    def series(self, Db, seed):
        """c_0..c_N of F at Db, with the relative noise"""
        n = N + 1
        a0 = DSTAR - Db                                     # t0 = a0 - h
        lin = [a0, mp.mpf(-1)]
        R = [mp.mpf(0)] * n
        S = [mp.mpf(0)] * n
        pw = [mp.mpf(1)] + [mp.mpf(0)] * N
        for j in range(3):
            if j:
                pw = smul(pw, lin, n)
            R = [R[k] + self.R[j] * pw[k] for k in range(n)]
            S = [S[k] + self.S[j] * pw[k] for k in range(n)]
        e0 = mp.exp(self.kap * (Db - DSTAR))
        R = [R[k] + mp.mpf(P_W) * mp.re(e0 * self.kap ** k / mp.factorial(k)) for k in range(n)]

        def rt(a):                                          # sqrt(a - h), principal, |h| < |a|
            return [mp.sqrt(a) * mp.binomial(mp.mpf(1) / 2, k) * (-1) ** k / a ** k for k in range(n)]
        fold = smul(S, rt(self.Dst - Db), n)
        r2, r3 = rt(mp.mpf(P_D2) - Db), rt(self.D3 - Db)
        C = [self.cont * mp.mpf(P_A1) * (r2[k] + 2 * mp.re(r3[k])) for k in range(n)]
        rng = np.random.default_rng(seed)
        return [mp.re(R[k] + fold[k] + C[k]) * (1 + mp.mpf(self.noise) * mp.mpf(rng.standard_normal())) for k in range(n)]


# ---- the two approximant families, matched at one base or two ----
def forms(inputs, deg, kind):
    """the polynomials, in h at the anchor inputs[0], of P f^2 + Q f + R (kind QP) or Q1 f' + Q0 f + R (kind DA) whose
    expansion vanishes at every base of inputs to all the orders its series gives (25 for QP, 24 for DA): exactly when
    the conditions match the unknowns, otherwise by least squares with each condition scaled to unit largest entry.
    Built in x = (D - D*)/W, normalized by the coefficient of f at x = 0 (nonzero for both: Q(D*) = -2 Delta* P(D*),
    Q0(D*) = -mu Q1'(D*)). Returns the polynomials and the residual norm."""
    l, m, n = deg
    need = l + m + n + 2
    kmax = N + 1 if kind == "QP" else N
    if need > kmax * len(inputs):
        raise ValueError("degrees %s need %d conditions" % (deg, need))
    rows = []
    for Db, c in inputs:
        ch = [c[k] * W ** k for k in range(N + 1)]
        xb = (Db - DSTAR) / W
        g1 = smul(ch, ch, N + 1) if kind == "QP" else [(k + 1) * ch[k + 1] for k in range(N)]
        for k in range(kmax):
            row = []
            for g, d in ((g1, l), (ch, m)):
                for i in range(d + 1):
                    row.append(mp.fsum(mp.binomial(i, j) * xb ** (i - j) * g[k - j] for j in range(min(i, k) + 1)))
            for i in range(n + 1):
                row.append(mp.binomial(i, k) * xb ** (i - k) if i >= k else mp.mpf(0))
            sc = max(abs(v) for v in row)
            rows.append([v / sc for v in row])
    j0 = l + 1                                              # the coefficient of f at x = 0
    A = mp.matrix([r[:j0] + r[j0 + 1:] for r in rows])
    b = mp.matrix([-r[j0] for r in rows])
    if len(rows) == need:
        x, res = mp.lu_solve(A, b), mp.mpf(0)
    else:
        x, res = mp.qr_solve(A, b)
    v = [x[i] for i in range(j0)] + [mp.mpf(1)] + [x[i] for i in range(j0, need)]
    xa = (inputs[0][0] - DSTAR) / W
    toh = lambda p: [y / W ** j for j, y in enumerate(pshift(p, xa))]   # noqa: E731
    P, Q, R = toh(v[:l + 1]), toh(v[l + 1:l + m + 2]), toh(v[l + m + 2:])
    if kind == "DA":
        P = [W * y for y in P]                              # Q1 multiplies df/dx = W df/dh
    return (P, Q, R), res


class QPade:
    """M2: the roots of P g^2 + Q g + R = 0"""
    def __init__(self, polys, Db, c0):
        self.P, self.Q, self.R = polys
        self.Db = Db
        PR = smul(self.P, self.R)
        QQ = smul(self.Q, self.Q)
        L = max(len(PR), len(QQ))
        self.disc = [(QQ[k] if k < len(QQ) else 0) - 4 * (PR[k] if k < len(PR) else 0) for k in range(L)]
        d0 = peval(self.disc, 0)
        self.ok = d0 > 0
        r0 = [(-peval(self.Q, 0) + sg * mp.sqrt(abs(d0))) / (2 * peval(self.P, 0)) for sg in (1, -1)]
        self.sig = 1 if abs(r0[0] - c0) < abs(r0[1] - c0) else -1
        self.res0 = min(abs(r0[0] - c0), abs(r0[1] - c0))
        rd = proots(self.disc)
        self.bad = [r.real for r in rd + proots(self.P) if abs(r.imag) < 1e-9]
        hs = float(DSTAR - Db)
        cand = [r for r in rd if abs(r.imag) < 1e-6 and abs(r.real - hs) < 0.02]
        self.hstar = mp.re(polish(self.disc, min(cand, key=lambda r: abs(r.real - hs)).real)) if cand else None

    def defined(self, h):
        """no real zero of the discriminant or of P between the base and h"""
        lo, hi = sorted((0.0, float(h)))
        return self.ok and not any(lo - 1e-12 <= x <= hi + 1e-12 for x in self.bad)

    def value(self, D, sheet):
        h = D - self.Db
        if not self.defined(h):
            return None
        return (-peval(self.Q, h) + sheet * self.sig * mp.sqrt(peval(self.disc, h))) / (2 * peval(self.P, h))

    def coeffs(self):
        """(D*, Delta*, a1, a2, a3, a4)"""
        if self.hstar is None:
            return None
        hs, n = self.hstar, 3
        neg = lambda p: [(-1) ** k * x for k, x in enumerate(pshift(p, hs))]   # noqa: E731  p(h* - t) in t
        Pt, Qt, dt = neg(self.P), neg(self.Q), neg(self.disc)
        E = sdiv([-x / 2 for x in Qt], Pt, n)
        O = sdiv([self.sig * x / 2 for x in ssqrt(dt[1:], n)], Pt, n)
        return [self.Db + hs, E[0], O[0], E[1], O[1], E[2]]


class DApprox:
    """M1: the differential approximant Q1 f' + Q0 f + R = 0, continued around its zero h* near D*"""
    def __init__(self, polys, Db, c0):
        self.Q1, self.Q0, self.R = polys
        self.Db, self.c0 = Db, c0
        self.zeros = [complex(z) for z in proots(self.Q1)]
        hs = float(DSTAR - Db)
        cand = [z for z in self.zeros if abs(z.imag) < 1e-6 and abs(z.real - hs) < 0.02]
        self.why = ""
        if not cand:
            self.hstar, self.why = None, "no zero of Q1 within 0.02 of D*"
            return
        z0 = min(cand, key=lambda z: abs(z.real - hs))
        self.hstar = mp.re(polish(self.Q1, z0.real))
        others = [z for z in self.zeros if z is not z0]
        dn = min((abs(z - complex(self.hstar)) for z in others), default=1.0)
        self.rho = min(RHO, mp.mpf(0.45 * dn))
        dQ1 = [k * self.Q1[k] for k in range(1, len(self.Q1))]
        self.mu = -peval(self.Q0, self.hstar) / peval(dQ1, self.hstar)
        self.zc = [mp.mpc(z) for z in others] + [mp.mpc(self.hstar)]
        if self.rho < mp.mpf("0.005"):
            self.why = "another zero of Q1 within 0.011 of h*"
        lo = min(0.0, float(self.hstar - self.rho), float(min(GRID) - Db))
        for z in others:
            if abs(z.imag) < 1e-3 and lo - 1e-3 <= z.real <= float(self.hstar) + 1e-3:
                self.why = "a zero of Q1 on the real path"

    def taylor(self, z, f):
        Q1, Q0, R = (pshift(p, z) for p in (self.Q1, self.Q0, self.R))
        phi = [f]
        for k in range(KT):
            s = mp.fsum(Q1[j] * (k - j + 1) * phi[k - j + 1] for j in range(1, min(k, len(Q1) - 1) + 1))
            s += mp.fsum(Q0[j] * phi[k - j] for j in range(0, min(k, len(Q0) - 1) + 1))
            s += R[k] if k < len(R) else 0
            phi.append(-s / (Q1[0] * (k + 1)))
        return phi

    def walk(self, z, f, z1):
        """f continued from z to z1 along the straight segment"""
        z, z1 = mp.mpc(z), mp.mpc(z1)
        for _ in range(MAXSTEP):
            if abs(z1 - z) < mp.mpf(10) ** (-DPS + 10):
                return z1, f
            d = min(abs(z - w) for w in self.zc)
            st = min(STEP * d, abs(z1 - z))
            u = (z1 - z) / abs(z1 - z) * st
            phi = self.taylor(z, f)
            f = mp.fsum(p * u ** k for k, p in enumerate(phi))
            z = z + u
        raise RuntimeError("the continuation did not reach its end in %d steps" % MAXSTEP)

    def sheets(self, Ds):
        """{D: {+1: chain, -1: mirror}} on the real axis, by the loop of radius rho"""
        hs, rho = self.hstar, self.rho
        z, f = self.walk(0, mp.mpc(self.c0), hs - rho)
        start = (z, f)
        nv = 64
        for j in range(1, nv + 1):
            z, f = self.walk(z, f, hs + rho * mp.expj(mp.pi + 2 * mp.pi * j / nv))
        out = {}
        for sheet, (z0, f0) in ((1, start), (-1, (mp.mpc(hs - rho), f))):
            left = sorted((D for D in Ds if D - self.Db <= hs - rho), reverse=True)
            right = sorted(D for D in Ds if D - self.Db > hs - rho)
            for seq in (left, right):
                z, f = z0, f0
                for D in seq:
                    z, f = self.walk(z, f, D - self.Db)
                    out.setdefault(D, {})[sheet] = f
        return out

    def coeffs(self):
        """(D*, Delta*, a1, a2, a3, a4, mu): Frobenius at h*, K matched at t = rho/2"""
        hs = self.hstar
        neg = lambda p: [(-1) ** k * x for k, x in enumerate(pshift(p, hs))]   # noqa: E731
        A = [-x for x in neg(self.Q1)]                      # A(t) df/dt + B f + C = 0, t = h* - h
        B, C = neg(self.Q0), neg(self.R)
        A[0] = mp.mpf(0)
        mu = -B[0] / A[1]
        nt = 160
        p, q = [], [mp.mpf(1)]
        for n in range(nt):
            s = mp.fsum(A[j] * p[n + 1 - j] * (n + 1 - j) for j in range(2, len(A)) if 0 <= n + 1 - j < n)
            s += mp.fsum(B[j] * p[n - j] for j in range(1, len(B)) if n - j >= 0)
            s += C[n] if n < len(C) else 0
            p.append(-s / (A[1] * n + B[0]))
        for n in range(1, nt):
            s = mp.fsum(A[j] * q[n + 1 - j] * (n + 1 - j + mu) for j in range(2, len(A)) if 0 <= n + 1 - j < n)
            s += mp.fsum(B[j] * q[n - j] for j in range(1, len(B)) if n - j >= 0)
            q.append(-s / (A[1] * n))
        t = self.rho / 2
        _, f = self.walk(0, mp.mpc(self.c0), hs - t)
        fp = mp.fsum(x * t ** k for k, x in enumerate(p))
        fh = t ** mu * mp.fsum(x * t ** k for k, x in enumerate(q))
        K = (f - fp) / fh
        return [self.Db + hs, p[0], K, p[1], K * q[1], p[2], mu]


# ---- one base (with its partner) ----
def run_base(inputs, targets, pstates, points):
    """both methods: per approximant the grid sheets, the + sheet at the states, the coefficients"""
    inputs = inputs[:points]
    Db, c0 = inputs[0][0], inputs[0][1][0]
    Ds = sorted(set(targets) | set(D for D, _ in pstates))
    res = dict(M1=[], M2=[])
    for deg in (DA_DEG2 if points == 2 else DA_DEG1):
        e = dict(deg=deg, why="")
        try:
            polys, rs = forms(inputs, deg, "DA")
            a = DApprox(polys, Db, c0)
            e.update(why=a.why, res=float(rs))
            if not a.why:
                sh = a.sheets(Ds)
                e.update(Dstar=float(Db + a.hstar), mu=float(a.mu), rho=float(a.rho),
                         grid={str(D): [sh[D][1], sh[D][-1]] for D in targets},
                         plus={str(D): sh[D][1] for D, _ in pstates}, coeffs=a.coeffs())
        except (RuntimeError, ZeroDivisionError, ValueError) as ex:
            e["why"] = str(ex)
        res["M1"].append(e)
    for deg in (QP_DEG2 if points == 2 else QP_DEG1):
        e = dict(deg=deg, why="")
        try:
            polys, rs = forms(inputs, deg, "QP")
            q = QPade(polys, Db, c0)
            e.update(why="" if q.ok else "discriminant <= 0 at the base", res=float(rs))
            if q.ok:
                e.update(Dstar=float(Db + q.hstar) if q.hstar is not None else None,
                         grid={str(D): [q.value(D, 1), q.value(D, -1)] for D in targets},
                         plus={str(D): q.value(D, 1) for D, _ in pstates}, coeffs=q.coeffs(), res0=float(q.res0))
        except (ZeroDivisionError, ValueError) as ex:
            e["why"] = str(ex)
        res["M2"].append(e)
    return res


def reduce_base(res, targets, pstates):
    """per method: the median over approximants at each D (mirror, chain), half ranges, |Im|, the + sheet"""
    out = {}
    for m in ("M1", "M2"):
        ok = [e for e in res[m] if not e["why"] and all(e["grid"][str(D)][1] is not None for D in targets)
              and all(e["plus"][str(D)] is not None for D, _ in pstates)]
        if not ok:
            out[m] = None
            continue
        g = {}
        for D in targets:
            mv = [mp.re(e["grid"][str(D)][1]) for e in ok]
            cv = [mp.re(e["grid"][str(D)][0]) for e in ok]
            im = max(abs(mp.im(e["grid"][str(D)][1])) for e in ok)
            g[str(D)] = dict(mirror=statistics.median(mv), mhalf=(max(mv) - min(mv)) / 2,
                             chain=statistics.median(cv), chalf=(max(cv) - min(cv)) / 2, im=im)
        p = {str(D): statistics.median([mp.re(e["plus"][str(D)]) for e in ok]) for D, _ in pstates}
        cf = [e["coeffs"] for e in ok if e.get("coeffs")]
        coef = [statistics.median([mp.re(x[i]) for x in cf]) for i in range(len(cf[0]))] if cf else None
        out[m] = dict(n=len(ok), of=len(res[m]), grid=g, plus=p, coeffs=coef,
                      why=sorted(set(e["why"] for e in res[m] if e["why"])))
    return out


def base_inputs(raw, plant, seed_base):
    """[(D_b, c, partner's (D, c))]: each base's series (the plant's, when given) and its partner's"""
    ser = []
    for i, jp in enumerate(jets(raw)):
        Db, c = delta_series(jp)
        ser.append((Db, plant.series(Db, seed_base + i) if plant is not None else c))
    out = []
    for i, (Db, c) in enumerate(ser):
        j = min((k for k in range(len(ser)) if k != i), key=lambda k: abs(float(ser[k][0]) - (float(Db) - PAIR)))
        out.append((Db, c, ser[j]))
    return out


def aggregate(red, usable, targets):
    """per method and D: value, error (see the docstring)"""
    agg = {}
    for m in ("M1", "M2"):
        rows = {}
        for D in targets:
            k = str(D)
            v = [red[b][m]["grid"][k]["mirror"] for b in usable]
            ch = [red[b][m]["grid"][k]["chain"] for b in usable]
            med, cmed = statistics.median(v), statistics.median(ch)
            err = max(max(abs(x - med) for x in v), max(red[b][m]["grid"][k]["mhalf"] for b in usable))
            cerr = max(max(abs(x - cmed) for x in ch), max(red[b][m]["grid"][k]["chalf"] for b in usable))
            im = max(red[b][m]["grid"][k]["im"] for b in usable)
            rows[k] = dict(mirror=med, err=err, chain=cmed, cerr=cerr, im=im,
                           spread_bases=max(v) - min(v))
        agg[m] = rows
    return agg


def fmt(x):
    return mp.nstr(x, 20) if x is not None else "-"


def cmd_planted(raw):
    mp.mp.dps = DPS
    shift = float(args.get("shift", 0))
    cont = float(args.get("cont", 0.3))
    noise = float(args.get("noise", 1e-30))
    plant = Plant(shift, cont, noise)
    truth = Plant(0.0, cont, noise)
    targets = GRID
    sts = [(D, plant.value(D)) for D, _ in states(raw)]
    print("planted: D* %s (judged against D* %s), contamination %s a1, noise %.0e" % (
        mp.nstr(plant.Dst, 10), mp.nstr(DSTAR, 10), cont, noise))
    red, rows = {}, {}
    points = int(args.get("points", 2))
    only = set(round(float(x), 4) for x in args["bases"].split(",")) if "bases" in args else None
    for Db, c, (Dp, cp) in base_inputs(raw, plant, 829):
        if only is not None and round(float(Db), 4) not in only:
            continue
        pst = [(D, v) for D, v in sts if abs(D - Db) > mp.mpf("1e-9") and (points == 1 or abs(D - Dp) > mp.mpf("1e-9"))]
        res = run_base([(Db, c), (Dp, cp)], targets, pst, points)
        r = reduce_base(res, targets, pst)
        b = float(Db)
        red[b] = r
        line = dict(D=b)
        ok_all = True
        for m in ("M1", "M2"):
            if r[m] is None:
                line[m] = dict(valid=0, why=sorted(set(e["why"] for e in res[m])))
                ok_all = False
                continue
            em = max(abs(r[m]["grid"][str(D)]["mirror"] - truth.value(D, -1)) for D in targets)
            ep = max(abs(r[m]["plus"][str(D)] - v) for D, v in pst)
            line[m] = dict(valid=r[m]["n"], err_mirror=float(em), err_plus=float(ep), why=r[m]["why"],
                           coeffs=[float(x) for x in r[m]["coeffs"]] if r[m]["coeffs"] else None)
            ok_all &= em <= TOL_M
        line["usable"] = ok_all
        rows[b] = line
        print("  D_b %.4f  M1 %s | M2 %s | %s" % (b, *[
            ("%d/%d mirror %.1e plus %.1e" % (line[m]["valid"], 7, line[m]["err_mirror"], line[m]["err_plus"])
             if line[m].get("err_mirror") is not None else "none valid (%s)" % "; ".join(line[m]["why"]))
            for m in ("M1", "M2")], "usable" if ok_all else "not usable"), flush=True)
    usable = [b for b in rows if rows[b]["usable"]]
    if "usable" in args:                                    # the control: judged at the given bases
        usable = [b for b in rows if round(b, 4) in set(round(float(x), 4) for x in args["usable"].split(","))]
    verdict, out = True, dict(shift=shift, cont=cont, noise=noise, points=points, bases=rows, usable=usable)
    if len(usable) == 0:
        print("planted: no usable base: FAIL")
        verdict = False
    else:
        agg = aggregate(red, usable, targets)
        for m in ("M1", "M2"):
            em = max(abs(agg[m][str(D)]["mirror"] - truth.value(D, -1)) for D in targets)
            ep = max(rows[b][m]["err_plus"] for b in usable)
            ok = em <= TOL_M and ep <= TOL_P
            verdict &= ok
            out[m] = dict(err_mirror=float(em), err_plus=float(ep), ok=ok,
                          largest_spread=float(max(agg[m][str(D)]["err"] for D in targets)))
            print("  %s over %d usable bases: mirror within %.1e of the plant's (bound %.0e), + sheet %.1e (bound %.0e), "
                  "stated error up to %.1e: %s" % (m, len(usable), em, TOL_M, ep, TOL_P,
                                                   out[m]["largest_spread"], "pass" if ok else "FAIL"))
        verdict &= len(usable) >= NCOUNT
    out["pass"] = bool(verdict)
    print("planted: %d usable bases (%s): %s" % (len(usable), ", ".join("%.4f" % b for b in usable),
                                                 "PASS" if verdict else "FAIL"))
    if "json" in args:
        json.dump(out, open(args["json"], "w"), indent=1, default=float)
    expect = shift == 0
    return 0 if verdict == expect else 1


def cmd_real(raw):
    mp.mp.dps = DPS
    pl = json.load(open(args["planted"]))
    usable = set(round(x, 4) for x in pl["usable"])
    targets = GRID
    sts = states(raw)
    red, rows = {}, {}
    print("real: the usable bases of %s: %s" % (args["planted"], ", ".join("%.4f" % x for x in sorted(usable))))
    points = int(pl.get("points", 2))
    for Db, c, (Dp, cp) in base_inputs(raw, None, 0):
        b = float(Db)
        if round(b, 4) not in usable:
            continue
        pst = [(D, v) for D, v in sts if abs(D - Db) > mp.mpf("1e-9") and (points == 1 or abs(D - Dp) > mp.mpf("1e-9"))]
        res = run_base([(Db, c), (Dp, cp)], targets, pst, points)
        r = reduce_base(res, targets, pst)
        red[b] = r
        line = dict(D=b)
        for m in ("M1", "M2"):
            if r[m] is None:
                line[m] = dict(valid=0, why=sorted(set(e["why"] for e in res[m])))
                continue
            ep = max(abs(r[m]["plus"][str(D)] - v) for D, v in pst)
            line[m] = dict(valid=r[m]["n"], err_plus=float(ep), why=r[m]["why"],
                           coeffs=[float(x) for x in r[m]["coeffs"]] if r[m]["coeffs"] else None,
                           mirror={str(D): float(r[m]["grid"][str(D)]["mirror"]) for D in targets})
        rows[b] = line
        print("  D_b %.4f  M1 %s | M2 %s" % (b, *[
            ("%d/%d + sheet %.1e" % (line[m]["valid"], 7, line[m]["err_plus"])) if line[m]["valid"] else
            "none valid (%s)" % "; ".join(line[m]["why"]) for m in ("M1", "M2")]), flush=True)
    good = [b for b in rows if all(rows[b][m]["valid"] for m in ("M1", "M2"))]
    agg = aggregate(red, good, targets)
    plus_ok = all(rows[b][m]["err_plus"] <= TOL_P for b in good for m in ("M1", "M2"))
    diff = max(abs(agg["M1"][str(D)]["mirror"] - agg["M2"][str(D)]["mirror"]) for D in targets)
    print("(a) + sheet against the measured Delta at the states in [%.2f, D*), %d bases: largest %.1e (bound %.0e): %s" % (
        S_LO, len(good), max(rows[b][m]["err_plus"] for b in good for m in ("M1", "M2")), TOL_P,
        "PASS" if plus_ok else "FAIL"))
    print("(b) M1 against M2 on the mirror: largest |difference| %.1e (bound %.0e): %s" % (
        diff, TOL_M, "PASS" if diff <= TOL_M else "FAIL"))
    print("     D        mirror (M1)               mirror (M2)               mean                 error     chain (+ sheet)")
    table = []
    for D in targets:
        k = str(D)
        v1, v2 = agg["M1"][k]["mirror"], agg["M2"][k]["mirror"]
        mean = (v1 + v2) / 2
        err = max(agg["M1"][k]["err"], agg["M2"][k]["err"], abs(v1 - v2) / 2)
        ch = (agg["M1"][k]["chain"] + agg["M2"][k]["chain"]) / 2
        cerr = max(agg["M1"][k]["cerr"], agg["M2"][k]["cerr"], abs(agg["M1"][k]["chain"] - agg["M2"][k]["chain"]) / 2)
        table.append(dict(D=float(D), mirror=mp.nstr(mean, 17), err=float(err), mirror_M1=mp.nstr(v1, 17),
                          err_M1=float(agg["M1"][k]["err"]), mirror_M2=mp.nstr(v2, 17), err_M2=float(agg["M2"][k]["err"]),
                          im_M1=float(agg["M1"][k]["im"]), spread_bases_M1=float(agg["M1"][k]["spread_bases"]),
                          spread_bases_M2=float(agg["M2"][k]["spread_bases"]), chain=mp.nstr(ch, 17), chain_err=float(cerr)))
        print("  %.2f  %s  %s  %s  %.1e  %s" % (D, mp.nstr(v1, 20), mp.nstr(v2, 20), mp.nstr(mean, 16), err, mp.nstr(ch, 16)))
    names = ["D*", "Delta*", "a1", "a2", "a3", "a4"]
    coef = {}
    for m in ("M1", "M2"):
        cs = [rows[b][m]["coeffs"] for b in good if rows[b][m]["coeffs"]]
        coef[m] = [dict(median=statistics.median(x[i] for x in cs), lo=min(x[i] for x in cs), hi=max(x[i] for x in cs))
                   for i in range(6)]
        if m == "M1":
            mus = [x[6] for x in cs]
            coef["mu_M1"] = dict(median=statistics.median(mus), lo=min(mus), hi=max(mus))
    print("Delta* and a1..a4 (median over bases [range over bases]):")
    for i, nm in enumerate(names):
        print("  %-6s M1 %+.12f [%+.12f, %+.12f]   M2 %+.12f [%+.12f, %+.12f]" % (
            nm, coef["M1"][i]["median"], coef["M1"][i]["lo"], coef["M1"][i]["hi"],
            coef["M2"][i]["median"], coef["M2"][i]["lo"], coef["M2"][i]["hi"]))
    print("  mu (M1) %.10f [%.10f, %.10f]" % (coef["mu_M1"]["median"], coef["mu_M1"]["lo"], coef["mu_M1"]["hi"]))
    if "json" in args:
        json.dump(dict(bases=rows, used=good, table=table, coeffs=coef, plus_ok=plus_ok, diff=float(diff)),
                  open(args["json"], "w"), indent=1, default=float)
    if "csv" in args:
        with open(args["csv"], "w") as fh:
            ks = list(table[0])
            fh.write(",".join(ks) + "\n")
            for t in table:
                fh.write(",".join(str(t[k]) for k in ks) + "\n")
    return 0 if plus_ok and diff <= TOL_M else 1


def cmd_literal(raw):
    """the truncated series re-expanded around D* with no model: it returns its starting value"""
    mp.mp.dps = DPS
    plant = Plant()
    Db = base_inputs(raw, None, 0)[-1][0]
    c = plant.series(Db, 1)
    hs, rho = DSTAR - Db, mp.mpf("0.06")
    z, p = mp.mpc(0), [mp.mpc(x) for x in c]
    pts = [hs - rho] + [hs + rho * mp.expj(mp.pi + 2 * mp.pi * j / 64) for j in range(1, 65)]
    for z1 in pts:
        p = pshift(p, z1 - z)
        z = z1
    D = Db + hs - rho
    print("literal: base %.4f, the 25-term series re-expanded at 65 centres around D* (radius %s) returns %s at D %s;"
          " the plant's chain %s, mirror %s" % (Db, rho, mp.nstr(p[0], 15), mp.nstr(D, 8), mp.nstr(plant.value(D), 15),
                                                mp.nstr(plant.value(D, -1), 15)))
    return 0


def cmd_noise(raw):
    """the jets' coefficient noise: the residual of the two-base least-squares forms at degrees 10..15 on three base
    pairs, on plants of relative noise 1e-30, 1e-33, 1e-36 and on the real series; no mirror is evaluated. The real
    noise is the real M2 residual at degree 15 over the 1e-33 plant's residual per unit noise"""
    mp.mp.dps = DPS
    pairs = [("259.8352", "259.7864"), ("259.8068", "259.7603"), ("259.7510", "259.7053")]
    ser = {"%.4f" % float(Db): (Db, c) for Db, c, _ in base_inputs(raw, None, 0)}
    out = {}
    for src in ("1e-30", "1e-33", "1e-36", "real"):
        for a, b in pairs:
            if src == "real":
                inputs = [ser[a], ser[b]]
            else:
                pl = Plant(0.0, 0.3, float(src))
                inputs = [(ser[a][0], pl.series(ser[a][0], 1)), (ser[b][0], pl.series(ser[b][0], 2))]
            row = {}
            for d in range(10, 16):
                for kind in ("QP", "DA"):
                    row["%s%d" % (kind, d)] = float(forms(inputs, (d, d, d), kind)[1])
            out["%s %s,%s" % (src, a, b)] = row
            print("%-5s %s,%s  " % (src, a, b) + " ".join("%s %.1e" % kv for kv in row.items()), flush=True)
    est = [out["real %s,%s" % pr]["QP15"] / (out["1e-33 %s,%s" % pr]["QP15"] / 1e-33) for pr in pairs]
    print("the jets' relative noise from M2's degree-15 residual: %.1e to %.1e; the plant carries the larger of it "
          "and 1e-30" % (min(est), max(est)))
    if "json" in args:
        json.dump(dict(residuals=out, noise=est), open(args["json"], "w"), indent=1)
    return 0


def cmd_posthoc(raw):
    """post hoc, from PLANTED.json and REAL.json: the plant's Puiseux data at D* against the methods' at the usable
    bases; the real result restricted to #818's 42 bases (to 259.8184); the stated spread over the actual error"""
    mp.mp.dps = DPS
    pl, re_ = json.load(open(args["planted"])), json.load(open(args["real"]))
    plant = Plant()
    E = lambda t: (lambda q: q[0] + q[2])(plant.parts(DSTAR - t))      # noqa: E731  R + C, the even part
    e = mp.taylor(E, 0, 2)
    sd = mp.taylor(lambda t: peval(plant.S, t), 0, 1)
    truth = [float(DSTAR), float(e[0]), float(sd[0]), float(e[1]), float(sd[1]), float(e[2])]
    names = ["Dstar", "Delta*", "a1", "a2", "a3", "a4"]
    use = [b for b in pl["bases"] if pl["bases"][b]["usable"]]
    worst = {nm: {m: max(abs(pl["bases"][b][m]["coeffs"][i] - truth[i]) for b in use) for m in ("M1", "M2")}
             for i, nm in enumerate(names)}
    worst["mu"] = dict(M1=max(abs(pl["bases"][b]["M1"]["coeffs"][6] - 0.5) for b in use))
    print("the plant's Puiseux data at D* against both methods at its %d usable bases (worst):" % len(use))
    for i, nm in enumerate(names):
        print("  %-6s truth %+.15f  M1 %.1e  M2 %.1e" % (nm, truth[i], worst[nm]["M1"], worst[nm]["M2"]))
    print("  mu (M1): |mu - 1/2| %.1e" % worst["mu"]["M1"])
    ratio = {m: pl[m]["largest_spread"] / pl[m]["err_mirror"] for m in ("M1", "M2")}
    print("the plant's largest stated spread over its largest actual error: M1 %.0f, M2 %.0f" % (ratio["M1"], ratio["M2"]))
    r8 = [b for b in re_["bases"] if float(b) <= 259.8185]
    t = {"%.2f" % x["D"]: x for x in re_["table"]}
    d7, d12 = 0.0, 0.0
    for k in re_["bases"][r8[0]]["M1"]["mirror"]:
        v1 = statistics.median([re_["bases"][b]["M1"]["mirror"][k] for b in r8])
        v2 = statistics.median([re_["bases"][b]["M2"]["mirror"][k] for b in r8])
        d7 = max(d7, abs((v1 + v2) / 2 - float(t["%.2f" % float(k)]["mirror"])))
        d12 = max(d12, abs(v1 - v2))
    print("#818's 42 bases only: %d usable (%s); their mirror against the table %.1e, M1 against M2 %.1e" % (
        len(r8), ", ".join("%.4f" % float(b) for b in r8), d7, d12))
    if "json" in args:
        json.dump(dict(truth=dict(zip(names, truth)), worst=worst, spread_over_error=ratio,
                       r818=dict(bases=[float(b) for b in r8], vs_table=d7, m1_m2=d12)), open(args["json"], "w"), indent=1)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = dict(planted=cmd_planted, real=cmd_real, literal=cmd_literal, noise=cmd_noise, posthoc=cmd_posthoc).get(cmd)
    if fn is None or not pos:
        print(__doc__)
        sys.exit(2)
    sys.exit(fn(pos[0]))
