#!/usr/bin/env python3
"""P2c review (#1017): (H8) of B8_constants.md section 8, recomputed independently (P2_remaining.md section 5).

Written from B8_constants.md 1.2 (the reduced system), 2 (the series and its gauge) and 8.2 (the conversion and the
three pieces), not from b8m_h8.py, b8m_formal.py, b8m_atoms.py, b8m_bounds.py or b8_alg.py (none opened). Arithmetic:
mpmath complex intervals (iv) at 256 bits.

M = 1, odd harmonics {-1, 1} (V, S, Y, P), even {-2, 0, 2} (beta, c), every product projected. With z = x^2/n and
eps = 1/n the reduced system (x d/dx = 2 z d/dz) reads, after multiplying the S- and beta-equations by eps,
    2 z V_z = Omega S,                                   2 z c_z = [c beta],
    (z - eps C^2) 2 z S_z = G,   G = (1 - eps) C^2 S + eps C B S - z Omega S - z Y + z B P - z C Omega P,
    (1 - eps) beta = (1 - eps)(eps [SS] + z [PP])/2 + eps^2 [beta [SS]]/2 + eps z [beta [PP]]/2 - eps [beta beta]
                     - 2 eps z beta_z,
with Y = V + S, P = C^-1 Y (Delta = x P), C f = [c f], B f = [[c beta] f], Omega = diag(i mu omega). This is 1.2's
system with D = x^2 - C^2, A = (n - 1) C^2 + C B - x^2 Omega - x^2 B and B = I - B C^-1 + C Omega C^-1, so that
x^2 B V + x^2 B S = x^2 (Y - B P + C Omega P). It is regular in eps: monomials eps^a z^j with a, j >= 0. The eps^0 part
is the inner system; Z^K is the part with a + j <= K (eps^k x^(2j) = eps^(k-j) z^j). Gauge (section 2): V(0) = [c(0) 2F]
at every order, and c^K = 1 at x = 1, i.e. sum over a + j = k of c_{a,j} = 0 for k >= 1 (and c_{0,0} = e_0).

    python3 p2c_h8.py --centre [--mutate=A|sign]   # series at the box centre: residual orders, leading terms (P1)
    python3 p2c_h8.py [--d0=0|1e7] [--mutate=A|sign|hS] [--point] [--out=DIR]   # the five numbers over the box
                                                   (--point: at the box centre, for the structural zeros' size)
"""
import json
import os
import sys
import time

from mpmath import iv, mp

iv.prec = 256
mp.prec = 256
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
ODD, EVEN = (-1, 1), (-2, 0, 2)
K = 4
GAMMA = iv.mpf(3) / 10
L0 = iv.mpf("73.89")                         # ln n0
A_C, W_C = iv.mpf("2.5421480475"), iv.mpf("0.7769940136")   # the box centre (B8_constants.md 8.4, item 3)
BOXR = iv.mpf("2e-6")
WINF = 2 * iv.pi / iv.mpf("1.64389375")
ZERO = iv.mpc(0)
IVC = type(ZERO)


def cx(v):
    """an iv complex from a number or an iv real or complex"""
    return v if isinstance(v, IVC) else iv.mpc(v)


# ----------------------------------------------------------------------------------------- double series in (eps, z)
class Ser:
    """sum c_{a,j} eps^a z^j over a truncation set (a downward-closed predicate keep(a, j))"""
    __slots__ = ("c", "keep")

    def __init__(self, c=None, keep=None):
        self.c = c if c is not None else {}
        self.keep = keep

    @staticmethod
    def const(v, keep):
        return Ser({(0, 0): cx(v)}, keep)

    def copy(self):
        return Ser(dict(self.c), self.keep)

    def __add__(self, o):
        if not isinstance(o, Ser):
            o = Ser.const(o, self.keep)
        out = dict(self.c)
        for k, v in o.c.items():
            out[k] = out[k] + v if k in out else v
        return Ser(out, self.keep)

    def __neg__(self):
        return Ser({k: -v for k, v in self.c.items()}, self.keep)

    def __sub__(self, o):
        return self + (-o if isinstance(o, Ser) else -cx(o))

    def __mul__(self, o):
        if not isinstance(o, Ser):
            o = cx(o)
            return Ser({k: v * o for k, v in self.c.items()}, self.keep)
        out = {}
        keep = self.keep
        for (a1, j1), v1 in self.c.items():
            for (a2, j2), v2 in o.c.items():
                k = (a1 + a2, j1 + j2)
                if keep(*k):
                    out[k] = out[k] + v1 * v2 if k in out else v1 * v2
        return Ser(out, keep)

    __rmul__ = __mul__

    def shift(self, da, dj):
        """times eps^da z^dj"""
        return Ser({(a + da, j + dj): v for (a, j), v in self.c.items() if self.keep(a + da, j + dj)}, self.keep)

    def zdz(self):
        """2 z d/dz (= x d/dx at fixed n)"""
        return Ser({(a, j): v * (2 * j) for (a, j), v in self.c.items() if j}, self.keep)

    def zint(self):
        """int_0^z f dz'/(2 z') for f with no z^0 terms"""
        return Ser({(a, j): v / (2 * j) for (a, j), v in self.c.items() if j}, self.keep)

    def at_z0(self):
        """the z^0 part, a series in eps"""
        return Ser({(a, j): v for (a, j), v in self.c.items() if j == 0}, self.keep)

    def inv(self):
        """1/f for f with a nonzero constant term (Neumann series, exact on the truncation set)"""
        c0 = self.c.get((0, 0), ZERO)
        r = Ser({k: v / c0 for k, v in self.c.items() if k != (0, 0)}, self.keep)
        out = Ser.const(1, self.keep)
        p = Ser.const(1, self.keep)
        for _ in range(200):
            p = p * r * (-1)
            if not p.c:
                break
            out = out + p
        else:
            raise RuntimeError("inv: the truncation set is not finite in the grading")
        return out * (1 / c0)

    def coef(self, a, j):
        return self.c.get((a, j), ZERO)


def svec(keys, keep):
    return {k: Ser({}, keep) for k in keys}


def pr(f, g, out_keys, keep):
    """[f g] projected on out_keys"""
    out = {k: Ser({}, keep) for k in out_keys}
    for i, fi in f.items():
        for j, gj in g.items():
            if i + j in out:
                out[i + j] = out[i + j] + fi * gj
    return out


def vadd(f, g):
    return {k: f[k] + g[k] for k in f}


def vsub(f, g):
    return {k: f[k] - g[k] for k in f}


def vscale(f, s):
    return {k: f[k] * s for k in f}


def vshift(f, da, dj):
    return {k: f[k].shift(da, dj) for k in f}


def op_of(c, keep):
    """the operator f -> [c f] on odd vectors, as a matrix M[mu][mu']"""
    return {m: {mp_: (c[m - mp_] if (m - mp_) in c else Ser({}, keep)) for mp_ in ODD} for m in ODD}


def apply(Mx, f):
    return {m: sum((Mx[m][mp_] * f[mp_] for mp_ in ODD), Ser({}, f[ODD[0]].keep)) for m in ODD}


def matmul(X, Y):
    keep = X[ODD[0]][ODD[0]].keep
    return {m: {q: sum((X[m][p] * Y[p][q] for p in ODD), Ser({}, keep)) for q in ODD} for m in ODD}


def matinv(X):
    a, b, c, d = X[1][1], X[1][-1], X[-1][1], X[-1][-1]
    di = (a * d - b * c).inv()
    return {1: {1: d * di, -1: -b * di}, -1: {1: -c * di, -1: a * di}}


# ---------------------------------------------------------------------------------------------------- the system
class Model:
    def __init__(self, A, wf, keep):
        self.A, self.keep = A, keep
        self.w = wf * WINF
        self.F = {1: A / 2, -1: A / 2}                    # B8's convention F_{+-1} = A/2
        self.sgn = -1 if MUT == "sign" else 1             # K3: the sign of the x^2 Omega term of A, flipped

    def Om(self, f, sgn=1):
        return {m: f[m] * (iv.mpc(0, 1) * m * self.w * sgn) for m in ODD}

    def eps(self, f):
        return vshift(f, 1, 0) if isinstance(f, dict) else f.shift(1, 0)

    def parts(self, Z):
        V, S, beta, c = Z
        keep = self.keep
        C = op_of(c, keep)
        Ci = matinv(C)
        Y = vadd(V, S)
        P = apply(Ci, Y)
        cb = pr(c, beta, EVEN, keep)
        return V, S, beta, c, C, Ci, Y, P, cb

    def G(self, Z, sgn=1):
        """G of the S-equation; sgn = -1 flips the x^2 Omega term"""
        V, S, beta, c, C, Ci, Y, P, cb = self.parts(Z)
        CCS = apply(C, apply(C, S))
        return vadd(vsub(CCS, self.eps(CCS)), self.G_rest(Z, sgn))          # (1 - eps) C^2 S + the rest

    def G_rest(self, Z, sgn=1):
        """G minus its (1 - eps) C^2 S term, computed directly (so that interval widths do not add up)"""
        V, S, beta, c, C, Ci, Y, P, cb = self.parts(Z)
        keep = self.keep
        BS = pr(cb, S, ODD, keep)
        g = self.eps(apply(C, BS))                                          # eps C B S
        g = vsub(g, vshift(self.Om(S, sgn), 0, 1))                          # - z Omega S
        g = vsub(g, vshift(Y, 0, 1))                                        # - z Y
        g = vadd(g, vshift(pr(cb, P, ODD, keep), 0, 1))                     # + z B P
        g = vsub(g, vshift(apply(C, self.Om(P)), 0, 1))                     # - z C Omega P
        return g

    def residuals(self, Z):
        """the four equations' residuals (left minus right), each a series"""
        V, S, beta, c, C, Ci, Y, P, cb = self.parts(Z)
        keep = self.keep
        rV = vsub({m: V[m].zdz() for m in ODD}, self.Om(S))
        rc = vsub({k: c[k].zdz() for k in EVEN}, cb)
        CC = matmul(C, C)
        zdzS = {m: S[m].zdz() for m in ODD}
        lhs = vsub(vshift(zdzS, 0, 1), self.eps(apply(CC, zdzS)))
        rS = vsub(lhs, self.G(Z))
        SS, PP = pr(S, S, EVEN, keep), pr(P, P, EVEN, keep)
        lhsb = vsub(beta, self.eps(beta))
        rhs = vscale(vadd(self.eps(SS), vshift(PP, 0, 1)), iv.mpf(1) / 2)
        rhs = vsub(rhs, self.eps(rhs))
        rhs = vadd(rhs, vscale(vshift(pr(beta, SS, EVEN, keep), 2, 0), iv.mpf(1) / 2))
        rhs = vadd(rhs, vscale(vshift(pr(beta, PP, EVEN, keep), 1, 1), iv.mpf(1) / 2))
        rhs = vsub(rhs, self.eps(pr(beta, beta, EVEN, keep)))
        rhs = vsub(rhs, self.eps({k: beta[k].zdz() for k in EVEN}))
        rb = vsub(lhsb, rhs)
        return dict(V=rV, S=rS, beta=rb, c=rc)

    def solve(self, gauge="lightcone", iters=None):
        """Picard iteration of the system on the truncation set; each step fixes one more grade (a + j)"""
        keep = self.keep
        V = {m: Ser.const(2 * self.F[m], keep) for m in ODD}
        S = svec(ODD, keep)
        beta = svec(EVEN, keep)
        c = {k: Ser.const(1 if k == 0 else 0, keep) for k in EVEN}
        n_it = iters or 40
        for _ in range(n_it):
            Z = (V, S, beta, c)
            V_, S_, beta_, c_, C, Ci, Y, P, cb = self.parts(Z)
            # c: 2 z c_z = [c beta], constants by the gauge
            cn = {k: cb[k].zint() for k in EVEN}
            for k in EVEN:
                base = {(0, 0): iv.mpc(1 if k == 0 else 0)}
                if gauge == "lightcone":
                    # c(z = eps) = 1 order by order: c_{a,0} = - sum_{j >= 1} c_{a-j, j}
                    for a in range(1, 64):
                        if not keep(a, 0):
                            break
                        s = ZERO
                        for (aa, jj), v in cn[k].c.items():
                            if aa + jj == a and jj >= 1:
                                s += v
                        base[(a, 0)] = -s
                cn[k] = cn[k] + Ser({kk: v for kk, v in base.items() if keep(*kk)}, keep)
            # V: 2 z V_z = Omega S, V(0) = [c(0) 2F]
            c0 = {k: cn[k].at_z0() for k in EVEN}
            F2 = {m: Ser.const(2 * self.F[m], keep) for m in ODD}
            Vn = vadd(pr(c0, F2, ODD, keep), {m: v.zint() for m, v in self.Om(S).items()})
            # S: (1 - eps) C^2 S = (z - eps C^2) 2 z S_z - [G without its (1 - eps) C^2 S term]
            CC = matmul(C, C)
            rest = self.G_rest(Z, self.sgn)                               # G - (1 - eps) C^2 S
            zdzS = {m: S[m].zdz() for m in ODD}
            lhs = vsub(vshift(zdzS, 0, 1), self.eps(apply(CC, zdzS)))
            one_m_eps_inv = (Ser.const(1, keep) - Ser.const(1, keep).shift(1, 0)).inv()
            Sn = apply(matinv(CC), vscale(vsub(lhs, rest), one_m_eps_inv))
            # beta
            SS, PP = pr(S, S, EVEN, keep), pr(P, P, EVEN, keep)
            t = vscale(vshift(pr(beta, SS, EVEN, keep), 2, 0), iv.mpf(1) / 2)
            t = vadd(t, vscale(vshift(pr(beta, PP, EVEN, keep), 1, 1), iv.mpf(1) / 2))
            t = vsub(t, self.eps(pr(beta, beta, EVEN, keep)))
            t = vsub(t, self.eps({k: beta[k].zdz() for k in EVEN}))
            bn = vadd(vscale(vadd(self.eps(SS), vshift(PP, 0, 1)), iv.mpf(1) / 2), vscale(t, one_m_eps_inv))
            V, S, beta, c = Vn, Sn, bn, cn
        return (V, S, beta, c)

    def convert(self, Z, keep2):
        """B5's variables (P, Phi, C, theta) from Z (any truncation), as series on keep2"""
        V, S, beta, c = [{k: Ser(dict(s.c), keep2) for k, s in part.items()} for part in Z]
        C = op_of(c, keep2)
        Ci = matinv(C)
        Y = vadd(V, S)
        P = apply(Ci, Y)
        Phi = {m: P[m].zdz() for m in ODD}
        one_m_eps_inv = (Ser.const(1, keep2) - Ser.const(1, keep2).shift(1, 0)).inv()
        Theta = {k: Ser({(a, j - 1): v for (a, j), v in beta[k].c.items() if keep2(a, j - 1)}, keep2) * one_m_eps_inv
                 for k in EVEN}
        theta = vsub(Theta, vscale(pr(P, P, EVEN, keep2), iv.mpf(1) / 2))
        for k in EVEN:                                   # beta = O(z): its z^0 terms must be (balls around) 0
            for (a, j), v in beta[k].c.items():
                if j == 0 and not (0 in v.real and 0 in v.imag):
                    raise RuntimeError(f"beta_{k} has a z^0 term at eps^{a}: {v}")
        return dict(P=P, Phi=Phi, C=c, theta=theta)


def keep_T(Kmax):
    return lambda a, j: a >= 0 and j >= 0 and a + j <= Kmax


def keep_inner(J):
    return lambda a, j: a == 0 and 0 <= j <= J


def keep_box(amax, jmax):
    return lambda a, j: 0 <= a <= amax and 0 <= j <= jmax


def params(centre):
    if centre:
        A, wf = iv.mpc(A_C), iv.mpc(W_C)
    else:
        A = iv.mpc(iv.mpf([A_C - BOXR, A_C + BOXR]), iv.mpf([-BOXR, BOXR]))
        wf = iv.mpc(iv.mpf([W_C - BOXR, W_C + BOXR]), iv.mpf([-BOXR, BOXR]))
    if MUT == "A":
        A = A * (1 + iv.mpf("1e-3"))                     # K1: the planted perturbation
    return A, wf


def mag(v):
    """upper bound of |v| (v an iv.mpc)"""
    return abs(v).b


def fmt(v, d=8):
    return mp.nstr(mp.mpc(v.real.mid, v.imag.mid), d)


# ------------------------------------------------------------------------------------- numbers in place of series
class Num:
    """a value standing in for a series evaluated at (eps, z): the same interface as Ser for G and convert"""
    __slots__ = ("v", "e", "z")

    def __init__(self, v, e, z):
        self.v, self.e, self.z = cx(v), e, z

    def _w(self, o):
        return o.v if isinstance(o, Num) else cx(o)

    def __add__(self, o):
        return Num(self.v + self._w(o), self.e, self.z)

    def __sub__(self, o):
        return Num(self.v - self._w(o), self.e, self.z)

    def __neg__(self):
        return Num(-self.v, self.e, self.z)

    def __mul__(self, o):
        return Num(self.v * self._w(o), self.e, self.z)

    __rmul__ = __mul__

    def __radd__(self, o):
        return Num(self.v + self._w(o), self.e, self.z)

    def shift(self, da, dj):
        return Num(self.v * self.e ** da * self.z ** dj, self.e, self.z)

    def inv(self):
        return Num(1 / self.v, self.e, self.z)


def ser_eval(s, e, z):
    out = ZERO
    for (a, j), v in s.c.items():
        out += v * e ** a * z ** j
    return out


class NumModel(Model):
    """Model.G and the conversion with every series replaced by its value at one (eps, z) box"""
    def __init__(self, A, wf, e, z):
        Model.__init__(self, A, wf, None)
        self.e, self.z = e, z

    def eps(self, f):
        return {k: v.shift(1, 0) for k, v in f.items()} if isinstance(f, dict) else f.shift(1, 0)


def numvec(f, e, z):
    return {k: Num(ser_eval(s, e, z), e, z) for k, s in f.items()}


# patch the helpers so that they also work on Num (the zero of a sum)
def _pr(f, g, out_keys, keep):
    first = next(iter(f.values()))
    zero = Num(0, first.e, first.z) if isinstance(first, Num) else Ser({}, keep)
    out = {k: zero for k in out_keys}
    for i, fi in f.items():
        for j, gj in g.items():
            if i + j in out:
                out[i + j] = out[i + j] + fi * gj
    return out


def _op_of(c, keep):
    first = next(iter(c.values()))
    zero = Num(0, first.e, first.z) if isinstance(first, Num) else Ser({}, keep)
    return {m: {mp_: (c[m - mp_] if (m - mp_) in c else zero) for mp_ in ODD} for m in ODD}


def _apply(Mx, f):
    first = f[ODD[0]]
    zero = Num(0, first.e, first.z) if isinstance(first, Num) else Ser({}, first.keep)
    return {m: sum((Mx[m][mp_] * f[mp_] for mp_ in ODD), zero) for m in ODD}


def _matmul(X, Y):
    first = X[ODD[0]][ODD[0]]
    zero = Num(0, first.e, first.z) if isinstance(first, Num) else Ser({}, first.keep)
    return {m: {q: sum((X[m][p] * Y[p][q] for p in ODD), zero) for q in ODD} for m in ODD}


pr, op_of, apply, matmul = _pr, _op_of, _apply, _matmul


def div_z(s):
    """f/z for a series with no z^0 terms"""
    for (a, j), v in s.c.items():
        if j == 0 and not (0 in v.real and 0 in v.imag):
            raise RuntimeError("div_z: z^0 term")
    return Ser({(a, j - 1): v for (a, j), v in s.c.items() if j >= 1}, s.keep)


def num_convert(M, ZK, e, z):
    """(P, theta) of Z^K at one (eps, z) box, by interval arithmetic"""
    V, S, beta, c = [numvec(part, e, z) for part in ZK]
    bz = {k: Num(ser_eval(div_z(ZK[2][k]), e, z), e, z) for k in EVEN}
    C = op_of(c, None)
    Ci = matinv(C)
    Y = vadd(V, S)
    P = apply(Ci, Y)
    inv1 = 1 / (1 - e)
    Theta = {k: bz[k] * inv1 for k in EVEN}
    theta = vsub(Theta, vscale(pr(P, P, EVEN, None), iv.mpf(1) / 2))
    return P, theta


def num_rhoS(M, ZK, e, z):
    """rho_S = (z - eps C^2) 2 z S_z - G at Z^K, at one (eps, z) box"""
    V, S, beta, c = [numvec(part, e, z) for part in ZK]
    dS = {m: Num(ser_eval(ZK[1][m].zdz(), e, z), e, z) for m in ODD}
    Zn = (V, S, beta, c)
    C = op_of(c, None)
    CC = matmul(C, C)
    lhs = {m: dS[m].shift(0, 1) - apply(CC, dS)[m].shift(1, 0) for m in ODD}
    G = M.G(Zn)
    return {m: lhs[m] - G[m] for m in ODD}


# ------------------------------------------------------------------------------------------------ the n-scalings
TAU0 = iv.exp(-L0 / 10)                       # tau = n^(-1/10) <= tau0 on n >= n0: eps = tau^10, |z| = tau^4
N0 = iv.exp(L0)


class Maj:
    """an upper bound tau^p * m(tau) with m increasing; stored as (p, M = m(tau0)) (M an iv upper end)"""
    __slots__ = ("p", "M")

    def __init__(self, p, M):
        self.p, self.M = p, iv.mpf(M).b if not isinstance(M, (int, float)) else iv.mpf(M).b

    def __add__(self, o):
        p = min(self.p, o.p)
        return Maj(p, (iv.mpf(self.M) * TAU0 ** (self.p - p) + iv.mpf(o.M) * TAU0 ** (o.p - p)).b)

    def __mul__(self, o):
        if isinstance(o, Maj):
            return Maj(self.p + o.p, (iv.mpf(self.M) * iv.mpf(o.M)).b)
        return Maj(self.p, (iv.mpf(self.M) * iv.mpf(o)).b)

    __rmul__ = __mul__

    def sh(self, k):
        return Maj(self.p + k, self.M)

    def inv1m(self):
        """bound of 1/(1 - x) for x <= this (p >= 1)"""
        assert self.p >= 1
        x = iv.mpf(self.M) * TAU0 ** self.p
        assert x.b < 1
        return Maj(0, (1 / (1 - x)).b)

    def value(self):
        """the bound at every n >= n0 (p >= 0 required)"""
        if self.p < 0:
            raise RuntimeError(f"a bound grows with n (tau power {self.p})")
        return (iv.mpf(self.M) * TAU0 ** self.p).b


def maj_ser(s, extra=None):
    """majorant of |s| on the arc: sum |c_{a,j}| tau^(10 a + 4 j)"""
    terms = [(10 * a + 4 * j, mag(v)) for (a, j), v in s.c.items() if mag(v) > 0]
    if extra is not None:
        terms += [(extra.p, extra.M)]
    if not terms:
        return Maj(99, 0)
    p = min(t[0] for t in terms)
    tot = iv.mpf(0)
    for q, m in terms:
        tot += iv.mpf(m) * TAU0 ** (q - p)
    return Maj(p, tot.b)


def maj_vec(f, extra=None):
    out = None
    for s in f.values():
        m = maj_ser(s)
        out = m if out is None else out + m
    return out if extra is None else out + extra


def centre_run():
    A, wf = params(True)
    t0 = time.time()
    M = Model(A, wf, keep_T(K))
    Z = M.solve(iters=20)
    big = keep_box(6, 10)
    Mb = Model(A, wf, big)
    Zb = tuple({k: Ser(dict(s.c), big) for k, s in part.items()} for part in Z)
    res = Mb.residuals(Zb)
    worst_in, lowest = 0.0, {}
    for name, r in res.items():
        for kk, s in r.items():
            for (a, j), v in s.c.items():
                m = float(mag(v))
                if a + j <= K:
                    worst_in = max(worst_in, m)
                elif m > 1e-40:
                    lowest[name] = min(lowest.get(name, 99), a + j)
    print(f"series Z^K (K = {K}) at the box centre" + (f", mutate {MUT}" if MUT else "")
          + f": largest residual coefficient on a + j <= K {worst_in:.2e}; "
          f"lowest grade a + j of a nonzero residual (V, S, beta, c): "
          + ", ".join(f"{k} {lowest.get(k, '-')}" for k in ("V", "S", "beta", "c")) + f" [{time.time() - t0:.1f} s]")
    ok_res = worst_in < 1e-60 and all(lowest.get(k, 99) == K + 1 for k in ("S", "beta", "c"))
    print(f"  (P1) residual orders (zero on a + j <= {K}, nonzero from grade {K + 1}): {'PASS' if ok_res else 'FAIL'}")
    T = M.convert(Z, keep_box(3, 6))
    lead = dict(Phi={m: T["Phi"][m].coef(1, 1) for m in ODD}, C={k: T["C"][k].coef(1, 0) for k in EVEN},
                theta={k: T["theta"][k].coef(1, 0) for k in EVEN}, P={m: T["P"][m].coef(1, 1) for m in ODD},
                P10={m: T["P"][m].coef(1, 0) for m in ODD})
    for nm, d in lead.items():
        print(f"  {nm}: " + ", ".join(f"[{k}] {fmt(v, 10)} (|.| {mp.nstr(abs(mp.mpc(v.real.mid, v.imag.mid)), 8)})"
                                       for k, v in d.items()))
    # (P1) against B8_constants.md 8.2's printed leading coefficients, to all their digits
    printed = dict(Phi={1: ("-15.714", "74.930"), -1: ("-15.714", "-74.930")},
                   C={0: ("-3.2313", "0"), 2: ("-1.6156", "0"), -2: ("-1.6156", "0")},
                   theta={0: ("-6.4625", "0"), 2: ("-3.2313", "0"), -2: ("-3.2313", "0")})
    ok_dig = True
    for nm, d in printed.items():
        for k, (re_s, im_s) in d.items():
            v = lead[nm][k]
            for part, s in ((v.real.mid.a, re_s), (v.imag.mid.a, im_s)):
                nd = len(s.split(".")[1]) if "." in s else 0
                mine = mp.nstr(mp.mpf(part), 40, min_fixed=-1, max_fixed=40)
                r = round(float(mp.mpf(part)), nd)
                ok_dig &= abs(r - float(s)) < 0.5 * 10 ** -nd
    print(f"  (P1) leading coefficients against B8_constants.md 8.2's printed digits: {'PASS' if ok_dig else 'FAIL'}")
    # (P1) the inner series against Theorem A's series_arb.py (not B8's code): z = xi^2, harmonic mu times
    # e^(-i mu pi/2)
    from flint import acb, arb, ctx
    ctx.prec = 256                                        # before any arb constant is made
    sys.path.insert(0, os.path.dirname(HERE))
    import series_arb as SA
    W0f = 2 * arb.pi() / arb("1.64389375")
    a_s = arb("2.5421480475") * (arb(1001) / 1000 if MUT == "A" else 1)
    sa = SA.SeriesArb({1: acb(0, -a_s / 2), -1: acb(0, a_s / 2)}, W0f * arb("0.7769940136"))
    dd, ss, gg = sa.series(31)
    Min = Model(A, wf, keep_inner(30))
    Zin = Min.solve(iters=70)
    Tin = Min.convert(Zin, keep_inner(30))
    rot = {-2: -1, 0: 1, 2: -1}
    worst, overlap = mp.mpf(0), True
    for j in range(31):
        pairs = [(Zin[3][k].coef(0, j) * rot[k], gg[j][k]) for k in EVEN]
        pairs += [(Tin["P"][k].coef(0, j) * (iv.mpc(0, -1) if k == 1 else iv.mpc(0, 1)), dd[j][k]) for k in ODD]
        for v, th in pairs:
            sc = max(mp.mpf(abs(th).mid().str(30, radius=False)), mp.mpf(10) ** -60)
            for pm, pt in ((v.real, th.real), (v.imag, th.imag)):
                tm, tr = mp.mpf(pt.mid().str(80, radius=False)), mp.mpf(pt.rad().str(20, radius=False))
                mm, mr = mp.mpf(pm.mid.a), mp.mpf(pm.delta.b) / 2
                overlap &= abs(mm - tm) <= mr + tr
                worst = max(worst, abs(mm - tm) / sc)
    ok_sa = overlap and worst < mp.mpf(10) ** -40
    print(f"  (P1) inner series, c and P to z^30, against series_arb.py: largest relative difference "
          f"{mp.nstr(worst, 3)}, balls overlap {overlap}: {'PASS' if ok_sa else 'FAIL'}")
    return ok_res and ok_dig and ok_sa, lead, Z, M


HERE = os.path.dirname(os.path.abspath(__file__))
S_EXP = dict(P=1, Phi=iv.mpf("1.4"), C=1, theta=1)             # n^s: s = 1 for P, C, theta; 2 - 2 gamma for Phi
TAUP = dict(P=10, Phi=14, C=10, theta=10)                      # the same as tau powers


def h_inputs(d0):
    name = "b8m_scan_K4_box.log" if d0 == "0" else f"b8m_scan_K4_d0_{d0}.log"
    for line in open(os.path.join(os.path.dirname(HERE), name)):
        r = json.loads(line)
        if abs(r["L0"] - 73.89) < 1e-9 and str(r["d0"]) == d0:
            h = r["hyp"]
            return dict(V=h["V"], S=h["S"], b=h["b"], c=h["c"]), name
    raise SystemExit(f"no ln n0 = 73.89 record for d0 = {d0} in {name}")


def contains0(v):
    return 0 in v.real and 0 in v.imag


def tail_series_maj(Mv, re, rz, amax, jmax, deriv=False):
    """Cauchy tail of a function bounded by Mv on |eps| <= re, |z| <= rz: sum over a > amax or j > jmax of
    Mv (|eps|/re)^a (|z|/rz)^j (times 2 j if deriv), as a Maj in tau"""
    qe0, qz0 = TAU0 ** 10 / re, TAU0 ** 4 / rz
    fa = 1 / ((1 - qe0) * (1 - qz0))
    if deriv:
        m1 = Mv * re ** -(amax + 1) * 2 / ((1 - qe0) * (1 - qz0) ** 2)
        m2 = Mv * rz ** -(jmax + 1) * 2 * ((jmax + 1) / (1 - qz0) + 1 / (1 - qz0) ** 2) / (1 - qe0)
    else:
        m1 = Mv * re ** -(amax + 1) * fa
        m2 = Mv * rz ** -(jmax + 1) * fa
    return Maj(10 * (amax + 1), m1.b) + Maj(4 * (jmax + 1), m2.b)


def polydisc(r):
    return iv.mpc(iv.mpf([-r, r]), iv.mpf([-r, r]))


def main_run(d0):
    out = {}
    t0 = time.time()
    A, wf = params("point" in OPTS)              # --point: the same run at the box centre (thin balls)
    om = mag(wf * WINF)
    # --- the series over the box
    M = Model(A, wf, keep_T(K))
    ZK = M.solve(iters=20)
    U = keep_box(6, 30)
    MU = Model(A, wf, U)
    ZU = tuple({k: Ser(dict(s.c), U) for k, s in part.items()} for part in ZK)
    T = MU.convert(ZU, U)
    Min = Model(A, wf, keep_inner(30))
    Zin = Min.solve(iters=70)                    # each grade takes about two passes (c lags beta)
    Tin = Min.convert(Zin, keep_inner(30))
    # Lemma F.3 at work: Z^K's eps^0 part is the inner series up to z^K
    agree = all(contains0(ZK[i][k].coef(0, j) - Zin[i][k].coef(0, j))
                for i in range(4) for k in ZK[i] for j in range(K + 1))
    print(f"series over the box: Z^K (K = {K}), its conversion on a <= 6, j <= 30, the inner series to z^30; "
          f"Z^K's eps^0 part equals the inner series to z^{K}: {agree} [{time.time() - t0:.0f} s]", flush=True)
    th_in = max(float(mag(v)) for k in EVEN for v in Tin["theta"][k].c.values())
    print(f"  the inner theta (identically 0): largest coefficient ball {th_in:.1e}")
    # --- T - Z_in, monomial by monomial
    re_, rz_ = iv.mpf("1e-3"), iv.mpf("1e-3")
    E, Zd = polydisc(re_), polydisc(rz_)
    Pn, thn = num_convert(M, ZK, E, Zd)
    MP = max((mag(Pn[m].v) for m in ODD), key=float)
    Mth = max((mag(thn[k].v) for k in EVEN), key=float)
    tails = dict(P=tail_series_maj(iv.mpf(MP), re_, rz_, 6, 30), Phi=tail_series_maj(iv.mpf(MP), re_, rz_, 6, 30, True),
                 C=Maj(99, 0), theta=tail_series_maj(iv.mpf(Mth), re_, rz_, 6, 30))
    zcut = dict(P=K, Phi=K, C=K, theta=K - 1)
    Tpart, lead, removed = {}, {}, {}
    for comp in ("P", "Phi", "C", "theta"):
        best = None
        rem = 0.0
        for k, s in T[comp].items():
            tot = None
            for (a, j), v in s.c.items():
                d = v - (Tin[comp][k].coef(0, j) if a == 0 else ZERO)
                structural = (a == 0 and j <= zcut[comp]) or (comp == "P" and a >= 1 and j == 0)
                if structural:
                    if not contains0(d):
                        raise SystemExit(f"{comp}[{k}] eps^{a} z^{j}: a structural zero is not a ball around 0: {d}")
                    rem = max(rem, float(mag(d)))
                    continue
                if mag(d) == 0:
                    continue
                e = 10 * a + 4 * j - TAUP[comp]
                if e < 0:
                    raise SystemExit(f"{comp}[{k}] eps^{a} z^{j} grows with n")
                m = Maj(10 * a + 4 * j, mag(d)).sh(-TAUP[comp])
                tot = m if tot is None else tot + m
                if (comp == "Phi" and (a, j) == (1, 1)) or (comp in ("C", "theta") and (a, j) == (1, 0)):
                    lead.setdefault(comp, {})[k] = d
            v = (tot + tails[comp].sh(-TAUP[comp])).value() if tot is not None else tails[comp].sh(-TAUP[comp]).value()
            best = v if best is None or v > best else best
        Tpart[comp] = best
        removed[comp] = rem
    # --- eta (lemma S's tail, B8_constants.md 8.2 (eta)): 5.5e-7 (|z|/0.0085)^12 (p, v), ^13 (c); Phi a factor 100
    Zs = iv.mpf("0.0085")
    eta = dict(P=(iv.mpf("5.5e-7") * (TAU0 ** 4 / Zs) ** 12 * TAU0 ** -10).b,
               Phi=(100 * iv.mpf("5.5e-7") * (TAU0 ** 4 / Zs) ** 12 * TAU0 ** -14).b,
               C=(iv.mpf("5.5e-7") * (TAU0 ** 4 / Zs) ** 13 * TAU0 ** -10).b, theta=iv.mpf(0).b)
    # --- R from h
    h, src = h_inputs(d0)
    if MUT == "hS":
        h["S"] *= 1 + 1e-3                                  # K2
    one = Maj(0, 1)
    tau3 = 1 + TAU0 ** 3
    w1, w = Maj(20, (tau3 ** 10).b), Maj(20, (tau3 ** 9).b)
    RV, RS, Rbh, Rc = w1 * h["V"], w * h["S"], w * h["b"], w1 * h["c"]
    ep, zz, x2, ix2 = Maj(10, 1), Maj(4, 1), Maj(-6, 1), Maj(6, 1)
    inv1e = ep.inv1m()
    Vm, Sm, bm, cm = (maj_vec(ZK[i]) for i in range(4))
    cm1 = maj_vec({k: ZK[3][k] - (1 if k == 0 else 0) for k in EVEN})
    Y = vadd(ZK[0], ZK[1])
    Ym, dYm, dSm = maj_vec(Y), maj_vec({m: Y[m].zdz() for m in ODD}), maj_vec({m: ZK[1][m].zdz() for m in ODD})
    W8 = keep_T(2 * K)
    c8 = {k: Ser(dict(s.c), W8) for k, s in ZK[3].items()}
    b8 = {k: Ser(dict(s.c), W8) for k, s in ZK[2].items()}
    cb8 = pr(c8, b8, EVEN, W8)
    rho4 = {k: c8[k].zdz() - cb8[k] for k in EVEN}
    for k in EVEN:
        for (a, j), v in list(rho4[k].c.items()):
            if a + j <= K:
                if not contains0(v):
                    raise SystemExit(f"rho_4 does not vanish at eps^{a} z^{j}")
                del rho4[k].c[(a, j)]
    cbm, rho4m = maj_vec(cb8), maj_vec(rho4)
    CKn, CKi = one + cm1, cm1.inv1m()
    PKm = CKi * Ym
    xPKm = CKi * (dYm + (cbm + rho4m) * PKm)
    # rho_S on U, its structural zeros on a + j <= K, and its Cauchy tail
    GU = MU.G(ZU)
    CU = op_of(ZU[3], U)
    CCU = matmul(CU, CU)
    dSU = {m: ZU[1][m].zdz() for m in ODD}
    rhoS = {m: dSU[m].shift(0, 1) - apply(CCU, dSU)[m].shift(1, 0) - GU[m] for m in ODD}
    for m in ODD:
        for (a, j), v in list(rhoS[m].c.items()):
            if a + j <= K:
                if not contains0(v):
                    raise SystemExit(f"rho_S does not vanish at eps^{a} z^{j}: {v}")
                del rhoS[m].c[(a, j)]
    rn = num_rhoS(NumModel(A, wf, E, Zd), ZK, E, Zd)
    Mrho = max((mag(rn[m].v) for m in ODD), key=float)
    rhoSm = maj_vec(rhoS) + tail_series_maj(iv.mpf(Mrho), re_, rz_, 6, 30)
    # the full solution's quantities
    Cn = CKn + Rc
    Ci = (cm1 + Rc).inv1m()
    bl = (one + bm * ep * inv1e)                               # |b^K|_1, b^K = 1 + beta^K/(n - 1)
    ell = (one + ep) * bl * (Sm + x2 * PKm * CKi)
    BK = one + cbm * CKi + CKn * om * CKi
    e_op = (CKn * cbm * ix2 + Maj(0, om) + BK) * ep * inv1e
    q = ell * e_op.inv1m() * (one + CKn * CKn * ix2) * ep * inv1e
    Rb = Rbh + q * RS
    WP = Ci * (RV + RS + Rc * PKm)
    WC = Rc
    Wth = Rb * Maj(-4, 1) * inv1e + WP * (PKm * 2 + WP) * iv.mpf("0.5")
    bfull = bm + Rb
    gdel = Rc * bfull + cm * Rb
    Bfull = cbm + gdel
    xRc = Rc * bfull + cm * Rb + rho4m
    Dei = Maj(-4, 1) * (Cn * Cn * ix2).inv1m()
    DeKi = Maj(-4, 1) * (CKn * CKn * ix2).inv1m()
    rS = DeKi * rhoSm
    PsiK = dSm + rS
    dD = ep * Rc * (Cn + CKn)
    GmG = ((one + ep) * (Cn * Cn * RS + Rc * (Cn + CKn) * Sm) + ep * (Cn * Bfull * RS + Cn * gdel * Sm + Rc * cbm * Sm)
           + zz * om * RS + zz * (RV + RS) + zz * (Bfull * WP + gdel * PKm) + zz * (Cn * om * WP + Rc * om * PKm))
    xRS = Dei * (GmG + dD * PsiK) + rS
    xU = RS * om + xRS + xRc * PKm + Rc * xPKm
    WPhi = Ci * (xU + Bfull * WP)
    Rpart = dict(P=WP.sh(-10).value(), Phi=WPhi.sh(-14).value(), C=WC.sh(-10).value(), theta=Wth.sh(-10).value())
    split = dict(xRS_main=(Dei * (Cn * Cn * RS)).sh(-14).value(), rS=rS.sh(-14).value(),
                 rhoS_lead=maj_vec(rhoS).p, q=q.value(), Ci=Ci.value())
    delta = {c: (iv.mpf(Tpart[c]) + iv.mpf(Rpart[c]) + iv.mpf(eta[c])).b for c in Tpart}
    out = dict(d0=d0, mutate=MUT, h=h, h_source=src, T=Tpart, R=Rpart, eta=eta, delta=delta, removed=removed,
               lead={c: {str(k): [mp.nstr(mp.mpf(v.real.mid.a), 10), mp.nstr(mp.mpf(v.imag.mid.a), 10),
                                  mp.nstr(mp.mpf((v.real.delta / 2).b), 2), mp.nstr(mp.mpf((v.imag.delta / 2).b), 2)]
                         for k, v in d.items()} for c, d in lead.items()},
               split=split, tails={c: tails[c].sh(-TAUP[c]).value() for c in tails}, MP=MP, Mth=Mth, Mrho=Mrho,
               sec=time.time() - t0)
    return out


def fl(v):
    return float(iv.mpf(v).b)


if __name__ == "__main__":
    if "centre" in OPTS:
        ok, lead, Z, M = centre_run()
        print("(P1) series checks:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    d0 = OPTS.get("d0", "0")
    r = main_run(d0)
    print(f"(H8) on the arc |x| = n^0.3 for every n >= e^73.89 over the box, d0 = {d0}"
          + (f", mutate {MUT}" if MUT else "") + f" (h from {r['h_source']}) [{r['sec']:.0f} s]")
    print(f"{'':6s} {'series (T)':>12s} {'remainder (R)':>14s} {'eta':>10s} {'delta':>12s}   delta_ref")
    for c, ref in (("P", 1), ("Phi", 200), ("C", 10), ("theta", 20)):
        print(f"{c:6s} {fl(r['T'][c]):12.6g} {fl(r['R'][c]):14.4g} {fl(r['eta'][c]):10.2g} "
              f"{fl(r['delta'][c]):12.6g}   {ref}")
    for c, d in r["lead"].items():
        print(f"  leading {c}{' at the centre' if 'point' in OPTS else ' over the box'}: "
              + ", ".join(f"[{k}] {v[0]} (+-{v[2]}) + i {v[1]} (+-{v[3]})" for k, v in d.items()))
    print("  structural zeros removed (largest ball): " + ", ".join(f"{c} {v:.1e}" for c, v in r["removed"].items()))
    print("  Phi's remainder: main term D^-1 C^2 R_S " + f"{fl(r['split']['xRS_main']):.4g}, the series' residual r_S "
          f"{fl(r['split']['rS']):.4g} (rho_S from tau^{r['split']['rhoS_lead']}); |q| {fl(r['split']['q']):.2e}")
    ok = all(fl(r["delta"][c]) <= ref for c, ref in (("P", 1), ("Phi", 200), ("C", 10), ("theta", 20)))
    print("(P2) delta <= delta_ref:", "PASS" if ok else "FAIL")
    if "out" in OPTS:
        os.makedirs(OPTS["out"], exist_ok=True)
        tag = f"h8_d0_{d0}" + ("_point" if "point" in OPTS else "") + (f"_mut_{MUT}" if MUT else "")
        with open(os.path.join(OPTS["out"], f"p2c_{tag}.json"), "w") as fh:
            json.dump({k: (str(v) if not isinstance(v, (dict, list, str, float, int, type(None))) else v)
                       for k, v in r.items()}, fh, indent=1, default=str)
    sys.exit(0 if ok else 1)
