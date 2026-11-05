#!/usr/bin/env python3
"""P2b review: the truncated formal series of the M = 1 model and every known quantity of Lemma R, as polynomials in
(eps, x). The series Z_k (k <= K) is computed EXACTLY, with coefficients that are polynomials in (A, omega) over the
Gaussian rationals Q(i); its coefficients are then enclosed over B8's complex parameter box with mpmath interval
arithmetic (iv.mpc), and the residuals and operators are formed in that interval arithmetic. The low eps-orders of
the residuals, which vanish identically by Lemma F, are recomputed EXACTLY (exact ring, products truncated in eps)
and verified to be zero; only then are those entries dropped from the interval results.

Shared with other code (recorded as AGENTS.md asks):
  - the order-by-order recursion is the one of solver/proof/review_p2/p2_formal.py (P2's derivation from the ORIGINAL
    p, m, b, c equations of proof/B8_constants.md 1.1), re-typed here for an exact coefficient ring; p2_formal works
    in 60-digit complex floats at the box centre only. Nothing is imported from solver/proof.
  - the box: A, omega/omega_inf in xhat + [-2e-6, 2e-6] + i[-2e-6, 2e-6] (B_chain.md 13.1, "within 2e-6 (1 + i)"),
    xhat = (2.5421480475, 0.7769940136), omega_inf = 2 pi / 1.64389375 (A_certify.md).
The parity and the vanishing orders at x = 0 are exact (absent dictionary keys); the residuals' low eps-orders are
verified exactly as said above.

Quantities (B8_constants.md 1.2, 4, 5; B_chain.md 13.3). Odd vectors: V, S, S/x^2, Y, Dl (the series Delta^K),
tauD, tauv, det*rho2. Even vectors: beta, beta/x^2, c - 1, cb = [c beta], cb/x^2, taub, rho4. Operators on odd
vectors (2 x 2, rows/columns mu = -1, +1): C, Gamma = (C^2 - I)/(x^2 - 1), det C - 1, det C * eta~, adj C.

    python3 p2b_series.py            # builds p2b_series.json (coefficient bounds over the box) and prints checks
    python3 p2b_series.py --centre   # the box of radius 0 (for comparison with p2_formal / b8m_atoms_K4.log)
"""
import json
import os
import sys
from fractions import Fraction as Fr

from mpmath import iv, mp

HERE = os.path.dirname(os.path.abspath(__file__))
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
ODD, EVEN = (-1, 1), (-2, 0, 2)
K = 4


# ---------------------------------------------------------------- the exact ring Q(i)[A, W] (W = omega)
class R:
    """polynomial in (A, W) with Gaussian-rational coefficients: {(i, j): (re, im)}"""
    __slots__ = ("t",)

    def __init__(self, t=None):
        self.t = t or {}

    @staticmethod
    def c(re, im=0):
        re, im = Fr(re), Fr(im)
        return R({(0, 0): (re, im)} if (re or im) else {})

    def iszero(self):
        return not self.t

    def __add__(self, o):
        if isinstance(o, (int, Fr)):
            o = R.c(o)
        out = dict(self.t)
        for k, (a, b) in o.t.items():
            if k in out:
                c, d = out[k]
                s = (a + c, b + d)
                if s[0] or s[1]:
                    out[k] = s
                else:
                    del out[k]
            else:
                out[k] = (a, b)
        return R(out)

    __radd__ = __add__

    def __neg__(self):
        return R({k: (-a, -b) for k, (a, b) in self.t.items()})

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, (int, Fr)):
            o = Fr(o)
            return R({k: (a * o, b * o) for k, (a, b) in self.t.items()}) if o else R()
        out = {}
        for (i1, j1), (a, b) in self.t.items():
            for (i2, j2), (c, d) in o.t.items():
                k = (i1 + i2, j1 + j2)
                re, im = a * c - b * d, a * d + b * c
                if k in out:
                    e, f = out[k]
                    out[k] = (e + re, f + im)
                else:
                    out[k] = (re, im)
        return R({k: v for k, v in out.items() if v[0] or v[1]})

    __rmul__ = __mul__

    def ev(self, A, W):
        """interval value at complex intervals A, W (iv.mpc)"""
        tot = iv.mpc(0)
        powA, powW = {}, {}
        for (i, j), (a, b) in self.t.items():
            if i not in powA:
                powA[i] = A ** i if i else iv.mpc(1)
            if j not in powW:
                powW[j] = W ** j if j else iv.mpc(1)
            co = iv.mpc(iv.mpf(a.numerator) / a.denominator, iv.mpf(b.numerator) / b.denominator)
            tot += co * powA[i] * powW[j]
        return tot


class IC:
    """an iv.mpc coefficient; iszero() only for the exact zero interval"""
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def iszero(self):
        v = self.v
        return v.real.a == 0 and v.real.b == 0 and v.imag.a == 0 and v.imag.b == 0

    def __add__(self, o):
        if isinstance(o, (int, Fr)):
            return IC(self.v + iv.mpf(o.numerator) / o.denominator) if isinstance(o, Fr) else IC(self.v + o)
        return IC(self.v + o.v)

    __radd__ = __add__

    def __neg__(self):
        return IC(-self.v)

    def __sub__(self, o):
        return self + (-o)

    def __mul__(self, o):
        if isinstance(o, Fr):
            return IC(self.v * (iv.mpf(o.numerator) / o.denominator))
        if isinstance(o, int):
            return IC(self.v * o)
        return IC(self.v * o.v)

    __rmul__ = __mul__


JMAX = None       # truncation of products in eps (exact low-order verification only)


ZERO = R()
ONE = R.c(1)
A_ = R({(1, 0): (Fr(1), Fr(0))})
W_ = R({(0, 1): (Fr(1), Fr(0))})
I_ = R.c(0, 1)


# ---------------------------------------------------------------- polynomials in x: {power: R}
def padd(*ps):
    out = {}
    for p in ps:
        for m, v in p.items():
            out[m] = out[m] + v if m in out else v
    return {m: v for m, v in out.items() if not v.iszero()}


def pscale(p, s):
    return {m: v * s for m, v in p.items() if not (v * s).iszero()}


def pmul(p, q):
    out = {}
    for a, u in p.items():
        for b, v in q.items():
            out[a + b] = out[a + b] + u * v if a + b in out else u * v
    return {m: v for m, v in out.items() if not v.iszero()}


def pder(p):
    return {m - 1: v * m for m, v in p.items() if m != 0}


def px(p, j=1):
    return {m + j: v for m, v in p.items()}


def vadd(*vs):
    out = {}
    for v in vs:
        for h, p in v.items():
            out[h] = padd(out.get(h, {}), p)
    return out


def vscale(v, s):
    return {h: pscale(p, s) for h, p in v.items()}


def proj(f, g, keep):
    out = {h: {} for h in keep}
    for a, p in f.items():
        for b, q in g.items():
            if a + b in out:
                out[a + b] = padd(out[a + b], pmul(p, q))
    return out


def vder(v):
    return {h: pder(p) for h, p in v.items()}


def vx(v, j=1):
    return {h: px(p, j) for h, p in v.items()}


def zero(keep):
    return {h: {} for h in keep}


def sprod(f, g, k, keep):
    out = zero(keep)
    for i in range(k + 1):
        if i < len(f) and k - i < len(g):
            out = vadd(out, proj(f[i], g[k - i], keep))
    return out


def formal(order, mutate=None):
    """the recursion of p2_formal.formal (P2), over the exact ring; F_{+-1} = A/2, Om = diag(i mu omega)"""
    F = {1: A_ * Fr(1, 2), -1: A_ * Fr(1, 2)}
    Om = {m: I_ * W_ * m for m in ODD}
    sgn = -1 if mutate == "sign" else 1
    S = [zero(ODD)]
    Dl = [{m: {1: F[m] * 2} for m in ODD}]
    be = [zero(EVEN)]
    c = [{0: {0: ONE}, 2: {}, -2: {}}]
    H = []
    for k in range(1, order + 1):
        cb = [sprod(c, be, j, EVEN) for j in range(k)]
        Sx = [vder(s) for s in S]
        t = vadd(vx(vder(Dl[k - 1]), 2), vscale(vx(sprod(c, Sx, k - 1, ODD)), -1),
                 {m: pscale(px(Dl[k - 1][m]), Om[m] * sgn) for m in ODD},
                 vscale(sprod(cb, S, k - 1, ODD), -1), sprod(c, S, k - 1, ODD))
        for j in range(1, k):
            t = vadd(t, vscale(proj(c[j], S[k - j], ODD), -1))
        S.append(t)
        H.append(vscale(vadd(sprod(S, S, k - 1, EVEN), sprod(Dl, Dl, k - 1, EVEN)), Fr(1, 2)))
        bk = vadd(be[k - 1], vscale(vx(vder(be[k - 1])), -1), vscale(sprod(be, be, k - 1, EVEN), -1), H[k - 1])
        if k >= 2:
            bk = vadd(bk, vscale(H[k - 2], -1), sprod(be, H, k - 2, EVEN))
        be.append(bk)
        cbk = sprod(c, be, k, EVEN)
        ck = {}
        for h in EVEN:
            p = cbk[h]
            if 0 in p:
                raise ValueError("[c beta] has a constant term")
            q = {m: v * Fr(1, m) for m, v in p.items()}
            s0 = ZERO
            for v in q.values():
                s0 = s0 + v
            if not s0.iszero():
                q[0] = -s0
            ck[h] = q
        c.append(ck)
        cbfull = [sprod(c, be, j, EVEN) for j in range(k + 1)]
        rhs = vadd(vscale(vx(vder(S[k]), 2), -1), {m: pscale(px(S[k][m]), -Om[m]) for m in ODD},
                   sprod(cbfull, Dl + [zero(ODD)], k, ODD))
        for j in range(1, k + 1):
            rhs = vadd(rhs, vscale(proj(c[j], Dl[k - j], ODD), -1), vx(proj(c[j], vder(Dl[k - j]), ODD)))
        dk = {}
        for m in ODD:
            if 1 in rhs[m]:
                raise ValueError(f"solvability: x^1 coefficient at order {k}")
            dk[m] = {p: v * Fr(1, 1 - p) for p, v in rhs[m].items()}
        Dl.append(dk)
    V = []
    for k in range(order + 1):
        cD = sprod(c, Dl, k, ODD)
        vk = {}
        for m in ODD:
            p = padd(cD[m], pscale(px(S[k][m]), -1))
            if 0 in p:
                raise ValueError("V not divisible by x")
            vk[m] = {e - 1: v for e, v in p.items()}
        V.append(vk)
    return dict(S=S, Dl=Dl, be=be, c=c, V=V, F=F, Om=Om)


# ---------------------------------------------------------------- bivariate (eps, x): {(j, m): R}
def b_add(*ps):
    out = {}
    for p in ps:
        for k, v in p.items():
            out[k] = out[k] + v if k in out else v
    return {k: v for k, v in out.items() if not v.iszero()}


def b_mul(p, q):
    out = {}
    for (j1, m1), u in p.items():
        for (j2, m2), v in q.items():
            if JMAX is not None and j1 + j2 > JMAX:
                continue
            k = (j1 + j2, m1 + m2)
            out[k] = out[k] + u * v if k in out else u * v
    return {k: v for k, v in out.items() if not v.iszero()}


def b_sc(p, s):
    return {k: v * s for k, v in p.items() if not (v * s).iszero()}


def b_xd(p):
    return {(j, m): v * m for (j, m), v in p.items() if m != 0}


def b_x(p, i=1):
    return {(j, m + i): v for (j, m), v in p.items()}


def b_eps(p, i=1):
    return {(j + i, m): v for (j, m), v in p.items()}


def b_divx(p, i=1):
    if any(m < i for (_, m) in p):
        raise ValueError("not divisible by x^%d" % i)
    return {(j, m - i): v for (j, m), v in p.items()}


def bv_add(*vs):
    out = {}
    for v in vs:
        for h, p in v.items():
            out[h] = b_add(out.get(h, {}), p)
    return out


def bv_sc(v, s):
    return {h: b_sc(p, s) for h, p in v.items()}


def bv_proj(f, g, keep):
    out = {h: {} for h in keep}
    for a, p in f.items():
        for b, q in g.items():
            if a + b in out:
                out[a + b] = b_add(out[a + b], b_mul(p, q))
    return out


def trunc(ser, Kt, keep):
    out = {h: {} for h in keep}
    for k in range(min(Kt + 1, len(ser))):
        for h, p in ser[k].items():
            for m, v in p.items():
                out[h] = b_add(out[h], {(k, m): v})
    return out


NM1 = {(-1, 0): ONE, (0, 0): -ONE}            # n - 1 = eps^-1 - 1


def mult_op(g):
    """f -> [g f] on odd vectors (g even): entry (mu, nu) = g_{mu - nu}"""
    return {(mu, nu): g.get(mu - nu, {}) for mu in ODD for nu in ODD}


def mat_mul(A, B):
    out = {}
    for mu in ODD:
        for nu in ODD:
            out[(mu, nu)] = b_add(*[b_mul(A[(mu, k)], B[(k, nu)]) for k in ODD])
    return out


def mat_add(*Ms):
    return {k: b_add(*[M[k] for M in Ms]) for k in Ms[0]}


def mat_sc(M, s):
    return {k: b_sc(p, s) for k, p in M.items()}


def mat_vec(M, v):
    return {mu: b_add(*[b_mul(M[(mu, nu)], v[nu]) for nu in ODD]) for mu in ODD}


def mat_scalar(M, p):
    return {k: b_mul(v, p) for k, v in M.items()}


def divide_x2m1(p):
    """exact division by x^2 - 1 for each eps-order; raises if the remainder is not exactly zero"""
    out = {}
    for j in sorted({j for j, _ in p}):
        a = {m: v for (jj, m), v in p.items() if jj == j}
        deg = max(a)
        for m in range(deg, 1, -1):
            qm = a.get(m, ZERO)
            if qm.iszero():
                continue
            out[(j, m - 2)] = qm
            a[m] = a[m] - qm
            a[m - 2] = a.get(m - 2, ZERO) + qm
        for v in a.values():
            if isinstance(v, IC):
                # the exact polynomial is divisible (c^K(+-1) = 1 exactly, checked in verify_low_orders), so the
                # remainder's enclosure must contain 0; the quotient's enclosure is that of the exact quotient
                if not (v.v.real.a <= 0 <= v.v.real.b and v.v.imag.a <= 0 <= v.v.imag.b):
                    raise ValueError("C^2 - I: interval remainder excludes 0")
            elif not v.iszero():
                raise ValueError("C^2 - I not divisible by x^2 - 1")
    return {k: v for k, v in out.items() if not v.iszero()}


def build(fs, conv, one, check_exact=False):
    """all known quantities from the series fs (exact), with coefficients converted by conv (exact: identity;
    intervals: R -> IC over the box); one = the unit of the coefficient type"""
    global JMAX
    S, Dl, be, c, V = (fs[k] for k in ("S", "Dl", "be", "c", "V"))

    def cv(v):
        return {h: {jm: conv(r) for jm, r in p.items()} for h, p in v.items()}

    St, Dt, bt, ct, Vt = (cv(trunc(s, K, keep)) for s, keep in ((S, ODD), (Dl, ODD), (be, EVEN), (c, EVEN),
                                                                  (V, ODD)))
    Om = {m: conv(fs["Om"][m]) for m in ODD}
    nm1 = {(-1, 0): one, (0, 0): -one}             # n - 1 = eps^-1 - 1
    Yt = bv_add(Vt, St)
    cm1 = {h: dict(p) for h, p in ct.items()}
    cm1[0] = b_add(cm1[0], {(0, 0): -one})
    cb = bv_proj(ct, bt, EVEN)
    # residuals of Z^K (B8_constants.md 4)
    tauD = bv_add(bv_proj(ct, {m: b_xd(p) for m, p in St.items()}, ODD),
                  {m: b_sc(b_x(b_xd(Dt[m])), -1) for m in ODD},
                  {m: b_mul(b_x(Dt[m]), {(0, 0): -Om[m]}) for m in ODD},
                  bv_proj(cb, St, ODD),
                  {m: b_mul(nm1, bv_proj(ct, St, ODD)[m]) for m in ODD})
    tauv = bv_add(bv_proj(ct, Dt, ODD), {m: b_sc(b_x(Yt[m]), -1) for m in ODD})
    Hb = bv_sc(bv_add(bv_proj(St, St, EVEN), bv_proj(Dt, Dt, EVEN)), Fr(1, 2))
    bH = bv_proj(bt, Hb, EVEN)
    taub = bv_add({h: b_xd(p) for h, p in bt.items()}, {h: b_mul(nm1, p) for h, p in bt.items()},
                  bv_proj(bt, bt, EVEN), bv_sc(Hb, -1), {h: b_eps(p) for h, p in Hb.items()},
                  {h: b_sc(b_eps(p), -1) for h, p in bH.items()})
    rho4 = bv_add({h: b_xd(p) for h, p in ct.items()}, bv_sc(cb, -1))
    out = dict(tauD=tauD, tauv=tauv, taub=taub, rho4=rho4)
    if check_exact:
        return out
    # the low eps-orders vanish identically (Lemma F, verified exactly by verify_low_orders before this is called)
    for nm, j0 in LOWEST.items():
        out[nm] = {h: {jm: v for jm, v in p.items() if jm[0] >= j0} for h, p in out[nm].items()}
    tauD, tauv, taub, rho4 = out["tauD"], out["tauv"], out["taub"], out["rho4"]
    # operators
    Cop = mult_op(ct)
    Iop = {(mu, nu): ({(0, 0): one} if mu == nu else {}) for mu in ODD for nu in ODD}
    C2mI = mat_add(mat_mul(Cop, Cop), mat_sc(Iop, -1))
    Gam = {k: divide_x2m1(p) for k, p in C2mI.items()}
    det = b_add(b_mul(Cop[(1, 1)], Cop[(-1, -1)]), b_sc(b_mul(Cop[(1, -1)], Cop[(-1, 1)]), -1))
    detm1 = b_add(det, {(0, 0): -one})
    adj = {(1, 1): Cop[(-1, -1)], (-1, -1): Cop[(1, 1)], (1, -1): b_sc(Cop[(1, -1)], -1),
           (-1, 1): b_sc(Cop[(-1, 1)], -1)}
    # eta = A^K - (n-1)(I - Gamma) = x^2 eta~ (B8_constants.md 5), derived here:
    #   eta~ = (n-1) Gamma + C Bb/x^2 - Om - I + Bb C^-1 - C Om C^-1,   Bb = [cb .];  det*eta~ is a polynomial
    cbx2 = {h: b_divx(p, 2) for h, p in cb.items()}
    Bb, Bbx2 = mult_op(cb), mult_op(cbx2)
    OmM = {(mu, nu): ({(0, 0): Om[mu]} if mu == nu else {}) for mu in ODD for nu in ODD}
    detI = {(mu, nu): (det if mu == nu else {}) for mu in ODD for nu in ODD}
    Neta = mat_add({k: b_mul(b_mul(nm1, det), v) for k, v in Gam.items()},
                   {k: b_mul(det, v) for k, v in mat_mul(Cop, Bbx2).items()},
                   mat_sc(mat_mul(detI, mat_add(OmM, Iop)), -1),
                   mat_mul(mat_add(Bb, mat_sc(mat_mul(Cop, OmM), -1)), adj))
    # det * rho2 (B8_constants.md 4, (I1)), with C^-1 = adj/det, all at Z^K:
    #   rho2 = x^2 M_rho4 C^-1 Y - C tauD - x^2 tauv' - x C Om C^-1 tauv + x M_{xc'} C^-1 tauv
    xcp = {h: b_xd(p) for h, p in ct.items()}
    adjY = mat_vec(adj, Yt)
    adjtv = mat_vec(adj, tauv)
    detrho2 = bv_add({m: b_x(p, 2) for m, p in bv_proj(rho4, adjY, ODD).items()},
                     {m: b_mul(det, p) for m, p in bv_sc(mat_vec(Cop, tauD), -1).items()},
                     {m: b_mul(det, b_sc(b_x(b_xd(tauv[m])), -1)) for m in ODD},
                     {m: b_x(p) for m, p in bv_sc(mat_vec(mat_mul(Cop, OmM), adjtv), -1).items()},
                     {m: b_x(p) for m, p in bv_proj(xcp, adjtv, ODD).items()})
    Sx2 = {m: b_divx(p, 2) for m, p in St.items()}
    bx2 = {h: b_divx(p, 2) for h, p in bt.items()}
    out.update(V=Vt, S=St, Sx2=Sx2, Y=Yt, Dl=Dt, detrho2=detrho2, beta=bt, betax2=bx2, cm1=cm1, cb=cb, cbx2=cbx2,
               C=Cop, Cm1=mult_op(cm1), Bb=Bb, Bbx2=Bbx2, Gam=Gam, detm1={0: detm1}, Neta=Neta, adj=adj)
    return out


# Lemma F: tauD, taub = O(eps^K), tauv, rho4 = O(eps^(K+1)); hence det*rho2 = O(eps^K)
LOWEST = dict(tauD=K, taub=K, tauv=K + 1, rho4=K + 1)


def verify_low_orders(fs):
    """recompute the residuals EXACTLY with products truncated at eps-order K + 2 (n - 1 contributes eps^-1 once per
    product chain, so nothing below K + 1 is lost) and check that every coefficient below LOWEST vanishes"""
    global JMAX
    JMAX = K + 2
    try:
        ex = build(fs, lambda r: r, ONE, check_exact=True)
    finally:
        JMAX = None
    bad = []
    # the light-cone gauge c_k(1) = 0 (k >= 1) and the evenness of c_k, exactly: then C^K(+-1) = I and
    # C^K^2 - I is divisible by x^2 - 1
    for k in range(1, K + 1):
        for h, p in fs["c"][k].items():
            s = ZERO
            for m, v in p.items():
                s = s + v
                if m % 2:
                    bad.append(("c odd power", h, k, m))
            if not s.iszero():
                bad.append(("c_k(1) != 0", h, k, None))
    for nm, j0 in LOWEST.items():
        for h, p in ex[nm].items():
            for (j, m), v in p.items():
                if j < j0 and not v.iszero():
                    bad.append((nm, h, j, m))
        # and the order j0 itself is present (the statement is sharp)
    return bad


# ---------------------------------------------------------------- enclosure over the box
def box(rA, rW):
    rA, rW = iv.mpf(rA), iv.mpf(rW)
    A = iv.mpc(iv.mpf("2.5421480475") + iv.mpf([-1, 1]) * rA, iv.mpf([-1, 1]) * rA)
    winf = 2 * iv.pi / iv.mpf("1.64389375")
    W = winf * iv.mpc(iv.mpf("0.7769940136") + iv.mpf([-1, 1]) * rW, iv.mpf([-1, 1]) * rW)
    return A, W


def hi(x):
    """exact upper endpoint (mp.mpf) of an iv.mpf"""
    return mp.make_mpf(iv.mpf(x)._mpi_[1])


def absup(z):
    """upper bound (mp.mpf, exact endpoint) of |z| for an iv.mpc"""
    re, im = z.real, z.imag
    a = max(hi(abs(re.a)), hi(abs(re.b)))
    b = max(hi(abs(im.a)), hi(abs(im.b)))
    return hi(iv.sqrt(iv.mpf(a) ** 2 + iv.mpf(b) ** 2))


def enclose(obj, A, W):
    """{key: {(j, m): upper bound of |coef| over the box (mpf)}} for vectors (key = harmonic) or operators
    (key = (row, col))"""
    out = {}
    for key, p in obj.items():
        out[key] = {jm: absup(v.ev(A, W)) for jm, v in p.items()}
    return out


def tables(rA=2e-6, rW=2e-6, mutate=None):
    """in-process: {name: list of columns [(j, m, C)]}, C an upper bound (mp.mpf) of |coef| over the box, summed over
    the rows (harmonics) of the column; vectors and scalars have one column. Also |A|, |omega| upper bounds."""
    prec0 = iv.prec
    iv.prec = 200
    fs = formal(K, mutate)
    bad = verify_low_orders(fs)
    if bad and mutate is None:
        raise ValueError(f"Lemma F's residual orders fail exactly: {bad[:4]}")
    A, W = box(rA, rW)
    q = build(fs, ic_conv(A, W), IC(iv.mpc(1)))
    out = {}
    for nm, obj in q.items():
        keys = list(obj.keys())
        if keys and isinstance(keys[0], tuple):             # operator: columns
            cols = {}
            for (r, c), p in obj.items():
                for jm, v in p.items():
                    if v.iszero():
                        continue
                    cols.setdefault(c, {})
                    cols[c][jm] = cols[c].get(jm, iv.mpf(0)) + iv.mpf(absup(v.v))
            out[nm] = [[(j, m, hi(C)) for (j, m), C in d.items()] for d in cols.values()]
        else:
            d = {}
            for h, p in obj.items():
                for jm, v in p.items():
                    if v.iszero():
                        continue
                    d[jm] = d.get(jm, iv.mpf(0)) + iv.mpf(absup(v.v))
            out[nm] = [[(j, m, hi(C)) for (j, m), C in d.items()]]
    out["_absA"] = absup(A)
    out["_absW"] = absup(W)
    out["_lowbad"] = bad
    iv.prec = prec0
    return out


def ic_conv(A, W):
    def f(r):
        return IC(r.ev(A, W))
    return f


def main():
    iv.prec = 200
    centre = "centre" in OPTS
    rA = rW = 0 if centre else 2e-6
    mutate = OPTS.get("mutate")
    fs = formal(K, mutate)
    bad = verify_low_orders(fs)
    if bad:
        print(f"Lemma F's residual orders FAIL exactly: {bad[:6]} ...")
        if not mutate:
            return 1
    else:
        print(f"Lemma F's residual orders verified exactly: lowest eps-orders {LOWEST}")
    A, W = box(rA, rW)
    q = build(fs, ic_conv(A, W), IC(iv.mpc(1)))
    js = {}
    for nm, obj in q.items():
        js[nm] = {str(k): {f"{j},{m}": mp.nstr(absup(v.v) * (1 + mp.mpf(10) ** -18), 20)
                           for (j, m), v in d.items() if not v.iszero()} for k, d in obj.items()}
    absW = absup(W)
    res = dict(K=K, rA=rA, rW=rW, lowest=LOWEST, absW=str(absW), absA=str(absup(A)), q=js,
               low_orders_exact_ok=not bad)
    fn = OPTS.get("json", "p2b_series_centre.json" if centre else "p2b_series.json")
    if mutate:
        fn = f"p2b_series_mut_{mutate}.json"
    with open(os.path.join(HERE, fn), "w") as fh:
        json.dump(res, fh)
    print(f"wrote {fn}; |A| <= {float(absup(A)):.10f}, |omega| <= {float(absW):.10f}")
    # descriptor j-sums (sum over keys and m) for comparison with the printed atoms (b8m_atoms_K4.log, rA = 0)
    for nm in ("V", "S", "beta", "cm1", "Y", "Dl", "tauD", "tauv", "taub", "rho4", "detrho2"):
        d = {}
        for key, cs in js[nm].items():
            for jm, v in cs.items():
                j = int(jm.split(",")[0])
                d[j] = d.get(j, 0) + float(v)
        print(f"  {nm:7s} j-sums: " + ", ".join(f"[{j}] {v:.6g}" for j, v in sorted(d.items())))
    for nm in ("Gam", "Neta"):
        cols = {}
        for key, cs in js[nm].items():
            col = key.split(",")[1].strip(" )")
            for jm, v in cs.items():
                j = int(jm.split(",")[0])
                cols.setdefault(col, {})
                cols[col][(j, jm)] = float(v)
        d = {}
        for col, dd in cols.items():
            for (j, jm), v in dd.items():
                pass
        # induced l1 norm per (j, m): max over columns of the column sums, summed over m
        per = {}
        for key, cs in js[nm].items():
            col = key.split(",")[1].strip(" )")
            for jm, v in cs.items():
                per.setdefault(jm, {})
                per[jm][col] = per[jm].get(col, 0) + float(v)
        dj = {}
        for jm, cc in per.items():
            j = int(jm.split(",")[0])
            dj[j] = dj.get(j, 0) + max(cc.values())
        print(f"  {nm:7s} j-sums (op): " + ", ".join(f"[{j}] {v:.6g}" for j, v in sorted(dj.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
