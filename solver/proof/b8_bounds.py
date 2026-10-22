#!/usr/bin/env python3
"""Lemma B8 (proof/B8_constants.md, sections 4-8): the bootstrap estimates for the one-harmonic (M = 1) model with
explicit constants, and the threshold n_0.

Bound algebra.  An upper bound is a sum of monomials c X^eX theta^eT rho^eR n^eN (X = 1 + |x|, theta = |x|/X,
rho = X^2/n), kept separately on the regions of Sigma^+_n:
    N : |x| <= x_F           (X <= X_F = 1 + x_F, rho <= X_F^2/n_0, l = X^2/|x^2-1| <= l_N),
    R : the ray arg x = alpha within N (same, with l <= l_ray),
    F : x_F <= |x| <= R_n    (X >= X_F, rho <= rho_max(n_0), l <= l_F = X_F/(x_F - 1)).
The sup of a monomial over a region is bounded factor by factor (Lemma W'), choosing the better of eliminating n via
n = X^2/rho or keeping n >= n_0.  Every quantity f carries bounds of |f|, |x f'| and (for explicit functions) |x (x f')'|;
products, sums and quotients (by a quantity with a one-monomial lower bound) follow the Leibniz rules.
All constants are python-flint balls; sups are upper bounds of balls; nothing is rounded inward.

    python3 b8_bounds.py K L0 [kappa1] [delta] [xF]   (L0 = ln(n_0); prints the bootstrap map and whether it closes)
"""
import sys
from flint import arb, acb
import b8_atoms as AT

PI = arb.pi()


def up(x):
    return arb(arb(x).upper())


def lo(x):
    return arb(arb(x).lower())


REG = ("N", "R", "F")


class Geo:
    def __init__(self, K, L0, kappa1="1", kappa="0.7", delta="0.05", xF="5", rA="0", rW="0"):
        self.K = K
        self.L0 = arb(L0)
        self.n0 = self.L0.exp()
        self.kappa1 = arb(kappa1)
        self.kappa = arb(kappa)
        self.delta = arb(delta)
        self.xF = arb(xF)
        self.XF = 1 + self.xF
        self.alpha = PI / 4 + self.kappa / 2
        self.sk = self.kappa.sin()
        self.calpha = (2 * self.alpha).sin() / (2 * self.alpha)
        self.phid = (self.delta / 2).asin()
        n = self.n0
        self.Rn = (self.kappa1 * n / n.log()).sqrt()
        self.rho_max = up((1 + self.Rn) ** 2 / n)        # decreasing in n for n >= e (notes, Lemma D)
        d2 = self.delta / 2
        lN = max(up((2 - d2) / d2), up((2 + d2) ** 2 / (d2 * (2 * (1 - d2) ** 2 + 2 - d2 * d2).sqrt())),
                 up((2 + d2) / d2))
        self.ell = {"N": lN, "R": up(2 * arb(2).sqrt()), "F": up(self.XF / (self.xF - 1))}
        self.box = {
            "N": dict(Xlo=arb(1), Xhi=self.XF, thlo=arb(0), thhi=up(self.xF / self.XF), rhohi=up(self.XF**2 / n)),
            "R": dict(Xlo=arb(1), Xhi=self.XF, thlo=arb(0), thhi=up(self.xF / self.XF), rhohi=up(self.XF**2 / n)),
            "F": dict(Xlo=self.XF, Xhi=None, thlo=lo(self.xF / self.XF), thhi=arb(1), rhohi=self.rho_max)}
        self.D, self.C = AT.atoms(K, rA, rW)


# ------------------------------------------------------------------ bounds: {region: {(eX, eT, eR, eN): arb}}
class B:
    def __init__(self, t=None):
        self.t = t if t is not None else {r: {} for r in REG}

    @staticmethod
    def mono(c, eX=0, eT=0, eR=0, eN=0, regions=REG):
        b = B()
        for r in regions:
            c_r = c[r] if isinstance(c, dict) else c
            b.t[r][(eX, eT, eR, eN)] = up(c_r)
        return b

    def __add__(self, o):
        out = B({r: dict(self.t[r]) for r in REG})
        for r in REG:
            for k, v in o.t[r].items():
                out.t[r][k] = up(out.t[r][k] + v) if k in out.t[r] else v
        return out

    def __mul__(self, o):
        if not isinstance(o, B):
            a = abs(arb(o))
            return B({r: {k: up(v * a) for k, v in self.t[r].items()} for r in REG})
        out = B()
        for r in REG:
            acc = out.t[r]
            for k1, v1 in self.t[r].items():
                for k2, v2 in o.t[r].items():
                    k = tuple(a + b for a, b in zip(k1, k2))
                    acc[k] = up(acc[k] + v1 * v2) if k in acc else up(v1 * v2)
        return out

    __rmul__ = __mul__

    def pw(self, k):
        out = B.mono(1)
        for _ in range(k):
            out = out * self
        return out


ZERO = None


def msup(geo, r, key, c):
    """upper bound of c X^eX theta^eT rho^eR n^eN over region r, for all n >= n_0 (None if unbounded)"""
    eX, eT, eR, eN = key
    bx = geo.box[r]
    # theta
    if eT >= 0:
        th = bx["thhi"] ** eT
    elif bx["thlo"] > 0:
        th = bx["thlo"] ** eT
    else:
        return None
    best = None
    # option 1: eliminate n = X^2/rho
    A, Rr = eX + 2 * eN, eR - eN
    if Rr >= 0:
        if A <= 0:
            xf = bx["Xlo"] ** A
        elif bx["Xhi"] is not None:
            xf = bx["Xhi"] ** A
        else:
            xf = None
        if xf is not None:
            best = up(c * th * xf * bx["rhohi"] ** Rr)
    # option 2: keep n >= n0
    if eR >= 0 and eN <= 0:
        if eX <= 0:
            xf = bx["Xlo"] ** eX
        elif bx["Xhi"] is not None:
            xf = bx["Xhi"] ** eX
        else:
            xf = None
        if xf is not None:
            v = up(c * th * xf * bx["rhohi"] ** eR * geo.n0 ** eN)
            best = v if best is None or v < best else best
    return best


def bsup(geo, b, regions):
    tot = arb(0)
    worst = None
    for r in regions:
        s = arb(0)
        for k, c in b.t[r].items():
            v = msup(geo, r, k, c)
            if v is None:
                raise ValueError(f"unbounded monomial {k} on region {r}")
            s = up(s + v)
        worst = s if worst is None or s > worst else worst
    return worst


def bsup_r(geo, b, r):
    return bsup(geo, b, (r,))


# ------------------------------------------------------------------ quantities with value / x-derivative bounds
class Q:
    """|f| <= v, |x f'| <= d, |x (x f')'| <= d2 (None if not tracked), and optionally |f| >= lo (one monomial/region)"""

    def __init__(self, v, d=None, d2=None, lo=None):
        self.v, self.d, self.d2, self.lo = v, d, d2, lo

    def __add__(self, o):
        return Q(self.v + o.v, _add(self.d, o.d), _add(self.d2, o.d2))

    def __mul__(self, o):
        if not isinstance(o, Q):
            return Q(self.v * o, None if self.d is None else self.d * o, None if self.d2 is None else self.d2 * o)
        d = None if (self.d is None or o.d is None) else self.d * o.v + self.v * o.d
        d2 = None
        if None not in (self.d, o.d, self.d2, o.d2):
            d2 = self.d2 * o.v + self.d * o.d * arb(2) + self.v * o.d2
        return Q(self.v * o.v, d, d2)

    __rmul__ = __mul__


def _add(a, b):
    return None if (a is None or b is None) else a + b


def inv_lo(lob):
    """1/|g| <= 1/lo for a one-monomial-per-region lower bound lob"""
    out = B()
    for r in REG:
        (k, c), = lob.t[r].items()
        out.t[r][tuple(-e for e in k)] = up(1 / c)
    return out


def qinv(g):
    """the quantity 1/g, from g.lo"""
    il = inv_lo(g.lo)
    d = None if g.d is None else g.d * il * il
    d2 = None
    if g.d is not None and g.d2 is not None:
        d2 = g.d2 * il * il + g.d * g.d * il * il * il * arb(2)
    return Q(il, d, d2)


def const(c):
    return Q(B.mono(c), B(), B())


def from_desc(geo, d, dd=None, dd2=None):
    """value bound theta^m0 X^w sum_j C_j rho^j with region discounts (Lemma W'), from a descriptor"""
    def mk(dsc):
        if dsc is None:
            return None
        m0, w = dsc["m0"], dsc["w"]
        jmin = min(j for j, _ in dsc["coef"])
        cr = {}
        for r in REG:
            bx = geo.box[r]
            s = arb(0)
            for (j, m), c in dsc["coef"].items():
                s += c * bx["thhi"] ** (m - m0) * bx["Xlo"] ** (m - 2 * j - w) * bx["rhohi"] ** (j - jmin)
            cr[r] = up(s)
        return B.mono(cr, eX=w, eT=m0, eR=jmin)
    return Q(mk(d), mk(dd), mk(dd2))


def lower_one_minus(geo, b, scale=None):
    """one-monomial lower bound 1 - sup(b) per region (b a small quantity), times optional monomial scale"""
    out = B()
    for r in REG:
        s = bsup_r(geo, b, r)
        v = lo(1 - s)
        if not v > 0:
            raise ValueError(f"lower bound 1 - {float(s):.3g} not positive on region {r}")
        out.t[r][(0, 0, 0, 0)] = v
    return out if scale is None else _scale_lo(out, scale)


def _scale_lo(lob, mono_b):
    out = B()
    for r in REG:
        (k1, c1), = lob.t[r].items()
        (k2, c2), = mono_b.t[r].items()
        out.t[r][tuple(a + b for a, b in zip(k1, k2))] = lo(c1 * c2)
    return out


def lo_mono(c, eX=0, eT=0, eR=0, eN=0):
    b = B()
    for r in REG:
        b.t[r][(eX, eT, eR, eN)] = lo(c[r] if isinstance(c, dict) else c)
    return b


# ------------------------------------------------------------------ the known (explicit) quantities
def known(geo):
    D, C = geo.D, geo.C
    K = geo.K
    one = const(1)
    x = Q(B.mono(1, eX=1, eT=1), B.mono(1, eX=1, eT=1), B.mono(1, eX=1, eT=1))
    x2 = Q(B.mono(1, eX=2, eT=2), B.mono(2, eX=2, eT=2), B.mono(4, eX=2, eT=2))
    eps = Q(B.mono(1, eN=-1), B(), B())
    nfac = up(geo.n0 / (geo.n0 - 1))
    eps1 = Q(B.mono(nfac, eN=-1), B(), B())                # 1/(n-1) <= (n0/(n0-1))/n
    nm1 = Q(B.mono(1, eN=1), B(), B())                      # n - 1 <= n
    k = {}
    xd2 = lambda name: None
    k["VK"] = from_desc(geo, D["V"], D["xV'"], D.get("x(xV')'"))
    k["SK"] = from_desc(geo, D["S"], D["xS'"], D.get("x(xS')'"))
    k["SKx2"] = from_desc(geo, D["S/x2"], D["x(S/x2)'"], D.get("x(x(S/x2)')'"))
    k["bK"] = from_desc(geo, D["beta"], D["xbeta'"], D.get("x(xbeta')'"))
    k["bKx2"] = from_desc(geo, D["beta/x2"], D["xbeta/x2'"], D.get("x(xbeta/x2')'"))
    cm1 = from_desc(geo, D["c-1"], D["xc'"], D.get("x(xc')'"))
    k["cK"] = Q(B.mono(1) + cm1.v, cm1.d, cm1.d2, lower_one_minus(geo, cm1.v))
    k["gam"] = from_desc(geo, D["gamma"], D["xgamma'"], D.get("x(xgamma')'"))
    Am = from_desc(geo, D["Am"], D["xA'"], D.get("x(xA')'"))
    k["AK"] = Q(B.mono(1, eN=1) + Am.v, Am.d, Am.d2, _scale_lo(lower_one_minus(geo, Am.v * B.mono(1, eN=-1)),
                                                                 B.mono(1, eN=1)))
    ell_lo = {r: 1 / geo.ell[r] for r in REG}
    x2m1 = Q(B.mono(1, eX=2), B.mono(2, eX=2, eT=2), B.mono(4, eX=2, eT=2), lo_mono(ell_lo, eX=2))
    omg = one + k["gam"] * (-1)
    omg = Q(omg.v, omg.d, omg.d2, lower_one_minus(geo, k["gam"].v))
    DK = x2m1 * omg
    DK.lo = _scale_lo(lo_mono(ell_lo, eX=2), lower_one_minus(geo, k["gam"].v))
    k["DK"] = DK
    k["BK"] = const(C["abs1piw"]) + k["bK"]
    k["bbK"] = one + k["bK"] * eps1                         # b^K = 1 + beta^K/(n-1)
    k["YK"] = k["VK"] + k["SK"]
    ne = from_desc(geo, D["ne"], D["x(ne)'"], D.get("x(x(ne)')'"))
    ope = one + ne * eps
    ope.lo = lower_one_minus(geo, ne.v * B.mono(1, eN=-1))
    icK = qinv(k["cK"])
    k["icK"] = icK
    k["q"] = k["bbK"] * (k["SKx2"] + k["YK"] * icK * icK) * DK * eps * qinv(ope)
    k["rho2"] = from_desc(geo, D["rho2"], D["xrho2'"])
    k["rho3t"] = from_desc(geo, D["rho3t"], D["xrho3t'"])
    k["rho4"] = from_desc(geo, D["rho4"], D["xrho4'"])
    k.update(one=one, x=x, x2=x2, eps=eps, eps1=eps1, nm1=nm1, x2m1=x2m1)
    return k



# ------------------------------------------------------------------ Lemma K: the fast kernel along the paths P_x
def kernel_consts(geo, k, nsub=3000):
    """s_* (ray decay margin), K_F and gamma_F (region F), gamma_N (region N: sup over 0 < r <= x_F at n = n_0)"""
    D = geo.D
    n0 = geo.n0
    nf = up(n0 / (n0 - 1))
    eta = from_desc(geo, D["eta"])
    gam = k["gam"]
    gmax = {r: bsup_r(geo, gam.v, r) for r in REG}
    # |h'| = |eta| / ((n-1)|1 - gamma|) on the ray (regions R and F)
    hp = {r: up(bsup_r(geo, eta.v * B.mono(nf, eN=-1), r) / (1 - gmax[r])) for r in ("R", "F")}
    sstar = lo(geo.sk - max(hp.values()))
    if not sstar > 0:
        raise ValueError("ray decay margin s_* not positive")
    # region F
    HF = up(bsup_r(geo, eta.v * B.mono(1, eX=-2), "F") * geo.ell["F"] / (1 - gmax["F"]))
    qF = up(1 + 2 * geo.sk / geo.xF**2 + 1 / geo.xF**4)
    NF = lo((1 - 1 / n0) / (geo.rho_max * qF))
    KF = up((HF * HF / (4 * NF * geo.calpha)).exp())
    gF = up(KF * qF.sqrt() * geo.rho_max * nf / sstar + KF * (PI / (geo.calpha * NF)).sqrt())
    # region N: exact r-dependence at n = n_0 (all terms decrease in n at fixed r)
    ce = D["eta"]["coef"]
    cg = D["gamma"]["coef"]
    d2 = geo.delta / 2
    al, sk, ca = geo.alpha, geo.sk, geo.calpha
    xF = geo.xF
    brk = [arb(0), lo(1 - d2), up(1 + d2), xF]
    grid = []
    for a, b in zip(brk[:-1], brk[1:]):
        m = nsub // 3
        grid += [(a + (b - a) * i / m, a + (b - a) * (i + 1) / m, a, b) for i in range(m)]
    gN, KN = arb(0), arb(0)
    for a, b, A0, B0 in grid:
        rr = arb(lo(a)).union(arb(up(b)))
        Ueta = sum((c * n0 ** (-j) * rr ** m for (j, m), c in ce.items()), arb(0))
        Ug = sum((c * n0 ** (-j) * rr ** m for (j, m), c in cg.items()), arb(0))
        if float(B0) <= float(1 - d2) + 1e-12:
            mr = 1 - rr * rr
        elif float(A0) >= float(1 + d2) - 1e-12:
            mr = rr * rr - 1
        else:
            mr = d2 * (2 * rr * rr + 2 - d2 * d2).sqrt()
        mr = lo(mr)
        if not mr > 0:
            raise ValueError("arc distance to the light cone not positive")
        Hp = up(Ueta / ((1 - Ug) * mr))
        Qa = up(1 + 2 * rr * rr * sk + rr ** 4)
        Np = lo((n0 - 1) * rr * rr / Qa)
        opt1 = up(Hp * al)
        if Np > 0:
            opt2 = up(Hp * Hp / (4 * Np * ca))
            lnK = opt1 if opt1 < opt2 else opt2
            arc2 = up((Hp * Hp / (4 * Np * ca)).exp() * (PI / (Np * ca)).sqrt())
        else:
            lnK = opt1
            arc2 = None
        arc1 = up(al * (Hp * al).exp())
        arcp = arc1 if arc2 is None or arc1 < arc2 else arc2
        Kr = up(lnK.exp())
        g = up(Kr * Qa.sqrt() / ((n0 - 1) * sstar) + arcp)
        gN = g if g > gN else gN
        KN = Kr if Kr > KN else KN
    return dict(sstar=sstar, hp=hp, HF=HF, NF=NF, KF=KF, gF=gF, gN=gN, KN=KN)


# ------------------------------------------------------------------ the improvement map (sections 5-7 of the notes)
def improve(geo, k, kc, b, verbose=False):
    K = geo.K
    n0 = geo.n0
    nf = up(n0 / (n0 - 1))
    w1 = B.mono(1, eX=2 * K + 2, eN=-(K + 1))                 # rho^(K+1)
    wS = B.mono(1, eX=2 * K + 2, eT=1, eN=-(K + 1))           # theta rho^(K+1)
    iwS = inv_lo(lo_mono(1, eX=2 * K + 2, eT=1, eN=-(K + 1)))
    absw = geo.C["absw"]
    one, x, x2, eps, eps1, nm1 = (k[s] for s in ("one", "x", "x2", "eps", "eps1", "nm1"))
    cK, SK, VK, bK, AK, DK, BK, bbK, YK, q, icK = (k[s] for s in ("cK", "SK", "VK", "bK", "AK", "DK", "BK", "bbK",
                                                                  "YK", "q", "icK"))
    x.lo = lo_mono(1, eX=1, eT=1)
    rho2, rho3t, rho4 = k["rho2"], k["rho3t"], k["rho4"]
    two = const(2)
    diag = {}

    def build(RS, RbH, Rc, RV):
        Rb = RbH + q * RS * two
        c = cK + Rc
        c.lo = B()
        for r in REG:
            (kk, cc), = cK.lo.t[r].items()
            v = lo(cc - bsup_r(geo, Rc.v, r))
            if not v > 0:
                raise ValueError("c lower bound")
            c.lo.t[r][kk] = v
        beta = bK + Rb
        S = SK + RS
        V = VK + RV
        Y = YK + RV + RS
        dD = (two * cK + Rc) * Rc
        # D = D^K - dD, |x D'| = |2 x^2 - 2 c^2 beta|
        D = Q(DK.v + dD.v, None)
        D.lo = B()
        for r in REG:
            (kk, cc), = DK.lo.t[r].items()
            rel = bsup_r(geo, dD.v * inv_lo(DK.lo), r)
            v = lo(cc * (1 - rel))
            if not v > 0:
                raise ValueError("D lower bound")
            D.lo.t[r][kk] = v
        D.d = (x2 * two).v + (c * c * beta * two).v
        ic = qinv(Q(c.v, (c * beta).v, None, c.lo))          # |x c'| = |c beta|
        dAc = nm1 * bbK * (two * cK + Rc) * Rc
        num1 = S * (dAc * DK + AK * dD) + x2 * BK * VK * dD + x2 * DK * BK * RV
        DA = Q(D.v * AK.v, None if (D.d is None or AK.d is None) else D.d * AK.v + D.v * AK.d)
        DA.lo = _scale_lo(D.lo, AK.lo)
        G1 = num1 * qinv(DA) + rho2 * qinv(AK)
        xD = Q((x * D).v, None if D.d is None else (x * D).d)
        xD.lo = _scale_lo(x.lo, D.lo)
        W2 = Rb * (S * (c * c + x2) + x2 * V) * qinv(xD)
        return dict(Rb=Rb, c=c, beta=beta, S=S, V=V, Y=Y, dD=dD, D=D, ic=ic, G1=G1, W2=W2)

    def gbeta(R, RS, RV, Rc, G1, W2, Rb):
        ic, S, Y, c = R["ic"], R["S"], R["Y"], R["c"]
        H = S * S + x2 * Y * Y * ic * ic
        xq = Q(q.d, q.d2)
        xlamG1 = AK * qinv(DK) * G1                       # x lambda^K G1 = (A^K/D^K) G1
        xgS = xlamG1 + x * W2
        gb = (Rb * (two * bK + Rb) + eps * Rb * H + bbK * (RS * RS + x2 * (RV + RS) * (RV + RS) * ic * ic)
              + bbK * x2 * RV * YK * ic * ic * two + bbK * x2 * RS * YK * R["dD"] * ic * ic * icK * icK * two
              + bbK * x2 * YK * YK * R["dD"] * ic * ic * icK * icK + rho3t * icK * icK
              + (xq * RS + q * xgS) * two)
        return gb, xgS

    # ---- pass 1: values
    RS = Q(wS * b["S"])
    RbH = Q(wS * b["b"])
    Rc = Q(w1 * b["c"])
    RV = Q(w1 * b["V"], wS * (absw * b["S"]))
    R1 = build(RS, RbH, Rc, RV)
    G1v, W2v = R1["G1"].v, R1["W2"].v
    RS_d = AK.v * inv_lo(DK.lo) * (RS.v + G1v) + x.v * W2v
    gb1, _ = gbeta(R1, RS, RV, Rc, R1["G1"], R1["W2"], R1["Rb"])
    RbH_d = gb1.v + nm1.v * RbH.v
    Rc_d = Rc.v * R1["beta"].v + cK.v * R1["Rb"].v + rho4.v
    # ---- pass 2: with derivatives
    RS = Q(wS * b["S"], RS_d)
    RbH = Q(wS * b["b"], RbH_d)
    Rc = Q(w1 * b["c"], Rc_d)
    R2 = build(RS, RbH, Rc, RV)
    G1, W2, Rb = R2["G1"], R2["W2"], R2["Rb"]
    gb, xgS = gbeta(R2, RS, RV, Rc, G1, W2, Rb)
    if G1.d is None or gb.d is None:
        raise ValueError("missing derivative bound")
    # ---- improved constants
    loc = G1.v * iwS
    integ = (G1.d + x.v * W2.v) * iwS
    bS_N = up(bsup_r(geo, loc, "N") + kc["gN"] * bsup_r(geo, integ, "N"))
    bS_F = up(bsup_r(geo, loc, "F") + kc["gF"] * bsup(geo, integ, ("R", "F")))
    bS = bS_N if bS_N > bS_F else bS_F
    gl = gb.v * B.mono(nf, eN=-1) * iwS
    gi = gb.d * B.mono(nf, eN=-1) * iwS
    fac = up(1 / n0 + geo.phid)
    bb_N = up(bsup_r(geo, gl, "N") + fac * bsup_r(geo, gi, "N"))
    bb_F = up(bsup_r(geo, gl, "F") + fac * bsup(geo, gi, ("N", "F")))
    bbn = bb_N if bb_N > bb_F else bb_F
    slow = up(arb(1) / (2 * K + 2) + geo.phid)
    bV = up(absw * slow * b["S"])
    chi = max(bsup_r(geo, R2["c"].v, r) for r in REG)
    clo = min(list(R2["c"].lo.t[r].values())[0] for r in REG)
    bc = up(chi / clo * slow * bsup(geo, (cK.v * Rb.v + rho4.v) * iwS, ("N", "F")))
    if verbose:
        diag = dict(G1loc_N=bsup_r(geo, loc, "N"), G1loc_F=bsup_r(geo, loc, "F"), integ_N=bsup_r(geo, integ, "N"),
                    integ_RF=bsup(geo, integ, ("R", "F")), gl_N=bsup_r(geo, gl, "N"), gl_F=bsup_r(geo, gl, "F"),
                    gi=bsup(geo, gi, ("N", "F")), chi=chi, clo=clo)
    return dict(V=bV, S=bS, b=bbn, c=bc), diag


def bootstrap(geo, iters=60, verbose=False):
    """iterate b <- P(b) from 0; closes if P(1.25 b*) < 1.25 b* componentwise"""
    k = known(geo)
    kc = kernel_consts(geo, k)
    b = dict(V=arb(0), S=arb(0), b=arb(0), c=arb(0))
    for it in range(iters):
        nb, dg = improve(geo, k, kc, b)
        done = all(abs(float(nb[s]) - float(b[s])) <= 1e-6 * (1 + float(nb[s])) for s in b)
        b = nb
        if done:
            break
        if any(float(b[s]) > 1e30 for s in b):
            return False, b, None, kc, {}
    trial = {s: up(b[s] * arb("1.25")) for s in b}
    Pt, dg = improve(geo, k, kc, trial, verbose=verbose)
    closes = all(Pt[s] < trial[s] for s in b)
    return closes, trial, Pt, kc, dg


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    L0 = sys.argv[2] if len(sys.argv) > 2 else "40"
    g = Geo(K, L0)
    ok, b, Pb, kc, dg = bootstrap(g, verbose=True)
    print(f"K = {K}, ln n0 = {L0}: kernel " + ", ".join(f"{a} {float(v):.4g}" for a, v in kc.items() if a != "hp"))
    print(f"  closes: {ok};  hypothesis b = " + ", ".join(f"{s} {float(v):.4g}" for s, v in b.items()))
    if Pb is not None:
        print(f"  P(b) = " + ", ".join(f"{s} {float(v):.4g}" for s, v in Pb.items()))
        print("  diag: " + ", ".join(f"{a} {float(v):.4g}" for a, v in dg.items()))
    k = known(g)
    print(f"K = {K}, ln n0 = {L0}: rho_max = {float(g.rho_max):.4g}, R_n = {float(g.Rn):.4g}, alpha = "
          f"{float(g.alpha):.4f}, phi_delta = {float(g.phid):.4f}, l = " + str({r: round(float(v), 2) for r, v in g.ell.items()}))
    W = {"VK": B.mono(1), "SK": B.mono(1, eX=2, eT=2, eN=-1), "bK": B.mono(1, eX=2, eT=2, eN=-1), "cK": B.mono(1),
         "AK": B.mono(1, eN=1), "DK": B.mono(1, eX=2), "q": B.mono(1, eX=2, eN=-1),
         "rho2": B.mono(1, eX=2 * K + 2, eT=2, eN=-K)}
    for name, wt in W.items():
        qq = k[name]
        iw = inv_lo(_scale_lo(lo_mono(1), wt))
        f = lambda bb: ", ".join(f"{r} {float(bsup_r(g, bb * iw, r)):.4g}" for r in REG)
        print(f"  {name:5s} (units {list(wt.t['N'])[0]}) v: {f(qq.v)}" + (f"   d: {f(qq.d)}" if qq.d is not None else "")
              + (f"   d2: {f(qq.d2)}" if qq.d2 is not None else ""))
