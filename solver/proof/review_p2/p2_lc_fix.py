#!/usr/bin/env python3
"""P2 REVIEW COPY of solver/proof/b8m_lc.py (NOT an independent implementation): the certificate with the two
corrections proposed in proof/P2_review.md, to measure their effect. Changes, marked "P2 FIX":
  (1) |x c'| for a member of the class K_h is bounded by |x c^K'| + |x R_c'| (the class bound D0["c"]), not by
      |c||beta| (that is the c-equation, which only solutions satisfy); |x D'| <= 2|x|^2 + 2|c||x c'| with that bound.
  (2) --rcpp=K multiplies the bound used for |R_c''| on L (needed in (dD/(x-1))') by K, to show that the closure and
      the Lipschitz matrix do not depend on it (it enters only through 1/(N_L - 1) and n0^-1 factors).
Everything else is b8m_lc.py unchanged (it imports b8_alg, b8m_atoms, b8m_bounds from solver/proof).

Original docstring:
Lemma LC (proof/B_chain.md section 13; plan in section 12): B8 continued to the light cone in the exact light-cone gauge, M = 1.

Setting as b8m_bounds.py (proof/B8_constants.md), with these changes (B_chain.md section 12.3):
  - the domain is the whole closed sector 0 <= arg x <= alpha, |x| <= R'_n (B8's domain plus the closed upper half
    of the light-cone disc, region L = {|x - 1| <= delta/2, Im x >= 0});
  - the gauge is c(1) = 1 (R_c(1) = 0); d := R_c(0) = -int_0^1 x R_c'(s) ds/s is part of the fixed point, and
    R_V(0) = [d 2F] (as in B8_constants.md 8.4);
  - the hypothesis constants are per radial band: N0 (|x| <= r1), Na (r1 <= |x| <= x_a, off the disc), L, Nb
    (x_a <= |x| <= x_F), F; on the ray: Ra (|x| <= x_a, hypothesis max(N0, Na)), Rb (Nb); the arc region Ac
    (the circle |x - 1| = delta/2 within the sector, hypothesis Na) is used for the data entering L;
  - in L every D^-1 is used in bounded form: D = (x - 1) Dt, D^K = (x - 1) Dt^K, Dt^K = (x + 1)(I - Gamma),
    dD/(x - 1) bounded by (2|c^K| + |R_c|) sup|R_c'| on [1, x] (R_c(1) = 0), x W2 = Lambda G2 with
    G2 = (A^K)^-1 Dt^K Dt^-1 (C M_h S + x^2 M_h C^-1 Y), and q x gS = ell x^-2 (I + e)^-1 eps1 A^K (G1 + G2);
  - S in L: from 0 along P_{x'} to the radial projection x' of x on the arc, then radially toward 1, with kernel
    |E(x, s)| <= (|x - 1|/|s - 1|)^N_L, N_L = (n0 - 1)(1 - c1 delta/2)/2 - eta_L; the hypothesis in L includes the
    derivative bound |x R_S'| <= h_Sp n w (needed because Lambda ~ n/(x - 1)).

Fixed point (B_chain.md 13.4): the closure P(h) < h shows that the integral operator T of the fixed-point formulation
maps the set K_h (value bounds h, derivative bounds D0(h) from the equations) into itself; --lip computes upper bounds
for its Lipschitz matrix in the norm max over (component, band) of sup|U|/(h w) and sup|x U'|/(h' m) (function
lipschitz: slopes of the convex majorant, as exact ball-mode secants per candidate) and checks Perron weights v with
M v < theta v, theta < 1: T is a contraction on K_h, so the light-cone-gauge solution exists, is unique in K_h and is
holomorphic in the parameters.

    python3 b8m_lc.py K L0 [--rA=2e-6 --rW=2e-6] [--xa=1.5] [--iters=200] [--log=file] [--lip [--eta=1e-6]]
                      [--mutate=NAME]
    mutation: rho2L (the residual rho2 dropped from G on the light-cone region L): the certificate still closes, but
    its bound on R_S in L must then fall below the numerically observed remainder (b8m_lc_numcheck.py with the
    mutated log must fail); b8m_lc_numcheck.py has its own mutations (d0, weight).
"""
import json
import os
import sys
import time
from flint import arb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # solver/proof (P2: run from review_p2)
import b8_alg as E
import b8m_atoms as AT
import b8m_bounds as MB

B, Q = E.B, E.Q
BALL = [False]          # ball mode: up/lo keep the ball (exact majorant arithmetic, for the Lipschitz secants)
_up0, _lo0 = E.up, E.lo


def up(x):
    return arb(x) if BALL[0] else _up0(x)


def lo(x):
    return arb(x) if BALL[0] else _lo0(x)


E.up, E.lo = up, lo     # b8_alg's internal calls resolve these at call time
PI = arb.pi()
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate", "")
RCPP = arb(OPTS.get("rcpp", "1"))            # P2 FIX (2)
FIXOFF = "fixoff" in OPTS                    # P2: reproduce b8m_lc.py exactly (for the comparison)

TARGETS = ("N0", "Na", "L", "Nb", "F")
COMPS = ("V", "S", "b", "c")
PATH_P = {"N0": ("N0",), "Na": ("Ra", "Na"), "Nb": ("Ra", "Rb", "Nb"), "F": ("Ra", "Rb", "F")}
PATH_Q = {"N0": ("N0",), "Na": ("N0", "Na"), "L": ("N0", "Na", "L"), "Nb": ("N0", "Na", "Nb"),
          "F": ("N0", "Na", "Nb", "F")}


class GeoLC:
    def __init__(self, K, L0, kappa="0.3", kappa1="0.6", delta="0.05", xF="20", r1="0.05", xa="1.5",
                 rA="0", rW="0", atoms=None):
        self.K = K
        self.L0 = arb(L0)
        self.n0 = self.L0.exp()
        self.kappa, self.kappa1, self.delta, self.xF = arb(kappa), arb(kappa1), arb(delta), arb(xF)
        self.r1, self.xa = arb(r1), arb(xa)
        self.gamma0 = arb("0.3")
        self.XF = 1 + self.xF
        self.alpha = PI / 4 + self.kappa / 2
        self.sk = self.kappa.sin()
        self.salpha = self.alpha.sin() / self.alpha
        self.phid = (self.delta / 2).asin()
        n = self.n0
        self.Rout = (self.kappa1 * n / (self.kappa.cos() * n.log())).sqrt()
        self.outer = "full"
        self.rho_max = up((1 + self.Rout) ** 2 / n)
        d2 = self.d2 = self.delta / 2
        r1, xa = self.r1, self.xa
        lN = max(up((2 - d2) / d2), up((2 + d2) ** 2 / (d2 * (2 * (1 - d2) ** 2 + 2 - d2 * d2).sqrt())),
                 up((2 + d2) / d2))
        # X^2/|x^2 - 1| per region (off the disc); L gets a dummy (never used: L-entries of D^-1 objects are replaced)
        self.ell = {"N0": up((1 + r1) / (1 - r1)), "Na": lN, "Nb": up((1 + xa) / (xa - 1)), "F": up(self.XF / (self.xF - 1)),
                    "Ra": up(2 * arb(2).sqrt()), "Rb": up(2 * arb(2).sqrt()), "Ac": lN, "L": arb(10) ** 6}
        th = lambda r: r / (1 + r)
        boxes = {
            "N0": dict(Xlo=arb(1), Xhi=up(1 + r1), thlo=arb(0), thhi=up(th(r1)), rho=up((1 + r1) ** 2 / n), far=False),
            "Na": dict(Xlo=lo(1 + r1), Xhi=up(1 + xa), thlo=lo(th(r1)), thhi=up(th(xa)), rho=up((1 + xa) ** 2 / n),
                       far=False),
            "L": dict(Xlo=lo(2 - d2), Xhi=up(2 + d2), thlo=lo(th(1 - d2)), thhi=up(th(1 + d2)),
                      rho=up((2 + d2) ** 2 / n), far=False),
            "Ac": dict(Xlo=lo(2 - d2), Xhi=up(2 + d2), thlo=lo(th(1 - d2)), thhi=up(th(1 + d2)),
                       rho=up((2 + d2) ** 2 / n), far=False),
            "Nb": dict(Xlo=lo(1 + xa), Xhi=self.XF, thlo=lo(th(xa)), thhi=up(th(self.xF)), rho=up(self.XF ** 2 / n),
                       far=False),
            "Ra": dict(Xlo=arb(1), Xhi=up(1 + xa), thlo=arb(0), thhi=up(th(xa)), rho=up((1 + xa) ** 2 / n), far=False),
            "Rb": dict(Xlo=lo(1 + xa), Xhi=self.XF, thlo=lo(th(xa)), thhi=up(th(self.xF)), rho=up(self.XF ** 2 / n),
                       far=False),
            "F": dict(Xlo=self.XF, Xhi=None, thlo=lo(th(self.xF)), thhi=arb(1), rho=self.rho_max, far=True)}
        E.configure(boxes, n)
        if atoms is None:
            atoms = AT.atoms(K, rA, rW)
        self.D, self.C = atoms


SUMHYP = [False]


def hyp_regions(h):
    """per-region hypothesis constants: the targets, the ray parts and the arc (Ra = N0 u Na part of the ray:
    max, or the sum in the Lipschitz runs, where each target region is perturbed separately)"""
    out = {r: h[r] for r in TARGETS}
    if SUMHYP[0]:
        out["Ra"] = h["N0"] + h["Na"]
    else:
        out["Ra"] = h["N0"] if h["N0"] > h["Na"] else h["Na"]
    out["Rb"] = h["Nb"]
    out["Ac"] = h["Na"]
    return out


# path regions containing points of each target region (for derivative-bound perturbations)
CONTAINS = {"N0": ("N0", "Ra"), "Na": ("Na", "Ra", "Ac"), "L": ("L",), "Nb": ("Nb", "Rb"), "F": ("F",)}


def mono_reg(spec):
    """a B with one term per region, spec[r] = (c, eX, eT, eN); and the inverse of the same as a lower bound"""
    b, bl = B(), B()
    for r in E.REG:
        c, eX, eT, eN = spec[r]
        m = B.mono(c, eX=eX, eT=eT, eN=eN)
        b.t[r] = dict(m.t[r])
        ml = E.lo_mono(c, eX=eX, eT=eT, eN=eN)
        bl.t[r] = dict(ml.t[r])
    return b, E.inv_lo(bl)


def der_weights(K):
    """derivative weights m: S: n w (n w / X^2 on F); b: n w; c, V: w (w = theta X^(2K+2) n^-(K+1))"""
    fastS = {r: (1, 2 * K + 2 - (2 if r == "F" else 0), 1, -K) for r in E.REG}
    fastb = {r: (1, 2 * K + 2, 1, -K) for r in E.REG}
    slow = {r: (1, 2 * K + 2, 1, -(K + 1)) for r in E.REG}
    return {"S": mono_reg(fastS), "b": mono_reg(fastb), "c": mono_reg(slow), "V": mono_reg(slow)}


def onlyL(v):
    """a constant (arb) placed on region L only (zero elsewhere)"""
    return B.mono({r: (v if r == "L" else arb(0)) for r in E.REG})


def onlyLd(v):
    return {r: (v if r == "L" else arb(0)) for r in E.REG}


def supLn(b):
    """the L-entries of b with X and theta replaced by their bounds on L, n^e kept (a B on L only)"""
    bx = E.BOX["L"]
    out = B()
    for (bT, e), (a, c) in b.t["L"].items():
        xf = bx["Xhi"] ** a if a > 0 else bx["Xlo"] ** a
        th = bx["thhi"] ** bT if bT >= 0 else bx["thlo"] ** bT
        key = (0, e)
        old = out.t["L"].get(key)
        v = up(c * xf * th)
        out.t["L"][key] = (0, v if old is None else up(old[1] + v))
    return out


def kernel_lc(geo, k, nsub=1500):
    n0 = geo.n0
    nf = up(n0 / (n0 - 1))
    eta = k["x2"] * k["etaT"]
    gmax = {r: E.bsup_r(k["Gam"].v, r) for r in E.REG}
    hp = {r: up(E.bsup_r(eta.v * B.mono(nf, eN=-1), r) / (1 - gmax[r])) for r in ("Ra", "Rb", "F")}
    sstar = lo(geo.sk - max(hp.values(), key=float))
    if not sstar > 0:
        raise ValueError("ray decay margin s_* not positive")
    HF = up(E.bsup_r(eta.v * B.mono(1, eX=-2), "F") * geo.ell["F"] / (1 - gmax["F"]))
    qF = up(1 + 2 * geo.sk / geo.xF ** 2 + 1 / geo.xF ** 4)
    NF = lo((1 - 1 / n0) / (geo.rho_max * qF))
    lnKF = up(HF * HF / (4 * NF * geo.salpha ** 2))
    KF = up(lnKF.exp())
    gF = up(KF * qF.sqrt() * geo.rho_max * nf / sstar + KF * (PI / (NF * geo.salpha ** 2)).sqrt())
    D = geo.D
    cN, cd, cg = D["Neta"]["coef"], D["det-1"]["coef"], D["Gamma"]["coef"]
    d2 = geo.d2
    al, sa = geo.alpha, geo.salpha
    sk = geo.sk
    brk = [arb(0), geo.r1, lo(1 - d2), up(1 + d2), geo.xa, geo.xF]
    gN = {"N0": arb(0), "Na": arb(0), "Nb": arb(0)}
    for a, b in zip(brk[:-1], brk[1:]):
        m = nsub // 5
        for i in range(m):
            ra, rb = a + (b - a) * i / m, a + (b - a) * (i + 1) / m
            rr = arb(lo(ra)).union(arb(up(rb)))
            ev = lambda cf: sum((c * n0 ** (-j) * rr ** mm for (j, mm), c in cf.items()), arb(0))
            Ueta = rr * rr * ev(cN) / (1 - ev(cd))
            Ug = ev(cg)
            if float(b) <= float(1 - d2) + 1e-12:
                mr = 1 - rr * rr
            elif float(a) >= float(1 + d2) - 1e-12:
                mr = rr * rr - 1
            else:
                mr = d2 * (2 * rr * rr + 2 - d2 * d2).sqrt()
            mr = lo(mr)
            if not mr > 0:
                raise ValueError("arc distance to the light cone")
            Hp = up(Ueta / ((1 - Ug) * mr))
            Qa = up(1 + 2 * rr * rr * sk + rr ** 4)
            Np = lo((n0 - 1) * rr * rr / Qa)
            opt1 = up(Hp * al)
            if Np > 0:
                opt2 = up(Hp * Hp / (4 * Np * sa * sa))
                lnK = opt1 if opt1 < opt2 else opt2
                arc2 = up(opt2.exp() * (PI / (Np * sa * sa)).sqrt())
            else:
                lnK, arc2 = opt1, None
            arc1 = up(al * opt1.exp())
            arcp = arc1 if arc2 is None or arc1 < arc2 else arc2
            Kr = up(lnK.exp())
            g = up(Kr * Qa.sqrt() / ((n0 - 1) * sstar) + arcp)
            if float(b) <= float(geo.r1) + 1e-15:
                reg = "N0"
            elif float(b) <= float(geo.xa) + 1e-12:
                reg = "Na"
            else:
                reg = "Nb"
            gN[reg] = g if g > gN[reg] else gN[reg]
    # region L: radial paths toward x = 1 (proof in B_chain.md section 12.3 / the LC section):
    # d/dr ln|1 - s^-2| >= 1/r - c1, c1 = 1/(2 - d2) + 2/(1 - d2); |lambda||H'| <= eta_L / r with
    # eta_L = sup_L ||eta|| / ((1 - ||Gamma||) |x| |x + 1|)
    c1 = up(1 / (2 - d2) + 2 / (1 - d2))
    etaL = up(E.bsup_r(eta.v, "L") / ((1 - gmax["L"]) * (1 - d2) * (2 - d2)))
    NL = lo((n0 - 1) * (1 - c1 * d2) / 2 - etaL)
    if not NL > 2:
        raise ValueError("N_L too small")
    return dict(sstar=sstar, HF=HF, NF=NF, KF=KF, gF=gF, gN0=gN["N0"], gNa=gN["Na"], gNb=gN["Nb"], c1=c1, etaL=etaL,
                NL=NL)


def wB(geo, h, eT):
    K = geo.K
    return B.mono(hyp_regions(h), eX=2 * K + 2, eT=eT, eN=-(K + 1))


def evaluate(geo, k, kc, h, Dinp=None, verbose=False, want_der=False):
    """h: dict comp -> dict target -> arb (comps V, S, b, c), plus h["Sp"] (arb): |x R_S'| <= h_Sp n w on L.
    Dinp: the derivative bounds |x R'| of the set (B objects per comp); None: from the equations (pass 1).
    Returns dict(P = the improved constants, cand = their candidate lists, D0 = the derivative bounds used, ...)."""
    K = geo.K
    n0 = geo.n0
    nf = up(n0 / (n0 - 1))
    d2 = geo.d2
    iwS = E.inv_lo(E.lo_mono(1, eX=2 * K + 2, eT=1, eN=-(K + 1)))
    iw1 = E.inv_lo(E.lo_mono(1, eX=2 * K + 2, eN=-(K + 1)))
    absw = geo.C["absw"]
    one, x, x2, eps, eps1, nm1, Om, half, two = (k[s] for s in ("one", "x", "x2", "eps", "eps1", "nm1", "Om", "half",
                                                                 "two"))
    cK, iCK, SK, VK, bK, AK, iAK, DK, iDK, BK, bbK, YK, q, xq = (k[s] for s in (
        "cK", "iCK", "S", "V", "beta", "AK", "iAK", "DK", "iDK", "BK", "bbK", "Y", "q", "xq"))
    MK, Dlt, DlK, rho2, rho3, rho4 = k["cb"], k["Dlt"], k["Dl"], k["rho2"], k["rho3"], k["rho4"]
    ope = k["ope"]
    ellx2 = k["ellx2"]
    Gam = k["Gam"]

    def build(RS, RbH, Rc, RV):
        Rb = RbH + q * RS
        Rb0 = q * RS
        c = cK + Rc
        c.lo = B()
        for r in E.REG:
            (kk, (aa, cc)), = cK.lo.t[r].items()
            v = lo(cc - E.bsup_r(Rc.v, r))
            if not v > 0:
                raise ValueError(f"c lower bound on {r}")
            c.lo.t[r][kk] = (aa, v)
        beta = bK + Rb
        # P2 FIX (1): the class bound |x c'| <= |x c^K'| + |x R_c'| (pass 2); pass 1 does not use c.d
        if Rc.d is not None and not FIXOFF:
            c.d = cK.d + Rc.d
        else:
            c.d = (c * beta).v if beta.v is not None else None
        c.d2 = None
        iC = E.qinv(Q(c.v, c.d, None, c.lo))
        S, V, Y = SK + RS, VK + RV, YK + RV + RS
        dC = Rc
        dD = (cK * two + Rc) * Rc
        D = Q(DK.v + dD.v)
        D.lo = B()
        for r in E.REG:
            (kk, (aa, cc)), = DK.lo.t[r].items()
            if r == "L":
                D.lo.t[r][kk] = (aa, cc)          # dummy, L-entries of D^-1 objects are never used
                continue
            rel = E.bsup_r(dD.v * E.inv_lo(DK.lo), r)
            v = lo(cc * (1 - rel))
            if not v > 0:
                raise ValueError(f"D lower bound on {r}")
            D.lo.t[r][kk] = (aa, v)
        if Rc.d is not None and not FIXOFF:
            D.d = (x2 * two).v + c.v * c.d * arb(2)            # P2 FIX (1): |x D'| <= 2|x|^2 + 2|c||x c'|
        else:
            D.d = (x2 * two).v + (c * c * beta * two).v
        iD = E.qinv(D)
        RcbK = Rc * bK

        def AB(Rbx):
            cRb = c * Rbx
            dB = (cRb + RcbK) * iC + MK * iC * dC * iCK + dC * Om * iC + cK * Om * iC * dC * iCK
            dA = nm1 * dD + dC * MK + c * (cRb + RcbK) + x2 * dB
            return dA, dB

        dA0, dB0 = AB(Rb0)
        A0, B0 = AK + dA0, BK + dB0
        G1 = iAK * (dA0 * S + dD * iD * (A0 * S + x2 * B0 * V) + x2 * (dB0 * V + BK * RV) + rho2)
        Mh = c * RbH
        W2 = iD * E.qinv(x) * (c * Mh * S + x2 * Mh * iC * Y)
        xgS = iDK * AK * G1 + x * W2
        return dict(Rb=Rb, c=c, iC=iC, beta=beta, S=S, V=V, Y=Y, dC=dC, dD=dD, D=D, G1=G1, W2=W2, xgS=xgS,
                    A0=A0, B0=B0, dA0=dA0, dB0=dB0, Mh=Mh, iD=iD)

    def build_L(R, Rc, Fc):
        """the bounded forms on L (only their L-entries are meaningful). Constants on L keep their n-dependence
        (supLn: X, theta replaced by their bounds on L, n^e kept), so that the bounds hold for every n >= n_0."""
        S, V, Y, c, iC, Mh = R["S"], R["V"], R["Y"], R["c"], R["iC"], R["Mh"]
        A0, B0, dA0, dB0 = R["A0"], R["B0"], R["dA0"], R["dB0"]
        RV = R["RV"]
        s1 = arb(1) / (1 - d2)
        # sup |R_c'| on L (R_c' = (x R_c')/x, |x| >= 1 - d2) and |x^2 R_c''| <= |x Fc'| + |Fc| (Fc = x R_c')
        rcp = supLn(Fc.v) * s1
        cKv, Rcv = supLn(cK.v), supLn(Rc.v)
        ddv = (cKv * 2 + Rcv) * rcp                           # |dD/(x - 1)| <= (2|c^K| + |R_c|) sup|R_c'|
        gm = E.bsup_r(Gam.v, "L")
        lDt = lo((2 - d2) * (1 - gm) - E.bsup_r(ddv, "L"))
        if not lDt > 0:
            raise ValueError("Dt lower bound on L")
        iDt_v = up(1 / lDt)
        DtK_v = up((2 + d2) * (1 + gm))
        dd_d = iDt_d = DtK_d = None
        if Fc.d is not None and Gam.d is not None:
            rcpp = (supLn(Fc.d) + supLn(Fc.v)) * (s1 * s1)
            rcpp = rcpp * RCPP                                  # P2 FIX (2): inflate the |R_c''| bound on L
            # |(dD/(x-1))'| <= 1/2 sup|dD''|, dD = C^K R_C + R_C C^K + R_C^2 (norm form):
            # |dD''| <= 2|c^K''||R_c| + 4|c^K'||R_c'| + 2|c^K||R_c''| + 2|R_c'|^2 + 2|R_c||R_c''|
            cKd = supLn(cK.d) * s1                                         # |c^K'|
            cKdd = (supLn(k["xc_d2"]) + supLn(cK.d)) * (s1 * s1)          # |c^K''|
            ddpp = cKdd * Rcv * 2 + cKd * rcp * 4 + cKv * rcpp * 2 + rcp * rcp * 2 + Rcv * rcpp * 2
            dd_d = ddpp * ((1 + d2) / 2)                                  # |x (dD/(x-1))'|
            gmd = supLn(Gam.d)
            DtK_d = B.mono(onlyLd(up((1 + d2) * (1 + gm)))) + gmd * (2 + d2)  # |x Dt^K'|, x Dt^K' = x(I-G) - (x+1) xG'
            iDt_d = (DtK_d + dd_d) * (iDt_v * iDt_v)
        iDt = Q(onlyL(iDt_v), iDt_d)
        ddL = Q(ddv, dd_d)
        DtK = Q(onlyL(DtK_v), DtK_d)
        inner = dA0 * S + ddL * iDt * (A0 * S + x2 * B0 * V) + x2 * (dB0 * V + BK * RV)
        G1L = iAK * (inner if MUT == "rho2L" else inner + rho2)
        G2L = iAK * DtK * iDt * (c * Mh * S + x2 * Mh * iC * Y)
        GL = G1L + G2L
        qxgSL = ellx2 * E.qinv(ope) * eps1 * AK * GL
        return dict(G1L=G1L, G2L=G2L, GL=GL, qxgSL=qxgSL, ddv=E.bsup_r(ddv, "L"), lDt=lDt,
                    iDtK=up(1 / ((2 - d2) * (1 - gm))))

    def gbeta(R, RS, RV, qxgS):
        iC, S, Y, Rb, dC = R["iC"], R["S"], R["Y"], R["Rb"], R["dC"]
        Dl = x * iC * Y
        H = (S * S + Dl * Dl) * half
        dDl = x * iC * (RV + RS) + iC * dC * Dlt
        g3 = bbK * (RS * RS * half + (x * iC * RV + iC * dC * (x * iCK * RS + Dlt)) * Dlt + dDl * dDl * half)
        return Rb * (bK * two + Rb) + eps * Rb * H + g3 + xq * RS + qxgS + rho3

    def mixL(std, Lq):
        """a Q whose L-entries come from Lq and the others from std"""
        def m(a, b):
            if a is None or b is None:
                return None
            out = B({r: dict(a.t[r]) for r in E.REG})
            out.t["L"] = dict(b.t["L"])
            return out
        return Q(m(std.v, Lq.v), m(std.d, Lq.d), m(std.d2, Lq.d2))

    hS, hb, hc, hV = h["S"], h["b"], h["c"], h["V"]
    RS = Q(wB(geo, hS, 1))
    RbH = Q(wB(geo, hb, 1))
    Rc = Q(wB(geo, hc, 0))
    RVv = wB(geo, hV, 0)
    if Dinp is None:
        # ---- pass 1: derivative bounds from the equations (the set K_h carries these bounds)
        RV = Q(RVv, wB(geo, hS, 1) * absw)
        R1 = build(RS, RbH, Rc, RV)
        R1["RV"] = RV
        Fc1 = Rc * R1["beta"] + cK * R1["Rb"] + rho4               # x R_c' (norm form)
        L1 = build_L(R1, Rc, Fc1)
        RS_d = (AK * iDK).v * (RS.v + R1["G1"].v) + x.v * R1["W2"].v
        # on L the derivative hypothesis: |x R_S'| <= h_Sp n w
        RS_d.t["L"] = dict(B.mono(h["Sp"], eX=2 * K + 2, eT=1, eN=-K).t["L"])
        qxgS1 = mixL(q * R1["xgS"], L1["qxgSL"])
        gb1 = gbeta(R1, RS, RV, qxgS1)
        D0 = dict(S=RS_d, b=gb1.v + nm1.v * RbH.v, c=Fc1.v, V=RV.d)
    else:
        D0 = Dinp
    # ---- pass 2
    RS = Q(RS.v, D0["S"])
    RbH = Q(RbH.v, D0["b"])
    Rc = Q(Rc.v, D0["c"])
    RV = Q(RVv, D0["V"])
    R2 = build(RS, RbH, Rc, RV)
    R2["RV"] = RV
    Fc2 = Rc * R2["beta"] + cK * R2["Rb"] + rho4
    L2 = build_L(R2, Rc, Fc2)
    G1, W2 = R2["G1"], R2["W2"]
    qxgS2 = mixL(q * R2["xgS"], L2["qxgSL"])
    gb = gbeta(R2, RS, RV, qxgS2)
    if G1.d is None or gb.d is None or L2["GL"].d is None:
        raise ValueError("missing derivative bound")
    loc = G1.v * iwS
    integ = (G1.d + x.v * W2.v) * iwS
    gam = {"N0": kc["gN0"], "Na": kc["gNa"], "Nb": kc["gNb"], "F": kc["gF"]}
    # every output is the max of a list of candidates, each a sum of term-wise suprema (the Lipschitz secants are
    # taken per candidate)
    cand = {}
    for r in ("N0", "Na", "Nb", "F"):
        lr = E.bsup_r(loc, r)
        cand[("S", r)] = [up(lr + gam[r] * E.bsup_r(integ, s)) for s in PATH_P[r]]
    # ---- S on L
    NL = kc["NL"]
    GL = L2["GL"]
    bx = E.BOX["L"]
    # w(s)/w(x) for s, x in the annulus X in [2 - d2, 2 + d2] (L and the arc Ac): w = theta X^(2K+2) n^-(K+1)
    ratS = up(bx["thhi"] * bx["Xhi"] ** (2 * K + 2) / (bx["thlo"] * bx["Xlo"] ** (2 * K + 2)))
    rat1 = up(bx["Xhi"] ** (2 * K + 2) / bx["Xlo"] ** (2 * K + 2))
    G2arc = E.bsup_r((iAK * DK * x * W2).v * iwS, "Ac")         # |G2(x')|/w(x') on the arc (standard form there)
    jarcs = [up(G2arc + gam["Na"] * E.bsup_r(integ, s)) for s in PATH_P["Na"]]   # |R_S + G|(x')/w(x') <= Jarc
    supGd = up(E.bsup_r(GL.d * iwS, "L") / (1 - d2))             # sup_L |G'|/w
    GLv = E.bsup_r(GL.v * iwS, "L")
    # R_S(x) = -G(x) + E(x,x')(R_S + G)(x') + int_{x'}^x E G', |E| <= (|x-1|/|s-1|)^N_L, int <= sup|G'| |x-1|/(N_L - 1)
    cand[("S", "L")] = [up(GLv + J * ratS + supGd * ratS * d2 / (NL - 1)) for J in jarcs]
    # derivative on L: |x R_S'| = |Lambda (R_S + G)|, ||Lambda|| |x - 1| <= ||A^K|| / ((2 - d2)(1 - ||Gamma||)),
    # |R_S + G|/|x - 1| <= Jarc w(x')/d2 + sup|G'|/(N_L - 1); hypothesis form |x R_S'| <= h_Sp n w
    lamt = up(E.bsup_r(AK.v * B.mono(1, eN=-1), "L") * L2["iDtK"])        # per unit n
    candSp = [up(lamt * (J / d2 + supGd / (NL - 1)) * ratS) for J in jarcs]
    # ---- b-mode
    gl = gb.v * B.mono(nf, eN=-1) * iwS
    gi = gb.d * B.mono(nf, eN=-1) * iwS
    for r in TARGETS:
        fac = up(nf / n0 + (0 if r == "L" else geo.phid))
        glr = E.bsup_r(gl, r)
        cand[("b", r)] = [up(glr + fac * E.bsup_r(gi, s)) for s in PATH_Q[r]]
    # ---- the gauge offset d = -int_0^1 x R_c' ds/s (R_c(1) = 0), |d| <= d0 n^-(K+1)
    Fc_noR4 = Rc * R2["beta"] + cK * R2["Rb"]
    K2 = 2 * K + 2
    pieces = (("N0", arb(0), geo.r1), ("Na", geo.r1, 1 - d2), ("L", 1 - d2, arb(1)))
    d0 = arb(0)
    for r, a, b in pieces:
        sr = E.bsup_r(Fc_noR4.v * iwS, r)
        d0 = up(d0 + sr * ((1 + b) ** K2 - (1 + a) ** K2) / K2)
    # rho4 exactly: sum |coef_{j m}| n^-j / m, times n^(K+1)
    r4 = arb(0)
    for (j, m), cf in geo.D["rho4"]["coef"].items():
        if m < 1:
            raise ValueError("rho4 has a constant term")
        r4 = up(r4 + cf * n0 ** (K + 1 - j) / m)
    d0 = up(d0 + r4)
    if MUT == "d0":
        d0 = arb(0)
    # ---- slow components
    slow = up(arb(1) / K2 + geo.phid)
    slowr = up(arb(1) / K2)
    hyS = hyp_regions(hS)
    for r in TARGETS:
        sl = slowr if r == "L" else slow
        cand[("V", r)] = [up(4 * geo.C["absf"] * d0 + absw * sl * hyS[s]) for s in PATH_Q[r]]
        if r == "L":
            # R_c(x) = int_1^x R_c', |R_c| <= |x - 1| sup_L |R_c'| <= d2 sup_L |x R_c'| / (1 - d2)
            cand[("c", r)] = [up(d2 / (1 - d2) * E.bsup_r(Fc2.v * iw1, "L") * rat1)]
        else:
            cand[("c", r)] = [up(d0 + sl * E.bsup_r(Fc2.v * iwS, s)) for s in PATH_Q[r]]
    P = {s: {} for s in COMPS}
    for (cp, r), lst in cand.items():
        P[cp][r] = max(lst, key=float)
    P["Sp"] = max(candSp, key=float)
    diag = dict(d0=d0, r4=r4, Jarc=max(jarcs, key=float), supGd=supGd, lDt=L2["lDt"], ddv=L2["ddv"], NL=NL)
    if verbose:
        for r in ("N0", "Na", "Nb", "F"):
            diag[f"loc_{r}"] = E.bsup_r(loc, r)
            diag[f"int_{r}"] = E.bsup(integ, PATH_P[r])
        diag["GL"] = GLv
    out = dict(P=P, diag=diag, cand=cand, candSp=candSp, D0=D0)
    if want_der:
        # derivative outputs of T(U), normalized by the weights m: value = A * (own value output) + rest
        Wt = der_weights(K)
        wSu = B.mono(1, eX=2 * K + 2, eT=1, eN=-(K + 1))
        der = {}
        for r in TARGETS:
            imS, imb, imc, imV = Wt["S"][1], Wt["b"][1], Wt["c"][1], Wt["V"][1]
            if r != "L":
                der[("S", r)] = (E.bsup_r((AK * iDK).v * wSu * imS, r),
                                 E.bsup_r(((AK * iDK).v * G1.v + x.v * W2.v) * imS, r))
            der[("b", r)] = (E.bsup_r(nm1.v * wSu * imb, r), E.bsup_r(gb.v * imb, r))
            der[("c", r)] = (arb(0), E.bsup_r(Fc2.v * imc, r))
            der[("V", r)] = (arb(0), E.bsup_r(RS.v * absw * imV, r))
        out["der"] = der
    return out


def improve_lc(geo, k, kc, h, verbose=False):
    o = evaluate(geo, k, kc, h, verbose=verbose)
    return o["P"], o["diag"]


def lipschitz(geo, k, kc, h, eta="1e-6", verbose=False):
    """Upper bounds for the Lipschitz matrix of T on K_h in the norm
        N(U) = max over (comp, region) of max( sup |U| / (h w), sup |x U'| / (h' m) )
    (values and derivatives; m = der_weights, h' = sup D0/m). By structural induction over the bound algebra, the
    difference of a node at U, U~ in K_h is bounded by the derivative, along the perturbed input, of its majorant
    (sum and product rules; inverses via 1/(l0 - s(t))); every candidate below is a sum of term-wise suprema of
    nonnegative-coefficient power series in t, hence convex in t, and its slope at t = 0 is bounded by the secant
    (M(eta) - M(0))/eta, computed in ball mode (no outward rounding between the two runs; both runs share the
    same B-object structure). Returns (rows, cols, matrix of arb upper bounds, scales)."""
    K = geo.K
    eta = arb(eta)
    SUMHYP[0] = True
    BALL[0] = True
    try:
        o0 = evaluate(geo, k, kc, h, want_der=True)
        D0 = o0["D0"]
        Wt = der_weights(K)
        hp = {(cp, r): E.bsup_r(D0[cp] * Wt[cp][1], r) for cp in COMPS for r in TARGETS}
        keys = [(cp, r) for cp in COMPS for r in TARGETS]
        rows = [("val",) + kk for kk in keys] + [("der",) + kk for kk in keys]
        cols = list(rows)
        M = {}

        def slopes(o, ob):
            sv = {}
            for kk in keys:
                sl = None
                for a, b in zip(o["cand"][kk], ob["cand"][kk]):
                    v = arb(((a - b) / eta).upper())
                    sl = v if sl is None or v > sl else sl
                sv[kk] = sl if sl > 0 else arb(0)
            ssp = None
            for a, b in zip(o["candSp"], ob["candSp"]):
                v = arb(((a - b) / eta).upper())
                ssp = v if ssp is None or v > ssp else ssp
            col = {}
            for kk in keys:
                col[("val",) + kk] = arb((sv[kk] / h[kk[0]][kk[1]]).upper())
                if kk == ("S", "L"):
                    col[("der",) + kk] = arb((max(ssp, arb(0)) / hp[kk]).upper())
                else:
                    A, rest = o["der"][kk]
                    _, rest0 = ob["der"][kk]
                    dr = arb(((rest - rest0) / eta).upper())
                    col[("der",) + kk] = arb(((A * sv[kk] + max(dr, arb(0))) / hp[kk]).upper())
            return col

        ob = evaluate(geo, k, kc, h, Dinp=D0, want_der=True)
        for c in cols:
            kind, cp, r = c
            if kind == "val":
                h2 = {s: dict(h[s]) for s in COMPS}
                h2["Sp"] = h["Sp"]
                h2[cp][r] = h[cp][r] * (1 + eta)
                o = evaluate(geo, k, kc, h2, Dinp=D0, want_der=True)
                M[c] = slopes(o, ob)
            else:
                m = Wt[cp][0]
                add0, add1 = B(), B()
                for rr in E.REG:
                    if rr in CONTAINS[r]:
                        add1.t[rr] = {kk: (a, cc * eta * hp[(cp, r)]) for kk, (a, cc) in m.t[rr].items()}
                        add0.t[rr] = {kk: (a, cc * 0) for kk, (a, cc) in m.t[rr].items()}
                Dz, De = dict(D0), dict(D0)
                Dz[cp] = D0[cp] + add0
                De[cp] = D0[cp] + add1
                obz = evaluate(geo, k, kc, h, Dinp=Dz, want_der=True)
                o = evaluate(geo, k, kc, h, Dinp=De, want_der=True)
                M[c] = slopes(o, obz)
            if verbose:
                big = max(M[c].items(), key=lambda kv: float(kv[1]))
                print(f"  column {c}: largest entry {big[0]} {float(big[1]):.3e}", flush=True)
    finally:
        SUMHYP[0] = False
        BALL[0] = False
    return rows, cols, M, hp


def contraction_check(rows, cols, M):
    """find v > 0 with M v < theta v: v = (I - M/lam)^-1 1 for lam slightly above the spectral radius (floats; then
    M v = lam (v - 1) < lam v), and theta = max_i (M v)_i / v_i is recomputed in arb from the upper bounds of M"""
    import numpy as np
    n = len(rows)
    idx = {r: i for i, r in enumerate(rows)}
    A = np.array([[float(M[c][r]) for c in cols] for r in rows])
    rho = float(max(abs(np.linalg.eigvals(A))))
    lam = min(0.5 * (rho + 1), rho + 0.05) if rho < 1 else rho
    try:
        v = np.linalg.solve(np.eye(n) - A / lam, np.ones(n))
    except np.linalg.LinAlgError:
        v = np.ones(n)
    if not (v > 0).all():
        v = np.ones(n)
    vv = [arb(float(x)) for x in v]
    theta = arb(0)
    for r in rows:
        i = idx[r]
        t = sum((M[c][r] * vv[idx[c]] for c in cols), arb(0)) / vv[i]
        t = arb(t.upper())
        theta = t if t > theta else theta
    return rho, theta, {"/".join(r): float(vv[idx[r]]) for r in rows}


def lipcheck_file(path):
    """recheck the contraction from a saved matrix (upper ends inflated by 1e-12 relative)"""
    d = json.load(open(path))
    M = {}
    rows = set()
    for key, val in d["M"].items():
        r, c = (tuple(x.split("/")) for x in key.split("<-"))
        rows.add(r)
        M.setdefault(c, {})[r] = arb(val) * (1 + arb("1e-12"))
    rows = sorted(rows)
    for c in rows:
        M.setdefault(c, {})
        for r in rows:
            M[c].setdefault(r, arb(0))
    return contraction_check(rows, rows, M)


def known_lc(geo):
    k = MB.known(geo)
    one, eps1, x = k["one"], k["eps1"], k["x"]
    e = (k["Om"] + k["BK"] + k["cK"] * k["cb/x2"]) * eps1
    ope = MB.one_plus(geo, e)
    ellx2 = k["bbK"] * (k["S/x2"] + k["DltX"] * k["iCK"])
    k["ope"] = ope
    k["ellx2"] = ellx2
    # |x (x c^K')'| (the d2 of c - 1)
    k["xc_d2"] = k["c-1"].d2
    return k


def flat(P):
    out = {}
    for s in COMPS:
        for r in TARGETS:
            out[f"{s}.{r}"] = P[s][r]
    out["Sp"] = P["Sp"]
    return out


def unflat(f):
    h = {s: {r: f[f"{s}.{r}"] for r in TARGETS} for s in COMPS}
    h["Sp"] = f["Sp"]
    return h


def zero_h():
    h = {s: {r: arb(0) for r in TARGETS} for s in COMPS}
    h["Sp"] = arb(0)
    return h


def closure(geo, iters=200, slacks=("1.1", "1.5"), eps="0.01", down=40, verbose=False):
    k = known_lc(geo)
    kc = kernel_lc(geo, k)
    h = zero_h()
    rec = dict(kernel={a: float(v) for a, v in kc.items()})
    last = None
    for it in range(iters):
        nh, dg = improve_lc(geo, k, kc, h)
        f0, f1 = flat(h), flat(nh)
        rel = max(abs(float(f1[s]) - float(f0[s])) / (1e-30 + float(f1[s])) for s in f1)
        h = nh
        last = rel
        if rel < 1e-6 or any(float(v) > 1e40 for v in f1.values()):
            break
    rec["iters"] = it + 1
    rec["last_rel"] = last
    rec["h_fixed"] = {s: float(v) for s, v in flat(h).items()}
    rec["closes"] = None
    for sl in slacks:
        ft = {s: up(v * arb(sl)) for s, v in flat(h).items()}
        ok = False
        try:
            for st in range(down):
                Pt, dg = improve_lc(geo, k, kc, unflat(ft), verbose=True)
                fP = flat(Pt)
                Th = {s: up(fP[s] * (1 + arb(eps))) for s in ft}
                if all(Th[s] <= ft[s] for s in ft):
                    ok = all(fP[s] < ft[s] for s in ft)
                    break
                ft = Th
        except ValueError as ex:
            rec[f"fail_{sl}"] = str(ex)
            continue
        if ok:
            rec["closes"] = float(arb(sl))
            rec["down_steps"] = st
            rec["hyp"] = {s: float(v) for s, v in ft.items()}
            rec["P_hyp"] = {s: float(v) for s, v in fP.items()}
            rec["ratio"] = {s: (float(fP[s] / ft[s]) if ft[s] > 0 else 0.0) for s in ft}
            rec["diag"] = {a: float(v) for a, v in dg.items()}
            return True, unflat(ft), Pt, k, kc, rec
    return False, h, None, k, kc, rec


def main():
    if "lipcheck" in OPTS:
        rho, theta, v = lipcheck_file(OPTS["lipcheck"])
        print(f"saved Lipschitz matrix {OPTS['lipcheck']}: spectral radius (float) {rho:.4f}; certified theta = "
              f"{float(theta.upper()):.4f} (< 1: {bool(theta < 1)})")
        return 0 if theta < 1 else 1
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    K = int(pos[0]) if pos else 4
    L0 = pos[1] if len(pos) > 1 else "73.89"
    t0 = time.time()
    at = AT.atoms(K, OPTS.get("rA", "0"), OPTS.get("rW", "0"))
    g = GeoLC(K, L0, xa=OPTS.get("xa", "1.5"), atoms=at)
    ok, h, Ph, k, kc, rec = closure(g, iters=int(OPTS.get("iters", "200")))
    if ok and "lip" in OPTS:
        t1 = time.time()
        rows, cols, M, hp = lipschitz(g, k, kc, h, eta=OPTS.get("eta", "1e-6"), verbose=True)
        Mj = {f"{'/'.join(r)}<-{'/'.join(c)}": float(M[c][r].upper()) for c in cols for r in rows}
        with open(OPTS.get("lipout", "b8m_lc_lip.json"), "w") as fh:
            json.dump(dict(M=Mj, hprime={f"{a}.{b}": float(c) for (a, b), c in hp.items()}), fh)
        lam, theta, v = contraction_check(rows, cols, M)
        print(f"Lipschitz matrix: Perron root (float) {lam:.4f}; certified theta = {float(theta.upper()):.4f} "
              f"(< 1: {bool(theta < 1)})", flush=True)
        rec["lip"] = dict(perron_float=lam, theta=float(theta.upper()), contraction=bool(theta < 1), weights=v,
                          hprime={f"{a}.{b}": float(c) for (a, b), c in hp.items()},
                          M={f"{'/'.join(r)}<-{'/'.join(c)}": float(M[c][r]) for c in cols for r in rows
                             if float(M[c][r]) > 1e-3},
                          time=round(time.time() - t1, 1))
    rec.update(K=K, L0=float(L0), rA=OPTS.get("rA", "0"), rW=OPTS.get("rW", "0"), xa=OPTS.get("xa", "1.5"),
               mutate=MUT or None, time=round(time.time() - t0, 1))
    line = json.dumps(rec)
    print(line, flush=True)
    if "log" in OPTS:
        with open(OPTS["log"], "a") as fh:
            fh.write(line + "\n")
    if MUT:
        print(f"MUTATION {MUT}: the record is written for b8m_lc_numcheck.py --mutlc (whose checks must then fail)")
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
