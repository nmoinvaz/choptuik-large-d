#!/usr/bin/env python3
"""P2 review: an independent implementation of the Lemma LC certificate (proof/B_chain.md section 13), written from
the notes (B8_constants.md sections 1-6, B_chain.md sections 12.3, 13.1-13.4), with its own atoms (p2_atoms.py: the
1/n series derived from the original model, complex interval arithmetic over the box), its own bound algebra
(p2_alg.py: mpmath with directed rounding) and its own Lipschitz method (forward differentiation of the majorant,
not secants).

It checks, for the class constants h of the certificate (b8m_lc_K4.log, "hyp", read as input) or for constants it
finds itself:
  (1) closure: every candidate bound P(h) < h (components V, S, beta^, c on the bands N0, Na, L, Nb, F; and h_Sp);
  (2) the gauge offset d0 (|c(0) - c^K(0)|_1 <= d0 n^-5) against B8's certified range 1e7;
  (3) the Lipschitz matrix of T in the norm of section 13.4 (values and x-derivatives x 4 components x 5 bands), and a
      Perron weight with M v <= theta v (exact rationals, p2_perron.perron).
Corrections F1-F3 of proof/P2_review.md are built in: R_c(1) = 0 and the R_c'' bound on L are part of the class (the
latter enters only through 1/(N_L - 1) and n0^-1, as in the certificate), and |x c'| is bounded by the class bound
|x c^K'| + |x R_c'|, not by the c-equation.

The norm, the regions, the paths and the lemmas are those of the notes (that is the statement being checked); the
code is independent. Nothing is imported from solver/proof.

    python3 p2_lc.py [--lip] [--mutate=NAME]
    Checks: (A) P(h) < h for every component and band, and h_Sp; (B) d0 <= 1e7; (C) agreement with the certificate's
    own P(h) (b8m_lc_K4.log "P_hyp") within 1 % for every component except beta on N0, where this implementation's
    bound must be the smaller one (the certificate's box atoms are wider there: tau_b's (4, 2) coefficient 276.8 over
    the box against 25.85 at the centre, ball wrapping); (D, with --lip) theta < 1 and the spectral radius within 0.01
    of the certificate's 0.7052.
    Mutations (each must make C or D fail): rho2 (the residual rho2 dropped), gF (the far fast-kernel gain halved),
    d0r4 (the exact rho4 part of the gauge offset dropped), phid (the arc angle phi_delta dropped from the slow and
    b-mode path factors).
"""
import json
import os
import sys
import time
from fractions import Fraction

from mpmath import iv, mp
from mpmath.libmp import fone, fzero

import p2_alg as G
from p2_alg import R, Lo, Bd, Q, Reg, bsup, inv_lo, lo_mono, lo_times, qinv, const, one_plus, from_desc, rmaxR
from p2_alg import up_iv, lo_iv, add_u, mul_u, div_u, sub_d, mul_d, pow_u, gt, fl
from p2_common import lo as LO, hi as HI
import p2_atoms as AT

OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate", "")
HERE = os.path.dirname(os.path.abspath(__file__))
K = 4
K2 = 2 * K + 2
TARGETS = ("N0", "Na", "L", "Nb", "F")
COMPS = ("V", "S", "b", "c")
PATH_P = {"N0": ("N0",), "Na": ("Ra", "Na"), "Nb": ("Ra", "Rb", "Nb"), "F": ("Ra", "Rb", "F")}
PATH_Q = {"N0": ("N0",), "Na": ("N0", "Na"), "L": ("N0", "Na", "L"), "Nb": ("N0", "Na", "Nb"), "F": ("N0", "Na", "Nb", "F")}
CONTAINS = {"N0": ("N0", "Ra"), "Na": ("Na", "Ra", "Ac"), "L": ("L",), "Nb": ("Nb", "Rb"), "F": ("F",)}
COLS = [("val", c, r) for c in COMPS for r in TARGETS] + [("der", c, r) for c in COMPS for r in TARGETS]
IDX = {k: i for i, k in enumerate(COLS)}


def ivr(x):
    return iv.mpf(x)


class Geo:
    def __init__(self, L0="73.89", kappa="0.3", kappa1="0.6", delta="0.05", xF="20", r1="0.05", xa="1.5"):
        self.L0 = ivr(L0)
        self.n0 = iv.exp(self.L0)
        kap, k1, de, xF_, r1_, xa_ = (ivr(v) for v in (kappa, kappa1, delta, xF, r1, xa))
        self.alpha = iv.pi / 4 + kap / 2
        self.sk = iv.sin(kap)
        self.salpha = iv.sin(self.alpha) / self.alpha
        self.d2 = de / 2
        # asin is not in mpmath.iv: evaluate it at 256 bits and widen by 1e-60 (far above the evaluation error)
        mp.prec = 256
        ph = mp.asin(mp.mpf(1) / 40)
        self.phid = iv.mpf([ph - mp.mpf(10) ** -60, ph + mp.mpf(10) ** -60])
        assert abs(HI(self.d2) - mp.mpf(1) / 40) < mp.mpf(10) ** -40   # delta/2 = 1/40
        n = self.n0
        Rout = iv.sqrt(k1 * n / (iv.cos(kap) * self.L0))
        self.rho_max = (1 + Rout) ** 2 / n
        self.xF, self.r1, self.xa = xF_, r1_, xa_
        d2 = self.d2
        th = lambda r: r / (1 + r)
        D = lambda x: lo_iv(x)
        U = lambda x: up_iv(x)
        box = lambda Xl, Xh, tl, th_, rho, far=False: dict(Xlo=D(Xl), Xhi=(None if Xh is None else U(Xh)), thlo=D(tl),
                                                            thhi=U(th_), rho=U(rho), far=far)
        self.boxes = {
            "N0": box(1, 1 + r1_, 0, th(r1_), (1 + r1_) ** 2 / n),
            "Na": box(1 + r1_, 1 + xa_, th(r1_), th(xa_), (1 + xa_) ** 2 / n),
            "L": box(2 - d2, 2 + d2, th(1 - d2), th(1 + d2), (2 + d2) ** 2 / n),
            "Ac": box(2 - d2, 2 + d2, th(1 - d2), th(1 + d2), (2 + d2) ** 2 / n),
            "Nb": box(1 + xa_, 1 + xF_, th(xa_), th(xF_), (1 + xF_) ** 2 / n),
            "Ra": box(1, 1 + xa_, 0, th(xa_), (1 + xa_) ** 2 / n),
            "Rb": box(1 + xa_, 1 + xF_, th(xa_), th(xF_), (1 + xF_) ** 2 / n),
            "F": box(1 + xF_, None, th(xF_), 1, self.rho_max, far=True)}
        # upper bounds of X^2/|x^2 - 1| per region (off the light-cone disc)
        lN = max(U((2 - d2) / d2), U((2 + d2) ** 2 / (d2 * iv.sqrt(2 * (1 - d2) ** 2 + 2 - d2 * d2))), U((2 + d2) / d2),
                 key=fl)
        self.ell = {"N0": U((1 + r1_) / (1 - r1_)), "Na": lN, "Nb": U((1 + xa_) / (xa_ - 1)), "F": U((1 + xF_) / (xF_ - 1)),
                    "Ra": U(2 * iv.sqrt(2)), "Rb": U(2 * iv.sqrt(2)), "Ac": lN, "L": None}
        Reg.set(self.boxes, self.n0)
        self.nf = U(n / (n - 1))


def mono(c, a=0, b=0, e=0):
    return Bd.mono(R(up_iv(c)) if not isinstance(c, (R, dict)) else c, a, b, e)


def fd(D, name, dname=None, d2name=None):
    g = lambda k: D.get(k) if k is not None else None
    return Q(from_desc(g(name)), from_desc(g(dname)) if dname else None, from_desc(g(d2name)) if d2name else None)


def fdv(D, base):
    return fd(D, base, "x" + base + "'", "x(x" + base + "')'")


def known(geo, D, consts):
    k = {}
    k["one"] = const(1)
    x = Q(mono(1, a=1, b=1), mono(1, a=1, b=1), mono(1, a=1, b=1))
    x.lo = lo_mono(fone, a=1, b=1)
    k["x"] = x
    k["x2"] = Q(mono(1, a=2, b=2), mono(2, a=2, b=2), mono(4, a=2, b=2))
    k["eps"] = Q(mono(1, e=-1), Bd(), Bd())
    k["eps1"] = Q(Bd.mono(R(geo.nf), e=-1), Bd(), Bd())
    k["nm1"] = Q(mono(1, e=1), Bd(), Bd())
    k["Om"] = Q(Bd.mono(R(mp.mpf(consts["absw"])._mpf_)), Bd(), Bd())
    k["half"] = const("0.5")
    k["two"] = const(2)
    for nm in ("V", "S", "S/x2", "beta", "beta/x2", "c-1", "cb", "cb/x2", "Y", "Dl", "tauD", "tauv", "taub", "rho4"):
        k[nm] = fdv(D, nm)
    k["xtauv"] = fd(D, "xtauv'", "x(xtauv')'")
    k["xc"] = fd(D, "xc-1'", "x(xc-1')'")
    k["Gam"] = fd(D, "Gamma", "xGamma'", "x(xGamma')'")
    dm1 = fd(D, "det-1", "xdet'", "x(xdet')'")
    det = one_plus(dm1)
    iDet = qinv(det)
    Neta = fd(D, "Neta", "xNeta'", "x(xNeta')'")
    etaT = Neta * iDet
    cK = one_plus(k["c-1"])
    iCK = qinv(cK)
    omG = one_plus(k["Gam"])
    x2m1 = Q(mono(1, a=2), mono(2, a=2, b=2), mono(4, a=2, b=2))
    x2m1.lo = lo_mono({r: (Lo(G.div_d(fone, geo.ell[r])) if geo.ell[r] is not None else Lo(up_iv("1e-300")))
                       for r in Reg.regs}, a=2)
    DK = x2m1 * omG
    DK.lo = lo_times(x2m1.lo, omG.lo)
    AK = k["nm1"] * omG + k["x2"] * etaT
    one_m = G.sub_d(fone, G.div_u(fone, lo_iv(geo.n0)))          # 1 - 1/n0 (lower bound)
    lob = {}
    eta_n = (k["x2"] * etaT).v * mono(1, e=-1)
    for r in Reg.regs:
        g = bsup(k["Gam"].v, r)
        l = Lo(mul_d(one_m, sub_d(fone, g.v))).minus(bsup(eta_n, r))
        if not l.positive():
            raise ValueError(f"A^K lower bound on {r}")
        lob[r] = (l, 0, 0, 1)
    AK.lo = lob
    Om = k["Om"]
    BK = k["one"] + k["cb"] * iCK + cK * Om * iCK
    bbK = k["one"] + k["beta"] * k["eps1"]
    DltX = iCK * k["Y"]
    Dlt = x * DltX
    e = (Om + BK + cK * k["cb/x2"]) * k["eps1"]
    ope = one_plus(e)
    ellx2 = bbK * (k["S/x2"] + DltX * iCK)
    q = ellx2 * qinv(ope) * DK * k["eps1"]
    rho2 = (k["x2"] * k["rho4"] * iCK * k["Y"] + cK * k["tauD"] + x * k["xtauv"] + x * k["xc"] * iCK * k["tauv"]
            + x * cK * Om * iCK * k["tauv"])
    if MUT == "rho2":
        rho2 = Q(Bd(), Bd(), Bd())
    rho3 = k["taub"] + bbK * iCK * k["tauv"] * (Dlt + k["Dl"]) * k["half"]
    k.update(det=det, iDet=iDet, Neta=Neta, etaT=etaT, cK=cK, iCK=iCK, omG=omG, x2m1=x2m1, DK=DK, iDK=qinv(DK), AK=AK,
             iAK=qinv(AK), BK=BK, bbK=bbK, DltX=DltX, Dlt=Dlt, e=e, ope=ope, iope=qinv(ope), ellx2=ellx2, q=q,
             xq=Q(q.d, q.d2), rho2=rho2, rho3=rho3, xc_d2=k["c-1"].d2)
    return k


# ------------------------------------------------------------------------------------------- kernel constants
def kernel(geo, k, D, nsub=1500):
    n0 = geo.n0
    eta = (k["x2"] * k["etaT"]).v
    gmax = {r: bsup(k["Gam"].v, r) for r in Reg.regs}
    hpv = {r: iv.mpf(mp.make_mpf(bsup(eta * Bd.mono(R(geo.nf), e=-1), r).v)) / (1 - iv.mpf(mp.make_mpf(gmax[r].v)))
           for r in ("Ra", "Rb", "F")}
    hmax = max(HI(v) for v in hpv.values())
    sstar = geo.sk - iv.mpf(hmax)
    if not LO(sstar) > 0:
        raise ValueError("s* not positive")
    sstar = iv.mpf(LO(sstar))
    gF_ = iv.mpf(mp.make_mpf(gmax["F"].v))
    HF = iv.mpf(mp.make_mpf(bsup(eta * mono(1, a=-2), "F").v)) * iv.mpf(mp.make_mpf(geo.ell["F"])) / (1 - gF_)
    qF = 1 + 2 * geo.sk / geo.xF ** 2 + 1 / geo.xF ** 4
    NF = (1 - 1 / n0) / (geo.rho_max * qF)
    NF = iv.mpf(LO(NF))
    lnKF = HF * HF / (4 * NF * geo.salpha ** 2)
    KF = iv.exp(lnKF)
    gF = KF * iv.sqrt(qF) * geo.rho_max * (n0 / (n0 - 1)) / sstar + KF * iv.sqrt(iv.pi / (NF * geo.salpha ** 2))
    if MUT == "gF":
        gF = gF / 2
    # bands N0, Na, Nb: 300 cells per piece, exact r-dependence of ||eta||, ||Gamma||, det at n0
    cN, cd, cg = D["Neta"]["coef"], D["det-1"]["coef"], D["Gamma"]["coef"]
    d2 = geo.d2
    al, sa, sk = geo.alpha, geo.salpha, geo.sk
    brk = [ivr(0), geo.r1, iv.mpf(LO(1 - d2)), iv.mpf(HI(1 + d2)), geo.xa, geo.xF]
    gN = {"N0": ivr(0), "Na": ivr(0), "Nb": ivr(0)}
    for a, b in zip(brk[:-1], brk[1:]):
        m = nsub // 5
        for i in range(m):
            ra, rb = a + (b - a) * i / m, a + (b - a) * (i + 1) / m
            rr = iv.mpf([LO(ra), HI(rb)])
            ev = lambda cf: sum((iv.mpf(c) * n0 ** (-j) * rr ** mm for (j, mm), c in cf.items()), ivr(0))
            Ueta = rr * rr * ev(cN) / (1 - ev(cd))
            Ug = ev(cg)
            if HI(b) <= LO(1 - d2):
                mr = 1 - rr * rr
            elif LO(a) >= HI(1 + d2):
                mr = rr * rr - 1
            else:
                mr = d2 * iv.sqrt(2 * rr * rr + 2 - d2 * d2)
            mr = iv.mpf(LO(mr))
            if not LO(mr) > 0:
                raise ValueError("arc distance")
            Hp = iv.mpf(HI(Ueta / ((1 - Ug) * mr)))
            Qa = iv.mpf(HI(1 + 2 * rr * rr * sk + rr * rr * rr * rr))
            Np = iv.mpf(LO((n0 - 1) * rr * rr / Qa))
            opt1 = Hp * al
            arc1 = al * iv.exp(opt1)
            if LO(Np) > 0:
                opt2 = Hp * Hp / (4 * Np * sa * sa)
                lnK = opt1 if HI(opt1) < HI(opt2) else opt2
                arc2 = iv.exp(opt2) * iv.sqrt(iv.pi / (Np * sa * sa))
                arcp = arc1 if HI(arc1) < HI(arc2) else arc2
            else:
                lnK, arcp = opt1, arc1
            g = iv.exp(lnK) * iv.sqrt(Qa) / ((n0 - 1) * sstar) + arcp
            reg = "N0" if HI(b) <= HI(geo.r1) else ("Na" if HI(b) <= HI(geo.xa) else "Nb")
            gN[reg] = g if HI(g) > HI(gN[reg]) else gN[reg]
    c1 = 1 / (2 - d2) + 2 / (1 - d2)
    gL = iv.mpf(mp.make_mpf(gmax["L"].v))
    etaL = iv.mpf(mp.make_mpf(bsup(eta, "L").v)) / ((1 - gL) * (1 - d2) * (2 - d2))
    NL = (n0 - 1) * (1 - c1 * d2) / 2 - etaL
    NL = iv.mpf(LO(NL))
    U = lambda v: R(up_iv(v))
    return dict(sstar=sstar, HF=HF, NF=NF, KF=KF, gF=U(gF), gN0=U(gN["N0"]), gNa=U(gN["Na"]), gNb=U(gN["Nb"]), c1=c1,
                etaL=etaL, NL=NL)


# ------------------------------------------------------------------------------------------- the operator T
def supLn(b):
    """L-entries with X, theta replaced by their bounds on L, n^e kept"""
    bx = Reg.regs["L"]
    out = Bd()
    for (bT, e), (a, c) in b.t["L"].items():
        xf = pow_u(bx["Xhi"], a) if a > 0 else pow_u(bx["Xlo"], a)
        th = pow_u(bx["thhi"], bT) if bT >= 0 else pow_u(bx["thlo"], bT)
        G._merge("L", out.t["L"], (0, e), 0, c.scale(mul_u(xf, th)))
    return out


def onlyL(v):
    return Bd.mono({r: (v if r == "L" else R(fzero)) for r in Reg.regs})


def hyp_regions(hc, summ):
    out = {r: hc[r] for r in TARGETS}
    out["Ra"] = (hc["N0"] + hc["Na"]) if summ else (hc["N0"] if gt(hc["N0"].v, hc["Na"].v) else hc["Na"])
    out["Rb"] = hc["Nb"]
    out["Ac"] = hc["Na"]
    return out


def wB(hc, b, summ):
    return Bd.mono(hyp_regions(hc, summ), a=K2, b=b, e=-(K + 1))


def der_weights():
    """m (and 1/m) per component: S: n w (n w/X^2 on F); b: n w; c, V: w"""
    out = {}
    for cp in COMPS:
        spec = {}
        for r in Reg.regs:
            if cp == "S":
                spec[r] = (K2 - (2 if r == "F" else 0), 1, -K)
            elif cp == "b":
                spec[r] = (K2, 1, -K)
            else:
                spec[r] = (K2, 1, -(K + 1))
        m = Bd()
        lob = {}
        for r, (a, b, e) in spec.items():
            mm = Bd.mono(R(fone), a=a, b=b, e=e)
            m.t[r] = dict(mm.t[r])
            lob[r] = lo_mono(fone, a=a, b=b, e=e)[r]
        out[cp] = (m, inv_lo(lob))
    return out


def evaluate(geo, k, kc, h, D0=None, summ=False, want_der=False):
    nf = geo.nf
    d2 = geo.d2
    d2u = up_iv(d2)
    iwS = inv_lo(lo_mono(fone, a=K2, b=1, e=-(K + 1)))
    iw1 = inv_lo(lo_mono(fone, a=K2, e=-(K + 1)))
    absw = k["Om"]
    one, x, x2, eps, eps1, nm1, Om, half, two = (k[s] for s in ("one", "x", "x2", "eps", "eps1", "nm1", "Om", "half", "two"))
    cK, iCK, SK, VK, bK, AK, iAK, DK, iDK, BK, bbK, YK, q, xq = (k[s] for s in (
        "cK", "iCK", "S", "V", "beta", "AK", "iAK", "DK", "iDK", "BK", "bbK", "Y", "q", "xq"))
    MK, Dlt, rho2, rho3, rho4, Gam = k["cb"], k["Dlt"], k["rho2"], k["rho3"], k["rho4"], k["Gam"]

    def build(RS, RbH, Rc, RV):
        Rb = RbH + q * RS
        Rb0 = q * RS
        c = cK + Rc
        c.lo = {r: (cK.lo[r][0].minus(bsup(Rc.v, r)), 0, 0, 0) for r in Reg.regs}
        for r in Reg.regs:
            if not c.lo[r][0].positive():
                raise ValueError(f"c lower bound on {r}")
        beta = bK + Rb
        if Rc.d is not None:
            c.d = cK.d + Rc.d                         # the class bound (F3)
        else:
            c.d = None
        iC = qinv(Q(c.v, c.d, None, c.lo))
        S, V, Y = SK + RS, VK + RV, YK + RV + RS
        dC = Rc
        dD = (cK * two + Rc) * Rc
        Dq = Q(DK.v + dD.v)
        lob = {}
        for r in Reg.regs:
            if r == "L":
                lob[r] = (Lo(up_iv("1e-300")), 0, 0, 0)    # never used: L-entries of D^-1 objects would explode
                continue
            (cl, a, b, e) = DK.lo[r]
            rel = bsup(dD.v * inv_lo({rr: DK.lo[rr] for rr in Reg.regs}), r)
            l = cl.times_one_minus(rel)
            if not l.positive():
                raise ValueError(f"D lower bound on {r}")
            lob[r] = (l, a, b, e)
        Dq.lo = lob
        Dq.d = (x2 * two).v + (c.v * c.d * R.c(2) if c.d is not None else Bd())
        if c.d is None:
            Dq.d = None
        iD = qinv(Dq)
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
        W2 = iD * qinv(x) * (c * Mh * S + x2 * Mh * iC * Y)
        xgS = iDK * AK * G1 + x * W2
        return dict(Rb=Rb, c=c, iC=iC, beta=beta, S=S, V=V, Y=Y, dC=dC, dD=dD, G1=G1, W2=W2, xgS=xgS, A0=A0, B0=B0,
                    dA0=dA0, dB0=dB0, Mh=Mh)

    def build_L(Rr, Rc, Fc):
        S, V, Y, c, iC, Mh = Rr["S"], Rr["V"], Rr["Y"], Rr["c"], Rr["iC"], Rr["Mh"]
        A0, B0, dA0, dB0, RV = Rr["A0"], Rr["B0"], Rr["dA0"], Rr["dB0"], Rr["RV"]
        s1 = G.div_u(fone, lo_iv(1 - d2))
        rcp = supLn(Fc.v) * s1                           # sup |R_c'| on L (x R_c' = F_c in the class, |x| >= 1 - d2)
        cKv, Rcv = supLn(cK.v), supLn(Rc.v)
        ddv = (cKv * R.c(2) + Rcv) * rcp                 # |dD/(x - 1)| (R_c(1) = 0, F1)
        gm = bsup(Gam.v, "L")
        lDt = Lo(lo_iv((2 - d2))).times_one_minus(gm).minus(bsup(ddv, "L"))
        if not lDt.positive():
            raise ValueError("Dt lower bound")
        iDt_v = lDt.inv()
        DtK_v = R(up_iv(2 + d2)) * (R(fone) + gm)
        dd_d = iDt_d = DtK_d = None
        if Fc.d is not None and Gam.d is not None:
            s1s = mul_u(s1, s1)
            rcpp = (supLn(Fc.d) + supLn(Fc.v)) * s1s * R.c(OPTS.get("rcpp", "1"))   # |R_c''| on L (class bound, F2)
            cKd = supLn(cK.d) * s1
            cKdd = (supLn(k["xc_d2"]) + supLn(cK.d)) * s1s
            ddpp = (cKdd * Rcv * R.c(2) + cKd * rcp * R.c(4) + cKv * rcpp * R.c(2) + rcp * rcp * R.c(2)
                    + Rcv * rcpp * R.c(2))
            dd_d = ddpp * R(up_iv((1 + d2) / 2))
            gmd = supLn(Gam.d)
            DtK_d = onlyL(R(up_iv(1 + d2)) * (R(fone) + gm)) + gmd * R(up_iv(2 + d2))
            iDt_d = (DtK_d + dd_d) * (iDt_v * iDt_v)
        iDt = Q(onlyL(iDt_v), iDt_d)
        ddL = Q(ddv, dd_d)
        DtK = Q(onlyL(DtK_v), DtK_d)
        inner = dA0 * S + ddL * iDt * (A0 * S + x2 * B0 * V) + x2 * (dB0 * V + BK * RV)
        G1L = iAK * (inner + rho2)
        G2L = iAK * DtK * iDt * (c * Mh * S + x2 * Mh * iC * Y)
        GL = G1L + G2L
        qxgSL = k["ellx2"] * k["iope"] * eps1 * AK * GL
        iDtK = Lo(lo_iv(2 - d2)).times_one_minus(gm).inv()
        return dict(GL=GL, qxgSL=qxgSL, lDt=lDt, iDtK=iDtK, ddv=bsup(ddv, "L"))

    def gbeta(Rr, RS, RV, qxgS):
        iC, S, Y, Rb, dC = Rr["iC"], Rr["S"], Rr["Y"], Rr["Rb"], Rr["dC"]
        Dl = x * iC * Y
        H = (S * S + Dl * Dl) * half
        dDl = x * iC * (RV + RS) + iC * dC * Dlt
        g3 = bbK * (RS * RS * half + (x * iC * RV + iC * dC * (x * iCK * RS + Dlt)) * Dlt + dDl * dDl * half)
        return Rb * (bK * two + Rb) + eps * Rb * H + g3 + xq * RS + qxgS + rho3

    def mixL(std, Lq):
        def m(a, b):
            if a is None or b is None:
                return None
            out = Bd({r: dict(a.t[r]) for r in Reg.regs})
            out.t["L"] = dict(b.t["L"])
            return out
        return Q(m(std.v, Lq.v), m(std.d, Lq.d), m(std.d2, Lq.d2))

    hS, hb, hc, hV = h["S"], h["b"], h["c"], h["V"]
    RSv = wB(hS, 1, summ)
    RbHv = wB(hb, 1, summ)
    Rcv0 = wB(hc, 0, summ)
    RVv = wB(hV, 0, summ)
    if D0 is None:
        # pass 1: the derivative bounds the equations give (the class carries them)
        RS = Q(RSv)
        RbH = Q(RbHv)
        Rc = Q(Rcv0)
        RV = Q(RVv, RSv * absw.v)
        R1 = build(RS, RbH, Rc, RV)
        R1["RV"] = RV
        Fc1 = Rc * R1["beta"] + cK * R1["Rb"] + rho4
        L1 = build_L(R1, Rc, Fc1)
        RS_d = (AK * iDK).v * (RS.v + R1["G1"].v) + x.v * R1["W2"].v
        RS_d.t["L"] = dict(Bd.mono(h["Sp"], a=K2, b=1, e=-K).t["L"])
        qxgS1 = mixL(q * R1["xgS"], L1["qxgSL"])
        gb1 = gbeta(R1, RS, RV, qxgS1)
        D0 = dict(S=RS_d, b=gb1.v + nm1.v * RbH.v, c=Fc1.v, V=RV.d)
    # pass 2
    RS = Q(RSv, D0["S"])
    RbH = Q(RbHv, D0["b"])
    Rc = Q(Rcv0, D0["c"])
    RV = Q(RVv, D0["V"])
    R2 = build(RS, RbH, Rc, RV)
    R2["RV"] = RV
    Fc2 = Rc * R2["beta"] + cK * R2["Rb"] + rho4
    L2 = build_L(R2, Rc, Fc2)
    G1, W2 = R2["G1"], R2["W2"]
    qxgS2 = mixL(q * R2["xgS"], L2["qxgSL"])
    gb = gbeta(R2, RS, RV, qxgS2)
    loc = G1.v * iwS
    integ = (G1.d + x.v * W2.v) * iwS
    gam = {"N0": kc["gN0"], "Na": kc["gNa"], "Nb": kc["gNb"], "F": kc["gF"]}
    cand = {}
    for r in ("N0", "Na", "Nb", "F"):
        lr = bsup(loc, r)
        cand[("S", r)] = [lr + gam[r] * bsup(integ, s) for s in PATH_P[r]]
    NL = kc["NL"]
    bx = Reg.regs["L"]
    ratS = div_u(mul_u(bx["thhi"], pow_u(bx["Xhi"], K2)), mul_d(bx["thlo"], G.pow_d(bx["Xlo"], K2)))
    rat1 = div_u(pow_u(bx["Xhi"], K2), G.pow_d(bx["Xlo"], K2))
    GL = L2["GL"]
    G2arc = bsup((iAK * DK * x * W2).v * iwS, "Ac")
    jarcs = [G2arc + gam["Na"] * bsup(integ, s) for s in PATH_P["Na"]]
    supGd = bsup(GL.d * iwS, "L").scale(G.div_u(fone, lo_iv(1 - d2)))
    GLv = bsup(GL.v * iwS, "L")
    inv_NL1 = G.div_u(fone, lo_iv(NL - 1))
    cand[("S", "L")] = [GLv + J.scale(ratS) + supGd.scale(mul_u(mul_u(ratS, d2u), inv_NL1)) for J in jarcs]
    lamt = bsup(AK.v * mono(1, e=-1), "L") * L2["iDtK"]
    candSp = [(lamt * (J.scale(G.div_u(fone, lo_iv(d2))) + supGd.scale(inv_NL1))).scale(ratS) for J in jarcs]
    gl = gb.v * Bd.mono(R(nf), e=-1) * iwS
    gi = gb.d * Bd.mono(R(nf), e=-1) * iwS
    phid_u = up_iv(geo.phid) if MUT != "phid" else fzero
    for r in TARGETS:
        fac = add_u(G.div_u(nf, lo_iv(geo.n0)), (fzero if r == "L" else phid_u))
        glr = bsup(gl, r)
        cand[("b", r)] = [glr + bsup(gi, s).scale(fac) for s in PATH_Q[r]]
    # gauge offset d0 = -int_0^1 F_c ds/s: N0 [0, r1], Na [r1, 1 - d2], L [1 - d2, 1]; rho4 exactly
    Fc_noR4 = Rc * R2["beta"] + cK * R2["Rb"]
    pieces = (("N0", ivr(0), geo.r1), ("Na", geo.r1, 1 - d2), ("L", 1 - d2, ivr(1)))
    d0 = R(fzero)
    for r, a, b in pieces:
        sr = bsup(Fc_noR4.v * iwS, r)
        d0 = d0 + sr.scale(up_iv(((1 + b) ** K2 - (1 + a) ** K2) / K2))
    r4 = fzero
    n0lo = iv.mpf(LO(geo.n0))
    for (j, m), cf in AT_DESC["rho4"]["coef"].items():
        if j < K + 1 or m < 1:
            raise ValueError("rho4: unexpected term")
        r4 = add_u(r4, up_iv(iv.mpf(cf) * n0lo ** (K + 1 - j) / m))
    if MUT != "d0r4":
        d0 = d0 + R(r4)
    slow = add_u(G.div_u(fone, mp.mpf(K2)._mpf_), phid_u)
    slowr = G.div_u(fone, mp.mpf(K2)._mpf_)
    hyS = hyp_regions(hS, summ)
    absf4 = mul_u(up_iv(4), mp.mpf(CONSTS["absf"])._mpf_)
    for r in TARGETS:
        sl = slowr if r == "L" else slow
        cand[("V", r)] = [d0.scale(absf4) + bsup(absw.v, r).scale(sl) * hyS[s] for s in PATH_Q[r]]
        if r == "L":
            cand[("c", r)] = [bsup(Fc2.v * iw1, "L").scale(mul_u(G.div_u(d2u, lo_iv(1 - d2)), rat1))]
        else:
            cand[("c", r)] = [d0 + bsup(Fc2.v * iwS, s).scale(sl) for s in PATH_Q[r]]
    P = {cp: {} for cp in COMPS}
    for (cp, r), lst in cand.items():
        P[cp][r] = rmaxR(lst)
    P["Sp"] = rmaxR(candSp)
    diag = dict(d0=d0, r4=R(r4), Jarc=rmaxR(jarcs), supGd=supGd, GL=GLv, NL=NL)
    for r in ("N0", "Na", "Nb", "F"):
        diag[f"loc_{r}"] = bsup(loc, r)
        diag[f"int_{r}"] = rmaxR([bsup(integ, s) for s in PATH_P[r]])
    out = dict(P=P, cand=cand, candSp=candSp, D0=D0, diag=diag)
    if want_der:
        Wt = der_weights()
        wSu = Bd.mono(R(fone), a=K2, b=1, e=-(K + 1))
        der = {}
        for r in TARGETS:
            imS, imb, imc, imV = Wt["S"][1], Wt["b"][1], Wt["c"][1], Wt["V"][1]
            if r != "L":
                der[("S", r)] = (bsup((AK * iDK).v * wSu * imS, r), bsup(((AK * iDK).v * G1.v + x.v * W2.v) * imS, r))
            der[("b", r)] = (bsup(nm1.v * wSu * imb, r), bsup(gb.v * imb, r))
            der[("c", r)] = (R(fzero), bsup(Fc2.v * imc, r))
            der[("V", r)] = (R(fzero), bsup(RS.v * absw.v * imV, r))
        out["der"] = der
    return out


AT_DESC = {}
CONSTS = {}


def setup(L0="73.89"):
    geo = Geo(L0)
    D, consts, rems = AT.atoms(K, "2e-6", "2e-6")
    AT_DESC.clear()
    AT_DESC.update(D)
    CONSTS.clear()
    CONSTS.update(consts)
    k = known(geo, D, consts)
    kc = kernel(geo, k, D)
    return geo, D, k, kc, rems


def h_from(rec):
    """class constants from a b8m_lc record's 'hyp' (floats of arb upper bounds; inflated by 1e-12)"""
    f = rec["hyp"]
    h = {cp: {r: R(up_iv(iv.mpf(repr(f[f"{cp}.{r}"])) * (1 + iv.mpf("1e-12")))) for r in TARGETS} for cp in COMPS}
    h["Sp"] = R(up_iv(iv.mpf(repr(f["Sp"])) * (1 + iv.mpf("1e-12"))))
    return h


def with_grad(h):
    """value columns: d h_{cp,r}/dt = h_{cp,r}"""
    out = {cp: {r: R(h[cp][r].v, {IDX[("val", cp, r)]: h[cp][r].v}) for r in TARGETS} for cp in COMPS}
    out["Sp"] = h["Sp"]
    return out


def lipschitz(geo, k, kc, h):
    """the 40 x 40 matrix by forward differentiation of the majorant"""
    o0 = evaluate(geo, k, kc, h, summ=True)
    D0 = o0["D0"]
    Wt = der_weights()
    hp = {(cp, r): bsup(D0[cp] * Wt[cp][1], r) for cp in COMPS for r in TARGETS}
    # derivative columns: D0[cp] + t h'_{cp,r} m_cp on the regions containing points of band r
    D0g = {}
    for cp in COMPS:
        B = D0[cp]
        add = Bd()
        for r in TARGETS:
            m = Wt[cp][0]
            for rr in CONTAINS[r]:
                for key, (a, c) in m.t[rr].items():
                    G._merge(rr, add.t[rr], key, a, R(fzero, {IDX[("der", cp, r)]: hp[(cp, r)].v}).scale(c.v))
        D0g[cp] = B + add
    hg = with_grad(h)
    o = evaluate(geo, k, kc, hg, D0=D0g, summ=True, want_der=True)
    M = {}
    for i, col in enumerate(COLS):
        M[col] = {}
    for cp in COMPS:
        for r in TARGETS:
            sv = {}
            for j in range(len(COLS)):
                best = fzero
                for cnd in o["cand"][(cp, r)]:
                    v = cnd.g.get(j, fzero)
                    best = v if gt(v, best) else best
                sv[j] = best
            for j, col in enumerate(COLS):
                M[col][("val", cp, r)] = G.div_u(sv[j], h[cp][r].v)
            if (cp, r) == ("S", "L"):
                for j, col in enumerate(COLS):
                    best = fzero
                    for cnd in o["candSp"]:
                        v = cnd.g.get(j, fzero)
                        best = v if gt(v, best) else best
                    M[col][("der", cp, r)] = G.div_u(best, hp[(cp, r)].v)
            else:
                A, rest = o["der"][(cp, r)]
                for j, col in enumerate(COLS):
                    t = add_u(mul_u(A.v, sv[j]), rest.g.get(j, fzero))
                    M[col][("der", cp, r)] = G.div_u(t, hp[(cp, r)].v)
    return M, hp


def perron(M):
    import numpy as np
    n = len(COLS)
    A = [[mp.make_mpf(M[COLS[j]][COLS[i]]) for j in range(n)] for i in range(n)]
    Af = np.array([[float(x) for x in row] for row in A])
    rho = max(abs(np.linalg.eigvals(Af)))
    v = np.ones(n)
    B = Af + 1e-14
    for _ in range(5000):
        w = B @ v
        v = w / w.max()
    V = [Fraction(float(max(x, 1e-30))) for x in v]
    Mq = [[Fraction(float(x)) * (1 + Fraction(1, 10 ** 15)) for x in row] for row in A]
    theta = max(sum((Mq[i][j] * V[j] for j in range(n)), Fraction(0)) / V[i] for i in range(n))
    return rho, theta


def main():
    t0 = time.time()
    geo, D, k, kc, rems = setup(OPTS.get("L0", "73.89"))
    print(f"atoms over the box: removed structural entries (Lemma F) {', '.join(f'{a} {mp.nstr(b, 3)}' for a, b in rems.items())}")
    print("kernel: " + ", ".join(f"{a} {mp.nstr(mp.make_mpf(v.v) if isinstance(v, R) else HI(v), 6)}" for a, v in kc.items()))
    cert = None
    for line in open(os.path.join(HERE, "..", "b8m_lc_K4.log")):
        r = json.loads(line)
        if r.get("closes") and not r.get("mutate"):
            cert = r
    h = h_from(cert)
    o = evaluate(geo, k, kc, h)
    P = o["P"]
    ok = True
    worst = 0.0
    rows = []
    for cp in COMPS:
        for r in TARGETS:
            p, hh = P[cp][r].f(), h[cp][r].f()
            ratio = p / hh
            worst = max(worst, ratio)
            ok &= gt(h[cp][r].v, P[cp][r].v)
            rows.append((f"{cp}.{r}", hh, p, ratio, cert["P_hyp"][f"{cp}.{r}"]))
    pSp, hSp = P["Sp"].f(), h["Sp"].f()
    ok &= gt(h["Sp"].v, P["Sp"].v)
    rows.append(("Sp", hSp, pSp, pSp / hSp, cert["P_hyp"]["Sp"]))
    print(f"closure at the certificate's h (b8m_lc_K4.log 'hyp'), this implementation's P(h):")
    for nm, hh, p, ratio, pc in rows:
        print(f"   {nm:5s}: h = {hh:.6e}, P(h) = {p:.6e} (ratio {ratio:.4f}); certificate's P(h) {pc:.6e} "
              f"(relative difference {abs(p - pc) / pc:.1e})")
    dg = o["diag"]
    print("diagnostics (this implementation vs b8m_lc_K4.log): " + ", ".join(
        f"{a} {fl(v.v) if isinstance(v, R) else float(HI(v)):.5g} ({cert['diag'].get(a, float('nan')):.5g})" for a, v in dg.items()))
    d0 = dg["d0"].f()
    okB = d0 < 1e7
    devs = {nm: (p - pc) / pc for nm, hh, p, ratio, pc in rows}
    okC = all(abs(v) <= 1e-2 for nm, v in devs.items() if nm != "b.N0") and devs["b.N0"] <= 0
    print(f"=> (A) closure P(h) < h: {ok} (largest ratio {worst:.4f}); (B) gauge offset d0 = {d0:.5e} <= 1e7: {okB}; "
          f"(C) agreement with the certificate's P(h): {okC} (largest |difference| outside beta.N0 "
          f"{max(abs(v) for nm, v in devs.items() if nm != 'b.N0'):.2e})")
    res = dict(closes=bool(ok), worst_ratio=worst, d0=d0, rows=rows, okC=okC, mutate=MUT or None)
    okD = True
    if "lip" in OPTS:
        t1 = time.time()
        M, hp = lipschitz(geo, k, kc, h)
        rho, theta = perron(M)
        okD = theta < 1 and abs(rho - 0.70521) <= 0.01
        print(f"Lipschitz matrix (forward differentiation): spectral radius {rho:.5f}; Perron-weight theta = "
              f"{float(theta):.5f} (< 1: {theta < 1}) [{time.time() - t1:.0f} s]; (D) {okD}")
        big = sorted(((fl(M[c][rw]), "/".join(rw), "/".join(c)) for c in COLS for rw in COLS if fl(M[c][rw]) > 1e-3),
                     reverse=True)
        for v, rw, c in big[:25]:
            print(f"   {rw} <- {c}: {v:.4e}")
        res.update(rho=rho, theta=float(theta), okD=okD, Mbig=[(rw, c, v) for v, rw, c in big])
        if not MUT:
            with open(os.path.join(HERE, "p2_lc_lip.json"), "w") as fh:
                json.dump({f"{'/'.join(rw)}<-{'/'.join(c)}": fl(M[c][rw]) for c in COLS for rw in COLS}, fh)
    with open(os.path.join(HERE, "p2_lc.json" if not MUT else f"p2_lc_mut_{MUT}.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(f"[{time.time() - t0:.0f} s]")
    allok = ok and okB and okC and okD
    if MUT:
        print(f"MUTATION {MUT}: a check must FAIL: {'caught' if not allok else 'NOT caught'}")
        return 0 if not allok else 3
    print(f"=> Lemma LC {'CONFIRMED' if allok else 'NOT confirmed'} by this implementation")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
