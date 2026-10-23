#!/usr/bin/env python3
"""Lemma B8 for the M = 1 model of b8m_model.py (b, c on {0, +-2}): the bootstrap estimates with explicit constants
(proof/B8_constants.md, sections 4-7) and the threshold n_0.

Bounds are on l1 norms over harmonics (operators: induced l1 norms), with the bound algebra of b8_bounds.py: upper
bounds of |f|, |x f'|, |x (x f')'| as sums of monomials c X^eX theta^eT rho^eR n^eN on the regions N (|x| <= x_F), R (the
ray arg x = alpha within N) and F (x_F <= |x| <= X_max - 1), every constant a python-flint ball.

The bootstrap quantities (hypothesis b, improved P(b)):
    |R_V| <= b_V rho^(K+1),  |R_S| <= b_S theta rho^(K+1),  |R^_beta| <= b_b theta rho^(K+1),  |R_c| <= b_c rho^(K+1),
with R = Z - Z^K, R^_beta = R_beta - q R_S (the decoupled b-mode, section 6), rho = (1+|x|)^2/n, theta = |x|/(1+|x|).
The domain: |x| <= X_max - 1 with X_max - 1 = R'_n = (kappa1 n/(cos(kappa) ln n))^(1/2) ("full", the working domain of
B5(1), proof/B5_inner.md section 3) or n^gamma0 ("arc").

    python3 b8m_bounds.py K L0 [--outer=full|arc] [--kappa=0.3] [--kappa1=0.6] [--gamma0=0.3] [--delta=0.05]
                              [--xF=5] [--rA=0] [--rW=0] [--d0=0] [--mutate=NAME]
    --d0: the normalization c(0) = c^K(0) + d with |d|_1 <= d0 n^-(K+1) (proof/B8_constants.md 8.4; default d = 0)
    L0 = ln(n_0).  --scan prints the smallest L0 (in steps) for which the bootstrap closes.
    mutations (each must make the bootstrap fail to close at an L0 where it closes): eta2 (the transport norm halved:
    must NOT be used to claim closure; instead it is checked that the kernel constant K_F changes), drop_rho2 (the
    S-residual dropped: the bound b_S must fall below the numerically observed remainder, see b8m_numcheck.py).
"""
import sys
from flint import arb, acb
import b8_alg as E             # compact bound algebra: B, Q, bsup, from_desc, qinv, ...
import b8m_atoms as AT

B, Q, REG = E.B, E.Q, E.REG
PI0 = None
up, lo = E.up, E.lo
PI = arb.pi()
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate", "")


class Geo:
    def __init__(self, K, L0, outer="full", kappa="0.3", kappa1="0.6", gamma0="0.3", delta="0.05", xF="20", r1="0.05",
                 rA="0", rW="0", atoms=None, arc=False):
        self.K = K
        self.L0 = arb(L0)
        self.n0 = self.L0.exp()
        self.kappa, self.kappa1, self.delta, self.xF = arb(kappa), arb(kappa1), arb(delta), arb(xF)
        self.gamma0, self.r1 = arb(gamma0), arb(r1)
        self.XF = 1 + self.xF
        self.alpha = PI / 4 + self.kappa / 2
        self.sk = self.kappa.sin()
        self.salpha = self.alpha.sin() / self.alpha
        self.phid = (self.delta / 2).asin()
        n = self.n0
        if outer == "full":
            self.Rout = (self.kappa1 * n / (self.kappa.cos() * n.log())).sqrt()
        else:
            self.Rout = n ** self.gamma0
        self.outer = outer
        # rho_max(n) = (1 + R_out(n))^2/n is decreasing in n for n >= n_0 (Lemma D of the notes)
        self.rho_max = up((1 + self.Rout) ** 2 / n)
        d2 = self.delta / 2
        r1 = self.r1
        lN = max(up((2 - d2) / d2), up((2 + d2) ** 2 / (d2 * (2 * (1 - d2) ** 2 + 2 - d2 * d2).sqrt())),
                 up((2 + d2) / d2))
        # ray arg x = alpha: |x^2 - 1|^2 = 1 + 2 r^2 sin(kappa) + r^4 >= 1 + r^4 and (1+r)^2/sqrt(1+r^4) <= 2 sqrt 2
        self.ell = {"N0": up((1 + r1) / (1 - r1)), "N1": lN, "R": up(2 * arb(2).sqrt()),
                    "F": up(self.XF / (self.xF - 1))}
        boxes = {
            "N0": dict(Xlo=arb(1), Xhi=up(1 + r1), thlo=arb(0), thhi=up(r1 / (1 + r1)), rho=up((1 + r1) ** 2 / n),
                       far=False),
            "N1": dict(Xlo=lo(1 + r1), Xhi=self.XF, thlo=lo(r1 / (1 + r1)), thhi=up(self.xF / self.XF),
                       rho=up(self.XF ** 2 / n), far=False),
            "R": dict(Xlo=arb(1), Xhi=self.XF, thlo=arb(0), thhi=up(self.xF / self.XF), rho=up(self.XF ** 2 / n),
                      far=False),
            "F": dict(Xlo=self.XF, Xhi=None, thlo=lo(self.xF / self.XF), thhi=arb(1), rho=self.rho_max, far=True)}
        if arc:
            # the inner arc |x| = n^gamma0 of B5 (a subset of F for n >= n_0), as its own region: X = 1 + n^gamma0 >= Xlo,
            # rho = (1 + n^gamma0)^2/n <= its value at n_0 (2 gamma0 < 1); evaluated by arc_sup below
            xa = n ** self.gamma0
            if not xa > self.xF:
                raise ValueError("inner arc below x_F")
            boxes["A"] = dict(Xlo=lo(1 + xa), Xhi=None, thlo=lo(xa / (1 + xa)), thhi=arb(1),
                              rho=up((1 + xa) ** 2 / n), far=True)
            self.ell["A"] = up((1 + xa) / (xa - 1))
        E.configure(boxes, n)
        if atoms is None:
            atoms = AT.atoms(K, rA, rW)
        self.D, self.C = atoms


# where the integrals run, for a target point in each region (section 5): the fast path P_x (ray, then arc at |x|)
# and the low path Q_x (radial, then an arc of angle <= phi_delta at |x|)
PATH_P = {"N0": ("N0",), "N1": ("R", "N1"), "F": ("R", "F")}
PATH_Q = {"N0": ("N0",), "N1": ("N0", "N1"), "F": ("N0", "N1", "F")}
XREG = ("N0", "N1", "F")


def fd(geo, name, dname=None, d2name=None):
    D = geo.D
    g = lambda k: D.get(k) if k is not None else None
    return E.from_desc(g(name), g(dname), g(d2name))


def fdv(geo, base):
    return fd(geo, base, "x" + base + "'", "x(x" + base + "')'")


def one_plus(geo, small):
    """the operator I + small (norm <= 1 + |small|), with lower bound 1 - sup|small| per region"""
    q = E.const(1) + small
    q.lo = E.lower_one_minus(small.v)
    return q


def known(geo):
    D, C = geo.D, geo.C
    n0 = geo.n0
    k = {}
    one = E.const(1)
    x = Q(B.mono(1, eX=1, eT=1), B.mono(1, eX=1, eT=1), B.mono(1, eX=1, eT=1))
    x.lo = E.lo_mono(1, eX=1, eT=1)
    x2 = Q(B.mono(1, eX=2, eT=2), B.mono(2, eX=2, eT=2), B.mono(4, eX=2, eT=2))
    eps = Q(B.mono(1, eN=-1), B(), B())
    nf = up(n0 / (n0 - 1))
    eps1 = Q(B.mono(nf, eN=-1), B(), B())
    nm1 = Q(B.mono(1, eN=1), B(), B())
    Om = E.const(C["absw"])            # || diag(i mu omega) || = |omega| for M = 1
    half = E.const(arb("0.5"))
    two = E.const(2)
    for nm in ("V", "S", "S/x2", "beta", "beta/x2", "c-1", "cb", "cb/x2", "Y", "Dl", "tauD", "tauv", "taub", "rho4"):
        k[nm] = fdv(geo, nm)
    k["xtauv"] = fd(geo, "xtauv'", "x(xtauv')'")
    k["xc"] = fd(geo, "xc-1'", "x(xc-1')'")
    k["Gam"] = fd(geo, "Gamma", "xGamma'", "x(xGamma')'")
    dm1 = fd(geo, "det-1", "xdet'", "x(xdet')'")
    det = one_plus(geo, dm1)
    iDet = E.qinv(det)
    Neta = fd(geo, "Neta", "xNeta'", "x(xNeta')'")
    if MUT == "eta2":
        Neta = Neta * arb("0.5")
    etaT = Neta * iDet
    cK = one_plus(geo, k["c-1"])
    iCK = E.qinv(cK)
    omG = one_plus(geo, k["Gam"])
    x2m1 = Q(B.mono(1, eX=2), B.mono(2, eX=2, eT=2), B.mono(4, eX=2, eT=2))
    x2m1.lo = E.lo_mono({r: 1 / geo.ell[r] for r in REG}, eX=2)
    DK = x2m1 * omG
    DK.lo = E.scale_lo(x2m1.lo, omG.lo)
    AK = nm1 * omG + x2 * etaT
    rel = k["Gam"].v + (one + k["Gam"]).v * B.mono(1, eN=-1) + (x2 * etaT).v * B.mono(1, eN=-1)
    AK.lo = E.scale_lo(E.lower_one_minus(rel), E.lo_mono(1, eN=1))
    BK = one + k["cb"] * iCK + cK * Om * iCK
    bbK = one + k["beta"] * eps1
    DltX = iCK * k["Y"]
    Dlt = x * DltX
    e = (Om + BK + cK * k["cb/x2"]) * eps1
    ope = one_plus(geo, e)
    ellx2 = bbK * (k["S/x2"] + DltX * iCK)
    q = ellx2 * E.qinv(ope) * DK * eps1
    # rho2 = x^2 M_rho4 C^-1 Y - C tau_D - x^2 tau_v' + x (x c') C^-1 tau_v - x C Om C^-1 tau_v (b8m_identities.py, I1).
    # The first term comes from x c^K' = [c^K beta^K] + rho4 for the truncated series; it was missing up to commit
    # 9d6e013 (b8m_identities.py --mutate=rho2old shows the identity failing without it).
    rho2 = (x2 * k["rho4"] * iCK * k["Y"] + cK * k["tauD"] + x * k["xtauv"] + x * k["xc"] * iCK * k["tauv"]
            + x * cK * Om * iCK * k["tauv"])
    if MUT == "drop_rho2":
        rho2 = E.const(0)
    rho3 = k["taub"] + bbK * iCK * k["tauv"] * (Dlt + k["Dl"]) * half
    k.update(one=one, x=x, x2=x2, eps=eps, eps1=eps1, nm1=nm1, Om=Om, half=half, two=two, det=det, iDet=iDet,
             etaT=etaT, cK=cK, iCK=iCK, omG=omG, x2m1=x2m1, DK=DK, iDK=E.qinv(DK), AK=AK, iAK=E.qinv(AK), BK=BK,
             bbK=bbK, DltX=DltX, Dlt=Dlt, e=e, q=q, xq=Q(q.d, q.d2), rho2=rho2, rho3=rho3, Neta=Neta)
    return k


# ------------------------------------------------------------------ Lemma K: the fast kernel (matrix version)
def kernel_consts(geo, k, nsub=1500):
    D = geo.D
    n0 = geo.n0
    nf = up(n0 / (n0 - 1))
    # eta = x^2 eta~; ||(n-1) H'|| <= ||eta|| / (1 - ||Gamma||)
    eta = k["x2"] * k["etaT"]
    gmax = {r: E.bsup_r(k["Gam"].v, r) for r in REG}
    hp = {r: up(E.bsup_r(eta.v * B.mono(nf, eN=-1), r) / (1 - gmax[r])) for r in ("R", "F")}
    sstar = lo(geo.sk - max(hp.values()))
    if not sstar > 0:
        raise ValueError("ray decay margin s_* not positive")
    HF = up(E.bsup_r(eta.v * B.mono(1, eX=-2), "F") * geo.ell["F"] / (1 - gmax["F"]))
    qF = up(1 + 2 * geo.sk / geo.xF**2 + 1 / geo.xF**4)
    NF = lo((1 - 1 / n0) / (geo.rho_max * qF))
    lnKF = up(HF * HF / (4 * NF * geo.salpha**2))
    KF = up(lnKF.exp())
    gF = up(KF * qF.sqrt() * geo.rho_max * nf / sstar + KF * (PI / (NF * geo.salpha**2)).sqrt())
    # region N: exact r-dependence at n = n_0
    cN, cd, cg = D["Neta"]["coef"], D["det-1"]["coef"], D["Gamma"]["coef"]
    if MUT == "eta2":
        cN = {kk: v * arb("0.5") for kk, v in cN.items()}
    d2 = geo.delta / 2
    al, sk, sa = geo.alpha, geo.sk, geo.salpha
    brk = [arb(0), geo.r1, lo(1 - d2), up(1 + d2), geo.xF]
    gN = {"N0": arb(0), "N1": arb(0)}
    KN = arb(0)
    for a, b in zip(brk[:-1], brk[1:]):
        m = nsub // 3
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
            reg = "N0" if float(b) <= float(geo.r1) + 1e-15 else "N1"
            gN[reg] = g if g > gN[reg] else gN[reg]
            KN = Kr if Kr > KN else KN
    return dict(sstar=sstar, HF=HF, NF=NF, KF=KF, gF=gF, gN0=gN["N0"], gN1=gN["N1"], KN=KN)


# ------------------------------------------------------------------ the improvement map
def improve(geo, k, kc, b, verbose=False, full=False):
    K = geo.K
    n0 = geo.n0
    nf = up(n0 / (n0 - 1))
    w1 = B.mono(1, eX=2 * K + 2, eN=-(K + 1))
    wS = B.mono(1, eX=2 * K + 2, eT=1, eN=-(K + 1))
    iwS = E.inv_lo(E.lo_mono(1, eX=2 * K + 2, eT=1, eN=-(K + 1)))
    absw = geo.C["absw"]
    one, x, x2, eps, eps1, nm1, Om, half, two = (k[s] for s in ("one", "x", "x2", "eps", "eps1", "nm1", "Om", "half",
                                                                 "two"))
    cK, iCK, SK, VK, bK, AK, iAK, DK, iDK, BK, bbK, YK, q, xq = (k[s] for s in (
        "cK", "iCK", "S", "V", "beta", "AK", "iAK", "DK", "iDK", "BK", "bbK", "Y", "q", "xq"))
    MK, Dlt, DlK, rho2, rho3, rho4 = k["cb"], k["Dlt"], k["Dl"], k["rho2"], k["rho3"], k["rho4"]

    def build(RS, RbH, Rc, RV):
        Rb = RbH + q * RS
        Rb0 = q * RS
        c = cK + Rc
        c.lo = B()
        for r in REG:
            (kk, (aa, cc)), = cK.lo.t[r].items()
            v = lo(cc - E.bsup_r(Rc.v, r))
            if not v > 0:
                raise ValueError("c lower bound")
            c.lo.t[r][kk] = (aa, v)
        beta = bK + Rb
        c.d = (c * beta).v if beta.v is not None else None          # |x c'| = |[c beta]|
        c.d2 = None
        iC = E.qinv(Q(c.v, c.d, None, c.lo))
        S, V, Y = SK + RS, VK + RV, YK + RV + RS
        dC = Rc
        dD = (cK * two + Rc) * Rc
        D = Q(DK.v + dD.v)
        D.lo = B()
        for r in REG:
            (kk, (aa, cc)), = DK.lo.t[r].items()
            rel = E.bsup_r(dD.v * E.inv_lo(DK.lo), r)
            v = lo(cc * (1 - rel))
            if not v > 0:
                raise ValueError("D lower bound")
            D.lo.t[r][kk] = (aa, v)
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
        return dict(Rb=Rb, c=c, iC=iC, beta=beta, S=S, V=V, Y=Y, dC=dC, dD=dD, D=D, G1=G1, W2=W2, xgS=xgS)

    def gbeta(R, RS, RV):
        iC, S, Y, Rb, dC = R["iC"], R["S"], R["Y"], R["Rb"], R["dC"]
        Dl = x * iC * Y
        H = (S * S + Dl * Dl) * half
        dDl = x * iC * (RV + RS) + iC * dC * Dlt
        g3 = bbK * (RS * RS * half + (x * iC * RV + iC * dC * (x * iCK * RS + Dlt)) * Dlt + dDl * dDl * half)
        gb = Rb * (bK * two + Rb) + eps * Rb * H + g3 + xq * RS + q * R["xgS"] + rho3
        return gb

    # ---- pass 1: values
    RS = Q(wS * b["S"])
    RbH = Q(wS * b["b"])
    Rc = Q(w1 * b["c"])
    RV = Q(w1 * b["V"], wS * (absw * b["S"]))
    R1 = build(RS, RbH, Rc, RV)
    RS_d = (AK * iDK).v * (RS.v + R1["G1"].v) + x.v * R1["W2"].v
    gb1 = gbeta(R1, RS, RV)
    RbH_d = gb1.v + nm1.v * RbH.v
    Rc_d = Rc.v * R1["beta"].v + cK.v * R1["Rb"].v + rho4.v
    # ---- pass 2
    RS = Q(wS * b["S"], RS_d)
    RbH = Q(wS * b["b"], RbH_d)
    Rc = Q(w1 * b["c"], Rc_d)
    R2 = build(RS, RbH, Rc, RV)
    G1, W2 = R2["G1"], R2["W2"]
    gb = gbeta(R2, RS, RV)
    if G1.d is None or gb.d is None:
        raise ValueError("missing derivative bound")
    loc = G1.v * iwS
    integ = (G1.d + x.v * W2.v) * iwS
    gam = {"N0": kc["gN0"], "N1": kc["gN1"], "F": kc["gF"]}
    bS = max((up(E.bsup_r(loc, r) + gam[r] * E.bsup(integ, PATH_P[r])) for r in XREG), key=float)
    gl = gb.v * B.mono(nf, eN=-1) * iwS
    gi = gb.d * B.mono(nf, eN=-1) * iwS
    fac = up(nf / n0 + geo.phid)          # radial part: int_0^r (t/r)^(n-1) dt/t = 1/(n-1) <= nf/n_0
    bbn = max((up(E.bsup_r(gl, r) + fac * E.bsup(gi, PATH_Q[r])) for r in XREG), key=float)
    slow = up(arb(1) / (2 * K + 2) + geo.phid)
    bV = up(absw * slow * b["S"])
    bc = up(slow * E.bsup((Rc.v * R2["beta"].v + cK.v * R2["Rb"].v + rho4.v) * iwS, ("N0", "N1", "F")))
    # normalization c(0) = c^K(0) + d, |d|_1 <= d0 n^-(K+1) (proof/B8_constants.md section 8.4, proof/B_chain.md
    # section 4): R_V(0) = [d 2F] and R_c(0) = d enter the slow integrals; since rho^(K+1) >= n^-(K+1) everywhere,
    # P_V gains |2F|_1 d0 <= 4 max|F_mu| d0 (M = 1) and P_c gains d0
    d0 = arb(OPTS.get("d0", "0"))
    if d0 > 0:
        bV = up(bV + 4 * geo.C["absf"] * d0)
        bc = up(bc + d0)
    diag = {}
    if verbose:
        for r in XREG:
            diag[f"loc_{r}"] = E.bsup_r(loc, r)
            diag[f"int_{r}"] = E.bsup(integ, PATH_P[r])
            diag[f"gl_{r}"] = E.bsup_r(gl, r)
            diag[f"gi_{r}"] = E.bsup(gi, PATH_Q[r])
    if full:
        # the hypothesis bounds with their derivative bounds (|x R'|), for the conversion on the inner arc (b8m_h8.py)
        return dict(V=bV, S=bS, b=bbn, c=bc), diag, dict(RS=RS, RbH=RbH, Rc=Rc, RV=RV, R2=R2)
    return dict(V=bV, S=bS, b=bbn, c=bc), diag


def bootstrap(geo, iters=80, verbose=False, slack="1.25"):
    k = known(geo)
    kc = kernel_consts(geo, k)
    b = dict(V=arb(0), S=arb(0), b=arb(0), c=arb(0))
    for it in range(iters):
        try:
            nb, _ = improve(geo, k, kc, b)
        except ValueError as ex:
            return False, b, None, kc, {"fail": str(ex)}
        done = all(abs(float(nb[s]) - float(b[s])) <= 1e-7 * (1 + float(nb[s])) for s in b)
        b = nb
        if done:
            break
        if any(float(b[s]) > 1e40 for s in b):
            return False, b, None, kc, {"fail": "diverges"}
    trial = {s: up(b[s] * arb(slack)) for s in b}
    try:
        Pt, dg = improve(geo, k, kc, trial, verbose=verbose)
    except ValueError as ex:
        return False, trial, None, kc, {"fail": str(ex)}
    closes = all(Pt[s] < trial[s] for s in b)
    return closes, trial, Pt, kc, dg


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    K = int(pos[0]) if pos else 3
    L0 = pos[1] if len(pos) > 1 else "74"
    kw = {kk: OPTS[kk] for kk in ("outer", "kappa", "kappa1", "gamma0", "delta", "xF", "r1", "rA", "rW")
          if kk in OPTS}
    if "scan" in OPTS:
        at = AT.atoms(K, OPTS.get("rA", "0"), OPTS.get("rW", "0"))
        for L in [float(v) for v in OPTS["scan"].split(",")]:
            g = Geo(K, str(L), atoms=at, **kw)
            ok, b, Pb, kc, dg = bootstrap(g)
            print(f"K = {K}, outer = {kw.get('outer', 'full')}, ln n0 = {L}: rho_max = {float(g.rho_max):.4g}, "
                  f"closes = {ok}, b = " + ", ".join(f"{s} {float(v):.4g}" for s, v in b.items())
                  + (f"  ({dg.get('fail')})" if not ok and dg else ""), flush=True)
        sys.exit(0)
    g = Geo(K, L0, **kw)
    ok, b, Pb, kc, dg = bootstrap(g, verbose=True)
    print(f"K = {K}, outer = {g.outer}, ln n0 = {L0}: rho_max = {float(g.rho_max):.5g}, R_out = {float(g.Rout):.4g}")
    print("  kernel: " + ", ".join(f"{a} {float(v):.4g}" for a, v in kc.items()))
    print(f"  closes: {ok};  hypothesis b = " + ", ".join(f"{s} {float(v):.5g}" for s, v in b.items()))
    if Pb is not None:
        print("  P(b)  = " + ", ".join(f"{s} {float(v):.5g}" for s, v in Pb.items()))
        print("  diag: " + ", ".join(f"{a} {float(v):.4g}" for a, v in dg.items()))
    elif dg:
        print("  ", dg)
