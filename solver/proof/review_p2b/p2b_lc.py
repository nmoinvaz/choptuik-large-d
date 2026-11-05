#!/usr/bin/env python3
"""P2b review: a second implementation of Lemma LC's bound algebra (B_chain.md 13): the closure P(h) < h of the class
K_h and the Lipschitz matrix of T in the h-weighted norm, written from the statements and derivations of
B_chain.md 13 and B8_constants.md 1-6 (Lemma R, Lemma K), NOT from solver/proof/b8_alg.py, b8m_bounds.py, b8m_lc.py
or review_p2/p2_lc_fix.py (none of which was opened before this script's numbers were committed).

Arithmetic: mpmath interval arithmetic (p2b_alg.py), exact formal series (p2b_series.py). Lipschitz matrix: forward
derivatives of the majorant (p2b_alg.py docstring), not secants. The class is the one of B_chain.md 13.1 with P2's
findings F1-F3 built in from the start: R_c(1) = 0 is a class condition (F1); on L the class bounds
|x (x R_c')'| <= h_cc n w and this is a 41st norm component (F2); |x c'| is bounded by the class bound
|x c^K'| + |x R_c'|, never by the c-equation (F3).

What each output is (per band N0, Na, L, Nb, F; w1 = rho^5, w = theta rho^5, n w (n w/X^2 on F for S)):
  P_V, P_c = sup |T_V|/w1, |T_c|/w1;  P_S, P_beta = sup |T_S|/w, |T_beta|/w;  D_V, D_c = sup |x T'|/w;
  D_S = sup |x T_S'|/(n w) (n w/X^2 on F; h_Sp on L);  D_beta = sup |x T_beta'|/(n w);  CC (L only).
The closure holds when every output is below its class constant. d0 = n^5 sup |d(U)|.

    python3 p2b_lc.py [--mode=cells|box] [--ln_n0=73.89] [--mutate=...] [--hfrom=FILE] [--log=FILE]
      --mode=box: one cell per band (like the boxes of B8_constants.md 1.3); default cells (finer suprema)
      --hfrom=FILE: evaluate at given constants (JSON {"h": {...}}) instead of iterating
"""
import json
import os
import sys
import time

from mpmath import iv, mp

import p2b_alg as Al
import p2b_series as Ser
from p2b_alg import (Bd, Jet, Cell, KJet, Poly, jadd, jmul, jprod, jscale, jinv, jconst, jmono, badd, bmul, bscale,
                     bto, bnum, binvlb, bmax, up, lo, mpup, mplo, ZEROI, ONEI, HD, PHID, ALPHA, SALPHA, SK, KAPPA)

HERE = os.path.dirname(os.path.abspath(__file__))
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
CRUDE_D = "crude_d" in OPTS      # the offset terms as in B8_constants.md 8.4: rho^(K+1) >= n^-(K+1), no X^-10
NOTESH = "notesH" in OPTS        # H_F as written in Lemma K.3: |eta|/X^2 <= |eta~| and X^2/|x^2-1| <= l_F = 21/19

COMP = ("V", "S", "B", "C")
BANDS = ("N0", "Na", "L", "Nb", "F")
W1, W, NW, NWF = (0, 0, 5), (0, 1, 5), (2, 1, 4), (0, 1, 4)
DW = {"V": W, "C": W, "B": NW}          # derivative weights (S: NW, NWF on F)


def vidx(c, b):
    return COMP.index(c) * 5 + BANDS.index(b)


def didx(c, b):
    return 20 + vidx(c, b)


CCIDX = 40
NIDX = 41


def idxname(i):
    if i == CCIDX:
        return "cc,L"
    k = "h" if i < 20 else "d"
    j = i % 20
    return f"{k}{COMP[j // 5]},{BANDS[j % 5]}"


# ---------------------------------------------------------------- geometry
def geom_points(a, b, n):
    a, b = mp.mpf(a), mp.mpf(b)
    return [a * (b / a) ** (mp.mpf(k) / n) for k in range(n + 1)]


def make_cells(st, mode):
    cells = []

    def add(band, pts, far=False, aux=False):
        for a, b in zip(pts[:-1], pts[1:]):
            kind = "sector"
            c = Cell(st, band, kind, a, b)
            c.aux = aux
            cells.append(c)
            if not aux:
                r = Cell(st, band, "ray", a, b)
                r.aux = False
                cells.append(r)

    if mode == "box":
        # one box per band (B8_constants.md 1.3) and, for the paths through the disc's arc Ac only, the annulus
        # 1 - delta/2 <= |x| <= 1 + delta/2 of Na (aux: not used for Na's own supremum or the radial sums)
        add("N0", [0, "0.05"])
        add("Na", ["0.05", "1.5"])
        add("Na", ["0.975", "1.025"], aux=True)
        add("Nb", ["1.5", "20"])
        add("F", ["20", None])
    else:
        add("N0", [0, "1e-12", "1e-9", "1e-6", "1e-4", "1e-3", "3e-3", "0.01", "0.02", "0.03", "0.04", "0.05"])
        lowpts = geom_points("0.05", "0.975", 28)
        lowpts[-1] = mp.mpf("0.975")
        mid = [mp.mpf("0.9875"), mp.mpf(1), mp.mpf("1.0125"), mp.mpf("1.025")]
        hipts = geom_points("1.025", "1.5", 10)[1:]
        add("Na", [str(p) for p in lowpts] + [str(p) for p in mid] + [str(p) for p in hipts])
        add("Nb", [str(p) for p in geom_points("1.5", "20", 30)])
        add("F", ["20", "29", "49", "99", "999", "1e5", "1e8", None])
    Lc = Cell(st, "L", "L", 0, 0)
    Lc.aux = False
    cells.append(Lc)
    return cells


# ---------------------------------------------------------------- known quantities per cell
class Known:
    NAMES = ("V", "S", "Sx2", "Y", "Dl", "beta", "betax2", "cm1", "cb", "cbx2", "tauD", "tauv", "taub", "rho4",
             "detrho2", "C", "Cm1", "Bb", "Bbx2", "Gam", "detm1", "Neta")

    def __init__(self, st, T):
        self.st = st
        self.kj = {nm: KJet(nm, T[nm]) for nm in self.NAMES}
        self.absA = iv.mpf(T["_absA"])
        self.absW = iv.mpf(T["_absW"])
        # rho4 integrated along Q_x: radial part int_0^r t^m dt/t = r^m/m, arc part <= phi_delta r^m
        cols = T["rho4"]
        self.rho4int = Poly("rho4int", [[(j, m, iv.mpf(C) * (1 / iv.mpf(m) + PHID)) for (j, m, C) in cols[0]]])
        self.rho4int.cols = [[(j, m, up(C)) for (j, m, C) in self.rho4int.cols[0]]]
        # the exact rho4 part of d0: n^5 int_0^1 |rho4| ds/s <= sum C nu0^(j-5)/m
        r4 = ZEROI
        for (j, m, C) in cols[0]:
            if m < 1 or j < 5:
                raise ValueError("rho4 structure")
            r4 = r4 + iv.mpf(C) * st.nu0 ** (j - 5) / m
        self.d0_rho4 = up(r4)
        if MUT == "rho4":
            self.d0_rho4 = ZEROI
        self.cache = {}

    def at(self, cell):
        if id(cell) in self.cache:
            return self.cache[id(cell)]
        st = self.st
        k = {nm: self.kj[nm](cell) for nm in self.NAMES}
        if MUT == "tauD":
            k["detrho2"] = jscale(k["detrho2"], iv.mpf("1.5"))
        if MUT == "taub":
            k["taub"] = jscale(k["taub"], iv.mpf("1.5"))
        one = jconst(ONEI)
        om = jconst(self.absW)
        x = jmono(1, 1, 0, 1, 1)
        x2 = jmono(2, 2, 0, 2, 4)
        eps1 = jconst(st.c1mnu, (-2, 0, 1))       # 1/(n - 1) <= nu/(1 - nu0), nu = rho X^-2
        nm1 = jconst(ONEI, (2, 0, -1))           # n - 1 <= n = X^2/rho
        K_ = dict(k)
        K_.update(one=one, om=om, x=x, x2=x2, eps1=eps1, nm1=nm1)
        CKinv = jinv(cell, k["Cm1"])
        detinv = jinv(cell, k["detm1"])
        etat = jmul(cell, k["Neta"], detinv)                      # |eta~| <= |det eta~|/|det|
        eps1eta = jprod(cell, eps1, x2, etat)                     # |eps1 eta| = eps1 |x|^2 |eta~|
        Gam = k["Gam"]
        AKinv = jmul(cell, eps1, jinv(cell, jadd(cell, Gam, eps1eta)))   # A^K = (n-1)(I - Gamma + eps1 eta)
        AKn = jadd(cell, one, Gam, eps1eta)                        # |A^K|/(n - 1)
        AK = jmul(cell, nm1, AKn)
        IGinv = jinv(cell, Gam)
        Dt = jprod(cell, x, CKinv, k["Y"])                         # Delta~ = x C^K^-1 Y^K
        Dtx = jmul(cell, CKinv, k["Y"])                            # Delta~/x
        bK = jadd(cell, one, jmul(cell, eps1, k["beta"]))          # |b^K|_1 <= 1 + eps1 |beta^K|
        cK = jadd(cell, one, k["cm1"])                             # |c^K|_1
        ellx2 = jmul(cell, bK, jadd(cell, k["Sx2"], jmul(cell, Dtx, CKinv)))   # |ell/x^2|
        BK = jadd(cell, one, jmul(cell, k["Bb"], CKinv), jprod(cell, om, k["C"], CKinv))
        e = jmul(cell, eps1, jadd(cell, jmul(cell, k["C"], k["Bbx2"]), om, BK))
        Ieinv = jinv(cell, e)
        if cell.kind == "L":
            x2m1 = Jet(Bd((0, 0, 0), HD * (2 + HD)), Bd((2, 2, 0), 2 * ONEI), Bd((2, 2, 0), 4 * ONEI))
        else:
            x2m1 = Jet(Bd((2, 0, 0), cell.d2hi), Bd((2, 2, 0), 2 * ONEI), Bd((2, 2, 0), 4 * ONEI))
        DK = jmul(cell, x2m1, jadd(cell, one, Gam))
        q = jprod(cell, eps1, ellx2, Ieinv, DK)                    # q = ell (I+e)^-1 D^K/((n-1) x^2)
        qLam = jprod(cell, ellx2, Ieinv, AKn)                      # q Lambda = (ell/x^2)(I+e)^-1 eps1 A^K
        rho2 = jmul(cell, k["detrho2"], detinv)
        rho3 = jadd(cell, k["taub"], jscale(jprod(cell, bK, CKinv, k["tauv"], jadd(cell, Dt, k["Dl"])),
                                            iv.mpf("0.5")))
        K_.update(CKinv=CKinv, etat=etat, eps1eta=eps1eta, AKinv=AKinv, AKn=AKn, AK=AK, IGinv=IGinv, Dt=Dt, bK=bK,
                  cK=cK, ellx2=ellx2, BK=BK, q=q, qLam=qLam, rho2=rho2, rho3=rho3)
        # scalar numbers for the kernel: sup |eta~|, |Gamma|, |x|^2/|x^2-1| (Lemma K)
        K_["n_etat"] = bnum(etat[0], cell).v
        K_["n_Gam"] = bnum(Gam[0], cell).v
        K_["n_eps1eta"] = bnum(eps1eta[0], cell).v
        if cell.kind != "L":
            # |Lambda| <= ((n-1)/|x^2 - 1|)(1 + |eps1 eta|/(1 - |Gamma|))
            lam = bmul(bmul(nm1[0], Bd((-2, 0, 0), up(1 / iv.mpf(cell.d2lo)))),
                       badd(cell, one[0], bmul(eps1eta[0], IGinv[0])))
            K_["Lam"] = lam
        self.cache[id(cell)] = K_
        return K_


# ---------------------------------------------------------------- inputs
JSEEDS = None      # Lemma J: {component: (shape weight, factor per cell)}: value inputs seeded with the shape s_j
PCIDX = "pc"


def seed_bd(w, val, idx, ad, cell=None):
    if JSEEDS is not None:
        if idx < 20:
            comp = COMP[idx // 5]
            sw, fac = JSEEDS[comp]
            return Bd(w, val, {comp: [(sw, fac(cell))]})
        return Bd(w, val)
    return Bd(w, val, {idx: up(val)} if ad else None)


def inputs_for(H, band, cell, ad):
    """the class bounds on a cell of `band` as jets (value, x-derivative)"""
    mS = NWF if cell.far else NW
    RV = Jet(seed_bd(W1, H[("h", "V", band)], vidx("V", band), ad, cell),
             seed_bd(W, H[("d", "V", band)], didx("V", band), ad, cell))
    RS = Jet(seed_bd(W, H[("h", "S", band)], vidx("S", band), ad, cell),
             seed_bd(mS, H[("d", "S", band)], didx("S", band), ad, cell))
    RB = Jet(seed_bd(W, H[("h", "B", band)], vidx("B", band), ad, cell),
             seed_bd(NW, H[("d", "B", band)], didx("B", band), ad, cell))
    RC = Jet(seed_bd(W1, H[("h", "C", band)], vidx("C", band), ad, cell),
             seed_bd(W, H[("d", "C", band)], didx("C", band), ad, cell))
    return RV, RS, RB, RC


# ---------------------------------------------------------------- one cell: Lemma R's forms (B8_constants.md 4)
def cell_eval(cell, kn, H, ad, st):
    c_ = cell
    one, om, x, x2, nm1, eps1 = kn["one"], kn["om"], kn["x"], kn["x2"], kn["nm1"], kn["eps1"]
    RV, RS, RB, RC = inputs_for(H, cell.band, cell, ad)
    RS1 = Jet(RS[0], RS[1])
    # operators built from c = c^K + R_c (jets of order 1)
    C = jadd(c_, kn["C"], RC)                                   # |C| <= |C^K| + |R_c|, |xC'| <= |xC^K'| + |xR_c'|
    Cm1 = jadd(c_, kn["Cm1"], RC)
    Cinv = jinv(c_, Cm1)                                        # F3: x(C^-1)' via the class bound of |x c'|
    c1 = jadd(c_, kn["cK"], RC)                                 # |c|_1
    dD = jadd(c_, jscale(jmul(c_, kn["C"], RC), 2), jmul(c_, RC, RC))   # delta D = -(C^K R_C + R_C C^K + R_C^2)
    S = jadd(c_, kn["S"], RS)
    V = jadd(c_, kn["V"], RV)
    Y = jadd(c_, kn["Y"], RV, RS)
    qRS = jmul(c_, kn["q"], RS)
    Rb = jadd(c_, RB, qRS)                                      # R_beta = R^_beta + q R_S
    beta = jadd(c_, kn["beta"], Rb)
    dBb = jadd(c_, jmul(c_, RC, kn["beta"]), jmul(c_, c1, qRS))   # delta Bfrak = [R_c beta^K + c q R_S .]
    CKinv = kn["CKinv"]
    B0mBK = jadd(c_, jmul(c_, dBb, Cinv), jprod(c_, kn["Bb"], Cinv, RC, CKinv),
                 jmul(c_, om, jadd(c_, jmul(c_, RC, Cinv), jprod(c_, kn["C"], Cinv, RC, CKinv))))
    B0 = jadd(c_, kn["BK"], B0mBK)
    A0mAK = jadd(c_, jmul(c_, nm1, dD), jmul(c_, RC, kn["Bb"]), jmul(c_, C, dBb), jmul(c_, x2, B0mBK))
    A0 = jadd(c_, kn["AK"], A0mAK)
    T1 = jmul(c_, A0mAK, S)
    T2 = jmul(c_, x2, jadd(c_, jmul(c_, B0mBK, V), jmul(c_, kn["BK"], RV)))
    A0SmB0V = jadd(c_, jmul(c_, A0, S), jprod(c_, x2, B0, V))
    Mh = jmul(c_, c1, RB)                                       # M_h = [c R^_beta .]
    MhTerm = jadd(c_, jprod(c_, C, Mh, S), jprod(c_, x2, Mh, Cinv, Y))   # C M_h S + x^2 M_h C^-1 Y
    out = {}
    if cell.kind == "L":
        # B_chain.md 13.3: R_c(1) = 0 (F1), P_c := R_C/(x - 1) with |P_c| <= sup_[1,x] |R_c'|,
        # |x P_c'| <= |x| sup |R_c''|/2, |R_c''| <= (|x(x R_c')'| + |x R_c'|)/|x|^2 (F2: class bound h_cc n w)
        spread = lambda w: up((iv.mpf(cell.Xhi) / cell.Xlo) ** abs(w[0] + 2 * w[2]) *
                              (iv.mpf(cell.thhi) / cell.thlo) ** abs(w[1]))
        dC = RC[1]                                               # |x R_c'| <= dC w
        cc = seed_bd(NW, H[("cc",)], CCIDX, ad)
        Pc0 = bscale(Bd((dC.w[0] - 1, dC.w[1] - 1, dC.w[2]), dC.v, dC.d), spread((dC.w[0] - 1, dC.w[1] - 1, dC.w[2])))
        if JSEEDS is not None:
            Pc0 = Bd(Pc0.w, Pc0.v, {PCIDX: [((0, 0, 0), ONEI)]})      # delta P_c <= p (pseudo-component)
        t = badd(c_, cc, dC)
        # |x| sup_s t(s)/|s|^2 = [t(s)/|s|] (|x|/|s|): the spread of the weight t/|x| and |x|/|s| <= (1+d/2)/(1-d/2)
        Pc1 = bscale(Bd((t.w[0] - 1, t.w[1] - 1, t.w[2]), t.v, t.d), up(iv.mpf("0.5") *
                     spread((t.w[0] - 1, t.w[1] - 1, t.w[2])) * (1 + HD) / (1 - HD)))
        Pc = Jet(Pc0, Pc1)
        QD = jadd(c_, jscale(jmul(c_, kn["C"], Pc), 2), jmul(c_, RC, Pc))     # delta D/(x - 1)
        Gm = kn["Gam"]
        # D~ = (x + 1)(I - Gamma) + QD: |D~^-1| <= 1/(|x+1|(1 - |Gamma|) - |QD|)
        Lt = iv.mpf(cell.xp1lo) * (1 - kn["n_Gam"])
        Dtinv0 = binvlb(Lt, bnum(QD[0], c_))
        # |x D~'| <= |x|(1 + |Gamma|) + |x + 1||x Gamma'| + |x QD'|
        xDtp = badd(c_, bmul(x[0], badd(c_, one[0], Gm[0])), bscale(Gm[1], cell.xp1hi), QD[1])
        Dtinv1 = bmul(bmul(Dtinv0, Dtinv0), bnum(xDtp, c_))
        Dtinv = Jet(Dtinv0, Dtinv1)
        DtK = Jet(bscale(badd(c_, one[0], Gm[0]), cell.xp1hi),
                  badd(c_, bmul(x[0], badd(c_, one[0], Gm[0])), bscale(Gm[1], cell.xp1hi)))
        T3 = jprod(c_, QD, Dtinv, A0SmB0V)
        G1 = jmul(c_, kn["AKinv"], jadd(c_, T1, T2, T3, kn["rho2"]))
        G2 = jprod(c_, kn["AKinv"], DtK, Dtinv, MhTerm)
        G = jadd(c_, G1, G2)
        out.update(G=G, G1=G1, G2=G2, spread=spread)
        qxgS = jmul(c_, kn["qLam"], G)                           # q x g_S = (ell/x^2)(I+e)^-1 eps1 A^K (G1 + G2)
    else:
        Gm = kn["Gam"]
        # |D^-1| <= X^-2/(inf(|x^2-1|/X^2)(1 - |Gamma|) - sup(|delta D|/X^2)),  |x D'| <= 2|x|^2 + 2|C||x C'|
        LD = iv.mpf(cell.d2lo) * (1 - kn["n_Gam"])
        s = bto(dD[0], c_, (2, 0, 0))
        Dinv0 = binvlb(LD, Bd((0, 0, 0), s.v, s.d), (-2, 0, 0))
        xDp = badd(c_, bscale(x2[0], 2), bscale(bmul(C[0], C[1]), 2))
        Dinv1 = bmul(bmul(Dinv0, Dinv0), xDp)
        Dinv = Jet(Dinv0, Dinv1)
        T3 = jprod(c_, dD, Dinv, A0SmB0V)
        G1 = jmul(c_, kn["AKinv"], jadd(c_, T1, T2, T3, kn["rho2"]))
        xW2 = jmul(c_, Dinv, MhTerm)                             # x W_2 = D^-1 (C M_h S + x^2 M_h C^-1 Y)
        out.update(G1=G1, xW2=xW2, Dinv=Dinv, parts=dict(T1=T1, T2=T2, T3=T3))
        # Lemma J's G = G1 + G2, G2 = (A^K)^-1 D^K D^-1 (...)
        G2 = jprod(c_, kn["AKinv"], jmul(c_, jmono(2, 0, 0, 0, 0, cell.d2hi), jadd(c_, one, Gm)), Dinv, MhTerm)
        out["G"] = jadd(c_, G1, G2)
        qxgS = jadd(c_, jmul(c_, kn["qLam"], G1), jmul(c_, kn["q"], xW2))
    # g^ (I3): -[R_beta(2 beta^K + R_beta)] + eps[R_beta H] + (1-eps)[b^K g3] - xq' R_S - q x g_S - rho3
    Hh = jscale(jadd(c_, jmul(c_, S, S), jprod(c_, x2, Cinv, Cinv, Y, Y)), iv.mpf("0.5"))
    dDl = jadd(c_, jprod(c_, x, Cinv, jadd(c_, RV, RS)), jprod(c_, Cinv, RC, kn["Dt"]))
    g3 = jadd(c_, jscale(jmul(c_, RS, RS), iv.mpf("0.5")),
              jmul(c_, kn["Dt"], jadd(c_, jprod(c_, x, Cinv, RV),
                                      jprod(c_, Cinv, RC, jadd(c_, jprod(c_, x, CKinv, RS), kn["Dt"])))),
              jscale(jmul(c_, dDl, dDl), iv.mpf("0.5")))
    q = kn["q"]
    xqp = Jet(q[1], q[2])
    eps = jconst(ONEI, (-2, 0, 1))
    ghat = jadd(c_, jmul(c_, Rb, jadd(c_, jscale(kn["beta"], 2), Rb)), jprod(c_, eps, Rb, Hh),
                jmul(c_, kn["bK"], g3), jmul(c_, xqp, RS), qxgS, kn["rho3"])
    Fc_nr = jadd(c_, jmul(c_, RC, beta), jmul(c_, kn["cK"], Rb))        # F_c without rho4
    Fc = jadd(c_, Fc_nr, kn["rho4"])
    out.update(ghat=ghat, Fc_nr=Fc_nr, Fc=Fc, RS=RS, Cinv=Cinv, Y=Y, Cj=C, S=S)
    # ratios
    out["G1w"] = bto(out["G1"][0], c_, W)
    if cell.kind != "L":
        out["Iw"] = bto(badd(c_, out["G1"][1], out["xW2"][0]), c_, W)        # (|x G1'| + |x W2|)/w
    out["gw"] = bto(bmul(eps1[0], ghat[0]), c_, W)                         # |g^|/((n-1) w)
    out["gpw"] = bto(bmul(eps1[0], ghat[1]), c_, W)                        # |x g^'|/((n-1) w)
    out["Icw"] = bto(Fc_nr[0], c_, W)                                     # (|R_c||beta| + |c^K||R_beta|)/w
    out["Fcw"] = bto(Fc[0], c_, W)                                        # |F_c|/w
    out["r4int"] = bto(kn_rho4int(c_), c_, W1) if cell.kind != "L" else None
    return out


KN = None


def kn_rho4int(cell):
    return KN.rho4int.bound(cell)


# ---------------------------------------------------------------- the kernel (Lemma K; B_chain.md 13.3)
def kernel_consts(st, cells, kres):
    """s_* = sin kappa - sup_ray |H'|, |H'| <= |eps1 eta|/(1 - |Gamma|)"""
    m = ZEROI
    for c in cells:
        if c.kind == "ray":
            kn = kres[id(c)]
            hp = up(kn["n_eps1eta"] / (1 - kn["n_Gam"]))
            if mpup(hp) > mpup(m):
                m = hp
    sstar = lo(SK - m)
    if mpup(sstar) <= 0:
        raise ValueError("s_* <= 0")
    return sstar, m


def cell_gains(st, cell, kn, sstar):
    """gains of Lemma K for a target cell (sector or far): arc g_arc, arc propagator max Kp, ray top piece g_top,
    and the factor for lower ray pieces (times |D_tb|)"""
    sa2 = SALPHA ** 2
    H = up(cell.x2d * kn["n_etat"] / (1 - kn["n_Gam"]))
    if cell.far:
        rlo = cell.rlo
        q = 1 + 2 * SK / rlo ** 2 + rlo ** -4
        N = lo((1 - st.nu0) / (iv.mpf(cell.rho_cap) * q))
        if cell.rhi is not None:
            f = cell.rhi ** 2 / (1 + 2 * cell.rhi ** 2 * SK + cell.rhi ** 4)
            N2 = lo((st.n0lo - 1) * f)
            if mpup(N2) > mpup(N):
                N = N2
        Dr_over_n1 = up(iv.mpf(cell.rho_cap) * iv.sqrt(q) / (1 - st.nu0))     # |D_r|/(n-1)
    else:
        def f(r):
            return r ** 2 / (1 + 2 * r ** 2 * SK + r ** 4)
        N = ZEROI if mpup(cell.rlo) == 0 else lo((st.n0lo - 1) * iv.mpf(min(mplo(f(cell.rlo)), mplo(f(cell.rhi)))))
        Dr = iv.sqrt(1 + 2 * cell.rhi ** 2 * SK + cell.rhi ** 4)
        Dr_over_n1 = up(Dr * st.eps1)
    if NOTESH and cell.far:
        H = up(kn["n_etat"] * (Al.XF + 1) / (Al.XF - 1) / (1 - kn["n_Gam"]))
    if MUT == "kernelH":
        H = ZEROI
    Ea = iv.exp(H * ALPHA)
    garc = up(ALPHA * Ea)
    Kp = up(Ea)
    if mpup(N) > 0:
        Kg = iv.exp(H ** 2 / (4 * N * sa2))
        g2 = up(Kg * iv.sqrt(iv.pi / (N * sa2)))
        if mpup(g2) < mpup(garc):
            garc = g2
        if mpup(Kg) < mpup(Kp):
            Kp = up(Kg)
    if MUT == "gamma3":
        garc = up(3 * garc)
    gtop = up(Kp * Dr_over_n1 / sstar)
    return dict(garc=garc, Kp=Kp, gtop=gtop, H=H, N=N)


# ---------------------------------------------------------------- the whole majorant
RADIAL_ORDER = ("N0", "Na", "Nb", "F")
TOP = {"N0": iv.mpf("0.05"), "Na": iv.mpf("1.5"), "Nb": iv.mpf(20)}


def Yf(r):
    return (1 + iv.mpf(r)) ** 10


def evaluate(st, cells, known, H, ad=False, detail=False):
    global KN
    KN = known
    kres = {id(c): known.at(c) for c in cells}
    res = {id(c): cell_eval(c, kres[id(c)], H, ad, st) for c in cells}
    sstar, hHray = kernel_consts(st, cells, kres)
    sec = [c for c in cells if c.kind in ("sector",) and not c.aux]
    auxc = [c for c in cells if c.kind == "sector" and c.aux]
    rays = [c for c in cells if c.kind == "ray"]
    Lc = [c for c in cells if c.kind == "L"][0]
    gains = {id(c): cell_gains(st, c, kres[id(c)], sstar) for c in sec + auxc}
    byband = {b: [c for c in sec if c.band == b] for b in RADIAL_ORDER}
    raysby = {b: [c for c in rays if c.band == b] for b in RADIAL_ORDER}
    near = [c for c in byband["Na"] + auxc if mplo(c.rlo) >= mplo(1 - HD) - 1e-30 and
            mpup(c.rhi) <= mpup(1 + HD) + 1e-30]
    if not near:
        near = [c for c in byband["Na"] if mplo(c.rlo) <= mpup(1 + HD) and mpup(c.rhi) >= mplo(1 - HD)]
    zero = Bd(W, ZEROI)

    def hb(comp, band):
        return Bd((0, 0, 0), H[("h", comp, band)], {vidx(comp, band): up(H[("h", comp, band)])} if ad else None)

    def rmax(items, key):
        xs = [res[id(c)][key] for c in items]
        return bmax(*xs) if xs else zero

    # ---- d0 (B_chain.md 13.2): |d| <= int_0^1 |F_c| ds/s, rho4 exactly, the rest per cell with int w ds/s
    # = nu^5 ((1 + t_b)^10 - (1 + t_a)^10)/10
    segs = []
    for c in byband["N0"]:
        segs.append((c, c.rlo, c.rhi))
    for c in byband["Na"]:
        a, b = c.rlo, c.rhi
        if mplo(a) >= mpup(1 - HD):
            continue
        if mpup(b) > mplo(1 - HD):
            b = 1 - HD
        segs.append((c, a, b))
    segs.append((Lc, 1 - HD, ONEI))
    d0 = Bd((0, 0, 0), known.d0_rho4)
    for c, a, b in segs:
        I = res[id(c)]["Icw"]
        fac = up((Yf(b) - Yf(a)) / 10)
        d0 = badd(Lc, d0, bscale(Bd((0, 0, 0), I.v, I.d), fac))
    if MUT == "nod0":
        d0 = Bd((0, 0, 0), ZEROI)
    twoF = up(2 * known.absA)                                   # |[d 2F]|_1 <= |d|_1 |2F|_1, |F|_1 = |A|

    # ---- radial integrals: sum over the cells below the target (all bands on the way) and the top piece
    def radial_sum(target_band, rtop_lo, key=None, comp=None):
        """(C_below as Bd weight 1: sum of I (Y(rhi) - Y(rlo))/10 over cells entirely below rtop_lo on the radial
        path, the top band, and Y(start of the top piece))"""
        Cb = Bd((0, 0, 0), ZEROI)
        path = RADIAL_ORDER[:RADIAL_ORDER.index(target_band if target_band != "L" else "Na") + 1]
        for b in path:
            for c in byband[b]:
                if mpup(c.rhi if c.rhi is not None else iv.mpf(10) ** 40) <= mplo(rtop_lo) + 0:
                    I = hb(comp, b) if key is None else Bd((0, 0, 0), res[id(c)][key].v, res[id(c)][key].d)
                    Cb = badd(Lc, Cb, bscale(I, up((Yf(c.rhi) - Yf(c.rlo)) / 10)))
        return Cb

    out = {}
    # ---- per target cell, bands N0, Na, Nb, F
    for b in RADIAL_ORDER:
        PV, PS, PB, PC, QV, QS, QB, QC = [], [], [], [], [], [], [], []
        for c in byband[b]:
            r = res[id(c)]
            g = gains[id(c)]
            # S: |T_S| <= |G1| + g_arc I(cell) + g_top I(ray of band, up to r) + sum lower pieces
            ps = badd(c, r["G1w"], bscale(r["Iw"], g["garc"]))
            top_rays = [rc for rc in raysby[b] if mplo(rc.rlo) <= mpup(c.rhi if c.rhi is not None else rc.rlo)]
            if top_rays:
                ps = badd(c, ps, bscale(rmax(top_rays, "Iw"), g["gtop"]))
            for bl in RADIAL_ORDER[:RADIAL_ORDER.index(b)]:
                Dtb = iv.sqrt(1 + 2 * TOP[bl] ** 2 * SK + TOP[bl] ** 4)
                ps = badd(c, ps, bscale(rmax(raysby[bl], "Iw"), up(g["Kp"] * Dtb * st.eps1 / sstar)))
            PS.append(ps)
            # beta: |T_beta| <= |g^|/(n-1) + (1/(n0-1)) sup_radial |x g^'|/(n-1) + phi_delta sup_arc |x g^'|/(n-1)
            radial_cells = [cc for bl in RADIAL_ORDER[:RADIAL_ORDER.index(b)] for cc in byband[bl]] + \
                [cc for cc in byband[b] if mplo(cc.rlo) <= mpup(c.rhi if c.rhi is not None else cc.rlo)]
            pb = badd(c, r["gw"], bscale(rmax(radial_cells, "gpw"), st.eps1), bscale(r["gpw"], up(PHID)))
            PB.append(pb)
            # V and c: radial pieces with exact int w dt/t, the arc part phi_delta sup theta, the offset d
            Cv = radial_sum(b, c.rlo, comp="S")
            Cc = radial_sum(b, c.rlo, key="Icw")
            Ya = Yf(c.rlo)
            vals_v, vals_c = [], []
            for Yr in ([Yf(c.rlo), Yf(c.rhi)] if c.rhi is not None else [Yf(c.rlo), None]):
                for C_, top, lst in ((Cv, hb("S", b), vals_v), (Cc, Bd((0, 0, 0), r["Icw"].v, r["Icw"].d), lst_dummy)):
                    pass
            # explicit: f(Y) = top/10 + (C_below - top Ya)/(10 Y), monotone in Y: evaluate at both ends
            def endpoint_vals(Cb, top):
                outs = []
                Ylist = [Yf(c.rlo)] + ([Yf(c.rhi)] if c.rhi is not None else [])
                for Yr in Ylist:
                    v = badd(c, bscale(Cb, up(1 / (10 * Yr))), bscale(top, up((Yr - Ya) / (10 * Yr))))
                    outs.append(v)
                if c.rhi is None:
                    outs.append(bscale(top, up(iv.mpf(1) / 10)))
                return bmax(*outs)
            rv = endpoint_vals(Cv, hb("S", b))
            pv = badd(c, bscale(Bd((0, 0, 0), d0.v, d0.d), up(twoF / (1 if CRUDE_D else Yf(c.rlo)))),
                      bscale(badd(c, rv, bscale(hb("S", b), up(PHID * c.thhi))), up(known.absW)))
            PV.append(Bd(W1, pv.v, pv.d))
            rc_ = endpoint_vals(Cc, Bd((0, 0, 0), r["Icw"].v, r["Icw"].d))
            pc = badd(c, bscale(Bd((0, 0, 0), d0.v, d0.d), up(1 / (1 if CRUDE_D else Yf(c.rlo)))), rc_,
                      bscale(Bd((0, 0, 0), r["Icw"].v, r["Icw"].d), up(PHID * c.thhi)),
                      Bd((0, 0, 0), r["r4int"].v, r["r4int"].d))
            PC.append(Bd(W1, pc.v, pc.d))
            # derivative bounds of T(U) (pass 1 with T's own value)
            QV.append(Bd(W, bscale(hb("S", b), up(known.absW)).v, bscale(hb("S", b), up(known.absW)).d))
            mS = NWF if c.far else NW
            kn = kres[id(c)]
            qs = badd(c, bmul(kn["Lam"], badd(c, Bd(W, ps.v, ps.d), r["G1"][0])), r["xW2"][0])
            QS.append(bto(qs, c, mS))
            qb = badd(c, r["ghat"][0], bmul(kn["nm1"][0], Bd(W, pb.v, pb.d)))
            QB.append(bto(qb, c, NW))
            QC.append(bto(r["Fc"][0], c, W))
        out[("h", "S", b)] = bmax(*[Bd((0, 0, 0), x.v, x.d) for x in PS])
        out[("h", "B", b)] = bmax(*[Bd((0, 0, 0), x.v, x.d) for x in PB])
        out[("h", "V", b)] = bmax(*[Bd((0, 0, 0), x.v, x.d) for x in PV])
        out[("h", "C", b)] = bmax(*[Bd((0, 0, 0), x.v, x.d) for x in PC])
        out[("d", "V", b)] = bmax(*[Bd((0, 0, 0), x.v, x.d) for x in QV])
        out[("d", "S", b)] = bmax(*[Bd((0, 0, 0), x.v, x.d) for x in QS])
        out[("d", "B", b)] = bmax(*[Bd((0, 0, 0), x.v, x.d) for x in QB])
        out[("d", "C", b)] = bmax(*[Bd((0, 0, 0), x.v, x.d) for x in QC])
    # ---- L (B_chain.md 13.3)
    r = res[id(Lc)]
    kn = kres[id(Lc)]
    spread = r["spread"]
    c1 = 1 / (2 - HD) + 2 / (1 - HD)
    etaL = up((1 + HD) * kn["n_etat"] / ((1 - kn["n_Gam"]) * (2 - HD)))
    NL = lo((st.n0lo - 1) * (1 - c1 * HD) / 2 - etaL)
    if MUT == "NL":
        NL = iv.mpf(2)
    if mpup(NL) <= 1:
        raise ValueError("N_L <= 1")
    # u(x') = G2(x') + int_{P_x'} E (G1' + W2): x' on Ac (radius in [1 - delta/2, 1 + delta/2]): Na cells there
    uint = []
    for c in near:
        g = gains[id(c)]
        t = badd(c, bscale(res[id(c)]["Iw"], g["garc"]),
                 bscale(rmax([rc for rc in raysby["Na"] if mplo(rc.rlo) <= mpup(c.rhi)], "Iw"), g["gtop"]),
                 bscale(rmax(raysby["N0"], "Iw"), up(g["Kp"] * iv.sqrt(1 + 2 * TOP["N0"] ** 2 * SK + TOP["N0"] ** 4)
                                                    * st.eps1 / sstar)))
        uint.append(Bd((0, 0, 0), t.v, t.d))
    uint = bmax(*uint)
    G2w = bto(r["G2"][0], Lc, W)
    uxp = badd(Lc, Bd((0, 0, 0), G2w.v, G2w.d), uint)                    # |u(x')|/w(x')
    Gw = bto(r["G"][0], Lc, W)
    xGp = bto(bscale(Bd((r["G"][1].w[0] - 1, r["G"][1].w[1] - 1, r["G"][1].w[2]), r["G"][1].v, r["G"][1].d),
                     spread((r["G"][1].w[0] - 1, r["G"][1].w[1] - 1, r["G"][1].w[2]))), Lc, W)   # sup |G'|/w
    sw = spread(W)
    psL = badd(Lc, Bd((0, 0, 0), Gw.v, Gw.d), bscale(uxp, sw),
               bscale(Bd((0, 0, 0), xGp.v, xGp.d), up(HD / (NL - 1) * sw)))
    out[("h", "S", "L")] = psL
    # |x T_S'| <= |(D~^K)^-1 A^K| (|u(x')|/(delta/2) + sup|G'|/(N_L - 1))
    DKA = bscale(bmul(kn["nm1"][0], kn["AKn"][0]), up(1 / (iv.mpf(Lc.xp1lo) * (1 - kn["n_Gam"]))))
    inner = badd(Lc, bscale(Bd(W, uxp.v, uxp.d), up(sw / HD)), bscale(Bd(W, xGp.v, xGp.d), up(sw / (NL - 1))))
    qsL = bto(bmul(DKA, inner), Lc, NW)
    out[("d", "S", "L")] = Bd((0, 0, 0), qsL.v, qsL.d)
    # beta on L: the integrand of T_beta is analytic on the closed sector (which contains L), so for x in L the
    # straight radial segment [0, x] can replace Q_x: it runs in N0, in Na for |s| < 1 - delta/2 and in Na or L
    # beyond; |(s/x)^(n-1)| <= 1 and int_0^|x| (t/|x|)^(n-1) dt/t = 1/(n-1), so no arc term
    radial_cells = byband["N0"] + byband["Na"] + [Lc]
    pbL = badd(Lc, r["gw"], bscale(rmax(radial_cells, "gpw"), st.eps1))
    if MUT == "Larc":
        # Q_x with its arc instead of the straight segment (both valid; shows what the arc term costs on L)
        pbL = badd(Lc, pbL, bscale(bmax(r["gpw"], rmax(near, "gpw")), up(PHID)))
    out[("h", "B", "L")] = Bd((0, 0, 0), pbL.v, pbL.d)
    qbL = bto(badd(Lc, r["ghat"][0], bmul(kn["nm1"][0], Bd(W, pbL.v, pbL.d))), Lc, NW)
    out[("d", "B", "L")] = Bd((0, 0, 0), qbL.v, qbL.d)
    # c on L: |T_c(x)| <= |x - 1| sup_L |F_c(s)|/|s|
    Fcx = bto(Bd((r["Fc"][0].w[0] - 1, r["Fc"][0].w[1] - 1, r["Fc"][0].w[2]), r["Fc"][0].v, r["Fc"][0].d), Lc, W1)
    pcL = bscale(Bd((0, 0, 0), Fcx.v, Fcx.d), up(HD * spread(W1)))
    out[("h", "C", "L")] = pcL
    out[("d", "C", "L")] = Bd((0, 0, 0), bto(r["Fc"][0], Lc, W).v, bto(r["Fc"][0], Lc, W).d)
    out[("cc",)] = Bd((0, 0, 0), bto(r["Fc"][1], Lc, NW).v, bto(r["Fc"][1], Lc, NW).d)
    # V on L: d term and the straight radial segment [0, x] (as for beta): N0, Na below 1 - delta/2, then Na or L
    CvL = radial_sum("L", 1 - HD, comp="S")
    hSm = bmax(hb("S", "Na"), hb("S", "L"))
    vals = []
    Ya = Yf(1 - HD)
    for Yr in (Yf(1 - HD), Yf(1 + HD)):
        vals.append(badd(Lc, bscale(CvL, up(1 / (10 * Yr))), bscale(hSm, up((Yr - Ya) / (10 * Yr)))))
    pvL = badd(Lc, bscale(Bd((0, 0, 0), d0.v, d0.d), up(twoF / (1 if CRUDE_D else Yf(1 - HD)))),
               bscale(bmax(*vals), up(known.absW)))
    out[("h", "V", "L")] = pvL
    out[("d", "V", "L")] = bscale(hb("S", "L"), up(known.absW))
    out["d0"] = d0
    if detail:
        out["_detail"] = dict(sstar=sstar, hHray=hHray, NL=NL, etaL=etaL, gains=gains, res=res, kres=kres,
                              uint=uint, Lc=Lc, near=near)
    return out


lst_dummy = []


# ---------------------------------------------------------------- closure
KEYS = [("h", c, b) for c in COMP for b in BANDS] + [("d", c, b) for c in COMP for b in BANDS] + [("cc",)]


def key_index(k):
    if k == ("cc",):
        return CCIDX
    return (vidx(k[1], k[2]) if k[0] == "h" else didx(k[1], k[2]))


def zero_h():
    return {k: ZEROI for k in KEYS}


def iterate(st, cells, known, H0=None, nit=400, tol=1e-6, log=print):
    H = H0 or zero_h()
    for it in range(nit):
        out = evaluate(st, cells, known, H)
        Hn = {k: up(out[k].v) for k in KEYS}
        ch = max((abs(mpup(Hn[k]) - mpup(H[k])) / max(mpup(Hn[k]), mp.mpf(10) ** -300)) for k in KEYS)
        H = Hn
        if it % 10 == 0:
            log(f"  iter {it}: max relative change {mp.nstr(ch, 3)}, d0 = {mp.nstr(mpup(out['d0'].v), 6)}")
        if ch < tol:
            log(f"  converged after {it} iterations")
            return H, out
    raise ValueError("iteration did not converge")


def descend(st, cells, known, Hstar, log=print, fac="1.01", start="1.1", nmax=200):
    """from start*h*, h <- fac*P(h) until fac*P(h) <= h for every component (then P(h) < h strictly)"""
    fac, start = iv.mpf(fac), iv.mpf(start)
    H = {k: up(start * v) for k, v in Hstar.items()}
    for it in range(nmax):
        out = evaluate(st, cells, known, H)
        ok = all(mpup(fac * out[k].v) <= mplo(H[k]) for k in KEYS)
        if ok:
            log(f"  descent: closed after {it} steps")
            return H, out
        H = {k: up(fac * out[k].v) for k in KEYS}
    raise ValueError("descent did not close")


# ---------------------------------------------------------------- Lipschitz matrix and Perron root
def lipschitz(st, cells, known, H):
    out = evaluate(st, cells, known, H, ad=True, detail=True)
    M = [[mp.mpf(0)] * NIDX for _ in range(NIDX)]
    for k in KEYS:
        i = key_index(k)
        hi_ = mplo(H[k])
        if hi_ <= 0:
            continue
        for j, dv in out[k].d.items():
            M[i][j] = mpup(iv.mpf(mpup(dv)) / hi_)
    return M, out


def perron(M, comps=None, lam=None):
    """spectral radius estimate (floats) and a certified theta: v = (I - M/lam)^-1 1 rounded to rationals, then
    M v <= theta v checked in exact rational arithmetic with M's upper bounds"""
    from fractions import Fraction as Fr
    import numpy as np
    idx = list(range(len(M))) if comps is None else comps
    A = np.array([[float(M[i][j]) for j in idx] for i in idx])
    ev = max(abs(np.linalg.eigvals(A)))
    lam = lam or min(ev * 1.03 + 1e-3, 0.999)
    v = np.linalg.solve(np.eye(len(idx)) - A / lam, np.ones(len(idx)))
    if (v <= 0).any():
        return ev, None, None
    vq = [Fr(float(x)).limit_denominator(10 ** 12) for x in v]
    Mq = [[mpf_to_frac_up(M[i][j]) for j in idx] for i in idx]
    theta = max(sum(Mq[a][b] * vq[b] for b in range(len(idx))) / vq[a] for a in range(len(idx)))
    return ev, theta, vq


def mpf_to_frac_up(x):
    from fractions import Fraction as Fr
    m, e = mp.mpf(x).man_exp
    return Fr(m) * (Fr(2) ** e) if e >= 0 else Fr(m, 2 ** (-e))


def main():
    t0 = time.time()
    mode = OPTS.get("mode", "cells")
    ln_n0 = OPTS.get("ln_n0", "73.89")
    logname = OPTS.get("log")
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    st = Al.Setting(ln_n0)
    T = Ser.tables()
    known = Known(st, T)
    cells = make_cells(st, mode)
    log(f"P2b Lemma LC second implementation: mode {mode}, ln n0 = {ln_n0}, {len(cells)} cells, mutate = {MUT}")
    log(f"  n0 = {mp.nstr(mpup(st.n0), 8)}, rho_max = {mp.nstr(mpup(st.rho_max), 8)}, |A| <= {mp.nstr(mpup(known.absA), 12)},"
        f" |omega| <= {mp.nstr(mpup(known.absW), 12)}")
    log(f"  exact rho4 part of d0: {mp.nstr(mpup(known.d0_rho4), 8)}")
    if "hfrom" in OPTS:
        with open(OPTS["hfrom"]) as fh:
            Hj = json.load(fh)["h"]
        H = {k: iv.mpf(Hj[idxname(key_index(k))]) for k in KEYS}
        out = evaluate(st, cells, known, H)
    else:
        Hs, _ = iterate(st, cells, known, log=log)
        H, out = descend(st, cells, known, Hs, log=log)
    res = report(st, cells, known, H, out, log)
    M, outad = lipschitz(st, cells, known, H)
    ev40, th40, _ = perron([[M[i][j] for j in range(40)] for i in range(40)])
    ev41, th41, _ = perron(M)
    det = outad["_detail"]
    log(f"  kernel: s_* = {mp.nstr(mpup(det['sstar']), 6)} (sup_ray |H'| = {mp.nstr(mpup(det['hHray']), 6)}),"
        f" N_L = {mp.nstr(mpup(det['NL']), 6)}, eta_L = {mp.nstr(mpup(det['etaL']), 6)}")
    gF = [g for c, g in [(c, det["gains"][id(c)]) for c in cells if c.kind == "sector" and c.band == "F"]]
    log("  far-band gains per cell (g_arc, g_top, H, N): " + "; ".join(
        f"({mp.nstr(mpup(g['garc']), 4)}, {mp.nstr(mpup(g['gtop']), 4)}, {mp.nstr(mpup(g['H']), 4)}, "
        f"{mp.nstr(mpup(g['N']), 4)})" for g in gF))
    log(f"  Lipschitz matrix 40 x 40: spectral radius {ev40:.5f}, certified theta {float(th40) if th40 else None:.5f}")
    log(f"  with the F2 component (41 x 41): spectral radius {ev41:.5f}, certified theta "
        f"{float(th41) if th41 else None:.5f}")
    res.update(rho40=ev40, theta40=float(th40) if th40 else None, rho41=ev41, theta41=float(th41) if th41 else None,
               sstar=float(mpup(det["sstar"])), NL=float(mpup(det["NL"])), etaL=float(mpup(det["etaL"])),
               M=[[float(M[i][j]) for j in range(NIDX)] for i in range(NIDX)], index=[idxname(i) for i in range(NIDX)],
               mode=mode, ln_n0=ln_n0, mutate=MUT, seconds=time.time() - t0)
    tag = OPTS.get("tag", f"{mode}" + ("_crude_d" if CRUDE_D else "") + ("_notesH" if NOTESH else "") +
                   (f"_mut_{MUT}" if MUT else "") +
                   ("_hcert" if "hfrom" in OPTS else ""))
    with open(os.path.join(HERE, f"p2b_lc_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    log(f"  wrote p2b_lc_{tag}.json ({time.time() - t0:.0f} s)")
    if logname:
        with open(os.path.join(HERE, logname), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    ok = res["closes"] and th40 is not None and th40 < 1
    return 0 if ok else 1


def report(st, cells, known, H, out, log):
    log("  constants h (rows V, S, beta, c; derivative constants D; P/h ratios):")
    ratios = {}
    for comp in COMP:
        row = []
        for b in BANDS:
            k = ("h", comp, b)
            rat = mpup(out[k].v) / mplo(H[k]) if mplo(H[k]) > 0 else mp.mpf(0)
            ratios[idxname(key_index(k))] = float(rat)
            row.append(f"{b} {mp.nstr(mpup(H[k]), 5)} ({mp.nstr(rat, 4)})")
        log(f"    h_{comp}: " + ", ".join(row))
    for comp in COMP:
        row = []
        for b in BANDS:
            k = ("d", comp, b)
            rat = mpup(out[k].v) / mplo(H[k]) if mplo(H[k]) > 0 else mp.mpf(0)
            ratios[idxname(key_index(k))] = float(rat)
            row.append(f"{b} {mp.nstr(mpup(H[k]), 5)} ({mp.nstr(rat, 4)})")
        log(f"    D_{comp}: " + ", ".join(row))
    k = ("cc",)
    rat = mpup(out[k].v) / mplo(H[k]) if mplo(H[k]) > 0 else mp.mpf(0)
    ratios["cc,L"] = float(rat)
    log(f"    h_cc (L): {mp.nstr(mpup(H[k]), 5)} ({mp.nstr(rat, 4)})")
    closes = all(v < 1 for v in ratios.values())
    log(f"  d0 = {mp.nstr(mpup(out['d0'].v), 6)}; h_Sp = D_S on L = {mp.nstr(mpup(H[('d', 'S', 'L')]), 6)};"
        f" closure: {'yes' if closes else 'NO'}, max ratio {max(ratios.values()):.4f}")
    return dict(h={idxname(key_index(k)): mp.nstr(mpup(H[k]), 17) for k in KEYS}, ratios=ratios, closes=closes,
                d0=float(mpup(out["d0"].v)))


if __name__ == "__main__":
    sys.exit(main())
