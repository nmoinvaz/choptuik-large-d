#!/usr/bin/env python3
"""P2 review: an independent check of Lemma B8's bootstrap (B8_constants.md sections 5-6, 8.4) for the normalization
offset |c(0) - c^K(0)|_1 <= d0 n^-5, d0 = 1e7, at ln n0 = 73.89: the constants h that (H8), B5(1) and B5(2) use for
the light-cone-gauge solution (B_chain.md section 4). Same atoms, bound algebra and known quantities as p2_lc.py
(nothing imported from solver/proof); B8's regions N0 (|x| <= r1), N1 (r1 <= |x| <= x_F, off the light-cone disc),
R (the ray, |x| <= x_F) and F, one constant per component.

Lemma B8.a (improvement): P_V = |omega| (1/(2K+2) + phi_d) h_S + 4 max|F| d0,
P_c = (1/(2K+2) + phi_d) sup |F_c|/w + d0, P_S = max_r [sup_r |G1|/w + gamma_r sup_{P(r)} (|x G1'| + |x||W2|)/w],
P_beta = max_r [sup_r |g^|/((n-1) w) + (1/(n0 - 1) + phi_d) sup_{Q(r)} |x g^'|/((n-1) w)], with the derivative bounds
of pass 1. Checked: P(h) < h at the h of b8m_scan_K4_d0_1e7.log ("hyp"), and the agreement with that log's P(h).

    python3 p2_b8.py [--d0=1e7] [--mutate=d0]     mutation d0: the offset terms dropped; P_V, P_c must move (caught by
                                                  the agreement check)
"""
import json
import os
import sys

from mpmath import iv, mp
from mpmath.libmp import fone, fzero

import p2_alg as G
from p2_alg import R, Lo, Bd, Q, Reg, bsup, inv_lo, lo_mono, qinv, up_iv, lo_iv, add_u, mul_u, fl, gt
from p2_common import lo as LO, hi as HI
import p2_atoms as AT
import p2_lc as LC

OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate", "")
HERE = os.path.dirname(os.path.abspath(__file__))
K, K2 = 4, 10
COMPS = ("V", "S", "b", "c")
XREG = ("N0", "N1", "F")
PATH_P = {"N0": ("N0",), "N1": ("R", "N1"), "F": ("R", "F")}
PATH_Q = {"N0": ("N0",), "N1": ("N0", "N1"), "F": ("N0", "N1", "F")}


class GeoB8:
    def __init__(self, L0="73.89"):
        g = LC.Geo(L0)                      # constants (kappa, delta, n0, rho_max, phi_d, ...) of the same setting
        self.__dict__.update({k: v for k, v in g.__dict__.items() if k not in ("boxes", "ell")})
        d2 = self.d2
        th = lambda r: r / (1 + r)
        D = lambda x: lo_iv(x)
        U = lambda x: up_iv(x)
        n = self.n0
        r1, xF = self.r1, self.xF
        box = lambda Xl, Xh, tl, th_, rho, far=False: dict(Xlo=D(Xl), Xhi=(None if Xh is None else U(Xh)), thlo=D(tl),
                                                            thhi=U(th_), rho=U(rho), far=far)
        self.boxes = {"N0": box(1, 1 + r1, 0, th(r1), (1 + r1) ** 2 / n),
                      "N1": box(1 + r1, 1 + xF, th(r1), th(xF), (1 + xF) ** 2 / n),
                      "R": box(1, 1 + xF, 0, th(xF), (1 + xF) ** 2 / n),
                      "F": box(1 + xF, None, th(xF), 1, self.rho_max, far=True)}
        lN = max(U((2 - d2) / d2), U((2 + d2) ** 2 / (d2 * iv.sqrt(2 * (1 - d2) ** 2 + 2 - d2 * d2))), U((2 + d2) / d2),
                 key=fl)
        self.ell = {"N0": U((1 + r1) / (1 - r1)), "N1": lN, "R": U(2 * iv.sqrt(2)), "F": U((1 + xF) / (xF - 1))}
        Reg.set(self.boxes, self.n0)


def kernel(geo, k, D, nsub=1500):
    n0 = geo.n0
    eta = (k["x2"] * k["etaT"]).v
    gmax = {r: bsup(k["Gam"].v, r) for r in Reg.regs}
    hpv = [iv.mpf(mp.make_mpf(bsup(eta * Bd.mono(R(geo.nf), e=-1), r).v)) / (1 - iv.mpf(mp.make_mpf(gmax[r].v)))
           for r in ("R", "F")]
    sstar = geo.sk - iv.mpf(max(HI(v) for v in hpv))
    sstar = iv.mpf(LO(sstar))
    gF_ = iv.mpf(mp.make_mpf(gmax["F"].v))
    HF = iv.mpf(mp.make_mpf(bsup(eta * LC.mono(1, a=-2), "F").v)) * iv.mpf(mp.make_mpf(geo.ell["F"])) / (1 - gF_)
    qF = 1 + 2 * geo.sk / geo.xF ** 2 + 1 / geo.xF ** 4
    NF = iv.mpf(LO((1 - 1 / n0) / (geo.rho_max * qF)))
    KF = iv.exp(HF * HF / (4 * NF * geo.salpha ** 2))
    gF = KF * iv.sqrt(qF) * geo.rho_max * (n0 / (n0 - 1)) / sstar + KF * iv.sqrt(iv.pi / (NF * geo.salpha ** 2))
    cN, cd, cg = D["Neta"]["coef"], D["det-1"]["coef"], D["Gamma"]["coef"]
    d2, al, sa, sk = geo.d2, geo.alpha, geo.salpha, geo.sk
    brk = [iv.mpf(0), geo.r1, iv.mpf(LO(1 - d2)), iv.mpf(HI(1 + d2)), geo.xF]
    gN = {"N0": iv.mpf(0), "N1": iv.mpf(0)}
    for a, b in zip(brk[:-1], brk[1:]):
        m = nsub // 3
        for i in range(m):
            ra, rb = a + (b - a) * i / m, a + (b - a) * (i + 1) / m
            rr = iv.mpf([LO(ra), HI(rb)])
            ev = lambda cf: sum((iv.mpf(c) * n0 ** (-j) * rr ** mm for (j, mm), c in cf.items()), iv.mpf(0))
            Ueta = rr * rr * ev(cN) / (1 - ev(cd))
            Ug = ev(cg)
            if HI(b) <= LO(1 - d2):
                mr = 1 - rr * rr
            elif LO(a) >= HI(1 + d2):
                mr = rr * rr - 1
            else:
                mr = d2 * iv.sqrt(2 * rr * rr + 2 - d2 * d2)
            mr = iv.mpf(LO(mr))
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
            reg = "N0" if HI(b) <= HI(geo.r1) else "N1"
            gN[reg] = g if HI(g) > HI(gN[reg]) else gN[reg]
    U = lambda v: R(up_iv(v))
    return dict(sstar=sstar, HF=HF, NF=NF, KF=KF, gF=U(gF), gN0=U(gN["N0"]), gN1=U(gN["N1"]))


def improve(geo, k, kc, h, d0):
    nf = geo.nf
    iwS = inv_lo(lo_mono(fone, a=K2, b=1, e=-(K + 1)))
    x, x2, eps, eps1, nm1, Om, half, two = (k[s] for s in ("x", "x2", "eps", "eps1", "nm1", "Om", "half", "two"))
    cK, iCK, SK, VK, bK, AK, iAK, DK, iDK, BK, bbK, YK, q, xq = (k[s] for s in (
        "cK", "iCK", "S", "V", "beta", "AK", "iAK", "DK", "iDK", "BK", "bbK", "Y", "q", "xq"))
    MK, Dlt, rho2, rho3, rho4 = k["cb"], k["Dlt"], k["rho2"], k["rho3"], k["rho4"]

    def build(RS, RbH, Rc, RV):
        Rb = RbH + q * RS
        Rb0 = q * RS
        c = cK + Rc
        c.lo = {r: (cK.lo[r][0].minus(bsup(Rc.v, r)), 0, 0, 0) for r in Reg.regs}
        beta = bK + Rb
        c.d = (cK.d + Rc.d) if Rc.d is not None else None
        iC = qinv(Q(c.v, c.d, None, c.lo))
        S, V, Y = SK + RS, VK + RV, YK + RV + RS
        dC = Rc
        dD = (cK * two + Rc) * Rc
        Dq = Q(DK.v + dD.v)
        lob = {}
        for r in Reg.regs:
            (cl, a, b, e) = DK.lo[r]
            rel = bsup(dD.v * inv_lo(DK.lo), r)
            lob[r] = (cl.times_one_minus(rel), a, b, e)
            if not lob[r][0].positive():
                raise ValueError(f"D lower bound on {r}")
        Dq.lo = lob
        Dq.d = (x2 * two).v + c.v * c.d * R.c(2) if c.d is not None else None
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
        return dict(Rb=Rb, c=c, iC=iC, beta=beta, S=S, Y=Y, dC=dC, G1=G1, W2=W2, xgS=xgS)

    def gbeta(Rr, RS, RV):
        iC, S, Y, Rb, dC = Rr["iC"], Rr["S"], Rr["Y"], Rr["Rb"], Rr["dC"]
        Dl = x * iC * Y
        H = (S * S + Dl * Dl) * half
        dDl = x * iC * (RV + RS) + iC * dC * Dlt
        g3 = bbK * (RS * RS * half + (x * iC * RV + iC * dC * (x * iCK * RS + Dlt)) * Dlt + dDl * dDl * half)
        return Rb * (bK * two + Rb) + eps * Rb * H + g3 + xq * RS + q * Rr["xgS"] + rho3

    w1 = Bd.mono(R(fone), a=K2, e=-(K + 1))
    wS = Bd.mono(R(fone), a=K2, b=1, e=-(K + 1))
    absw = Om.v
    RS = Q(wS * h["S"])
    RbH = Q(wS * h["b"])
    Rc = Q(w1 * h["c"])
    RV = Q(w1 * h["V"], wS * absw * h["S"])
    R1 = build(RS, RbH, Rc, RV)
    RS_d = (AK * iDK).v * (RS.v + R1["G1"].v) + x.v * R1["W2"].v
    gb1 = gbeta(R1, RS, RV)
    RbH_d = gb1.v + nm1.v * RbH.v
    Rc_d = Rc.v * R1["beta"].v + cK.v * R1["Rb"].v + rho4.v
    RS = Q(RS.v, RS_d)
    RbH = Q(RbH.v, RbH_d)
    Rc = Q(Rc.v, Rc_d)
    R2 = build(RS, RbH, Rc, RV)
    G1, W2 = R2["G1"], R2["W2"]
    gb = gbeta(R2, RS, RV)
    loc = G1.v * iwS
    integ = (G1.d + x.v * W2.v) * iwS
    gam = {"N0": kc["gN0"], "N1": kc["gN1"], "F": kc["gF"]}
    PS = LC.rmaxR([bsup(loc, r) + gam[r] * LC.rmaxR([bsup(integ, s) for s in PATH_P[r]]) for r in XREG])
    gl = gb.v * Bd.mono(R(nf), e=-1) * iwS
    gi = gb.d * Bd.mono(R(nf), e=-1) * iwS
    fac = add_u(G.div_u(nf, lo_iv(geo.n0)), up_iv(geo.phid))
    Pb = LC.rmaxR([bsup(gl, r) + LC.rmaxR([bsup(gi, s) for s in PATH_Q[r]]).scale(fac) for r in XREG])
    slow = add_u(G.div_u(fone, mp.mpf(K2)._mpf_), up_iv(geo.phid))
    PV = (bsup(absw, "F") * h["S"]).scale(slow)
    Fc = (Rc.v * R2["beta"].v + cK.v * R2["Rb"].v + rho4.v) * iwS
    Pc = LC.rmaxR([bsup(Fc, r) for r in XREG]).scale(slow)
    if MUT != "d0":
        PV = PV + R(mul_u(mul_u(up_iv(4), mp.mpf(LC.CONSTS["absf"])._mpf_), up_iv(d0)))
        Pc = Pc + R(up_iv(d0))
    return dict(V=PV, S=PS, b=Pb, c=Pc)


def main():
    geo = GeoB8(OPTS.get("L0", "73.89"))
    D, consts, rems = AT.atoms(K, "2e-6", "2e-6")
    LC.CONSTS.clear()
    LC.CONSTS.update(consts)
    k = LC.known(geo, D, consts)
    kc = kernel(geo, k, D)
    print("kernel: " + ", ".join(f"{a} {mp.nstr(mp.make_mpf(v.v) if isinstance(v, R) else HI(v), 6)}" for a, v in kc.items()))
    d0s = OPTS.get("d0", "1e7")
    rec = None
    for line in open(os.path.join(HERE, "..", f"b8m_scan_K4_d0_{d0s}.log")):
        r = json.loads(line)
        if r.get("closes") and abs(r["L0"] - 73.89) < 1e-9:
            rec = r
    h = {c: R(up_iv(iv.mpf(repr(rec["hyp"][c])) * (1 + iv.mpf("1e-12")))) for c in COMPS}
    P = improve(geo, k, kc, h, iv.mpf(d0s))
    ok = all(gt(h[c].v, P[c].v) for c in COMPS)
    dev = {c: (P[c].f() - rec["P_hyp"][c]) / rec["P_hyp"][c] for c in COMPS}
    for c in COMPS:
        print(f"   {c}: h = {h[c].f():.6e}, P(h) = {P[c].f():.6e} (ratio {P[c].f() / h[c].f():.4f}); "
              f"b8m_scan's P(h) {rec['P_hyp'][c]:.6e} (relative difference {dev[c]:.1e})")
    okC = all(abs(v) <= 1e-2 for v in dev.values())
    print(f"=> B8 at d0 = {d0s}: closure P(h) < h: {ok}; agreement with b8m_scan_K4_d0_{d0s}.log within 1 %: {okC}")
    allok = ok and okC
    if MUT:
        print(f"MUTATION {MUT}: a check must FAIL: {'caught' if not allok else 'NOT caught'}")
        return 0 if not allok else 3
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
