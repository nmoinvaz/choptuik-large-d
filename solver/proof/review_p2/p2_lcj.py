#!/usr/bin/env python3
"""P2 review: an independent implementation of the Lemma J certificate (proof/B_chain.md section 14), on the atoms,
bound algebra and known quantities of p2_lc.py (nothing imported from solver/proof).

The jump v = R+ - R- of the light-cone-gauge solution and its mirror solves, on the Stokes segment (1, x*], the
differences of the remainder equations; its secant coefficients are bounded locally by forward differentiation:
a node carries its value bound and, for each component j, a bound l_j(x) (a function of x, a Bd) with
|delta node(x)| <= sum_j l_j(x) |v_j(x)| (+ l_cp sup_[1,x] |v_c'| on L). Lower bounds of the inverses are those of the
class (both R+ and R- are in it), so |delta(a^-1)| <= |delta a| / lo^2. The actual solutions satisfy the c-equation, so
|x c'| = |[c beta]| is legitimate here.

With the norm N(v) = max_j sup |v_j|/(s_j m), m = e^{(n-1) Phi(x)/2}, s_S = 1, s_V = s_c = rho, s_b = rho/n + rho^2,
the map T0 (S backward from x* with the endpoint from the label condition; V, c, b forward from 1 with zero data)
satisfies N(T0 v)_i <= sum_j Gamma_ij N_j(v), with (re-derived in P2_review.md section 2.2)
    Gamma_{S,j} = (1 + hH)/(1/2 - hH) sup(l_{G,j} s_j) + e_j,    Gamma_{V,S} = 2|omega| n/(n - 1),
    Gamma_{c,j} = 2n/(n - 1) sup(l_{Fc,j} s_j),    Gamma_{b,j} = n/(n - 1) sup(l_{g,j} s_j / (n s_b)),
and Gamma0 (m = 1 on (1, 1 + delta/2]) for the a priori argument. A Perron weight with Gamma w <= theta w, theta < 1,
proves (H-J).

    python3 p2_lcj.py [--mutate=kernel|Om]
    mutations (theta must exceed 1): kernel (the backward factor multiplied by n0, a kernel without decay), Om (|omega|
    multiplied by 1e4)
"""
import json
import os
import sys
from fractions import Fraction

from mpmath import iv, mp
from mpmath.libmp import fone, fzero

import p2_alg as G
from p2_alg import R, Lo, Bd, Q, Reg, bsup, inv_lo, lo_mono, up_iv, lo_iv, add_u, mul_u, div_u, fl, gt
from p2_common import lo as LO, hi as HI
import p2_lc as LC

OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate", "")
HERE = os.path.dirname(os.path.abspath(__file__))
COMPS = ("V", "S", "b", "c")
SEG = ("L", "Na", "Nb", "F")


class DQ:
    __slots__ = ("v", "g")

    def __init__(self, v, g=None):
        self.v = v
        self.g = dict(g or {})

    def __add__(self, o):
        o = lift(o)
        g = dict(self.g)
        for j, b in o.g.items():
            g[j] = g[j] + b if j in g else b
        return DQ(self.v + o.v, g)

    def __mul__(self, o):
        o = lift(o)
        g = {j: b * o.v for j, b in self.g.items()}
        for j, b in o.g.items():
            t = self.v * b
            g[j] = g[j] + t if j in g else t
        return DQ(self.v * o.v, g)

    __rmul__ = __mul__


def lift(o):
    if isinstance(o, DQ):
        return o
    if isinstance(o, Q):
        return DQ(o.v)
    if isinstance(o, Bd):
        return DQ(o)
    raise TypeError(type(o))


def dinv(a, lob):
    il = inv_lo(lob)
    return DQ(il, {j: b * il * il for j, b in a.g.items()})


def mixL(std, Lq):
    def m(a, b):
        out = Bd({r: dict(a.t[r]) for r in Reg.regs})
        out.t["L"] = dict(b.t["L"])
        return out
    keys = set(std.g) | set(Lq.g)
    return DQ(m(std.v, Lq.v), {j: m(std.g.get(j, Bd()), Lq.g.get(j, Bd())) for j in keys})


def build(geo, k, h):
    d2 = geo.d2
    one = Bd.mono(R(fone))
    x, x2, eps, eps1, nm1, Om, half, two = (k[s] for s in ("x", "x2", "eps", "eps1", "nm1", "Om", "half", "two"))
    cK, iCK, SK, VK, bK, AK, iAK, DK, BK, bbK, YK, q, xq = (k[s] for s in (
        "cK", "iCK", "S", "V", "beta", "AK", "iAK", "DK", "BK", "bbK", "Y", "q", "xq"))
    MK, Dlt, rho2, rho3, rho4 = k["cb"], k["Dlt"], k["rho2"], k["rho3"], k["rho4"]
    RS = DQ(LC.wB(h["S"], 1, False), {"S": one})
    RbH = DQ(LC.wB(h["b"], 1, False), {"b": one})
    Rc = DQ(LC.wB(h["c"], 0, False), {"c": one})
    RV = DQ(LC.wB(h["V"], 0, False), {"V": one})
    Rb = RbH + q * RS
    Rb0 = q * RS
    c = lift(cK) + Rc
    clo = {r: (k["cK"].lo[r][0].minus(bsup(Rc.v, r)), 0, 0, 0) for r in Reg.regs}
    iC = dinv(c, clo)
    beta = lift(bK) + Rb
    S, V, Y = lift(SK) + RS, lift(VK) + RV, lift(YK) + RV + RS
    dC = Rc
    dD = (lift(cK) * two + Rc) * Rc
    D = lift(DK) + dD
    Dlo = {}
    for r in Reg.regs:
        if r == "L":
            Dlo[r] = (Lo(up_iv("1e-300")), 0, 0, 0)
            continue
        (cl, a, b, e) = DK.lo[r]
        rel = bsup(dD.v * inv_lo(DK.lo), r)
        Dlo[r] = (cl.times_one_minus(rel), a, b, e)
    iD = dinv(D, Dlo)
    RcbK = Rc * bK

    def AB(Rbx):
        cRb = c * Rbx
        dB = (cRb + RcbK) * iC + MK * iC * dC * iCK + dC * Om * iC + cK * Om * iC * dC * iCK
        dA = nm1 * dD + dC * MK + c * (cRb + RcbK) + x2 * dB
        return dA, dB

    dA0, dB0 = AB(Rb0)
    A0, B0 = lift(AK) + dA0, lift(BK) + dB0
    Mh = c * RbH
    G1 = iAK * (dA0 * S + dD * iD * (A0 * S + x2 * B0 * V) + x2 * (dB0 * V + BK * RV) + rho2)
    G2 = iAK * DK * iD * (c * Mh * S + x2 * Mh * iC * Y)
    Fc = Rc * beta + cK * Rb + rho4
    # the L forms
    s1 = G.div_u(fone, lo_iv(1 - d2))
    rcp = LC.supLn(Fc.v) * s1
    cKv, Rcv = LC.supLn(cK.v), LC.supLn(Rc.v)
    ddv = (cKv * R.c(2) + Rcv) * rcp
    gm = bsup(k["Gam"].v, "L")
    lDt = Lo(lo_iv(2 - d2)).times_one_minus(gm).minus(bsup(ddv, "L"))
    if not lDt.positive():
        raise ValueError("Dt lower bound")
    ddL = DQ(ddv, {"cp": (cKv * R.c(2) + Rcv * R.c(2))})       # |delta(dD/(x-1))| <= (2|c^K| + 2|R_c|) sup|v_c'|
    DtK = DQ(LC.onlyL(R(up_iv(2 + d2)) * (R(fone) + gm)))
    iDt = dinv(DtK + ddL, {r: ((lDt if r == "L" else Lo(fone)), 0, 0, 0) for r in Reg.regs})
    G1L = iAK * (dA0 * S + ddL * iDt * (A0 * S + x2 * B0 * V) + x2 * (dB0 * V + BK * RV) + rho2)
    G2L = iAK * DtK * iDt * (c * Mh * S + x2 * Mh * iC * Y)
    Gt = mixL(G1 + G2, G1L + G2L)
    qxgS = k["ellx2"] * k["iope"] * eps1 * AK * Gt
    Dl = x * iC * Y
    H = (S * S + Dl * Dl) * half
    dDl = x * iC * (RV + RS) + iC * dC * Dlt
    g3 = bbK * (RS * RS * half + (x * iC * RV + iC * dC * (x * iCK * RS + Dlt)) * Dlt + dDl * dDl * half)
    gb = Rb * (lift(bK) * two + Rb) + eps * Rb * H + g3 + xq * RS + qxgS + rho3
    P = iC * Y
    xcP = (c * beta) * P
    return dict(G=Gt, gb=gb, Fc=Fc, xcP=xcP, iC=iC, c=c, beta=beta, S=S, Y=Y, P=P)


def supseg(b, regs=SEG):
    return max((bsup(b, r) for r in regs), key=lambda v: fl(v.v))


def perron(Gm):
    idx = list(COMPS)
    w = {i: 1.0 for i in idx}
    for _ in range(2000):
        nw = {i: sum(fl(Gm[i][j].v) * w[j] for j in idx) + 1e-300 for i in idx}
        mx = max(nw.values())
        w = {i: max(nw[i] / mx, 1e-12) for i in idx}
    W = {i: Fraction(w[i]) for i in idx}
    Mq = {i: {j: Fraction(fl(Gm[i][j].v)) * (1 + Fraction(1, 10 ** 15)) for j in idx} for i in idx}
    th = max(sum((Mq[i][j] * W[j] for j in idx), Fraction(0)) / W[i] for i in idx)
    return th, w


def main():
    geo, D, k, kc, rems = LC.setup(OPTS.get("L0", "73.89"))
    cert = None
    for line in open(os.path.join(HERE, "..", "b8m_lc_K4.log")):
        r = json.loads(line)
        if r.get("closes") and not r.get("mutate"):
            cert = r
    h = LC.h_from(cert)
    o = LC.evaluate(geo, k, kc, h)
    if not all(gt(h[cp][r].v, o["P"][cp][r].v) for cp in COMPS for r in LC.TARGETS):
        print("the LC class does not close in this implementation")
        return 1
    D0 = o["D0"]
    dq = build(geo, k, h)
    nf = geo.nf
    d2 = geo.d2
    one = Bd.mono(R(fone))
    rhoB = Bd.mono(R(fone), a=2, e=-1)
    sig = {"S": one, "V": rhoB, "c": rhoB, "b": Bd.mono(R(fone), a=2, e=-2) + Bd.mono(R(fone), a=4, e=-2)}
    inv_rho = inv_lo(lo_mono(fone, a=2, e=-1))
    inv_nrho2 = inv_lo(lo_mono(fone, a=4, e=-1))
    absw = bsup(k["Om"].v, "F")
    if MUT == "Om":
        absw = absw.scale(up_iv(10000))
    eta = (k["x2"] * k["etaT"]).v
    hHs = []
    for r in SEG:
        e_ = bsup(eta * Bd.mono(R(nf), e=-1), r)
        g_ = bsup(k["Gam"].v, r)
        hHs.append(iv.mpf(mp.make_mpf(e_.v)) / (1 - iv.mpf(mp.make_mpf(g_.v))))
    hH = max(HI(v) for v in hHs)
    if not hH < mp.mpf("0.5"):
        raise ValueError("hH >= 1/2")
    kb = R(up_iv((1 + iv.mpf(hH)) / (iv.mpf("0.5") - iv.mpf(hH))))
    if MUT == "kernel":
        kb = kb.scale(up_iv(geo.n0))
    s1 = G.div_u(fone, lo_iv(1 - d2))

    def coef(node, j, regs=SEG):
        gj = node.g.get(j)
        return R(fzero) if gj is None else supseg(gj * sig[j], regs)

    cpL = {j: coef(dq["Fc"], j, ("L",)).scale(s1) for j in COMPS}

    def fB(node, j, sg, cpv):
        out = Bd()
        if j in node.g:
            out = out + node.g[j] * sg[j]
        if "cp" in node.g:
            out = out + node.g["cp"] * cpv[j]
        return out

    # endpoint at x* (region F)
    Linv = bsup((k["iAK"] * k["DK"]).v, "F")
    xSd = k["S"].d + D0["S"]
    Phim = bsup(dq["iC"].v * (k["Om"].v * dq["S"].v + xSd + (dq["c"] * dq["beta"]).v * dq["P"].v), "F")
    e = {}
    for j in COMPS:
        ej = bsup(dq["G"].g[j] * sig[j], "F") if j in dq["G"].g else R(fzero)
        ex = bsup(dq["xcP"].g[j] * sig[j], "F") if j in dq["xcP"].g else R(fzero)
        extra = absw if j == "S" else (Phim * bsup(sig["c"], "F") if j == "c" else R(fzero))
        e[j] = ej + Linv * (ex + extra)
    Gam = {i: {j: R(fzero) for j in COMPS} for i in COMPS}
    for j in COMPS:
        Gam["S"][j] = kb * supseg(fB(dq["G"], j, sig, cpL)) + e[j]
        Gam["c"][j] = supseg(fB(dq["Fc"], j, sig, cpL)).scale(mul_u(up_iv(2), nf))
        fb = fB(dq["gb"], j, sig, cpL)
        best = []
        for r in SEG:
            opts = []
            for inv in (inv_rho, inv_nrho2):
                try:
                    opts.append(bsup(fb * inv, r))
                except ValueError:
                    pass
            best.append(min(opts, key=lambda v: fl(v.v)))
        Gam["b"][j] = max(best, key=lambda v: fl(v.v)).scale(nf)
    Gam["V"]["S"] = Gam["V"]["S"] + absw.scale(mul_u(up_iv(2), nf))
    # the a priori norm (m = 1 on (1, 1 + delta/2]; s0_V = s0_c = 1, s0_b = 1/n)
    sig0 = {"S": one, "V": one, "c": one, "b": Bd.mono(R(fone), e=-1)}
    kb0 = R(up_iv((1 + iv.mpf(hH)) / (1 - iv.mpf(hH))))
    GcpL = bsup(dq["G"].g["cp"], "L") if "cp" in dq["G"].g else R(fzero)
    gcpL = bsup(dq["gb"].g["cp"], "L") if "cp" in dq["gb"].g else R(fzero)
    fcpL = bsup(dq["Fc"].g["cp"], "L") if "cp" in dq["Fc"].g else R(fzero)
    Gam0 = {i: {j: R(fzero) for j in COMPS} for i in COMPS}
    for j in COMPS:
        cj = lambda node: (bsup(node.g[j] * sig0[j], "L") if j in node.g else R(fzero))
        cpj = cj(dq["Fc"]).scale(s1)
        Gam0["S"][j] = kb0 * (cj(dq["G"]) + GcpL * cpj)
        Gam0["c"][j] = (cj(dq["Fc"]) + fcpL * cpj).scale(up_iv(d2))
        Gam0["b"][j] = (cj(dq["gb"]) + gcpL * cpj).scale(nf)
    Gam0["V"]["S"] = Gam0["V"]["S"] + absw.scale(up_iv(d2))
    th, w = perron(Gam)
    th0, w0 = perron(Gam0)
    ref = json.load(open(os.path.join(HERE, "..", "b8m_lcj.json")))
    print(f"Lemma J (independent): sup ||H'|| = {float(hH):.4f}, backward factor {fl(kb.v):.4f}; ||Lambda^-1(x*)|| <= "
          f"{fl(Linv.v):.4e} (b8m_lcj: {ref['Linv']:.4e}); |Phi-| <= {fl(Phim.v):.4f} ({ref['Phim']:.4f})")
    for nm, Gm, rf in (("Gamma", Gam, ref["Gamma"]), ("Gamma0", Gam0, ref["Gamma0"])):
        print(f"  {nm} (this implementation / b8m_lcj.json):")
        for i in COMPS:
            print("    " + i + ": " + "  ".join(f"{j} {fl(Gm[i][j].v):.3e} / {rf[i][j]:.3e}" for j in COMPS))
    ok = th < 1 and th0 < 1
    print(f"  theta = {float(th):.5f} (b8m_lcj: {ref['theta']:.5f}); theta0 = {float(th0):.3e} ({ref['theta0']:.3e})")
    out = dict(theta=float(th), theta0=float(th0), hH=float(hH), Gamma={i: {j: fl(Gam[i][j].v) for j in COMPS} for i in COMPS},
               Gamma0={i: {j: fl(Gam0[i][j].v) for j in COMPS} for i in COMPS}, ok=bool(ok), mutate=MUT or None)
    with open(os.path.join(HERE, "p2_lcj.json" if not MUT else f"p2_lcj_mut_{MUT}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    if MUT:
        print(f"MUTATION {MUT}: theta must exceed 1: {'caught' if not ok else 'NOT caught'}")
        return 0 if not ok else 3
    print(f"=> (H-J) {'CONFIRMED' if ok else 'NOT confirmed'} by this implementation")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
