#!/usr/bin/env python3
"""Lemma J (proof/B_chain.md section 14): the outer jump is determined by its label (hypothesis (H-J)), M = 1.

Setting. Y+ is the light-cone-gauge solution of Lemma LC (b8m_lc.py) on the closed upper sector, Y- its mirror image,
both with remainders R+-, Z^K + R+- , in the class K_h of the LC certificate (b8m_lc_K4.log) on the Stokes segment
(1, x*], x* = (n u*)^(1/2), u* = 0.0081. The jump v = Y+ - Y- = R+ - R- solves the secant system (differences of the
remainder equations of B8_constants.md Lemma R in the bounded form of Lemma LC):
    x v_V' = Om v_S,  x v_S' = Lambda (v_S + dG),  x v_b' + (n - 1) v_b = dg,  x v_c' = dFc,
with dG = G(R+) - G(R-) (G = G1 + G2), dg, dFc the differences of g^ and of the (I4) forcing. Their secant coefficients
are bounded locally, |dG(x)| <= sum_j l_{G,j}(x) |v_j(x)|, by forward differentiation of the bound algebra (class DQ;
inverses with their lower bounds held fixed: |g^-1 - g~^-1| <= |g - g~| / lo^2; on L the term dD/(x - 1) brings in
v_c' := sup_[1,x] |v_c'|).

Claim. If v -> 0 as x -> 1+ and the label of v at u* vanishes (Phi+(u*) = Phi-(u*) in B5's inner variables), v = 0.
Proof. In the norm N(v) = max_j sup |v_j| / (s_j m), m = e^{(n-1) Phi(x)/2} (half the fast exponent, Phi(x) =
(1/2) ln(1 - x^-2)), s_S = 1, s_V = s_c = rho, s_b = 1/n, the map T0 (fast from x* backward, with the endpoint value
from the label condition Lambda (v_S + dG) = -Om v_S + dC Phi- + d[x c' P]; slow and b-mode from 1 forward with zero
data) satisfies N(T0 v)_i <= sum_j Gam_ij N_j(v) with
    Gam_{S,j} = (1 + hH)/(1/2 - hH) sup(l_{G,j} s_j) + e_j s_j(x*)      (backward kernel |E(x,y)| <= e^{-(n-1)(1-hH)(Phi(y)-Phi(x))})
    Gam_{V,S} = 2|omega| n/(n-1),   Gam_{c,j} = 2 n/(n-1) sup(l_{Fc,j} s_j),   Gam_{b,j} = n/(n-1) sup(l_{g,j} s_j)
(hH = sup ||H'||). A positive vector w with Gam w < theta w, theta < 1, makes T0 a contraction. The jump is a fixed
point of T0 with N < infinity (a priori: on (1, 1 + delta/2] the same estimates with m = 1 make the affine map with the
actual endpoint value a contraction in both norms), so v = 0.

    python3 b8m_lcj.py [--log=b8m_lc_K4.log] [--L0=73.89] [--mutate=NAME]
    mutations (the contraction must fail): kernel (the backward factor 1/(1/2 - hH) replaced by the full rate, i.e. the
    fast kernel without decay: Gam_S multiplied by n), Om (the slow coupling |omega| multiplied by 1e4)
"""
import json
import sys
from flint import arb
import b8_alg as E
import b8m_atoms as AT
import b8m_lc as LC

B, Q = E.B, E.Q
up, lo = LC.up, LC.lo
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate", "")
SEG = ("L", "Na", "Nb", "F")          # the bands that contain the Stokes segment (1, x*]
COMPS = ("V", "S", "b", "c")


class DQ:
    """a node of the bound algebra with its local secant coefficients: |value| <= v, |delta value| <= sum_j g_j |v_j|"""

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
        if isinstance(o, (int, arb)):
            s = arb(o)
            return DQ(self.v * s, {j: b * s for j, b in self.g.items()})
        o = lift(o)
        g = {j: b * o.v for j, b in self.g.items()}
        for j, b in o.g.items():
            g[j] = g[j] + self.v * b if j in g else self.v * b
        return DQ(self.v * o.v, g)

    __rmul__ = __mul__


def lift(o):
    if isinstance(o, DQ):
        return o
    if isinstance(o, Q):
        return DQ(o.v)
    if isinstance(o, B):
        return DQ(o)
    raise TypeError(type(o))


def dinv(a, lob):
    il = E.inv_lo(lob)
    return DQ(il, {j: b * il * il for j, b in a.g.items()})


def lowB(vals):
    out = B()
    for r in E.REG:
        out.t[r][(0, 0)] = (0, vals[r])
    return out


def mixL(std, Ld):
    keys = set(std.g) | set(Ld.g)
    zero = B()

    def m(a, b):
        out = B({r: dict(a.t[r]) for r in E.REG})
        out.t["L"] = dict(b.t["L"])
        return out
    return DQ(m(std.v, Ld.v), {j: m(std.g.get(j, zero), Ld.g.get(j, zero)) for j in keys})


def build_dq(geo, k, h):
    K = geo.K
    d2 = geo.d2
    one = B.mono(1)
    kk = {s: lift(k[s]) for s in ("x", "x2", "eps", "eps1", "nm1", "Om", "half", "cK", "iCK", "S", "V", "beta", "AK",
                                  "iAK", "DK", "BK", "bbK", "Y", "q", "xq", "cb", "Dlt", "rho2", "rho3", "rho4",
                                  "ellx2")}
    x, x2, eps, eps1, nm1, Om, half = (kk[s] for s in ("x", "x2", "eps", "eps1", "nm1", "Om", "half"))
    cK, iCK, SK, VK, bK, AK, iAK, DK, BK, bbK, YK, q, xq = (kk[s] for s in (
        "cK", "iCK", "S", "V", "beta", "AK", "iAK", "DK", "BK", "bbK", "Y", "q", "xq"))
    MK, Dlt, rho2, rho3, rho4, ellx2 = (kk[s] for s in ("cb", "Dlt", "rho2", "rho3", "rho4", "ellx2"))
    iope = lift(E.qinv(k["ope"]))
    RS = DQ(LC.wB(geo, h["S"], 1), {"S": one})
    RbH = DQ(LC.wB(geo, h["b"], 1), {"b": one})
    Rc = DQ(LC.wB(geo, h["c"], 0), {"c": one})
    RV = DQ(LC.wB(geo, h["V"], 0), {"V": one})
    Rb = RbH + q * RS
    Rb0 = q * RS
    c = cK + Rc
    cloB = B()
    for r in E.REG:
        (key, (aa, cc)), = k["cK"].lo.t[r].items()
        v = lo(cc - E.bsup_r(Rc.v, r))
        if not v > 0:
            raise ValueError("c lower bound")
        cloB.t[r][key] = (aa, v)
    iC = dinv(c, cloB)
    beta = bK + Rb
    S, V, Y = SK + RS, VK + RV, YK + RV + RS
    dC = Rc
    dD = (cK * 2 + Rc) * Rc
    D = DK + dD
    DloB = B()
    for r in E.REG:
        (key, (aa, cc)), = k["DK"].lo.t[r].items()
        if r == "L":
            DloB.t[r][key] = (aa, cc)             # dummy: the L-entries of D^-1 are never used (L forms instead)
            continue
        rel = E.bsup_r(dD.v * E.inv_lo(k["DK"].lo), r)
        v = lo(cc * (1 - rel))
        if not v > 0:
            raise ValueError("D lower bound")
        DloB.t[r][key] = (aa, v)
    iD = dinv(D, DloB)
    RcbK = Rc * bK

    def AB(Rbx):
        cRb = c * Rbx
        dB = (cRb + RcbK) * iC + MK * iC * dC * iCK + dC * Om * iC + cK * Om * iC * dC * iCK
        dA = nm1 * dD + dC * MK + c * (cRb + RcbK) + x2 * dB
        return dA, dB

    dA0, dB0 = AB(Rb0)
    A0, B0 = AK + dA0, BK + dB0
    Mh = c * RbH
    G1 = iAK * (dA0 * S + dD * iD * (A0 * S + x2 * B0 * V) + x2 * (dB0 * V + BK * RV) + rho2)
    G2 = iAK * DK * iD * (c * Mh * S + x2 * Mh * iC * Y)
    Fc = Rc * beta + cK * Rb + rho4
    # the L forms (B_chain.md 13.3): dD/(x - 1) and Dt = D/(x - 1)
    s1 = arb(1) / (1 - d2)
    rcp = LC.supLn(Fc.v) * s1
    cKv, Rcv = LC.supLn(k["cK"].v), LC.supLn(Rc.v)
    ddv = (cKv * 2 + Rcv) * rcp
    gm = E.bsup_r(k["Gam"].v, "L")
    lDt = lo((2 - d2) * (1 - gm) - E.bsup_r(ddv, "L"))
    if not lDt > 0:
        raise ValueError("Dt lower bound")
    # |delta (dD/(x - 1))| <= (2|c^K| + 2|R_c|) sup_[1,x] |v_c'|  (pseudo-component "cp")
    ddL = DQ(ddv, {"cp": (cKv * 2 + Rcv * 2)})
    DtK = DQ(LC.onlyL(up((2 + d2) * (1 + gm))))
    iDt = dinv(DtK + ddL, lowB({r: (lDt if r == "L" else arb(1)) for r in E.REG}))
    G1L = iAK * (dA0 * S + ddL * iDt * (A0 * S + x2 * B0 * V) + x2 * (dB0 * V + BK * RV) + rho2)
    G2L = iAK * DtK * iDt * (c * Mh * S + x2 * Mh * iC * Y)
    G = mixL(G1 + G2, G1L + G2L)
    qxgS = ellx2 * iope * eps1 * AK * G
    Dl = x * iC * Y
    H = (S * S + Dl * Dl) * half
    dDl = x * iC * (RV + RS) + iC * dC * Dlt
    g3 = bbK * (RS * RS * half + (x * iC * RV + iC * dC * (x * iCK * RS + Dlt)) * Dlt + dDl * dDl * half)
    gb = Rb * (bK * 2 + Rb) + eps * Rb * H + g3 + xq * RS + qxgS + rho3
    P = iC * Y
    xcP = (c * beta) * P
    return dict(G=G, gb=gb, Fc=Fc, xcP=xcP, iC=iC, c=c, beta=beta, S=S, Y=Y, P=P)


def main():
    K = 4
    L0 = OPTS.get("L0", "73.89")
    rec = None
    for line in open(OPTS.get("log", "b8m_lc_K4.log")):
        r = json.loads(line)
        if r.get("closes") and abs(r["L0"] - float(L0)) < 1e-9 and not r.get("mutate"):
            rec = r
    if rec is None:
        print("no closing LC record")
        return 2
    at = AT.atoms(K, rec["rA"], rec["rW"])
    g = LC.GeoLC(K, L0, xa=rec.get("xa", "1.5"), atoms=at)
    k = LC.known_lc(g)
    kc = LC.kernel_lc(g, k)
    h = LC.unflat({s: arb(repr(v)) for s, v in rec["hyp"].items()})
    o = LC.evaluate(g, k, kc, h)
    P = o["P"]
    if not all(P[s][r] < h[s][r] for s in COMPS for r in LC.TARGETS) or not P["Sp"] < h["Sp"]:
        print("the LC hypothesis does not close (recheck)")
        return 1
    D0 = o["D0"]
    n0 = g.n0
    nf = up(n0 / (n0 - 1))
    dq = build_dq(g, k, h)
    one = B.mono(1)
    rhoB = B.mono(1, eX=2, eN=-1)
    # shapes (nondecreasing in x): s_S = 1, s_V = s_c = rho, s_b = rho/n + rho^2
    sig = {"S": one, "V": rhoB, "c": rhoB, "b": B.mono(1, eX=2, eN=-2) + B.mono(1, eX=4, eN=-2)}
    inv_rho = E.inv_lo(E.lo_mono(1, eX=2, eN=-1))           # 1/rho
    inv_nrho2 = E.inv_lo(E.lo_mono(1, eX=4, eN=-1))         # 1/(n rho^2)
    absw = g.C["absw"]
    if MUT == "Om":
        absw = absw * 10000

    def supseg(b, regs=SEG):
        return max((E.bsup_r(b, r) for r in regs), key=float)

    def coef(node, j, regs=SEG):
        gj = node.g.get(j)
        if gj is None:
            return arb(0)
        return supseg(gj * sig[j], regs)

    def fB(node, j, sg):
        """the B-object of the coefficient of N_j (input weight sg_j m) in |delta node|, the v_c' substitution on L
        included (v_c'(x) <= cpL[j] m(x) N_j)"""
        out = B()
        if j in node.g:
            out = out + node.g[j] * sg[j]
        if "cp" in node.g:
            out = out + node.g["cp"] * cpL[j]
        return out

    # hH = sup ||H'|| on the segment, ||H'|| <= ||eta|| / ((n-1)(1 - ||Gamma||))
    eta = k["x2"] * k["etaT"]
    hH = max((up(E.bsup_r(eta.v * B.mono(nf, eN=-1), r) / (1 - E.bsup_r(k["Gam"].v, r))) for r in SEG), key=float)
    if not hH < arb("0.5"):
        raise ValueError("hH >= 1/2")
    kb = up((1 + hH) / (arb("0.5") - hH))
    if MUT == "kernel":
        kb = kb * n0
    # the pseudo-component v_c' = sup_[1,x] |v_c'| on L: |v_c'(y)| = |dFc(y)|/|y| <= sum_j l_{Fc,j} s_j m N_j / (1 - d2)
    cpL = {j: up(coef(dq["Fc"], j, ("L",)) / (1 - g.d2)) for j in COMPS}
    GcpL = E.bsup_r(dq["G"].g["cp"], "L") if "cp" in dq["G"].g else arb(0)
    gcpL = E.bsup_r(dq["gb"].g["cp"], "L") if "cp" in dq["gb"].g else arb(0)
    fcpL = E.bsup_r(dq["Fc"].g["cp"], "L") if "cp" in dq["Fc"].g else arb(0)
    # the endpoint at x* (region F): Lambda (v_S + dG) = -Om v_S + dC Phi- + d[x c' P]
    iAK, DK = k["iAK"], k["DK"]
    Linv = E.bsup_r((iAK * DK).v, "F")                          # ||Lambda^-1|| = ||(A^K)^-1 D^K||
    xSd = k["S"].d + D0["S"]                                     # |x S'| <= |x S^K'| + |x R_S'|
    Phim = E.bsup_r((dq["iC"].v * (k["Om"].v * dq["S"].v + xSd + (dq["c"] * dq["beta"]).v * dq["P"].v)), "F")
    e = {}
    for j in COMPS:
        ej = E.bsup_r(dq["G"].g[j] * sig[j], "F") if j in dq["G"].g else arb(0)
        ex = E.bsup_r(dq["xcP"].g[j] * sig[j], "F") if j in dq["xcP"].g else arb(0)
        extra = arb(0)
        if j == "S":
            extra = absw
        if j == "c":
            extra = Phim * E.bsup_r(sig["c"], "F")
        e[j] = up(ej + Linv * (ex + extra))
    Gam = {i: {j: arb(0) for j in COMPS} for i in COMPS}
    for j in COMPS:
        Gam["S"][j] = up(kb * supseg(fB(dq["G"], j, sig)) + e[j])
        Gam["c"][j] = up(2 * nf * supseg(fB(dq["Fc"], j, sig)))
        fb = fB(dq["gb"], j, sig)
        # b-row: sup_y f(y) / (n s_b(y)), n s_b = rho + n rho^2 >= max(rho, n rho^2) (per band the better one)
        best = []
        for r in SEG:
            opts = []
            for inv in (inv_rho, inv_nrho2):
                try:
                    opts.append(E.bsup_r(fb * inv, r))
                except ValueError:
                    pass
            if not opts:
                raise ValueError(f"b-row unbounded on {r}")
            best.append(min(opts, key=float))
        Gam["b"][j] = up(nf * max(best, key=float))
    Gam["V"]["S"] = up(Gam["V"]["S"] + 2 * absw * nf)
    # the a priori norm (m = 1 on (1, 1 + delta/2], s_V = s_c = 1): for the identification of the jump
    sig0 = {"S": one, "V": one, "c": one, "b": B.mono(1, eN=-1)}
    Gam0 = {i: {j: arb(0) for j in COMPS} for i in COMPS}
    kb0 = up((1 + hH) / (1 - hH))
    for j in COMPS:
        cj = lambda node: (E.bsup_r(node.g[j] * sig0[j], "L") if j in node.g else arb(0))
        cpj = up(cj(dq["Fc"]) / (1 - g.d2))
        Gam0["S"][j] = up(kb0 * (cj(dq["G"]) + GcpL * cpj))
        Gam0["c"][j] = up(g.d2 * (cj(dq["Fc"]) + fcpL * cpj))
        Gam0["b"][j] = up(nf * (cj(dq["gb"]) + gcpL * cpj))
    Gam0["V"]["S"] = up(Gam0["V"]["S"] + absw * g.d2)

    def perron(Gm):
        idx = list(COMPS)
        w = {i: 1.0 for i in idx}
        for _ in range(500):
            nw = {i: sum(float(Gm[i][j]) * w[j] for j in idx) + 1e-30 for i in idx}
            mx = max(nw.values())
            w = {i: max(nw[i] / mx, 1e-12) for i in idx}
        wa = {i: arb(w[i]) for i in idx}
        th = max((up(sum((Gm[i][j] * wa[j] for j in idx), arb(0)) / wa[i]) for i in idx), key=float)
        return th, w

    th, w = perron(Gam)
    th0, w0 = perron(Gam0)
    ok = bool(th < 1) and bool(th0 < 1)
    print(f"Lemma J (n >= e^{L0}): sup ||H'|| = {float(hH):.4f}, backward factor {float(kb):.3f}; "
          f"||Lambda^-1(x*)|| <= {float(Linv):.3e}, |Phi-| <= {float(Phim):.3f}")
    for nm, Gm in (("Gamma (m = e^{(n-1)Phi/2})", Gam), ("Gamma0 (m = 1 on (1, 1 + delta/2])", Gam0)):
        print(f"  {nm}:")
        for i in COMPS:
            print("    " + i + ": " + "  ".join(f"{j} {float(Gm[i][j]):.3e}" for j in COMPS))
    print(f"  contraction constants: theta = {float(th):.4e} (weights {w}), theta0 = {float(th0):.4e}")
    out = dict(L0=float(L0), hH=float(hH), kb=float(kb), Linv=float(Linv), Phim=float(Phim), theta=float(th),
               theta0=float(th0), Gamma={i: {j: float(Gam[i][j]) for j in COMPS} for i in COMPS},
               Gamma0={i: {j: float(Gam0[i][j]) for j in COMPS} for i in COMPS}, ok=ok, mutate=MUT or None)
    if "json" in OPTS:
        json.dump(out, open(OPTS["json"], "w"), indent=1)
    if MUT:
        print(f"MUTATION {MUT}: the contraction must FAIL: {'caught' if not ok else 'NOT caught'}")
        return 0 if not ok else 3
    print(f"=> (H-J): {'PROVED for these data' if ok else 'not shown'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
