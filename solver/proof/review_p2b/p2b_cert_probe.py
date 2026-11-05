#!/usr/bin/env python3
"""P2b review, comparison step only: probes INSIDE the Lemma LC certificate's bound algebra (solver/proof/b8m_lc.py,
b8m_bounds.py, b8_alg.py, python-flint), to locate the differences found by p2b_compare_lc.py. It imports the
certificate's modules and re-assembles the pieces of its G1, W2 and g^ on each region at the certified constants
(the assembly lines are copied from b8m_lc.evaluate.build, which is a nested function); it is not part of the second
implementation and shares everything with the certificate.

    python3 p2b_cert_probe.py         # G1 pieces per region, g^ terms on N0 (b8m_lc_K4_preP2b.log, the certified run)
    python3 p2b_cert_probe.py lcj     # Lemma J's beta-row per band
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.dirname(HERE)
sys.path.insert(0, PROOF)
ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
from flint import arb  # noqa: E402
import b8_alg as E  # noqa: E402
import b8m_atoms as AT  # noqa: E402
import b8m_lc as LC  # noqa: E402

B, Q = E.B, E.Q


def main():
    rec = None
    for line in open(os.path.join(PROOF, "b8m_lc_K4_preP2b.log")):
        r = json.loads(line)
        if r.get("closes") and not r.get("mutate"):
            rec = r
    K = 4
    at = AT.atoms(K, rec["rA"], rec["rW"])
    g = LC.GeoLC(K, "73.89", atoms=at)
    k = LC.known_lc(g)
    h = LC.unflat({s: arb(repr(v)) for s, v in rec["hyp"].items()})
    iwS = E.inv_lo(E.lo_mono(1, eX=2 * K + 2, eT=1, eN=-(K + 1)))
    x, x2, nm1, Om, two = (k[s] for s in ("x", "x2", "nm1", "Om", "two"))
    cK, iCK, SK, VK, bK, AK, iAK, DK, BK, YK, q = (k[s] for s in ("cK", "iCK", "S", "V", "beta", "AK", "iAK", "DK",
                                                                  "BK", "Y", "q"))
    MK, rho2 = k["cb"], k["rho2"]
    RS = Q(LC.wB(g, h["S"], 1))
    RbH = Q(LC.wB(g, h["b"], 1))
    Rc = Q(LC.wB(g, h["c"], 0))
    RV = Q(LC.wB(g, h["V"], 0))
    Rb0 = q * RS
    c = cK + Rc
    c.lo = B()
    for r in E.REG:
        (kk, (aa, cc)), = cK.lo.t[r].items()
        c.lo.t[r][kk] = (aa, LC.lo(cc - E.bsup_r(Rc.v, r)))
    iC = E.qinv(Q(c.v, None, None, c.lo))
    S, V, Y = SK + RS, VK + RV, YK + RV + RS
    dC = Rc
    dD = (cK * two + Rc) * Rc
    D = Q(DK.v + dD.v)
    D.lo = B()
    for r in E.REG:
        (kk, (aa, cc)), = DK.lo.t[r].items()
        if r == "L":
            D.lo.t[r][kk] = (aa, cc)
            continue
        rel = E.bsup_r(dD.v * E.inv_lo(DK.lo), r)
        D.lo.t[r][kk] = (aa, LC.lo(cc * (1 - rel)))
    iD = E.qinv(D)
    RcbK = Rc * bK
    cRb = c * Rb0
    dB = (cRb + RcbK) * iC + MK * iC * dC * iCK + dC * Om * iC + cK * Om * iC * dC * iCK
    dA = nm1 * dD + dC * MK + c * (cRb + RcbK) + x2 * dB
    A0, B0 = AK + dA, BK + dB
    parts = dict(T1=iAK * (dA * S), T3=iAK * (dD * iD * (A0 * S + x2 * B0 * V)), T2=iAK * (x2 * (dB * V + BK * RV)),
                 rho2=iAK * rho2)
    print("certificate's G1 pieces at its constants, sup over the region of |piece|/w:")
    for r in ("N0", "Na", "Nb", "F"):
        print(f"  {r}: " + ", ".join(f"{nm} {float(E.bsup_r(p.v * iwS, r)):.5g}" for nm, p in parts.items()) +
              f"; |rho2| alone {float(E.bsup_r(rho2.v * iwS * B.mono(1, eN=-1), r)):.5g} (x n^-1)")
    # g^ on N0 and L (pass 2 with the certified derivative bounds D0), as in b8m_lc.evaluate.gbeta
    o = LC.evaluate(g, k, LC.kernel_lc(g, k), h)
    D0 = o["D0"]
    half, eps, eps1, xq, bbK, Dlt, rho3 = (k[s] for s in ("half", "eps", "eps1", "xq", "bbK", "Dlt", "rho3"))
    RS2 = Q(RS.v, D0["S"])
    RbH2 = Q(RbH.v, D0["b"])
    Rc2 = Q(Rc.v, D0["c"])
    RV2 = Q(RV.v, D0["V"])
    Rb2 = RbH2 + q * RS2
    c2 = cK + Rc2
    beta2 = bK + Rb2
    c2.d = (c2 * beta2).v
    c2.d2 = None
    c2.lo = c.lo
    iC2 = E.qinv(Q(c2.v, c2.d, None, c2.lo))
    S2, Y2 = SK + RS2, YK + RV2 + RS2
    Dl = x * iC2 * Y2
    H = (S2 * S2 + Dl * Dl) * half
    dDl = x * iC2 * (RV2 + RS2) + iC2 * Rc2 * Dlt
    g3 = bbK * (RS2 * RS2 * half + (x * iC2 * RV2 + iC2 * Rc2 * (x * iCK * RS2 + Dlt)) * Dlt + dDl * dDl * half)
    nf = LC.up(g.n0 / (g.n0 - 1))
    terms = dict(t1=Rb2 * (bK * two + Rb2), t2=eps * Rb2 * H, t3=g3, t4=xq * RS2, rho3=rho3)
    for r in ("N0",):
        print(f"certificate's g^ terms on {r}: sup |term|/((n-1) w) and sup |x term'|/((n-1) w):")
        for nm, t in terms.items():
            a = E.bsup_r(t.v * B.mono(nf, eN=-1) * iwS, r)
            b = E.bsup_r(t.d * B.mono(nf, eN=-1) * iwS, r) if t.d is not None else None
            print(f"    {nm}: {float(a):.5g}, {float(b) if b is not None else None:.5g}")
    return 0


if __name__ == "__main__" and "lcj" not in ARGS:
    sys.exit(main())


def probe_lcj():
    """the certificate's Lemma J b-row per band (b8m_lcj.main's loop, with both normalizations printed)"""
    import b8m_lcj as LJ
    rec = None
    for line in open(os.path.join(PROOF, "b8m_lc_K4_preP2b.log")):
        r = json.loads(line)
        if r.get("closes") and not r.get("mutate"):
            rec = r
    K = 4
    at = AT.atoms(K, rec["rA"], rec["rW"])
    g = LC.GeoLC(K, "73.89", atoms=at)
    k = LC.known_lc(g)
    h = LC.unflat({s: arb(repr(v)) for s, v in rec["hyp"].items()})
    dq = LJ.build_dq(g, k, h)
    one = B.mono(1)
    rhoB = B.mono(1, eX=2, eN=-1)
    sig = {"S": one, "V": rhoB, "c": rhoB, "b": B.mono(1, eX=2, eN=-2) + B.mono(1, eX=4, eN=-2)}
    inv_rho = E.inv_lo(E.lo_mono(1, eX=2, eN=-1))
    inv_nrho2 = E.inv_lo(E.lo_mono(1, eX=4, eN=-1))
    print("certificate's Lemma J beta-row: sup over each band of (ell_g,j s_j)/rho and /(n rho^2) (None: unbounded)")
    for j in ("S", "V", "b", "c"):
        fb = dq["gb"].g[j] * sig[j] if j in dq["gb"].g else B()
        row = []
        for r in LJ.SEG:
            vals = []
            for inv in (inv_rho, inv_nrho2):
                try:
                    vals.append(f"{float(E.bsup_r(fb * inv, r)):.4g}")
                except ValueError:
                    vals.append("None")
            row.append(f"{r} ({vals[0]}, {vals[1]})")
        print(f"  {j}: " + "; ".join(row))


if __name__ == "__main__" and "lcj" in ARGS:
    probe_lcj()
