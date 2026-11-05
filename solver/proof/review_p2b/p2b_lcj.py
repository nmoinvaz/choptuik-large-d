#!/usr/bin/env python3
"""P2b review: a second implementation of Lemma J's secant coefficients ell and of its Gamma matrix (B_chain.md 14),
written from the statement and proof there, NOT from solver/proof/b8m_lcj.py (not opened before these numbers were
committed). The coefficients are the forward derivatives of p2b_lc.py's bound algebra (mpmath intervals) at the class
constants of Lemma LC, on cells covering the Stokes segment (1, x*], x* = (n u*)^(1/2), u* = 0.0081.

Pointwise secant bound (from the forward derivatives d_j of a node f with weight W_f, inputs |R_j| <= h_j w_j):
  |f(R+) - f(R-)|(x) <= sum_j (d_j/h_j) W_f(x)/w_j(x) |v_j(x)|,  so  ell_{f,j} s_j <= (d_j/h_j) sup W_f s_j/w_j .
On L, delta D/(x - 1) brings in p = sup_[1,x] |v_c'| <= sup |dF_c(s)|/|s|; with s_j m nondecreasing, p <= sum_j
sup_L(ell_{F_c,j} s_j/|s|) m(x) N_j, added to the L coefficients.

Gamma (B_chain.md 14): Gamma_{S,j} = (1 + h_H)/(1/2 - h_H) sup(ell_{G,j} s_j) + e_j s_j(x*), Gamma_{V,S} = 2|omega| n/(n-1),
Gamma_{c,j} = 2n/(n-1) sup(ell_{F_c,j} s_j), Gamma_{beta,j} = n/(n-1) sup(ell_{g,j} s_j/(n s_beta)), shapes s_S = 1,
s_V = s_c = rho, s_beta = rho/n + rho^2 (in [rho^2, rho^2 (1 + X^-2)]). The endpoint (label condition, derived here):
  v_S(x*) = -dG(x*) + Lambda^-1 (-Omega v_S + dC Phi^- + d[x c' P]),  d[x c' P] = [x v_c' P+] + [x c-' dP],
  |dP| <= |C^-1|(|v_V| + |v_S|) + |C^-1|^2 |Y| |v_c|,  |Phi^-| <= |C^-1|(|omega||S| + |x S'| + |x c'||P|),
  e_j = ell_{G,j}(x*) + |Lambda^-1(x*)| k_j,  k_V = |P| ell_{F_c,V} + |x c'||C^-1|,  k_S = |omega| + |P| ell_{F_c,S}
  + |x c'||C^-1|,  k_beta = |P| ell_{F_c,beta},  k_c = |Phi^-| + |P| ell_{F_c,c} + |x c'||C^-1|^2 |Y|.

    python3 p2b_lcj.py --h=p2b_lc_cells.json [--mutate=noend|omega|shape]
"""
import json
import os
import sys
from fractions import Fraction as Fr

from mpmath import iv, mp

import p2b_alg as Al
import p2b_lc as L
import p2b_series as Ser
from p2b_alg import Cell, Bd, bto, bnum, bmul, badd, up, lo, mpup, mplo, ZEROI, ONEI, HD

HERE = os.path.dirname(os.path.abspath(__file__))
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
USTAR = iv.mpf("0.0081")
COMPJ = ("S", "V", "B", "C")          # the order of B_chain.md 14 (S, V, beta, c)
WIN = {"V": L.W1, "S": L.W, "B": L.W, "C": L.W1}
SHAPE = {"S": (0, 0, 0), "V": (0, 0, 1), "C": (0, 0, 1), "B": (0, 0, 2)}


def jcells(st, rho_star):
    cells = []
    Lc = Cell(st, "L", "L", 0, 0)
    Lc.aux = False
    cells.append(Lc)
    for band, pts in (("Na", L.geom_points("1.025", "1.5", 12)), ("Nb", L.geom_points("1.5", "20", 30))):
        for a, b in zip(pts[:-1], pts[1:]):
            c = Cell(st, band, "real", str(a), str(b))
            c.aux = False
            cells.append(c)
    for a, b in zip(["20", "29", "49", "99", "999", "1e5", "1e8"], ["29", "49", "99", "999", "1e5", "1e8", None]):
        c = Cell(st, "F", "real", a, b)
        c.aux = False
        if mpup(c.rho_cap) > mpup(rho_star):
            c.rho_cap = rho_star
        cells.append(c)
    return cells


def coef(node, j, cell, extra=(0, 0, 0)):
    """sup over the cell of ell_{node,j} s_j / W_extra (own-weight forward derivative with the seed s_j)"""
    return Al.dsup(node, cell, j, extra)


def main():
    st = Al.Setting(OPTS.get("ln_n0", "73.89"))
    hfile = OPTS.get("h", "p2b_lc_cells.json")
    if hfile == "cert":
        # the certificate's class constants (b8m_lc_K4.log hyp, b8m_lc_lip.json hprime); h_cc from p2b's closure
        CM = {"V": "V", "S": "S", "B": "b", "C": "c"}
        rec = None
        for line in open(os.path.join(os.path.dirname(HERE), "b8m_lc_K4.log")):
            r_ = json.loads(line)
            if r_.get("closes") and not r_.get("mutate"):
                rec = r_
        lip = json.load(open(os.path.join(os.path.dirname(HERE), "b8m_lc_lip.json")))
        own = json.load(open(os.path.join(HERE, "p2b_lc_cells.json")))
        H = {}
        for c_ in L.COMP:
            for b_ in L.BANDS:
                H[("h", c_, b_)] = iv.mpf(repr(rec["hyp"][f"{CM[c_]}.{b_}"]))
                H[("d", c_, b_)] = iv.mpf(repr(lip["hprime"][f"{CM[c_]}.{b_}"]))
        H[("d", "S", "L")] = iv.mpf(repr(rec["hyp"]["Sp"]))
        H[("cc",)] = iv.mpf(own["h"]["cc,L"])
    else:
        with open(os.path.join(HERE, hfile)) as fh:
            Hj = json.load(fh)["h"]
        H = {k: iv.mpf(Hj[L.idxname(L.key_index(k))]) for k in L.KEYS}
    T = Ser.tables()
    known = L.Known(st, T)
    L.KN = known
    Al.OWNW = True
    shape_fac = {j: (lambda c: ONEI) for j in COMPJ}
    shape_fac["B"] = lambda c: up(1 + iv.mpf(1) / iv.mpf(c.Xlo) ** 2)        # s_beta <= rho^2 (1 + X^-2)
    sh = dict(SHAPE)
    if MUT == "shape":
        sh["V"] = (0, 0, 0)                                                   # s_V = 1 instead of rho
    L.JSEEDS = {j: (sh[j], shape_fac[j]) for j in COMPJ}
    rho_star = up((iv.sqrt(st.nu0) + iv.sqrt(USTAR)) ** 2)          # rho(x*) for every n >= n0
    cells = jcells(st, rho_star)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log(f"P2b Lemma J second implementation: class constants from {hfile}, ln n0 = {st.ln_n0}, mutate = {MUT}")
    log(f"  u* = 0.0081, rho(x*) <= {mp.nstr(mpup(rho_star), 8)}; {len(cells)} cells on the Stokes segment")
    sup = {(f, j): ZEROI for f in ("G", "g", "F") for j in COMPJ}
    hH = ZEROI
    Lc = cells[0]
    res = {}
    for c in cells:
        kn = known.at(c)
        r = L.cell_eval(c, kn, H, False, st)
        res[id(c)] = r
        hp = up(kn["n_eps1eta"] / (1 - kn["n_Gam"]))
        if mpup(hp) > mpup(hH):
            hH = hp
        b = c.band
        loc = {}
        for j in COMPJ:
            loc[("G", j)] = coef(r["G"][0], j, c)
            loc[("F", j)] = coef(r["Fc"][0], j, c)
            # ell_g s_j/(n s_beta) <= ell_g s_j/(n rho^2) = ell_g s_j/(X^2 rho)
            loc[("g", j)] = coef(r["ghat"][0], j, c, extra=(2, 0, 1))
        if c.kind == "L":
            # the pseudo-component p = sup_[1,x] |v_c'| enters G and g through P_c: p <= sum_j Q_j m(x) N_j,
            # Q_j = sup_L ell_{F_c,j} s_j/|s|
            Qj = {j: coef(r["Fc"][0], j, c, extra=(1, 1, 0)) for j in COMPJ}
            for f, node, extra in (("G", r["G"][0], (0, 0, 0)), ("g", r["ghat"][0], (2, 0, 1))):
                lp = coef(node, L.PCIDX, c, extra)
                for j in COMPJ:
                    loc[(f, j)] = up(loc[(f, j)] + lp * Qj[j])
            log(f"  L: pseudo-component: ell_G,pc = {mp.nstr(mpup(coef(r['G'][0], L.PCIDX, c)), 4)}, "
                f"Q_j = sup_L(ell_Fc,j s_j/|s|): " + ", ".join(f"{j} {mp.nstr(mpup(Qj[j]), 4)}" for j in COMPJ))
        for k, v in loc.items():
            if mpup(v) > mpup(sup[k]):
                sup[k] = v
        res[id(c)]["loc"] = loc
    log(f"  h_H = sup |H'| on the segment = {mp.nstr(mpup(hH), 6)}")
    if mpup(hH) >= 0.5:
        raise ValueError("h_H >= 1/2")
    # endpoint at x*: the last far cell (it contains x* for every n >= n0)
    cF = cells[-1]
    r = res[id(cF)]
    kn = known.at(cF)
    num = lambda B: bnum(B, cF).v
    # |Lambda^-1| = |(A^K)^-1 D^K| <= |(A^K)^-1| |x^2 - 1| (1 + |Gamma|)
    x2m1 = Bd((2, 0, 0), cF.d2hi)
    Laminv = num(bmul(bmul(kn["AKinv"][0], x2m1), badd(cF, kn["one"][0], kn["Gam"][0])))
    Cinv = num(r["Cinv"][0])
    Y = num(r["Y"][0])
    P = up(Cinv * Y)
    xcp = num(r["Cj"][1])                          # |x c'| <= |x c^K'| + |x R_c'| (class)
    xSp = num(r["S"][1])
    Phi = up(Cinv * (known.absW * num(r["S"][0]) + xSp + xcp * P))
    lF = {j: res[id(cF)]["loc"][("F", j)] for j in COMPJ}
    lG = {j: res[id(cF)]["loc"][("G", j)] for j in COMPJ}
    sx = {"S": ONEI, "V": rho_star, "C": rho_star, "B": up(rho_star ** 2 * (1 + st.nu0 / USTAR))}
    # ell_{F_c,j} at x* times s_j(x*) is lF[j]; the k_j multiply |v_j(x*)| = N_j s_j(x*) m(x*)
    kap = {"V": up(P * lF["V"] + xcp * Cinv * sx["V"]),
           "S": up(known.absW + P * lF["S"] + xcp * Cinv),
           "B": up(P * lF["B"]),
           "C": up(Phi * sx["C"] + P * lF["C"] + xcp * Cinv * Cinv * Y * sx["C"])}
    e = {j: up(lG[j] + Laminv * kap[j]) for j in COMPJ}
    if MUT == "noend":
        e = {j: ZEROI for j in COMPJ}
    log(f"  endpoint at x*: |Lambda^-1| <= {mp.nstr(mpup(Laminv), 5)}, |C^-1| <= {mp.nstr(mpup(Cinv), 5)}, |P| <= "
        f"{mp.nstr(mpup(P), 5)}, |x c'| <= {mp.nstr(mpup(xcp), 5)}, |Phi^-| <= {mp.nstr(mpup(Phi), 5)}")
    log("  e_j s_j(x*): " + ", ".join(f"{j} {mp.nstr(mpup(e[j]), 5)}" for j in COMPJ))
    om = known.absW if MUT != "omega" else known.absW * 10 ** 4
    kS = up((1 + hH) / (iv.mpf("0.5") - hH))
    if MUT == "nodecay":
        kS = up(kS * st.n0)                      # the backward kernel without decay (as b8m_lcj's own mutation)
    nn = up(1 / (1 - st.nu0))
    G = {}
    for j in COMPJ:
        G[("S", j)] = up(kS * sup[("G", j)] + e[j])
        G[("V", j)] = up(2 * om * nn) if j == "S" else ZEROI
        G[("C", j)] = up(2 * nn * sup[("F", j)])
        G[("B", j)] = up(nn * sup[("g", j)])
    log("  sup ell s_j over the segment (G, F_c, g/(n s_beta)):")
    for f in ("G", "F", "g"):
        log(f"    {f}: " + ", ".join(f"{j} {mp.nstr(mpup(sup[(f, j)]), 5)}" for j in COMPJ))
    log("  Gamma (rows S, V, beta, c; columns S, V, beta, c):")
    for i in COMPJ:
        log("    " + "  ".join(f"{mp.nstr(mpup(G[(i, j)]), 5):>11s}" for j in COMPJ))
    M = [[float(mpup(G[(i, j)])) for j in COMPJ] for i in COMPJ]
    import numpy as np
    ev = max(abs(np.linalg.eigvals(np.array(M))))
    # certified theta: positive v with Gamma v <= theta v in exact rationals (upper bounds)
    lam = ev * 1.02 + 1e-4
    v = np.linalg.solve(np.eye(4) - np.array(M) / lam, np.ones(4))
    vq = [Fr(float(a)).limit_denominator(10 ** 12) for a in v]
    Gq = [[L.mpf_to_frac_up(mpup(G[(i, j)])) for j in COMPJ] for i in COMPJ]
    theta = max(sum(Gq[a][b] * vq[b] for b in range(4)) / vq[a] for a in range(4))
    log(f"  Perron root of Gamma {ev:.5f}; certified theta = {float(theta):.5f} (weight {[float(a) for a in vq]})")
    out = dict(Gamma=M, order=list(COMPJ), rho=ev, theta=float(theta), hH=float(mpup(hH)),
               e={j: float(mpup(e[j])) for j in COMPJ}, Laminv=float(mpup(Laminv)), Phi=float(mpup(Phi)),
               sup={f"{f},{j}": float(mpup(v_)) for (f, j), v_ in sup.items()}, hfile=hfile, mutate=MUT)
    tag = OPTS.get("tag", os.path.splitext(hfile)[0].replace("p2b_lc_", "") + (f"_mut_{MUT}" if MUT else ""))
    out["Gamma_cert"] = None
    with open(os.path.join(HERE, f"p2b_lcj_{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    if OPTS.get("log"):
        with open(os.path.join(HERE, OPTS["log"]), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    ok = theta < 1
    if MUT:
        print(f"MUTATION {MUT}: theta = {float(theta):.4g}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        print(f"FAILED: {exc}" + (f" (mutation {MUT}: caught)" if MUT else ""))
        sys.exit(2)
