#!/usr/bin/env python3
"""P2c review (#988): Lemma J's a priori contraction theta0 on (1, 1 + delta/2], recomputed independently of
solver/proof/b8m_lcj.py (not opened before the numbers below were committed).

The statement (B_chain.md 14, "Psi_out lies in the space"): on I = (1, x1], x1 = 1 + delta/2, with m = 1, the affine
map with the actual endpoint value v_S(x1) is a contraction, theta0 < 1. Its linear part, derived here from the
difference equations of B_chain.md 14 (x v_V' = Omega v_S, x v_S' = Lambda (v_S + dG), x v_b' + (n - 1) v_b = dg,
x v_c' = dF_c), in the plain sup norms sup_I |v_j| (m = 1, every shape 1):
  - S, backward from x1 with the kernel bound |E(x, y)| <= exp(-(n - 1)(1 - h_H)(Phi(y) - Phi(x))) and
    |Lambda| / y <= (n - 1)(1 + h_H) Phi'(y): Gamma0_{S,j} = (1 + h_H)/(1 - h_H) sup_I ell_{G,j}
    (the weighted norm of the main contraction has 1/2 - h_H in place of 1 - h_H, because m takes half the decay);
  - V, forward from 1 with v_V(1) = 0 and |Omega| <= |omega|: Gamma0_{V,S} = |omega| ln(x1);
  - c, forward from 1 with v_c(1) = 0: Gamma0_{c,j} = ln(x1) sup_I ell_{F_c,j};
  - beta, forward with the gain of (y/x)^(n - 1): Gamma0_{b,j} = sup_I ell_{g,j} / (n - 1);
  - on L the term dD/(x - 1) brings in p(x) = sup_[1,x] |v_c'| <= sum_j sup_I(ell_{F_c,j}/|s|) N_j, added to the
    G and g coefficients through their pseudo-component (as P2b does in the weighted norm).
theta0 is certified as P2b certifies theta: a positive rational weight v with Gamma0 v <= theta0 v componentwise, the
entries of Gamma0 rounded up to rationals.

The secant coefficients ell are P2b's (review_p2b/p2b_lc.py, p2b_lcj.py: mpmath intervals, written from the
statements, sharing no code with b8m_lc.py or b8m_lcj.py), evaluated on P2b's L cell (1, 1.025] with unit shapes, at
either P2b's own class constants (--h=p2b_lc_cells.json) or the certificate's (--h=cert, as p2b_lcj.py --h=cert).

    python3 p2c_theta0.py [--h=p2b_lc_cells.json|cert] [--rate=full|half] [--mutate=nodecay] [--log=FILE]
    python3 p2c_theta0.py --planted [--mutate=perron]

--planted runs the Gamma0 assembly and the Perron certification on planted inputs with a known Perron root
(a rank-one matrix u w^T, root w.u); --planted --mutate=perron raises one entry so that the root exceeds 1, and the
check must fail. --mutate=nodecay multiplies the backward kernel by n0 (b8m_lcj's own mutation, as in B_chain.md 14);
the protocol fixed before the run required theta0 > 1 from it, and it gave 0.564: in the m = 1 norm the S row's
coefficients are O(rho) = O(1/n), so a factor n0 leaves them O(1) (P2_remaining.md 2.2). Added after that run, with that
reason: --mutate=gaininv multiplies the beta row by (n0 - 1) instead of dividing by n - 1 (an inverted gain at n = n0;
the inversion for every n >= n0 is unbounded, and P2b's algebra refuses it), and theta0
must then exceed 1 (exit 1). --rate=half uses 1/2 - h_H, the weighted-norm form, for the comparison only.
"""
import json
import os
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
P2B = os.path.join(os.path.dirname(HERE), "review_p2b")
sys.path.insert(0, P2B)

from mpmath import iv, mp                     # noqa: E402

OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
COMPJ = ("S", "V", "B", "C")                 # S, V, beta, c (B_chain.md 14's order)
DELTA = iv.mpf("0.05")                       # B8's delta: L = (1, 1 + delta/2] = (1, 1.025]


def to_frac_up(x):
    """an upper bound of the mpf x as an exact rational"""
    m_, e_ = mp.mpf(x).man_exp
    return Fr(int(m_)) * (Fr(2) ** e_) if e_ >= 0 else Fr(int(m_), 2 ** (-e_))


def certify(G):
    """theta with G v <= theta v for a positive rational v (G: 4 x 4 upper bounds as Fractions)"""
    import numpy as np
    M = np.array([[float(g) for g in row] for row in G])
    ev = max(abs(np.linalg.eigvals(M)))
    lam = ev * 1.02 + 1e-12
    v = np.linalg.solve(np.eye(len(G)) - M / lam, np.ones(len(G)))
    if not all(a > 0 for a in v):
        v = np.ones(len(G))
    vq = [Fr(float(a)).limit_denominator(10 ** 15) for a in v]
    theta = max(sum(G[a][b] * vq[b] for b in range(len(G))) / vq[a] for a in range(len(G)))
    return ev, theta, vq


def assemble(ell, hH, om, lnx1, gain, rate="full", nodecay=None):
    """Gamma0 from the sup coefficients ell[(f, j)] (f in G, F, g), all upper bounds (mpf)"""
    den = (1 - hH) if rate == "full" else (mp.mpf("0.5") - hH)
    kS = (1 + hH) / den
    if nodecay is not None:
        kS = kS * nodecay
    G = {}
    for j in COMPJ:
        G[("S", j)] = kS * ell[("G", j)]
        G[("V", j)] = om * lnx1 if j == "S" else mp.mpf(0)
        G[("C", j)] = lnx1 * ell[("F", j)]
        G[("B", j)] = ell[("g", j)] * gain
    return G


def planted(mutate):
    mp.prec = 106
    u = [mp.mpf("0.3"), mp.mpf("0.02"), mp.mpf("0.001"), mp.mpf("0.4")]
    w = [mp.mpf("0.5"), mp.mpf("0.25"), mp.mpf("2"), mp.mpf("0.125")]
    G = [[to_frac_up(u[a] * w[b]) for b in range(4)] for a in range(4)]
    root = sum(u[a] * w[a] for a in range(4))           # the Perron root of u w^T
    if mutate == "perron":
        G[3][3] = G[3][3] + 2
    ev, theta, _ = certify(G)
    ok = theta >= float(root) * (1 - 1e-12) and theta <= float(root) * 1.03 + 1e-10 and theta < 1
    print(f"planted: Perron root of u w^T {float(root):.6f}, numerical {ev:.6f}, certified theta {float(theta):.6f}: "
          f"{'PASS' if ok else 'FAIL'}" + (f" (mutation {mutate})" if mutate else ""))
    # the assembly on planted coefficients: Gamma0 entries against hand values
    ell = {(f, j): mp.mpf("1e-3") for f in ("G", "F", "g") for j in COMPJ}
    Gm = assemble(ell, mp.mpf("0.1"), mp.mpf("4"), mp.log(mp.mpf("1.025")), mp.mpf("1e-30"))
    hand = {("S", "S"): mp.mpf("1.1") / mp.mpf("0.9") * mp.mpf("1e-3"), ("V", "S"): 4 * mp.log(mp.mpf("1.025")),
            ("V", "V"): 0, ("C", "B"): mp.log(mp.mpf("1.025")) * mp.mpf("1e-3"), ("B", "C"): mp.mpf("1e-33")}
    ok2 = all(abs(Gm[k] - v) <= mp.mpf("1e-25") * (1 + abs(v)) for k, v in hand.items())
    print(f"planted: Gamma0 assembly against hand values: {'PASS' if ok2 else 'FAIL'}")
    return ok and ok2


def main():
    if "planted" in OPTS:
        return 0 if planted(MUT) else 1
    import p2b_alg as Al
    import p2b_lc as L
    import p2b_series as Ser
    from p2b_alg import Cell, up, mpup, ONEI
    st = Al.Setting(OPTS.get("ln_n0", "73.89"))
    hfile = OPTS.get("h", "p2b_lc_cells.json")
    if hfile == "cert":
        CM = {"V": "V", "S": "S", "B": "b", "C": "c"}
        rec = None
        for line in open(os.path.join(os.path.dirname(HERE), "b8m_lc_K4_preP2b.log")):
            r_ = json.loads(line)
            if r_.get("closes") and not r_.get("mutate"):
                rec = r_
        lip = json.load(open(os.path.join(os.path.dirname(HERE), "b8m_lc_lip_preP2b.json")))
        own = json.load(open(os.path.join(P2B, "p2b_lc_cells.json")))
        H = {}
        for c_ in L.COMP:
            for b_ in L.BANDS:
                H[("h", c_, b_)] = iv.mpf(repr(rec["hyp"][f"{CM[c_]}.{b_}"]))
                H[("d", c_, b_)] = iv.mpf(repr(lip["hprime"][f"{CM[c_]}.{b_}"]))
        H[("d", "S", "L")] = iv.mpf(repr(rec["hyp"]["Sp"]))
        H[("cc",)] = iv.mpf(own["h"]["cc,L"])
    else:
        with open(os.path.join(P2B, hfile)) as fh:
            Hj = json.load(fh)["h"]
        H = {k: iv.mpf(Hj[L.idxname(L.key_index(k))]) for k in L.KEYS}
    T = Ser.tables()
    known = L.Known(st, T)
    L.KN = known
    Al.OWNW = True
    L.JSEEDS = {j: ((0, 0, 0), (lambda c: ONEI)) for j in COMPJ}          # m = 1 and every shape 1
    Lc = Cell(st, "L", "L", 0, 0)
    Lc.aux = False
    kn = known.at(Lc)
    r = L.cell_eval(Lc, kn, H, False, st)
    hH = up(kn["n_eps1eta"] / (1 - kn["n_Gam"]))
    coef = lambda node, j, extra=(0, 0, 0): Al.dsup(node, Lc, j, extra)  # noqa: E731
    # 1/(n - 1) <= c1mnu nu, nu = rho X^-2: carried in the weight so that the sup over n >= n0 is taken jointly
    gain_w = (2, 0, -1)
    ell = {}
    for j in COMPJ:
        ell[("G", j)] = coef(r["G"][0], j)
        ell[("F", j)] = coef(r["Fc"][0], j)
        ell[("g", j)] = up(coef(r["ghat"][0], j, gain_w) * st.c1mnu)
        if MUT == "gaininv":
            ell[("g", j)] = up(ell[("g", j)] * (st.n0 - 1) ** 2)        # times (n0 - 1) instead of over (n - 1)
    Q = {j: coef(r["Fc"][0], j, (1, 1, 0)) for j in COMPJ}                 # sup_I ell_{F_c,j}/|s|
    lpG = coef(r["G"][0], L.PCIDX)
    lpg = up(coef(r["ghat"][0], L.PCIDX, gain_w) * st.c1mnu)
    if MUT == "gaininv":
        lpg = up(lpg * (st.n0 - 1) ** 2)
    lpF = coef(r["Fc"][0], L.PCIDX)                                      # F_c's own pseudo-component (P2b: none)
    for j in COMPJ:
        ell[("G", j)] = up(ell[("G", j)] + lpG * Q[j])
        ell[("g", j)] = up(ell[("g", j)] + lpg * Q[j])
    om = known.absW
    lnx1 = up(iv.log(1 + DELTA / 2))
    mp.prec = 106
    E = {k: mpup(v) for k, v in ell.items()}
    G = assemble(E, mpup(hH), mpup(om), mpup(lnx1), mp.mpf(1), OPTS.get("rate", "full"),
                 nodecay=mpup(st.n0) if MUT == "nodecay" else None)
    Gq = [[to_frac_up(G[(i, j)]) for j in COMPJ] for i in COMPJ]
    ev, theta, vq = certify(Gq)
    lines = [f"P2c theta0 (Lemma J's a priori map on (1, 1 + delta/2], m = 1, unit shapes): class constants {hfile}, "
             f"ln n0 = {st.ln_n0}, rate = {OPTS.get('rate', 'full')}, mutate = {MUT}",
             f"  h_H on L = {mp.nstr(mpup(hH), 6)}, |omega| = {mp.nstr(mpup(om), 8)}, ln(1 + delta/2) = "
             f"{mp.nstr(mpup(lnx1), 8)}",
             "  sup_I ell (G with the pseudo-component, F_c, g/(n - 1)):"]
    for f in ("G", "F", "g"):
        lines.append(f"    {f}: " + ", ".join(f"{j} {mp.nstr(E[(f, j)], 5)}" for j in COMPJ))
    lines.append(f"  pseudo-component: ell_G,pc = {mp.nstr(mpup(lpG), 5)}, ell_g,pc/(n-1) = {mp.nstr(mpup(lpg), 5)}, "
                 f"ell_Fc,pc = {mp.nstr(mpup(lpF), 5)}, "
                 "Q_j = " + ", ".join(f"{j} {mp.nstr(mpup(Q[j]), 5)}" for j in COMPJ))
    lines.append("  Gamma0 (rows S, V, beta, c; columns S, V, beta, c):")
    for i in COMPJ:
        lines.append("    " + "  ".join(f"{mp.nstr(G[(i, j)], 5):>11s}" for j in COMPJ))
    lines.append(f"  Perron root {ev:.6e}; certified theta0 = {float(theta):.6e} (weight {[float(a) for a in vq]})")
    ok = theta < 1
    lines.append(f"  theta0 < 1: {'yes' if ok else 'NO'}" + (f" (mutation {MUT})" if MUT else ""))
    print("\n".join(lines))
    if OPTS.get("log"):
        with open(os.path.join(HERE, OPTS["log"]), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    out = dict(hfile=hfile, rate=OPTS.get("rate", "full"), mutate=MUT, hH=float(mpup(hH)), theta0=float(theta),
               perron=float(ev), Gamma0=[[float(G[(i, j)]) for j in COMPJ] for i in COMPJ], order=list(COMPJ),
               ell={f"{f},{j}": float(v) for (f, j), v in E.items()})
    tag = OPTS.get("tag", ("cert" if hfile == "cert" else "own") + f"_{OPTS.get('rate', 'full')}"
                   + (f"_mut_{MUT}" if MUT else ""))
    with open(os.path.join(HERE, f"p2c_theta0_{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
