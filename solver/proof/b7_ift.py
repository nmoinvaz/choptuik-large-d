#!/usr/bin/env python3
"""Lemma B7 for M = 1 with explicit constants (proof/B_chain.md, section 8): the implicit function theorem that
continues Theorem A(M = 1)'s simple zero to the outer selection condition, and the threshold n0.

Notation (real p = (A, w/w_inf) in Theorem A's box X = x^ + [-h, h]^2, h = 1e-9). S~(p) = V^-1 k(p) the inner
multiplier vector in eigen-coordinates (b5_label.py); J_D(p) = psi_{D,+1}(0.3; p) Theorem A's function (lemma6_path.py
runs: J_D(x^) and the interval Jacobian J(X)). The outer selection condition is L(Psi_out) = 0 (L the label map at
u* = 0.0081); by B5(2) (b5_jump.py) V^-1 L(Psi_out) = (I + R~) S~ + E~.

Lemma P (b7_adjoint.py) gives J_D = chi^T psi(0.09) with chi enclosed over a complex box X_C; with lemma L's
representation psi(0.09) = T Psi(u*) V S~ this reads J_D(p) = l~(p) S~(p), l~ a holomorphic row 2-vector with
    l~ = l~0 (I + R~) + l~rest,   l~0 = u* e^{-1/(2u*)} chi_S^T C^-1 V u*^Lambda,   |l~rest| explicit (section 8.2).
Then (real p):
  (i)  if L(Psi_out) = 0 then S~ = -(I + R~)^-1 E~, so F(p) := J_D(p) + Q(p) = 0 with Q := l~ (I + R~)^-1 E~;
  (ii) Krawczyk's test for F on X, with J(X) + DQ(X): K(X) in int X gives exactly one zero p_n of F in X, so at most
       one p in X satisfies the selection condition, and |p_n - p*| <= ||J(X)^-1|| sup |Q|;
  (iii) existence: the homotopy g_t = [V((I + t R~) S~ + t E~)]_{+1} from g_0 = k_{+1} (one simple zero p* in X,
       degree +-1) to g_1 = [L(Psi_out)]_{+1} has no zero on the boundary of X when
       (1 - ||V|| ||V^-1|| eps_L) min_bd |J_D| / (||l~|| ||V^-1||) > ||V|| |E~|; then g_1 has a zero in X, and since
       L(Psi_out) is antisymmetric for real p, L(Psi_out) = 0 there.
Derivatives: DQ by the product rule, with |D l~| <= sup_{X_C} |l~| / r_P (Cauchy, r_P the polydisc radius of X_C
around X) and |D R~| <= eps_L / 1e-6, |D E~| <= C_E / (n 1e-6) (Cauchy on B5(1)'s box, b5_jump.py).

    python3 b7_ift.py [--adj=b7_adjoint.json] [--jump=b5_jump_d0_1e6.json] [--json=b7_ift.json]
    python3 b7_ift.py --mutate=JD    Theorem A's centre value J_D(x^) shifted by 5e-11 (a wrong input: the zero of J_D
                                      then leaves X): no n may pass
"""
import json
import math
import os
import sys

import numpy as np
from flint import arb, acb, acb_mat, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b5_bounds as BB  # noqa: E402
import b5_label as LB  # noqa: E402

OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
ODD, EVEN = (-1, 1), (-2, 0, 2)
up = BB.up


def p_acb(p):
    return acb(arb(p[0]), arb(p[1]))


def rowabs1(v):
    return sum((up(x) for x in v), arb(0))


def ell_bound(adj, lab_rep, ustar):
    """sup over the complex box of ||l~||_1 (row acting on the inf-norm of S~), and l~0 at the box"""
    ctx.prec = 192
    rep, dp, V, Vi, lam, o = lab_rep
    chi = [p_acb(p) for p in adj["chi"]]
    chiD, chiS, chiC = chi[0:2], chi[2:4], chi[4:7]
    u = arb(ustar)
    xi = u.sqrt()
    z = acb(u)
    q = BB.cell_inner(rep, dp, z, up(z), arb(0))       # inner solution at u* over K (contains Y+ and Y-)
    C = q["C"]
    Cm = acb_mat([[C.get(a - b, acb(0)) for b in ODD] for a in ODD])
    Ci = Cm.inv()
    Dg = acb_mat([[(1 + acb(0, 1) * a * rep["w"]) if a == b else acb(0) for b in ODD] for a in ODD])
    upl = acb_mat([[(lam[j] * u.log()).exp() if i == j else acb(0) for j in range(2)] for i in range(2)])
    ef = (-1 / (2 * u)).exp()
    rowS = acb_mat([[chiS[0], chiS[1]]])
    rowD = acb_mat([[chiD[0], chiD[1]]])
    l0 = rowS * Ci * V * upl * acb(u * ef)                # 1 x 2
    # remainder (lemma L bounds, A <= |S~| e^I):
    lab = json.load(open(os.path.join(HERE, "b5_label.json")))
    I = arb(lab["int_g"])
    mu_lo, mu_hi = arb(lab["mu_lo"]), arb(lab["mu_hi"])
    kP, kC = arb(lab["coef_sup"]["kP"]), arb(lab["coef_sup"]["kC"])
    eI = I.exp()
    pt = ef * u ** (mu_lo + 1) * eI                       # max_j |p~_j| per unit |S~|
    psC = 2 * (kC * u).exp() * kP * u ** (mu_hi + 3) * ef * eI     # |psi_C|_inf per unit |S~|
    t1 = xi * rowabs1([(rowD * V)[0, j] for j in range(2)]) * pt
    t2 = u * rowabs1([(rowS * Ci * Dg * V)[0, j] for j in range(2)]) * pt
    sQ = sum((up(q["Q"][m]) for m in ODD), arb(0))
    t3 = u * rowabs1([(rowS * Ci)[0, j] for j in range(2)]) * sQ * psC
    t4 = rowabs1(chiC) * psC
    rest = t1 + t2 + t3 + t4
    epsL = arb(lab["epsL"])
    lnorm = rowabs1([l0[0, j] for j in range(2)]) * (1 + epsL) + rest
    return dict(l0=l0, lnorm=lnorm, rest=rest, l0norm=rowabs1([l0[0, j] for j in range(2)]))


def theoremA_data():
    pt = json.load(open(os.path.join(HERE, "lemma6_path_point.json")))
    bx = json.load(open(os.path.join(HERE, "lemma6_path_box_1e-9.json")))
    Dm1, Dp1 = p_acb(pt["Dm1"]), p_acb(pt["Dp1"])
    dDm1 = [p_acb(p) for p in bx["dDm1"]]
    dDp1 = [p_acb(p) for p in bx["dDp1"]]
    JD = Dp1 - Dm1.conjugate()
    dJ = [dDp1[k] - dDm1[k].conjugate() for k in range(2)]
    JX = [[dJ[0].real, dJ[1].real], [dJ[0].imag, dJ[1].imag]]
    return JD, JX, arb(bx["h"])


def krawczyk(xhat, Fhat, JF, h):
    Jm = np.array([[float(JF[i][j].mid()) for j in range(2)] for i in range(2)])
    Yf = np.linalg.inv(Jm)
    Y = [[arb(float(Yf[i][j])) for j in range(2)] for i in range(2)]
    dX = [arb(0, h.upper()), arb(0, h.upper())]
    K = []
    for i in range(2):
        t = xhat[i] - (Y[i][0] * Fhat[0] + Y[i][1] * Fhat[1])
        for j in range(2):
            Mij = (arb(1) if i == j else arb(0)) - (Y[i][0] * JF[0][j] + Y[i][1] * JF[1][j])
            t = t + Mij * dX[j]
        K.append(t)
    ok = all(abs(K[i] - xhat[i]) < h for i in range(2))
    margins = [float((h - abs(K[i] - xhat[i]).upper()).lower()) for i in range(2)]
    return bool(ok), margins


def inv_norm(JX):
    """an upper bound of ||M^-1||_inf (inf-norm on (A, wf)) for every M in the interval matrix JX"""
    a, b, c, d = JX[0][0], JX[0][1], JX[1][0], JX[1][1]
    det = a * d - b * c
    if det.contains(0):
        return None
    inv = [[d / det, -b / det], [-c / det, a / det]]
    return max(up(inv[i][0]) + up(inv[i][1]) for i in range(2))


def min_bd_JD(JD, JX, h, nsub=40):
    """lower bound of |J_D| on the boundary of X from J_D(p) in J_D(x^) + J(X)(p - x^), faces subdivided"""
    best = None
    for side in (0, 1):
        for sgn in (-1, 1):
            for k in range(nsub):
                t = -h + 2 * h * arb(k) / nsub
                seg = arb((t + h / nsub).mid(), (h / nsub).upper())
                dp = [None, None]
                dp[side] = sgn * h
                dp[1 - side] = seg
                re = JD.real + JX[0][0] * dp[0] + JX[0][1] * dp[1]
                im = JD.imag + JX[1][0] * dp[0] + JX[1][1] * dp[1]
                # |J_D| >= sqrt(lower|re|^2 + lower|im|^2) on the segment (lower|x| = 0 if the ball contains 0)
                val = arb((arb(abs(re).lower()) ** 2 + arb(abs(im).lower()) ** 2).sqrt().lower())
                best = val if best is None or val < best else best
    return best


def main():
    ctx.prec = 192
    adj_path = OPTS.get("adj", "b7_adjoint.json")
    adj = json.load(open(os.path.join(HERE, adj_path)))
    box = json.load(open(os.path.join(HERE, adj["box"])))
    jump = json.load(open(os.path.join(HERE, OPTS.get("jump", "b5_jump_d0_1e6.json"))))
    lab = json.load(open(os.path.join(HERE, "b5_label.json")))
    out, rp = LB.run(ZL=lab["ZL"], Zf=lab["Zf"], ncell=lab["ncell"], verbose=False)
    if abs(out["epsL"] - lab["epsL"]) > 1e-12:
        print("lemma L rerun differs from b5_label.json")
        return 1
    V, Vi = rp["V"], rp["Vi"]
    eb = ell_bound(adj, (rp["rep"], rp["dp"], V, Vi, rp["lam"], rp["o"]), jump["ustar"])
    JD, JX, h = theoremA_data()
    if MUT == "JD":
        JD = JD + acb(arb("5e-11"), arb("5e-11"))
    xhat = [arb("2.5421480475"), arb("0.7769940136")]
    hr, hi = arb(box["hr"]), arb(box["hi"])
    rP = arb(min(float((hr - h).lower()), float(hi.lower())))
    if not rP > 0:
        print("the complex box does not contain a polydisc around X")
        return 1
    epsL = arb(jump["epsL"])
    Vn, Vin = arb(lab["V_norm"]), arb(lab["Vinv_norm"])
    CE = arb(jump["CE"])
    rC = arb(jump["cauchy"])
    L = eb["lnorm"]
    dL = L / rP
    inm = inv_norm(JX)
    mbd = min_bd_JD(JD, JX, h)
    Rharm = Vn * Vin * epsL
    print(f"Lemma P: adjoint run {adj['steps']} steps over X_C = x^ + [+-{box['hr']}]^2 + i[+-{box['hi']}]^2; "
          f"sup ||l~||_1 <= {float(L.upper()):.4e} (l~0 part {float(eb['l0norm'].upper()):.4e}, rest "
          f"{float(eb['rest'].upper()):.2e}); |D l~| <= {float(dL.upper()):.4e} (Cauchy radius {float(rP):.1e})")
    print(f"Theorem A data: J_D(x^) = {JD.real.mid().str(6)} + {JD.imag.mid().str(6)} i, ||J(X)^-1|| <= "
          f"{float(inm):.4g}, min |J_D| on the boundary of X >= {float(mbd):.4e}")
    print(f"B5(2): C_E = e^{jump['lnCE']:.4f}, eps_L = {jump['epsL']:.4f}; harmonic ||R|| <= {float(Rharm.upper()):.4f}")
    first = None
    rows = []
    for L10 in range(7400, 11001, 25):
        lnn = arb(L10) / 100
        En = CE * (-lnn).exp()
        dEn = En / rC
        Qb = L * En / (1 - epsL)
        dQ = dL * En / (1 - epsL) + L * (epsL / rC * En / (1 - epsL) ** 2 + dEn / (1 - epsL))
        Fhat = [JD.real + arb(0, Qb.upper()), JD.imag + arb(0, Qb.upper())]
        JF = [[JX[i][j] + arb(0, dQ.upper()) for j in range(2)] for i in range(2)]
        okK, margins = krawczyk(xhat, Fhat, JF, h)
        # existence (degree): (1 - ||R||) min|J_D| / (||l~|| ||V^-1||) > ||V|| |E~|
        lhs = (1 - Rharm) * mbd / (L * Vin)
        okD = bool(lhs > Vn * En)
        loc = inm * Qb if inm is not None else None
        rows.append((float(lnn), okK, okD, margins, float(Qb.upper()), float(loc.upper()) if loc is not None else None))
        if okK and okD and first is None:
            first = float(lnn)
    for r in rows[::4]:
        print(f"   ln n = {r[0]:6.2f}: Krawczyk {r[1]} (margins {r[3][0]:.2e}, {r[3][1]:.2e}), degree {r[2]}, "
              f"|Q| <= {r[4]:.2e}, |p_n - p*| <= {r[5]:.2e}")
    if first is None:
        print("no n in the scan passes" + (f"   [MUTATION {MUT}]" if MUT else ""))
        if MUT:
            print(f"MUTATION {MUT}: the test must FAIL: caught")
            return 0
        return 1
    # n enters only through E_n = C_E/n, which decreases; the Krawczyk test is inclusion-monotone in the widths
    # (|Q|, |DQ| are increasing in E_n) and the degree condition is monotone in E_n, so passing at n0 gives every n >= n0.
    # Sanity check of that monotonicity on the scan:
    if not all(r[1] and r[2] for r in rows if r[0] >= first):
        print("a row beyond the first passing one fails: the monotonicity argument is contradicted")
        return 1
    C7 = float((inm * L / (1 - epsL) * CE).upper())
    print(f"=> for every n >= n0 = e^{first:.2f} (D = n + 2): exactly one p_n in X satisfies the outer selection "
          f"condition L(Psi_out) = 0, and |p_n - p*|_inf <= C7/n with C7 = {C7:.4e} = e^{math.log(C7):.3f}"
          + (f"   [MUTATION {MUT}]" if MUT else ""))
    res = dict(ln_n0=first, C7=C7, lnC7=math.log(C7), lnorm=float(L.upper()), dl=float(dL.upper()),
               invJ=float(inm), minbd=float(mbd), rows=rows, adj=adj_path, jump=OPTS.get("jump", "b5_jump_d0_1e6.json"))
    if "json" in OPTS:
        with open(os.path.join(HERE, OPTS["json"]), "w") as fh:
            json.dump(res, fh, indent=1)
    if MUT:
        print(f"MUTATION {MUT}: the test must FAIL: NOT caught")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
