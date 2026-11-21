#!/usr/bin/env python3
"""P2 item 4 (#1061): Lemma B8(1)'s bootstrap P(h) < h, a second implementation (notes/P2_remaining.md section 7).

Written from B8_constants.md sections 3-7 (Lemma W, Lemma R's identities, Lemma K, Lemma B8.a's map P, the offset of
8.4), not from b8_alg.py, b8m_bounds.py or b8m_scan.py (not opened before the results were committed). The series and
residuals are p2c_b8_series.py's; the arithmetic is p2c_b8_alg.py's (python-flint 0.9 balls, 256 bits).

What differs from the certificate's design, by choice (both are valid readings of the statements):
- the formulas are written once and run in two arithmetics: exact values at points (Lemma R's identities, planted
  truth) and cell majorants (the bounds);
- the bounds are affine in h (products of two remainders folded with h <= Hmax), so P is computed once per n0 and
  iterated exactly;
- N0, N1 and the ray R are covered by radial cells, each with its own kernel gain (Lemma K.1-K.2 evaluated on the cell);
- three algebraic rearrangements, each checked by the identities: Lambda G1 = (D^K)^-1 N1,
  q (Lambda G1 + x W2) = l^ (I + e)^-1 [N1 + (I - dD D^-1) N2] / (n - 1), and the (n - 1)(C^2 - C^K2) S part of N1
  combined with its dD D^-1 partner into -(n - 1) x^2 dD D^-1 S;
- closure is tested at h = h* + t u with (I - J) u = h*, J the Jacobian of P at its fixed point h*.

    python3 p2c_b8_boot.py --check=identities [--mutate=drop_rho2|drop_W2|sign]
    python3 p2c_b8_boot.py --check=instances --L0=73.89 [--mutate=prodrule]
    python3 p2c_b8_boot.py --check=gains [--mutate=kappa]
    python3 p2c_b8_boot.py --check=kernel [--mutate=kappa]
    python3 p2c_b8_boot.py --check=closure_planted
    python3 p2c_b8_boot.py --L0=62 [--d0=1e7] [--domain=D|Dprime] [--kappa=0.3] [--mutate=tauD10] [--json=out.json]
    python3 p2c_b8_boot.py --bisect=58,74 [--d0=1e7] [--domain=...]
"""
import json
import math
import random
import sys
import time

from flint import acb, arb, ctx

import p2c_b8_alg as ALG
import p2c_b8_series as SER
from p2c_b8_alg import BoundError, Cell, CellArith, PointArith, apow
from p2c_b8_series import K, add, sub, scale, hp, matvec, matmat, M_e, eye, norm

OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
KAPPA = arb(OPTS.get("kappa", "0.3"))
KAPPA1 = arb("0.6")
DELTA = arb("0.05")
XF = arb(20)
R1 = arb("0.05")
ZB = arb("0.0085")
HALF = arb("0.5")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- formulas (both arithmetics)

def cst(A, c):
    if A.point:
        return [acb(c), acb(0), acb(0)]
    return [ALG.const(A.cl, arb(c)), ALG.Maj(A.cl), ALG.Maj(A.cl)]


def kquant(A):
    """Quantities built from Z^K only (Lemma R, Lemma K), as jets to second order."""
    Q = {}
    C = A.K("C")
    dCi = A.sinv(A.K("detCm1"))                      # 1/det C^K
    Q["C"] = C
    Q["CKinv"] = A.sc(dCi, A.K("adjC"))
    Gam = A.K("Gam")
    Q["Gam"] = Gam
    Q["IGinv"] = A.ninv(A.neg(Gam))                   # (I - Gamma)^-1
    etat = A.sc(dCi, A.K("etan"))                     # eta~ = eta / x^2
    Q["eta"] = A.sc(A.x(2), etat)
    Ht = A.sc(A.inv_nm1(), A.mm(Q["IGinv"], etat))    # H~ = (I - Gamma)^-1 eta~ / (n - 1)
    Q["Hp"] = A.sc(A.x(2), Ht)                        # H' of Lemma K
    IHt, IHp = A.ninv(Ht), A.ninv(Q["Hp"])
    Q["AKinv"] = A.sc(A.inv_nm1(), A.mm(IHp, Q["IGinv"]))        # A^K = (n - 1)(I - Gamma)(I + H')
    Q["DKinv"] = A.sc(A.inv_x2m1(), Q["IGinv"])                   # D^K = (x^2 - 1)(I - Gamma)
    Q["Lam"] = A.sc(A.inv_x2m1(), A.add(A.sc(A.nm1(), A.I()), A.mm(Q["IGinv"], Q["eta"])))
    lhat = A.sc(A.mul(dCi, dCi), A.K("lhatn"))       # l / x^2
    Q["q"] = A.sc(A.mul(A.inv_nm1(), A.x2m1()), A.mm(lhat, IHt))  # (I + e)^-1 D^K = (x^2 - 1)(I + H~)^-1
    Q["lIe"] = A.mm(lhat, A.mm(IHt, Q["IGinv"]))      # l^ (I + e)^-1
    Q["rho2"] = A.sc(dCi, A.K("rho2n"))
    Q["rho3"] = A.sc(A.mul(dCi, dCi), A.K("rho3n"))
    Q["Dt"] = A.sc(dCi, A.K("Dtn"))                   # Delta~ = x C^K-1 Y^K
    Q["BK"] = A.add(A.I(), A.mm(A.sub(A.mm(C, A.Om()), A.K("Bfr")), Q["CKinv"]))
    Q["Hgain"] = A.mm(A.inv_x2m1(), A.mm(Q["eta"], Q["IGinv"])) if not A.point else None
    return Q


def rquant(A, Q, R):
    """The remainder terms of Lemma R and the derivative bounds of pass 1, from remainders R (jets)."""
    RV, RS, Rhb, Rc = R["RV"], R["RS"], R["Rhb"], R["Rc"]
    SK, VK, YK, bK, cK = A.K("S"), A.K("V"), A.K("Y"), A.K("beta"), A.K("c")
    Om, x1, x2, half = A.Om(), A.x(1), A.x(2), cst(A, HALF if not A.point else acb("0.5"))
    RC = A.Me(Rc)
    C = A.add(Q["C"], RC)
    Cinv = A.mm(A.ninv(A.mm(Q["CKinv"], RC)), Q["CKinv"])        # (C^K + R_C)^-1
    S, V, Y = A.add(SK, RS), A.add(VK, RV), A.add(YK, A.add(RV, RS))
    qRS = A.mv(Q["q"], RS)
    Rb = A.add(Rhb, qRS)                                           # R_beta = R^_beta + q R_S
    c = A.add(cK, Rc)
    dcb = A.add(A.hp(Rc, bK), A.hp(c, qRS))                        # [c beta_0] - [c^K beta^K]
    dB0 = A.Me(dcb)
    B0fr = A.add(A.K("Bfr"), dB0)
    Mh = A.Me(A.hp(c, Rhb))
    dD = A.neg(A.add(A.add(A.mm(Q["C"], RC), A.mm(RC, Q["C"])), A.mm(RC, RC)))   # D - D^K
    Dinv = A.mm(A.ninv(A.mm(Q["DKinv"], dD)), Q["DKinv"])
    dCinv = A.neg(A.mm(A.mm(Cinv, RC), Q["CKinv"]))               # C^-1 - C^K-1
    dBB = A.add(A.mm(A.sub(A.mm(C, Om), B0fr), dCinv), A.mm(A.sub(A.mm(RC, Om), dB0), Q["CKinv"]))  # B_0 - B^K
    B0 = A.add(Q["BK"], dBB)
    dAt = A.sub(A.add(A.mm(RC, B0fr), A.mm(Q["C"], dB0)), A.sc(x2, dBB))       # A~_0 - A~^K
    At0 = A.sub(A.sub(A.mm(C, B0fr), A.sc(x2, Om)), A.sc(x2, B0))               # A~_0 = A_0 - (n - 1) C^2
    dDDinv = A.mm(dD, Dinv)
    N1 = A.neg(A.sc(A.mul(A.nm1(), x2), A.mv(dDDinv, S)))
    N1 = A.add(N1, A.mv(dAt, S))
    N1 = A.sub(N1, A.sc(x2, A.add(A.mv(dBB, V), A.mv(Q["BK"], RV))))
    N1 = A.sub(N1, A.mv(dDDinv, A.sub(A.mv(At0, S), A.sc(x2, A.mv(B0, V)))))
    if not (MUT == "bound_rho2" and not A.point):            # cell-only drop: the planted check must fire
        N1 = A.sub(N1, Q["rho2"])
    G1 = A.mv(Q["AKinv"], N1)
    N2 = A.add(A.mv(A.mm(C, Mh), S), A.sc(x2, A.mv(A.mm(Mh, Cinv), Y)))
    W2x = A.mv(Dinv, N2)                                            # x W2
    if MUT == "drop_W2" or (MUT == "bound_W2" and not A.point):
        W2x = A.sc(cst(A, 0), W2x)
    dRS = A.add(A.add(A.mv(Q["Lam"], RS), A.mv(Q["DKinv"], N1)), W2x)
    # g^
    Dl = A.sc(x1, A.mv(Cinv, Y))
    H = A.sc(half, A.add(A.hp(S, S), A.hp(Dl, Dl)))
    Dt = Q["Dt"]
    CiRC = A.mm(Cinv, RC)
    dDl = A.sub(A.sc(x1, A.mv(Cinv, A.add(RV, RS))), A.mv(CiRC, Dt))
    inner = A.sub(A.sc(x1, A.mv(Cinv, RV)), A.mv(CiRC, A.add(A.sc(x1, A.mv(Q["CKinv"], RS)), Dt)))
    g3 = A.add(A.add(A.sc(half, A.hp(RS, RS)), A.hp(Dt, inner)), A.sc(half, A.hp(dDl, dDl)))
    QLG = A.sc(A.inv_nm1(), A.mv(Q["lIe"], A.sub(A.add(N1, N2), A.mv(dDDinv, N2))))   # q (Lambda G1 + x W2)
    two = cst(A, 2)
    terms = {"RbRb": A.neg(A.hp(Rb, A.add(A.sc(two, bK), Rb))), "RbH": A.sc(A.eps(), A.hp(Rb, H)),
             "g3": A.hp(A.K("bK1"), g3), "dqRS": A.neg(A.mv(A.dshift(Q["q"]), RS)), "QLG": A.neg(QLG),
             "rho3": A.neg(Q["rho3"])}
    gh = None
    for v in terms.values():
        gh = v if gh is None else A.add(gh, v)
    beta = A.add(bK, Rb)
    dRc = A.sub(A.add(A.hp(Rc, beta), A.hp(cK, Rb)), A.K("rho4"))
    return {"G1": G1, "W2x": W2x, "gh": gh, "dRS": dRS, "dRhb": A.sub(gh, A.sc(A.nm1(), Rhb)), "dRc": dRc,
            "dRV": A.mv(Om, RS), "N1": N1, "N2": N2, "terms": terms, "Dinv": Dinv, "Cinv": Cinv, "Rb": Rb, "qRS": qRS}


# ---------------------------------------------------------------- point checks: Lemma R's identities

def omega_mat(par):
    return par["Omega"]


def check_identities(at, ntrial=6, seed=1):
    random.seed(seed)
    par = at["par"]
    Om = par["Omega"]
    worst = {}

    def rnd(kind, scale_=1.0):
        m = 2 if kind == "o" else 3
        return [acb(random.uniform(-1, 1), random.uniform(-1, 1)) * scale_ for _ in range(m)]

    ok = True
    for t in range(ntrial):
        r = random.uniform(0.1, 30)
        ph = random.uniform(0, 0.93)
        x = acb(r * math.cos(ph), r * math.sin(ph))
        n = acb(random.choice([50, 300, 2000, 1e5]) * random.uniform(1, 2), random.uniform(-3, 3))
        eps = 1 / n
        A = PointArith(at, x, n, Om)
        Q = kquant(A)
        sz = 1e-3
        Rv = {"RV": rnd("o", sz), "RS": rnd("o", sz), "Rhb": rnd("e", sz), "Rc": rnd("e", sz)}
        dR = {k: rnd("o" if k in ("RV", "RS") else "e", sz * 7) for k in Rv}
        R1 = {k: [v] for k, v in Rv.items()}
        P = rquant(A, Q, R1)
        ev = lambda nm, p=0: SER.peval(at[nm], eps, x, p)
        x2 = x * x
        # direct K operators
        CK = M_e(ev("c"))
        CKi = inv2(CK)
        BfK = M_e(hp(ev("c"), ev("beta")))
        BK = add(eye(2), matmat(sub(matmat(CK, Om), BfK), CKi))
        DK = sub(scale(eye(2), x2), matmat(CK, CK))
        AK = sub(sub(add(scale(matmat(CK, CK), n - 1), matmat(CK, BfK)), scale(Om, x2)), scale(BK, x2))
        res = {}
        # factorizations
        Lam_d = matmat(inv2(DK), AK)
        res["Lambda"] = rel(sub(Q["Lam"][0], Lam_d), Lam_d)
        res["AKinv"] = rel(sub(Q["AKinv"][0], inv2(AK)), inv2(AK))
        bK1 = ev("bK1")
        Dt = scale(matvec(CKi, ev("Y")), x)
        CKif = lambda f: matvec(CKi, f)
        lop = [[None] * 2 for _ in range(3)]
        for j in range(2):
            f = [acb(1) if i == j else acb(0) for i in range(2)]
            col = hp(bK1, add(hp(ev("S"), f), hp(Dt, scale(CKif(f), x))))     # (1 - eps)[b^K(...)]
            for i in range(3):
                lop[i][j] = col[i]
        q_d = matmat(lop, inv2(add(scale(eye(2), n - 1), Lam_d)))
        res["q"] = rel(sub(Q["q"][0], q_d), q_d)
        # I1
        lhs1 = add(sub(matvec(DK, ev("S", 1)), matvec(AK, ev("S"))), scale(matvec(BK, ev("V")), x2))
        res["I1"] = rel(sub(lhs1, Q["rho2"][0]), lhs1)
        # the full solution candidate
        qv = Q["q"][0]
        dq = Q["q"][1]
        Rb = add(Rv["Rhb"], matvec(qv, Rv["RS"]))
        dRb = add(add(dR["Rhb"], matvec(dq, Rv["RS"])), matvec(qv, dR["RS"]))
        S = add(ev("S"), Rv["RS"])
        dS = add(ev("S", 1), dR["RS"])
        V = add(ev("V"), Rv["RV"])
        bet = add(ev("beta"), Rb)
        dbet = add(ev("beta", 1), dRb)
        cc = add(ev("c"), Rv["Rc"])
        dcc = add(ev("Dc"), dR["Rc"])
        Cf = M_e(cc)
        Cfi = inv2(Cf)
        Y = add(V, S)
        Dl = scale(matvec(Cfi, Y), x)
        H = scale(add(hp(S, S), hp(Dl, Dl)), acb("0.5"))
        Bf = M_e(hp(cc, bet))
        Bm = add(eye(2), matmat(sub(matmat(Cf, Om), Bf), Cfi))
        Dm = sub(scale(eye(2), x2), matmat(Cf, Cf))
        Am = sub(sub(add(scale(matmat(Cf, Cf), n - 1), matmat(Cf, Bf)), scale(Om, x2)), scale(Bm, x2))
        defS = sub(matvec(Dm, dS), sub(matvec(Am, S), scale(matvec(Bm, V), x2)))
        defb = sub(add(add(dbet, scale(bet, n - 1)), hp(bet, bet)), add(scale(H, 1 - eps), scale(hp(bet, H), eps)))
        defc = sub(dcc, hp(cc, bet))
        Dmi = inv2(Dm)
        # I2: x R_S' = Lambda (R_S + G1) + x W2 + D^-1 defect_S
        rhs2 = add(add(matvec(Lam_d, add(Rv["RS"], P["G1"][0])), P["W2x"][0]), matvec(Dmi, defS))
        res["I2"] = rel(sub(dR["RS"], rhs2), dR["RS"])
        # I3: x R^_beta' + (n - 1) R^_beta = g^ + defect_beta - q D^-1 defect_S
        lhs3 = add(dR["Rhb"], scale(Rv["Rhb"], n - 1))
        rhs3 = sub(add(P["gh"][0], defb), matvec(qv, matvec(Dmi, defS)))
        res["I3"] = rel(sub(lhs3, rhs3), lhs3)
        # I4: x R_c' = [R_c beta] + [c^K R_beta] - rho_4 + defect_c
        rhs4 = add(sub(add(hp(Rv["Rc"], bet), hp(ev("c"), Rb)), ev("rho4")), defc)
        res["I4"] = rel(sub(dR["Rc"], rhs4), dR["Rc"])
        # x V^K' = Omega S^K exactly
        res["V"] = rel(sub(ev("V", 1), matvec(Om, ev("S"))), ev("V", 1))
        for k, v in res.items():
            worst[k] = max(worst.get(k, 0.0), v)
    lim = 1e-40
    rep = []
    for k, v in worst.items():
        good = v < lim
        ok = ok and good
        rep.append(f"{k}: {v:.2e} {'ok' if good else 'FAIL'}")
    return ok, rep


def inv2(M):
    d = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    return [[M[1][1] / d, -M[0][1] / d], [-M[1][0] / d, M[0][0] / d]]


def rel(diff, ref):
    a, b = norm(diff), norm(ref)
    return float((a.upper() / max(b.lower(), arb("1e-300"))).upper().mid())


# ---------------------------------------------------------------- settings for one n0

class Setting:
    def __init__(self, L0, d0=0, domain="D", kappa=KAPPA, Hmax=None):
        self.L0 = arb(L0)
        self.n0 = self.L0.exp()
        self.eps0 = 1 / self.n0
        self.kappa = kappa
        self.alpha = arb.pi() / 4 + kappa / 2
        self.s_alpha = self.alpha.sin() / self.alpha
        self.phid = (DELTA / 2).asin()
        if domain == "D":
            Rp = (KAPPA1 * self.n0 / (kappa.cos() * self.L0)).sqrt()
        else:
            Rp = (ZB * self.n0).sqrt()
        self.Rp = Rp
        self.rho_max = (1 + Rp) * (1 + Rp) / self.n0
        self.d0 = arb(d0)
        self.domain = domain
        self.Hmax = Hmax


def make_cells(st, nz=1, n0c=24, n1c=160, ra="1e-6"):
    """Radial cells: Z [0, ra], N0 [ra, 0.05] and N1 [0.05, 20] (log-spaced, with breaks at 1 -+ delta/2 and 1)."""
    ra = arb(ra)
    def logspace(a, b, m):
        la, lb = math.log(float(a.mid())), math.log(float(b.mid()))
        return [arb(repr(math.exp(la + (lb - la) * i / m))) for i in range(1, m)]
    edgesN0 = [ra] + logspace(ra, R1, n0c) + [R1]
    br = [arb("0.975"), arb("0.99"), arb(1), arb("1.01"), arb("1.025")]
    mid = logspace(R1, XF, n1c)
    pts = sorted(set([float(p.mid()) for p in mid] + [float(b.mid()) for b in br]))
    edgesN1 = [R1] + [arb(repr(p)) for p in pts if 0.05 < p < 20] + [XF]
    out = []
    for kind in ("sector", "ray"):
        out.append(("N0", Cell("Z", st.eps0, st.Hmax, st.kappa, DELTA, arb(0), ra, kind=kind)))
        for a, b in zip(edgesN0[:-1], edgesN0[1:]):
            out.append(("N0", Cell("B", st.eps0, st.Hmax, st.kappa, DELTA, a, b, kind=kind)))
        for a, b in zip(edgesN1[:-1], edgesN1[1:]):
            out.append(("N1", Cell("B", st.eps0, st.Hmax, st.kappa, DELTA, a, b, kind=kind)))
    out.append(("F", Cell("F", st.eps0, st.Hmax, st.kappa, DELTA, xF=XF, rho_max=st.rho_max)))
    return out


def hyp(A):
    return {"RV": A.rem("RV", 1, True), "RS": A.rem("RS", 2), "Rhb": A.rem("Rhb", 3), "Rc": A.rem("Rc", 4, True)}


def cell_forms(cell, at, om_abs):
    A = CellArith(cell, at, om_abs)
    Q = kquant(A)
    R1 = hyp(A)
    P1 = rquant(A, Q, R1)
    R2 = {"RV": R1["RV"] + [P1["dRV"][0]], "RS": R1["RS"] + [P1["dRS"][0]],
          "Rhb": R1["Rhb"] + [P1["dRhb"][0]], "Rc": R1["Rc"] + [P1["dRc"][0]]}
    if MUT == "prodrule":
        ALG_mul = A.mul
        def mul_drop(a, b):
            out = ALG_mul(a, b)
            if len(out) > 1:
                out[1] = a[0] * b[1]
            return out
        A.mul = A.mv = A.mm = A.hp = A.sc = mul_drop
    P2 = rquant(A, Q, R2)
    f = {
        "locS": P1["G1"][0].ratio("w"),
        "IS": (P2["G1"][1] + P2["W2x"][0]).ratio("w"),
        "locB": P1["gh"][0].ratio("wn"),
        "IB": P2["gh"][1].ratio("wn"),
        "Ic": P1["dRc"][0].ratio("w"),
        "Hgain": Q["Hgain"][0].sup(),
        "Hp": Q["Hp"][0].sup(),
        "Gam": Q["Gam"][0].sup(),
    }
    return f, (A, Q, P1, P2)


def gains(st, cells, forms):
    """Kernel gains per target cell (Lemma K.1-K.3) and the decay margin s_*."""
    sk = st.kappa.sin()
    hp_sup = arb(0)
    for (reg, cl), f in zip(cells, forms):
        if cl.kind == "ray" or cl.mode == "F":
            hp_sup = max((hp_sup, f["Hp"]), key=lambda z: z.upper())
    s_star = sk - hp_sup
    if MUT == "kappa":
        s_star = arb("0.4").sin() - hp_sup
    if s_star.lower() <= 0:
        raise BoundError(f"decay margin s_* = {s_star.str(4)} not positive")
    sa2 = st.s_alpha * st.s_alpha
    al = st.alpha
    n0 = st.n0
    g = []
    for (reg, cl), f in zip(cells, forms):
        H = f["Hgain"]
        if cl.mode == "F":
            qF = 1 + 2 * sk / (XF * XF) + 1 / apow(XF, 4)
            NF = (1 - 1 / n0) / (st.rho_max * qF)
            KF = (H * H / (4 * NF * sa2)).exp()
            E = min((KF, (H * al).exp()), key=lambda z: z.upper())
            ray = E * qF.sqrt() * st.rho_max / s_star * n0 / (n0 - 1)
            arc = min((KF * (arb.pi() / (NF * sa2)).sqrt(), al * (H * al).exp()), key=lambda z: z.upper())
            g.append({"gamma": ray + arc, "KF": KF, "NF": NF, "H": H, "s_star": s_star})
            continue
        if cl.kind == "ray":
            g.append(None)
            continue
        fr = lambda r: r * r / (1 + 2 * r * r * sk + apow(r, 4))
        Nlo = (n0 - 1) * min((fr(cl.r_lo), fr(cl.r_hi)), key=lambda z: z.lower())
        eHa = (H * al).exp()
        if Nlo.lower() > 0:
            Kc = (H * H / (4 * Nlo * sa2)).exp()
            E = min((Kc, eHa), key=lambda z: z.upper())
            arc = min((Kc * (arb.pi() / (Nlo * sa2)).sqrt(), al * eHa), key=lambda z: z.upper())
        else:
            E, arc = eHa, al * eHa
        Dr = (1 + 2 * cl.r_hi * cl.r_hi * sk + apow(cl.r_hi, 4)).sqrt()
        ray = E * Dr / ((n0 - 1) * s_star)
        g.append({"gamma": ray + arc, "H": H})
    return g, s_star


# ---------------------------------------------------------------- the map P and its closure

def aff(form, h):
    return form[0] + form[1] * h[0] + form[2] * h[1] + form[3] * h[2] + form[4] * h[3]


class Pmap:
    def __init__(self, st, cells, forms, g, s_star, absF):
        self.st, self.cells, self.forms, self.g, self.s_star = st, cells, forms, g, s_star
        self.slow = 1 / arb(2 * K + 2) + st.phid
        self.om = None
        self.absF = absF
        self.beta_fac = 1 / (st.n0 - 1) + st.phid
        # index structure
        self.sector = [i for i, (r, c) in enumerate(cells) if c.kind == "sector" and c.mode != "F"]
        self.ray = [i for i, (r, c) in enumerate(cells) if c.kind == "ray"]
        self.F = [i for i, (r, c) in enumerate(cells) if c.mode == "F"][0]
        self.sector.sort(key=lambda i: float(cells[i][1].r_lo.mid()))
        self.ray.sort(key=lambda i: float(cells[i][1].r_lo.mid()))
        self.fl = {k: [[float(v.upper().mid()) for v in f[k]] for f in forms] for k in ("locS", "IS", "locB", "IB", "Ic")}
        self.gfl = [None if gg is None else float(gg["gamma"].upper().mid()) for gg in g]

    def __call__(self, h, exact=False, detail=False):
        """P(h); floats (upper ends, for the iteration) or balls (exact=True, for the certificate)."""
        if exact:
            fv = lambda k, i: aff(self.forms[i][k], h)
            gam = lambda i: self.g[i]["gamma"]
            mx = lambda xs: max(xs, key=lambda z: z.upper())
            zero = arb(0)
            bfac = self.beta_fac
        else:
            fv = lambda k, i: aff(self.fl[k][i], h)
            gam = lambda i: self.gfl[i]
            mx = max
            zero = 0.0
            bfac = float(self.beta_fac.upper().mid())
        cl = self.cells
        # prefix maxima over the ray (for P_S) and over the sector (for P_beta)
        ray_r = [(float(cl[i][1].r_hi.mid()), fv("IS", i)) for i in self.ray]
        PS, PB, Ic, arg = [], [], [], {}
        run_b = zero
        k = 0
        run_ray = zero
        for i in self.sector:
            c = cl[i][1]
            rh = float(c.r_hi.mid())
            while k < len(ray_r) and ray_r[k][0] <= rh * (1 + 1e-12):
                run_ray = mx([run_ray, ray_r[k][1]])
                k += 1
            IS = mx([fv("IS", i), run_ray])
            PS.append((fv("locS", i) + gam(i) * IS, i))
            run_b = mx([run_b, fv("IB", i)])
            PB.append((fv("locB", i) + bfac * run_b, i))
            Ic.append(fv("Ic", i))
        iF = self.F
        all_ray = mx([v for _, v in ray_r])
        PS.append((fv("locS", iF) + gam(iF) * mx([fv("IS", iF), all_ray]), iF))
        PB.append((fv("locB", iF) + bfac * mx([run_b, fv("IB", iF)]), iF))
        Ic.append(fv("Ic", iF))
        st = self.st
        if exact:
            om, slow, d0, F1 = self.omega_abs, self.slow, st.d0, self.absF
        else:
            om, slow, d0, F1 = (float(self.omega_abs.upper().mid()), float(self.slow.upper().mid()),
                                float(st.d0.upper().mid()), float(self.absF.upper().mid()))
        PV = om * slow * h[1] + 2 * F1 * d0
        Pc = slow * mx(Ic) + d0
        PSv = mx([v for v, _ in PS])
        PBv = mx([v for v, _ in PB])
        out = [PV, PSv, PBv, Pc]
        if detail:
            key = (lambda t: float(t[0].upper().mid())) if exact else (lambda t: t[0])
            return out, {"S_at": max(PS, key=key)[1], "B_at": max(PB, key=key)[1]}
        return out


    def pieces(self):
        """P as a max of affine maps: per component, a list of (b, a) with P_i(h) = max_k b_k + a_k . h (floats)."""
        fl, cl = self.fl, self.cells
        bfac = float(self.beta_fac.upper().mid())
        slow = float(self.slow.upper().mid())
        add = lambda f, g, c: [f[i] + c * g[i] for i in range(5)]
        PS, PB, Pc = [], [], []
        rays = [(float(cl[i][1].r_hi.mid()), i) for i in self.ray]
        seen_ib = []
        for i in self.sector:
            rh = float(cl[i][1].r_hi.mid())
            g = self.gfl[i]
            for j in [i] + [jr for r, jr in rays if r <= rh * (1 + 1e-12)]:
                PS.append(add(fl["locS"][i], fl["IS"][j], g))
            seen_ib.append(i)
            for j in seen_ib:
                PB.append(add(fl["locB"][i], fl["IB"][j], bfac))
            Pc.append([slow * v for v in fl["Ic"][i]])
        iF = self.F
        for j in [iF] + [jr for _, jr in rays]:
            PS.append(add(fl["locS"][iF], fl["IS"][j], self.gfl[iF]))
        for j in seen_ib + [iF]:
            PB.append(add(fl["locB"][iF], fl["IB"][j], bfac))
        Pc.append([slow * v for v in fl["Ic"][iF]])
        om = float(self.omega_abs.upper().mid())
        d0 = float(self.st.d0.upper().mid())
        F1 = float(self.absF.upper().mid())
        PV = [[2 * F1 * d0, 0.0, om * slow, 0.0, 0.0]]
        Pc = [[v[0] + d0] + v[1:] for v in Pc]
        return [PV, PS, PB, Pc]


def closure_lp(P, hstar, Hmax):
    """{h : P(h) < h} is convex and polyhedral: maximize the margin s in b_k + a_k . h <= h_i - s h*_i over the
    pieces of every component, then take the point of least sum h_i/h*_i at half that margin (y = h/h*)."""
    import numpy as np
    from scipy.optimize import linprog
    rows, rhs = [], []
    for i, comp in enumerate(P.pieces()):
        for pc in comp:
            r = [pc[1 + j] * hstar[j] / hstar[i] for j in range(4)]
            r[i] -= 1.0
            rows.append(r + [1.0])
            rhs.append(-pc[0] / hstar[i])
    A, b = np.array(rows), np.array(rhs)
    ymax = [Hmax[j] / hstar[j] for j in range(4)]
    bnds = [(0, ymax[j]) for j in range(4)] + [(-1, 1)]
    r1 = linprog([0, 0, 0, 0, -1], A_ub=A, b_ub=b, bounds=bnds, method="highs")
    if r1.status != 0 or r1.x[4] <= 0:
        return None, (float(r1.x[4]) if r1.status == 0 else None), len(rows)
    s = float(r1.x[4]) / 2
    bnds2 = [(0, ymax[j]) for j in range(4)] + [(s, s)]
    r2 = linprog([1, 1, 1, 1, 0], A_ub=A, b_ub=b, bounds=bnds2, method="highs")
    y = r2.x[:4] if r2.status == 0 else r1.x[:4]
    return [float(y[j] * hstar[j]) for j in range(4)], float(r1.x[4]), len(rows)


def iterate(P, Hmax, steps=4000, tol=1e-12):
    h = [0.0] * 4
    for it in range(steps):
        hn = P(h)
        if any(hn[i] > Hmax[i] for i in range(4)):
            return None, it, hn
        if max(abs(hn[i] - h[i]) / max(hn[i], 1e-300) for i in range(4)) < tol:
            return hn, it, hn
        h = hn
    return None, steps, h


def jacobian(P, h):
    J = [[0.0] * 4 for _ in range(4)]
    base = P(h)
    for j in range(4):
        dh = max(h[j], 1.0) * 1e-6
        hp_ = list(h)
        hp_[j] += dh
        v = P(hp_)
        for i in range(4):
            J[i][j] = (v[i] - base[i]) / dh
    return J


def solve4(M, b):
    import numpy as np
    return [float(v) for v in np.linalg.solve(np.array(M, dtype=float), np.array(b, dtype=float))]


def spectral_radius(J):
    import numpy as np
    return float(max(abs(np.linalg.eigvals(np.array(J, dtype=float)))))


def closure_test(P, hstar, Hmax=None):
    """Find h with P(h) < h strictly (balls): first h = h* + t u, (I - J) u = h*; then the polyhedral fallback."""
    J = jacobian(P, hstar)
    rho = spectral_radius(J)
    IJ = [[(1.0 if i == j else 0.0) - J[i][j] for j in range(4)] for i in range(4)]
    try:
        u = solve4(IJ, hstar)
    except Exception:
        return None, rho
    if any(ui <= 0 for ui in u):
        u = [abs(ui) + 1e-3 * hstar[i] for i, ui in enumerate(u)]
    for t in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3):
        h = [hstar[i] + t * u[i] for i in range(4)]
        if Hmax is not None and any(h[i] > Hmax[i] for i in range(4)):
            break                                  # the bounds hold for h <= Hmax only
        hb = [arb(repr(float(v))) for v in h]
        Pb = P(hb, exact=True)
        if all(Pb[i].upper() < hb[i].lower() for i in range(4)):
            return {"t": t, "h": hb, "P": Pb, "ratio": [float((Pb[i] / hb[i]).upper().mid()) for i in range(4)],
                    "rhoJ": rho, "method": "h*+tu"}, rho
    if Hmax is not None:
        h, smax, nrows = closure_lp(P, hstar, Hmax)
        if h is not None:
            hb = [arb(repr(float(v))) for v in h]
            Pb = P(hb, exact=True)
            if all(Pb[i].upper() < hb[i].lower() for i in range(4)):
                return {"t": None, "h": hb, "P": Pb, "ratio": [float((Pb[i] / hb[i]).upper().mid()) for i in range(4)],
                        "rhoJ": rho, "method": f"polyhedral ({nrows} pieces, margin {smax:.3g})"}, rho
    return None, rho


def build(st, at, om_abs, absF, res=None):
    res = res or {}
    cells = make_cells(st, **res)
    t0 = time.time()
    forms = []
    for reg, cl in cells:
        f, _ = cell_forms(cl, at, om_abs)
        forms.append(f)
    g, s_star = gains(st, cells, forms)
    P = Pmap(st, cells, forms, g, s_star, absF)
    P.omega_abs = om_abs
    return P, time.time() - t0


def run_L0(L0, d0, domain, at, Hmax0=None, res=None, quiet=False):
    om_abs = at["par"]["omega"].abs_upper()
    absF = norm(at["par"]["F"])
    Hmax = Hmax0 or [3e7, 8e7, 5e6, 2e6]
    if float(d0) > 0 and Hmax0 is None:
        Hmax = [3e8, 3e8, 3e7, 3e7]
    for attempt in range(10):
        st = Setting(L0, d0, domain, KAPPA, Hmax)
        try:
            P, dt = build(st, at, om_abs, absF, res)
        except BoundError as e:
            return {"L0": L0, "closed": False, "why": str(e), "Hmax": Hmax}
        hstar, it, hlast = iterate(P, Hmax)
        if hstar is None:
            if it >= 4000:
                return {"L0": L0, "closed": False, "why": "iteration did not converge", "Hmax": Hmax}
            # iterate exceeded Hmax: retry larger (the bounds were only valid for h <= Hmax)
            Hmax = [max(float(Hmax[i]), 3 * float(hlast[i])) for i in range(4)]
            if max(Hmax) > 1e11:
                return {"L0": L0, "closed": False, "why": "the iterate passes 1e11 (no fixed point found)", "Hmax": Hmax}
            if not quiet:
                log(f"  L0 = {L0}: iterate passed Hmax, retrying with Hmax = {[f'{v:.3g}' for v in Hmax]}")
            continue
        # self-consistent Hmax: h* <= Hmax <= 1.2 h* (the bounds hold for h <= Hmax; folding stays sharp)
        if any(not (1.0 <= Hmax[i] / hstar[i] <= 1.2) for i in range(4)):
            Hmax = [1.1 * v for v in hstar]
            continue
        cert, rho = closure_test(P, hstar, Hmax)
        if cert and all(cert["h"][i].upper() <= Hmax[i] for i in range(4)):
            _, det = P(cert["h"], exact=True, detail=True)
            return {"L0": L0, "closed": True, "hstar": hstar, "h": [float(v.upper().mid()) for v in cert["h"]],
                    "P": [float(v.upper().mid()) for v in cert["P"]], "ratio": cert["ratio"], "t": cert["t"],
                    "method": cert["method"],
                    "rhoJ": rho, "Hmax": Hmax, "s_star": float(P.s_star.mid()),
                    "gamma_F": float(P.g[P.F]["gamma"].upper().mid()), "K_F": float(P.g[P.F]["KF"].upper().mid()),
                    "H_F": float(P.g[P.F]["H"].upper().mid()), "N_F": float(P.g[P.F]["NF"].mid()),
                    "rho_max": float(st.rho_max.mid()), "S_at": P.cells[det["S_at"]][1].label(),
                    "B_at": P.cells[det["B_at"]][1].label(), "seconds": dt, "ncells": len(P.cells)}
        return {"L0": L0, "closed": False, "why": f"no strict closure near h* (rho(J) = {rho:.4f})", "hstar": hstar,
                "Hmax": Hmax}
    return {"L0": L0, "closed": False, "why": "Hmax retries exhausted", "Hmax": Hmax}


# ---------------------------------------------------------------- checks

def check_instances(at, L0, nsamp=6, seed=3, domain="D"):
    """Planted truth: random remainders under (B_h) at sample points; every exact quantity below its cell bound.
    The bounds are over the parameter box (at); the instances are at the box centre (atc), a point of the box."""
    random.seed(seed)
    atc = SER.atoms(SER.params(True))
    par = atc["par"]
    om_abs = at["par"]["omega"].abs_upper()
    h = [2.4e6, 6.3e6, 6.2e5, 1.7e5]
    Hmax = [2 * v for v in h]
    st = Setting(L0, 0, domain, KAPPA, Hmax)
    cells = make_cells(st, n0c=6, n1c=24)
    nsec = sum(1 for _, c in cells if c.kind == "sector" and c.mode != "F")
    pick = [cells[i] for i in (0, 3, 9, 15, 20, 24, 30, nsec, nsec + 12, nsec + 26, len(cells) - 1)]
    worst = 0.0
    rep = []
    hb = [arb(repr(v)) for v in h]
    for reg, cl in pick:
        f, _ = cell_forms(cl, at, om_abs)
        bounds = {k: float(aff(f[k], hb).upper().mid()) for k in ("locS", "IS", "locB", "IB", "Ic")}
        mx = {k: 0.0 for k in bounds}
        for s in range(nsamp):
            x, n = sample_point(cl, st)
            A = PointArith(atc, x, n, par["Omega"])
            Q = kquant(A)
            w, w1 = weights(x, n)
            def rv(kind, size):
                m = 2 if kind == "o" else 3
                v = [acb(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(m)]
                nv = norm(v)
                return [z * size / nv for z in v]
            R = {"RV": [rv("o", h[0] * w1)], "RS": [rv("o", h[1] * w)], "Rhb": [rv("e", h[2] * w)],
                 "Rc": [rv("e", h[3] * w1)]}
            P1 = rquant(A, Q, R)
            R2 = {"RV": R["RV"] + [P1["dRV"][0]], "RS": R["RS"] + [P1["dRS"][0]],
                  "Rhb": R["Rhb"] + [P1["dRhb"][0]], "Rc": R["Rc"] + [P1["dRc"][0]]}
            P2 = rquant(A, Q, R2)
            nm1 = n - 1
            vals = {"locS": norm(P1["G1"][0]) / w, "IS": (norm(P2["G1"][1]) + norm(P2["W2x"][0])) / w,
                    "locB": norm(P1["gh"][0]) / (w * nm1.real), "IB": norm(P2["gh"][1]) / (w * nm1.real),
                    "Ic": norm(P1["dRc"][0]) / w}
            for k, v in vals.items():
                r = float(v.upper().mid()) / bounds[k]
                mx[k] = max(mx[k], r)
        worst = max(worst, max(mx.values()))
        rep.append(f"{cl.label():28s} " + " ".join(f"{k}={v:.3f}" for k, v in mx.items()))
    return worst < 1, worst, rep


def weights(x, n):
    r = x.abs_upper()
    X = 1 + r
    rho = X * X / n.real
    w1 = apow(rho, K + 1)
    return w1 * r / X, w1


def sample_point(cl, st):
    al = float(st.alpha.mid())
    if cl.mode == "F":
        # X in [21, ...], n >= n0 with rho <= rho_max
        L = float(st.L0.mid()) + random.uniform(0, 3)
        n = math.exp(L)
        rho = random.choice([random.uniform(0, 1) * float(st.rho_max.mid()), float(st.rho_max.mid()) * 0.999])
        X = max(21.0, math.sqrt(rho * n))
        r = X - 1
        ph = random.uniform(0, al)
        return acb(arb(repr(r)) * arb(repr(ph)).cos(), arb(repr(r)) * arb(repr(ph)).sin()), acb(arb(L).exp())
    lo, hi = float(cl.r_lo.mid()), float(cl.r_hi.mid())
    while True:
        r = lo + (hi - lo) * random.random()
        ph = al if cl.kind == "ray" else random.uniform(0, al)
        if abs(complex(r * math.cos(ph), r * math.sin(ph)) - 1) >= float(DELTA.mid()) / 2:
            break
    L = float(st.L0.mid()) + random.uniform(0, 3)
    rr, pp = arb(repr(r)), arb(repr(ph))
    return acb(rr * pp.cos(), rr * pp.sin()), acb(arb(L).exp())


def check_kernel(st):
    """Lemma K.1 at the domain's ray: Re(lambda dx)/|lambda dx| <= -sin(kappa_margin) for t in (0, 1e6]."""
    kap_m = arb("0.4") if MUT == "kappa" else st.kappa
    e = acb(st.alpha.cos(), st.alpha.sin())
    worst = -10.0
    for i in range(1, 4001):
        t = arb(10) ** (arb(-4) + arb(10) * i / 4000)
        x = e * t
        lam = 1 / (x * (x * x - 1))
        v = lam * e
        ratio = v.real / v.abs_lower()
        worst = max(worst, float((ratio + kap_m.sin()).upper().mid()))
    return worst <= 0, worst


def check_gains(st_n=300.0, seed=5):
    """Planted truth for Lemma K.3: the scalar kernel exp((n-1)(Phi(x) - Phi(s))) (H' = 0) integrated numerically along
    P_x at a moderate n; the integral over w(x) must stay below the gain of the target's cell."""
    import mpmath as mp
    mp.mp.dps = 30
    random.seed(seed)
    n = st_n
    kap = float(KAPPA.mid())
    kap_m = 0.4 if MUT == "kappa" else kap
    al = math.pi / 4 + kap / 2
    sa = math.sin(al) / al
    sk = math.sin(kap_m)
    Phi = lambda z: 0.5 * mp.log(1 - z ** -2)
    w = lambda r: (r / (1 + r)) * ((1 + r) ** 2 / n) ** (K + 1)
    worst = 0.0
    rows = []
    for r in (0.3, 0.8, 1.5, 3.0, 8.0, 14.0):
        phx = random.uniform(0, al) if abs(r - 1) > 0.05 else al
        x = mp.mpc(r * math.cos(phx), r * math.sin(phx))
        Px = Phi(x)
        e = mp.expj(al)
        f_ray = lambda t: mp.exp((n - 1) * mp.re(Px - Phi(t * e))) * w(t) / t
        ray = mp.quad(f_ray, [0, r * 0.5, r * 0.9, r * 0.99, r])
        f_arc = lambda ph: mp.exp((n - 1) * mp.re(Px - Phi(r * mp.expj(ph)))) * w(r)
        arc = mp.quad(f_arc, [phx, (phx + al) / 2, al])
        true = float((ray + arc) / w(r))
        Q = 1 + 2 * r * r * math.sin(kap) + r ** 4
        N = (n - 1) * r * r / Q
        Dr = math.sqrt(Q)
        gam = Dr / ((n - 1) * sk) + min(math.sqrt(math.pi / (N * sa * sa)), al)
        worst = max(worst, true / gam)
        rows.append(f"r={r:5.2f} arg={phx:.3f}: integral/w = {true:.4e}, gain = {gam:.4e}, ratio {true / gam:.3f}")
    return worst <= 1, worst, rows


def check_closure_planted():
    """The closure search on a planted affine map with a known threshold: P_L(h) = M(L) h + f, rho(M(L)) = 1 at L = 60."""
    def mk(L):
        a = 0.9 * math.exp(-(L - 60) / 10)        # spectral radius crosses 1 at L = 60 + 10 ln 0.9 = 58.946
        M = [[0, a * 0.5, 0, 0], [0, 0, a, 0], [0, a, 0, 0], [0, 0.1, 0, 0]]
        f = [1e5, 2e5, 3e5, 1e4]

        class Pl:
            def __call__(self, h, exact=False, detail=False):
                if exact:
                    return [arb(repr(f[i])) + sum((arb(repr(M[i][j])) * h[j] for j in range(4)), arb(0))
                            for i in range(4)]
                return [f[i] + sum(M[i][j] * h[j] for j in range(4)) for i in range(4)]
        return Pl()
    L_true = 60 + 10 * math.log(0.9)
    res = []
    for L in (L_true - 0.2, L_true + 0.2):
        P = mk(L)
        hstar, it, _ = iterate(P, [1e15] * 4, steps=20000)
        ok = hstar is not None and closure_test(P, hstar)[0] is not None
        res.append((L, ok))
    good = (not res[0][1]) and res[1][1]
    return good, L_true, res


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    at = SER.atoms(SER.params("centre" in OPTS), mutate=MUT if MUT in ("atom", "tauD10", "drop_rho2") else None)
    ok_s, rep_s = SER.check_structure(at, "centre" in OPTS)
    chk = OPTS.get("check")
    log(f"# p2c_b8_boot {' '.join(sys.argv[1:])}  (python-flint, {SER.PREC} bits)")
    if not ok_s:
        for r in rep_s:
            log("  ", r)
        log("STRUCTURE FAIL (Lemma F's structural zeros)")
        sys.exit(1)
    if chk == "identities":
        atc = SER.atoms(SER.params(True), mutate=MUT if MUT in ("atom", "tauD10", "drop_rho2") else None)
        ok, rep = check_identities(atc)
        for r in rep:
            log("  ", r)
        log("IDENTITIES", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    if chk == "instances":
        ok, worst, rep = check_instances(at, OPTS.get("L0", "73.89"))
        for r in rep:
            log("  ", r)
        log(f"INSTANCES {'PASS' if ok else 'FAIL'}: largest exact/bound = {worst:.6f}")
        sys.exit(0 if ok else 1)
    if chk == "kernel":
        st = Setting(OPTS.get("L0", "73.89"))
        ok, worst = check_kernel(st)
        log(f"KERNEL {'PASS' if ok else 'FAIL'}: max of Re(lam dx)/|lam dx| + sin(kappa_margin) = {worst:.4e}")
        sys.exit(0 if ok else 1)
    if chk == "gains":
        ok, worst, rows = check_gains()
        for r in rows:
            log("  ", r)
        log(f"GAINS {'PASS' if ok else 'FAIL'}: largest integral/gain = {worst:.4f}")
        sys.exit(0 if ok else 1)
    if chk == "closure_planted":
        ok, Lt, res = check_closure_planted()
        log(f"  threshold {Lt:.4f}; " + ", ".join(f"L = {L:.3f}: closed = {c}" for L, c in res))
        log("CLOSURE_PLANTED", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    d0 = OPTS.get("d0", "0")
    domain = OPTS.get("domain", "D")
    res = {}
    if "n1c" in OPTS:
        res["n1c"] = int(OPTS["n1c"])
    if "n0c" in OPTS:
        res["n0c"] = int(OPTS["n0c"])
    out = []
    if "bisect" in OPTS:
        lo, hi = (float(v) for v in OPTS["bisect"].split(","))
        tol = float(OPTS.get("tol", "0.25"))
        rhi = run_L0(repr(hi), d0, domain, at, res=res)
        log(fmt(rhi))
        out.append(rhi)
        if not rhi["closed"]:
            log("BISECT: no closure at the upper end")
        rlo = run_L0(repr(lo), d0, domain, at, res=res)
        log(fmt(rlo))
        out.append(rlo)
        while hi - lo > tol and rhi["closed"] and not rlo["closed"]:
            mid = round((lo + hi) / 2, 4)
            r = run_L0(repr(mid), d0, domain, at, res=res)
            log(fmt(r))
            out.append(r)
            if r["closed"]:
                hi = mid
            else:
                lo = mid
        log(f"BISECT d0 = {d0}, domain {domain}: closes at ln n0 = {hi}, not at {lo}")
    else:
        for L0 in OPTS.get("L0", "73.89").split(","):
            r = run_L0(L0, d0, domain, at, res=res)
            log(fmt(r))
            out.append(r)
    if "json" in OPTS:
        with open(OPTS["json"], "w") as fh:
            json.dump(out, fh, indent=1, default=str)
    log(f"# {time.time() - t0:.0f} s")
    sys.exit(0 if all(r["closed"] for r in out[:1]) else 2)


def fmt(r):
    if not r["closed"]:
        return f"L0 = {r['L0']}: NOT CLOSED ({r['why']})"
    h = ", ".join(f"{v:.4e}" for v in r["h"])
    q = ", ".join(f"{v:.4f}" for v in r["ratio"])
    return (f"L0 = {r['L0']}: CLOSED h = ({h}), P(h)/h = ({q}), rho(J) = {r['rhoJ']:.4f}, by {r['method']}, "
            f"rho_max = {r['rho_max']:.5f}, s_* = {r['s_star']:.4f}, K_F = {r['K_F']:.4f}, H_F = {r['H_F']:.3f}, "
            f"N_F = {r['N_F']:.1f}, gamma_F = {r['gamma_F']:.4f}, S at {r['S_at']}, beta at {r['B_at']}, "
            f"{r['ncells']} cells, {r['seconds']:.0f} s")


if __name__ == "__main__":
    main()
