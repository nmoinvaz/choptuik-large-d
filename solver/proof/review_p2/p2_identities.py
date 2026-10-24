#!/usr/bin/env python3
"""P2 review: independent checks of the algebra behind B8 and Lemma LC, at random complex points, in mpmath (80
digits), with the formal series of p2_formal.py (independent of b8m_formal.py / b8m_model.py / b8m_identities.py).

  T1 (reduction, B8_constants.md 1.2): take random values (S, Dl, beta, c) at random complex x, n, and solve the
     ORIGINAL M = 1 equations (sum and difference of the p- and m-equations, b, c) for (S', Dl', beta', c'). With
     V := ([c Dl] - x S)/x the reduced equations x V' = Om S and x D S' = A S - x^2 B V must then hold.
  T2 (Lemma R, B8_constants.md 4, as stated there): (I1) for the truncated series Z^K (K = 4); (I2), (I3), (I4) for
     Z = Z^K + R with R, R' random, including the defect terms; both forms of q.
  T3 (Lemma LC, B_chain.md 13.3): x W2 = Lambda G2 with G2 = (A^K)^-1 D^K D^-1 (C M_h S + x^2 M_h C^-1 Y), and
     q (Lambda G1 + x W2) = ell x^-2 (I + e)^-1 eps1 A^K (G1 + G2), eps1 = 1/(n - 1).
  T4 (Lemma J, B_chain.md 14): with P = C^-1 Y (B5's P = Dl/x), Phi := x P' = C^-1 (x Y' - [x c' P]).

    python3 p2_identities.py [--mutate=rho2|q|g3]
    mutations: rho2 (the x^2 M_rho4 C^-1 Y term dropped from rho2: I1 and I2 must fail), q (q without (I + e)^-1: the
    q-forms and I3 must fail), g3 (the factor 1/2 of [dDl dDl] in g3 dropped: I3 must fail).
"""
import random
import sys

from mpmath import mp, mpc, mpf, matrix, inverse, pi, fabs

import p2_formal as PF

mp.dps = 80
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
ODD, EVEN = (-1, 1), (-2, 0, 2)
RNG = random.Random(20261023)


def rc(s=1):
    return mpc(RNG.uniform(-s, s), RNG.uniform(-s, s))


# ------------------------------------------------------------------ vectors, operators
def proj(f, g, keep):
    out = {h: mpc(0) for h in keep}
    for a, u in f.items():
        for b, v in g.items():
            if a + b in out:
                out[a + b] += u * v
    return out


def vadd(*vs):
    out = {}
    for v in vs:
        for h, u in v.items():
            out[h] = out.get(h, 0) + u
    return out


def vsc(v, s):
    return {h: u * s for h, u in v.items()}


def col(v):
    return matrix([[v[m]] for m in ODD])


def vec(m):
    return {ODD[i]: m[i, 0] for i in range(2)}


def mop(g):
    """the operator f -> [g f] on odd vectors (g even)"""
    return matrix([[g.get(a - b, mpc(0)) for b in ODD] for a in ODD])


def ap(M, v):
    return vec(M * col(v))


def nrm(v):
    return max(fabs(u) for u in v.values()) if isinstance(v, dict) else max(fabs(v[i, j]) for i in range(v.rows)
                                                                          for j in range(v.cols))


I2 = matrix([[1, 0], [0, 1]])


class Model:
    """the reduced-variable operators of B8_constants.md 1.2 at a point"""

    def __init__(self, x, n, w):
        self.x, self.n, self.w = x, n, w
        self.eps = 1 / n
        self.Om = matrix([[mpc(0, ODD[i] * w) if i == j else 0 for j in range(2)] for i in range(2)])

    def ops(self, V, S, be, c):
        x, n, Om = self.x, self.n, self.Om
        C = mop(c)
        Ci = inverse(C)
        cb = proj(c, be, EVEN)
        Bb = mop(cb)
        D = x * x * I2 - C * C
        Bm = I2 - Bb * Ci + C * Om * Ci
        A = (n - 1) * C * C + C * Bb - x * x * Om - x * x * Bm
        Y = vadd(V, S)
        Dl = vsc(ap(Ci, Y), x)
        H = vsc(vadd(proj(S, S, EVEN), proj(Dl, Dl, EVEN)), mpf("0.5"))
        return dict(C=C, Ci=Ci, cb=cb, Bb=Bb, D=D, B=Bm, A=A, Y=Y, Dl=Dl, H=H)


# ------------------------------------------------------------------ T1: the reduction
def t1(trials=6):
    worst = mpf(0)
    for _ in range(trials):
        x, n, w = rc() + 1.5, mpf(RNG.uniform(5, 50)) + rc(2), mpf(RNG.uniform(1, 4))
        md = Model(x, n, w)
        eps = 1 / n
        S = {m: rc() for m in ODD}
        Dl = {m: rc() for m in ODD}
        be = {k: rc() for k in EVEN}
        c = {k: rc(0.3) + (1 if k == 0 else 0) for k in EVEN}
        Om = {m: mpc(0, m * w) for m in ODD}
        C = mop(c)
        cb = proj(c, be, EVEN)
        # original: x^2 S' - x[c Dl'] = -x Om S + [[c beta] Dl] - [c Dl];  x^2 Dl' - x[c S'] = -x Om Dl + [[c beta] S]
        #           + (n - 1)[c S];  x beta' = (1 - e) H + e [beta H] - (n - 1) beta - [beta beta];  x c' = [c beta]
        r1 = vadd({m: -x * Om[m] * S[m] for m in ODD}, proj(cb, Dl, ODD), vsc(proj(c, Dl, ODD), -1))
        r2 = vadd({m: -x * Om[m] * Dl[m] for m in ODD}, proj(cb, S, ODD), vsc(proj(c, S, ODD), n - 1))
        Mbig = matrix(4, 4)
        for i in range(2):
            for j in range(2):
                Mbig[i, j] = x * x * (1 if i == j else 0)
                Mbig[i, 2 + j] = -x * C[i, j]
                Mbig[2 + i, j] = -x * C[i, j]
                Mbig[2 + i, 2 + j] = x * x * (1 if i == j else 0)
        sol = inverse(Mbig) * matrix([[r1[-1]], [r1[1]], [r2[-1]], [r2[1]]])
        Sp = {-1: sol[0, 0], 1: sol[1, 0]}
        Dlp = {-1: sol[2, 0], 1: sol[3, 0]}
        cp = vsc(cb, 1 / x)
        # V and V' from the definition V = ([c Dl] - x S)/x
        V = vsc(vadd(proj(c, Dl, ODD), vsc(S, -x)), 1 / x)
        Vp = vadd(vsc(vadd(proj(cp, Dl, ODD), proj(c, Dlp, ODD), vsc(S, -1), vsc(Sp, -x)), 1 / x), vsc(V, -1 / x))
        o = md.ops(V, S, be, c)
        # Dl consistency: x C^-1 Y must equal Dl
        e0 = nrm(vadd(o["Dl"], vsc(Dl, -1)))
        e1 = nrm(vadd(vsc(Vp, x), vsc(ap(md.Om, S), -1)))
        lhs = vsc(ap(o["D"], Sp), x)
        rhs = vadd(ap(o["A"], S), vsc(ap(o["B"], V), -x * x))
        e2 = nrm(vadd(lhs, vsc(rhs, -1))) / max(nrm(lhs), 1)
        worst = max(worst, e0, e1, e2)
    return worst


# ------------------------------------------------------------------ the truncated series at a point
class Series:
    def __init__(self, K=4, order=None):
        A = mpf("2.5421480475")
        self.w = 2 * pi / mpf("1.64389375") * mpf("0.7769940136")
        self.K = K
        fs = PF.formal(order or K, A, self.w)
        self.fs = fs
        self.F = fs["F"]

    def ev(self, name, keys, x, n, d=0):
        out = {}
        for h in keys:
            tot = mpc(0)
            for k in range(self.K + 1):
                p = self.fs[name][k][h]
                if d == 0:
                    v = sum((u * x ** m for m, u in p.items()), mpc(0))
                elif d == 1:
                    v = sum((u * m * x ** (m - 1) for m, u in p.items() if m >= 1), mpc(0))
                else:
                    v = sum((u * m * (m - 1) * x ** (m - 2) for m, u in p.items() if m >= 2), mpc(0))
                tot += v / n ** k
            out[h] = tot
        return out

    def at(self, x, n):
        """Z^K and derivatives (first, second) at x"""
        z = {}
        for nm, keys in (("V", ODD), ("S", ODD), ("be", EVEN), ("c", EVEN), ("Dl", ODD)):
            z[nm] = self.ev(nm, keys, x, n)
            z[nm + "'"] = self.ev(nm, keys, x, n, 1)
            z[nm + "''"] = self.ev(nm, keys, x, n, 2)
        return z


def lemma_r(ser, x, n, R, Rp, mut=None):
    """residuals of (I1)-(I4) and of the two q forms; R = (RV, RS, Rb, Rc) values, Rp their x-derivatives"""
    md = Model(x, n, ser.w)
    eps = 1 / n
    Om = md.Om
    z = ser.at(x, n)
    VK, SK, bK, cK, DlK = z["V"], z["S"], z["be"], z["c"], z["Dl"]
    oK = md.ops(VK, SK, bK, cK)
    CK, CKi, DK, AK, BK = oK["C"], oK["Ci"], oK["D"], oK["A"], oK["B"]
    YK = oK["Y"]
    # residuals of Z^K (Dl^K the series)
    xSp = vsc(z["S'"], x)
    tauD = vadd(ap(CK, xSp), vsc(z["Dl'"], -x * x), vsc(ap(Om, DlK), -x), ap(oK["Bb"], SK), vsc(ap(CK, SK), n - 1))
    tauv = vadd(ap(CK, DlK), vsc(YK, -x))
    xcK = vsc(z["c'"], x)
    tauvp = vadd(proj(z["c'"], DlK, ODD), proj(cK, z["Dl'"], ODD), vsc(YK, -1), vsc(vadd(z["V'"], z["S'"]), -x))
    HK = vsc(vadd(proj(SK, SK, EVEN), proj(DlK, DlK, EVEN)), mpf("0.5"))
    taub = vadd(vsc(z["be'"], x), vsc(bK, n - 1), proj(bK, bK, EVEN), vsc(HK, -(1 - eps)), vsc(proj(bK, HK, EVEN), -eps))
    rho4 = vadd(xcK, vsc(oK["cb"], -1))
    rho2 = vadd(vsc(ap(mop(rho4) * CKi, YK), x * x), vsc(ap(CK, tauD), -1), vsc(tauvp, -x * x),
                vsc(ap(mop(xcK) * CKi, tauv), x), vsc(ap(CK * Om * CKi, tauv), -x))
    if mut == "rho2":
        rho2 = vadd(rho2, vsc(ap(mop(rho4) * CKi, YK), -x * x))
    # (I1)
    lhs1 = vadd(vsc(ap(DK, z["S'"]), x), vsc(ap(AK, SK), -1), vsc(ap(BK, VK), x * x))
    e_I1 = nrm(vadd(lhs1, vsc(rho2, -1))) / max(nrm(rho2), mpf(10) ** -60)
    # q and its pieces
    Lam = inverse(DK) * AK
    bKv = {k: (1 if k == 0 else 0) + bK[k] / (n - 1) for k in EVEN}
    Dt = vsc(ap(CKi, YK), x)

    def ell(f):
        return vsc(proj(bKv, vadd(proj(SK, f, EVEN), proj(Dt, vsc(ap(CKi, f), x), EVEN)), EVEN), 1 - eps)

    def ellmat():
        cols = []
        for j in ODD:
            f = {m: mpc(1 if m == j else 0) for m in ODD}
            cols.append(ell(f))
        return matrix([[cols[jj][k] for jj in range(2)] for k in EVEN])       # 3 x 2

    eK = (CK * oK["Bb"] / (x * x) - Om - BK) / (n - 1)
    if mut == "q":
        q1 = ellmat() * DK / ((n - 1) * x * x)
    else:
        q1 = ellmat() * inverse(I2 + eK) * DK / ((n - 1) * x * x)
    q2 = ellmat() * inverse((n - 1) * I2 + Lam)
    e_q = nrm(q1 - q2) / nrm(q2)
    q = q1

    def qop(v):
        r = q * col(v)
        return {EVEN[k]: r[k, 0] for k in range(3)}

    # x q'(x) by differentiating the formula (q2 form) numerically
    def qmat_at(xx):
        mdx = Model(xx, n, ser.w)
        zz = ser.at(xx, n)
        o = mdx.ops(zz["V"], zz["S"], zz["be"], zz["c"])
        LamX = inverse(o["D"]) * o["A"]
        bKx = {k: (1 if k == 0 else 0) + zz["be"][k] / (n - 1) for k in EVEN}
        Dtx = vsc(ap(o["Ci"], o["Y"]), xx)
        cols = []
        for j in ODD:
            f = {m: mpc(1 if m == j else 0) for m in ODD}
            v = vsc(proj(bKx, vadd(proj(zz["S"], f, EVEN), proj(Dtx, vsc(ap(o["Ci"], f), xx), EVEN)), EVEN), 1 - eps)
            cols.append(v)
        Lm = matrix([[cols[jj][k] for jj in range(2)] for k in EVEN])
        return Lm * inverse((n - 1) * I2 + LamX)

    dq = mp.diff(lambda t: qmat_at(t), x) if False else None
    hstep = mpf(10) ** -25
    xqp = (qmat_at(x + hstep) - qmat_at(x - hstep)) * (x / (2 * hstep))
    # the full Z = Z^K + R
    RV, RS, Rb, Rc = R
    RVp, RSp, Rbp, Rcp = Rp
    V, S, be, c = vadd(VK, RV), vadd(SK, RS), vadd(bK, Rb), vadd(cK, Rc)
    o = md.ops(V, S, be, c)
    C, Ci, D, A, Bm, Y = o["C"], o["Ci"], o["D"], o["A"], o["B"], o["Y"]
    Sp = vadd(z["S'"], RSp)
    bep = vadd(z["be'"], Rbp)
    cp = vadd(z["c'"], Rcp)
    defS = vadd(vsc(ap(D, Sp), x), vsc(ap(A, S), -1), vsc(ap(Bm, V), x * x))
    defb = vadd(vsc(bep, x), vsc(be, n - 1), proj(be, be, EVEN), vsc(o["H"], -(1 - eps)), vsc(proj(be, o["H"], EVEN), -eps))
    defc = vadd(vsc(cp, x), vsc(o["cb"], -1))
    # A0, B0 with beta -> beta^K + q R_S
    Rb0 = qop(RS)
    o0 = md.ops(V, S, vadd(bK, Rb0), c)
    A0, B0 = o0["A"], o0["B"]
    dD = D - DK
    RbH = vadd(Rb, vsc(Rb0, -1))
    Mh = mop(proj(c, RbH, EVEN))
    inner = vadd(ap(A0 - AK, S), vsc(vadd(ap(B0, V), vsc(ap(BK, VK), -1)), -x * x),
                 vsc(ap(dD * inverse(D), vadd(ap(A0, S), vsc(ap(B0, V), -x * x))), -1), vsc(rho2, -1))
    G1 = ap(inverse(AK), inner)
    W2 = ap(inverse(D) / x, vadd(ap(C * Mh, S), vsc(ap(Mh * Ci, Y), x * x)))
    lhs2 = vsc(RSp, x)
    rhs2 = vadd(ap(Lam, vadd(RS, G1)), vsc(W2, x), ap(inverse(D), defS))
    e_I2 = nrm(vadd(lhs2, vsc(rhs2, -1))) / nrm(lhs2)
    # (I3): R^_beta' = R_beta' - (q R_S)' = R_beta' - q' R_S - q R_S'
    xqR = (xqp * col(RS))
    xRbHp = vadd(vsc(Rbp, x), {EVEN[k]: -xqR[k, 0] for k in range(3)}, vsc(qop(RSp), -x))
    lhs3 = vadd(xRbHp, vsc(RbH, n - 1))
    dDl = vadd(vsc(ap(Ci, vadd(RV, RS)), x), vsc(ap(Ci * mop(Rc), Dt), -1))
    half = mpf("0.5") if mut != "g3" else mpf(1)
    g3 = vadd(vsc(proj(RS, RS, EVEN), mpf("0.5")),
              proj(Dt, vadd(vsc(ap(Ci, RV), x), vsc(ap(Ci * mop(Rc), vadd(vsc(ap(CKi, RS), x), Dt)), -1)), EVEN),
              vsc(proj(dDl, dDl, EVEN), half))
    rho3 = vadd(taub, vsc(proj(bKv, proj(ap(CKi, tauv), vadd(Dt, DlK), EVEN), EVEN), (1 - eps) / 2))
    xgS = vadd(ap(Lam, G1), vsc(W2, x))
    gh = vadd(vsc(proj(Rb, vadd(vsc(bK, 2), Rb), EVEN), -1), vsc(proj(Rb, o["H"], EVEN), eps),
              vsc(proj(bKv, g3, EVEN), 1 - eps), {EVEN[k]: -xqR[k, 0] for k in range(3)}, vsc(qop(xgS), -1),
              vsc(rho3, -1))
    rhs3 = vadd(gh, defb, vsc(qop(ap(inverse(D), defS)), -1))
    e_I3 = nrm(vadd(lhs3, vsc(rhs3, -1))) / nrm(lhs3)
    # (I4)
    lhs4 = vsc(Rcp, x)
    rhs4 = vadd(proj(Rc, be, EVEN), proj(cK, Rb, EVEN), vsc(rho4, -1), defc)
    e_I4 = nrm(vadd(lhs4, vsc(rhs4, -1))) / nrm(lhs4)
    # T3: the L forms
    G2 = ap(inverse(AK) * DK * inverse(D), vadd(ap(C * Mh, S), vsc(ap(Mh * Ci, Y), x * x)))
    e_G2 = nrm(vadd(vsc(W2, x), vsc(ap(Lam, G2), -1))) / nrm(W2)
    lhsq = qop(xgS)
    eps1 = 1 / (n - 1)
    Gs = vadd(G1, G2)
    rq = ellmat() * inverse(I2 + eK) * AK * col(Gs) * (eps1 / (x * x))
    rhsq = {EVEN[k]: rq[k, 0] for k in range(3)}
    e_qL = nrm(vadd(lhsq, vsc(rhsq, -1))) / nrm(lhsq)
    # T4: Phi = x P' with P = C^-1 Y
    xYp = vadd(vsc(vadd(z["V'"], RVp), x), vsc(Sp, x))
    P = ap(Ci, Y)
    hh = mpf(10) ** -25
    # P(x) along the series + R (R frozen in x is not a solution, so use only the identity x(CP)' = xC'P + C xP')
    xPp = ap(Ci, vadd(xYp, vsc(proj(vsc(cp, x), P, ODD), -1)))
    e_Phi = nrm(vadd(ap(C, xPp), proj(vsc(cp, x), P, ODD), vsc(xYp, -1))) / nrm(xYp)
    return dict(I1=e_I1, q=e_q, I2=e_I2, I3=e_I3, I4=e_I4, G2=e_G2, qL=e_qL, Phi=e_Phi)


def t2(trials=4):
    ser = Series(4)
    worst = {}
    for _ in range(trials):
        x = mpf(RNG.uniform(0.3, 2.0)) * mp.expjpi(RNG.uniform(0, 0.25))
        n = mpf(RNG.uniform(20, 200))
        sc = mpf(10) ** -3
        R = tuple({h: rc(sc) for h in keys} for keys in (ODD, ODD, EVEN, EVEN))
        Rp = tuple({h: rc(sc) for h in keys} for keys in (ODD, ODD, EVEN, EVEN))
        e = lemma_r(ser, x, n, R, Rp, MUT)
        for k, v in e.items():
            worst[k] = max(worst.get(k, 0), v)
    return worst


def main():
    w1 = t1()
    print(f"T1 reduction (original M = 1 equations -> B8's reduced system): worst residual {mp.nstr(w1, 3)}")
    w2 = t2()
    print("T2-T4 relative residuals: " + ", ".join(f"{k} {mp.nstr(v, 3)}" for k, v in w2.items()))
    tol = mpf(10) ** -40
    ok = w1 < tol and all(v < tol for v in w2.values())
    if MUT:
        fails = [k for k, v in w2.items() if not v < tol]
        expect = {"rho2": {"I1", "I2"}, "q": {"q", "I3"}, "g3": {"I3"}}[MUT]
        caught = expect <= set(fails)
        print(f"MUTATION {MUT}: failing identities {fails}; expected {sorted(expect)}: {'caught' if caught else 'NOT caught'}")
        return 0 if caught else 3
    print(f"=> {'ALL IDENTITIES HOLD' if ok else 'FAILURE'} (tolerance 1e-40)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
