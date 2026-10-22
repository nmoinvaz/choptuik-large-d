#!/usr/bin/env python3
"""Lemma B8 (proof/B8_constants.md, sections 4-6): the algebraic identities behind the remainder equations, checked in
mpmath (80 digits) at random complex points x, n, for M = 1 and M = 3, with the truncated series of b8m_formal.py.

Notation (b8m_model.py): C f = [c f], Bb f = [[c beta] f] on odd harmonics, Om = diag(i mu omega), D = x^2 - C^2,
A = (n-1) C^2 + C Bb - x^2 Om - x^2 B, B = I - Bb C^-1 + C Om C^-1, Delta = x C^-1 Y, Y = V + S,
H = ([S S] + [Delta Delta])/2.  Superscript K: built from the truncated series (Delta^K = the series Delta).
Residuals of the truncated series: tau_D = x C S' - x^2 Delta' - x Om Delta + Bb S + (n-1) C S, tau_v = C Delta - x Y,
tau_b = x beta' + (n-1) beta + [beta beta] - (1-eps) H - eps [beta H], rho4 = x c' - [c beta].

  (I1) x D^K S^K' - A^K S^K + x^2 B^K V^K = rho2,
       rho2 = x^2 M_rho4 C^-1 Y - C tau_D - x^2 tau_v' + x (x c') C^-1 tau_v - x C Om C^-1 tau_v   (all ^K).
  (I2) for ANY Z = Z^K + R (R, R' arbitrary) with defect_S = x D S' - (A S - x^2 B V):
       x R_S' = Lam (R_S + G1) + x W2 + D^-1 defect_S,   Lam = (D^K)^-1 A^K,
       G1 = (A^K)^-1 [ (A0 - A^K) S - x^2 (B0 V - B^K V^K) - dD D^-1 (A0 S - x^2 B0 V) - rho2 ],  dD = D - D^K,
       W2 = (x D)^-1 (C M_h S + x^2 M_h C^-1 Y),  M_h f = [[c R^_beta] f],
       A0, B0 = A, B with beta replaced by beta^K + q R_S,  R^_beta = R_beta - q R_S,
       q = l (I + e)^-1 D^K / ((n-1) x^2) = l ((n-1) I + Lam)^-1  (both forms checked),
       l f = (1-eps) [b^K ([S^K f] + [Dt x C^K^-1 f])],  Dt = x C^K^-1 Y^K,  b^K = 1 + beta^K/(n-1),
       e = (C^K Bb^K / x^2 - Om - B^K)/(n-1).
  (I3) with defect_b = x beta' + (n-1) beta + [beta beta] - (1-eps) H - eps [beta H]:
       x R^_beta' + (n-1) R^_beta = g^ + defect_b - q D^-1 defect_S,
       g^ = -[R_b (2 beta^K + R_b)] + eps [R_b H] + (1-eps)[b^K g3] - x q' R_S - q (Lam G1 + x W2) - rho3,
       g3 = [R_S R_S]/2 + [Dt (x C^-1 R_V - C^-1 R_C (x C^K^-1 R_S + Dt))] + [dDl dDl]/2,
       dDl = x C^-1 (R_V + R_S) - C^-1 R_C Dt,  R_C f = [R_c f],
       rho3 = tau_b + (1-eps) [b^K [(C^K^-1 tau_v)(Dt + Delta^K)]]/2.
  (I4) x R_c' = [R_c beta] + [c^K R_beta] - rho4 + defect_c,  defect_c = x c' - [c beta].
Mutations: --mutate=rho2old drops the x^2 M_rho4 C^-1 Y term of rho2 (I1 must fail); --mutate=q drops (I + e)^-1
in q (I2's q check and I3 must fail).  Exit status 0 iff all identities hold to 1e-40 (relative).

    python3 b8m_identities.py [--mutate=rho2old|q]
"""
import random
import sys
import mpmath as mp
from flint import ctx
import b8m_formal as FM
import b8m_model as BM

mp.mp.dps = 80
MUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), "")


def to_mp(z):
    return mp.mpc(mp.mpf(z.real.mid().str(85, radius=False)), mp.mpf(z.imag.mid().str(85, radius=False)))


def series_eval(fs, name, keys, x, n, d=0):
    out = {}
    for h in keys:
        tot = mp.mpc(0)
        for k in range(len(fs[name])):
            cs = [to_mp(c) for c in fs[name][k][h].coeffs()]
            if d == 0:
                v = sum(cs[m] * x ** m for m in range(len(cs)))
            elif d == 1:
                v = sum(m * cs[m] * x ** (m - 1) for m in range(1, len(cs)))
            else:
                v = sum(m * (m - 1) * cs[m] * x ** (m - 2) for m in range(2, len(cs)))
            tot += v / mp.mpf(n) ** k
        out[h] = tot
    return out


def run(M, K, rnd):
    O, E = BM.harm(M)
    ctx.prec = 300
    F, om = FM.params(M=M)
    if M > 1:      # a generic profile for M >= 3 (the identities do not depend on it)
        F = {m: F[m] if abs(m) == 1 else F[1] * (0.3 / m) for m in O}
    fs = FM.formal(K, F, om, M)
    w = to_mp(om)
    I = mp.eye(len(O))
    Om = mp.diag([1j * m * w for m in O])
    vec = lambda d: BM.vec(d, O)
    unv = lambda v: BM.unvec(v, O)
    pr = BM.proj
    mm = lambda g: BM.mult_matrix(g, O)

    def evr(x, n):
        Z = {}
        for nm, keys in (("V", O), ("S", O), ("be", E), ("c", E), ("Dl", O)):
            Z[nm] = series_eval(fs, nm, keys, x, n)
            Z[nm + "'"] = series_eval(fs, nm, keys, x, n, 1)
        return Z

    def ops(c, be, x, n):
        Cm = mm(c)
        Bb = mm(pr(c, be, E))
        Ci = Cm ** -1
        B = I - Bb * Ci + Cm * Om * Ci
        A = (n - 1) * Cm * Cm + Cm * Bb - x ** 2 * Om - x ** 2 * B
        D = x ** 2 * I - Cm * Cm
        return Cm, Ci, Bb, B, A, D

    worst = {}

    def rec(name, lhs, rhs):
        lv = lhs if isinstance(lhs, mp.matrix) else mp.matrix(list(lhs))
        rv = rhs if isinstance(rhs, mp.matrix) else mp.matrix(list(rhs))
        num = max(abs(lv[i] - rv[i]) for i in range(lv.rows))
        den = max([abs(lv[i]) for i in range(lv.rows)] + [abs(rv[i]) for i in range(rv.rows)] + [mp.mpf(10) ** -70])
        worst[name] = max(worst.get(name, mp.mpf(0)), num / den)

    for trial in range(3):
        x = mp.mpc(rnd.uniform(0.4, 3.0), rnd.uniform(0.2, 1.5))
        n = mp.mpf(rnd.uniform(30, 300))
        eps = 1 / n
        Zk = evr(x, n)
        VK, SK, bK, cK, DlK = (Zk[s] for s in ("V", "S", "be", "c", "Dl"))
        dSK, dbK, dcK, dDlK, dVK = (Zk[s + "'"] for s in ("S", "be", "c", "Dl", "V"))
        CK, CKi, BbK, BK, AK, DK = ops(cK, bK, x, n)
        YK = {m: VK[m] + SK[m] for m in O}
        dYK = {m: dVK[m] + dSK[m] for m in O}
        # residuals
        tauv = {m: pr(cK, DlK, O)[m] - x * YK[m] for m in O}
        dtauv = {m: pr(dcK, DlK, O)[m] + pr(cK, dDlK, O)[m] - YK[m] - x * dYK[m] for m in O}
        tauD = vec({m: x * pr(cK, dSK, O)[m] - x * x * dDlK[m] - x * 1j * m * w * DlK[m] for m in O}) \
            + BbK * vec(SK) + (n - 1) * CK * vec(SK)
        HKs = {k: (pr(SK, SK, E)[k] + pr(DlK, DlK, E)[k]) / 2 for k in E}
        taub = {k: x * dbK[k] + (n - 1) * bK[k] + pr(bK, bK, E)[k] - (1 - eps) * HKs[k] - eps * pr(bK, HKs, E)[k]
                for k in E}
        rho4 = {k: x * dcK[k] - pr(cK, bK, E)[k] for k in E}
        # (I1)
        lhs = x * DK * vec(dSK) - AK * vec(SK) + x ** 2 * BK * vec(VK)
        rho2 = (-CK * tauD - x ** 2 * vec(dtauv) + x * (x * mm(dcK)) * CKi * vec(tauv) - x * CK * Om * CKi * vec(tauv))
        if MUT != "rho2old":
            rho2 = rho2 + x ** 2 * mm(rho4) * CKi * vec(YK)
        rec("I1 rho2", lhs, rho2)
        # a random remainder R and the exact-structure quantities of Z = Z^K + R
        rc = lambda a: mp.mpc(rnd.uniform(-a, a), rnd.uniform(-a, a))
        RV = {m: rc(1e-3) for m in O}
        RS = {m: rc(1e-3) for m in O}
        Rb = {k: rc(1e-3) for k in E}
        Rc = {k: rc(1e-3) for k in E}
        dRS = {m: rc(1e-2) for m in O}
        dRb = {k: rc(1e-2) for k in E}
        dRc = {k: rc(1e-2) for k in E}
        V = {m: VK[m] + RV[m] for m in O}
        S = {m: SK[m] + RS[m] for m in O}
        be = {k: bK[k] + Rb[k] for k in E}
        c = {k: cK[k] + Rc[k] for k in E}
        dS = {m: dSK[m] + dRS[m] for m in O}
        dbe = {k: dbK[k] + dRb[k] for k in E}
        dc = {k: dcK[k] + dRc[k] for k in E}
        Cm, Ci, Bb, B, A, D = ops(c, be, x, n)
        Y = {m: V[m] + S[m] for m in O}
        defS = x * D * vec(dS) - (A * vec(S) - x ** 2 * B * vec(V))
        Lam = DK ** -1 * AK
        # q: l (I + e)^-1 D^K / ((n-1) x^2) and l ((n-1) I + Lam)^-1, as matrices E x O
        Dt = unv(x * CKi * vec(YK))
        bbK = {k: (1 if k == 0 else 0) + bK[k] / (n - 1) for k in E}

        def l_of(xx, nn, cKx, CKix, SKx, Dtx, bbx):
            cols = []
            for j in range(len(O)):
                f = {m: (1 if i == j else 0) for i, m in enumerate(O)}
                xf = unv(xx * CKix * vec(f))
                inner = {k: pr(SKx, f, E)[k] + pr(Dtx, xf, E)[k] for k in E}
                cols.append([(1 - 1 / nn) * pr(bbx, inner, E)[k] for k in E])
            return mp.matrix([[cols[j][i] for j in range(len(O))] for i in range(len(E))])

        def q_of(xx, nn, mut=MUT):
            Z2 = evr(xx, nn)
            CK2, CKi2, BbK2, BK2, AK2, DK2 = ops(Z2["c"], Z2["be"], xx, nn)
            Y2 = {m: Z2["V"][m] + Z2["S"][m] for m in O}
            Dt2 = unv(xx * CKi2 * vec(Y2))
            bb2 = {k: (1 if k == 0 else 0) + Z2["be"][k] / (nn - 1) for k in E}
            L2 = l_of(xx, nn, Z2["c"], CKi2, Z2["S"], Dt2, bb2)
            e2 = (CK2 * BbK2 / xx ** 2 - Om - BK2) / (nn - 1)
            ie = I if mut == "q" else (I + e2) ** -1
            return L2 * ie * DK2 / ((nn - 1) * xx ** 2), L2 * ((nn - 1) * I + DK2 ** -1 * AK2) ** -1

        q, q_alt = q_of(x, n)
        for i in range(q.rows):
            rec("I2 q forms", q[i, :].T, q_alt[i, :].T)
        hh = mp.mpf(10) ** -25
        qp = (q_of(x + hh, n)[0] - q_of(x - hh, n)[0]) / (2 * hh)
        qRS = q * vec(RS)
        Rhat = {k: Rb[k] - qRS[i] for i, k in enumerate(E)}
        b0 = {k: bK[k] + qRS[i] for i, k in enumerate(E)}
        _, _, _, B0, A0, _ = ops(c, b0, x, n)
        dD = D - DK
        G1 = AK ** -1 * ((A0 - AK) * vec(S) - x ** 2 * (B0 * vec(V) - BK * vec(VK))
                         - dD * D ** -1 * (A0 * vec(S) - x ** 2 * B0 * vec(V)) - rho2)
        Mh = mm(pr(c, Rhat, E))
        W2 = (x * D) ** -1 * (Cm * Mh * vec(S) + x ** 2 * Mh * Ci * vec(Y))
        rec("I2 S-remainder", x * vec(dRS), Lam * (vec(RS) + G1) + x * W2 + D ** -1 * defS)
        # (I3)
        Dl = unv(x * Ci * vec(Y))
        H = {k: (pr(S, S, E)[k] + pr(Dl, Dl, E)[k]) / 2 for k in E}
        defb = {k: x * dbe[k] + (n - 1) * be[k] + pr(be, be, E)[k] - (1 - eps) * H[k] - eps * pr(be, H, E)[k]
                for k in E}
        RC = mm(Rc)
        xCKiRS = unv(x * CKi * vec(RS))
        dDl = unv(x * Ci * vec({m: RV[m] + RS[m] for m in O}) - Ci * RC * vec(Dt))
        inner = unv(x * Ci * vec(RV) - Ci * RC * vec({m: xCKiRS[m] + Dt[m] for m in O}))
        g3 = {k: pr(RS, RS, E)[k] / 2 + pr(Dt, inner, E)[k] + pr(dDl, dDl, E)[k] / 2 for k in E}
        CKitv = unv(CKi * vec(tauv))
        rho3 = {k: taub[k] + (1 - eps) * pr(bbK, pr(CKitv, {m: Dt[m] + DlK[m] for m in O}, E), E)[k] / 2 for k in E}
        xgS = Lam * G1 + x * W2
        qx = q * xgS
        xqpRS = x * qp * vec(RS)
        ghat = {k: -pr(Rb, {kk: 2 * bK[kk] + Rb[kk] for kk in E}, E)[k] + eps * pr(Rb, H, E)[k]
                + (1 - eps) * pr(bbK, g3, E)[k] - xqpRS[i] - qx[i] - rho3[k] for i, k in enumerate(E)}
        dRhat = mp.matrix([dRb[k] for k in E]) - qp * vec(RS) - q * vec(dRS)
        lhs3 = x * dRhat + (n - 1) * mp.matrix([Rhat[k] for k in E])
        qD = q * D ** -1 * defS
        rhs3 = mp.matrix([ghat[k] + defb[k] - qD[i] for i, k in enumerate(E)])
        rec("I3 b-remainder", lhs3, rhs3)
        # (I4)
        lhs4 = mp.matrix([x * dRc[k] for k in E])
        rhs4 = mp.matrix([pr(Rc, be, E)[k] + pr(cK, Rb, E)[k] - rho4[k] + (x * dc[k] - pr(c, be, E)[k]) for k in E])
        rec("I4 c-remainder", lhs4, rhs4)
    return worst


if __name__ == "__main__":
    rnd = random.Random(20261022)
    ok = True
    for M, K in ((1, 3), (1, 4), (3, 2)):
        wst = run(M, K, rnd)
        line = ", ".join(f"{k}: {mp.nstr(v, 3)}" for k, v in wst.items())
        good = all(v < mp.mpf(10) ** -40 for v in wst.values())
        ok &= good
        print(f"M = {M}, K = {K}{' (mutate=' + MUT + ')' if MUT else ''}: relative residuals {line}  "
              f"{'OK' if good else 'FAIL'}", flush=True)
    print("ALL IDENTITIES HOLD" if ok else "SOME IDENTITY FAILS")
    sys.exit(0 if ok else 1)
