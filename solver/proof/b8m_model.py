#!/usr/bin/env python3
"""Lemma B8 (proof/B8_constants.md, section 1): the reduced form of the M-harmonic outer model with the projection
conventions of the certified inner problem (Theorem A(M = 1): p, m on odd |mu| <= M, b, c on even |nu| <= M + 1, every
binary product projected onto its result's harmonics; G p = [[c(b-1)] p], b (p^2 + m^2) = [b [p^2 + m^2]]), the model of
solver/proof/b5_system.py:
    x c_x = (n-1)[c(b-1)],   x b_x = [b[p^2+m^2]] - (n-1)[b(b-1)],
    x^2 p_x - x[c p_x] = -w x p_th + (n-1)[[c(b-1)] p] - [c p] + n/2 [c(p+m)],
    x^2 m_x + x[c m_x] = -w x m_th - (n-1)[[c(b-1)] m] + [c m] - n/2 [c(p+m)].
Reduced variables (p = eps^(1/2) P etc.): S = P + M, Delta = P - M, V = ([c Delta] - x S)/x, beta = (n-1)(b-1).
With the operators C f = [c f] (on odd harmonics), Bb f = [[c beta] f], Om = diag(i mu w):
    x V'              = Om S                                                        (exact, any M)
    x (x^2 - C^2) S'  = A S - x^2 B V,   A = (n-1) C^2 + C Bb - x^2 Om - x^2 B,   B = I - Bb C^-1 + C Om C^-1
    x beta' + (n-1) beta + [beta beta] = (1-eps)(H + [beta H]/(n-1)),   H = ([S S] + [Delta Delta])/2
    x c'              = [c beta],          Delta = x C^-1 (V + S).
Checked numerically (mpmath, 50 digits) at random complex points for M = 1, 3, 5: the derivatives of (V, S, beta, c)
computed from the model by the chain rule equal the reduced right-hand side.  Exits non-zero on failure.
--mutate=proj uses [c(b-1) p] (one projection of the triple product) in the p-equation; for M = 1 the two conventions
coincide (reported), for M >= 3 the check must fail.  --mutate=sign flips the sign of Om in the reduced S-equation.

    python3 b8m_model.py [--mutate=proj|sign]
"""
import random
import sys
import mpmath as mp

mp.mp.dps = 50
MUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), "")


def harm(M):
    return [m for m in range(-M, M + 1) if m % 2], [k for k in range(-(M + 1), M + 2) if k % 2 == 0]


def proj(A, B, keep):
    out = {k: mp.mpc(0) for k in keep}
    for i, a in A.items():
        for j, b in B.items():
            if i + j in out:
                out[i + j] += a * b
    return out


def full(A, B):
    out = {}
    for i, a in A.items():
        for j, b in B.items():
            out[i + j] = out.get(i + j, mp.mpc(0)) + a * b
    return out


def mult_matrix(g, O):
    return mp.matrix([[g.get(mu - nu, mp.mpc(0)) for nu in O] for mu in O])


def vec(d, O):
    return mp.matrix([d[m] for m in O])


def unvec(v, O):
    return {m: v[i] for i, m in enumerate(O)}


def model_rhs(M, x, n, w, pf, mf, bf, cf, mutate=""):
    O, E = harm(M)
    bm1 = {k: bf[k] - (1 if k == 0 else 0) for k in E}
    G = {k: (n - 1) * v for k, v in proj(cf, bm1, E).items()}
    p2m2 = {k: proj(pf, pf, E)[k] + proj(mf, mf, E)[k] for k in E}
    dc = {k: G[k] / x for k in E}
    db = {k: (proj(bf, p2m2, E)[k] - (n - 1) * proj(bf, bm1, E)[k]) / x for k in E}
    Cm = mult_matrix(cf, O)
    if mutate == "proj":
        G3 = {k: (n - 1) * v for k, v in full(cf, bm1).items()}
        Gp, Gm = proj(G3, pf, O), proj(G3, mf, O)
    else:
        Gp, Gm = proj(G, pf, O), proj(G, mf, O)
    cp, cm = proj(cf, pf, O), proj(cf, mf, O)
    cpm = proj(cf, {m: pf[m] + mf[m] for m in O}, O)
    rp = mp.matrix([-w * x * 1j * m * pf[m] + Gp[m] - cp[m] + n / 2 * cpm[m] for m in O])
    rm = mp.matrix([-w * x * 1j * m * mf[m] - Gm[m] + cm[m] - n / 2 * cpm[m] for m in O])
    I = mp.eye(len(O))
    dp = mp.lu_solve(x * x * I - x * Cm, rp)
    dm = mp.lu_solve(x * x * I + x * Cm, rm)
    return unvec(dp, O), unvec(dm, O), db, dc


def reduced_rhs(M, x, n, w, V, S, be, c, mutate=""):
    O, E = harm(M)
    eps = 1 / n
    Cm = mult_matrix(c, O)
    Bb = mult_matrix(proj(c, be, E), O)
    sg = -1 if mutate == "sign" else 1
    Om = mp.diag([sg * 1j * m * w for m in O])
    I = mp.eye(len(O))
    Ci = Cm ** -1
    B = I - Bb * Ci + Cm * Om * Ci
    A = (n - 1) * Cm * Cm + Cm * Bb - x**2 * Om - x**2 * B
    Sv, Vv = vec(S, O), vec(V, O)
    dS = mp.lu_solve(x * (x**2 * I - Cm * Cm), A * Sv - x**2 * B * Vv)
    dV = Om * Sv / x
    D = unvec(x * Ci * (Vv + Sv), O)
    H = {k: (proj(S, S, E)[k] + proj(D, D, E)[k]) / 2 for k in E}
    bH, bb = proj(be, H, E), proj(be, be, E)
    dbe = {k: (-(n - 1) * be[k] - bb[k] + (1 - eps) * (H[k] + bH[k] / (n - 1))) / x for k in E}
    dc = {k: proj(c, be, E)[k] / x for k in E}
    return unvec(dV, O), unvec(dS, O), dbe, dc


def check_M(M, npts=3, mutate=""):
    O, E = harm(M)
    rnd = random.Random(1234 + M)
    rc = lambda a: mp.mpc(rnd.uniform(-a, a), rnd.uniform(-a, a))
    worst = mp.mpf(0)
    for _ in range(npts):
        x = mp.mpc(rnd.uniform(0.3, 1.7), rnd.uniform(0.1, 0.9))
        n = mp.mpf(rnd.uniform(20, 60))
        w = mp.mpf(rnd.uniform(1, 3))
        pf = {m: rc(0.3) for m in O}
        mf = {m: rc(0.3) for m in O}
        bf = {k: (1 if k == 0 else 0) + rc(0.05) for k in E}
        cf = {k: (1 if k == 0 else 0) + rc(0.05) for k in E}
        dp, dm, db, dc = model_rhs(M, x, n, w, pf, mf, bf, cf, mutate if mutate == "proj" else "")
        sq = mp.sqrt(n)
        S = {m: sq * (pf[m] + mf[m]) for m in O}
        Dl = {m: sq * (pf[m] - mf[m]) for m in O}
        dS = {m: sq * (dp[m] + dm[m]) for m in O}
        dD = {m: sq * (dp[m] - dm[m]) for m in O}
        cD = proj(cf, Dl, O)
        V = {m: (cD[m] - x * S[m]) / x for m in O}
        d_cD = {m: proj(dc, Dl, O)[m] + proj(cf, dD, O)[m] for m in O}
        dV = {m: (d_cD[m] - S[m] - x * dS[m]) / x - V[m] / x for m in O}
        be = {k: (n - 1) * (bf[k] - (1 if k == 0 else 0)) for k in E}
        dbe = {k: (n - 1) * db[k] for k in E}
        rV, rS, rbe, rc_ = reduced_rhs(M, x, n, w, V, S, be, cf, mutate if mutate == "sign" else "")
        for m in O:
            worst = max(worst, abs(rV[m] - dV[m]), abs(rS[m] - dS[m]))
        for k in E:
            worst = max(worst, abs(rbe[k] - dbe[k]), abs(rc_[k] - dc[k]))
    return worst


if __name__ == "__main__":
    ok = True
    for M in (1, 3, 5):
        wst = check_M(M, mutate=MUT)
        good = wst < mp.mpf("1e-40")
        print(f"M = {M}{' (mutate=' + MUT + ')' if MUT else ''}: max |reduced - model| = {mp.nstr(wst, 3)}  "
              f"{'OK' if good else 'FAIL'}")
        ok &= good
    print("ALL OK" if ok else "SOME CHECK FAILED")
    sys.exit(0 if ok else 1)
