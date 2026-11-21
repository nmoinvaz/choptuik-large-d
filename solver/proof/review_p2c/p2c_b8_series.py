#!/usr/bin/env python3
"""P2 item 4 (#1061): the truncated outer series Z^K of B8 and its residuals, recomputed in python-flint balls.

Written from B8_constants.md sections 1.1-1.2 (the model and its reduced form) and 2 (the series, its gauge and Lemma F),
not from b8m_formal.py or b8m_atoms.py (not opened). M = 1: odd harmonics (-1, 1), even (-2, 0, 2), every product
projected onto its result's set. eps = 1/n, n - 1 = (1 - eps)/eps, theta-derivative D = x d/dx.

Order-by-order recursion (derived here from the reduced system; Z_0 = (2F, 0, 0, 1), Delta_0 = 2F x):
    tau_D at eps^k:   S_{k+1} = (CS)_k - sum_{j=1..k} C_j S_{k+1-j} - (C DS)_k + x D Delta_k + x Omega Delta_k - (BfrakS)_k
    beta at eps^k:    beta_{k+1} = beta_k - D beta_k - (beta beta)_k + H_k - H_{k-1} + (beta H)_{k-1}
    c at eps^k:       D c_k = (c beta)_k, c_k(1) = 0
    V:                V_k = Omega int_0^x S_k ds/s + [c_k(0) 2F]
    Delta:            (C Delta)_k = x (V_k + S_k)
with H = ([SS] + [Delta Delta])/2 (series Delta). Products of order k are Cauchy sums over orders.

Objects: odd vector = list of 2 acb, even vector = list of 3 acb, operator = list of rows, scalar = acb. A polynomial
in (eps, x) is a dict {(j, m): object}.

    python3 p2c_b8_series.py [--centre] [--mutate=atom]   # Lemma F's structural zeros and the atom sizes
"""
import sys

from flint import acb, arb, ctx

PREC = 256
ctx.prec = PREC
K = 4
ODD, EVEN = (-1, 1), (-2, 0, 2)
A_STAR, W_STAR = "2.5421480475", "0.7769940136"     # box centre of Theorem A(M = 1) (shared constant)
BOX = "2e-6"                                          # A, omega/omega_inf in centre +- 2e-6 (1 + i)
OMEGA_INF = 2 * arb.pi() / arb("1.64389375")


# ---------------------------------------------------------------- harmonic objects

def is_mat(a):
    return isinstance(a, list) and a and isinstance(a[0], list)


def zero_like(a):
    if isinstance(a, list):
        return [zero_like(b) for b in a]
    return acb(0)


def add(a, b):
    if isinstance(a, list):
        return [add(x, y) for x, y in zip(a, b)]
    return a + b


def sub(a, b):
    if isinstance(a, list):
        return [sub(x, y) for x, y in zip(a, b)]
    return a - b


def scale(a, s):
    if isinstance(a, list):
        return [scale(x, s) for x in a]
    return a * s


def harm(v):
    return ODD if len(v) == 2 else EVEN


def hp(f, g):
    """Projected harmonic product [fg] of two vectors (odd*odd -> even, even*odd -> odd, even*even -> even)."""
    hf, hg = harm(f), harm(g)
    res = EVEN if hf == hg else ODD
    out = [acb(0) for _ in res]
    for i, mu in enumerate(hf):
        for j, nu in enumerate(hg):
            s = mu + nu
            if s in res:
                out[res.index(s)] += f[i] * g[j]
    return out


def M_e(c):
    """Operator f -> [c f] on odd vectors, c even."""
    return [[hp(c, [acb(1) if i == j else acb(0) for i in range(2)])[r] for j in range(2)] for r in range(2)]


def M_o(s):
    """Operator f -> [s f] from odd to even vectors, s odd."""
    return [[hp(s, [acb(1) if i == j else acb(0) for i in range(2)])[r] for j in range(2)] for r in range(3)]


def matvec(A, v):
    return [sum((A[r][j] * v[j] for j in range(len(v))), acb(0)) for r in range(len(A))]


def matmat(A, B):
    return [[sum((A[r][k] * B[k][c] for k in range(len(B))), acb(0)) for c in range(len(B[0]))]
            for r in range(len(A))]


def eye(n):
    return [[acb(1) if i == j else acb(0) for j in range(n)] for i in range(n)]


def adj2(Cm):
    return [[Cm[1][1], -Cm[0][1]], [-Cm[1][0], Cm[0][0]]]


def det2(Cm):
    return Cm[0][0] * Cm[1][1] - Cm[0][1] * Cm[1][0]


def norm(a):
    """l1 norm of a vector, induced l1 norm (largest column sum) of an operator, modulus of a scalar (upper ends)."""
    if is_mat(a):
        return max((sum((a[r][c].abs_upper() for r in range(len(a))), arb(0)) for c in range(len(a[0]))),
                   key=lambda z: z.upper())
    if isinstance(a, list):
        return sum((z.abs_upper() for z in a), arb(0))
    return a.abs_upper()


def nmax(xs):
    """Largest upper end of a list of nonnegative balls, returned as a ball."""
    best = None
    for z in xs:
        if best is None or z.upper() > best.upper():
            best = z
    return best


# ---------------------------------------------------------------- polynomials in (eps, x)

def padd(P, Q):
    R = dict(P)
    for k, v in Q.items():
        R[k] = add(R[k], v) if k in R else v
    return R


def pneg(P):
    return {k: scale(v, acb(-1)) for k, v in P.items()}


def psub(P, Q):
    return padd(P, pneg(Q))


def pscale(P, s):
    return {k: scale(v, s) for k, v in P.items()}


def pmul(P, Q, f):
    """Bilinear product of two polynomials with object product f."""
    R = {}
    for (j1, m1), a in P.items():
        for (j2, m2), b in Q.items():
            k = (j1 + j2, m1 + m2)
            v = f(a, b)
            R[k] = add(R[k], v) if k in R else v
    return R


def pshift(P, dj=0, dm=0):
    return {(j + dj, m + dm): v for (j, m), v in P.items()}


def ptheta(P, power=1):
    """(x d/dx)^power."""
    return {(j, m): scale(v, acb(m ** power)) for (j, m), v in P.items() if m != 0}


def pconst(obj):
    return {(0, 0): obj}


def peps_factor(P, a0, a1):
    """(a0 + a1 eps) P."""
    return padd(pscale(P, acb(a0)), pshift(pscale(P, acb(a1)), dj=1))


def pnm1(P):
    """(n - 1) P = (1 - eps)/eps P; the eps^{-1} coefficient must vanish (returned separately)."""
    Q = pshift(peps_factor(P, 1, -1), dj=-1)
    low = {k: v for k, v in Q.items() if k[0] < 0}
    return {k: v for k, v in Q.items() if k[0] >= 0}, low


def pdiv_x2(P):
    """P / x^2 for P = O(x^2); the dropped coefficients are returned."""
    low = {k: v for k, v in P.items() if k[1] < 2}
    return {(j, m - 2): v for (j, m), v in P.items() if m >= 2}, low


def peval(P, eps, x, power=0):
    """Value of (x d/dx)^power P at (eps, x)."""
    out = None
    for (j, m), v in P.items():
        if power and m == 0:
            continue
        t = scale(v, (eps ** j if j else acb(1)) * (x ** m if m else acb(1)) * acb(m ** power))
        out = t if out is None else add(out, t)
    return out


def split_order(P, jmin):
    """Drop the eps^j terms with j < jmin (Lemma F's residual orders); return (kept, dropped)."""
    return ({k: v for k, v in P.items() if k[0] >= jmin}, {k: v for k, v in P.items() if k[0] < jmin})


def max_norm(P):
    vals = [norm(v) for v in P.values()]
    return nmax(vals) if vals else arb(0)


def all_contain_zero(P):
    for v in P.values():
        flat = v if not isinstance(v, list) else (sum(v, []) if is_mat(v) else v)
        for z in (flat if isinstance(flat, list) else [flat]):
            if not z.contains(acb(0)):
                return False
    return True


# ---------------------------------------------------------------- the series

def params(centre=False):
    if centre:
        A = acb(arb(A_STAR))
        wf = acb(arb(W_STAR))
    else:
        A = acb(arb(A_STAR, arb(BOX)), arb(0, arb(BOX)))
        wf = acb(arb(W_STAR, arb(BOX)), arb(0, arb(BOX)))
    om = wf * OMEGA_INF
    F = [A / 2, A / 2]
    Om = [[acb(0, -1) * om, acb(0)], [acb(0), acb(0, 1) * om]]     # diag(i mu omega), mu = -1, 1
    return {"A": A, "wf": wf, "omega": om, "F": F, "Omega": Om}


def xmul(P, Q, f):
    R = {}
    for m1, a in P.items():
        for m2, b in Q.items():
            v = f(a, b)
            R[m1 + m2] = add(R[m1 + m2], v) if (m1 + m2) in R else v
    return R


def xadd(P, Q):
    R = dict(P)
    for k, v in Q.items():
        R[k] = add(R[k], v) if k in R else v
    return R


def xscale(P, s):
    return {k: scale(v, s) for k, v in P.items()}


def xtheta(P):
    return {m: scale(v, acb(m)) for m, v in P.items() if m != 0}


def xshift(P, d):
    return {m + d: v for m, v in P.items()}


def ordsum(Fs, Gs, k, f, i0=0, i1=None):
    """sum_{i=i0..i1} F_i G_{k-i} (x-polynomials)."""
    i1 = k if i1 is None else i1
    R = {}
    for i in range(i0, i1 + 1):
        if 0 <= k - i < len(Gs) and i < len(Fs):
            R = xadd(R, xmul(Fs[i], Gs[k - i], f))
    return R


def series(par, Kmax=K, mutate=None):
    """Z_k = (V_k, S_k, beta_k, c_k) and Delta_k for k = 0..Kmax as x-polynomials {m: vector}."""
    F, Om = par["F"], par["Omega"]
    twoF = scale(F, acb(2))
    e0 = [acb(0), acb(1), acb(0)]
    zo, ze = [acb(0), acb(0)], [acb(0)] * 3
    V, S, B, C, Dl = [{0: twoF}], [{}], [{}], [{0: e0}], [{1: twoF}]
    cA = [{}]                                       # c_k = sum_m a_{k,m} (x^m - 1), the a's
    H = []
    opv = lambda Op, v: matvec(Op, v)
    for k in range(Kmax):
        # orders up to k are known
        Hk = xscale(xadd(ordsum(S, S, k, hp), ordsum(Dl, Dl, k, hp)), acb("0.5"))
        H.append(Hk)
        cb = [ordsum(C, B, i, hp) for i in range(k + 1)]                   # (c beta)_i, even
        CS_k = ordsum(C, S, k, hp)
        CDS_k = ordsum(C, [xtheta(s) for s in S], k, hp)
        BS_k = ordsum(cb, S, k, hp)
        Sn = xadd(CS_k, xscale(xadd(CDS_k, BS_k), acb(-1)))
        Sn = xadd(Sn, xshift(xtheta(Dl[k]), 1))
        Sn = xadd(Sn, xshift({m: opv(Om, v) for m, v in Dl[k].items()}, 1))
        Sn = xadd(Sn, xscale(ordsum(C, S, k + 1, hp, 1, k), acb(-1)))
        Sn = {m: v for m, v in Sn.items() if m >= 1}
        if mutate == "atom" and k + 1 == 2:
            Sn = {m: (scale(v, acb("1.001")) if m == max(Sn) else v) for m, v in Sn.items()}
        S.append(Sn)
        # beta_{k+1}
        Bn = xadd(B[k], xscale(xadd(xtheta(B[k]), ordsum(B, B, k, hp)), acb(-1)))
        Bn = xadd(Bn, Hk)
        if k >= 1:
            Bn = xadd(Bn, xscale(H[k - 1], acb(-1)))
            Bn = xadd(Bn, ordsum(B, H, k - 1, hp))
        Bn = {m: v for m, v in Bn.items() if m >= 1}
        B.append(Bn)
        # c_{k+1}: D c = (c beta)_{k+1}, c(1) = 0
        rhs = xadd(Bn, ordsum(C, B, k + 1, hp, 1, k))
        a = {m: scale(v, acb(1) / m) for m, v in rhs.items()}
        cn = dict(a)
        cn[0] = scale(sum_objs(list(a.values()), ze), acb(-1))
        C.append(cn)
        cA.append(a)
        # V_{k+1}
        Vn = {m: opv(Om, scale(v, acb(1) / m)) for m, v in Sn.items()}
        Vn = xadd(Vn, {0: hp(cn[0], twoF)})
        V.append(Vn)
        # Delta_{k+1}
        Dn = xshift(xadd(Vn, Sn), 1)
        Dn = xadd(Dn, xscale(ordsum(C, Dl, k + 1, hp, 1, k + 1), acb(-1)))
        Dl.append(Dn)
    return {"V": V, "S": S, "beta": B, "c": C, "Delta": Dl, "cA": cA}


def sum_objs(objs, z):
    out = z
    for o in objs:
        out = add(out, o)
    return out


def to_poly(lst, Kmax=K):
    return {(k, m): v for k in range(Kmax + 1) for m, v in lst[k].items()}


# ---------------------------------------------------------------- atoms and residuals

def atoms(par, mutate=None, Kmax=K):
    """The (eps, x) polynomials of Z^K, the operators and the residuals; structural zeros dropped and recorded."""
    ser = series(par, Kmax, mutate)
    Om = par["Omega"]
    V, S, beta, c, Dl = (to_poly(ser[k], Kmax) for k in ("V", "S", "beta", "c", "Delta"))
    Y = padd(V, S)
    e0 = [acb(0), acb(1), acb(0)]
    cm1 = psub(c, pconst(e0))
    Cop = {k: M_e(v) for k, v in c.items()}
    cb = pmul(c, beta, hp)
    Bfr = {k: M_e(v) for k, v in cb.items()}
    detC = {}
    for (j1, m1), A1 in Cop.items():
        for (j2, m2), A2 in Cop.items():
            kk = (j1 + j2, m1 + m2)
            v = A1[0][0] * A2[1][1] - A1[0][1] * A2[1][0]
            detC[kk] = detC[kk] + v if kk in detC else v
    adjC = {k: adj2(v) for k, v in Cop.items()}
    opmul = lambda A_, B_: matmat(A_, B_)
    opv = lambda A_, v: matvec(A_, v)
    drops = {}
    # gamma_c = (c^K - 1)/(x^2 - 1), exactly from c_k = sum a_{k,m}(x^m - 1)
    gc = {}
    for k in range(1, Kmax + 1):
        for m, a in ser["cA"][k].items():
            for i in range(0, m - 1, 2):
                gc[(k, i)] = add(gc[(k, i)], a) if (k, i) in gc else a
    Gam = pmul({k: M_e(v) for k, v in gc.items()}, padd(Cop, pconst(eye(2))), opmul)
    # residuals at Z^K
    DS = ptheta(S)
    tauD = padd(pmul(Cop, DS, opv), pneg(pshift(ptheta(Dl), dm=1)))
    tauD = padd(tauD, pneg(pshift({k: opv(Om, v) for k, v in Dl.items()}, dm=1)))
    tauD = padd(tauD, pmul(Bfr, S, opv))
    nCS, low = pnm1(pmul(Cop, S, opv))
    drops["tauD_nm1"] = low
    tauD = padd(tauD, nCS)
    tauv = psub(pmul(Cop, Dl, opv), pshift(Y, dm=1))
    HK = pscale(padd(pmul(S, S, hp), pmul(Dl, Dl, hp)), acb("0.5"))
    nb, low = pnm1(beta)
    drops["taub_nm1"] = low
    taub = padd(padd(ptheta(beta), nb), pmul(beta, beta, hp))
    taub = psub(taub, peps_factor(HK, 1, -1))
    taub = psub(taub, pshift(pmul(beta, HK, hp), dj=1))
    rho4 = psub(ptheta(c), cb)
    tauD, drops["tauD"] = split_order(tauD, Kmax)
    taub, drops["taub"] = split_order(taub, Kmax)
    tauv, drops["tauv"] = split_order(tauv, Kmax + 1)
    rho4, drops["rho4"] = split_order(rho4, Kmax + 1)
    if mutate == "tauD10":
        tauD = pscale(tauD, acb(10))
    # rho_2 det C = x^2 [rho4 (adjC Y)] - det C C tau_D - det C x D tau_v + x [(D c)(adjC tau_v)] - x C Omega adjC tau_v
    adjY = pmul(adjC, Y, opv)
    adjtv = pmul(adjC, tauv, opv)
    sc = lambda s, v: scale(v, s)
    r2 = pshift(pmul(rho4, adjY, hp), dm=2)
    if mutate == "drop_rho2":
        r2 = {}
    r2 = padd(r2, pneg(pmul(detC, pmul(Cop, tauD, opv), sc)))
    r2 = padd(r2, pneg(pshift(pmul(detC, ptheta(tauv), sc), dm=1)))
    r2 = padd(r2, pshift(pmul(ptheta(c), adjtv, hp), dm=1))
    COm = {k: matmat(v, Om) for k, v in Cop.items()}
    r2 = padd(r2, pneg(pshift(pmul(COm, adjtv, opv), dm=1)))
    # bK1 = (1 - eps) b^K = (1 - eps) + eps beta^K
    bK1 = padd(peps_factor(pconst(e0), 1, -1), pshift(beta, dj=1))
    # rho_3 det C^2 = det C^2 tau_b + 1/2 [bK1 [(adjC tau_v)(x adjC Y + det C Delta^K)]]
    det2C = pmul(detC, detC, lambda a_, b_: a_ * b_)
    inner = padd(pshift(adjY, dm=1), pmul(detC, Dl, sc))
    r3 = padd(pmul(det2C, taub, sc), pscale(pmul(bK1, pmul(adjtv, inner, hp), hp), acb("0.5")))
    # eta~ det C = det C [(n - 1) Gamma + C Bfrak/x^2 - Omega - I] - (C Omega - Bfrak) adjC
    nGam, low = pnm1(Gam)
    drops["Gamma_nm1"] = low
    cb_x2, low = pdiv_x2(cb)
    drops["cb_x2"] = low
    Bfr_x2 = {k: M_e(v) for k, v in cb_x2.items()}
    T = padd(nGam, pmul(Cop, Bfr_x2, opmul))
    T = psub(T, pconst(Om))
    T = psub(T, pconst(eye(2)))
    eta_num = psub(pmul(detC, T, sc), pmul(psub(COm, Bfr), adjC, opmul))
    # l^ det C^2 f = [bK1 (det C^2 [(S/x^2) f] + [(adjC Y)(adjC f)])]
    S_x2, low = pdiv_x2(S)
    drops["S_x2"] = low
    part1 = pmul(det2C, {k: M_o(v) for k, v in S_x2.items()}, sc)
    part2 = pmul({k: M_o(v) for k, v in adjY.items()}, adjC, opmul)
    lhat = pmul({k: M_ee(v) for k, v in bK1.items()}, padd(part1, part2), opmul)
    return {"par": par, "ser": ser, "V": V, "S": S, "Y": Y, "Delta": Dl, "beta": beta, "c": c, "cm1": cm1,
            "C": Cop, "adjC": adjC, "detCm1": psub(detC, pconst(acb(1))), "Bfr": Bfr, "Bfr_x2": Bfr_x2,
            "Gam": Gam, "tauD": tauD, "tauv": tauv, "taub": taub, "rho4": rho4, "rho2n": r2, "rho3n": r3,
            "etan": eta_num, "lhatn": lhat, "S_x2": S_x2, "bK1": bK1, "Dtn": pshift(adjY, dm=1), "adjY": adjY,
            "Dc": ptheta(c), "drops": drops}


def M_ee(a):
    """Operator b -> [a b] on even vectors, a even."""
    return [[hp(a, [acb(1) if i == j else acb(0) for i in range(3)])[r] for j in range(3)] for r in range(3)]


def check_structure(at, centre):
    """Lemma F: the dropped coefficients contain 0 (over the box) and are at rounding level (at the centre)."""
    ok = True
    rep = []
    for name, P in at["drops"].items():
        if not P:
            rep.append((name, 0, "none"))
            continue
        cz = all_contain_zero(P)
        mx = max_norm(P)
        good = cz and (not centre or mx.upper() < arb("1e-60"))
        ok = ok and good
        rep.append((name, len(P), f"{'ok' if good else 'FAIL'} contain0={cz} max={mx.upper().str(3, radius=False)}"))
    return ok, rep


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:]
            if a.startswith("--")}
    centre = "centre" in opts
    at = atoms(params(centre), mutate=opts.get("mutate"))
    ok, rep = check_structure(at, centre)
    print(f"# p2c_b8_series: K = {K}, prec = {PREC}, {'centre' if centre else 'box'}, mutate = {opts.get('mutate')}")
    for name, n, s in rep:
        print(f"  dropped {name:10s} {n:3d} terms: {s}")
    for name in ("V", "S", "beta", "cm1", "Gam", "tauD", "taub", "tauv", "rho4"):
        P = at[name]
        js = sorted({j for j, _ in P})
        row = []
        for j in js:
            s = sum((norm(v) for (jj, _), v in P.items() if jj == j), arb(0))
            row.append(f"j={j}: {s.upper().str(4, radius=False)}")
        deg = max((m - 2 * j for j, m in P), default=None)
        print(f"  {name:5s} max(m - 2j) = {deg}; sum_m |coef|: " + ", ".join(row))
    print("STRUCTURE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
