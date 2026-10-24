#!/usr/bin/env python3
"""P2 review: descriptors of the truncated 1/n series of the M = 1 model OVER THE COMPLEX PARAMETER BOX of Lemma LC
(A, omega/omega_inf within 2e-6 (1 + i) of the centre), for the independent Lemma LC check (p2_lc.py).

Independent of b8m_formal.py / b8m_atoms.py: the recursion is that of p2_formal.py (derived from the original p, m,
b, c equations), run in complex rectangular interval arithmetic (mpmath.iv, p2_common.C). For a vector (or operator) of
bivariate polynomials sum c_jm eps^j x^m the descriptor is the table of upper bounds of sum over components |c_jm|
(operators: per (j, m) the largest column sum), with m0 = min m, w = max (m - 2j). Lemma W of B8_constants.md then
gives |p(x)| <= theta^m0 X^w sum_j (sum_m |c_jm|) rho^j.

Structural zeros (Lemma F of B8_constants.md: the residuals of the truncated series are O(eps^K) for tau_D, tau_b and
O(eps^(K+1)) for tau_v, rho4, identically in the parameters, because every order of the recursion is solved exactly)
are imposed: the interval entries at those orders contain 0 and are dropped. At the box centre, in 60-digit point
arithmetic (p2_formal.py), they vanish to 1e-50 (checked here). Everything else (parity, vanishing orders at x = 0) is
produced exactly by the recursion.
"""
from mpmath import iv, mp, mpf, pi

from p2_common import C, hi, lo

ODD, EVEN = (-1, 1), (-2, 0, 2)
ZERO = C(0, 0)


def cpoly_add(*ps):
    out = {}
    for p in ps:
        for m, v in p.items():
            out[m] = out[m] + v if m in out else v
    return out


def cpoly_mul(p, q):
    out = {}
    for a, u in p.items():
        for b, v in q.items():
            out[a + b] = out[a + b] + u * v if a + b in out else u * v
    return out


def cpoly_sc(p, s):
    return {m: v * s for m, v in p.items()}


def cpoly_der(p):
    return {m - 1: v * m for m, v in p.items() if m != 0}


def cpoly_x(p, j=1):
    return {m + j: v for m, v in p.items()}


def vadd(*vs):
    out = {}
    for v in vs:
        for h, p in v.items():
            out[h] = cpoly_add(out.get(h, {}), p)
    return out


def vsc(v, s):
    return {h: cpoly_sc(p, s) for h, p in v.items()}


def proj(f, g, keep):
    out = {h: {} for h in keep}
    for a, p in f.items():
        for b, q in g.items():
            if a + b in out:
                out[a + b] = cpoly_add(out[a + b], cpoly_mul(p, q))
    return out


def vder(v):
    return {h: cpoly_der(p) for h, p in v.items()}


def vx(v, j=1):
    return {h: cpoly_x(p, j) for h, p in v.items()}


def zero(keep):
    return {h: {} for h in keep}


def sprod(f, g, k, keep):
    out = zero(keep)
    for i in range(k + 1):
        if i < len(f) and k - i < len(g):
            out = vadd(out, proj(f[i], g[k - i], keep))
    return out


def params(rA="2e-6", rW="2e-6"):
    rA, rW = iv.mpf(rA), iv.mpf(rW)
    A = C(iv.mpf("2.5421480475") + iv.mpf([-hi(rA), hi(rA)]), iv.mpf([-hi(rA), hi(rA)]))
    w_inf = 2 * iv.pi / iv.mpf("1.64389375")
    wf = C(iv.mpf("0.7769940136") + iv.mpf([-hi(rW), hi(rW)]), iv.mpf([-hi(rW), hi(rW)]))
    w = wf * C(w_inf, 0)
    return A, w


def formal_iv(order, A, w):
    """the recursion of p2_formal.formal in complex interval arithmetic"""
    half = iv.mpf("0.5")
    F = {1: A * C(half, 0), -1: A * C(half, 0)}
    Om = {m: C(0, m) * w for m in ODD}
    S = [zero(ODD)]
    Dl = [{m: {1: F[m] * C(2, 0)} for m in ODD}]
    be = [zero(EVEN)]
    c = [{0: {0: C(1, 0)}, 2: {}, -2: {}}]
    H = []
    for k in range(1, order + 1):
        cb = [sprod(c, be, j, EVEN) for j in range(k)]
        Sx = [vder(s) for s in S]
        t = vadd(vx(vder(Dl[k - 1]), 2), vsc(vx(sprod(c, Sx, k - 1, ODD)), C(-1, 0)),
                 {m: cpoly_sc(cpoly_x(Dl[k - 1][m]), Om[m]) for m in ODD},
                 vsc(sprod(cb, S, k - 1, ODD), C(-1, 0)), sprod(c, S, k - 1, ODD))
        for j in range(1, k):
            t = vadd(t, vsc(proj(c[j], S[k - j], ODD), C(-1, 0)))
        S.append(t)
        H.append(vsc(vadd(sprod(S, S, k - 1, EVEN), sprod(Dl, Dl, k - 1, EVEN)), C(half, 0)))
        bk = vadd(be[k - 1], vsc(vx(vder(be[k - 1])), C(-1, 0)), vsc(sprod(be, be, k - 1, EVEN), C(-1, 0)), H[k - 1])
        if k >= 2:
            bk = vadd(bk, vsc(H[k - 2], C(-1, 0)), sprod(be, H, k - 2, EVEN))
        be.append(bk)
        cbk = sprod(c, be, k, EVEN)
        ck = {}
        for h in EVEN:
            p = cbk[h]
            if 0 in p:
                raise ValueError("[c beta] has a constant term (structurally impossible)")
            q = {m: v * C(iv.mpf(1) / m, 0) for m, v in p.items()}
            s = ZERO
            for v in q.values():
                s = s + v
            q[0] = ZERO - s
            ck[h] = q
        c.append(ck)
        cbfull = [sprod(c, be, j, EVEN) for j in range(k + 1)]
        rhs = vadd(vsc(vx(vder(S[k]), 2), C(-1, 0)), {m: cpoly_sc(cpoly_x(S[k][m]), ZERO - Om[m]) for m in ODD},
                   sprod(cbfull, Dl + [zero(ODD)], k, ODD))
        for j in range(1, k + 1):
            rhs = vadd(rhs, vsc(proj(c[j], Dl[k - j], ODD), C(-1, 0)), vx(proj(c[j], vder(Dl[k - j]), ODD)))
        dk = {}
        for m in ODD:
            # the x^1 coefficient is structurally zero (it cancels exactly; p2_formal checks it at the centre)
            dk[m] = {p: v * C(iv.mpf(1) / (1 - p), 0) for p, v in rhs[m].items() if p != 1}
        Dl.append(dk)
    V = []
    for k in range(order + 1):
        cD = sprod(c, Dl, k, ODD)
        num = {m: cpoly_add(cD[m], cpoly_sc(cpoly_x(S[k][m]), C(-1, 0))) for m in ODD}
        V.append({m: {p - 1: v for p, v in num[m].items() if p != 0} for m in ODD})
    return dict(S=S, Dl=Dl, be=be, c=c, V=V, F=F, Om=Om)


# ---------------------------------------------------------------- bivariate objects {(j, m): C}
def tv(ser, K, keep):
    out = {h: {} for h in keep}
    for k in range(min(K + 1, len(ser))):
        for h, p in ser[k].items():
            for m, v in p.items():
                key = (k, m)
                out[h][key] = out[h][key] + v if key in out[h] else v
    return out


def badd(*ps):
    out = {}
    for p in ps:
        for key, v in p.items():
            out[key] = out[key] + v if key in out else v
    return out


def bmul(p, q):
    out = {}
    for (j1, m1), u in p.items():
        for (j2, m2), v in q.items():
            key = (j1 + j2, m1 + m2)
            out[key] = out[key] + u * v if key in out else u * v
    return out


def bsc(p, s):
    return {k: v * s for k, v in p.items()}


def bxd(p):
    return {(j, m): v * m for (j, m), v in p.items() if m != 0}


def bx(p, i=1):
    return {(j, m + i): v for (j, m), v in p.items()}


def beps(p, i=1):
    return {(j + i, m): v for (j, m), v in p.items()}


def bvproj(f, g, keep):
    out = {h: {} for h in keep}
    for a, p in f.items():
        for b, q in g.items():
            if a + b in out:
                out[a + b] = badd(out[a + b], bmul(p, q))
    return out


def bvadd(*vs):
    out = {}
    for v in vs:
        for h, p in v.items():
            out[h] = badd(out.get(h, {}), p)
    return out


def bvsc(v, s):
    return {h: bsc(p, s) for h, p in v.items()}


NM1 = {(-1, 0): C(1, 0), (0, 0): C(-1, 0)}


def drop_low(v, jmin):
    """impose the structural zeros of Lemma F: entries with j < jmin removed; returns (cleaned, max |removed|)"""
    out, rem = {}, mp.mpf(0)
    for h, p in v.items():
        q = {}
        for (j, m), u in p.items():
            if j < jmin:
                rem = max(rem, u.absup())
            else:
                q[(j, m)] = u
        out[h] = q
    return out, rem


def desc(entries):
    """descriptor of the l1 sum of a list of bivariate polynomials"""
    coef = {}
    for p in entries:
        for key, u in p.items():
            a = u.absup()
            if a > 0:
                coef[key] = coef.get(key, mp.mpf(0)) + a
    if not coef:
        return None
    return dict(m0=min(m for _, m in coef), w=max(m - 2 * j for j, m in coef), coef=coef,
                jmin=min(j for j, _ in coef))


def mdesc(Mx):
    """operators on odd vectors: per (j, m) the largest column sum"""
    cols = {}
    for (r, cc), p in Mx.items():
        for key, u in p.items():
            a = u.absup()
            if a > 0:
                cols.setdefault(cc, {})
                cols[cc][key] = cols[cc].get(key, mp.mpf(0)) + a
    coef = {}
    for d in cols.values():
        for key, a in d.items():
            coef[key] = max(coef.get(key, mp.mpf(0)), a)
    if not coef:
        return None
    return dict(m0=min(m for _, m in coef), w=max(m - 2 * j for j, m in coef), coef=coef,
                jmin=min(j for j, _ in coef))


def mult_op(g):
    return {(mu, nu): g.get(mu - nu, {}) for mu in ODD for nu in ODD}


def mat_mul(A, B):
    out = {}
    for (r, k1), a in A.items():
        for (k2, cc), b in B.items():
            if k1 == k2:
                out[(r, cc)] = badd(out.get((r, cc), {}), bmul(a, b))
    return out


def mat_add(*Ms):
    out = {}
    for Mx in Ms:
        for k, p in Mx.items():
            out[k] = badd(out.get(k, {}), p)
    return out


def mat_sc(Mx, s):
    return {k: bsc(p, s) for k, p in Mx.items()}


def div_x2m1(p):
    """quotient of the exact division by x^2 - 1 (per eps-order), synthetic division from the top"""
    out = {}
    for j in sorted({j for j, _ in p}):
        coefs = {m: v for (jj, m), v in p.items() if jj == j}
        deg = max(coefs)
        a = {m: coefs.get(m, ZERO) for m in range(deg + 1)}
        for m in range(deg, 1, -1):
            qm = a[m]
            out[(j, m - 2)] = qm
            a[m - 2] = a[m - 2] + qm
    return out


def with_derivs(name, v, D, entries=lambda v: list(v.values()), deriv=2):
    """descriptor of v, x v' and x (x v')' (vectors)"""
    D[name] = desc(entries(v))
    d1 = {h: bxd(p) for h, p in v.items()}
    D["x" + name + "'"] = desc(entries(d1))
    if deriv >= 2:
        d2 = {h: bxd(p) for h, p in d1.items()}
        D["x(x" + name + "')'"] = desc(entries(d2))
    return v


def atoms(K=4, rA="2e-6", rW="2e-6"):
    A, w = params(rA, rW)
    fs = formal_iv(K + 1, A, w)
    Om = fs["Om"]
    St, Dt, bt, ct, Vt = (tv(fs[nm], K, keep) for nm, keep in (("S", ODD), ("Dl", ODD), ("be", EVEN), ("c", EVEN),
                                                                 ("V", ODD)))
    D = {}
    rems = {}
    Yt = bvadd(Vt, St)
    cm1 = {h: dict(p) for h, p in ct.items()}
    cm1[0] = badd(cm1[0], {(0, 0): C(-1, 0)})
    cm1[0] = {k: v for k, v in cm1[0].items() if not (k == (0, 0))}      # exact: c_0 = 1
    cb = bvproj(ct, bt, EVEN)
    with_derivs("V", Vt, D)
    with_derivs("S", St, D)
    with_derivs("S/x2", {h: {(j, m - 2): u for (j, m), u in p.items()} for h, p in St.items()}, D)
    with_derivs("beta", bt, D)
    with_derivs("beta/x2", {h: {(j, m - 2): u for (j, m), u in p.items()} for h, p in bt.items()}, D)
    with_derivs("c-1", cm1, D)
    with_derivs("cb", cb, D)
    with_derivs("cb/x2", {h: {(j, m - 2): u for (j, m), u in p.items()} for h, p in cb.items()}, D)
    with_derivs("Y", Yt, D)
    with_derivs("Dl", Dt, D)
    # operators
    Cop = mult_op(ct)
    I2 = {(mu, nu): ({(0, 0): C(1, 0)} if mu == nu else {}) for mu in ODD for nu in ODD}
    C2mI = mat_add(mat_mul(Cop, Cop), mat_sc(I2, C(-1, 0)))
    Gam = {k: div_x2m1(p) for k, p in C2mI.items()}
    D["Gamma"] = mdesc(Gam)
    xG = {k: bxd(p) for k, p in Gam.items()}
    D["xGamma'"] = mdesc(xG)
    D["x(xGamma')'"] = mdesc({k: bxd(p) for k, p in xG.items()})
    det = badd(bmul(Cop[(1, 1)], Cop[(-1, -1)]), bsc(bmul(Cop[(1, -1)], Cop[(-1, 1)]), C(-1, 0)))
    detm1 = {k: v for k, v in badd(det, {(0, 0): C(-1, 0)}).items() if k != (0, 0)}
    D["det-1"] = desc([detm1])
    D["xdet'"] = desc([bxd(detm1)])
    D["x(xdet')'"] = desc([bxd(bxd(detm1))])
    adj = {(1, 1): Cop[(-1, -1)], (-1, -1): Cop[(1, 1)], (1, -1): bsc(Cop[(1, -1)], C(-1, 0)),
           (-1, 1): bsc(Cop[(-1, 1)], C(-1, 0))}
    cbx2 = {h: {(j, m - 2): u for (j, m), u in p.items()} for h, p in cb.items()}
    Mcb, Mcbx2 = mult_op(cb), mult_op(cbx2)
    OmM = {(mu, nu): ({(0, 0): Om[mu]} if mu == nu else {}) for mu in ODD for nu in ODD}
    detI = {(mu, nu): (det if mu == nu else {}) for mu in ODD for nu in ODD}
    t1 = {k: bmul(bmul(NM1, det), v) for k, v in Gam.items()}
    t2 = {k: bmul(det, v) for k, v in mat_mul(Cop, Mcbx2).items()}
    t3 = mat_sc(mat_mul(detI, mat_add(OmM, I2)), C(-1, 0))
    t4 = mat_mul(mat_add(Mcb, mat_sc(mat_mul(Cop, OmM), C(-1, 0))), adj)
    Neta = mat_add(t1, t2, t3, t4)
    # (n - 1) det Gamma: the eps^-1 part vanishes structurally (Gamma = O(eps)); entries with j < 0 are removed
    Neta = {k: {kk: v for kk, v in p.items() if kk[0] >= 0} for k, p in Neta.items()}
    D["Neta"] = mdesc(Neta)
    xN = {k: bxd(p) for k, p in Neta.items()}
    D["xNeta'"] = mdesc(xN)
    D["x(xNeta')'"] = mdesc({k: bxd(p) for k, p in xN.items()})
    # residuals of Z^K
    tauD = bvadd(bvproj(ct, {m: bxd(p) for m, p in St.items()}, ODD), {m: bsc(bx(bxd(Dt[m])), C(-1, 0)) for m in ODD},
                 {m: bsc(bx(Dt[m]), ZERO - Om[m]) for m in ODD}, bvproj(cb, St, ODD),
                 {m: bmul(NM1, bvproj(ct, St, ODD)[m]) for m in ODD})
    tauD, rems["tauD"] = drop_low(tauD, K)
    with_derivs("tauD", tauD, D)
    tauv = bvadd(bvproj(ct, Dt, ODD), {m: bsc(bx(Yt[m]), C(-1, 0)) for m in ODD})
    tauv, rems["tauv"] = drop_low(tauv, K + 1)
    with_derivs("tauv", tauv, D)
    xtauvp = {h: bxd(p) for h, p in tauv.items()}
    D["xtauv'"] = desc(list(xtauvp.values()))
    D["x(xtauv')'"] = desc([bxd(p) for p in xtauvp.values()])
    Hb = bvsc(bvadd(bvproj(St, St, EVEN), bvproj(Dt, Dt, EVEN)), C(iv.mpf("0.5"), 0))
    bH = bvproj(bt, Hb, EVEN)
    taub = bvadd({h: bxd(p) for h, p in bt.items()}, {h: bmul(NM1, p) for h, p in bt.items()}, bvproj(bt, bt, EVEN),
                 bvsc(Hb, C(-1, 0)), {h: beps(p) for h, p in Hb.items()}, {h: bsc(beps(p), C(-1, 0)) for h, p in bH.items()})
    taub, rems["taub"] = drop_low(taub, K)
    with_derivs("taub", taub, D)
    rho4 = bvadd({h: bxd(p) for h, p in ct.items()}, bvsc(cb, C(-1, 0)))
    rho4, rems["rho4"] = drop_low(rho4, K + 1)
    with_derivs("rho4", rho4, D)
    xc = {h: bxd(p) for h, p in cm1.items()}
    D["xc-1'"] = desc(list(xc.values()))
    D["x(xc-1')'"] = desc([bxd(p) for p in xc.values()])
    consts = dict(absw=w.absup(), absf=max(fs["F"][m].absup() for m in ODD))
    return D, consts, rems


def jsum(d):
    out = {}
    for (j, m), v in d["coef"].items():
        out[j] = out.get(j, 0) + v
    return out


if __name__ == "__main__":
    import sys
    rA = sys.argv[1] if len(sys.argv) > 1 else "2e-6"
    D, cs, rems = atoms(4, rA, rA)
    print(f"atoms over the box (rA = rW = {rA}); |omega| <= {mp.nstr(cs['absw'], 8)}, max|F| <= {mp.nstr(cs['absf'], 8)}")
    print("removed structural entries (box width; Lemma F): " + ", ".join(f"{k} {mp.nstr(v, 3)}" for k, v in rems.items()))
    for k, d in D.items():
        if d is None:
            print(f"  {k:12s} (zero)")
            continue
        js = sorted(jsum(d).items())
        print(f"  {k:12s} m0 {d['m0']}, w {d['w']:+d}: " + ", ".join(f"[{j}] {mp.nstr(v, 5)}" for j, v in js[:4]))
