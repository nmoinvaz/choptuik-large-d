#!/usr/bin/env python3
"""Lemma B8 (proof/B8_constants.md, section 3): weighted bounds of every known function in the M = 1 estimates.

For a bivariate polynomial p = sum_j eps^j p_j(x) whose terms eps^j x^m satisfy m >= m0, m even and m <= 2j + w,
Lemma W of the notes gives, for every complex x,
    |p(x)| <= theta^m0 X^w sum_j ||p_j||_1 rho^j,    X = 1 + |x|, theta = |x|/X, rho = X^2/n, eps = 1/n.
This module computes (m0, w, {j: ||p_j||_1}) rigorously (python-flint balls over a complex parameter polydisc) for:
    V^K, x V^K', S^K, x S^K', beta^K, x beta^K', c^K - 1, x c^K', gamma = (1 - c^K^2)/(1 - x^2), x gamma',
    Am = A^K - n, x A^K',  eta = A^K - n + 1 + (n-1) gamma  (the order-1 part of the fast rate, section 4),
    ne = n e = -2 - 2 i mu w + beta^K + c^K^2 beta^K/x^2  (the denominator of the decoupling q, section 6),
    rho2, x rho2', rho3t, x rho3t', rho4, x rho4'  (the residuals of b8_formal.py),
for mu = +1 and -1 (the maximum of the two norms is kept).  Structural zeros (Lemma F) are imposed as in b8_formal.

    python3 b8_atoms.py [K] [rA] [rW]
"""
import sys
from flint import acb, arb, acb_poly
import b8_formal as BF

X1 = acb_poly([0, 1])


def struct(p, jmin=None, m0=0):
    """impose structural zeros: odd x-powers, x-powers below m0, eps-orders below jmin"""
    out = {}
    for j, q in p.items():
        if jmin is not None and j < jmin:
            continue
        cs = [cf if (m >= m0 and m % 2 == 0) else acb(0) for m, cf in enumerate(q.coeffs())]
        qq = acb_poly(cs)
        if any(not (c == 0) for c in cs):
            out[j] = qq
    return out


def desc(p):
    """(m0, w, {j: ||p_j||_1}, {(j, m): |p_jm|}) with arb upper bounds"""
    m0, w, norms = BF.wnorm(p)
    coef = {}
    for j, q in p.items():
        for m, cf in enumerate(q.coeffs()):
            if not (cf == 0):
                coef[(j, m)] = arb(abs(cf).upper())
    return dict(m0=m0, w=w, norms=norms, coef=coef)


def dmax(a, b):
    """componentwise max of two descriptors (same polynomial for mu = +1, -1)"""
    if a is None:
        return b
    norms = dict(a["norms"])
    for j, v in b["norms"].items():
        norms[j] = v if j not in norms else (v if v > norms[j] else norms[j])
    coef = dict(a["coef"])
    for k, v in b["coef"].items():
        coef[k] = v if k not in coef else (v if v > coef[k] else coef[k])
    m0 = min(a["m0"], b["m0"])
    w = max(a["w"], b["w"])
    return dict(m0=m0, w=w, norms=norms, coef=coef)


def xd(p):
    return BF.bx(BF.bdx(p))


def atoms(K, rA="0", rW="0"):
    f, P, om = BF.params(rA, rW)
    T, raw = BF.truncated(K, f, P, om)
    V, S, be, c = T["V"], T["S"], T["be"], T["c"]
    one = BF.bconst(acb(1))
    x2 = {0: X1 * X1}
    c2 = BF.bmul(c, c)
    D = {}
    # gamma = (1 - c^2)/(1 - x^2), order by order (exact division; remainder checked to contain 0)
    omc2 = BF.badd(one, BF.bscale(c2, -1))
    gam = {}
    for j, q in omc2.items():
        qq, r = divmod(q, acb_poly([1, 0, -1]))
        for cf in r.coeffs():
            if not cf.contains(0):
                raise ValueError("gamma: 1 - c^2 not divisible by 1 - x^2")
        gam[j] = qq
    gam = struct(gam, 1, 0)
    D["gamma"] = desc(gam)
    D["xgamma'"] = desc(struct(xd(gam), 1, 2))
    D["x(xgamma')'"] = desc(struct(xd(xd(gam)), 1, 2))
    D["c-1"] = desc(struct(BF.badd(c, BF.bscale(one, -1)), 1, 0))
    D["xc'"] = desc(struct(xd(c), 1, 2))
    D["x(xc')'"] = desc(struct(xd(xd(c)), 1, 2))
    D["beta"] = desc(struct(be, 1, 2))
    D["xbeta'"] = desc(struct(xd(be), 1, 2))
    D["x(xbeta')'"] = desc(struct(xd(xd(be)), 1, 2))
    # beta/x^2
    bx2 = {j: acb_poly(list(q.coeffs())[2:]) for j, q in struct(be, 1, 2).items()}
    D["beta/x2"] = desc(bx2)
    D["xbeta/x2'"] = desc(struct(xd(bx2), 1, 2))
    D["x(xbeta/x2')'"] = desc(struct(xd(xd(bx2)), 1, 2))
    for key in ("V", "xV'", "x(xV')'", "S", "xS'", "x(xS')'", "S/x2", "x(S/x2)'", "x(x(S/x2)')'", "Am", "xA'",
                "x(xA')'", "eta", "ne", "rho2", "xrho2'"):
        D[key] = None
    for mu in (1, -1):
        imw = acb(0, mu) * om
        D["V"] = dmax(D["V"], desc(struct(V[mu], 0, 0)))
        D["xV'"] = dmax(D["xV'"], desc(struct(xd(V[mu]), 1, 2)))
        D["x(xV')'"] = dmax(D["x(xV')'"], desc(struct(xd(xd(V[mu])), 1, 2)))
        Sm = struct(S[mu], 1, 2)
        D["S"] = dmax(D["S"], desc(Sm))
        D["xS'"] = dmax(D["xS'"], desc(struct(xd(S[mu]), 1, 2)))
        D["x(xS')'"] = dmax(D["x(xS')'"], desc(struct(xd(xd(S[mu])), 1, 2)))
        Sx2 = {j: acb_poly(list(q.coeffs())[2:]) for j, q in Sm.items()}
        D["S/x2"] = dmax(D["S/x2"], desc(Sx2))
        D["x(S/x2)'"] = dmax(D["x(S/x2)'"], desc(struct(xd(Sx2), 1, 2)))
        D["x(x(S/x2)')'"] = dmax(D["x(x(S/x2)')'"], desc(struct(xd(xd(Sx2)), 1, 2)))
        # A^K = (n-1) c^2 b + ... = c^2 (1 - eps)/eps + c^2 beta - x^2 (1 + 2 i mu w - beta)
        A = BF.badd(BF.bshift(BF.badd(c2, BF.bscale(BF.bshift(c2, 1), -1)), -1), BF.bmul(c2, be),
                    BF.bscale(x2, -(1 + 2 * imw)), BF.bmul(x2, be))
        Am = BF.badd(A, BF.bscale(BF.bconst(acb(1), -1), -1))           # A - n  (n = eps^-1)
        Am = struct(Am, 0, 0)
        D["Am"] = dmax(D["Am"], desc(Am))
        D["xA'"] = dmax(D["xA'"], desc(struct(xd(A), 0, 2)))
        D["x(xA')'"] = dmax(D["x(xA')'"], desc(struct(xd(xd(A)), 0, 2)))
        # eta = A - n + 1 + (n - 1) gamma
        eta = BF.badd(Am, one, BF.bshift(gam, -1), BF.bscale(gam, -1))
        eta = struct(eta, 0, 0)
        D["eta"] = dmax(D["eta"], desc(eta))
        ne = BF.badd(BF.bconst(-2 - 2 * imw), be, BF.bmul(c2, bx2))
        D["ne"] = dmax(D["ne"], desc(struct(ne, 0, 0)))
        D["x(ne)'"] = dmax(D.get("x(ne)'"), desc(struct(xd(ne), 1, 2)))
        D["x(x(ne)')'"] = dmax(D.get("x(x(ne)')'"), desc(struct(xd(xd(ne)), 1, 2)))
    R = BF.residuals(K, f, P, om, T)
    r2 = None
    xr2 = None
    for mu in (1, -1):
        rr = struct(R[f"rho2_{mu:+d}"], K, 2)
        r2 = dmax(r2, desc(rr))
        xr2 = dmax(xr2, desc(struct(xd(rr), K, 2)))
    D["rho2"], D["xrho2'"] = r2, xr2
    r3 = struct(R["rho3t"], K, 2)
    D["rho3t"], D["xrho3t'"] = desc(r3), desc(struct(xd(r3), K, 2))
    r4 = struct(R["rho4"], K + 1, 2)
    D["rho4"], D["xrho4'"] = desc(r4), desc(struct(xd(r4), K + 1, 2))
    consts = dict(absw=arb(abs(om).upper()), abs1piw=arb(max(abs(1 + acb(0, 1) * om), abs(1 - acb(0, 1) * om)).upper()),
                  absf=arb(abs(f).upper()), absP=arb(abs(P).upper()),
                  c1=arb(max(abs(1 + 2 * acb(0, s) * om - 4 * P) for s in (1, -1)).upper()))
    return D, consts


def wval(d, rho):
    """sum_j ||p_j|| rho^j (an upper bound once multiplied by theta^m0 X^w)"""
    return sum((v * rho**j for j, v in d["norms"].items()), arb(0))


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rA = sys.argv[2] if len(sys.argv) > 2 else "0"
    rW = sys.argv[3] if len(sys.argv) > 3 else "0"
    D, C = atoms(K, rA, rW)
    print(f"K = {K}, polydisc rA = {rA}, rW = {rW};  |omega| <= {float(C['absw']):.6f}, |1 +- i w| <= "
          f"{float(C['abs1piw']):.6f}, c1 = max|1 +- 2iw - 2<F^2>| <= {float(C['c1']):.6f}")
    for k, d in D.items():
        ns = sorted(d["norms"].items())
        print(f"  {k:10s} m0 {d['m0']}, w {d['w']:+d}, j {ns[0][0]}..{ns[-1][0]}: "
              + ", ".join(f"[{j}] {float(v):.4g}" for j, v in ns[:5]))
