#!/usr/bin/env python3
"""Lemma B8 (proof/B8_constants.md, section 2): the formal 1/n solution of the one-harmonic (M = 1) outer system in the
variables of b8_system.py, in ball arithmetic over a complex parameter polydisc, and the residuals of its truncations.

Variables (mu = +-1; b8_system.py checks the equations):
    x V_mu' = i mu w S_mu
    x (x^2 - c^2) S_mu' = [(n-1) c^2 b - x^2 (1 + 2 i mu w - beta)] S_mu - (1 + i mu w - beta) x^2 V_mu      (E2)
    x beta' + (n-1) beta + beta^2 = (1 - eps) b H,   H = S_+ S_- + x^2 Y_+ Y_- / c^2,  Y = V + S              (E3)
    x c' = c beta                                                                                               (E4)
with (n-1) b = (n-1) + beta.  Free data: F_+ = F_- = f (the scaling V_+ -> t V_+, V_- -> V_-/t is a symmetry, so only
P = F_+ F_- = <F^2>/2 = A^2/4 matters), omega, and the gauge c_k(1) = 0 (the light cone at x = 1 to all orders) with
V(0) = 2 f c(0) (no K_k corrections; an n-dependent F is covered by letting f vary in the polydisc).
Each order Z_k (k >= 1) is an even polynomial of degree <= 2k; S_k, beta_k are O(x^2).  The truncation
Z^[K] = sum_{k<=K} eps^k Z_k leaves residuals
    rho2 = x (x^2 - c^2) S' - A S + B x^2 V                          = sum_{j>=K} eps^j rho2_j(x)
    rho3t = c^2 [x beta' + (n-1) beta + beta^2] - (1-eps+eps beta)(c^2 S_+S_- + x^2 Y_+Y_-)   = sum_{j>=K} eps^j ...
    rho4 = x c' - c beta                                              = sum_{j>=K+1} eps^j ...
(all at Z^[K]); degree of eps^j part <= 2j + 2 (rho2, rho3t) and <= 2j (rho4).

A bivariate polynomial sum_j eps^j p_j(x) is stored as {j: acb_poly}.  Weighted bound used throughout (proved in the
notes, Lemma W): if m0 <= deg_x-terms and every x^m eps^j term has m <= 2j + w, then for all complex x,
    |p(x)| <= theta^m0 X^w sum_j ||p_j||_1 rho^j,   X = 1 + |x|, theta = |x|/X, rho = X^2/n.

    python3 b8_formal.py [K] [--check]   (--check: cross-check against solver/largeD_late_orders.py at the centre)
"""
import os
import sys
from flint import acb, arb, acb_poly, arb_poly

sys.setrecursionlimit(10000)
HERE = os.path.dirname(os.path.abspath(__file__))
W0 = 2 * arb.pi() / arb("1.64389375")
A_STAR, WF_STAR = arb("2.5421480475"), arb("0.7769940136")   # Theorem A(M = 1) zero (A, omega/omega_inf)


def params(rA="0", rW="0"):
    """(f, P, omega) as complex balls: A in the complex disc |A - A*| <= rA, omega/omega_inf in |.| <= rW."""
    A = acb(arb(A_STAR.mid(), arb(rA)), arb(0, arb(rA)))
    om = acb(W0 * WF_STAR) + acb(arb(0, arb(rW)), arb(0, arb(rW))) * W0
    f = A / 2
    return f, f * f, om


# ------------------------------------------------------------------ bivariate polynomials {j: acb_poly}
def bp(d):
    return {j: p for j, p in d.items() if p.degree() >= 0}


def badd(*ps):
    out = {}
    for p in ps:
        for j, q in p.items():
            out[j] = out[j] + q if j in out else q
    return bp(out)


def bscale(p, s):
    return bp({j: q * s for j, q in p.items()})


def bmul(p, q):
    out = {}
    for i, a in p.items():
        for j, b in q.items():
            out[i + j] = out[i + j] + a * b if i + j in out else a * b
    return bp(out)


def bdx(p):
    return bp({j: q.derivative() for j, q in p.items()})


def bshift(p, k):            # eps^k p
    return {j + k: q for j, q in p.items()}


X1 = acb_poly([0, 1])


def bx(p, k=1):              # x^k p
    return bp({j: q * X1 ** k for j, q in p.items()})


def bconst(c, j=0):
    return {j: acb_poly([c])}


def series_to_bp(lst, K):
    return bp({k: lst[k] for k in range(min(K + 1, len(lst)))})


def wnorm(p):
    """(m0, w, {j: ||p_j||_1 as arb upper bound}) for the weighted bound theta^m0 X^w sum_j ||p_j|| rho^j"""
    m0, w, norms = None, None, {}
    for j, q in p.items():
        cs = q.coeffs()
        nz = [m for m, c in enumerate(cs) if not (c == 0)]
        if not nz:
            continue
        lo, hi = min(nz), max(nz)
        m0 = lo if m0 is None else min(m0, lo)
        w = hi - 2 * j if w is None else max(w, hi - 2 * j)
        s = arb(0)
        for c in cs:
            s += abs(c)
        norms[j] = arb(s.upper())
    return m0, w, norms


# ------------------------------------------------------------------ the formal recursion
def ser_mul(a, b, k):
    """[a b]_k for series given as lists of acb_poly (missing entries are 0)"""
    acc = acb_poly([])
    for i in range(k + 1):
        if i < len(a) and k - i < len(b):
            acc += a[i] * b[k - i]
    return acc


def int_over_x(p):
    """int_0^x p(s)/s ds for a polynomial with p(0) = 0 (checked)"""
    cs = p.coeffs()
    if cs and not (cs[0] == 0):
        if not cs[0].contains(0):
            raise ValueError("int_over_x: constant term not zero")
    return acb_poly([acb(0)] + [cs[m] / m for m in range(1, len(cs))])


def formal(K, f, P, om):
    """orders 0..K of (V_+, V_-, S_+, S_-, beta, c) as lists of acb_poly"""
    one = acb_poly([1])
    x2 = X1 * X1
    V = {mu: [acb_poly([2 * f])] for mu in (1, -1)}
    S = {mu: [acb_poly([])] for mu in (1, -1)}
    be = [acb_poly([])]
    c = [one]
    for k in range(1, K + 1):
        c2 = [ser_mul(c, c, j) for j in range(k + 1)]
        c2m = [c2[j] - (c2[j - 1] if j >= 1 else acb_poly([])) for j in range(k + 1)]   # c^2 (1 - eps)
        for mu in (1, -1):
            imw = acb(0, mu) * om
            Sx = [s.derivative() for s in S[mu]]
            # [x (x^2 - c^2) S']_{k-1}
            t1 = X1 * (x2 * Sx[k - 1] - ser_mul(c2, Sx, k - 1))
            t2 = ser_mul(c2m, S[mu], k)          # [c^2 (1-eps) S]_k without the S_k term (S has orders < k here)
            t3 = ser_mul(ser_mul_list(c2, be, k), S[mu], k - 1)
            Sk = (t1 - t2 - t3 + x2 * (1 + 2 * imw) * S[mu][k - 1] - x2 * ser_mul(be, S[mu], k - 1)
                  + (1 + imw) * x2 * V[mu][k - 1] - x2 * ser_mul(be, V[mu], k - 1))
            S[mu].append(Sk)
        # beta_k = beta_{k-1} - x beta'_{k-1} - [beta^2]_{k-1} + H_{k-1} - H_{k-2} + [beta H]_{k-2}
        H = H_series(S, V, c, k - 1)
        bk = (be[k - 1] - X1 * be[k - 1].derivative() - ser_mul(be, be, k - 1) + H[k - 1]
              - (H[k - 2] if k >= 2 else acb_poly([])) + (ser_mul(be, H, k - 2) if k >= 2 else acb_poly([])))
        be.append(bk)
        cb = ser_mul(c, be, k)
        ck = int_over_x(cb)
        ck = ck - acb_poly([ck(acb(1))])               # c_k(1) = 0
        c.append(ck)
        for mu in (1, -1):
            imw = acb(0, mu) * om
            V[mu].append(imw * int_over_x(S[mu][k]) + acb_poly([2 * f * ck(acb(0))]))
    return V, S, be, c


def ser_mul_list(a, b, k):
    return [ser_mul(a, b, j) for j in range(k + 1)]


def inv_series(a, k):
    out = [acb_poly([1]) * (1 / a[0](acb(0)))] if a[0].degree() == 0 else None
    if out is None:
        raise ValueError
    for j in range(1, k + 1):
        acc = acb_poly([])
        for i in range(1, j + 1):
            if i < len(a):
                acc += a[i] * out[j - i]
        out.append(-acc * out[0])
    return out


def H_series(S, V, c, kmax):
    """H = S_+ S_- + x^2 Y_+ Y_- / c^2, orders 0..kmax"""
    Y = {mu: [V[mu][j] + (S[mu][j] if j < len(S[mu]) else acb_poly([])) for j in range(len(V[mu]))] for mu in (1, -1)}
    c2 = [ser_mul(c, c, j) for j in range(kmax + 1)]
    ic2 = inv_series(c2, kmax)
    YY = [ser_mul(Y[1], Y[-1], j) for j in range(kmax + 1)]
    out = []
    for j in range(kmax + 1):
        out.append(ser_mul(S[1], S[-1], j) + X1 * X1 * ser_mul(YY, ic2, j))
    return out


# ------------------------------------------------------------------ truncations as bivariate polynomials, residuals
def truncated(K, f, P, om):
    V, S, be, c = formal(K, f, P, om)
    T = {"V": {mu: series_to_bp(V[mu], K) for mu in (1, -1)},
         "S": {mu: series_to_bp(S[mu], K) for mu in (1, -1)},
         "be": series_to_bp(be, K), "c": series_to_bp(c, K)}
    return T, (V, S, be, c)


def residuals(K, f, P, om, T=None):
    if T is None:
        T, _ = truncated(K, f, P, om)
    V, S, be, c = T["V"], T["S"], T["be"], T["c"]
    x2 = {0: X1 * X1}
    c2 = bmul(c, c)
    one = bconst(acb(1))
    out = {}
    for mu in (1, -1):
        imw = acb(0, mu) * om
        # (n-1) c^2 b = c^2 (1 - eps)/eps + c^2 beta
        nm1c2b = badd(bshift(badd(c2, bscale(bshift(c2, 1), -1)), -1), bmul(c2, be))
        Acoef = badd(nm1c2b, bscale(x2, -(1 + 2 * imw)), bmul(x2, be))
        Bcoef = badd(bconst(1 + imw), bscale(be, -1))
        r2 = badd(bmul(bx(badd(x2, bscale(c2, -1))), bdx(S[mu])), bscale(bmul(Acoef, S[mu]), -1),
                  bmul(bmul(Bcoef, x2), V[mu]))
        out[f"rho2_{mu:+d}"] = r2
    Y = {mu: badd(V[mu], S[mu]) for mu in (1, -1)}
    lhs = badd(bx(bdx(be)), bshift(badd(be, bscale(bshift(be, 1), -1)), -1), bmul(be, be))
    rhs = bmul(badd(one, bscale(bshift(one, 1), -1), bshift(be, 1)),
               badd(bmul(c2, bmul(S[1], S[-1])), bmul(x2, bmul(Y[1], Y[-1]))))
    out["rho3t"] = badd(bmul(c2, lhs), bscale(rhs, -1))
    out["rho4"] = badd(bx(bdx(c)), bscale(bmul(c, be), -1))
    return out


def structural(p, jmin, m0):
    """impose the structural zeros proved in the notes (Lemma F): no eps^j with j < jmin, no x^m with m < m0, no odd m.
    Ball arithmetic over a parameter polydisc cannot see these exact cancellations.  Returns (cleaned, largest |removed|)
    so that the caller can confirm at the centre (point parameters) that the removed entries are rounding noise."""
    out, removed = {}, arb(0)
    for j, q in p.items():
        cs = list(q.coeffs())
        keep = []
        for m, cf in enumerate(cs):
            if j < jmin or m < m0 or m % 2 == 1:
                u = arb(abs(cf).upper())
                removed = u if u > removed else removed
                keep.append(acb(0))
            else:
                keep.append(cf)
        qq = acb_poly(keep)
        if qq.degree() >= 0 and any(not (c == 0) for c in keep):
            out[j] = qq
    return out, arb(removed.upper())


STRUCT = {"rho2": (None, 2), "rho3t": (None, 2), "rho4": (None, 2)}   # jmin set from K below


def clean_res(name, r, K):
    jmin = K + 1 if name.startswith("rho4") else K
    return structural(r, jmin, 2)


def report(K, f, P, om, verbose=True):
    T, raw = truncated(K, f, P, om)
    V, S, be, c = raw
    res = {}
    if verbose:
        print(f"K = {K}: degrees of Z_k (V_+, S_+, beta, c):",
              [(V[1][k].degree(), S[1][k].degree(), be[k].degree(), c[k].degree()) for k in range(K + 1)])
    R = residuals(K, f, P, om, T)
    for name, r in R.items():
        r, rem = clean_res(name, r, K)
        m0, w, norms = wnorm(r)
        jmin = min(norms) if norms else None
        res[name] = (m0, w, norms, rem)
        if verbose:
            print(f"  {name:8s}: removed <= {float(rem):.1e}; eps-orders {jmin}..{max(norms) if norms else None}, lowest x-power {m0}, weight {w}; "
                  + ", ".join(f"||.||_{j} = {float(v):.4g}" for j, v in sorted(norms.items())[:4]))
    return T, raw, res


def crosscheck(K=4):
    """compare with solver/largeD_late_orders.py (independent recursion in (S, D, G, B), theta grid), M = 1, centre"""
    import numpy as np
    sys.path.insert(0, os.path.join(HERE, ".."))
    import largeD_late_orders as L
    f, P, om = params()
    A = float(2 * f.real.mid())
    omf = float(om.real.mid())
    th = 2 * np.pi * np.arange(L.NT) / L.NT
    Sg, Dg, Gg, Bg, _ = L.run(A * np.sin(th), omf, K, verbose=False, jmax_of=lambda k: 1)
    V, S, be, c = formal(K, f, P, om)
    # map: F = A sin(theta) has F_+ = A/(2i) = -i A/2; our normalisation F_+ = f = A/2 differs by the phase -i
    # (a theta-shift), so harmonic +1 of the grid fields equals -i x (ours, mu = +1).
    worst = 0.0
    for xv in (0.3, 0.8, 1.7):
        for k in range(1, K + 1):
            h = lambda arr: np.fft.fft(L.at(arr, xv))[1] / L.NT
            Sk = complex(S[1][k](acb(xv)).mid()) * (-1j)
            ck = complex(c[k](acb(xv)).mid())
            Dk = h(Dg[k])
            # Delta = x (V + S)/c: compare the full truncated Delta at this order via the series relation is longer;
            # check S_k and c_k = -G_k directly, and beta_k against B: beta = (n-1)(b-1) = sum eps^k (B_{k+1} - B_k)
            Sg_k = h(Sg[k])
            Gk = np.fft.fft(L.at(Gg[k], xv))[0].real / L.NT
            bk = complex(be[k](acb(xv)).mid())
            Bk1 = np.fft.fft(L.at(Bg[k + 1], xv))[0].real / L.NT
            Bk = np.fft.fft(L.at(Bg[k], xv))[0].real / L.NT if k >= 2 else 0.0
            e1 = abs(Sk - Sg_k) / max(1e-300, abs(Sg_k))
            e2 = abs(ck + Gk) / max(1e-300, abs(Gk))
            e3 = abs(bk - (Bk1 - Bk)) / max(1e-300, abs(Bk1 - Bk))
            worst = max(worst, e1, e2, e3)
            print(f"  x {xv:3.1f} k {k}: rel diff S_k {e1:.1e}  c_k vs -G_k {e2:.1e}  beta_k vs B_k+1 - B_k {e3:.1e}")
    print(f"crosscheck: worst relative difference {worst:.2e}")
    return worst


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 3
    f, P, om = params()
    print(f"centre: f = A/2 = {f.real}, omega = {om.real}")
    report(K, f, P, om)
    f, P, om = params("0.02", "0.005")
    print(f"polydisc |A - A*| <= 0.02, |omega/omega_inf - w*| <= 0.005 (complex):")
    report(K, f, P, om)
    if "--check" in sys.argv:
        w = crosscheck(max(K, 3))
        sys.exit(0 if w < 1e-9 else 1)
