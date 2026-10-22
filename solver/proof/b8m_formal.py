#!/usr/bin/env python3
"""Lemma B8 (proof/B8_constants.md, section 2): the formal 1/n solution of the M-harmonic model of b8m_model.py
(p, m on odd |mu| <= M; b, c on even |nu| <= M + 1; products projected), in the variables (V, S, beta, c) plus the
auxiliary Delta, in ball arithmetic over a complex parameter polydisc.

Recursion (eps = 1/n; Z = sum_k eps^k Z_k; [.] = projected product; Om = diag(i mu w); F the profile):
    S_k = -sum_{i=1}^k [c_i S_{k-i}] + [c S]_{k-1} + x^2 Delta'_{k-1} + x Om Delta_{k-1} - [[c beta] S]_{k-1}
          - x [c S']_{k-1}                                    (from x[c S'] = x^2 Delta' + x Om Delta - [[c beta]S] - (n-1)[c S])
    beta_k = beta_{k-1} - x beta'_{k-1} - [beta beta]_{k-1} + H_{k-1} - H_{k-2} + [beta H]_{k-2},  H = ([SS] + [Delta Delta])/2
    x c_k' = [c beta]_k,  c_k(1) = 0 on every harmonic (the light cone at x = 1 to all orders)
    V_k = Om int_0^x S_k/s ds + [c_k(0) 2F],   Delta_k = x (V_k + S_k) - sum_{i=1}^k [c_i Delta_{k-i}]
with V_0 = 2F, Delta_0 = 2 x F, S_0 = beta_0 = 0, c_0 = 1 (gauge: V(0) = [c(0) 2F], i.e. Delta'(0) = 2F exactly).
Each Z_k (k >= 1) is a polynomial of degree <= 2k (Delta_k: odd, <= 2k+1); S_k, beta_k = O(x^2).

Checks (independent of the recursion):
  --check  (1) the residual of the ORIGINAL projected equations (b8m_model.model_rhs, p, m, b, c form) at the truncated
               series, at fixed x and n = 1e3, 1e4, 1e5: its size relative to the leading terms must fall like n^-(K+1)
               (the fitted exponent is printed and must be within 0.15 of K + 1);
           (2) the top-degree coefficients (x^{2k+1} of Delta_k, x^{2k} of S_k and c_k) equal the certified inner
               series of series_arb.py (Theorem A(M = 1)), which shares no code with this file.
  --mutate=coef multiplies the Om Delta term of the S-recursion by 1.001: both checks must fail.

    python3 b8m_formal.py [M] [K] [--check] [--mutate=coef]
"""
import os
import sys
from flint import acb, arb, acb_poly, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), "")
def W0():
    return 2 * arb.pi() / arb("1.64389375")


A_STAR_S, WF_STAR_S = "2.5421480475", "0.7769940136"       # parsed at the working precision when used
X1 = acb_poly([0, 1])
Z0 = acb_poly([])


def harm(M):
    return [m for m in range(-M, M + 1) if m % 2], [k for k in range(-(M + 1), M + 2) if k % 2 == 0]


def params(rA="0", rW="0", M=1, F=None):
    """(F dict, omega): F = A sin(theta) normalised by the scaling symmetry to F_+1 = F_-1 = A/2 for M = 1 (other
    harmonics of F zero unless given); A in |A - A*| <= rA, omega/omega_inf in |. - w*| <= rW (complex discs)"""
    A = acb(arb(arb(A_STAR_S).mid(), arb(rA)), arb(0, arb(rA)))
    om = acb(W0() * arb(WF_STAR_S)) + acb(arb(0, arb(rW)), arb(0, arb(rW))) * W0()
    O, _ = harm(M)
    if F is None:
        F = {m: (A / 2 if abs(m) == 1 else acb(0)) for m in O}
    return F, om


def pmul(A, B, keep):
    out = {k: Z0 for k in keep}
    for i, a in A.items():
        for j, b in B.items():
            if i + j in out:
                out[i + j] = out[i + j] + a * b
    return out


def padd(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out[k] + v if k in out else v
    return out


def pscale(d, s):
    return {k: v * s for k, v in d.items()}


def pder(d):
    return {k: v.derivative() for k, v in d.items()}


def pmulx(d, j=1):
    return {k: v * X1 ** j for k, v in d.items()}


def int_over_x(p):
    cs = p.coeffs()
    if cs and not cs[0].contains(0):
        raise ValueError("int_over_x: constant term not zero")
    return acb_poly([acb(0)] + [cs[m] / m for m in range(1, len(cs))])


def ser(lst, k, keep, other=None, prod=False):
    """[a b]_k for series of harmonic dicts"""
    a, b = lst, other
    acc = {kk: Z0 for kk in keep}
    for i in range(k + 1):
        if i < len(a) and k - i < len(b):
            acc = padd(acc, pmul(a[i], b[k - i], keep))
    return acc


def formal(K, F, om, M=1, mutate="", gauge0_from=None):
    """orders 0..K; with gauge0_from = K0 the orders k > K0 use the gauge c_k(0) = 0 instead of c_k(1) = 0, which is the
    asymptotic expansion of the solution normalized by c(0) = c^K0(0) (b8m_numcheck.py, check A1)"""
    O, E = harm(M)
    Om = {m: acb(0, m) * om for m in O}
    V = [{m: acb_poly([2 * F[m]]) for m in O}]
    S = [{m: Z0 for m in O}]
    Dl = [{m: X1 * (2 * F[m]) for m in O}]
    be = [{k: Z0 for k in E}]
    c = [{k: (acb_poly([1]) if k == 0 else Z0) for k in E}]
    H = []
    fac = acb(1) + (acb("0.001") if mutate == "coef" else 0)
    for k in range(1, K + 1):
        # S_k
        t = {m: Z0 for m in O}
        for i in range(1, k + 1):
            if i < len(c) and k - i < len(S):
                t = padd(t, pscale(pmul(c[i], S[k - i], O), -1))
        cS = ser(c, k - 1, O, S)
        cb = [ser(c, j, E, be) for j in range(k)]
        cbS = ser(cb, k - 1, O, S)
        Sx = [pder(s) for s in S]
        cSx = ser(c, k - 1, O, Sx)
        Dlx = pder(Dl[k - 1])
        Sk = padd(t, cS, pmulx(Dlx, 2), {m: X1 * Dl[k - 1][m] * Om[m] * fac for m in O}, pscale(cbS, -1),
                  pscale(pmulx(cSx), -1))
        S.append(Sk)
        # H_{k-1}
        H.append({kk: (ser(S, k - 1, E, S)[kk] + ser(Dl, k - 1, E, Dl)[kk]) * acb("0.5") for kk in E})
        bk = padd(be[k - 1], pscale(pmulx(pder(be[k - 1])), -1), pscale(ser(be, k - 1, E, be), -1), H[k - 1])
        if k >= 2:
            bk = padd(bk, pscale(H[k - 2], -1), ser(be, k - 2, E, H))
        be.append(bk)
        cbk = ser(c, k, E, be)
        ck = {}
        for kk in E:
            p = int_over_x(cbk[kk])
            ck[kk] = p if (gauge0_from is not None and k > gauge0_from) else p - acb_poly([p(acb(1))])
        c.append(ck)
        c0 = {kk: acb_poly([ck[kk](acb(0))]) for kk in E}
        Vk = padd({m: int_over_x(Sk[m]) * Om[m] for m in O}, pmul(c0, {m: acb_poly([2 * F[m]]) for m in O}, O))
        V.append(Vk)
        Dk = pmulx(padd(Vk, Sk))
        for i in range(1, k + 1):
            Dk = padd(Dk, pscale(pmul(c[i], Dl[k - i], O), -1))
        Dl.append(Dk)
    return dict(V=V, S=S, Dl=Dl, be=be, c=c, O=O, E=E, Om=Om)


# ------------------------------------------------------------------ checks
def check_model_residual(K, M=1, xs=("0.6+0.3j", "1.7+0.9j"), ns=(1e3, 1e4, 1e5), mutate=""):
    import mpmath as mp
    import b8m_model as BM
    mp.mp.dps = 60
    ctx.prec = 300
    F, om = params(M=M)
    fs = formal(K, F, om, M, mutate)
    O, E = fs["O"], fs["E"]
    to_mp = lambda z: mp.mpc(mp.mpf(z.real.mid().str(80, radius=False)), mp.mpf(z.imag.mid().str(80, radius=False)))
    w = to_mp(om)

    def ev(series, comp, x, n, der=False):
        tot = mp.mpc(0)
        for k in range(len(series)):
            p = series[k][comp]
            cs = [to_mp(cc) for cc in p.coeffs()]
            if der:
                v = sum(m * cs[m] * x ** (m - 1) for m in range(1, len(cs)))
            else:
                v = sum(cs[m] * x ** m for m in range(len(cs)))
            tot += v / mp.mpf(n) ** k
        return tot

    slopes = []
    for xs_ in xs:
        x = mp.mpc(complex(xs_))
        rel = []
        for n in ns:
            n = mp.mpf(n)
            S = {m: ev(fs["S"], m, x, n) for m in O}
            Dl = {m: ev(fs["Dl"], m, x, n) for m in O}
            dS = {m: ev(fs["S"], m, x, n, True) for m in O}
            dDl = {m: ev(fs["Dl"], m, x, n, True) for m in O}
            be = {k: ev(fs["be"], k, x, n) for k in E}
            dbe = {k: ev(fs["be"], k, x, n, True) for k in E}
            c = {k: ev(fs["c"], k, x, n) for k in E}
            dc = {k: ev(fs["c"], k, x, n, True) for k in E}
            h = 1 / mp.sqrt(n)
            pf = {m: h * (S[m] + Dl[m]) / 2 for m in O}
            mf = {m: h * (S[m] - Dl[m]) / 2 for m in O}
            bf = {k: (1 if k == 0 else 0) + be[k] / (n - 1) for k in E}
            dpf = {m: h * (dS[m] + dDl[m]) / 2 for m in O}
            dmf = {m: h * (dS[m] - dDl[m]) / 2 for m in O}
            dbf = {k: dbe[k] / (n - 1) for k in E}
            rp, rm, rb, rc = BM.model_residual(M, x, n, w, pf, mf, bf, c, dpf, dmf, dbf, dc)
            # residuals of (p)+(m) (no n-terms) and of the c equation: O(eps^(K+1)) relative to their terms;
            # of (p)-(m) (contains n [c S]) and of the b equation ((n-1)[b beta]): O(eps^K)
            scale = max(abs(x * x * h * dDl[m]) for m in O)
            rs = max([abs(rp[m] + rm[m]) / scale for m in O] + [abs(rc[k]) for k in E])
            rd = max([abs(rp[m] - rm[m]) / scale for m in O] + [abs(rb[k]) * n / (1 + abs(dbe[0])) for k in E])
            rel.append((rs, rd))
        for j, name, tgt in ((0, "sum + c   ", K + 1), (1, "diff + b  ", K)):
            sl = [float(mp.log(rel[i][j] / rel[i + 1][j]) / mp.log(mp.mpf(ns[i + 1]) / ns[i]))
                  for i in range(len(ns) - 1)]
            slopes.append((sl[-1], tgt))
            print(f"  x = {xs_}, {name}: relative residual at n = {list(ns)}: "
                  + ", ".join(f"{float(v[j]):.2e}" for v in rel) + f"; decay exponents "
                  + ", ".join(f"{v:.3f}" for v in sl) + f" (expected {tgt})")
    return slopes


def check_inner(K, M=1, mutate=""):
    from series_arb import SeriesArb
    ctx.prec = 256
    A = arb(A_STAR_S)
    F = {1: acb(0, -1) * A / 2, -1: acb(0, 1) * A / 2}
    om = acb(W0() * arb(WF_STAR_S))
    fs = formal(K, F, om, M, mutate)
    sa = SeriesArb(F, om, M=1, prec=256)
    d, s, g = sa.series(K + 1)
    worst = 0.0
    for k in range(1, K + 1):
        for m in fs["O"]:
            a = fs["Dl"][k][m].coeffs()[2 * k + 1]
            worst = max(worst, float(abs(a - d[k][m]).upper()) / float(abs(d[k][m]).upper()))
            a = fs["S"][k][m].coeffs()[2 * k]
            worst = max(worst, float(abs(a - s[k - 1][m]).upper()) / float(abs(s[k - 1][m]).upper()))
        for kk in fs["E"]:
            cs = fs["c"][k][kk].coeffs()
            if len(cs) > 2 * k:
                ref = g[k][kk]
                den = float(abs(ref).upper())
                if den > 1e-30:
                    worst = max(worst, float(abs(cs[2 * k] - ref).upper()) / den)
    print(f"  top-degree coefficients vs the certified inner series (k <= {K}): worst relative difference {worst:.2e}")
    return worst


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    M = int(pos[0]) if pos else 1
    K = int(pos[1]) if len(pos) > 1 else 3
    ctx.prec = 256
    F, om = params(M=M)
    fs = formal(K, F, om, M)
    print(f"M = {M}, K = {K}: degrees (V, S, beta, c, Delta) of the orders:",
          [(max(p.degree() for p in fs[n_][k].values()) for n_ in ("V", "S", "be", "c", "Dl")) for k in range(K + 1)]
          and [tuple(max(p.degree() for p in fs[n_][k].values()) for n_ in ("V", "S", "be", "c", "Dl"))
               for k in range(K + 1)])
    if "--check" in sys.argv:
        ok = True
        if M == 1:
            wi = check_inner(K, M, MUT)
            ok &= wi < 1e-30
        sl = check_model_residual(K, M, mutate=MUT)
        ok &= all(abs(s - t) < 0.15 for s, t in sl)
        print("CHECKS OK" if ok else "CHECK FAILED")
        sys.exit(0 if ok else 1)
