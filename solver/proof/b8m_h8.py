#!/usr/bin/env python3
"""Lemma B8 -> hypothesis (H8)(ii) of Lemma B5(1) (proof/B5_inner.md, sections 4 and 7), M = 1.

On the inner arc |x| = n^gamma, 0 <= arg x <= alpha = pi/4 + kappa/2 (z = x^2/n, |z| = r_n = n^(2 gamma - 1)), with
W = Y_out - Z_in in B5's variables (P, Phi, C, theta), Lemma 1 of B5:
    P = C^-1 (V + S),  Phi = x P' = C^-1 (x Y' - [x c' P])  (Y = V + S),  C = c,
    theta = Theta - [P P]/2,  Theta = beta / ((1 - eps) z)            (eps = 1/n; V, S, beta, c of b8m_model.py),
this script proves, for every harmonic and every n >= n_0 (every constant a python-flint ball),
    |W_P| <= d_P / n,  |W_Phi| <= d_Phi n^(-2 + 2 gamma),  |W_C| <= d_C / n,  |W_theta| <= d_theta / n,
by W = (Y_out - T) + (T - Z_in), T the outer series truncated at eps^K and converted:

 (R) Y_out - T.  The bootstrap hypothesis constants h (read from a b8m_scan.py log line) are re-verified here,
     P(h) < h, with the atoms over the parameter box (so Lemma B8 holds with |R_V| <= h_V rho^(K+1), ...), and the
     converted differences are bounded in b8m_bounds' algebra on the arc region "A" of Geo(arc=True) (X = 1 + n^gamma):
        W_P = C^-1 (R_V + R_S - [R_c P^K]),   W_Phi = x W_P'  (product rule, |x R_S'|, |x R_c'| from the equations),
        W_C = R_c,   W_theta = R_beta / ((1 - eps) z) - [W_P (2 P^K + W_P)]/2,   R_beta = R^_beta + q R_S.
     A term c X^a n^e of the arc bound gives n^s c X^a n^e <= c (1 + n_0^-gamma)^max(a,0) n_0^(gamma a + e + s) for
     every n >= n_0 when gamma a + e + s <= 0 (otherwise the script stops: the bound is not of the required order).

 (T) T - Z_in.  T is a function of (eps, z): each eps^k x^m of the formal series is eps^(k - m/2) z^(m/2), and on the
     arc |eps^i z^j| = n^-wt, wt = i + (1 - 2 gamma) j.  T is computed as a Taylor model in (eps, z): the monomials of
     weight <= W_cut exactly (balls), the rest as a remainder |R| <= rem (n_0/n)^W' (W' > W_cut) on the arc; products
     are projected harmonic convolutions (l1 norms), C^-1 is a Neumann series with its tail, 1/(1 - eps) a geometric
     series with its tail.  The eps^0 part of T agrees with the inner formal series up to z^K for P, Phi, C and up to
     z^(K-1) for theta (Lemma F(iii) of the notes: the top-degree outer coefficients are the inner series, checked
     here against series_arb.py at the box centre); those terms are removed.  Z_in = Y_N + eta (lemma S, N = 31
     terms, as rerun by b5_bounds.py over B5's box): the inner coefficients K < j <= 30 are subtracted, and lemma S's
     eta is bounded by B5's certified number |eta| <= 5.5e-7 (|z|/0.0085)^12 on |z| <= 0.0085 (b5_bounds_k0.3_Z0.0085.log;
     the weights of fixed_point.py's norm are |z|^k for p, v and |z|^(k+1) for c, k = 12), times a factor 100 for Phi
     (Phi = v + [(C-1)(v + (1+s)p)] + [eta_c Q] + ..., |1+s| <= 5, |Q| <= 14 on the arc).

    python3 b8m_h8.py K L0 --log=b8m_scan_....log [--rA=2e-6 --rW=2e-6] [--W=2.6] [--gamma=0.3]
           [--centre]  (validation of the structural cancellation at the box centre only)
    exit status 0 iff (H8)(ii) is proved with delta <= delta_ref = (1, 200, 10, 20).
"""
import json
import sys
from fractions import Fraction as Fr
from flint import arb, acb, ctx
import b8_alg as E
import b8m_atoms as AT
import b8m_bounds as MB
import b8m_formal as FM
from series_arb import SeriesArb

OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
up, lo = E.up, E.lo
DREF = dict(P=arb(1), Phi=arb(200), C=arb(10), th=arb(20))
CT = {}


# ---------------------------------------------------------------------------------------------- (T) Taylor models
def wt(i, j):
    return i + CT["s"] * j


def nsc(w):
    """n_0^-w for a rational weight w"""
    return (-(arb(w.numerator) / w.denominator) * CT["L0"]).exp()


class TM:
    """sum t[(i,j)][h] eps^i z^j + R, ||R||_1 <= rem (n_0/n)^W' with W' > W_cut, on the arc, n >= n_0"""

    def __init__(self, t=None, rem=None):
        self.t = t if t is not None else {}
        self.rem = arb(0) if rem is None else rem

    def maj(self):
        s = arb(0)
        for k, v in self.t.items():
            s += sum((abs(c) for c in v.values()), arb(0)) * nsc(wt(*k))
        return up(s)

    def minwt(self):
        ws = [wt(*k) for k, v in self.t.items() if any(not c.is_zero() for c in v.values())]
        if self.rem > 0:
            ws.append(CT["W"])
        return min(ws)


def put(t, k, h, c):
    d = t.setdefault(k, {})
    d[h] = d.get(h, acb(0)) + c


def tm_add(A, B, sb=1):
    t = {k: dict(v) for k, v in A.t.items()}
    for k, v in B.t.items():
        for h, c in v.items():
            put(t, k, h, sb * c)
    return TM(t, up(A.rem + B.rem))


def tm_scale(A, s):
    return TM({k: {h: c * s for h, c in v.items()} for k, v in A.t.items()}, up(A.rem * abs(acb(s))))


def tm_mul(A, B, keep):
    t, rem = {}, arb(0)
    for k1, v1 in A.t.items():
        for k2, v2 in B.t.items():
            k = (k1[0] + k2[0], k1[1] + k2[1])
            prod = {}
            for h1, c1 in v1.items():
                for h2, c2 in v2.items():
                    if h1 + h2 in keep:
                        prod[h1 + h2] = prod.get(h1 + h2, acb(0)) + c1 * c2
            if wt(*k) <= CT["W"]:
                for h, c in prod.items():
                    put(t, k, h, c)
            else:
                rem += sum((abs(c) for c in prod.values()), arb(0)) * nsc(wt(*k))
    rem += A.maj() * B.rem + B.maj() * A.rem + A.rem * B.rem
    return TM(t, up(rem))


def tm_series(ser, keep, K, op=None):
    """the truncated formal series sum_{k<=K} eps^k Z_k(x) as a Taylor model; op = 'zdz2' applies 2 z d/dz (= x d/dx),
    'divz' divides by z (the x^0 coefficients must contain 0; beta(0) = 0 structurally)"""
    t, rem = {}, arb(0)
    for k in range(K + 1):
        for h in keep:
            cs = ser[k][h].coeffs()
            for m, c in enumerate(cs):
                if m % 2:
                    if not c.contains(0):
                        raise ValueError("odd power in an even series")
                    continue
                i, j = k - m // 2, m // 2
                if i < 0:
                    raise ValueError("degree above 2k")
                if op == "zdz2":
                    c = c * (2 * j)
                elif op == "divz":
                    if j == 0:
                        if not c.contains(0):
                            raise ValueError("beta(0) != 0")
                        continue
                    j -= 1
                if wt(i, j) <= CT["W"]:
                    put(t, (i, j), h, c)
                else:
                    rem += abs(c) * nsc(wt(i, j))
    return TM(t, up(rem))


def tm_geom():
    """1/(1 - eps) on the scalar harmonic 0"""
    t, i = {}, 0
    while wt(i, 0) <= CT["W"]:
        t[(i, 0)] = {0: acb(1)}
        i += 1
    return TM(t, up(nsc(wt(i, 0)) / (1 - nsc(Fr(1)))))


def tm_cinv(Ec, y, O):
    """(I + [Ec .])^-1 y on odd harmonics: Neumann sum up to the power m with (m+1) w_E > W_cut, plus the tail
    ||E||^(m+1)/(1 - ||E||) ||y|| (||E(n)|| <= ||E||_0 (n_0/n)^w_E, so the tail has weight > W_cut)"""
    wE = Ec.minwt()
    if not wE > 0:
        raise ValueError("C - 1 has a weight-0 part")
    u, acc, m = y, y, 0
    while (m + 1) * wE <= CT["W"]:
        u = tm_scale(tm_mul(Ec, u, O), -1)
        acc = tm_add(acc, u)
        m += 1
    nE = up(Ec.maj() + Ec.rem)
    if not nE < 1:
        raise ValueError("||C - 1|| >= 1")
    tail = up(nE ** (m + 1) / (1 - nE) * (y.maj() + y.rem))
    return TM(acc.t, up(acc.rem + tail))


def part_T(K, rA, rW, N=31, verbose=True):
    F, om = FM.params(rA, rW)
    fs = FM.formal(K, F, om)
    O, Ev = fs["O"], fs["E"]
    V, S = tm_series(fs["V"], O, K), tm_series(fs["S"], O, K)
    xY = tm_add(tm_series(fs["V"], O, K, "zdz2"), tm_series(fs["S"], O, K, "zdz2"))
    c = tm_series(fs["c"], Ev, K)
    xc = tm_series(fs["c"], Ev, K, "zdz2")
    Ec = tm_add(c, TM({(0, 0): {0: acb(1)}}), -1)
    P = tm_cinv(Ec, tm_add(V, S), O)
    Phi = tm_cinv(Ec, tm_add(xY, tm_mul(xc, P, O), -1), O)
    Th = tm_mul(tm_series(fs["be"], Ev, K, "divz"), tm_geom(), Ev)
    th = tm_add(Th, tm_scale(tm_mul(P, P, Ev), acb("0.5")), -1)
    T = dict(P=P, Phi=Phi, C=c, th=th)
    # the inner formal series in B5's variables (P_j, Phi_j = 2 j P_j, C_j; theta = 0), j <= N - 1
    sa = SeriesArb(F, om, M=1, prec=ctx.prec)
    d, s, g = sa.series(N)
    inner = dict(P=lambda j, h: d[j][h], Phi=lambda j, h: 2 * j * d[j][h], C=lambda j, h: g[j][h],
                 th=lambda j, h: acb(0))
    keep = dict(P=O, Phi=O, C=Ev, th=Ev)
    jmatch = dict(P=K, Phi=K, C=K, th=K - 1)
    sexp = dict(P=Fr(1), Phi=2 - 2 * CT["g"], C=Fr(1), th=Fr(1))
    out, info = {}, {}
    for f, A in T.items():
        t = {k: dict(v) for k, v in A.t.items()}
        rem = A.rem
        worst = 0.0
        for j in range(N):
            for h in keep[f]:
                a = inner[f](j, h)
                if j <= jmatch[f]:
                    tv = t.get((0, j), {}).get(h, acb(0))
                    worst = max(worst, float(abs(tv - a).upper()))
                    if not (tv - a).contains(0):
                        raise ValueError(f"{f}: eps^0 z^{j} coefficient differs from the inner series")
                elif wt(0, j) <= CT["W"]:
                    put(t, (0, j), h, -a)
                else:
                    rem += abs(a) * nsc(wt(0, j))
            if j <= jmatch[f]:
                t.pop((0, j), None)
        rem = up(rem)
        # bound per harmonic: sum |t_ij| n_0^(s - wt) (wt >= s required) + rem n_0^s
        sx = sexp[f]
        if not CT["W"] >= sx:
            raise ValueError("W_cut below the required order")
        per = {}
        lead = {}
        for h in keep[f]:
            tot = rem * nsc(-sx)
            for k, v in t.items():
                cc = v.get(h)
                if cc is None or cc.is_zero():
                    continue
                if wt(*k) < sx:
                    raise ValueError(f"{f}: term eps^{k[0]} z^{k[1]} of weight {wt(*k)} < {sx} does not cancel")
                tot += abs(cc) * nsc(wt(*k) - sx)
                if wt(*k) == sx:
                    lead[(k, h)] = cc
            per[h] = up(tot)
        out[f] = max(per.values(), key=float)
        info[f] = dict(per={h: float(v) for h, v in per.items()}, rem=float(rem), match=worst,
                       lead={f"eps^{k[0]} z^{k[1]} [{h}]": (float(c.real.mid()), float(c.imag.mid()), float(abs(c)))
                             for (k, h), c in lead.items()})
    return out, info


def eta_part(L0, gam):
    """lemma S's eta on the arc (B5's certified bound, see the docstring), relative to the required orders"""
    n0 = arb(L0).exp()
    base = arb("5.5e-7") / arb("0.0085") ** 12            # |eta| <= base |z|^12, |z| = n^(2 gamma - 1)
    e = (2 * gam - 1) * 12
    res = {}
    for f, s, fac in (("P", arb(1), 1), ("Phi", 2 - 2 * gam, 100), ("C", arb(1), 1), ("th", arb(1), 0)):
        if not e + s <= 0:
            raise ValueError("eta not small enough")
        res[f] = up(fac * base * n0 ** (e + s))
    return res


# ---------------------------------------------------------------------------------------------- (R) remainder
def arc_sup(bb, s, geo):
    n0, gam = geo.n0, geo.gamma0
    tot = arb(0)
    for (bT, e), (a, c) in bb.t["A"].items():
        ex = gam * a + e + s
        if not ex <= 0:
            raise ValueError(f"arc term X^{a} n^{e}: exponent {float(ex):.3f} > 0 after n^{float(s):.2f}")
        f = (1 + n0 ** (-gam)) ** max(a, 0)
        tot = up(tot + c * f * n0 ** ex)
    return tot


def part_R(K, L0, hyp, kw, at, gamma):
    g = MB.Geo(K, L0, atoms=at, arc=True, gamma0=gamma, **kw)
    k = MB.known(g)
    kc = MB.kernel_consts(g, k)
    Pb, dg, R = MB.improve(g, k, kc, hyp, full=True)
    closes = all(Pb[s] < hyp[s] for s in hyp)
    RS, RbH, Rc, RV, R2 = (R[s] for s in ("RS", "RbH", "Rc", "RV", "R2"))
    iC = R2["iC"]
    PK = k["iCK"] * k["Y"]
    WP = iC * (RV + RS + Rc * PK)
    nf = up(g.n0 / (g.n0 - 1))
    n_x2 = E.Q(E.B.mono(1, eX=-2, eT=-2, eN=1))
    Wth = R2["Rb"] * n_x2 * nf + WP * (PK * 2 + WP) * k["half"]
    gam = g.gamma0
    d = dict(P=arc_sup(WP.v, arb(1), g), Phi=arc_sup(WP.d, 2 - 2 * gam, g), C=arc_sup(Rc.v, arb(1), g),
             th=arc_sup(Wth.v, arb(1), g))
    return closes, Pb, d, kc


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    K, L0 = int(pos[0]), pos[1]
    ctx.prec = 256
    gamma = OPTS.get("gamma", "0.3")
    CT.update(g=Fr(gamma), s=1 - 2 * Fr(gamma), W=Fr(OPTS.get("W", "2.6")), L0=arb(L0))
    rA, rW = OPTS.get("rA", "2e-6"), OPTS.get("rW", "2e-6")
    if "centre" in OPTS:
        out, info = part_T(K, "0", "0")
        print(f"K = {K}, ln n0 = {L0}, box centre: (T) delta = " + ", ".join(f"{f} {float(v):.6g}" for f, v in out.items()))
        for f, v in info.items():
            print(f"  {f}: structural cancellation |T - inner| on eps^0 z^j, j <= match: {v['match']:.2e}; "
                  f"leading {v['lead']}; rem {v['rem']:.2e}")
        return 0
    kw = {kk: OPTS[kk] for kk in ("outer", "kappa", "kappa1", "delta", "xF", "r1") if kk in OPTS}
    hyp = None
    for line in open(OPTS["log"]):
        r = json.loads(line)
        if r["K"] == K and abs(r["L0"] - float(L0)) < 1e-9 and r.get("closes"):
            hyp = {s: up(arb(repr(v))) for s, v in r["hyp"].items()}
    if hyp is None:
        print("no closing hypothesis in the log for this K, L0")
        return 2
    at = AT.atoms(K, rA, rW)
    closes, Pb, dR, kc = part_R(K, L0, hyp, kw, at, gamma)
    dT, info = part_T(K, rA, rW)
    dE = eta_part(L0, arb(gamma))
    tot = {f: up(dR[f] + dT[f] + dE[f]) for f in dT}
    ok = closes and all(tot[f] <= DREF[f] for f in tot)
    print(f"K = {K}, ln n0 = {L0}, box rA = {rA}, rW = {rW}, gamma = {gamma}, W_cut = {OPTS.get('W', '2.6')}")
    print("  bootstrap re-verified over the box: P(h) < h: " + str(closes) + "; h = "
          + ", ".join(f"{s} {float(v):.5g}" for s, v in hyp.items()) + "; P(h) = "
          + ", ".join(f"{s} {float(v):.5g}" for s, v in Pb.items()))
    for f in ("P", "Phi", "C", "th"):
        print(f"  delta_{f:3s} <= {float(tot[f]):.6g}   (remainder {float(dR[f]):.3g}, series {float(dT[f]):.6g}, "
              f"eta {float(dE[f]):.2g}; delta_ref {float(DREF[f]):g})")
    for f, v in info.items():
        print(f"  series {f}: per harmonic {v['per']}; leading terms {v['lead']}; remainder {v['rem']:.2e}")
    print(f"(H8)(ii) with delta <= delta_ref: {'PROVED' if ok else 'NOT PROVED'}")
    rec = dict(K=K, L0=float(L0), rA=rA, rW=rW, gamma=gamma, closes=closes, hyp={s: float(v) for s, v in hyp.items()},
               P_hyp={s: float(v) for s, v in Pb.items()}, delta={f: float(v) for f, v in tot.items()},
               delta_R={f: float(v) for f, v in dR.items()}, delta_T={f: float(v) for f, v in dT.items()},
               delta_eta={f: float(v) for f, v in dE.items()}, ok=ok)
    print(json.dumps(rec))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
