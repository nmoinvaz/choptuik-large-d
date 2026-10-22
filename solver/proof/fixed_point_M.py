#!/usr/bin/env python3
"""Start-value fixed point (step (i) of proof/A_certify.md, sections 4.4-4.5) for the M-harmonic inner problem, with
explicit constants and parameter derivatives. Rigorous: python-flint arb/acb ball arithmetic throughout.
Statement, proof and results: proof/A_lemmas_M.md, section 2.

Profile and parameters (those of solver/inner_stokes/hier.py, inner_np.Fh_from):
    F = A sin(theta) + sum_{nu = 3, 5, .., M} (B_nu sin(nu theta) + C_nu cos(nu theta)),
    x = (A, w/w_inf, B_3, C_3, ..., B_M, C_M)   (M + 1 real parameters; w_inf = 2 pi / 1.64389375 as in the solvers),
    F_{+-1} = -+ i A/2,   F_{+-nu} = -+ i B_nu/2 + C_nu/2   (holomorphic in x, so complex parameter boxes are allowed).
System: the truncated inner system of lateral_np.py / inner_np.py, D, S on odd |mu| <= M and C on even |nu| <= M + 1,
every product projected, in the variables of fixed_point.py (z = xi^2):
    D = xi P,  S = z Q,  V = Q - (1 + s_mu) P,  s_mu = i mu w,  W = V + [(C-1)Q],  E^ = [P^2/2]
    P' = (V + [(C-1)Q]) / (2z)
    V' = V/(2z^2) + ([(C-1)Q] + [(C-1)W])/(2z^2) + (-(2+s)Q - (1+s)W + [C E^ P])/(2z)
    C' = [C E^]/2
The fast eigenvalue repeats across harmonics: V/(2z^2) is the same scalar on every V_mu, so the inverse operator, its
domain and its kernel bound are those of M = 1. The labels enter only through s_mu = i mu w in the O(1/z) terms. Two
splittings of the fast row are implemented:
  * "plain" (fixed_point.py's): principal part V/(2z^2) only, kernel e^{1/(2t) - 1/(2z)}. Valid on rays and (with the
    strict kernel bound of lemma6_sector.py) on the paths of lemma S. The label terms are perturbations of size
    |3 + 2 s_mu| R / (2 a) (or |3 + 2 s_mu| / (2k) on a ray), which grows like |mu| w: the contraction costs ~ M w R.
  * "diag" (rays only, --diag): principal part V/(2z^2) - (3 + 2 s_mu) V/(2z), kernel
    e^{1/(2t) - 1/(2z)} (t/z)^{(3 + 2 s_mu)/2}. On a ray t/z is real and positive, so |(t/z)^{s_mu}| = 1 for every label
    and |kernel| <= e^{-a(1/|t| - 1/|z|)} (|t|/|z|)^{3/2} <= e^{-a(1/|t| - 1/|z|)}: the kernel bounds of fixed_point.py
    hold unchanged and the label-dependent diagonal terms leave the Lipschitz matrix. (On a non-ray path the factor is
    |(t/z)^{s_mu}| = e^{-mu w (arg t - arg z)}, the label weight; that is why lemma S uses "plain".)
The integral equation, norms, majorants, forcing and contraction check are those of fixed_point.py (whose helpers are
imported unchanged), with the harmonic sets of M. M = 1 with "plain" is fixed_point.py exactly (checked by --check-m1).

    python3 fixed_point_M.py --M=3 [--x=A,wf,B3,C3] [--r0=0.1] [--arg=3pi/8] [--box=h | --cbox=c] [--sector=d] [--diag]
                             [--k=40] [--N=..]
        the lemma at a point or uniformly over a parameter box; --box=h: real box x +- h, --cbox=c: complex box
        x + c(1 + i); --arg: arg xi of the start ray (3pi/8 = fixed_point.py, 5pi/16 = inner_np.py); --sector=d: every
        ray |arg xi - arg| <= d
    python3 fixed_point_M.py --check-m1        M = 1 reproduces fixed_point.py: constants, start values, start data
    python3 fixed_point_M.py --M=3 --data --box=1e-6 [--cauchy=1e-3]   start data with derivatives over a real box
    python3 fixed_point_M.py --M=3 --check-series   the ball series against lateral.py's mpmath recursion (independent
                                                    code) at the start point
    mutations (each must make the check fail; exit status 0 only if it does):
    python3 fixed_point_M.py --M=3 --mutate=series     one coefficient of the harmonic-M series off by a relative 1e-6
    python3 fixed_point_M.py --M=3 --mutate=cdrop      the series built without the c harmonics +-(M+1) (the M-1 c-set),
                                                       the residual with them: the low orders no longer vanish
    python3 fixed_point_M.py --M=3 --mutate=diagpath   the "diag" splitting on a lemma-S path, where it is invalid: the
                                                       harness must refuse (the kernel bound there needs "plain")
    negative control: python3 fixed_point_M.py --M=3 --r0=0.3     start radius too large: the contraction must fail

Interface for the validated integration (the point certificate, solver/proof/certify_M*.py):
    import fixed_point_M as FPM
    xi0, vals, ders, (etab, detab) = FPM.start_data_M(x, M, r0="0.1", arg=None, theta_sign=+1, cauchy="1e-3",
                                                       diag=True)
    x: list of M + 1 real arb balls (the box X, created at >= 320 bits); vals: the state at xi0 in lateral_np's order
    [D_mu (odd mu ascending), S_mu, C_nu (even nu ascending)], valid for every x in X; ders[i]: d(state)/d x_i over X.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fixed_point as FP  # noqa: E402
from fixed_point import Maj, MD, conv, poly_maj, up  # noqa: E402
from series_arb import SeriesArb  # noqa: E402
from flint import arb, acb, acb_poly, ctx  # noqa: E402

W0_STR = "1.64389375"
# numerical zeros (double precision, not rigorous) of solver/inner_stokes/hier.py / nested.py:
ZEROS = {
    1: ["2.5421480475", "0.7769940136"],                                         # A_certify.md section 5
    3: ["2.7265", "1.0156", "0.14450", "0.01118"],                               # README (nested.py), 4-5 digits
    5: ["2.718579189", "0.999228404", "0.149203575", "0.014719052", "0.008405347", "0.001470345"],  # hier.py --schur
}


def W0():
    return 2 * arb.pi() / arb(W0_STR)


def parse_arg(s):
    """'3pi/8' -> arb; plain decimals are radians"""
    if s is None:
        return 3 * arb.pi() / 8
    s = str(s).replace(" ", "")
    if "pi" in s:
        num, _, den = s.partition("pi")
        num = arb(num) if num not in ("", "+") else arb(1)
        den = arb(den.lstrip("/")) if den else arb(1)
        return num * arb.pi() / den
    return arb(s)


def fh_from(x, M):
    """harmonics of F from x = (A, wf, B3, C3, ..): {+-1: -+ i A/2, +-nu: -+ i B/2 + C/2}; holomorphic in x"""
    I = acb(0, 1)
    A = acb(x[0])
    h = {1: -I * A / 2, -1: I * A / 2}
    for i, nu in enumerate(range(3, M + 1, 2)):
        B, C = acb(x[2 + 2 * i]), acb(x[3 + 2 * i])
        h[nu] = -I * B / 2 + C / 2
        h[-nu] = I * B / 2 + C / 2
    return h


def a_rate(arg_xi, d=None):
    """the fast-kernel rate a = -cos(arg z)/2 on the ray arg xi (arg z = 2 arg xi), or its minimum over the closed sector
    |arg xi - arg| <= d (an exact lower bound as arb); must be > 0"""
    th = 2 * arb(arg_xi)
    if d is None:
        a = -th.cos() / 2
    else:
        d2 = 2 * arb(d)
        a1, a2 = -(th - d2).cos() / 2, -(th + d2).cos() / 2
        a = a1 if a1 < a2 else (a2 if a2 < a1 else a1.union(a2))
        # the minimum of -cos over an interval is at an endpoint unless the interval contains a multiple of 2 pi,
        # i.e. arg z = 0 (mod 2 pi); exclude that
        if (th - d2) <= 0 or (th + d2) >= 2 * arb.pi():
            raise ValueError("sector reaches arg z = 0")
    a = arb(a.lower())
    if not a > 0:
        raise ValueError("the fast kernel is not bounded on this ray/sector (a <= 0)")
    return a


def setup_M(x, M, r0, prec=320, N_in=None, mutate=None):
    """formal series (x acb, possibly boxes), optimal truncation at |xi| = r0 (or the given N), polynomials in z"""
    ctx.prec = prec
    w = W0() * acb(x[1])
    Fh = fh_from(x, M)
    J = int(1 / (2 * float(r0) ** 2)) + 30
    if N_in is not None:
        J = max(J, N_in + 1)
    if mutate == "cdrop":
        # MUTATION: the series with the c harmonics |nu| <= M - 1 only (inconsistent with the system checked below)
        sa0 = SeriesArb(Fh, w, M=M, prec=prec)
        sa0.even = [nu for nu in sa0.even if abs(nu) <= M - 1]
        d, s, g = sa0.series(J)
        sa = SeriesArb(Fh, w, M=M, prec=prec)
        g = [{nu: gj.get(nu, acb(0)) for nu in sa.even} for gj in g]
    else:
        sa = SeriesArb(Fh, w, M=M, prec=prec)
        d, s, g = sa.series(J)
    R = arb(r0) ** 2
    mags = [max(abs(d[j][mu]) for mu in sa.odd) * R ** j for j in range(J)]
    N = N_in or min(range(1, J), key=lambda j: float(mags[j].mid()))
    P = {mu: acb_poly([d[j][mu] for j in range(N)]) for mu in sa.odd}
    Q = {mu: acb_poly([s[j][mu] for j in range(N)]) for mu in sa.odd}
    C = {nu: acb_poly([g[j][nu] for j in range(N)]) for nu in sa.even}
    if mutate == "series":
        # MUTATION: one coefficient of the top harmonic's D series (order z^5) off by a relative 1e-6
        cs = P[M].coeffs()
        cs[5] = cs[5] * (1 + acb("1e-6"))
        P[M] = acb_poly(cs)
    return sa, N, P, Q, C, R


def lipschitz_M(sa, P, Q, C, k, rho, wts, diag=False):
    """fixed_point.lipschitz_and_forcing for any harmonic set, with the optional "diag" splitting of the fast row:
    (2+s)Q + (1+s)W = (3+2s)V + (2+s)(1+s)P + (1+s)[(C-1)Q], and with "diag" the (3+2s)V part is in the principal part.
    Returns (L, comps, kw, rows); L_ij bounds the weighted norm of d(T eta)_i / d eta_j over the ball."""
    odd, even = sa.odd, sa.even
    sm = {mu: sa.I * mu * sa.w for mu in odd}
    comps = [("p", mu) for mu in odd] + [("v", mu) for mu in odd] + [("c", nu) for nu in even]
    kw = {cp: (k + 1 if cp[0] == "c" else k) for cp in comps}
    ball = {cp: arb(rho) / arb(wts[i]) for i, cp in enumerate(comps)}
    V = {mu: Q[mu] - (1 + sm[mu]) * P[mu] for mu in odd}
    Pm = {mu: MD(poly_maj(P[mu]) + Maj(ball[("p", mu)], kw[("p", mu)]), {("p", mu): Maj(1, kw[("p", mu)])}) for mu in odd}
    Vm = {mu: MD(poly_maj(V[mu]) + Maj(ball[("v", mu)], kw[("v", mu)]), {("v", mu): Maj(1, kw[("v", mu)])}) for mu in odd}
    Cm1 = {nu: MD(poly_maj(C[nu] - (1 if nu == 0 else 0)) + Maj(ball[("c", nu)], kw[("c", nu)]),
                  {("c", nu): Maj(1, kw[("c", nu)])}) for nu in even}
    Cm = {nu: (Cm1[nu] + MD(Maj(1, 0)) if nu == 0 else Cm1[nu]) for nu in even}
    Qm = {mu: Vm[mu] + Pm[mu] * (1 + sm[mu]) for mu in odd}
    CQ = conv(Cm1, Qm, set(odd))
    Wm = {mu: Vm[mu] + CQ[mu] for mu in odd}
    CW = conv(Cm1, Wm, set(odd))
    Eh = {kk: v * arb(0.5) for kk, v in conv(Pm, Pm, set(even)).items()}
    CE = conv(Cm, Eh, set(even))
    CEP = conv(CE, Pm, set(odd))
    Fp = {mu: (Vm[mu] + CQ[mu]).shift(1) * arb(0.5) for mu in odd}
    if diag:
        G1 = {mu: Pm[mu] * ((2 + sm[mu]) * (1 + sm[mu])) + CQ[mu] * (1 + sm[mu]) + CEP[mu] for mu in odd}
    else:
        G1 = {mu: Qm[mu] * (2 + sm[mu]) + Wm[mu] * (1 + sm[mu]) + CEP[mu] for mu in odd}
    Fv = {mu: ((CQ[mu] + CW[mu]).shift(2) + G1[mu].shift(1)) * arb(0.5) for mu in odd}
    Fc = {nu: CE[nu] * arb(0.5) for nu in even}
    rows = {("p", mu): (Fp[mu], "slow") for mu in odd}
    rows.update({("v", mu): (Fv[mu], "fast") for mu in odd})
    rows.update({("c", nu): (Fc[nu], "slow") for nu in even})
    L = [[FP.intbound(rows[ci][0].der.get(cj), rows[ci][1], kw[ci]) for cj in comps] for ci in comps]
    return L, comps, kw, rows


def perron(L, sym=None):
    Lf = np.array([[float(v.upper()) for v in row] for row in L]) + 1e-12
    ev, evec = np.linalg.eig(Lf)
    i = int(np.argmax(ev.real))
    u = np.abs(evec[:, i].real) + 1e-6 * np.abs(evec[:, i].real).max()
    w = list(1.0 / u)
    if sym:
        w = [min(w[a], w[b]) for a, b in enumerate(sym)]
    return w


def wnorm(L, wts):
    """max_i sum_j w_i L_ij / w_j as an exact upper bound"""
    return arb(max((sum(arb(wts[i]) * L[i][j] / arb(wts[j]) for j in range(len(L)))).upper() for i in range(len(L))))


def sym_index(comps):
    idx = {cp: i for i, cp in enumerate(comps)}
    return [idx[(t, -m)] for (t, m) in comps]


class rate:
    """use the given fast-kernel rate (and optionally another intbound) inside fixed_point; restore afterwards"""

    def __init__(self, a, intbound=None):
        self.a, self.ib = arb(a), intbound

    def __enter__(self):
        self.saved = FP.A_FAST, FP.intbound
        FP.A_FAST = self.a
        if self.ib is not None:
            FP.intbound = self.ib
        return self

    def __exit__(self, *exc):
        FP.A_FAST, FP.intbound = self.saved


def run_M(x, M, r0="0.1", arg=None, sector=None, prec=320, verbose=True, kmins_in=None, N_in=None, kcap=40, k_in=None,
          diag=False, a_in=None, intbound=None, sym=False, mutate=None):
    """the contraction of fixed_point.run for M harmonics. x: list of M + 1 acb (points or boxes).
    arg: arg xi of the ray (arb; default 3pi/8); sector: half-width d of the sector of rays; a_in: an explicit kernel
    rate instead (lemma S); intbound: replacement for fixed_point.intbound (lemma S's strict bound)."""
    ctx.prec = prec
    if diag and intbound is not None:
        raise ValueError("the diag splitting is valid on rays only; refusing it with a non-ray kernel bound")
    argx = parse_arg(arg) if not isinstance(arg, arb) else arg
    a = arb(a_in) if a_in is not None else a_rate(argx, sector)
    sa, N, P, Q, C, R = setup_M(x, M, r0, prec, N_in, mutate)
    Maj.R = R
    Np, Nv, Nc, V = FP.residual_numerators(sa, P, Q, C)
    res, kmins, lows = {}, [], []
    kin = iter(kmins_in) if kmins_in else None
    for mu in sa.odd:
        m, km, ok = FP.tail_maj(Np[mu], 1, arb(0.5), next(kin) if kin else None)
        res[("p", mu)] = m; kmins.append(km); lows.append(ok)
        m, km, ok = FP.tail_maj(Nv[mu], 2, arb(0.5), next(kin) if kin else None)
        res[("v", mu)] = m; kmins.append(km); lows.append(ok)
    for nu in sa.even:
        m, km, ok = FP.tail_maj(Nc[nu], 0, arb(1), next(kin) if kin else None)
        res[("c", nu)] = m; kmins.append(km); lows.append(ok)
    kmax = min(min(res[("p", mu)].q + 1 for mu in sa.odd), min(res[("v", mu)].q + 2 for mu in sa.odd),
               min(res[("c", nu)].q for nu in sa.even))
    k = min(kmax, kcap) if k_in is None else k_in
    if k > kmax:
        raise ValueError(f"weight exponent {k} exceeds the forcing orders (max {kmax})")
    ncomp = len(sa.odd) * 2 + len(sa.even)
    with rate(a, intbound):
        L, comps, kw, rows = lipschitz_M(sa, P, Q, C, k, arb("1e-20"), [1.0] * ncomp, diag)
        wts = perron(L, sym_index(comps) if sym else None)
        Fv = [FP.intbound(res[cp], "fast" if cp[0] == "v" else "slow", kw[cp]) for cp in comps]
        F = arb(max((arb(wts[j]) * Fv[j]).upper() for j in range(len(comps))))
        lip0 = wnorm(L, wts)
        rho = 4 * F / (1 - lip0) if lip0 < 1 else arb(1)
        L, comps, kw, rows = lipschitz_M(sa, P, Q, C, k, rho, wts, diag)
        lip = wnorm(L, wts)
    selfmap = F + lip * rho
    ok = bool(lip < 1) and bool(selfmap < rho) and all(lows)
    out = dict(M=M, N=N, k=k, kmax=kmax, kmins=kmins, low_ok=all(lows), lip=lip, F=F, rho=rho, selfmap=selfmap, ok=ok,
               wts=wts, comps=comps, kw=kw, L=L, sa=sa, P=P, Q=Q, C=C, R=R, r0=arb(r0), a=a, arg=argx, diag=diag,
               sector=sector)
    if verbose:
        report(out, x)
    return out


def report(out, x=None):
    comps, L = out["comps"], out["L"]
    print(f"M = {out['M']}, r0 = {out['r0'].str(5)} (R = r0^2), arg xi = {float(out['arg'].mid()):.6f}"
          + (f" +- {out['sector']}" if out["sector"] else "") + f", kernel rate a = {float(out['a']):.5f}, splitting "
          f"{'diag' if out['diag'] else 'plain'}: N = {out['N']} terms, weight exponent k = {out['k']} "
          f"(forcing allows {out['kmax']})")
    print(f"  residual numerators: lower coefficients all contain 0: {out['low_ok']}; first nonzero orders "
          f"{sorted(set(out['kmins']))}")
    print(f"  forcing ||T(0)||_w <= {float(out['F'].upper()):.3e}")
    big = sorted(((float(L[i][j].upper()), comps[i], comps[j]) for i in range(len(L)) for j in range(len(L))),
                 reverse=True)[:6]
    print("  largest Lipschitz entries (unweighted): " + ", ".join(f"{t[1][0]}{t[1][1]:+d}<-{t[2][0]}{t[2][1]:+d} "
                                                                     f"{t[0]:.3g}" for t in big))
    print(f"  contraction constant (weighted max norm) <= {float(out['lip'].upper()):.4f}")
    print(f"  ball rho = {float(out['rho'].upper()):.3e}: ||T(0)|| + Lip rho = {float(out['selfmap'].upper()):.3e} "
          f"< rho: {bool(out['selfmap'] < out['rho'])}")
    eb = eta_bounds(out)
    print(f"  => fixed point exists and is unique in the ball: {out['ok']}; |eta| <= "
          f"{max(float(v.upper()) for v in eb.values()):.2e} at |z| = R (p, v, c variables)")


def eta_bounds(out):
    """|eta_j(z)| <= (rho / w_j) |z|^{k_j} on the ray up to |z| = R: the bound at |z| = R, per component"""
    R, k, rho, wts, comps = out["R"], out["k"], out["rho"], out["wts"], out["comps"]
    return {cp: rho / arb(wts[i]) * R ** (k + 1 if cp[0] == "c" else k) for i, cp in enumerate(comps)}


def xi_start(out, theta_sign=+1):
    phi = theta_sign * out["arg"]
    r0 = out["r0"]
    return acb(r0 * phi.cos(), r0 * phi.sin())


def start_values(out, theta_sign=+1):
    """enclosures of y+(xi0) (y- for theta_sign = -1): {mu: D}, {mu: S}, {nu: C} at xi0 = r0 e^{+- i arg}"""
    sa, P, Q, C, R, r0 = out["sa"], out["P"], out["Q"], out["C"], out["R"], out["r0"]
    xi0 = xi_start(out, theta_sign)
    z0 = xi0 * xi0
    eb = eta_bounds(out)
    sm = {mu: sa.I * mu * sa.w for mu in sa.odd}

    def disc(e):
        r = arb(0, e.upper())
        return acb(r, r)
    Dv, Sv, Cv = {}, {}, {}
    for mu in sa.odd:
        e_p = eb[("p", mu)]
        e_q = eb[("v", mu)] + up(1 + sm[mu]) * e_p
        Dv[mu] = xi0 * P[mu](z0) + disc(r0 * e_p)
        Sv[mu] = z0 * Q[mu](z0) + disc(R * e_q)
    for nu in sa.even:
        Cv[nu] = C[nu](z0) + disc(eb[("c", nu)])
    return xi0, Dv, Sv, Cv


class JetN:
    """first-order jet in P real parameters: value v and derivatives d[0..P-1], all acb balls"""
    __slots__ = ("v", "d")

    def __init__(self, v, d):
        self.v, self.d = v, d

    @staticmethod
    def const(v, P):
        return JetN(acb(v), [acb(0)] * P)

    def __add__(self, o):
        return JetN(self.v + o.v, [a + b for a, b in zip(self.d, o.d)])

    def __sub__(self, o):
        return JetN(self.v - o.v, [a - b for a, b in zip(self.d, o.d)])

    def __mul__(self, o):
        if isinstance(o, JetN):
            return JetN(self.v * o.v, [self.v * b + a * o.v for a, b in zip(self.d, o.d)])
        o = acb(o)
        return JetN(self.v * o, [a * o for a in self.d])

    def __truediv__(self, o):
        o = acb(o)
        return JetN(self.v / o, [a / o for a in self.d])


def series_jet_M(x, M, J):
    """series_arb.SeriesArb.series with JetN values in x = (A, wf, B3, C3, ..): the coefficients and their derivatives
    in every parameter (balls over a box allowed)"""
    P = M + 1
    I = acb(0, 1)

    def var(i):
        d = [acb(0)] * P
        d[i] = acb(1)
        return JetN(acb(x[i]), d)
    w = var(1) * W0()
    Aj = var(0)
    Fh = {1: Aj * (-I / 2), -1: Aj * (I / 2)}
    for i, nu in enumerate(range(3, M + 1, 2)):
        B, C = var(2 + 2 * i), var(3 + 2 * i)
        Fh[nu] = B * (-I / 2) + C * acb(0.5)
        Fh[-nu] = B * (I / 2) + C * acb(0.5)
    odd = [mu for mu in range(-M, M + 1) if mu % 2]
    even = [nu for nu in range(-(M + 1), M + 2) if nu % 2 == 0]
    Z = JetN.const(0, P)

    def mul(Ad, Bd, keep):
        out = {kk: Z for kk in keep}
        for i, ai in Ad.items():
            for j, bj in Bd.items():
                if i + j in out:
                    out[i + j] = out[i + j] + ai * bj
        return out
    d = [{mu: Fh[mu] * 2 for mu in odd}]
    s, g = [], [{nu: JetN.const(1 if nu == 0 else 0, P) for nu in even}]
    e = [{nu: Z for nu in even}]
    ce = [{nu: Z for nu in even}]
    for j in range(J):
        sj = {mu: d[j][mu] * (2 * j + 1) + d[j][mu] * w * (I * mu) for mu in odd}
        for i in range(1, j + 1):
            t = mul(g[i], s[j - i], odd)
            for mu in odd:
                sj[mu] = sj[mu] - t[mu]
        s.append(sj)
        m = j + 1
        em = {nu: Z for nu in even}
        for i in range(m):
            t = mul(d[i], d[m - 1 - i], even)
            for nu in even:
                em[nu] = em[nu] + t[nu] / 2
        e.append(em)
        cm = {nu: Z for nu in even}
        for i in range(m):
            t = mul(g[i], e[m - i], even)
            for nu in even:
                cm[nu] = cm[nu] + t[nu]
        ce.append(cm)
        g.append({nu: cm[nu] / (2 * m) for nu in even})
        num = {mu: s[j][mu] * (2 * j + 2) + s[j][mu] * w * (I * mu) for mu in odd}
        for i in range(1, j + 2):
            t = mul(g[i], d[j + 1 - i], odd)
            for mu in odd:
                num[mu] = num[mu] - t[mu] * (2 * (j + 1 - i))
        for mm in range(1, j + 2):
            t = mul(ce[mm], d[j + 1 - mm], odd)
            for mu in odd:
                num[mu] = num[mu] - t[mu]
        d.append({mu: num[mu] / (2 * j + 2) for mu in odd})
    return odd, even, d, s, g


def start_data_M(x, M, r0="0.1", arg=None, theta_sign=+1, cauchy=None, centre=None, diag=False, kcap=40,
                 sector=None, verbose=False):
    """C^1 start data over a real parameter box X (the interface of fixed_point.start_data, for M harmonics).
    x: list of M + 1 real arb balls. Y = Y_N + eta: Y_N and its derivatives exactly by jets over X; eta bounded by the
    fixed point over X; its derivatives by Cauchy's estimate |d eta / d x_i| <= sup |eta| / c over the complex box
    X + c(1 + i) (eta is holomorphic in the parameters there: a uniform limit of holomorphic Picard iterates).
    Returns (xi0, vals, ders, (eta bound over X, Cauchy bound of its derivatives)); vals in lateral_np's state order
    [D_mu (odd mu ascending), S_mu, C_nu (even nu ascending)], ders[i] the same list of d/dx_i."""
    ctx.prec = 320
    x = [arb(v) for v in x]
    cauchy = arb("1e-3") if cauchy is None else arb(cauchy)
    xc = [acb(v + arb(0, cauchy), arb(0, cauchy)) for v in x]
    isbox = any(v.rad() > 0 for v in x)
    if centre is None:
        Maj.BOX = False
        centre = run_M([acb(v.mid()) for v in x], M, r0, arg, sector, verbose=False, kcap=kcap, diag=diag)
    Maj.BOX = isbox
    outX = run_M([acb(v) for v in x], M, r0, arg, sector, verbose=False, kmins_in=centre["kmins"], N_in=centre["N"],
                 k_in=centre["k"], diag=diag)
    Maj.BOX = True
    outC = run_M(xc, M, r0, arg, sector, verbose=False, kmins_in=centre["kmins"], N_in=centre["N"], k_in=centre["k"],
                 diag=diag)
    Maj.BOX = False
    if not (outX["ok"] and outC["ok"]):
        raise RuntimeError(f"fixed point not verified (box {outX['ok']}, Cauchy box {outC['ok']})")
    N, R = centre["N"], outX["R"]
    eb, ebC = eta_bounds(outX), eta_bounds(outC)
    odd, even, d, s, g = series_jet_M([acb(v) for v in x], M, N)
    xi0 = xi_start(outX, theta_sign)
    z0 = xi0 * xi0
    zp = [acb(1)]
    for _ in range(N):
        zp.append(zp[-1] * z0)
    P = M + 1
    sm_up = {mu: up(1 + acb(0, 1) * mu * W0() * xc[1]) for mu in odd}
    r0a = arb(r0)

    def ball(e):
        r = arb(0, e.upper())
        return acb(r, r)
    vals, ders = [], [[] for _ in range(P)]

    def add(jet, e, eC):
        vals.append(jet.v + ball(e))
        for i in range(P):
            ders[i].append(jet.d[i] + ball(eC / cauchy))

    def jsum(coefs, mu):
        acc = JetN.const(0, P)
        for j in range(N):
            acc = acc + coefs[j][mu] * zp[j]
        return acc
    for mu in odd:
        add(jsum(d, mu) * xi0, r0a * eb[("p", mu)], r0a * ebC[("p", mu)])
    for mu in odd:
        add(jsum(s, mu) * z0, R * (eb[("v", mu)] + sm_up[mu] * eb[("p", mu)]),
            R * (ebC[("v", mu)] + sm_up[mu] * ebC[("p", mu)]))
    for nu in even:
        add(jsum(g, nu), eb[("c", nu)], ebC[("c", nu)])
    etab = max(eb.values(), key=lambda v: float(v.upper()))
    etabC = max(ebC.values(), key=lambda v: float(v.upper()))
    if verbose:
        print(f"start data over X (M = {M}, N = {N}, k = {centre['k']}): contraction over X {float(outX['lip'].upper()):.4f}, "
              f"over the Cauchy box (c = {cauchy.str(3)}) {float(outC['lip'].upper()):.4f}; sup |eta| over X "
              f"{float(etab.upper()):.2e}, Cauchy part of the derivatives {float((etabC / cauchy).upper()):.2e}")
    return xi0, vals, ders, (etab, etabC / cauchy)


def state_order(M):
    odd = [mu for mu in range(-M, M + 1) if mu % 2]
    even = [nu for nu in range(-(M + 1), M + 2) if nu % 2 == 0]
    return [f"D{mu:+d}" for mu in odd] + [f"S{mu:+d}" for mu in odd] + [f"C{nu:+d}" for nu in even]


# ---------------------------------------------------------------- checks


def check_m1():
    """M = 1 reproduces fixed_point.py: the constants of run, the start values on both rays, and start_data over a box"""
    ctx.prec = 320
    fails = []
    for box in (None, "1e-3"):
        Maj.BOX = False
        if box is None:
            A, wf = acb(FP.A_STAR), acb(FP.WF_STAR)
            kin = Nin = None
        else:
            c0 = FP.run(acb(FP.A_STAR), acb(FP.WF_STAR), "0.1", verbose=False)
            kin, Nin = c0["kmins"], c0["N"]
            Maj.BOX = True
            A = acb(arb(FP.A_STAR, arb(box)), arb(0, arb(box)))
            wf = acb(arb(FP.WF_STAR, arb(box)), arb(0, arb(box)))
        o1 = FP.run(A, wf, "0.1", verbose=False, kmins_in=kin, N_in=Nin)
        # the same kernel-rate ball as fixed_point.py: its sqrt(2)/4 is made at import time at 53 bits, so its bounds
        # carry balls of relative radius ~1e-17, and it takes Python's max of balls (an undecided comparison keeps the
        # first ball) where this module takes the largest upper bound. The constants therefore agree to ~1e-16
        # relative, not bit for bit; the start values and start data (below) agree exactly.
        o2 = run_M([A, wf], 1, "0.1", None, verbose=False, kmins_in=kin, N_in=Nin, a_in=FP.A_FAST)
        Maj.BOX = False
        rels = {key: float((abs(o1[key].upper() - o2[key].upper()) / o1[key].upper()).upper())
                for key in ("lip", "F", "rho", "selfmap")}
        rel = max(rels.values())
        print(f"  relative differences of the upper bounds: " + ", ".join(f"{k_} {v:.0e}" for k_, v in rels.items()))
        same = all(o1[key] == o2[key] for key in ("N", "k", "kmins")) and o1["wts"] == o2["wts"] and rel < 1e-15 \
            and o1["ok"] and o2["ok"]
        print(f"run{' over the complex box +-' + box + '(1+i)' if box else ' at x^'}: fixed_point.py N {o1['N']}, k "
              f"{o1['k']}, lip {float(o1['lip'].upper()):.6f}, F {float(o1['F'].upper()):.6e}, rho "
              f"{float(o1['rho'].upper()):.6e}; fixed_point_M: N {o2['N']}, k {o2['k']}, lip "
              f"{float(o2['lip'].upper()):.6f}, F {float(o2['F'].upper()):.6e}, rho {float(o2['rho'].upper()):.6e}; same N, "
              f"k, vanishing orders and weights, constants equal to relative {rel:.0e}: {same}")
        if not same:
            fails.append(f"run box={box}")
        if box is None:
            for sg in (+1, -1):
                _, D1, S1, C1 = FP.start_values(o1, sg)
                _, D2, S2, C2 = start_values(o2, sg)
                v1 = [D1[m] for m in (-1, 1)] + [S1[m] for m in (-1, 1)] + [C1[n] for n in (-2, 0, 2)]
                v2 = [D2[m] for m in (-1, 1)] + [S2[m] for m in (-1, 1)] + [C2[n] for n in (-2, 0, 2)]
                dm = max(float(abs(acb(a.mid()) - acb(b.mid())).upper()) for a, b in zip(v1, v2))
                dr = max(abs(float(a.real.rad()) - float(b.real.rad())) / float(a.real.rad()) for a, b in zip(v1, v2))
                eq = dm < 1e-60 and dr < 1e-6 and all(a.overlaps(b) for a, b in zip(v1, v2))
                print(f"  start values on the ray arg xi = {'+' if sg > 0 else '-'}3pi/8: midpoints differ by {dm:.0e}, "
                      f"radii by relative {dr:.0e}: {eq}; D+1 = {v2[1].str(14)}, radius {float(v2[1].real.rad()):.2e}")
                if not eq:
                    fails.append(f"start values {sg}")
    # start data over the real box 1e-9 (the box of section 11) with the default Cauchy radius of fixed_point (1e-2)
    t0 = time.time()
    Ab, wb = arb(FP.A_STAR, arb("1e-9")), arb(FP.WF_STAR, arb("1e-9"))
    xi1, va1, dA1, dw1, eb1 = FP.start_data(Ab, wb, "0.1", +1, arb("1e-3"))
    xi2, va2, dd2, eb2 = start_data_M([Ab, wb], 1, "0.1", None, +1, "1e-3")
    worst_mid, worst_rad, contain = 0.0, 0.0, True
    for a, b in zip(va1 + dA1 + dw1, va2 + dd2[0] + dd2[1]):
        worst_mid = max(worst_mid, float(abs(acb(a.mid()) - acb(b.mid())).upper()))
        ra = max(float(a.real.rad()), float(a.imag.rad()))
        rb = max(float(b.real.rad()), float(b.imag.rad()))
        worst_rad = max(worst_rad, abs(rb - ra) / max(ra, 1e-300))
        contain = contain and a.overlaps(b)
    print(f"start_data over X = x^ +- 1e-9 (Cauchy 1e-3), 7 values and 14 derivatives: max midpoint difference "
          f"{worst_mid:.1e}, max relative radius difference {worst_rad:.1e}, all overlap: {contain} "
          f"({time.time() - t0:.0f} s)")
    if not (contain and worst_rad < 1e-6 and worst_mid < 1e-40):
        fails.append("start_data")
    print(f"M = 1 REPRODUCES fixed_point.py: {not fails}" + (f" (differences: {fails})" if fails else ""))
    return not fails


def check_series(M, x, r0="0.085", arg=None):
    """the ball series (series_arb, as used here) against lateral.py's mpmath recursion (independent code) at the
    start point: the truncated sums at the same truncation, in double precision"""
    sys.path.insert(0, os.path.join(HERE, "..", "inner_stokes"))
    import mpmath as mp
    from lateral import Model
    ctx.prec = 320
    argx = parse_arg(arg)
    out = run_M([acb(v) for v in x], M, r0, argx, verbose=False)
    xi0, Dv, Sv, Cv = start_values(out, +1)
    mp.mp.dps = 40
    xf = [float(v) for v in x]
    Fh = {1: xf[0] / 2j, -1: -xf[0] / 2j}
    for i, nu in enumerate(range(3, M + 1, 2)):
        Fh[nu] = xf[2 + 2 * i] / 2j + xf[3 + 2 * i] / 2
        Fh[-nu] = -xf[2 + 2 * i] / 2j + xf[3 + 2 * i] / 2
    wmp = 2 * mp.pi / mp.mpf(W0_STR) * mp.mpf(str(x[1]))
    m = Model({k: mp.mpc(v) for k, v in Fh.items()}, wmp, M, gravity=True)
    d, s, g = m.series(out["N"] + 2)
    xm = mp.mpc(float(xi0.real.mid()), float(xi0.imag.mid()))
    # sums with exactly N terms (the same truncation as Y_N)
    N = out["N"]
    worst = 0.0
    for mu in out["sa"].odd:
        Dm = sum(d[j][mu] * xm ** (2 * j + 1) for j in range(N))
        Sm = sum(s[j][mu] * xm ** (2 * j + 2) for j in range(N))
        worst = max(worst, abs(complex(Dm) - complex(Dv[mu].mid())), abs(complex(Sm) - complex(Sv[mu].mid())))
    for nu in out["sa"].even:
        Cm = sum(g[j][nu] * xm ** (2 * j) for j in range(N))
        worst = max(worst, abs(complex(Cm) - complex(Cv[nu].mid())))
    print(f"M = {M}: Y_N at xi0 = {r0} e^(i {float(argx.mid()):.4f}) (N = {N}) vs lateral.py's mpmath series: max "
          f"difference {worst:.1e}")
    return worst


def main():
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    if "check-m1" in opts:
        sys.exit(0 if check_m1() else 3)
    M = int(opts.get("M", "3"))
    xs = opts["x"].split(",") if "x" in opts else ZEROS[M]
    r0 = opts.get("r0", "0.1")
    arg = opts.get("arg", "3pi/8")
    sector = opts.get("sector")
    diag = "diag" in opts
    kcap = int(opts.get("k", "40"))
    mutate = opts.get("mutate")
    ctx.prec = 320
    if "check-series" in opts:
        w = check_series(M, xs, r0, arg)
        sys.exit(0 if w < 1e-12 else 3)
    if mutate == "diagpath":
        try:
            run_M([acb(v) for v in xs], M, r0, arg, verbose=False, diag=True, intbound=FP.intbound)
        except ValueError as e:
            print(f"MUTATION diagpath: refused ({e}) -> caught")
            sys.exit(0)
        print("MUTATION diagpath: NOT refused")
        sys.exit(3)
    if "data" in opts:
        h = opts.get("box", "0")
        xb = [arb(v, arb(h)) if float(h) > 0 else arb(v) for v in xs]
        t0 = time.time()
        xi0, vals, ders, eb = start_data_M(xb, M, r0, arg, +1, opts.get("cauchy", "1e-3"), diag=diag, kcap=kcap,
                                           sector=sector, verbose=True)
        names = state_order(M)
        pn = ["A", "wf"] + [f"{t}{nu}" for nu in range(3, M + 1, 2) for t in ("B", "C")]
        print(f"xi0 = {xi0.str(10)}; {time.time() - t0:.0f} s")
        for i, nm in enumerate(names):
            v = vals[i]
            print(f"  {nm:5s} {acb(v.mid()).str(16):48s} rad {max(float(v.real.rad()), float(v.imag.rad())):.1e}   "
                  + "  ".join(f"d/d{pn[p]} rad {max(float(ders[p][i].real.rad()), float(ders[p][i].imag.rad())):.1e}"
                              for p in range(M + 1)))
        if "json" in opts:
            def sj(z):
                return [z.real.str(60, radius=True), z.imag.str(60, radius=True)]
            with open(opts["json"], "w") as fh:
                json.dump({"M": M, "x": xs, "box": h, "r0": r0, "arg": arg, "diag": diag, "xi0": sj(xi0),
                           "order": names, "params": pn, "vals": [sj(v) for v in vals],
                           "ders": [[sj(v) for v in dl] for dl in ders]}, fh, indent=1)
        return 0
    if "box" in opts or "cbox" in opts:
        Maj.BOX = False
        c0 = run_M([acb(v) for v in xs], M, r0, arg, sector, verbose=False, kcap=kcap, diag=diag)
        Maj.BOX = True
        if "box" in opts:
            h = arb(opts["box"])
            x = [acb(arb(v, h)) for v in xs]
            print(f"real box x +- {opts['box']} (N, k and vanishing orders from the centre)")
        else:
            c = arb(opts["cbox"])
            x = [acb(arb(v, c), arb(0, c)) for v in xs]
            print(f"complex box x + {opts['cbox']}(1 + i) (N, k and vanishing orders from the centre)")
        out = run_M(x, M, r0, arg, sector, kmins_in=c0["kmins"], N_in=c0["N"], k_in=c0["k"], diag=diag)
        Maj.BOX = False
    else:
        out = run_M([acb(v) for v in xs], M, r0, arg, sector, kcap=kcap, diag=diag, mutate=mutate)
    if mutate:
        print(f"MUTATION {mutate}: the lemma holds: {out['ok']} (must be False)")
        return 0 if not out["ok"] else 3
    if "values" in opts:
        for sg in (+1, -1):
            xi0, Dv, Sv, Cv = start_values(out, sg)
            print(f"start values at xi0 = {xi0.str(8)}:")
            for mu in out["sa"].odd:
                print(f"   D{mu:+d} = {Dv[mu].str(14)}")
            for mu in out["sa"].odd:
                print(f"   S{mu:+d} = {Sv[mu].str(14)}")
            for nu in out["sa"].even:
                print(f"   C{nu:+d} = {Cv[nu].str(14)}")
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
