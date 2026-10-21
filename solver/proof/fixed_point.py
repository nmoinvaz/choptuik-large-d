#!/usr/bin/env python3
"""Rigorous start-value lemma (step i of proof/A_certify.md) for the one-harmonic (M = 1) inner problem.

Claim checked here, in ball arithmetic (python-flint arb/acb): on the ray z = xi^2 = r e^{i theta0}, 0 < r <= R
(theta0 = +3pi/4 or -3pi/4, i.e. arg xi = +-3pi/8, R = r0^2; with --sector=d, every ray |arg xi -+ 3pi/8| <= d), the
truncated inner system has a solution Y = Y_N + eta with Y_N the optimally truncated formal series and eta in an explicit
ball of a weighted space, and it is the only solution of the system with eta = o(r^(k+1)) as r -> 0 along the ray (every
bound below is non-decreasing in R, so the contraction holds on (0, R'] for all R' <= R, and the integral operator is of
Volterra type). The lateral sum y+ (y- on the lower ray) is such a solution, so y+(xi0) is enclosed.

Variables (z = xi^2, every harmonic projected as in lateral_np.py):
    D = xi P,  S = z Q,  V = Q - (1+s) P,  s = i mu w,  W = V + [(C-1)Q],  E^ = [P^2/2]_even
System (equivalent to (I1)-(I3) of A_certify.md for z != 0):
    P'  = (V + [(C-1)Q]) / (2z)
    V'  = V/(2z^2) + ([(C-1)Q] + [(C-1)W])/(2z^2) + (-(2+s)Q - (1+s)W + [C E^ P])/(2z)
    C'  = [C E^]/2
Only V/(2z^2) is singular at rank one; it is kept as the principal part, and the integral equation is
    p(z) = int_0^z Phi_p,   v(z) = int_0^z exp(1/(2t) - 1/(2z)) Phi_v,   c(z) = int_0^z Phi_c   (t along the ray)
with Phi = rho + F(Y_N + eta) - F(Y_N) - (0, v/(2z^2), 0) and rho = F(Y_N) - Y_N' the residual. On the ray
|exp(1/(2t) - 1/(2z))| = exp(-a (1/|t| - 1/|z|)), a = sqrt(2)/4, so the fast kernel is bounded and localizing:
    int_0^r exp(-a (1/rho - 1/r)) rho^m drho <= r^(m+2)/a   (m >= -2),   int_0^r rho^m drho = r^(m+1)/(m+1).
Norms: ||p||, ||v|| = sup |.|/r^k, ||c|| = sup |.|/r^(k+1); weighted max norm over the 7 components.
All functions are bounded by "majorants" K |t|^q on 0 < |t| <= R (sums of |coefficients|), products and the harmonic
convolutions by the triangle inequality, so every bound is rigorous. The residual numerators vanish identically below
their first nonzero order (they are combinations of the order-by-order equations the recursion solves exactly); the
computation checks that each of those coefficient balls contains 0.

    python3 fixed_point.py [r0] [--box=rhoA,rhow] [--sector=d]
        default r0 = 0.1; --box: complex parameter box radii; --sector: prove it on every ray |arg xi -+ 3pi/8| <= d
"""
import sys
import os
import numpy as np
from flint import arb, acb, acb_poly, ctx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from series_arb import SeriesArb  # noqa: E402

A_STAR, WF_STAR = "2.5421480475", "0.7769940136"


def up(x):
    """rigorous upper bound of |x| as an exact arb"""
    return arb(abs(x).upper())


class Maj:
    """|f(t)| <= K |t|^q for 0 < |t| <= R"""
    R = None
    BOX = False

    def __init__(self, K, q):
        self.K, self.q = arb(K), q

    def __add__(self, o):
        if o is None:
            return self
        q = min(self.q, o.q)
        return Maj(self.K * Maj.R ** (self.q - q) + o.K * Maj.R ** (o.q - q), q)

    __radd__ = __add__

    def __mul__(self, o):
        if o is None:
            return None
        if isinstance(o, Maj):
            return Maj(self.K * o.K, self.q + o.q)
        return Maj(self.K * up(o), self.q)

    def shift(self, m):
        return Maj(self.K, self.q - m)


def madd(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def mmul(a, b):
    if a is None or b is None:
        return None
    return a * b


class MD:
    """majorant of a value and of its derivatives in the 7 directions (absolute values throughout)"""

    def __init__(self, val, der=None):
        self.val, self.der = val, dict(der or {})

    def __add__(self, o):
        keys = set(self.der) | set(o.der)
        return MD(madd(self.val, o.val), {k: madd(self.der.get(k), o.der.get(k)) for k in keys})

    def __mul__(self, o):
        if isinstance(o, MD):
            keys = set(self.der) | set(o.der)
            der = {k: madd(mmul(self.val, o.der.get(k)), mmul(self.der.get(k), o.val)) for k in keys}
            return MD(mmul(self.val, o.val), der)
        return MD(mmul(self.val, o) if self.val is not None else None, {k: mmul(v, o) for k, v in self.der.items()})

    def shift(self, m):
        return MD(self.val.shift(m) if self.val is not None else None,
                  {k: (v.shift(m) if v is not None else None) for k, v in self.der.items()})


def conv(Ad, Bd, keep):
    out = {}
    for i, a in Ad.items():
        for j, b in Bd.items():
            if i + j in keep:
                t = a * b
                out[i + j] = t if i + j not in out else out[i + j] + t
    return out


def poly_maj(p):
    """K |t|^0 majorant of a polynomial in z on |z| <= R"""
    return Maj(sum((up(c) * Maj.R ** k for k, c in enumerate(p.coeffs())), arb(0)), 0)


def setup(A, w, r0, prec=320, N_in=None):
    """formal series (A, w acb, possibly boxes), optimal truncation at |xi| = r0 (or the given N), polynomials in z"""
    ctx.prec = prec
    sa = SeriesArb({1: -acb(0, 1) * A / 2, -1: acb(0, 1) * A / 2}, w, prec=prec)
    J = int(1 / (2 * float(r0) ** 2)) + 30
    d, s, g = sa.series(J)
    R = arb(r0) ** 2
    mags = [max(abs(d[j][mu]) for mu in sa.odd) * R ** j for j in range(J)]
    N = N_in or min(range(1, J), key=lambda j: float(mags[j].mid()))
    P = {mu: acb_poly([d[j][mu] for j in range(N)]) for mu in sa.odd}
    Q = {mu: acb_poly([s[j][mu] for j in range(N)]) for mu in sa.odd}
    C = {nu: acb_poly([g[j][nu] for j in range(N)]) for nu in sa.even}
    return sa, N, P, Q, C, R


def pconv(Ad, Bd, keep):
    out = {k: acb_poly([0]) for k in keep}
    for i, a in Ad.items():
        for j, b in Bd.items():
            if i + j in out:
                out[i + j] = out[i + j] + a * b
    return out


def residual_numerators(sa, P, Q, C):
    """numerators: rho_p = Np/(2z), rho_v = Nv/(2z^2), rho_c = Nc (exact polynomials in ball arithmetic)"""
    I, w, odd, even = sa.I, sa.w, sa.odd, sa.even
    X = acb_poly([0, 1])
    sm = {mu: I * mu * w for mu in odd}
    V = {mu: Q[mu] - (1 + sm[mu]) * P[mu] for mu in odd}
    Cm1 = {nu: C[nu] - (1 if nu == 0 else 0) for nu in even}
    CQ = pconv(Cm1, Q, odd)
    Wd = {mu: V[mu] + CQ[mu] for mu in odd}
    CW = pconv(Cm1, Wd, odd)
    Eh = {k: v * acb(0.5) for k, v in pconv(P, P, even).items()}
    CE = pconv(C, Eh, even)
    CEP = pconv(CE, P, odd)
    Np = {mu: V[mu] + CQ[mu] - 2 * X * P[mu].derivative() for mu in odd}
    Nv = {mu: V[mu] + CQ[mu] + CW[mu] + X * (-(2 + sm[mu]) * Q[mu] - (1 + sm[mu]) * Wd[mu] + CEP[mu])
          - 2 * X * X * V[mu].derivative() for mu in odd}
    Nc = {nu: CE[nu] * acb(0.5) - C[nu].derivative() for nu in even}
    return Np, Nv, Nc, V


def tail_maj(p, shift, scale, kmin=None):
    """majorant of p(t)/t^shift * scale, using that coefficients below order kmin vanish identically. kmin is detected
    at a parameter point (first coefficient ball not containing 0); for a parameter box it must be passed in (from the
    box centre: the vanishing orders are structural, the same for every parameter value), since wide balls contain 0"""
    cs = p.coeffs()
    if kmin is None:
        kmin = next((k for k, c in enumerate(cs) if not c.contains(0)), len(cs))
    # consistency check of the identity: each lower coefficient is a ball containing 0 whose contribution on |t| <= R is
    # negligible (rounding only); the lemma (exact recursion) says it is exactly 0
    # (for parameter boxes the balls widen, so only containment of 0 is required there)
    low_ok = all(c.contains(0) and (Maj.BOX or float((abs(c) * Maj.R ** k).upper()) < 1e-60)
                 for k, c in enumerate(cs[:kmin]))
    K = sum((up(c) * Maj.R ** (k - kmin) for k, c in enumerate(cs) if k >= kmin), arb(0))
    return Maj(K * scale, kmin - shift), kmin, low_ok


A_FAST = arb(2).sqrt() / 4          # -cos(3pi/4)/2: on the ray arg z = +-3pi/4


def set_sector(delta_xi):
    """make every bound hold uniformly on the closed sectors |arg xi -+ 3pi/8| <= delta_xi: on the ray arg z = theta the
    kernel rate is a = -cos(theta)/2, smallest at |theta -+ 3pi/4| = 2 delta_xi, a = (sqrt 2/4)(cos 2d - sin 2d); every other
    bound is a sum of |coefficients| and does not depend on the direction"""
    global A_FAST
    d2 = 2 * arb(delta_xi)
    A_FAST = (arb(2).sqrt() / 4 * (d2.cos() - d2.sin())).lower()
    A_FAST = arb(A_FAST)
    if not A_FAST > 0:
        raise ValueError("sector too wide: the fast kernel is no longer bounded")


def intbound(m, kind, k_i):
    """weighted-norm bound (sup over 0 < r <= R of r^-k_i * |integral|) of an integrand majorant m"""
    if m is None:
        return arb(0)
    q, K, R = m.q, m.K, Maj.R
    best = None
    if q > -1 and q + 1 >= k_i:
        best = K * R ** (q + 1 - k_i) / (q + 1)
    if kind == "fast" and q >= -2 and q + 2 >= k_i:
        b2 = K * R ** (q + 2 - k_i) / A_FAST
        best = b2 if best is None else (b2 if b2 < best else best)
    if best is None:
        raise ValueError(f"integrand t^{q} not admissible for weight {k_i} ({kind})")
    return best


def lipschitz_and_forcing(sa, P, Q, C, k, rho, wts):
    """returns (L, F, comps): Lipschitz matrix and forcing in the weighted norms, value majorants over the ball
    ||eta_j|| <= rho / w_j"""
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
    Fv = {mu: ((CQ[mu] + CW[mu]).shift(2) + (Qm[mu] * (2 + sm[mu]) + Wm[mu] * (1 + sm[mu]) + CEP[mu]).shift(1)) * arb(0.5)
          for mu in odd}
    Fc = {nu: CE[nu] * arb(0.5) for nu in even}
    rows = {("p", mu): (Fp[mu], "slow") for mu in odd}
    rows.update({("v", mu): (Fv[mu], "fast") for mu in odd})
    rows.update({("c", nu): (Fc[nu], "slow") for nu in even})
    L = [[intbound(rows[ci][0].der.get(cj), rows[ci][1], kw[ci]) for cj in comps] for ci in comps]
    return L, comps, kw, rows


def run(A, wf, r0="0.1", prec=320, verbose=True, kmins_in=None, N_in=None):
    ctx.prec = prec
    W0 = 2 * arb.pi() / arb("1.64389375")
    w = acb(W0) * acb(wf)
    sa, N, P, Q, C, R = setup(acb(A), w, r0, prec, N_in)
    Maj.R = R
    Np, Nv, Nc, V = residual_numerators(sa, P, Q, C)
    res = {}
    kmins, lows = [], []
    kin = iter(kmins_in) if kmins_in else None
    for mu in sa.odd:
        m, km, ok = tail_maj(Np[mu], 1, arb(0.5), next(kin) if kin else None)
        res[("p", mu)] = m; kmins.append(km); lows.append(ok)
        m, km, ok = tail_maj(Nv[mu], 2, arb(0.5), next(kin) if kin else None)
        res[("v", mu)] = m; kmins.append(km); lows.append(ok)
    for nu in sa.even:
        m, km, ok = tail_maj(Nc[nu], 0, arb(1), next(kin) if kin else None)
        res[("c", nu)] = m; kmins.append(km); lows.append(ok)
    # weight exponent: largest k admissible for every forcing term
    k = min(res[("p", mu)].q + 1 for mu in sa.odd)
    k = min(k, min(res[("v", mu)].q + 2 for mu in sa.odd), min(res[("c", nu)].q + 1 - 1 for nu in sa.even))
    k = min(k, 40)
    comps0 = [("p", mu) for mu in sa.odd] + [("v", mu) for mu in sa.odd] + [("c", nu) for nu in sa.even]
    # first pass: unit weights, tiny ball, to get the Lipschitz matrix shape
    wts = [1.0] * len(comps0)
    L, comps, kw, rows = lipschitz_and_forcing(sa, P, Q, C, k, arb("1e-20"), wts)
    Lf = np.array([[float(x.upper()) for x in row] for row in L]) + 1e-12
    ev, evec = np.linalg.eig(Lf)
    i = int(np.argmax(ev.real))
    u = np.abs(evec[:, i].real) + 1e-6 * np.abs(evec[:, i].real).max()
    wts = list(1.0 / u)
    Fv = [intbound(res[cp], "fast" if cp[0] == "v" else "slow", kw[cp]) for cp in comps]
    F = max(arb(wts[j]) * Fv[j] for j in range(len(comps)))
    # Lipschitz in the weighted max norm with the ball radius rho = 2 F / (1 - lip0)
    lip0 = max(sum(arb(wts[i_]) * L[i_][j] / arb(wts[j]) for j in range(len(comps))) for i_ in range(len(comps)))
    rho = 4 * F / (1 - lip0) if lip0 < 1 else arb(1)
    L, comps, kw, rows = lipschitz_and_forcing(sa, P, Q, C, k, rho, wts)
    lip = max(sum(arb(wts[i_]) * L[i_][j] / arb(wts[j]) for j in range(len(comps))) for i_ in range(len(comps)))
    selfmap = F + lip * rho
    ok = bool(lip < 1) and bool(selfmap < rho) and all(lows)
    out = dict(N=N, k=k, kmins=kmins, low_ok=all(lows), lip=lip, F=F, rho=rho, selfmap=selfmap, ok=ok,
               wts=wts, comps=comps, kw=kw, L=L, sa=sa, P=P, Q=Q, C=C, R=R, r0=arb(r0))
    if verbose:
        print(f"A = {A}, w/w_inf = {wf}, r0 = {r0} (R = r0^2): N = {N} terms, weight exponent k = {k}")
        print(f"  residual numerators: first nonzero orders {kmins}; all lower coefficients contain 0: {all(lows)}")
        print(f"  forcing ||T(0)||_w <= {float(F.upper()):.3e}")
        print("  Lipschitz matrix (rows/cols " + ", ".join(f"{t}{m:+d}" for t, m in comps) + "):")
        for i_, row in enumerate(L):
            print("    " + " ".join(f"{float(x.upper()):9.2e}" for x in row))
        print(f"  weights {np.round(wts, 4).tolist()}")
        print(f"  contraction constant (weighted max norm) <= {float(lip.upper()):.4f}")
        print(f"  ball radius rho = {float(rho.upper()):.3e}: ||T(0)|| + Lip rho = {float(selfmap.upper()):.3e} "
              f"<= rho: {bool(selfmap < rho)}")
        print(f"  => fixed point exists and is unique in the ball: {ok}")
    return out


def start_values(out, theta_sign=+1):
    """enclosures of y+(xi0) (D, S on odd mu, C on even nu) at xi0 = r0 e^{+-3 i pi/8}, from Y_N + eta"""
    sa, P, Q, C, R, r0, k = out["sa"], out["P"], out["Q"], out["C"], out["R"], out["r0"], out["k"]
    phi = theta_sign * 3 * arb.pi() / 8
    xi0 = acb(r0 * phi.cos(), r0 * phi.sin())
    z0 = xi0 * xi0
    wts, comps, rho = out["wts"], out["comps"], out["rho"]
    eb = {cp: rho / arb(wts[i]) * R ** (k + 1 if cp[0] == "c" else k) for i, cp in enumerate(comps)}
    sm = {mu: sa.I * mu * sa.w for mu in sa.odd}
    Dv, Sv, Cv = {}, {}, {}
    for mu in sa.odd:
        e_p = eb[("p", mu)]
        e_q = eb[("v", mu)] + up(1 + sm[mu]) * e_p
        Dv[mu] = xi0 * P[mu](z0) + acb(arb(0, (r0 * e_p).upper()), arb(0, (r0 * e_p).upper()))
        Sv[mu] = z0 * Q[mu](z0) + acb(arb(0, (R * e_q).upper()), arb(0, (R * e_q).upper()))
    for nu in sa.even:
        e_c = eb[("c", nu)]
        Cv[nu] = C[nu](z0) + acb(arb(0, e_c.upper()), arb(0, e_c.upper()))
    return xi0, Dv, Sv, Cv


class Jet:
    """first-order jet in the real parameters (A, wf): value and partial derivatives, all acb balls"""

    def __init__(self, v, dA=None, dw=None):
        self.v = acb(v)
        self.dA = acb(0) if dA is None else acb(dA)
        self.dw = acb(0) if dw is None else acb(dw)

    def __add__(self, o):
        o = o if isinstance(o, Jet) else Jet(o)
        return Jet(self.v + o.v, self.dA + o.dA, self.dw + o.dw)

    __radd__ = __add__

    def __sub__(self, o):
        o = o if isinstance(o, Jet) else Jet(o)
        return Jet(self.v - o.v, self.dA - o.dA, self.dw - o.dw)

    def __rsub__(self, o):
        return Jet(o) - self

    def __neg__(self):
        return Jet(-self.v, -self.dA, -self.dw)

    def __mul__(self, o):
        if isinstance(o, Jet):
            return Jet(self.v * o.v, self.v * o.dA + self.dA * o.v, self.v * o.dw + self.dw * o.v)
        o = acb(o)
        return Jet(self.v * o, self.dA * o, self.dw * o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = acb(o)
        return Jet(self.v / o, self.dA / o, self.dw / o)


def series_jet(A, wf, J, M=1):
    """the recursion of series_arb.SeriesArb.series with Jet values: coefficients and their (A, wf) derivatives.
    A, wf: acb (balls allowed: then the derivatives are enclosed over the box)"""
    W0 = 2 * arb.pi() / arb("1.64389375")
    I = acb(0, 1)
    w = Jet(acb(W0) * acb(wf), 0, acb(W0))
    Aj = Jet(acb(A), 1, 0)
    odd = [mu for mu in range(-M, M + 1) if mu % 2]
    even = [nu for nu in range(-(M + 1), M + 2) if nu % 2 == 0]
    Fh = {1: Aj * (-I / 2), -1: Aj * (I / 2)}

    def mul(Ad, Bd, keep):
        out = {k: Jet(0) for k in keep}
        for i, ai in Ad.items():
            for j, bj in Bd.items():
                if i + j in out:
                    out[i + j] = out[i + j] + ai * bj
        return out

    d = [{mu: Fh[mu] * 2 for mu in odd}]
    s, g = [], [{nu: Jet(1 if nu == 0 else 0) for nu in even}]
    e = [{nu: Jet(0) for nu in even}]
    ce = [{nu: Jet(0) for nu in even}]
    for j in range(J):
        sj = {mu: d[j][mu] * (2 * j + 1) + d[j][mu] * w * (I * mu) for mu in odd}
        for i in range(1, j + 1):
            t = mul(g[i], s[j - i], odd)
            for mu in odd:
                sj[mu] = sj[mu] - t[mu]
        s.append(sj)
        m = j + 1
        em = {nu: Jet(0) for nu in even}
        for i in range(m):
            t = mul(d[i], d[m - 1 - i], even)
            for nu in even:
                em[nu] = em[nu] + t[nu] / 2
        e.append(em)
        cm = {nu: Jet(0) for nu in even}
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


def start_data(A, wf, r0="0.1", theta_sign=+1, cauchy=None, centre=None):
    """C^1 start data for the validated integration (the interface step (ii) consumes), over a parameter box:
    value Y(xi0) and derivatives dY/dA, dY/dwf, in lateral_np's state order [D_-1, D_+1, S_-1, S_+1, C_-2, C_0, C_+2].
    A, wf: real balls (the box X). Y = Y_N + eta: Y_N and its derivatives exactly by jets over the box; eta bounded by
    the fixed point on the complex box X + cauchy*(1+i), its derivatives by Cauchy's estimate |d eta| <= sup|eta|/cauchy.
    Returns (xi0, value list, dA list, dw list, eta bound)."""
    ctx.prec = 320
    A, wf = arb(A), arb(wf)
    if cauchy is None:
        cauchy = arb("1e-2")
    cA = acb(A + arb(0, cauchy), arb(0, cauchy))
    cw = acb(wf + arb(0, cauchy), arb(0, cauchy))
    if centre is None:
        Maj.BOX = False
        centre = run(acb(A.mid()), acb(wf.mid()), r0, verbose=False)
    Maj.BOX = A.rad() > 0 or wf.rad() > 0
    outX = run(acb(A), acb(wf), r0, verbose=False, kmins_in=centre["kmins"], N_in=centre["N"])     # values over X
    Maj.BOX = True
    outC = run(cA, cw, r0, verbose=False, kmins_in=centre["kmins"], N_in=centre["N"])              # Cauchy box
    Maj.BOX = False
    if not (outX["ok"] and outC["ok"]):
        raise RuntimeError("fixed point not verified")

    def eta_bounds(out):
        R_, k_, rho_, wts_, comps_ = out["R"], out["k"], out["rho"], out["wts"], out["comps"]
        return {cp: rho_ / arb(wts_[i]) * R_ ** (k_ + 1 if cp[0] == "c" else k_) for i, cp in enumerate(comps_)}
    N, R = centre["N"], outX["R"]
    eb, ebC = eta_bounds(outX), eta_bounds(outC)
    odd, even, d, s, g = series_jet(acb(A), acb(wf), N)
    phi = theta_sign * 3 * arb.pi() / 8
    r0a = arb(r0)
    xi0 = acb(r0a * phi.cos(), r0a * phi.sin())
    z0 = xi0 * xi0
    zp = [acb(1)]
    for _ in range(N):
        zp.append(zp[-1] * z0)
    W0 = 2 * arb.pi() / arb("1.64389375")
    sm_up = {mu: up(1 + acb(0, 1) * mu * acb(W0) * cw) for mu in odd}
    vals, dAs, dws = [], [], []

    def ball(e):
        r = arb(0, e.upper())
        return acb(r, r)

    def add_eta(jv, e, eC):
        """value + eta over X; derivatives + Cauchy bound sup|eta| / cauchy (eta holomorphic in the parameters)"""
        return jv.v + ball(e), jv.dA + ball(eC / cauchy), jv.dw + ball(eC / cauchy)

    for mu in odd:
        Dj = sum((d[j][mu] * zp[j] for j in range(N)), Jet(0)) * xi0
        v, a, w_ = add_eta(Dj, r0a * eb[("p", mu)], r0a * ebC[("p", mu)])
        vals.append(v); dAs.append(a); dws.append(w_)
    for mu in odd:
        Sj = sum((s[j][mu] * zp[j] for j in range(N)), Jet(0)) * z0
        v, a, w_ = add_eta(Sj, R * (eb[("v", mu)] + sm_up[mu] * eb[("p", mu)]),
                           R * (ebC[("v", mu)] + sm_up[mu] * ebC[("p", mu)]))
        vals.append(v); dAs.append(a); dws.append(w_)
    for nu in even:
        Cj = sum((g[j][nu] * zp[j] for j in range(N)), Jet(0))
        v, a, w_ = add_eta(Cj, eb[("c", nu)], ebC[("c", nu)])
        vals.append(v); dAs.append(a); dws.append(w_)
    etab = max(eb.values(), key=lambda x: float(x.upper()))
    etabC = max(ebC.values(), key=lambda x: float(x.upper()))
    return xi0, vals, dAs, dws, (etab, etabC / cauchy)


if __name__ == "__main__":
    r0 = next((a for a in sys.argv[1:] if not a.startswith("--")), "0.1")
    box = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--box=")), None)
    sec = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--sector=")), None)
    if sec:
        set_sector(sec)
        print(f"sector: every ray |arg xi -+ 3pi/8| <= {sec} (arg z within 2*{sec} of +-3pi/4); fast-kernel rate a = "
              f"{A_FAST.str(8)} (sqrt(2)/4 = 0.35355339 on the central ray)")
    ctx.prec = 320
    Maj.BOX = bool(box)
    kmins_in = None
    if box:
        rA, rw = (arb(x) for x in box.split(","))
        A = acb(arb(A_STAR, rA), arb(0, rA))
        wf = acb(arb(WF_STAR, rw), arb(0, rw))
        print(f"complex parameter box: A = {A_STAR} + [+-{box.split(',')[0]}](1+i), w/w_inf = {WF_STAR} + [+-{box.split(',')[1]}](1+i)")
        Maj.BOX = False
        c0 = run(acb(A_STAR), acb(WF_STAR), r0, verbose=False)     # structural truncation and vanishing orders
        kmins_in, N_in = c0["kmins"], c0["N"]
        Maj.BOX = True
        print(f"  truncation N = {N_in} and vanishing orders {kmins_in} taken from the box centre")
    else:
        A, wf = acb(A_STAR), acb(WF_STAR)
        N_in = None
    out = run(A, wf, r0, kmins_in=kmins_in, N_in=N_in)
    for sgn in (+1, -1):
        xi0, Dv, Sv, Cv = start_values(out, sgn)
        print(f"start values at xi0 = {r0} e^({'+' if sgn > 0 else '-'}3 i pi/8):")
        def fmt(z):
            return f"{z.real.mid().str(14)} + {z.imag.mid().str(14)}i (radii {float(z.real.rad()):.1e}, {float(z.imag.rad()):.1e})"
        for mu in out["sa"].odd:
            print(f"   D_{mu:+d} = {fmt(Dv[mu])}")
            print(f"   S_{mu:+d} = {fmt(Sv[mu])}")
        for nu in out["sa"].even:
            print(f"   C_{nu:+d} = {fmt(Cv[nu])}")
