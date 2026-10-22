#!/usr/bin/env python3
"""The truncated M-harmonic inner system for the M-harmonic certificate (proof/A_M.md), in ball arithmetic.

The system is the one solver/inner_stokes/inner_np.py and lateral_np.py integrate, for any odd M:
    xi^2 D'  + i mu w xi D  = [C S]_mu
    xi^2 S'  + i mu w xi S  = [C xi D']_mu + [(C E) D]_mu - [C D]_mu,     E = [D D]/2
    xi C'                   = [C E]_nu
with D, S on the odd harmonics |mu| <= M and C, E on the even harmonics |nu| <= M + 1, every product projected onto
its result's harmonics. State order (as lateral_np.NP.pack): [D_mu (mu = -M..M odd), S_mu, C_nu (nu = -(M+1)..M+1
even)], then w (a constant state). n = 3M + 4 complex unknowns: 7 at M = 1, 13 at M = 3, 19 at M = 5. The profile
F = A sin t + sum_{nu >= 3} (B_nu sin nu t + C_nu cos nu t) enters only the start values (D = 2 F xi + O(xi^3)).

At M = 1 the class SysM is validated_int.Sys without its second constant state A (A enters only the start data): the
right-hand sides and linearizations agree to rounding (self-test below). validated_int.Lohner (the QR-Lohner
integrator of proof/A_certify.md section 9) works with SysM unchanged; it only uses sys_.n, sys_.nd, rhs, lin.

Also here:
  * LegM: the path legs of validated_int.Leg with the start-ray angle phi0 as a parameter (validated_int hard-codes
    3 pi/8): ray xi = (r0 + (r1 - r0) t) e^{i phi0}, arc xi = r1 e^{i phi0 (1 - t)}, real xi = r0 + (r1 - r0) t.
  * series_generic_M: the formal series recursion (series_arb.SeriesArb, validated_int.series_generic) for any M and
    any number type, so acb_series jets give exact parameter derivatives of the truncated series.
  * placeholder_start: NON-RIGOROUS start data for M >= 3 (the optimally truncated series plus an ASSUMED error ball),
    used until the M-harmonic start-value lemma (solver/proof/fixed_point_M.py, another agent) exists. Every result
    obtained with it is labelled "placeholder start values" and is not a certificate.

    python3 certify_M_sys.py selftest          rhs/lin against validated_int.Sys (M = 1) and inner_np.Inner.rhs (M = 3, 5);
                                               series against series_arb.SeriesArb; jets against finite differences
    python3 certify_M_sys.py selftest --mutate=coupling   the same with one coupling sign flipped: must FAIL
"""
import os
import sys
import time

import numpy as np
from flint import acb, arb, acb_series, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

I = acb(0, 1)
W0_STR = "1.64389375"          # w_inf := 2 pi / 1.64389375, exact from the decimal string (the codes' convention)
MUTATE = set()                 # self-test mutations ("coupling")


def W0():
    """w_inf = 2 pi / 1.64389375 at the current precision"""
    return 2 * arb.pi() / arb(W0_STR)


def harmonics(M):
    odd = [mu for mu in range(-M, M + 1) if mu % 2]
    even = [nu for nu in range(-(M + 1), M + 2) if nu % 2 == 0]
    return odd, even


def pr(X, Y, out, zero):
    """projected product: {k: sum_{a+b=k} X_a Y_b} for k in out (validated_int.pr)"""
    r = {}
    for k in out:
        acc = zero
        for a, xa in X.items():
            yb = Y.get(k - a)
            if yb is not None:
                acc = acc + xa * yb
        r[k] = acc
    return r


class SysM:
    """state: D[mu], S[mu] (odd |mu| <= M), C[nu] (even |nu| <= M + 1) and w; packed as a list of nd + 1"""

    def __init__(self, M):
        self.M = M
        self.ODD, self.EVEN = harmonics(M)
        self.no, self.ne = len(self.ODD), len(self.EVEN)
        self.keys = [("D", m) for m in self.ODD] + [("S", m) for m in self.ODD] + [("C", v) for v in self.EVEN]
        self.nd = len(self.keys)
        self.n = self.nd + 1
        self.idx = {k: i for i, k in enumerate(self.keys)}
        self.OSET, self.ESET = set(self.ODD), set(self.EVEN)

    def unpack(self, Y):
        no, ne = self.no, self.ne
        D = {m: Y[i] for i, m in enumerate(self.ODD)}
        S = {m: Y[no + i] for i, m in enumerate(self.ODD)}
        C = {v: Y[2 * no + i] for i, v in enumerate(self.EVEN)}
        return D, S, C, Y[self.nd]

    def pack(self, D, S, C, w):
        return [D[m] for m in self.ODD] + [S[m] for m in self.ODD] + [C[v] for v in self.EVEN] + [w]

    def rhs(self, Y, xi, ixi, zero):
        """dY/dxi (w constant) and the value-level products needed by the linearization"""
        ODD, EVEN = self.ODD, self.EVEN
        D, S, C, w = self.unpack(Y)
        ixi2 = ixi * ixi
        CS = pr(C, S, ODD, zero)
        dD = {m: (CS[m] - I * m * w * xi * D[m]) * ixi2 for m in ODD}
        E = {k: v / 2 for k, v in pr(D, D, EVEN, zero).items()}
        cE = pr(C, E, EVEN, zero)
        xdD = {m: xi * dD[m] for m in ODD}
        a, b, c = pr(C, xdD, ODD, zero), pr(cE, D, ODD, zero), pr(C, D, ODD, zero)
        if "coupling" in MUTATE:
            r2 = {m: a[m] - b[m] - c[m] for m in ODD}
        else:
            r2 = {m: a[m] + b[m] - c[m] for m in ODD}
        dC = {v: cE[v] * ixi for v in EVEN}
        dS = {m: (r2[m] - I * m * w * xi * S[m]) * ixi2 for m in ODD}
        return self.pack(dD, dS, dC, zero), (D, S, C, w, E, cE, dD)

    def lin(self, cache, V, xi, ixi, zero):
        """derivative of rhs at the cached point in the direction V (nd + 1 components, V[-1] = dw)"""
        ODD, EVEN = self.ODD, self.EVEN
        D, S, C, w, E, cE, dD = cache
        dD_, dS_, dC_, dw = self.unpack(V)
        ixi2 = ixi * ixi
        p1, p2 = pr(dC_, S, ODD, zero), pr(C, dS_, ODD, zero)
        ddD = {m: (p1[m] + p2[m] - I * m * (dw * xi * D[m] + w * xi * dD_[m])) * ixi2 for m in ODD}
        dE = pr(D, dD_, EVEN, zero)
        q1, q2 = pr(dC_, E, EVEN, zero), pr(C, dE, EVEN, zero)
        dcE = {v: q1[v] + q2[v] for v in EVEN}
        xdD = {m: xi * dD[m] for m in ODD}
        xddD = {m: xi * ddD[m] for m in ODD}
        t = [pr(dC_, xdD, ODD, zero), pr(C, xddD, ODD, zero), pr(dcE, D, ODD, zero), pr(cE, dD_, ODD, zero),
             pr(dC_, D, ODD, zero), pr(C, dD_, ODD, zero)]
        sg = -1 if "coupling" in MUTATE else 1
        dr2 = {m: t[0][m] + t[1][m] + sg * (t[2][m] + t[3][m]) - t[4][m] - t[5][m] for m in ODD}
        ddC = {v: dcE[v] * ixi for v in EVEN}
        ddS = {m: (dr2[m] - I * m * (dw * xi * S[m] + w * xi * dS_[m])) * ixi2 for m in ODD}
        return self.pack(ddD, ddS, ddC, zero)


    def lin_unit(self, cache, j, xi, ixi, zero):
        """lin(cache, e_j): the same formulas with sparse dicts (a missing key is an exact zero), so that products with
        zero series are skipped. Self-tested against lin."""
        ODD, EVEN = self.OSET, self.ESET
        D, S, C, w, E, cE, dD = cache
        dD_, dS_, dC_, dw = {}, {}, {}, None
        if j < self.nd:
            kind, h = self.keys[j]
            {"D": dD_, "S": dS_, "C": dC_}[kind][h] = None      # placeholder, replaced by the unit series below
        one = zero + 1
        for dct in (dD_, dS_, dC_):
            for k in dct:
                dct[k] = one
        if j == self.nd:
            dw = one
        p = spadd(spr(dC_, S, ODD), spr(C, dS_, ODD))
        keys = set(p) | set(dD_) | (set(self.ODD) if dw is not None else set())
        ddD = {}
        for m in keys:
            t = p.get(m, zero)
            u = None
            if dw is not None:
                u = dw * xi * D[m]
            if m in dD_:
                u = (u + w * xi * dD_[m]) if u is not None else w * xi * dD_[m]
            if u is not None:
                t = t - I * m * u
            ddD[m] = t * (ixi * ixi)
        dE = spr(D, dD_, EVEN)
        dcE = spadd(spr(dC_, E, EVEN), spr(C, dE, EVEN))
        xdD = {m: xi * dD[m] for m in self.ODD}
        xddD = {m: xi * v for m, v in ddD.items()}
        dr2 = spadd(spr(dC_, xdD, ODD), spr(C, xddD, ODD), spr(dcE, D, ODD), spr(cE, dD_, ODD))
        dr2 = spsub(dr2, spadd(spr(dC_, D, ODD), spr(C, dD_, ODD)))
        if "coupling" in MUTATE:
            dr2 = spsub(spsub(dr2, spr(dcE, D, ODD)), spadd(spr(dcE, D, ODD), spr(cE, dD_, ODD), spr(cE, dD_, ODD)))
        ddC = {v: dcE[v] * ixi for v in dcE}
        keysS = set(dr2) | set(dS_) | (set(self.ODD) if dw is not None else set())
        ddS = {}
        for m in keysS:
            t = dr2.get(m, zero)
            u = None
            if dw is not None:
                u = dw * xi * S[m]
            if m in dS_:
                u = (u + w * xi * dS_[m]) if u is not None else w * xi * dS_[m]
            if u is not None:
                t = t - I * m * u
            ddS[m] = t * (ixi * ixi)
        return ([ddD.get(m, zero) for m in self.ODD] + [ddS.get(m, zero) for m in self.ODD]
                + [ddC.get(v, zero) for v in self.EVEN] + [zero])


def spr(X, Y, out):
    """sparse projected product: {k: sum_{a+b=k} X_a Y_b} over the nonzero entries only, k in the set out"""
    r = {}
    for a, xa in X.items():
        for b, yb in Y.items():
            k = a + b
            if k in out:
                r[k] = r[k] + xa * yb if k in r else xa * yb
    return r


def spadd(*ds):
    r = {}
    for d in ds:
        for k, v in d.items():
            r[k] = r[k] + v if k in r else v
    return r


def spsub(a, b):
    r = dict(a)
    for k, v in b.items():
        r[k] = r[k] - v if k in r else -v
    return r


class LegM:
    """xi(t), t in [0, 1] exactly: 'ray' xi = (r0 + (r1 - r0) t) e^{i phi0}; 'arc' xi = r1 e^{i phi0 (1 - t)};
    'real' xi = r0 + (r1 - r0) t. validated_int.Leg with phi0 a parameter (an arb)."""

    def __init__(self, kind, r0, r1, phi0):
        self.kind, self.r0, self.r1, self.phi0 = kind, arb(r0), arb(r1), arb(phi0)
        self.u = (I * self.phi0).exp() if kind == "ray" else acb(1)

    def xi(self, t):
        if self.kind in ("ray", "real"):
            return (acb(self.r0) + acb(self.r1 - self.r0) * t) * self.u
        return acb(self.r1) * (I * self.phi0 * (1 - t)).exp()

    def dxi(self, xi):
        return acb(self.r1 - self.r0) * self.u if self.kind in ("ray", "real") else -I * self.phi0 * xi

    def series(self, t0, L):
        x0 = self.xi(t0)
        if self.kind in ("ray", "real"):
            v = acb(self.r1 - self.r0) * self.u
            return acb_series([x0, v], prec=L), acb_series([v], prec=L)
        e = acb_series([0, -I * self.phi0], prec=L).exp()
        xs = x0 * e
        return xs, -I * self.phi0 * xs


def legs_for(path):
    """path = (r0, phi0, rho, xi1): ray r0 -> rho at phi0, arc at rho to the real axis, real axis rho -> xi1 (omitted
    when xi1 == rho). r0, rho, xi1 decimal strings; phi0 an arb."""
    r0, phi0, rho, xi1 = path
    L = [LegM("ray", r0, rho, phi0), LegM("arc", r0, rho, phi0)]
    if arb(xi1) != arb(rho):
        L.append(LegM("real", rho, xi1, phi0))
    return L


# ---------------------------------------------------------------- the profile and the formal series

def profile(p, M, one):
    """F_mu (mu odd, |mu| <= M) from p = (A, wf, B3, C3, ..., BM, CM): F_1 = A/(2i), F_nu = B/(2i) + C/2,
    F_-nu = conj(F_nu) (for real parameters). Entries of p may be acb or acb_series jets; 'one' fixes the type."""
    h = {1: p[0] * (-I / 2) * one, -1: p[0] * (I / 2) * one}
    for k, nu in enumerate(range(3, M + 1, 2)):
        B, Cc = p[2 + 2 * k], p[3 + 2 * k]
        h[nu] = B * (-I / 2) + Cc / 2
        h[-nu] = B * (I / 2) + Cc / 2
    return h


def series_generic_M(Fh, w, J, M):
    """coefficients d_j, s_j, g_j (j < J+1) of the formal series D = sum d_j xi^(2j+1), S = sum s_j xi^(2j+2),
    C = sum g_j xi^(2j), d_0 = 2F, g_0 = 1, for any number type (acb, or acb_series jets in a parameter). The recursion
    is validated_int.series_generic / series_arb.SeriesArb.series for general M."""
    odd, even = harmonics(M)
    z0 = w * 0
    zero, one = z0, z0 + 1

    def mul(X, Y, keep):
        out = {k: zero for k in keep}
        for i, xi in X.items():
            for j, yj in Y.items():
                if i + j in out:
                    out[i + j] = out[i + j] + xi * yj
        return out
    d = [{mu: 2 * Fh[mu] for mu in odd}]
    sl, g = [], [{nu: (one if nu == 0 else zero) for nu in even}]
    e = [{nu: zero for nu in even}]
    ce = [{nu: zero for nu in even}]
    for j in range(J):
        sj = {mu: (2 * j + 1 + I * mu * w) * d[j][mu] for mu in odd}
        for i in range(1, j + 1):
            t = mul(g[i], sl[j - i], odd)
            for mu in odd:
                sj[mu] = sj[mu] - t[mu]
        sl.append(sj)
        m = j + 1
        em = {nu: zero for nu in even}
        for i in range(m):
            t = mul(d[i], d[m - 1 - i], even)
            for nu in even:
                em[nu] = em[nu] + t[nu] / 2
        e.append(em)
        cm = {nu: zero for nu in even}
        for i in range(m):
            t = mul(g[i], e[m - i], even)
            for nu in even:
                cm[nu] = cm[nu] + t[nu]
        ce.append(cm)
        g.append({nu: cm[nu] / (2 * m) for nu in even})
        num = {mu: (2 * j + 2 + I * mu * w) * sl[j][mu] for mu in odd}
        for i in range(1, j + 2):
            t = mul(g[i], d[j + 1 - i], odd)
            for mu in odd:
                num[mu] = num[mu] - 2 * (j + 1 - i) * t[mu]
        for mm in range(1, j + 2):
            t = mul(ce[mm], d[j + 1 - mm], odd)
            for mu in odd:
                num[mu] = num[mu] - t[mu]
        d.append({mu: num[mu] / (2 * j + 2) for mu in odd})
    return d, sl, g


def series_state(d, sl, g, xi0, n, M):
    """the first n terms of the series at xi0, in SysM order (without w)"""
    odd, even = harmonics(M)
    x2 = xi0 * xi0
    z = d[0][odd[0]] * 0
    D, S, C = {}, {}, {}
    for mu in odd:
        acc_d, acc_s, pw = z, z, xi0
        for j in range(n):
            acc_d = acc_d + d[j][mu] * pw
            acc_s = acc_s + sl[j][mu] * pw * xi0
            pw = pw * x2
        D[mu], S[mu] = acc_d, acc_s
    for nu in even:
        acc, pw = z, acb(1)
        for j in range(n):
            acc = acc + g[j][nu] * pw
            pw = pw * x2
        C[nu] = acc
    return [D[m] for m in odd] + [S[m] for m in odd] + [C[v] for v in even]


def optimal_n(p, M, r0):
    """optimal truncation of the series at |xi| = r0 (smallest term of D over all harmonics); p: list of arb"""
    w = W0() * p[1]
    Fh = profile([acb(x) for x in p], M, acb(1))
    J = int(1 / (2 * float(r0) ** 2)) + 40
    d, _, _ = series_generic_M(Fh, acb(w), J, M)
    x2 = arb(r0) ** 2
    odd, _ = harmonics(M)
    mags = [max(float(abs(d[j][mu]).mid()) for mu in odd) * float((x2 ** j).mid()) for j in range(J)]
    jt = min(range(1, J), key=lambda j: mags[j])
    return jt, mags[jt]


def start_series(p, M, r0, phi0, n, dirs=None):
    """the truncated series at xi0 = r0 e^{i phi0} and its directional derivatives along the parameter directions
    dirs (list of (M+1)-vectors of arb; default the unit vectors), exact (jets), for parameters p (arb balls allowed:
    then everything is enclosed over the box). Returns (xi0, values, [derivative lists])."""
    m = M + 1
    if dirs is None:
        dirs = [[arb(1) if i == k else arb(0) for i in range(m)] for k in range(m)]
    xi0 = acb(arb(r0)) * (I * arb(phi0)).exp()
    W = W0()
    pa = [acb(x) for x in p]
    Fh = profile(pa, M, acb(1))
    d, sl, g = series_generic_M(Fh, acb(W * p[1]), n, M)
    vals = series_state(d, sl, g, xi0, n, M)
    ders = []
    ctx.cap = 2
    for dv in dirs:
        pj = [acb_series([pa[i], acb(dv[i])], prec=2) for i in range(m)]
        Fj = profile(pj, M, acb_series([1], prec=2))
        wj = acb_series([acb(W * p[1]), acb(W * dv[1])], prec=2)
        dj, slj, gj = series_generic_M(Fj, wj, n, M)
        vj = series_state(dj, slj, gj, acb_series([xi0], prec=2), n, M)
        ders.append([v.coeffs()[1] if len(v.coeffs()) > 1 else acb(0) for v in vj])
    return xi0, vals, ders


def placeholder_start(p, M, r0, phi0, dirs=None, eps_factor=100.0, verbose=False):
    """NON-RIGOROUS start data for M >= 3 (until fixed_point_M.py exists): the series truncated at its smallest term
    plus an ASSUMED ball eps0 = eps_factor * smallest term (values) and eps_factor * smallest term / r0^2 (derivatives).
    Same return shape as start_series, plus the assumed radii."""
    n, smallest = optimal_n([arb(x.mid()) for x in p], M, r0)
    xi0, vals, ders = start_series(p, M, r0, phi0, n, dirs)
    e0 = eps_factor * smallest * float(r0)
    e1 = eps_factor * smallest / float(r0) ** 2
    eb0 = acb(arb(0, e0), arb(0, e0))
    eb1 = acb(arb(0, e1), arb(0, e1))
    vals = [v + eb0 for v in vals]
    ders = [[v + eb1 for v in col] for col in ders]
    if verbose:
        print(f"   placeholder start values (NOT rigorous): series to {n} terms at r0 = {r0}, smallest term {smallest:.1e}, "
              f"assumed radii {e0:.1e} (values), {e1:.1e} (derivatives)", flush=True)
    return xi0, vals, ders, {"n": n, "smallest": smallest, "eps_val": e0, "eps_der": e1, "rigorous": False}


# ---------------------------------------------------------------- self-tests

def selftest():
    """rhs and lin of SysM against validated_int.Sys (M = 1), inner_np.Inner.rhs (M = 3, 5; double precision), and
    the series recursion against series_arb.SeriesArb (M = 3); jets against central differences. Returns True if all
    agree within their tolerances."""
    import validated_int as VI
    rng = np.random.default_rng(7)
    ok = True
    ctx.prec = 256
    # M = 1 against validated_int.Sys (with A appended as the second constant state there)
    s1, v1 = SysM(1), VI.Sys(gravity=True)
    Y = [acb(complex(*rng.normal(size=2))) for _ in range(s1.nd)] + [acb(3.1)]
    xi = acb(0.17, 0.05)
    F1, c1 = s1.rhs(Y, xi, 1 / xi, acb(0))
    F2, c2 = v1.rhs(Y + [acb(2.5)], xi, 1 / xi, acb(0))
    e1 = max(float(abs(F1[i] - F2[i]).upper()) for i in range(s1.nd))
    V = [acb(complex(*rng.normal(size=2))) for _ in range(s1.nd)] + [acb(0.7)]
    L1 = s1.lin(c1, V, xi, 1 / xi, acb(0))
    L2 = v1.lin(c2, V + [acb(0)], xi, 1 / xi, acb(0))
    e2 = max(float(abs(L1[i] - L2[i]).upper()) for i in range(s1.nd))
    good = e1 < 1e-60 and e2 < 1e-60
    ok &= good
    print(f"M = 1: SysM vs validated_int.Sys: rhs {e1:.1e}, lin {e2:.1e}  {'ok' if good else 'MISMATCH'}")
    # M = 3, 5 against inner_np.Inner.rhs (double precision)
    sys.path.insert(0, os.path.join(HERE, "..", "inner_stokes"))
    import inner_np as IN
    for M in (3, 5):
        s = SysM(M)
        w = 3.8
        Fh = {mu: complex(*rng.normal(size=2)) for mu in s.ODD}
        m = IN.Inner(M, w, Fh, True)
        Yc = rng.normal(size=s.nd) + 1j * rng.normal(size=s.nd)
        xi_c = 0.15 + 0.07j
        ref = m.rhs(xi_c, Yc)
        F, cache = s.rhs([acb(complex(y)) for y in Yc] + [acb(w)], acb(xi_c), 1 / acb(xi_c), acb(0))
        e = max(abs(complex(F[i].mid()) - ref[i]) / (1 + abs(ref[i])) for i in range(s.nd))
        # lin against a central difference of rhs (in acb at 256 bits: truncation error only)
        V = rng.normal(size=s.nd) + 1j * rng.normal(size=s.nd)
        dw = 0.3
        Lv = s.lin(cache, [acb(complex(v)) for v in V] + [acb(dw)], acb(xi_c), 1 / acb(xi_c), acb(0))
        hh = arb("1e-30")
        Yp = [acb(complex(Yc[i])) + hh * acb(complex(V[i])) for i in range(s.nd)] + [acb(w) + hh * dw]
        Ym = [acb(complex(Yc[i])) - hh * acb(complex(V[i])) for i in range(s.nd)] + [acb(w) - hh * dw]
        Fp, _ = s.rhs(Yp, acb(xi_c), 1 / acb(xi_c), acb(0))
        Fm, _ = s.rhs(Ym, acb(xi_c), 1 / acb(xi_c), acb(0))
        el = max(float(abs((Fp[i] - Fm[i]) / (2 * hh) - Lv[i]).upper()) / (1 + float(abs(Lv[i]).upper()))
                 for i in range(s.nd))
        good = e < 1e-13 and el < 1e-40
        ok &= good
        print(f"M = {M}: SysM vs inner_np.Inner.rhs (double): {e:.1e}; lin vs central difference: {el:.1e}  "
              f"{'ok' if good else 'MISMATCH'}")
    # series recursion against series_arb.SeriesArb (M = 3), and the jets against central differences (M = 3)
    from series_arb import SeriesArb
    M = 3
    p = [arb("2.7264758856"), arb("1.0156030948"), arb("0.1444194785"), arb("0.0110874259")]
    w = W0() * p[1]
    Fh = profile([acb(x) for x in p], M, acb(1))
    sa = SeriesArb({k: v for k, v in Fh.items()}, w, M)
    d1, s1_, g1 = sa.series(30)
    d2, s2_, g2 = series_generic_M(Fh, acb(w), 30, M)
    es = max(float(abs(d1[j][mu] - d2[j][mu]).upper()) / (1 + float(abs(d1[j][mu]).upper()))
             for j in range(30) for mu in sa.odd)
    overlap = all((d1[j][mu] - d2[j][mu]).contains(0) for j in range(30) for mu in sa.odd)
    es = es if overlap else 1.0         # disjoint balls: a genuine mismatch; otherwise es is a sum of ball radii
    phi0 = 3 * arb.pi() / 8
    xi0, vals, ders = start_series(p, M, "0.09", phi0, 40)
    hh = arb("1e-25")
    ej = 0.0
    for k in range(M + 1):
        pp = [x + (hh if i == k else 0) for i, x in enumerate(p)]
        pm = [x - (hh if i == k else 0) for i, x in enumerate(p)]
        _, vp, _ = start_series(pp, M, "0.09", phi0, 40, dirs=[])
        _, vm, _ = start_series(pm, M, "0.09", phi0, 40, dirs=[])
        ej = max(ej, max(float(abs((vp[i] - vm[i]) / (2 * hh) - ders[k][i]).upper()) for i in range(len(vals))))
    good = es < 1e-40 and ej < 1e-30
    ok &= good
    print(f"M = 3: series_generic_M vs series_arb.SeriesArb: {es:.1e}; parameter jets vs central differences: {ej:.1e}  "
          f"{'ok' if good else 'MISMATCH'}")
    # lin_unit (sparse) against lin on unit vectors, on series, M = 3 and 5
    from flint import acb_series
    for M in (3, 5):
        s = SysM(M)
        ctx.cap = 6
        Ys = [acb_series([acb(complex(*rng.normal(size=2))) for _ in range(6)], prec=6) for _ in range(s.nd)]
        Ys.append(acb_series([acb(3.8)], prec=6))
        xs = acb_series([acb(0.15, 0.07), acb(0.01, 0.02)], prec=6)
        zs = acb_series([0], prec=6)
        _, cache = s.rhs(Ys, xs, 1 / xs, zs)
        eu = 0.0
        for j in range(s.n):
            e = [acb_series([1 if i == j else 0], prec=6) for i in range(s.n)]
            a = s.lin(cache, e, xs, 1 / xs, zs)
            b = s.lin_unit(cache, j, xs, 1 / xs, zs)
            for i in range(s.nd):
                d = a[i] - b[i]
                eu = max([eu] + [float(abs(c).upper()) for c in d.coeffs()])
        good = eu < 1e-60
        ok &= good
        print(f"M = {M}: lin_unit vs lin on all unit vectors (series): {eu:.1e}  {'ok' if good else 'MISMATCH'}")
    # one Lohner step of certify_M_lohner.LohnerM against validated_int.Lohner at M = 1 (same set, same step)
    ok &= step_check()
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return ok


def step_check():
    """one step, M = 1, from the same Lohner set: LohnerM (growing Picard, sparse Jacobian columns, NJ = N) against
    validated_int.Lohner with lemma6_common installed; the enclosures must overlap and agree to rounding"""
    import validated_int as VI
    import lemma6_common
    lemma6_common.install()
    import certify_M_lohner as CL
    from flint import acb_mat
    ctx.prec = 256
    s1, sv = SysM(1), VI.Sys(gravity=True)
    p = [arb("2.5421480475"), arb("0.7769940136")]
    phi0 = 3 * arb.pi() / 8
    xi0, vals, ders, _ = placeholder_start(p, 1, "0.1", phi0)
    w = W0() * p[1]
    c = [acb(v.mid()) for v in vals] + [acb(w.mid())]
    q = [vals[i] - c[i] for i in range(s1.nd)] + [acb(w) - c[s1.nd]]
    n1 = s1.n
    Q1 = acb_mat([[1 if i == j else 0 for j in range(n1)] for i in range(n1)])
    a = CL.LohnerM(s1, c, Q1, q, 24)
    Qv = acb_mat([[1 if i == j else 0 for j in range(n1 + 1)] for i in range(n1 + 1)])
    b = VI.Lohner(sv, c + [acb(p[0])], Qv, q + [acb(0)], 24)
    lega = LegM("ray", "0.1", "0.2", phi0)
    legb = VI.Leg("ray", "0.1", "0.2")
    oka, okb = a.step(lega, 0.0, 2.0 ** -12), b.step(legb, 0.0, 2.0 ** -12)
    Xa, Xb = a.box(), b.box()
    ov = all(Xa[i].overlaps(Xb[i]) for i in range(s1.nd))
    dm = max(float(abs(Xa[i].mid() - Xb[i].mid())) for i in range(s1.nd))
    ra = max(float(Xa[i].rad()) for i in range(s1.nd)) if False else max(max(float(x.real.rad()), float(x.imag.rad())) for x in Xa[:s1.nd])
    rb = max(max(float(x.real.rad()), float(x.imag.rad())) for x in Xb[:s1.nd])
    good = oka and okb and ov and dm < 1e-40
    print(f"M = 1: one step of LohnerM vs validated_int.Lohner: both ok {oka and okb}, overlap {ov}, midpoints differ "
          f"{dm:.1e}, radii {ra:.1e} vs {rb:.1e}  {'ok' if good else 'MISMATCH'}")
    # the same step with the Jacobian at order 12 (N = 24): a different but valid enclosure, overlapping
    c2 = CL.LohnerM(s1, c, Q1, q, 24)
    c2.NJ = 12
    okc = c2.step(lega, 0.0, 2.0 ** -12)
    Xc = c2.box()
    ov2 = all(Xc[i].overlaps(Xb[i]) for i in range(s1.nd))
    rc = max(max(float(x.real.rad()), float(x.imag.rad())) for x in Xc[:s1.nd])
    good2 = okc and ov2 and rc < 10 * rb + 1e-30
    print(f"M = 1: the same step with NJ = 12: ok {okc}, overlaps validated_int's {ov2}, radius {rc:.1e}  "
          f"{'ok' if good2 else 'MISMATCH'}")
    return good and good2


def timing(M, N=24, prec=256, NJ=None):
    """time one validated_int.Lohner step with SysM (diagnostic)"""
    import validated_int as VI
    import lemma6_common
    lemma6_common.install()
    from flint import acb_mat
    ctx.prec = prec
    s = SysM(M)
    p = {1: ["2.5421480475", "0.7769940136"], 3: ["2.7264758856", "1.0156030948", "0.1444194785", "0.0110874259"],
         5: ["2.718579189", "0.999228404", "0.149203575", "0.014719052", "0.008405347", "0.001470345"]}[M]
    pa = [arb(x) for x in p]
    phi0 = 3 * arb.pi() / 8
    r0 = "0.08"
    xi0, vals, ders, info = placeholder_start(pa, M, r0, phi0, verbose=True)
    c = [acb(v.mid()) for v in vals] + [acb((W0() * pa[1]).mid())]
    Q = acb_mat([[1 if i == j else 0 for j in range(s.n)] for i in range(s.n)])
    q = [vals[i] - c[i] for i in range(s.nd)] + [acb(0)]
    import certify_M_lohner as CL
    lo = CL.LohnerM(s, c, Q, q, N)
    lo.NJ = NJ
    leg = LegM("ray", r0, "0.2", phi0)
    for k in range(10, 20):
        t = time.time()
        okk = lo.step(leg, 0.0, 2.0 ** -k)
        t1 = time.time() - t
        print(f"M = {M}: n = {s.n}, one Lohner step h = 2^-{k} (order {N}, {prec} bits): {t1:.2f} s, ok {okk}", flush=True)
        if okk:
            break


if __name__ == "__main__":
    args = sys.argv[1:]
    for a in args:
        if a.startswith("--mutate="):
            MUTATE.add(a.split("=", 1)[1])
    if args and args[0] == "selftest":
        sys.exit(0 if selftest() else 1)
    if args and args[0] == "timing":
        timing(int(args[1]), int(args[2]) if len(args) > 2 else 24, 256, int(args[3]) if len(args) > 3 else None)
