"""The truncated M-harmonic inner system in the forms the lemma-6 scripts integrate (proof/A_lemmas_M.md), for any odd M.
Ball arithmetic (python-flint); the classes plug into validated_int.Lohner unchanged, as validated_int.Sys (M = 1) and
lemma6_usys.SysU (M = 1) do.

  * SysXiM: the xi-form of lateral_np.py / inner_np.py, state [D_mu (odd mu ascending), S_mu, C_nu (even nu ascending),
    w, A], d/dxi:
        xi^2 D'  + i mu w xi D = [C S]_mu
        xi^2 S'  + i mu w xi S = [C xi D']_mu + [(C E) D]_mu - [C D]_mu,    E = [D D]/2
        xi C' = [C E]_nu
  * SysUM: the variables of fixed_point.py with u = 1/xi^2 as the independent variable (lemma6_usys.py for M = 1),
    state [P_mu, V_mu, kappa (C_nu - delta_nu0), w, A], D = xi P, S = z Q, V = Q - (1 + s_mu) P, s_mu = i mu w, z = 1/u:
        P_u = -z (V + [(C-1)Q]) / 2
        V_u = -V/2 - ([(C-1)Q] + [(C-1)W])/2 - z (-(2+s) Q - (1+s) W + [C E^ P]) / 2,   W = V + [(C-1)Q],  E^ = [P P]/2
        C_u = -z^2 [C E^] / 2
    The fast mode is e^{-u/2} for every label (the eigenvalue repeats across harmonics); the labels enter through s_mu.
  * rhs_gen_M: SysUM's right-hand side for any ring type (acb, acb_series, acb_poly), for the disc lemma.

    python3 lemma6_usys_M.py [--M=3]     self-tests (exit status 0 only if all pass):
        (a) M = 1: SysXiM = validated_int.Sys and SysUM = lemma6_usys.SysU (rhs and lin, random states)
        (b) SysXiM against inner_np.Inner.rhs (solver/inner_stokes, numpy convolutions, the code hier.py integrates)
        (c) SysUM against SysXiM through the exact change of variables and the chain rule (power series in xi)
        (d) fixed_point.residual_numerators (the system the start-value lemma encodes) against SysUM
        (e) lin against the series derivative of rhs, for both classes and the w direction
        (f) rhs_gen_M against SysUM.rhs
    python3 lemma6_usys_M.py --mutate=label   mutation: s_mu = i |mu| w in SysUM (the label sign lost); (c), (d) must fail
"""
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flint import acb, arb, acb_series, acb_poly, ctx  # noqa: E402
import validated_int as VI  # noqa: E402

I = acb(0, 1)
KAPPA = 100
HALF = acb(0.5)
PHI_U = 3 * arb.pi() / 4


def harmonics(M):
    odd = [mu for mu in range(-M, M + 1) if mu % 2]
    even = [nu for nu in range(-(M + 1), M + 2) if nu % 2 == 0]
    return odd, even


def pr(X, Y, out, zero):
    r = {}
    for k in out:
        acc = zero
        for a, xa in X.items():
            yb = Y.get(k - a)
            if yb is not None:
                acc = acc + xa * yb
        r[k] = acc
    return r


class SysXiM:
    """the xi-form (validated_int.Sys for M harmonics); state [D, S, C, w, A]"""

    def __init__(self, M):
        self.M = M
        self.odd, self.even = harmonics(M)
        self.no, self.ne = len(self.odd), len(self.even)
        self.nd = 2 * self.no + self.ne
        self.n = self.nd + 2

    def unpack(self, Y):
        no = self.no
        D = {m: Y[i] for i, m in enumerate(self.odd)}
        S = {m: Y[no + i] for i, m in enumerate(self.odd)}
        C = {n: Y[2 * no + i] for i, n in enumerate(self.even)}
        return D, S, C, Y[self.nd]

    def pack(self, D, S, C, w):
        return [D[m] for m in self.odd] + [S[m] for m in self.odd] + [C[n] for n in self.even] + [w, w]

    def rhs(self, Y, xi, ixi, zero):
        D, S, C, w = self.unpack(Y)
        odd, even = self.odd, self.even
        ixi2 = ixi * ixi
        CS = pr(C, S, odd, zero)
        dD = {m: (CS[m] - I * m * w * xi * D[m]) * ixi2 for m in odd}
        E = {k: v / 2 for k, v in pr(D, D, even, zero).items()}
        cE = pr(C, E, even, zero)
        xdD = {m: xi * dD[m] for m in odd}
        a, b, c = pr(C, xdD, odd, zero), pr(cE, D, odd, zero), pr(C, D, odd, zero)
        r2 = {m: a[m] + b[m] - c[m] for m in odd}
        dC = {n: cE[n] * ixi for n in even}
        dS = {m: (r2[m] - I * m * w * xi * S[m]) * ixi2 for m in odd}
        return self.pack(dD, dS, dC, zero), (D, S, C, w, E, cE, dD)

    def lin(self, cache, V, xi, ixi, zero):
        D, S, C, w, E, cE, dD = cache
        odd, even = self.odd, self.even
        dD_, dS_, dC_, dw = self.unpack(V)
        ixi2 = ixi * ixi
        p1, p2 = pr(dC_, S, odd, zero), pr(C, dS_, odd, zero)
        dCS = {m: p1[m] + p2[m] for m in odd}
        ddD = {m: (dCS[m] - I * m * (dw * xi * D[m] + w * xi * dD_[m])) * ixi2 for m in odd}
        dE = pr(D, dD_, even, zero)
        p1, p2 = pr(dC_, E, even, zero), pr(C, dE, even, zero)
        dcE = {n: p1[n] + p2[n] for n in even}
        xdD = {m: xi * dD[m] for m in odd}
        xddD = {m: xi * ddD[m] for m in odd}
        t = [pr(dC_, xdD, odd, zero), pr(C, xddD, odd, zero), pr(dcE, D, odd, zero), pr(cE, dD_, odd, zero),
             pr(dC_, D, odd, zero), pr(C, dD_, odd, zero)]
        dr2 = {m: t[0][m] + t[1][m] + t[2][m] + t[3][m] - t[4][m] - t[5][m] for m in odd}
        ddC = {n: dcE[n] * ixi for n in even}
        ddS = {m: (dr2[m] - I * m * (dw * xi * S[m] + w * xi * dS_[m])) * ixi2 for m in odd}
        return self.pack(ddD, ddS, ddC, zero)


class SysUM:
    """the u-form (lemma6_usys.SysU for M harmonics); state [P, V, kappa (C - delta), w, A]"""

    def __init__(self, M, kappa=KAPPA, mutate=None):
        self.M = M
        self.odd, self.even = harmonics(M)
        self.no, self.ne = len(self.odd), len(self.even)
        self.nd = 2 * self.no + self.ne
        self.n = self.nd + 2
        self.kap = acb(kappa)
        self.mut = mutate        # MUTATION tests only: 'sign' (V_u = +V/2 + ...), 'label' (s_mu = i |mu| w)

    def sfac(self, m):
        return abs(m) if self.mut == "label" else m

    def unpack(self, Y):
        no = self.no
        P = {m: Y[i] for i, m in enumerate(self.odd)}
        V = {m: Y[no + i] for i, m in enumerate(self.odd)}
        C = {n: Y[2 * no + i] for i, n in enumerate(self.even)}
        return P, V, C, Y[self.nd]

    def pack(self, P, V, C, w):
        return [P[m] for m in self.odd] + [V[m] for m in self.odd] + [C[n] for n in self.even] + [w, w]

    def rhs(self, Y, u, z, zero):
        odd, even = self.odd, self.even
        P, V, Ct, w = self.unpack(Y)
        s = {m: I * self.sfac(m) * w for m in odd}
        Cm1 = {n: Ct[n] / self.kap for n in even}
        C = {n: (Cm1[n] + 1 if n == 0 else Cm1[n]) for n in even}
        Q = {m: V[m] + (1 + s[m]) * P[m] for m in odd}
        CQ = pr(Cm1, Q, odd, zero)
        W = {m: V[m] + CQ[m] for m in odd}
        CW = pr(Cm1, W, odd, zero)
        Eh = {k: v / 2 for k, v in pr(P, P, even, zero).items()}
        CE = pr(C, Eh, even, zero)
        CEP = pr(CE, P, odd, zero)
        dP = {m: -z * (V[m] + CQ[m]) / 2 for m in odd}
        dV = {m: -V[m] / 2 - (CQ[m] + CW[m]) / 2 - z * (-(2 + s[m]) * Q[m] - (1 + s[m]) * W[m] + CEP[m]) / 2
              for m in odd}
        dC = {n: -z * z * CE[n] / 2 * self.kap for n in even}
        if self.mut == "sign":
            dV = {m: dV[m] + V[m] for m in odd}
        return self.pack(dP, dV, dC, zero), (P, V, C, w, s, Cm1, Q, CQ, W, Eh, CE)

    def lin(self, cache, dY, u, z, zero):
        odd, even = self.odd, self.even
        P, V, C, w, s, Cm1, Q, CQ, W, Eh, CE = cache
        dP_, dV_, dCt, dw = self.unpack(dY)
        dC_ = {n: dCt[n] / self.kap for n in even}
        ds = {m: I * self.sfac(m) * dw for m in odd}
        dQ = {m: dV_[m] + (1 + s[m]) * dP_[m] + ds[m] * P[m] for m in odd}
        a, b = pr(dC_, Q, odd, zero), pr(Cm1, dQ, odd, zero)
        dCQ = {m: a[m] + b[m] for m in odd}
        dW = {m: dV_[m] + dCQ[m] for m in odd}
        a, b = pr(dC_, W, odd, zero), pr(Cm1, dW, odd, zero)
        dCW = {m: a[m] + b[m] for m in odd}
        dEh = pr(P, dP_, even, zero)
        a, b = pr(dC_, Eh, even, zero), pr(C, dEh, even, zero)
        dCE = {n: a[n] + b[n] for n in even}
        CEP_a, CEP_b = pr(dCE, P, odd, zero), pr(CE, dP_, odd, zero)
        ddP = {m: -z * (dV_[m] + dCQ[m]) / 2 for m in odd}
        ddV = {m: -dV_[m] / 2 - (dCQ[m] + dCW[m]) / 2
               - z * (-(2 + s[m]) * dQ[m] - ds[m] * Q[m] - (1 + s[m]) * dW[m] - ds[m] * W[m] + CEP_a[m] + CEP_b[m]) / 2
               for m in odd}
        ddC = {n: -z * z * dCE[n] / 2 * self.kap for n in even}
        if self.mut == "sign":
            ddV = {m: ddV[m] + dV_[m] for m in odd}
        return self.pack(ddP, ddV, ddC, zero)


def rhs_gen_M(M, Y, z, w, zero, mutate=None):
    """SysUM.rhs (the nd dynamic components) for any ring type with right multiplication by acb (acb, acb_series,
    acb_poly). Y: nd ring elements; z = 1/u (ring element); w: acb."""
    odd, even = harmonics(M)
    no = len(odd)
    kap = acb(KAPPA)
    ikap = acb(1) / kap
    P = {m: Y[i] for i, m in enumerate(odd)}
    V = {m: Y[no + i] for i, m in enumerate(odd)}
    Cm1 = {n: Y[2 * no + i] * ikap for i, n in enumerate(even)}
    C = {n: (Cm1[n] + 1 if n == 0 else Cm1[n]) for n in even}
    s = {m: I * m * w for m in odd}
    Q = {m: V[m] + P[m] * (1 + s[m]) for m in odd}
    CQ = pr(Cm1, Q, odd, zero)
    W = {m: V[m] + CQ[m] for m in odd}
    CW = pr(Cm1, W, odd, zero)
    Eh = {k: v * HALF for k, v in pr(P, P, even, zero).items()}
    CE = pr(C, Eh, even, zero)
    CEP = pr(CE, P, odd, zero)
    dP = [(z * (V[m] + CQ[m])) * (-HALF) for m in odd]
    dV = [V[m] * (-HALF) - (CQ[m] + CW[m]) * HALF - (z * (Q[m] * (-(2 + s[m])) - W[m] * (1 + s[m]) + CEP[m])) * HALF
          for m in odd]
    if mutate == "sign":
        dV = [dV[i] + V[m] for i, m in enumerate(odd)]
    dC = [(z * z * CE[n]) * (-HALF * kap) for n in even]
    return dP + dV + dC


def dsc_to_pvc(M, Y, xi, w):
    """[D, S, C] at xi -> [P, V, kappa (C - delta)] (exact; xi, w may be balls or series)"""
    odd, even = harmonics(M)
    no = len(odd)
    z = xi * xi
    out = [Y[i] / xi for i in range(no)]
    for i, m in enumerate(odd):
        out.append(Y[no + i] / z - (1 + I * m * w) * Y[i] / xi)
    return out + [KAPPA * (Y[2 * no + j] - (1 if n == 0 else 0)) for j, n in enumerate(even)]


def pvc_to_dsc(M, Y, xi, w):
    odd, even = harmonics(M)
    no = len(odd)
    z = xi * xi
    out = [xi * Y[i] for i in range(no)]
    for i, m in enumerate(odd):
        out.append(z * (Y[no + i] + (1 + I * m * w) * Y[i]))
    return out + [Y[2 * no + j] / KAPPA + (1 if n == 0 else 0) for j, n in enumerate(even)]


class UArc:
    """u = rho e^{-i (3pi/4)(1 - t)} (as lemma6_usys.UArc), or from the angle phi_a to phi_b in general"""
    kind = "uarc"

    def __init__(self, rho, phi_a=None, phi_b=None):
        self.rho = arb(rho)
        self.pa = -PHI_U if phi_a is None else arb(phi_a)
        self.pb = arb(0) if phi_b is None else arb(phi_b)
        self.dp = self.pb - self.pa

    def xi(self, t):
        return acb(self.rho) * (I * (self.pa + self.dp * t)).exp()

    def dxi(self, u):
        return I * self.dp * u

    def series(self, t0, L):
        x0 = self.xi(t0)
        e = acb_series([0, I * self.dp], prec=L).exp()
        xs = x0 * e
        return xs, I * self.dp * xs


class URay:
    """u = (ra + (rb - ra) t) e^{i phi}, t: 0 -> 1 (default phi = -3pi/4, the start ray arg xi = 3pi/8)"""
    kind = "uray"

    def __init__(self, ra, rb, phi=None):
        self.ra, self.rb = arb(ra), arb(rb)
        self.e = (I * (-PHI_U if phi is None else arb(phi))).exp()

    def xi(self, t):
        return (acb(self.ra) + acb(self.rb - self.ra) * t) * self.e

    def dxi(self, u):
        return acb(self.rb - self.ra) * self.e

    def series(self, t0, L):
        x0 = self.xi(t0)
        v = acb(self.rb - self.ra) * self.e
        return acb_series([x0, v], prec=L), acb_series([v], prec=L)


# ---------------------------------------------------------------- self-tests


def _rnd(scale=1.0):
    return acb(random.uniform(-scale, scale), random.uniform(-scale, scale))


def selftest(M=3, trials=3, seed=7, mutate=None):
    ctx.prec = 200
    random.seed(seed)
    res = {}
    wv = VI.W0 * arb("0.9992")
    w = acb(wv)
    # (a) M = 1 against the existing classes (exact formulas, same arithmetic)
    import lemma6_usys as U1
    SX1, SU1, X1, Uo = SysXiM(1), SysUM(1), VI.Sys(gravity=True), U1.SysU()
    worst = 0.0
    for _ in range(trials):
        Y = [_rnd() for _ in range(7)] + [w, acb(2.5)]
        dY = [_rnd() for _ in range(7)] + [acb(0.3), acb(0)]
        xi = acb(random.uniform(0.05, 0.3), random.uniform(0, 0.2))
        for A_, B_ in ((SX1, X1), (SU1, Uo)):
            x0 = xi if A_ is SX1 else 1 / (xi * xi)
            F1, c1 = A_.rhs(Y, x0, 1 / x0, acb(0))
            F2, c2 = B_.rhs(Y, x0, 1 / x0, acb(0))
            L1, L2 = A_.lin(c1, dY, x0, 1 / x0, acb(0)), B_.lin(c2, dY, x0, 1 / x0, acb(0))
            worst = max(worst, max(float(abs(F1[i] - F2[i]).upper()) + float(abs(L1[i] - L2[i]).upper())
                                   for i in range(7)))
    res["a"] = worst
    # (b) SysXiM against inner_np.Inner.rhs (numpy, double precision)
    sys.path.insert(0, os.path.join(HERE, "..", "inner_stokes"))
    import inner_np
    SX = SysXiM(M)
    odd, even = SX.odd, SX.even
    worst = 0.0
    for _ in range(trials):
        Fh = {}
        for nu in range(1, M + 1, 2):
            Fh[nu] = complex(random.uniform(-1, 1), random.uniform(-1, 1)); Fh[-nu] = np.conj(Fh[nu])
        m = inner_np.Inner(M, float(wv.mid()), Fh, True)
        Ynp = np.array([complex(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(SX.nd)])
        xi = complex(random.uniform(0.05, 0.3), random.uniform(0, 0.2))
        Fn = m.rhs(xi, Ynp)
        Y = [acb(v.real, v.imag) for v in Ynp] + [w, acb(2.5)]
        Fb, _ = SX.rhs(Y, acb(xi.real, xi.imag), 1 / acb(xi.real, xi.imag), acb(0))
        worst = max(worst, max(abs(complex(Fb[i].mid()) - Fn[i]) / (1 + abs(Fn[i])) for i in range(SX.nd)))
    res["b"] = worst
    # (c) SysUM against SysXiM by the change of variables (chain rule as power series in xi)
    SU = SysUM(M, mutate=mutate)
    worst = 0.0
    for _ in range(trials):
        xi0 = acb(random.uniform(0.05, 0.3), random.uniform(0.0, 0.2))
        Y0 = [_rnd() for _ in range(SX.nd)] + [w, acb(2.5)]
        F, _ = SX.rhs(Y0, xi0, 1 / xi0, acb(0))
        ctx.cap = 2
        xs = acb_series([xi0, 1], prec=2)
        pv = dsc_to_pvc(M, [acb_series([Y0[i], F[i]], prec=2) for i in range(SX.nd)], xs, w)
        lhs = [(p.coeffs()[1] if len(p.coeffs()) > 1 else acb(0)) * (-xi0 ** 3 / 2) for p in pv]
        Yu = [p.coeffs()[0] for p in pv] + [w, acb(2.5)]
        u0 = 1 / (xi0 * xi0)
        Fu, _ = SU.rhs(Yu, u0, 1 / u0, acb(0))
        worst = max(worst, max(float(abs(lhs[i] - Fu[i]).upper()) / (1 + float(abs(Fu[i]).upper()))
                               for i in range(SX.nd)))
    res["c"] = worst
    # (d) fixed_point.residual_numerators: for constant polynomials P, Q, C the residual numerators are the right-hand
    #     side times (2z, 2z^2, 1): rho_p = Np/(2z) = P', rho_v = Nv/(2z^2) = V', rho_c = Nc = C'
    import fixed_point as FP
    from series_arb import SeriesArb
    worst = 0.0
    for _ in range(trials):
        Y = [_rnd() for _ in range(SX.nd)]
        sa = SeriesArb({mu: acb(0) for mu in odd}, w, M=M, prec=200)
        no = len(odd)
        Pp = {m: acb_poly([Y[i]]) for i, m in enumerate(odd)}
        Qp = {m: acb_poly([Y[no + i] + (1 + I * m * w) * Y[i]]) for i, m in enumerate(odd)}
        Cp = {n: acb_poly([Y[2 * no + j] / KAPPA + (1 if n == 0 else 0)]) for j, n in enumerate(even)}
        Np, Nv, Nc, _ = FP.residual_numerators(sa, Pp, Qp, Cp)
        u0 = acb(random.uniform(30, 200), random.uniform(-100, 0))
        z0 = 1 / u0
        Fu, _ = SU.rhs(Y + [w, acb(2.5)], u0, z0, acb(0))
        # d/du = -z^2 d/dz
        ref = [Np[m](z0) / (2 * z0) for m in odd] + [Nv[m](z0) / (2 * z0 * z0) for m in odd] + \
              [Nc[n](z0) * KAPPA for n in even]
        ref = [-z0 * z0 * r for r in ref]
        worst = max(worst, max(float(abs(ref[i] - Fu[i]).upper()) / (1 + float(abs(Fu[i]).upper()))
                               for i in range(SX.nd)))
    res["d"] = worst
    # (e) lin against the series derivative of rhs (both classes, w direction included)
    worst = 0.0
    for sys_ in (SX, SU):
        for _ in range(trials):
            Y = [_rnd() for _ in range(sys_.nd)] + [w, acb(2.5)]
            dY = [_rnd() for _ in range(sys_.nd)] + [acb(0.2), acb(0)]
            x0 = acb(random.uniform(0.05, 0.3), random.uniform(0, 0.2))
            if sys_ is SU:
                x0 = 1 / (x0 * x0)
            Fx, cx = sys_.rhs(Y, x0, 1 / x0, acb(0))
            L = sys_.lin(cx, dY, x0, 1 / x0, acb(0))
            ctx.cap = 2
            Ye = [acb_series([Y[i], dY[i]], prec=2) for i in range(len(Y))]
            Fe, _ = sys_.rhs(Ye, acb_series([x0], prec=2), acb_series([1 / x0], prec=2), acb_series([0], prec=2))
            d = [Fe[i].coeffs()[1] if len(Fe[i].coeffs()) > 1 else acb(0) for i in range(sys_.nd)]
            worst = max(worst, max(float(abs(d[i] - L[i]).upper()) for i in range(sys_.nd)))
    res["e"] = worst
    # (f) rhs_gen_M against SysUM.rhs
    worst = 0.0
    for _ in range(trials):
        Y = [_rnd(2) for _ in range(SU.nd)]
        u = acb(random.uniform(-30, 30), random.uniform(-30, 30))
        F1, _ = SU.rhs(Y + [w, acb(2.5)], u, 1 / u, acb(0))
        F2 = rhs_gen_M(M, Y, 1 / u, w, acb(0))
        worst = max(worst, max(float(abs(F1[i] - F2[i]).upper()) for i in range(SU.nd)))
    res["f"] = worst
    tol = {"a": 1e-50, "b": 1e-13, "c": 1e-50, "d": 1e-50, "e": 1e-50, "f": 1e-50}
    ok = {k: res[k] < tol[k] for k in res}
    print(f"M = {M}" + (f" [MUTATION {mutate}]" if mutate else "") + ": " + "; ".join(
        f"({k}) {res[k]:.1e} {'ok' if ok[k] else 'FAIL'}" for k in sorted(res)))
    return res, ok


if __name__ == "__main__":
    opts = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
    mut = opts.get("mutate")
    Ms = [int(opts["M"])] if "M" in opts else [3, 5]
    allok = True
    for M in Ms:
        res, ok = selftest(M, mutate=mut)
        allok = allok and all(ok.values())
    if mut:
        # the mutation is in SysUM only: (c) (against the xi-form) and (d) (against fixed_point's encoding) must fire
        fired = not all(ok[k] for k in ("c", "d"))
        print(f"MUTATION {mut}: caught: {fired}")
        sys.exit(0 if fired else 3)
    print(f"ALL SELF-TESTS PASS: {allok}")
    sys.exit(0 if allok else 3)
