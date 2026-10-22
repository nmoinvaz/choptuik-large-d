"""The truncated M = 1 system in the variables of fixed_point.py with u = 1/z = 1/xi^2 as the independent variable
(proof/A_certify.md section 11, used by the lemma 6b sweep).

State [P_-1, P_+1, V_-1, V_+1, k C_-2, k (C_0 - 1), k C_+2, w, A] (k = KAPPA = 100, a fixed linear rescaling) with
D = xi P, S = z Q, V = Q - (1 + s) P, s = i mu w. SysUR adds the arc radius and SysUD a path offset as a tenth state.
`python3 lemma6_usys.py` checks rhs and lin of all three against validated_int.Sys at random states. With
' = d/dz the system is (fixed_point.py, checked exactly in review R1)
    P' = (V + [(C-1)Q]) / (2z),
    V' = V/(2z^2) + ([(C-1)Q] + [(C-1)W])/(2z^2) + (-(2+s) Q - (1+s) W + [C E P])/(2z),   W = V + [(C-1)Q],
    C' = [C E]/2,   E = [P^2/2],
and d/du = -z^2 d/dz turns it into
    P_u = -z (V + [(C-1)Q]) / 2,
    V_u = -V/2 - ([(C-1)Q] + [(C-1)W])/2 - z (-(2+s) Q - (1+s) W + [C E P]) / 2,
    C_u = -z^2 [C E] / 2.
The fast mode is V ~ e^{-u/2} with a constant rate: the stiffness of the xi-form (rates 1/xi^3) is gone, and a ball in
the path only enters through z = 1/u in the O(z) terms. The classes below plug into validated_int.Lohner unchanged:
the integrator calls rhs(Y, x, 1/x, zero) and lin(...) with x = u (so the second argument it passes is z).
"""
from flint import acb, arb, acb_series

import validated_int as VI

ODD, EVEN = (-1, 1), (-2, 0, 2)
I = acb(0, 1)
RSCALE = 1000                     # the radius state of SysUR is stored as rho / RSCALE: with an orthogonal Lohner basis,
                                  # a radius interval of width ~1 next to state widths ~1e-3 wraps the set
KAPPA = 100                       # the C components are stored as KAPPA (C - delta_{nu 0}) (a fixed linear change of
                                  # variables: it makes the a-priori Picard step of validated_int work at |du| ~ 1)
PHI_U = 3 * arb.pi() / 4          # arg u runs from -3pi/4 (the start ray arg xi = 3pi/8) to 0 (the real axis)


def pr(X, Y, out, zero):
    return VI.pr(X, Y, out, zero)


class SysU:
    def __init__(self, kappa=KAPPA, mutate=False):
        self.nd, self.n = 7, 9
        self.kap = acb(kappa)
        self.mut = mutate          # MUTATION test only: V_u = +V/2 + ... (wrong sign of the fast term)

    def unpack(self, Y):
        P = {m: Y[i] for i, m in enumerate(ODD)}
        V = {m: Y[2 + i] for i, m in enumerate(ODD)}
        C = {n: Y[4 + i] for i, n in enumerate(EVEN)}
        return P, V, C, Y[7]

    def pack(self, P, V, C, w):
        return [P[m] for m in ODD] + [V[m] for m in ODD] + [C[n] for n in EVEN] + [w, w]

    def rhs(self, Y, u, z, zero):
        P, V, Ct, w = self.unpack(Y)
        s = {m: I * m * w for m in ODD}
        Cm1 = {n: Ct[n] / self.kap for n in EVEN}            # the state holds kappa (C - delta_{nu 0})
        C = {n: (Cm1[n] + 1 if n == 0 else Cm1[n]) for n in EVEN}
        Q = {m: V[m] + (1 + s[m]) * P[m] for m in ODD}
        CQ = pr(Cm1, Q, ODD, zero)
        W = {m: V[m] + CQ[m] for m in ODD}
        CW = pr(Cm1, W, ODD, zero)
        Eh = {k: v / 2 for k, v in pr(P, P, EVEN, zero).items()}
        CE = pr(C, Eh, EVEN, zero)
        CEP = pr(CE, P, ODD, zero)
        dP = {m: -z * (V[m] + CQ[m]) / 2 for m in ODD}
        dV = {m: -V[m] / 2 - (CQ[m] + CW[m]) / 2 - z * (-(2 + s[m]) * Q[m] - (1 + s[m]) * W[m] + CEP[m]) / 2
              for m in ODD}
        dC = {n: -z * z * CE[n] / 2 * self.kap for n in EVEN}
        if self.mut:
            dV = {m: dV[m] + V[m] for m in ODD}
        return self.pack(dP, dV, dC, zero), (P, V, C, w, s, Cm1, Q, CQ, W, Eh, CE)

    def lin(self, cache, dY, u, z, zero):
        P, V, C, w, s, Cm1, Q, CQ, W, Eh, CE = cache
        dP_, dV_, dCt, dw = self.unpack(dY)
        dC_ = {n: dCt[n] / self.kap for n in EVEN}
        ds = {m: I * m * dw for m in ODD}
        dQ = {m: dV_[m] + (1 + s[m]) * dP_[m] + ds[m] * P[m] for m in ODD}
        a, b = pr(dC_, Q, ODD, zero), pr(Cm1, dQ, ODD, zero)
        dCQ = {m: a[m] + b[m] for m in ODD}
        dW = {m: dV_[m] + dCQ[m] for m in ODD}
        a, b = pr(dC_, W, ODD, zero), pr(Cm1, dW, ODD, zero)
        dCW = {m: a[m] + b[m] for m in ODD}
        dEh = pr(P, dP_, EVEN, zero)
        a, b = pr(dC_, Eh, EVEN, zero), pr(C, dEh, EVEN, zero)
        dCE = {n: a[n] + b[n] for n in EVEN}
        CEP_a, CEP_b = pr(dCE, P, ODD, zero), pr(CE, dP_, ODD, zero)
        ddP = {m: -z * (dV_[m] + dCQ[m]) / 2 for m in ODD}
        ddV = {m: -dV_[m] / 2 - (dCQ[m] + dCW[m]) / 2
               - z * (-(2 + s[m]) * dQ[m] - ds[m] * Q[m] - (1 + s[m]) * dW[m] - ds[m] * W[m] + CEP_a[m] + CEP_b[m]) / 2
               for m in ODD}
        ddC = {n: -z * z * dCE[n] / 2 * self.kap for n in EVEN}
        if self.mut:
            ddV = {m: ddV[m] + dV_[m] for m in ODD}
        return self.pack(ddP, ddV, ddC, zero)


class SysUR(SysU):
    """SysU along the arcs u = rho E(t), E = e^{-i (3pi/4)(1 - t)}, with the radius rho as a tenth, constant state
    [P-1, P+1, V-1, V+1, kC-2, k(C0-1), kC+2, w, A, rho]. The integrator is given the unit arc E(t) (leg UArc(1)) and
    multiplies by dE/dt, so rhs returns rho F(rho E, Y), whose rho-derivative is F - z dF/dz (explicit z = 1/u).
    With rho a state, a radius interval is carried linearly by the Lohner set instead of as a ball in the path."""

    def __init__(self, kappa=KAPPA, mutate=False, rscale=RSCALE):
        super().__init__(kappa, mutate)
        self.n = 10
        self.rs = acb(rscale)      # the state holds rho / rscale (a linear change of variables, see RSCALE)

    def pack(self, P, V, C, w):
        return [P[m] for m in ODD] + [V[m] for m in ODD] + [C[n] for n in EVEN] + [w, w, w]

    def rhs(self, Y, E, iE, zero):
        rho = Y[9] * self.rs
        u, z = rho * E, iE / rho
        F, cache = SysU.rhs(self, Y, u, z, zero)
        P, V, C, w, s, Cm1, Q, CQ, W, Eh, CE = cache
        CEP = pr(CE, P, ODD, zero)
        Gz = [-(V[m] + CQ[m]) / 2 for m in ODD]
        Gz += [-(-(2 + s[m]) * Q[m] - (1 + s[m]) * W[m] + CEP[m]) / 2 for m in ODD]
        Gz += [-z * CE[n] * self.kap for n in EVEN]
        Frho = [(F[i] - z * Gz[i]) * self.rs for i in range(7)]   # d(rho F)/d(rho / rscale)
        return [rho * F[i] for i in range(7)] + [zero, zero, zero], (cache, rho, u, z, Frho)

    def lin(self, cache, dY, E, iE, zero):
        inner, rho, u, z, Frho = cache
        L = SysU.lin(self, inner, dY, u, z, zero)
        return [rho * L[i] + dY[9] * Frho[i] for i in range(7)] + [zero, zero, zero]


class SysUD(SysU):
    """SysU along a translated path u = u_c(t) + delta, with the complex offset delta as a tenth, constant state
    [P-1, P+1, V-1, V+1, kC-2, k(C0-1), kC+2, w, A, delta]. The integrator is given the reference path u_c(t) (a UArc or
    URay leg) and multiplies by du_c/dt = du/dt, so the velocity does not depend on delta; delta enters only through
    z = 1/u in the O(z) terms, and dF/d delta = -z^2 dF/dz (explicit z)."""

    def __init__(self, kappa=KAPPA, mutate=False):
        super().__init__(kappa, mutate)
        self.n = 10

    def pack(self, P, V, C, w):
        return [P[m] for m in ODD] + [V[m] for m in ODD] + [C[n] for n in EVEN] + [w, w, w]

    def rhs(self, Y, uc, iuc, zero):
        u = uc + Y[9]
        z = 1 / u
        F, cache = SysU.rhs(self, Y, u, z, zero)
        P, V, C, w, s, Cm1, Q, CQ, W, Eh, CE = cache
        CEP = pr(CE, P, ODD, zero)
        Gz = [-(V[m] + CQ[m]) / 2 for m in ODD]
        Gz += [-(-(2 + s[m]) * Q[m] - (1 + s[m]) * W[m] + CEP[m]) / 2 for m in ODD]
        GzC = [-CE[n] * self.kap for n in EVEN]                    # d Gz / dz (only the C rows depend on z)
        # dF/d delta = -z^2 Gz(y, z), z = 1/(uc + delta), in mean value form in delta at fixed y:
        #   Fd(delta) = Fd(0) + delta * Fd'(z~),  Fd' = z^2 (2 z Gz + z^2 dGz/dz),  z~ in the ball of z over the box.
        # Evaluated directly over the delta box, Fd is about 10 times wider than its true variation, and that excess
        # (first order in the box) accumulates along the tube.
        z0 = iuc
        d = Y[9]
        if isinstance(d, acb):                                      # the segment [0, delta] (its hull)
            dseg = d.union(acb(0))
        else:
            dseg = acb_series([d.coeffs()[0].union(acb(0)) if d.coeffs() else acb(0)], prec=d.prec)
        zs = 1 / (uc + dseg)
        Gz0 = Gz + [z0 * g for g in GzC]
        Gzs = Gz + [zs * g for g in GzC]
        Fd0 = [-z0 * z0 * Gz0[i] for i in range(7)]
        dFd = [zs * zs * (2 * zs * Gzs[i] + (zs * zs * GzC[i - 4] if i >= 4 else 0)) for i in range(7)]
        Fd = [Fd0[i] + d * dFd[i] for i in range(7)]               # dF/d delta (mean value form in delta)
        return [F[i] for i in range(7)] + [zero, zero, zero], (cache, u, z, Fd)

    def lin(self, cache, dY, uc, iuc, zero):
        inner, u, z, Fd = cache
        L = SysU.lin(self, inner, dY, u, z, zero)
        return [L[i] + dY[9] * Fd[i] for i in range(7)] + [zero, zero, zero]


class UArc:
    """u = rho e^{-i (3pi/4)(1 - t)}, t: 0 -> 1 (the arc |xi| = rho^-1/2 from arg xi = 3pi/8 to 0); rho may be a ball"""
    kind = "uarc"

    def __init__(self, rho):
        self.rho = arb(rho)

    def xi(self, t):
        return acb(self.rho) * (-I * PHI_U * (1 - t)).exp()

    def dxi(self, u):
        return I * PHI_U * u

    def series(self, t0, L):
        x0 = self.xi(t0)
        e = acb_series([0, I * PHI_U], prec=L).exp()
        xs = x0 * e
        return xs, I * PHI_U * xs


class URay:
    """u = (ra + (rb - ra) t) e^{-3 i pi/4}, t: 0 -> 1 (the ray arg xi = 3pi/8, |u| from ra to rb)"""
    kind = "uray"

    def __init__(self, ra, rb):
        self.ra, self.rb = arb(ra), arb(rb)
        self.e = (-I * PHI_U).exp()

    def xi(self, t):
        return (acb(self.ra) + acb(self.rb - self.ra) * t) * self.e

    def dxi(self, u):
        return acb(self.rb - self.ra) * self.e

    def series(self, t0, L):
        x0 = self.xi(t0)
        v = acb(self.rb - self.ra) * self.e
        return acb_series([x0, v], prec=L), acb_series([v], prec=L)


def dsc_to_pvc(Y, xi, w):
    """[D-1, D+1, S-1, S+1, C-2, C0, C+2] at xi -> [P-1, P+1, V-1, V+1, kappa C-2, kappa (C0 - 1), kappa C+2] (exact;
    xi, w may be balls)"""
    z = xi * xi
    out = [Y[0] / xi, Y[1] / xi]
    for i, m in enumerate(ODD):
        out.append(Y[2 + i] / z - (1 + I * m * w) * Y[i] / xi)
    return out + [KAPPA * (Y[4 + j] - (1 if n == 0 else 0)) for j, n in enumerate(EVEN)]


def pvc_to_dsc(Y, xi, w):
    z = xi * xi
    out = [xi * Y[0], xi * Y[1]]
    for i, m in enumerate(ODD):
        out.append(z * (Y[2 + i] + (1 + I * m * w) * Y[i]))
    return out + [Y[4 + j] / KAPPA + (1 if n == 0 else 0) for j, n in enumerate(EVEN)]


def selftest(trials=4, seed=1):
    """the u-system against validated_int.Sys (the xi-form the certificate integrates), at random complex states:
    rhs via the exact change of variables and the chain rule (as power series in xi); lin against the series derivative
    of rhs, for SysU, SysUR and SysUD (including the w, rho and delta directions). Prints the largest mismatches."""
    import random
    from flint import ctx
    ctx.prec = 200
    random.seed(seed)
    S, SU, SR, SD = VI.Sys(gravity=True), SysU(), SysUR(), SysUD()
    w = acb(VI.W0 * arb("0.7769940136"))

    def rnd():
        return acb(random.uniform(-1, 1), random.uniform(-1, 1))
    worst = [0.0, 0.0, 0.0, True]
    for _ in range(trials):
        xi0 = acb(random.uniform(0.05, 0.3), random.uniform(0.0, 0.2))
        Y0 = [rnd() for _ in range(7)] + [w, acb(2.5)]
        F, _ = S.rhs(Y0, xi0, 1 / xi0, acb(0))
        ctx.cap = 2
        xs = acb_series([xi0, 1], prec=2)
        pv = dsc_to_pvc([acb_series([Y0[i], F[i]], prec=2) for i in range(7)], xs, w)
        lhs = [(p.coeffs()[1] if len(p.coeffs()) > 1 else acb(0)) * (-xi0 ** 3 / 2) for p in pv]
        Yu = [p.coeffs()[0] for p in pv] + [w, acb(2.5)]
        u0 = 1 / (xi0 * xi0)
        Fu, _ = SU.rhs(Yu, u0, 1 / u0, acb(0))
        worst[0] = max(worst[0], max(float(abs(lhs[i] - Fu[i]).upper()) / (1 + float(abs(Fu[i]).upper()))
                                     for i in range(7)))
        for sys_ in (SU, SR, SD):
            if sys_ is SU:
                Y, x0 = list(Yu), u0
            elif sys_ is SR:
                Y, x0 = Yu + [acb(random.uniform(25, 260)) / RSCALE], (-I * PHI_U * random.uniform(0, 1)).exp()
            else:
                Y, x0 = Yu + [rnd()], u0
            dY = [rnd() for _ in range(7)] + [acb(0.2), acb(0)] + ([rnd()] if sys_ is not SU else [])
            Fx, cx = sys_.rhs(Y, x0, 1 / x0, acb(0))
            L = sys_.lin(cx, dY, x0, 1 / x0, acb(0))
            Ye = [acb_series([Y[i], dY[i]], prec=2) for i in range(len(Y))]
            Fe, _ = sys_.rhs(Ye, acb_series([x0], prec=2), acb_series([1 / x0], prec=2), acb_series([0], prec=2))
            d = [Fe[i].coeffs()[1] for i in range(7)]
            if sys_ is SD:                       # SysUD's delta column is a mean value enclosure: containment
                worst[3] = worst[3] and all(L[i].contains(d[i]) for i in range(7))
            else:
                k = 1 if sys_ is SU else 2
                worst[k] = max(worst[k], max(float(abs(d[i] - L[i]).upper()) for i in range(7)))
    print(f"u-system vs validated_int.Sys: rhs mismatch {worst[0]:.1e}; SysU lin {worst[1]:.1e}; SysUR lin "
          f"{worst[2]:.1e}; SysUD lin encloses the derivative: {worst[3]}")
    return worst


if __name__ == "__main__":
    selftest()
