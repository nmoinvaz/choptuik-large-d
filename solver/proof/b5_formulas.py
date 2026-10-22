#!/usr/bin/env python3
"""Lemma B5(1) (proof/B5_inner.md, section 2): the formulas the rigorous bounds use, written generically (any number
type with + - * /, and a linear solver), and checked here against b5_system.Model at random points (mpmath, 50 digits).

Notation: state (P, Phi, C, vt) of the exact M-model in inner variables (b5_system.py), Phi = z Pt (Pt = Phi/z),
eps = t z, D = 1 + s, D2 = 2 + s, Cop X = [C X], E0 = [P P]/2, Theta = E0 + vt, B0 = [[C E0] .], C0' = [C E0]/2,
Q0 = Cop^-1 (Phi + D P).

  g(z, t, P, Pt, C, vt)   = z * (Phi' of the exact model)  -- no division by z, so it can be evaluated on a disc
                            containing z = 0 (used for the eps-part Lipschitz bound by Cauchy/Schwarz estimates)
  zG0(z, P, Phi, Pt, C)   = z * (Phi' - Phi/(2 z^2)) of the eps -> 0 system: the non-principal part of the inner
                            Phi-rate, scaled; its Jacobian blocks J_PhiP, J_PhiPhi, J_PhiC (all O(1) as z -> 0)
  res_phi(z, eps, ...)    = z r_Phi / eps, r_Phi = (Phi' of the exact model) - (Phi' of the inner system) at the inner
                            solution: the exact decomposition of proof/B5_inner.md section 2, every term O(1); it
                            needs Q' = S/(2 z^2) (S = Cop Phi + A2^0 = 2 z^2 Q0' on the inner solution)
  tF_vt(z, eps, ...)      = z * (forcing of the vt-row) at the inner solution (vt = 0)

    python3 b5_formulas.py                 checks: g = z f_eps; zG0 = z(f_0 - Phi/(2z^2)); res_phi against
                                           (f_eps - f_0) z/eps at an exact inner-like state; tF_vt against f_eps
    python3 b5_formulas.py --mutate=res    drops the (1-eps) Cop[[C H]P]/2 term of res_phi: its check must fail
"""
import random
import sys

import mpmath as mp

import b5_system as BS

ODD, EVEN = BS.ODD, BS.EVEN
proj = BS.proj
MUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)


class Ops:
    """operator algebra on harmonic dicts; num: callable making a constant; solve(A, b) for small matrices"""

    def __init__(self, w, solve, zero, one, I):
        self.solve_ = solve
        self.zero, self.one, self.I = zero, one, I
        self.s = {m: I * m * w for m in ODD}

    def D(self, X, k=1):
        return {m: (k + self.s[m]) * X[m] for m in ODD}

    def op(self, F, X):
        """[F X] (F even field, X odd vector)"""
        return proj(F, X, ODD, self.zero)

    def Cinv(self, C, X):
        A = [[C.get(a - b, self.zero) for b in ODD] for a in ODD]
        sol = self.solve_(A, [X[m] for m in ODD])
        return dict(zip(ODD, sol))

    def ee(self, A, B):
        return proj(A, B, EVEN, self.zero)

    @staticmethod
    def add(*Xs):
        out = dict(Xs[0])
        for X in Xs[1:]:
            for k in out:
                out[k] = out[k] + X[k]
        return out

    @staticmethod
    def sc(c, X):
        return {k: c * v for k, v in X.items()}


def g_scaled(o, z, t, P, Pt, C, vt):
    """z * Phi' of the exact model at the state (P, Phi = z Pt, C, vt), eps = t z; no division by z"""
    e = t * z
    e1 = e * (1 - e)
    one = o.one
    Phi = o.sc(z, Pt)
    E0 = {k: v / 2 for k, v in o.ee(P, P).items()}
    Th = o.add(E0, vt)
    CT = o.ee(C, Th)
    Bop = lambda X: o.op(CT, X)
    Cop = lambda X: o.op(C, X)
    # Q from Eop Q = Mop Phi + D P - e1 Cop Bop P, Mop = I - t Cop^2, Eop = Cop (I - e D) + e1 z Bop
    rhs = o.add(Phi, o.sc(-t, Cop(Cop(Phi))), o.D(P), o.sc(-e1, Cop(Bop(P))))
    A = [[sum((C.get(a - c_, o.zero) * ((one - e * (1 + o.s[c_])) if c_ == b else o.zero) for c_ in ODD), o.zero)
          + e1 * z * CT.get(a - b, o.zero) for b in ODD] for a in ODD]
    Q = dict(zip(ODD, o.solve_(A, [rhs[m] for m in ODD])))
    QQ = o.ee(Q, Q)
    Ee = {k: E0[k] + e * z * QQ[k] / 2 for k in EVEN}
    hh = {k: Ee[k] - (1 + e) * Th[k] + e * z * (o.ee(Th, Ee)[k] - (1 - e) * o.ee(Th, Th)[k]) for k in EVEN}
    Cp = {k: (1 - e) * CT[k] / 2 for k in EVEN}            # C'
    CpT = o.ee(Cp, Th)
    Ch2 = {k: v / 2 for k, v in o.ee(C, hh).items()}       # [C h/2]  (= [C eps z Theta'])
    # T1 = Eop (Cop Pt + A2/z)/2, A2/z = -D2 Q + (1-e) Bop P
    A2z = o.add(o.sc(-one, o.D(Q, 2)), o.sc(1 - e, Bop(P)))
    X1 = o.add(Cop(Pt), A2z)
    EX1 = o.add(Cop(o.add(X1, o.sc(-e, o.D(X1)))), o.sc(e1 * z, Bop(X1)))
    T1 = o.sc(one / 2, EX1)
    # T2 = -z Mop' Phi = -t z Cop^2 Pt + t z^2 (Cop' Cop + Cop Cop') Pt
    T2 = o.add(o.sc(-t * z, Cop(Cop(Pt))), o.sc(t * z * z, o.add(o.op(Cp, Cop(Pt)), Cop(o.op(Cp, Pt)))))
    # T3 = z Eop' Q, Eop' = Cop'(I - e D) + e1 Bop + e1 z Bop',  e1 z Bop' X = e1 z [[C' Th] X] + (1-e)[[C h/2] X]
    T3 = o.sc(z, o.add(o.op(Cp, o.add(Q, o.sc(-e, o.D(Q)))), o.sc(e1, Bop(Q)), o.sc(e1 * z, o.op(CpT, Q)),
                       o.sc(1 - e, o.op(Ch2, Q))))
    # T4 = -(D - e1 Cop Bop) Phi/2 = -(D - e1 Cop Bop) z Pt / 2
    T4 = o.sc(-z / 2, o.add(o.D(Pt), o.sc(-e1, Cop(Bop(Pt)))))
    # T5 = z e1 (Cop' Bop + Cop Bop') P = z e1 Cop' Bop P + z e1 Cop [[C' Th] P] + (1-e) Cop [[C h/2] P]
    T5 = o.add(o.sc(z * e1, o.op(Cp, Bop(P))), o.sc(z * e1, Cop(o.op(CpT, P))), o.sc(1 - e, Cop(o.op(Ch2, P))))
    K = o.add(T1, T2, T3, T4, T5)
    # Mop^-1
    Mm = [[(one if a == b else o.zero) - t * sum((C.get(a - c_, o.zero) * C.get(c_ - b, o.zero) for c_ in ODD), o.zero)
           for b in ODD] for a in ODD]
    return dict(zip(ODD, o.solve_(Mm, [K[m] for m in ODD])))


def zG0(o, z, P, Phi, Pt, C):
    """z (Phi' - Phi/(2z^2)) of the inner system, = (Cop^2 - I) Pt/2 + Cop(B0 P - D2 Q0)/2 + z [C0' Q0] - D Phi/2"""
    E0 = {k: v / 2 for k, v in o.ee(P, P).items()}
    CE = o.ee(C, E0)
    Cop = lambda X: o.op(C, X)
    Q0 = o.Cinv(C, o.add(Phi, o.D(P)))
    a = o.sc(o.one / 2, o.add(Cop(Cop(Pt)), o.sc(-o.one, Pt)))
    b = o.sc(o.one / 2, Cop(o.add(o.op(CE, P), o.sc(-o.one, o.D(Q0, 2)))))
    c = o.sc(z / 2, o.op(CE, Q0))
    d = o.sc(-o.one / 2, o.D(Phi))
    return o.add(a, b, c, d), Q0


def jac_blocks(o, z, P, Phi, Pt, Ct, C, dP=None, dPhi=None, dC=None):
    """directional derivatives of zG0: J_PhiP dP, J_PhiPhi dPhi, J_PhiC dC (analytic, no division by z).
    Ct = (C - 1)/z (so (Cop^2 - I)/z = (Cop + I) Ct-op)"""
    E0 = {k: v / 2 for k, v in o.ee(P, P).items()}
    CE = o.ee(C, E0)
    Cop = lambda X: o.op(C, X)
    Q0 = o.Cinv(C, o.add(Phi, o.D(P)))
    out = {}
    if dP is not None:
        dE = o.ee(P, dP)                                     # d E0 = [P dP]
        dCE = o.ee(C, dE)
        dQ = o.Cinv(C, o.D(dP))
        out["P"] = o.add(o.sc(o.one / 2, Cop(o.add(o.op(dCE, P), o.op(CE, dP), o.sc(-o.one, o.D(dQ, 2))))),
                         o.sc(z / 2, o.add(o.op(dCE, Q0), o.op(CE, dQ))))
    if dPhi is not None:
        dQ = o.Cinv(C, dPhi)
        CtX = o.op(Ct, dPhi)
        out["F"] = o.add(o.sc(o.one / 2, o.add(Cop(CtX), CtX)), o.sc(-o.one / 2, Cop(o.D(dQ, 2))),
                         o.sc(z / 2, o.op(CE, dQ)), o.sc(-o.one / 2, o.D(dPhi)))
    if dC is not None:
        dCE = o.ee(dC, E0)
        dQ = o.sc(-o.one, o.Cinv(C, o.op(dC, Q0)))
        out["C"] = o.add(o.sc(o.one / 2, o.add(o.op(dC, Cop(Pt)), Cop(o.op(dC, Pt)))),
                         o.sc(o.one / 2, o.op(dC, o.add(o.op(CE, P), o.sc(-o.one, o.D(Q0, 2))))),
                         o.sc(o.one / 2, Cop(o.op(dCE, P))), o.sc(-o.one / 2, Cop(o.D(dQ, 2))),
                         o.sc(z / 2, o.add(o.op(dCE, Q0), o.op(CE, dQ))))
    return out


def res_phi(o, z, e, P, Pt, C, Qp):
    """z r_Phi / eps at the inner solution (vt = 0); Qp = Q0' (= S/(2 z^2)); e = eps, t = e/z enters only via
    Mop^-1 = (I - t Cop^2)^-1, passed as tz = t (the caller gives t separately through o.t)"""
    t = o.t
    one = o.one
    Phi = o.sc(z, Pt)
    E0 = {k: v / 2 for k, v in o.ee(P, P).items()}
    CE = o.ee(C, E0)
    B0 = lambda X: o.op(CE, X)
    Cop = lambda X: o.op(C, X)
    C0p = {k: v / 2 for k, v in CE.items()}
    Cp0 = lambda X: o.op(C0p, X)
    Q0 = o.Cinv(C, o.add(Phi, o.D(P)))
    # Eop and its inverse applied
    A = [[sum((C.get(a - c_, o.zero) * ((one - e * (1 + o.s[c_])) if c_ == b else o.zero) for c_ in ODD), o.zero)
          + e * (1 - e) * z * CE.get(a - b, o.zero) for b in ODD] for a in ODD]
    Einv = lambda X: dict(zip(ODD, o.solve_(A, [X[m] for m in ODD])))
    Eop = lambda X: o.add(Cop(o.add(X, o.sc(-e, o.D(X)))), o.sc(e * (1 - e) * z, B0(X)))
    dQh = Einv(o.add(o.sc(-one, Cop(Cop(Pt))), o.sc(-(1 - e), Cop(B0(P))), Cop(o.D(Q0)), o.sc(-(1 - e) * z, B0(Q0))))
    Q = o.add(Q0, o.sc(e, dQh))
    QQ = o.ee(Q, Q)
    H = {k: -E0[k] + z * QQ[k] / 2 + e * z * o.ee(E0, E0)[k] + e * z * z / 2 * o.ee(E0, QQ)[k] for k in EVEN}
    CH = o.ee(C, H)
    C0pE = o.ee(C0p, E0)
    Phip = o.add(Cop(Qp), Cp0(Q0), o.sc(-one / 2, o.D(Pt)))          # Phi_in' = Cop Q0' + C0' Q0 - D Pt/2
    calA = o.add(Cp0(dQh), o.sc(-one, Cp0(Q)), o.sc(-(1 - e), Cp0(o.D(Q))), o.sc(1 - e, B0(Q)),
                 o.sc((1 - e) ** 2 * z, o.op(C0pE, Q)), o.sc((1 - e) / 2, o.op(CH, Q)))
    terms = [Cop(Cop(Phip)),
             o.sc(z, o.add(o.sc(-one, Cop(o.D(Qp))), o.sc((1 - e) * z, B0(Qp)))),
             o.sc(-one / 2, Eop(o.add(o.D(dQh, 2), B0(P)))),
             o.sc(-one, Cop(Cop(Pt))),
             o.sc((1 - e) * z, o.add(Cp0(Cop(Pt)), Cop(Cp0(Pt)))),
             o.sc(z, calA),
             o.sc((1 - e) * z / 2, Cop(B0(Pt))),
             o.sc(z * (1 - e) ** 2, Cp0(B0(P))),
             o.sc(z * (1 - e) ** 2, Cop(o.op(C0pE, P)))]
    if MUT != "res":
        terms.append(o.sc((1 - e) / 2, Cop(o.op(CH, P))))
    Kb = o.add(*terms)
    Mm = [[(one if a == b else o.zero) - t * sum((C.get(a - c_, o.zero) * C.get(c_ - b, o.zero) for c_ in ODD), o.zero)
           for b in ODD] for a in ODD]
    return dict(zip(ODD, o.solve_(Mm, [Kb[m] for m in ODD])))


def tF_vt(o, z, e, P, Pt, Q):
    """z * (vt-row forcing) at vt = 0: (-E0 + z[Q Q]/2 - z[P Pt])/2 + eps z ([E0 E0] + (z/2)[E0 [Q Q]])/2"""
    E0 = {k: v / 2 for k, v in o.ee(P, P).items()}
    QQ = o.ee(Q, Q)
    PPt = o.ee(P, Pt)
    EE = o.ee(E0, E0)
    EQQ = o.ee(E0, QQ)
    return {k: (-E0[k] + z * QQ[k] / 2 - z * PPt[k]) / 2 + e * z * (EE[k] + z / 2 * EQQ[k]) / 2 for k in EVEN}


# ----------------------------------------------------------------------------------------------------------------------
def mp_ops(w):
    def solve(A, b):
        return list(mp.lu_solve(mp.matrix(A), mp.matrix(b)))
    return Ops(w, solve, mp.mpc(0), mp.mpf(1), mp.mpc(0, 1))


def checks(npts=5):
    mp.mp.dps = 50
    rnd = random.Random(2024)
    worst = {"g": 0, "zG0": 0, "jac": 0, "res": 0, "tF": 0}
    for _ in range(npts):
        cr = lambda: mp.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1))
        z = mp.mpc(rnd.uniform(0.002, 0.02), rnd.uniform(-0.005, 0.005))
        w = mp.mpf(rnd.uniform(1, 4))
        eps = mp.mpf(rnd.uniform(1e-6, 1e-5))
        o = mp_ops(w)
        P = {m: 2 * cr() for m in ODD}
        Pt = {m: 5 * cr() for m in ODD}
        C = {k: (1 if k == 0 else 0) + z * cr() for k in EVEN}
        vt = {k: cr() for k in EVEN}
        Phi = o.sc(z, Pt)
        # g vs z f_eps
        gv = g_scaled(o, z, eps / z, P, Pt, C, vt)
        _, dPhi, _, _ = BS.Model(w, eps, mp).f_eps(z, P, Phi, C, vt)
        worst["g"] = max(worst["g"], max(abs(gv[m] - z * dPhi[m]) / abs(gv[m]) for m in ODD))
        # zG0 vs z (f_0 - Phi/(2 z^2))
        zg, Q0 = zG0(o, z, P, Phi, Pt, C)
        _, dPhi0, _, G0 = BS.Model(w, 0, mp).f_0(z, P, Phi, C)
        worst["zG0"] = max(worst["zG0"], max(abs(zg[m] - z * G0[m]) / abs(zg[m]) for m in ODD))
        # Jacobian blocks vs finite differences of zG0 (mp.diff)
        Ct = {k: (C[k] - (1 if k == 0 else 0)) / z for k in EVEN}
        for blk, comps in (("P", ODD), ("F", ODD), ("C", EVEN)):
            for j in comps:
                unit = {k: (mp.mpf(1) if k == j else mp.mpf(0)) for k in comps}
                J = jac_blocks(o, z, P, Phi, Pt, Ct, C, **{{"P": "dP", "F": "dPhi", "C": "dC"}[blk]: unit})[blk]
                for m in ODD:
                    def f(h, m=m):
                        if blk == "P":
                            Pp = {k: P[k] + (h if k == j else 0) for k in ODD}
                            return zG0(o, z, Pp, Phi, Pt, C)[0][m]
                        if blk == "F":
                            Fp = {k: Phi[k] + (h if k == j else 0) for k in ODD}
                            return zG0(o, z, P, Fp, {k: v / z for k, v in Fp.items()}, C)[0][m]
                        Cp = {k: C[k] + (h if k == j else 0) for k in EVEN}
                        return zG0(o, z, P, Phi, Pt, Cp)[0][m]
                    worst["jac"] = max(worst["jac"], abs(mp.diff(f, 0) - J[m]) / (1 + abs(J[m])))
        # residual: build an exact inner-like state: P, C arbitrary, Q0' arbitrary, then res_phi vs (f_eps - f_0) z/eps
        # with the inner identities imposed: Phi' (inner) uses S = 2 z^2 Q0'; f_0 at the state gives Phi' = Cop Q0' +
        # C0' Q0 - D Pt/2 only if S/(2 z^2) = Q0', so take Q0' := S/(2 z^2) computed from the state
        E0 = {k: v / 2 for k, v in o.ee(P, P).items()}
        Q0 = o.Cinv(C, o.add(Phi, o.D(P)))
        S = o.add(o.op(C, Phi), o.sc(-z, o.D(Q0, 2)), o.sc(z, o.op(o.ee(C, E0), P)))
        Qp = o.sc(1 / (2 * z * z), S)
        o.t = eps / z
        rp = res_phi(o, z, eps, P, Pt, C, Qp)
        vt0 = {k: mp.mpc(0) for k in EVEN}
        _, dPe, _, dvte = BS.Model(w, eps, mp).f_eps(z, P, Phi, C, vt0)
        _, dP0, _, _ = BS.Model(w, 0, mp).f_0(z, P, Phi, C, vt0)
        worst["res"] = max(worst["res"], max(abs(rp[m] - (dPe[m] - dP0[m]) * z / eps) / (1 + abs(rp[m])) for m in ODD))
        # vt forcing: f_eps's vt' at vt = 0 equals tF_vt / z (the -(1+eps) vt/(2 eps z) term vanishes at vt = 0)
        Qex = BS.Model(w, eps, mp).Q_of(z, P, Phi, C, {k: E0[k] for k in EVEN})
        tf = tF_vt(o, z, eps, P, Pt, Qex)
        worst["tF"] = max(worst["tF"], max(abs(tf[k] - z * dvte[k]) / (1 + abs(tf[k])) for k in EVEN))
    return worst


if __name__ == "__main__":
    wv = checks()
    thr = {"g": 1e-40, "zG0": 1e-40, "jac": 1e-30, "res": 1e-30, "tF": 1e-40}
    ok = {k: wv[k] < thr[k] for k in wv}
    for k in wv:
        print(f"  {k:4s}: worst relative difference {mp.nstr(wv[k], 3)} ({'PASS' if ok[k] else 'FAIL'}, threshold {thr[k]})")
    if MUT:
        bad = not all(ok.values())
        print(f"MUTATION {MUT}: some check must FAIL: {'caught' if bad else 'NOT caught'}")
        sys.exit(0 if bad else 3)
    sys.exit(0 if all(ok.values()) else 1)
