#!/usr/bin/env python3
"""P2c review (#1019): Lemma B5(1) (B5_inner.md sections 1-4) recomputed independently (P2_remaining.md section 6).

Written from B5_inner.md, not from b5_bounds.py, b5_formulas.py, b5_numeric.py, b5_system.py or b5_data.py (none opened).
M = 1: odd harmonics {-1, 1} (P, Phi, Q), even {-2, 0, 2} (C, Theta, theta), every product projected; s = i mu omega,
D = 1 + s, D2 = 2 + s on odd harmonics; C X = [C X], B X = [[C Theta] X].

Stage A (this file's checks):
  --model    Lemma 1 (B5_inner.md 1.2) against B8's reduced system (B8_constants.md 1.2, coded in p2c_h8.py), at random
             complex states: the B8 rates converted to (P, Phi, Q, C, Theta) must satisfy Lemma 1's four relations.
  --phiz     section 1.3's Phi_z formula against the total derivative of the Phi relation along B8's flow (central
             differences at 256 bits), and Lemma 2's G0 against the eps -> 0 limit of Phi_z - Phi/(2 z^2).
             --mutate=G0 drops G0's C B0 P/(2z) term (K2): the G0 check must fail.
Stage B:
  --residual Lemma 3 from the eps-expansion: z^2 f^eps_Phi - z^2 f^0_Phi on the inner series as a power series in
             (tau = eps/z, z), its structural zeros (tau^0: Lemma 2; z^0, z^1 at every tau order: Lemma 3), and
             F_R = t r_Phi/eps = r/(tau z^2) at t -> 0 (R(0), B5_inner.md 2: 35.18).
             --mutate=offsol (K1 as the protocol wrote it) perturbs the inner series by A (1 + 1e-3); A enters only
             through P(0) = 2F, so this is another solution, not off the solution (R(0) moves, Lemma 3 still holds).
             --mutate=offphi (K1b) puts the state off the solution, Phi + 1e-3 z: Lemma 3's zeros must fail.
"""
import os
import random
import sys

from mpmath import mp, mpc, mpf

mp.prec = 256
HERE = os.path.dirname(os.path.abspath(__file__))
OPTS = {a.split("=", 1)[0][2:]: (a.split("=", 1)[1] if "=" in a else "1") for a in sys.argv[1:] if a.startswith("--")}
MUT = OPTS.get("mutate")
ODD, EVEN = (-1, 1), (-2, 0, 2)
WINF = 2 * mp.pi / mpf("1.64389375")


# ------------------------------------------------------------------------------- vectors and products (numbers)
def pr(f, g, out):
    r = {k: mpc(0) for k in out}
    for i, a in f.items():
        for j, b in g.items():
            if i + j in r:
                r[i + j] += a * b
    return r


def add(*fs):
    return {k: sum((f[k] for f in fs), mpc(0)) for k in fs[0]}


def sc(f, s):
    return {k: v * s for k, v in f.items()}


def sub(f, g):
    return {k: f[k] - g[k] for k in f}


def opm(c):
    """the operator X -> [c X] on odd vectors"""
    return {m: {q: c.get(m - q, mpc(0)) for q in ODD} for m in ODD}


def ap(M, f):
    return {m: sum((M[m][q] * f[q] for q in ODD), mpc(0)) for m in ODD}


def mm(X, Y):
    return {m: {q: sum((X[m][p] * Y[p][q] for p in ODD), mpc(0)) for q in ODD} for m in ODD}


def inv(X):
    a, b, c, d = X[1][1], X[1][-1], X[-1][1], X[-1][-1]
    det = a * d - b * c
    return {1: {1: d / det, -1: -b / det}, -1: {1: -c / det, -1: a / det}}


def madd(*Xs):
    return {m: {q: sum((X[m][q] for X in Xs), mpc(0)) for q in ODD} for m in ODD}


def msc(X, s):
    return {m: {q: X[m][q] * s for q in ODD} for m in ODD}


def eye(s=1):
    return {m: {q: (mpc(s) if m == q else mpc(0)) for q in ODD} for m in ODD}


def diag(f):
    return {m: {q: (f[m] if m == q else mpc(0)) for q in ODD} for m in ODD}


def norm(f):
    return max(abs(v) for v in f.values())


# ------------------------------------------------------------------------------------ B8's reduced system, (eps, z)
def b8_rates(V, S, beta, c, z, eps, w):
    """rates d/dz of (V, S, beta, c) from B8_constants.md 1.2 in (eps, z) form (p2c_h8.py's statement)"""
    Om = {m: 1j * m * w for m in ODD}
    C = opm(c)
    Ci = inv(C)
    Y = add(V, S)
    P = ap(Ci, Y)
    cb = pr(c, beta, EVEN)
    B = opm(cb)
    CC = mm(C, C)
    G = add(sc(ap(CC, S), 1 - eps), sc(ap(C, ap(B, S)), eps), sc({m: Om[m] * S[m] for m in ODD}, -z), sc(Y, -z),
            sc(ap(B, P), z), sc(ap(C, {m: Om[m] * P[m] for m in ODD}), -z))
    # (z - eps C^2) 2 z S_z = G
    Dm = madd(eye(z), msc(CC, -eps))
    S_z = sc(ap(inv(Dm), G), 1 / (2 * z))
    V_z = {m: Om[m] * S[m] / (2 * z) for m in ODD}
    c_z = {k: v / (2 * z) for k, v in cb.items()}
    SS, PP = pr(S, S, EVEN), pr(P, P, EVEN)
    # (1 - eps) beta = (1 - eps)(eps [SS] + z [PP])/2 + eps^2 [beta [SS]]/2 + eps z [beta [PP]]/2 - eps [beta beta]
    #                  - 2 eps z beta_z
    rhs0 = add(sc(add(sc(SS, eps), sc(PP, z)), (1 - eps) / 2), sc(pr(beta, SS, EVEN), eps ** 2 / 2),
               sc(pr(beta, PP, EVEN), eps * z / 2), sc(pr(beta, beta, EVEN), -eps))
    beta_z = {k: (rhs0[k] - (1 - eps) * beta[k]) / (2 * eps * z) for k in EVEN}
    return V_z, S_z, beta_z, c_z


# ------------------------------------------------------------------------------------ B5's Lemma 1 and section 1.3
class B5:
    def __init__(self, w, eps):
        self.w, self.eps = w, eps
        self.s = {m: 1j * m * w for m in ODD}
        self.D = diag({m: 1 + self.s[m] for m in ODD})
        self.D2 = diag({m: 2 + self.s[m] for m in ODD})

    def ops(self, C, Theta):
        Cm = opm(C)
        Bm = opm(pr(C, Theta, EVEN))
        return Cm, Bm

    def E(self, Cm, Bm, z):
        e = self.eps
        return madd(mm(Cm, madd(eye(), msc(self.D, -e))), msc(Bm, e * (1 - e) * z))

    def lemma1(self, P, Phi, Q, C, Theta, z, P_z, Q_z, C_z, Theta_z):
        """the four relations of Lemma 1 (left minus right)"""
        e = self.eps
        Cm, Bm = self.ops(C, Theta)
        CC = mm(Cm, Cm)
        r0 = sub(sc(P_z, 2 * z), Phi)
        lhs = ap(madd(eye(), msc(CC, -e / z)), Phi)
        rhs = add(ap(self.E(Cm, Bm, z), Q), sc(ap(self.D, P), -1), sc(ap(Cm, ap(Bm, P)), e * (1 - e)))
        r1 = sub(lhs, rhs)
        A2 = add(sc(ap(self.D2, Q), -z), sc(ap(Bm, P), (1 - e) * z))
        r2 = sub(sc(Q_z, 2 * z * z), add(ap(Cm, Phi), A2))
        r3 = sub(C_z, sc(pr(C, Theta, EVEN), (1 - e) / 2))
        PP, QQ = pr(P, P, EVEN), pr(Q, Q, EVEN)
        Ee = sc(add(PP, sc(QQ, e * z)), mpf(1) / 2)
        h = add(Ee, sc(Theta, -(1 + e)), sc(sub(pr(Theta, Ee, EVEN), sc(pr(Theta, Theta, EVEN), 1 - e)), e * z))
        r4 = sub(sc(Theta_z, 2 * e * z), h)
        return r0, r1, r2, r3, r4

    def Q_of(self, P, Phi, C, Theta, z):
        """Q from the Phi relation: E Q = M Phi + D P - eps(1 - eps) C B P"""
        e = self.eps
        Cm, Bm = self.ops(C, Theta)
        CC = mm(Cm, Cm)
        rhs = add(ap(madd(eye(), msc(CC, -e / z)), Phi), ap(self.D, P), sc(ap(Cm, ap(Bm, P)), -e * (1 - e)))
        return ap(inv(self.E(Cm, Bm, z)), rhs)

    def h_of(self, P, Q, Theta, z):
        e = self.eps
        PP, QQ = pr(P, P, EVEN), pr(Q, Q, EVEN)
        Ee = sc(add(PP, sc(QQ, e * z)), mpf(1) / 2)
        return add(Ee, sc(Theta, -(1 + e)), sc(sub(pr(Theta, Ee, EVEN), sc(pr(Theta, Theta, EVEN), 1 - e)), e * z))

    def phi_z(self, P, Phi, C, theta, z):
        """section 1.3's Phi_z, with Theta = theta + [PP]/2 and Q from the Phi relation"""
        e = self.eps
        ep = e * (1 - e)
        Theta = add(theta, sc(pr(P, P, EVEN), mpf(1) / 2))
        Q = self.Q_of(P, Phi, C, Theta, z)
        Cm, Bm = self.ops(C, Theta)
        CC = mm(Cm, Cm)
        Mm = madd(eye(), msc(CC, -e / z))
        Cz = sc(pr(C, Theta, EVEN), (1 - e) / 2)
        Czm = opm(Cz)
        Theta_z = sc(self.h_of(P, Q, Theta, z), 1 / (2 * e * z))
        Bzm = opm(add(pr(Cz, Theta, EVEN), pr(C, Theta_z, EVEN)))
        Mz = madd(msc(CC, e / z ** 2), msc(madd(mm(Czm, Cm), mm(Cm, Czm)), -e / z))
        Ez = madd(mm(Czm, madd(eye(), msc(self.D, -e))), msc(madd(Bm, msc(Bzm, z)), ep))
        A2 = add(sc(ap(self.D2, Q), -z), sc(ap(Bm, P), (1 - e) * z))
        bracket = add(sc(ap(self.E(Cm, Bm, z), add(ap(Cm, Phi), A2)), 1 / (2 * z * z)), sc(ap(Mz, Phi), -1),
                      ap(Ez, Q), sc(ap(madd(self.D, msc(mm(Cm, Bm), -ep)), Phi), -1 / (2 * z)),
                      sc(ap(madd(mm(Czm, Bm), mm(Cm, Bzm)), P), ep))
        return ap(inv(Mm), bracket)

    def G0(self, P, Phi, C, z):
        """Lemma 2: G0 = (C^2 - I) Phi/(2 z^2) + C(B0 P - D2 Q0)/(2z) + [C0' Q0] - D Phi/(2z)"""
        Cm = opm(C)
        E0 = sc(pr(P, P, EVEN), mpf(1) / 2)
        CE0 = pr(C, E0, EVEN)
        B0 = opm(CE0)
        Q0 = ap(inv(Cm), add(Phi, ap(self.D, P)))
        t1 = sc(ap(madd(mm(Cm, Cm), eye(-1)), Phi), 1 / (2 * z * z))
        inner = ap(B0, P) if MUT != "G0" else {m: mpc(0) for m in ODD}      # K2 drops C B0 P/(2z)
        t2 = sc(ap(Cm, sub(inner, ap(self.D2, Q0))), 1 / (2 * z))
        t3 = pr(sc(CE0, mpf(1) / 2), Q0, ODD)
        t4 = sc(ap(self.D, Phi), -1 / (2 * z))
        return add(t1, t2, t3, t4)


def rvec(rnd, keys, scale=1.0):
    return {k: mpc(rnd.uniform(-scale, scale), rnd.uniform(-scale, scale)) for k in keys}


def check_model():
    rnd = random.Random(1019)
    worst = mpf(0)
    for _ in range(10):
        w = mpf(rnd.uniform(2, 4)) + 1j * mpf(rnd.uniform(-0.1, 0.1))
        z = mpc(rnd.uniform(0.001, 0.01), rnd.uniform(0, 0.01))
        eps = mpf(10) ** -rnd.uniform(3, 12)
        V, S = rvec(rnd, ODD), rvec(rnd, ODD, 0.01)
        beta = rvec(rnd, EVEN, 0.01)
        c = {k: (1 if k == 0 else 0) + v for k, v in rvec(rnd, EVEN, 0.05).items()}
        V_z, S_z, beta_z, c_z = b8_rates(V, S, beta, c, z, eps, w)
        P = ap(inv(opm(c)), add(V, S))
        Pz = ap(inv(opm(c)), sub(add(V_z, S_z), pr(c_z, P, ODD)))
        Phi = sc(Pz, 2 * z)
        Q = sc(S, 1 / z)
        Q_z = sub(sc(S_z, 1 / z), sc(S, 1 / z ** 2))
        Theta = sc(beta, 1 / ((1 - eps) * z))
        Theta_z = sc(sub(sc(beta_z, 1 / z), sc(beta, 1 / z ** 2)), 1 / (1 - eps))
        rs = B5(w, eps).lemma1(P, Phi, Q, c, Theta, z, Pz, Q_z, c_z, Theta_z)
        scale = max(norm(Phi), norm(Q), norm(P), 1)
        for r in rs:
            worst = max(worst, norm(r) / scale)
    ok = worst < mpf(10) ** -40
    print(f"--model: Lemma 1 against B8's reduced system at 10 random states: largest relative residual "
          f"{mp.nstr(worst, 3)}: {'PASS' if ok else 'FAIL'}")
    return ok


def check_phiz():
    rnd = random.Random(10191)
    worst_tot, worst_g0 = mpf(0), mpf(0)
    for _ in range(10):
        w = mpf(rnd.uniform(2, 4)) + 1j * mpf(rnd.uniform(-0.1, 0.1))
        z0 = mpc(rnd.uniform(0.001, 0.01), rnd.uniform(0, 0.01))
        eps = mpf(10) ** -rnd.uniform(3, 8)
        V, S = rvec(rnd, ODD), rvec(rnd, ODD, 0.01)
        beta = rvec(rnd, EVEN, 0.01)
        c = {k: (1 if k == 0 else 0) + v for k, v in rvec(rnd, EVEN, 0.05).items()}

        def state(V, S, beta, c, z):
            V_z, S_z, beta_z, c_z = b8_rates(V, S, beta, c, z, eps, w)
            P = ap(inv(opm(c)), add(V, S))
            Phi = sc(ap(inv(opm(c)), sub(add(V_z, S_z), pr(c_z, P, ODD))), 2 * z)
            Theta = sc(beta, 1 / ((1 - eps) * z))
            return P, Phi, c, sub(Theta, sc(pr(P, P, EVEN), mpf(1) / 2)), (V_z, S_z, beta_z, c_z)

        P, Phi, C, th, rates = state(V, S, beta, c, z0)
        # total derivative of Phi along the flow: central differences in z with the state moved by its rates
        hh = mpf(10) ** -32
        vals = []
        for sgn in (1, -1):
            dz = sgn * hh
            mv = lambda f, fz: {k: f[k] + fz[k] * dz for k in f}
            # second-order step (Taylor with the rates; the O(h^2) state error cancels in the central difference)
            Vp, Sp, bp, cp = mv(V, rates[0]), mv(S, rates[1]), mv(beta, rates[2]), mv(c, rates[3])
            vals.append(state(Vp, Sp, bp, cp, z0 + dz)[1])
        tot = {m: (vals[0][m] - vals[1][m]) / (2 * hh) for m in ODD}
        f = B5(w, eps).phi_z(P, Phi, C, th, z0)
        worst_tot = max(worst_tot, norm(sub(f, tot)) / max(norm(f), 1))
        # Lemma 2: eps -> 0 with (P, Phi, C, theta) fixed
        tiny = B5(w, mpf(10) ** -70)
        f0 = tiny.phi_z(P, Phi, C, th, z0)
        g0 = add(sc(Phi, 1 / (2 * z0 * z0)), tiny.G0(P, Phi, C, z0))
        worst_g0 = max(worst_g0, norm(sub(f0, g0)) / max(norm(f0), 1))
    ok1 = worst_tot < mpf(10) ** -25
    ok2 = worst_g0 < mpf(10) ** -40
    print(f"--phiz: section 1.3's Phi_z against the total derivative along B8's flow (central differences, h 1e-32): "
          f"largest relative difference {mp.nstr(worst_tot, 3)}: {'PASS' if ok1 else 'FAIL'}")
    print(f"--phiz: Lemma 2's G0 + Phi/(2z^2) against Phi_z at eps 1e-70 with (P, Phi, C, theta) fixed: "
          f"largest relative difference {mp.nstr(worst_g0, 3)}" + (f" [mutate {MUT}]" if MUT else "")
          + f": {'PASS' if ok2 else 'FAIL'}")
    return ok1 and ok2


# ------------------------------------------------------------------------------ stage B: series in (tau, z)
def series_tools():
    import p2c_h8 as H
    return H


def inner_b5(H, A, wf, J):
    """the inner series in B5's variables (P, Phi, C, Theta, Q) from #1017's solver (eps^0 part of B8's system)"""
    Min = H.Model(A, wf, H.keep_inner(J))
    Z = Min.solve(iters=2 * J + 10)
    V, S, beta, c = Z
    keep = H.keep_inner(J)
    C = H.op_of(c, keep)
    P = H.apply(H.matinv(C), H.vadd(V, S))
    Phi = {m: P[m].zdz() for m in ODD}
    Theta = {k: H.div_z(beta[k]) for k in EVEN}
    Q = {m: H.div_z(S[m]) for m in ODD}
    return P, Phi, c, Theta, Q


def residual_series(H, w, P, Phi, C, keep):
    """r = z^2 (f^eps_Phi - f^0_Phi) on the state (P, Phi, C, theta = 0), a power series in (tau, z); eps = tau z"""
    Ser = H.Ser
    one = Ser.const(1, keep)
    E = lambda f: f.shift(1, 1)                       # times eps = tau z
    T = lambda f: f.shift(1, 0)                       # times tau = eps/z
    Zs = lambda f: f.shift(0, 1)                      # times z
    vE = lambda f: {k: E(v) for k, v in f.items()}
    vT = lambda f: {k: T(v) for k, v in f.items()}
    vZ = lambda f: {k: Zs(v) for k, v in f.items()}
    mE = lambda X: {m: {q: E(X[m][q]) for q in ODD} for m in ODD}
    mT = lambda X: {m: {q: T(X[m][q]) for q in ODD} for m in ODD}
    mZ = lambda X: {m: {q: Zs(X[m][q]) for q in ODD} for m in ODD}
    madd_ = lambda *Xs: {m: {q: sum((X[m][q] for X in Xs), Ser({}, keep)) for q in ODD} for m in ODD}
    msub_ = lambda X, Y: {m: {q: X[m][q] - Y[m][q] for q in ODD} for m in ODD}
    vadd_ = lambda *fs: {k: sum((f[k] for f in fs), Ser({}, keep)) for k in fs[0]}
    vsub_ = lambda f, g: {k: f[k] - g[k] for k in f}
    vsc = lambda f, s: {k: v * s for k, v in f.items()}
    I2 = {m: {q: (one if m == q else Ser({}, keep)) for q in ODD} for m in ODD}
    from mpmath import iv
    Dg = {m: 1 + iv.mpc(0, m) * w for m in ODD}
    D2g = {m: 2 + iv.mpc(0, m) * w for m in ODD}
    Dv = lambda f: {m: f[m] * Dg[m] for m in ODD}
    D2v = lambda f: {m: f[m] * D2g[m] for m in ODD}
    Dm = {m: {q: (Ser.const(Dg[m], keep) if m == q else Ser({}, keep)) for q in ODD} for m in ODD}
    ap_, mm_, pr_, op_, inv_ = H.apply, H.matmul, H.pr, H.op_of, H.matinv
    half = mpf(1) / 2
    PP = pr_(P, P, EVEN, keep)
    Theta = vsc(PP, half)                              # theta = 0: Theta = E0 = [PP]/2
    Cm = op_(C, keep)
    CT = pr_(C, Theta, EVEN, keep)
    Bm = op_(CT, keep)
    CC = mm_(Cm, Cm)
    epf = lambda f: vsub_(vE(f), vE(vE(f)))           # eps (1 - eps) on vectors
    mep = lambda X: msub_(mE(X), mE(mE(X)))
    # E = C (I - eps D) + eps' z B ; M = I - tau C^2
    Emat = madd_(msub_(Cm, mE(mm_(Cm, Dm))), mZ(mep(Bm)))
    Mmat = msub_(I2, mT(CC))
    # Q from E Q = M Phi + D P - eps' C B P
    Q = ap_(inv_(Emat), vsub_(vadd_(ap_(Mmat, Phi), Dv(P)), epf(ap_(Cm, ap_(Bm, P)))))
    QQ = pr_(Q, Q, EVEN, keep)
    Ee = vsc(vadd_(PP, vE(vZ(QQ))), half)              # E_eps = ([PP] + eps z [QQ])/2
    # C_z = (1 - eps)[C Theta]/2 ; z Theta_z = (h/eps)/2 with h/eps = z[QQ]/2 - Theta + z([Theta E_eps] - (1 - eps)[Theta Theta])
    Cz = vsc(vsub_(CT, vE(CT)), half)
    TT = pr_(Theta, Theta, EVEN, keep)
    hoe = vadd_(vsc(vZ(QQ), half), vsc(Theta, -1), vZ(vsub_(pr_(Theta, Ee, EVEN, keep), vsub_(TT, vE(TT)))))
    zThz = vsc(hoe, half)
    Czm = op_(Cz, keep)
    # eps' B_z = (1 - eps) eps op(C_z Theta) + (1 - eps) tau op(C z Theta_z)
    CzT = pr_(Cz, Theta, EVEN, keep)
    CzTh = pr_(C, zThz, EVEN, keep)
    epBz = madd_(mep(op_(CzT, keep)), msub_(mT(op_(CzTh, keep)), mE(mT(op_(CzTh, keep)))))
    # eps' z B_z = z * eps' B_z
    A2 = vadd_(vsc(vZ(D2v(Q)), -1), vZ(vsub_(ap_(Bm, P), vE(ap_(Bm, P)))))
    Sx = vadd_(ap_(Cm, Phi), A2)                       # C Phi + A2
    # z^2 M_z Phi = eps C^2 Phi - eps z (C_z C + C C_z) Phi
    z2MzPhi = vsub_(vE(ap_(CC, Phi)), vE(vZ(ap_(madd_(mm_(Czm, Cm), mm_(Cm, Czm)), Phi))))
    # E_z = C_z (I - eps D) + eps' B + eps' z B_z
    Ez = madd_(msub_(Czm, mE(mm_(Czm, Dm))), mep(Bm), mZ(epBz))
    br = vadd_(vsc(ap_(Emat, Sx), half), vsc(z2MzPhi, -1), vZ(vZ(ap_(Ez, Q))),
               vsc(vZ(vsub_(Dv(Phi), epf(ap_(Cm, ap_(Bm, Phi))))), -half),
               vZ(vZ(vadd_(epf(ap_(Czm, ap_(Bm, P))), ap_(Cm, ap_(epBz, P))))))
    Fe = ap_(inv_(Mmat), br)
    # z^2 f^0 = Phi/2 + (C^2 - I) Phi/2 + z C (B0 P - D2 Q0)/2 + z^2 [C0' Q0] - z D Phi/2
    E0 = Theta
    CE0 = pr_(C, E0, EVEN, keep)
    B0 = op_(CE0, keep)
    Q0 = ap_(inv_(Cm), vadd_(Phi, Dv(P)))
    F0 = vadd_(vsc(Phi, half), vsc(ap_(msub_(CC, I2), Phi), half), vsc(vZ(ap_(Cm, vsub_(ap_(B0, P), D2v(Q0)))), half),
               vZ(vZ(pr_(vsc(CE0, half), Q0, ODD, keep))), vsc(vZ(Dv(Phi)), -half))
    return vsub_(Fe, F0)


def check_residual():
    H = series_tools()
    from mpmath import iv
    A, wf = H.params(True)
    Ain = A * (1 + iv.mpf("1e-3")) if MUT == "offsol" else A
    J = 24
    P, Phi, C, Theta, Q = inner_b5(H, Ain, wf, J)
    if MUT == "offphi":                                # K1b: off the solution, Phi + 1e-3 z (B5_inner.md 6's kind)
        for m in ODD:
            Phi[m] = Phi[m] + H.Ser({(0, 1): iv.mpc(iv.mpf("1e-3"))}, Phi[m].keep)
    keep = H.keep_box(3, J)
    emb = lambda f: {k: H.Ser(dict(v.c), keep) for k, v in f.items()}
    w = wf * H.WINF
    r = residual_series(H, w, emb(P), emb(Phi), emb(C), keep)
    worst0, worst01, R0 = mpf(0), mpf(0), {}
    for m in ODD:
        for (a, j), v in r[m].c.items():
            mag = mpf(abs(v).b)
            if a == 0:                                 # relative to the size of the state's coefficient there
                ref = max(mpf(abs(Phi[m].coef(0, j)).b), mpf(abs(Phi[m].coef(0, max(j - 1, 0))).b), mpf(10) ** -300)
                worst0 = max(worst0, mag / ref)
            elif j <= 1:
                worst01 = max(worst01, mag)
            if (a, j) == (1, 2):
                R0[m] = mpc(mp.mpf(v.real.mid.a), mp.mpf(v.imag.mid.a))
    ok0 = worst0 < mpf(10) ** -40
    # K1's check (protocol 6.1): |t r_Phi/eps| at tau -> 0 on |t| = 1e-6 Z against |t| = Z, 0 <= arg t <= pi/2 + kappa
    Zr = mpf("0.0085")
    az = mp.pi / 2 + mpf("0.3")
    def FR(tt):
        out = mpf(0)
        for m in ODD:
            s = mpc(0)
            for (a, j), v in r[m].c.items():
                if a == 1:
                    s += mpc(mp.mpf(v.real.mid.a), mp.mpf(v.imag.mid.a)) * tt ** (j - 2)
            out = max(out, abs(s))
        return out
    pts = [az * k / 31 for k in range(32)]
    big = max(FR(Zr * mp.expj(f)) for f in pts)
    small = max(FR(Zr * mpf("1e-6") * mp.expj(f)) for f in pts)
    okK1 = small <= 3 * big
    ok01 = worst01 < mpf(10) ** -40
    r0 = max(abs(v) for v in R0.values())
    okR = abs(r0 - mpf("35.18")) < mpf("0.005")
    print(f"--residual: z^2 (f^eps - f^0) on the inner series (z^{J}), as a series in (tau, z)"
          + (f" [mutate {MUT}]" if MUT else "") + ":")
    print(f"  tau^0 part (Lemma 2: zero): largest coefficient relative to the state's {mp.nstr(worst0, 3)}: "
          f"{'PASS' if ok0 else 'FAIL'}")
    print(f"  z^0 and z^1 at tau >= 1 (Lemma 3: zero): largest coefficient {mp.nstr(worst01, 3)}: "
          f"{'PASS' if ok01 else 'FAIL'}")
    print("  R(0) = t r_Phi/eps at t -> 0: " + ", ".join(f"[{m}] {mp.nstr(v, 8)} (|.| {mp.nstr(abs(v), 6)})"
                                                     for m, v in R0.items())
          + f"; against B5_inner.md 2's 35.18: {'PASS' if okR else 'FAIL'}")
    print(f"  K1's check: max |t r_Phi/eps| on |t| = 1e-6 Z {mp.nstr(small, 5)}, on |t| = Z {mp.nstr(big, 5)}; "
          f"at most 3 times: {'PASS' if okK1 else 'FAIL'}")
    return ok0 and ok01 and okR and okK1


if __name__ == "__main__":
    ok = True
    if "residual" in OPTS:
        ok &= check_residual()
    if "model" in OPTS:
        ok &= check_model()
    if "phiz" in OPTS:
        ok &= check_phiz()
    sys.exit(0 if ok else 1)
