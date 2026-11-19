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


if __name__ == "__main__":
    ok = True
    if "model" in OPTS:
        ok &= check_model()
    if "phiz" in OPTS:
        ok &= check_phiz()
    sys.exit(0 if ok else 1)
