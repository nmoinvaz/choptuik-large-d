#!/usr/bin/env python3
"""Linear perturbations of the D = infinity inner CSS solution of the HE kappa-model (issue #376).

    python3 css_inner_pert.py check [KAPPA]                    the checks below (exit 1 if one fails or a control does not fire)
    python3 css_inner_pert.py det KAPPA LAM [--r0= --r1=]      det M(lambda) and its smallest singular value
    python3 css_inner_pert.py root KAPPA LAM [--set=..]        secant root of det M at each (r0, r1) setting
    python3 css_inner_pert.py count KAPPA RE0 RE1 IM0 IM1 [--set=..] [--plant=Z] [--mutate=M] [--K=6]
                                                              winding number of det M around the box
    python3 css_inner_pert.py kpath K0 K1 STEP LAM [--set=..]   the root followed in kappa (background re-solved)

All commands take --out=FILE (jsonl, appended).  Settings (r0, r1): A = (0.1, 0.3), B = (0.1, 0.25), C = (0.09, 0.3).

Equations.  The inner PDE of matter/CSS.md section 8 (css_inner_pde.py), with tau = -ln(-t), xi = sqrt(2 s)/(-t),
X = xi d_xi + d_tau (so d_t = e^tau X and d_s = e^{2 tau} xi^-1 d_xi), z = xi^2, theta = xi d_xi:
    c^2 theta F = z [ (X^2 + X) F - (X c / c) X F + (2 kappa Fbar / rho) (X F)^2 ],
    c theta c   = 2 z h (X F)(X Fbar),        rho = 1 - kappa F Fbar,  h = rho^-2.
Perturbation (A and B independent, the real system complexified):
    F = e^{-i om tau} (f + eps e^{lam tau} A),  Fbar = e^{i om tau} (g + eps e^{lam tau} B),  c = b + eps e^{lam tau} beta,
so X acts as theta - i om on f, theta + i om on g, theta on b, and theta + lam - i om, theta + lam + i om, theta + lam
on A, B, beta.  Gauge: beta(0) = 0 (the lapse at xi = 0 is fixed by the matching; a time reparametrization
t -> t + eta(t) with eta ~ (-t)^{1 - lam} is a perturbation with beta(0) = lam - 1, so this removes it for lam != 1).
Regular data A(0), B(0); the series in z is divergent like the background's (fast mode e^{-b^2/(2 xi^2)}).
A growth rate is a lam at which a regular perturbation has a zero Stokes jump, as for the background (GAMMA.md
section 5).  The jumps of (A, B) at xi = r1 between the upper and the lower lateral solution (optimally truncated series
at r0 e^{+-3 i pi/8}, ray to |xi| = r1, arc to the real axis; both sides integrated, so complex lam is allowed) are
linear in (A(0), B(0)): J = M(lam) (A(0), B(0)).  The growth rates are zeros of det M at every (r0, r1); a zero that
moves with (r0, r1) is an artefact of the jump at one point (det of the fast modes' (A, B) values at r1).

Exact modes (from the symmetries; checked as series identities and as null vectors of M):
    lam = 0      phase rotation:          A = i f, B = -i g, beta = 0
    lam = 1      t-translation:           A = theta f - i om f, B = theta g + i om g, beta = theta b
    lam = 2      s-translation + gauge:   A = f'/xi - 2 b1 (theta f - i om f), B = g'/xi - 2 b1 (theta g + i om g),
                                          beta = b'/xi - 2 b1 (theta b + b),   b = 1 + b1 xi^2 + ...
    lam = +-i om target isometries:       (A, B) = (1, -kappa g^2) at +i om, (-kappa f^2, 1) at -i om, beta = 0

Controls (AGENTS.md): --mutate=nometric (beta not fed back into the A, B equations) and --mutate=hamsign (the sign of
the beta equation's source flipped) must remove the lam = 1 and lam = 2 zeros; --plant=Z multiplies det M by
(lam - Z) and must add one to a box holding Z; the series identities must fail for a wrong lam.
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import css_inner as ci  # noqa: E402

SETTINGS = {"A": (0.1, 0.3), "B": (0.1, 0.25), "C": (0.09, 0.3)}
PHI0 = 3 * np.pi / 8
MUTATE = [None]


# --- lazy power series in z (coefficients computed on demand and cached) -------------------------------------------
class LS:
    def __init__(self):
        self.c = []

    def __getitem__(self, k):
        c = self.c
        while len(c) <= k:
            c.append(self.coef(len(c)))
        return c[k]


class Const(LS):
    def __init__(self, v):
        super().__init__()
        self.v = list(v)

    def coef(self, k):
        return self.v[k] if k < len(self.v) else mp.mpc(0)


class Lin(LS):
    def __init__(self, a, ca, b=None, cb=0):
        super().__init__()
        self.a, self.ca, self.b, self.cb = a, ca, b, cb

    def coef(self, k):
        return self.ca * self.a[k] + (self.cb * self.b[k] if self.b is not None else 0)


class Mul(LS):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b

    def coef(self, k):
        self.a[k]; self.b[k]
        return mp.fdot(self.a.c[:k + 1], self.b.c[k::-1] if k else self.b.c[:1])


class Theta(LS):
    """(theta + nu) a, theta z^k = 2 k z^k."""
    def __init__(self, a, nu):
        super().__init__()
        self.a, self.nu = a, nu

    def coef(self, k):
        return (2 * k + self.nu) * self.a[k]


class Inv(LS):
    def __init__(self, a):
        super().__init__()
        self.a = a

    def coef(self, k):
        if k == 0:
            return 1 / self.a[0]
        self.a[k]
        return -mp.fdot(self.a.c[1:k + 1], self.c[k - 1::-1]) / self.a.c[0]


class Rec(LS):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def coef(self, k):
        return self.fn(k, self.c)


ONE = Const([mp.mpc(1)])


class Background:
    """The inner solution's formal series (css_inner.series) and the lambda-independent nodes of the linearization."""

    def __init__(self, kap, f0, om, J):
        self.kap, self.f0, self.om, self.J = kap, f0, om, J
        a, ag, be = ci.series(kap, f0, om, J)
        self.coef = (a, ag, be)
        I = mp.mpc(0, 1)
        omm = mp.mpf(om)
        self.f, self.g, self.b = Const(a), Const(ag), Const(be)
        self.XF0 = Theta(self.f, -I * omm)
        self.XG0 = Theta(self.g, I * omm)
        self.Xc0 = Theta(self.b, 0)
        self.rho0 = Lin(ONE, 1, Mul(self.f, self.g), -mp.mpf(kap))
        self.irho0 = Inv(self.rho0)
        self.irho0sq = Mul(self.irho0, self.irho0)
        self.ib = Inv(self.b)
        self.ibsq = Mul(self.ib, self.ib)
        self.b2 = Mul(self.b, self.b)
        self.T0 = Mul(self.Xc0, self.ib)
        self.XF0sq = Mul(self.XF0, self.XF0)
        self.XG0sq = Mul(self.XG0, self.XG0)
        self.XFG0 = Mul(self.XF0, self.XG0)
        self.girho0 = Mul(self.g, self.irho0)
        self.firho0 = Mul(self.f, self.irho0)


def pert_series(bg, lam, A0, B0, J, mutate=None):
    """Coefficients (lists of mpc, length J) of A, B, beta in z for the data (A0, B0) at growth rate lam."""
    I = mp.mpc(0, 1)
    om, kap = mp.mpf(bg.om), mp.mpf(bg.kap)
    lam = mp.mpc(lam)
    nuA, nuB = lam - I * om, lam + I * om
    met = 0 if mutate == "nometric" else 1
    hs = -1 if mutate == "hamsign" else 1
    a, ag, be = bg.coef
    st = {}

    def fA(k, c):
        if k == 0:
            return mp.mpc(A0)
        s = st["br"][k - 1]
        s -= mp.fsum(bg.b2[i] * 2 * (k - i) * c[k - i] for i in range(1, k))
        s -= met * mp.fsum(st["tbb"][i] * 2 * (k - i) * a[k - i] for i in range(1, k))
        return s / (2 * k * bg.b2[0])

    def fB(k, c):
        if k == 0:
            return mp.mpc(B0)
        s = st["brb"][k - 1]
        s -= mp.fsum(bg.b2[i] * 2 * (k - i) * c[k - i] for i in range(1, k))
        s -= met * mp.fsum(st["tbb"][i] * 2 * (k - i) * ag[k - i] for i in range(1, k))
        return s / (2 * k * bg.b2[0])

    def fbe(k, c):
        if k == 0:
            return mp.mpc(0)
        s = st["r2"][k - 1]
        s -= mp.fsum(be[i] * 2 * (k - i) * c[k - i] for i in range(1, k))
        s -= mp.fsum(c[i] * 2 * (k - i) * be[k - i] for i in range(1, k))
        return s / (2 * k * be[0])

    A, B, bet = Rec(fA), Rec(fB), Rec(fbe)
    betm = bet if met else Const([])
    XF1, XG1, Xc1 = Theta(A, nuA), Theta(B, nuB), Theta(betm, lam)
    X2F1 = Lin(Theta(XF1, nuA), 1, XF1, 1)
    X2G1 = Lin(Theta(XG1, nuB), 1, XG1, 1)
    rho1 = Lin(Mul(bg.f, B), -kap, Mul(bg.g, A), -kap)
    irho1 = Lin(Mul(bg.irho0sq, rho1), -1)
    h1 = Lin(Mul(bg.irho0, irho1), 2)
    ib1 = Lin(Mul(bg.ibsq, betm), -1)
    T1 = Lin(Mul(Xc1, bg.ib), 1, Mul(bg.Xc0, ib1), 1)
    st["tbb"] = Lin(Mul(bg.b, bet), 2)
    # bracket of the F equation, dual part
    kterm = Lin(Mul(Lin(Mul(B, bg.irho0), 1, Mul(bg.g, irho1), 1), bg.XF0sq), 1, Mul(bg.girho0, Mul(bg.XF0, XF1)), 2)
    st["br"] = Lin(Lin(X2F1, 1, Lin(Mul(T1, bg.XF0), 1, Mul(bg.T0, XF1), 1), -1), 1, kterm, 2 * kap)
    ktermb = Lin(Mul(Lin(Mul(A, bg.irho0), 1, Mul(bg.f, irho1), 1), bg.XG0sq), 1, Mul(bg.firho0, Mul(bg.XG0, XG1)), 2)
    st["brb"] = Lin(Lin(X2G1, 1, Lin(Mul(T1, bg.XG0), 1, Mul(bg.T0, XG1), 1), -1), 1, ktermb, 2 * kap)
    # c theta c = 2 z h XF XFbar, dual part: 2 [h1 XF0 XG0 + h0 (XF1 XG0 + XF0 XG1)]
    st["r2"] = Lin(Lin(Mul(h1, bg.XFG0), 1, Mul(bg.irho0sq, Lin(Mul(XF1, bg.XG0), 1, Mul(bg.XF0, XG1), 1)), 1), 2 * hs)
    A[J - 1]; B[J - 1]; bet[J - 1]
    return A.c[:J], B.c[:J], bet.c[:J]


def series_eval(cA, cB, cbe, xi, jt=None):
    """(A, A', B, B', beta) at complex xi from the optimally truncated series (truncation index from A and B)."""
    xi = mp.mpc(xi)
    z = xi * xi
    if jt is None:
        terms = [abs(cA[j] * z ** j) + abs(cB[j] * z ** j) for j in range(len(cA))]
        jt = min(range(5, len(cA)), key=lambda j: terms[j])
    A = mp.fsum(cA[j] * z ** j for j in range(jt))
    B = mp.fsum(cB[j] * z ** j for j in range(jt))
    Ap = mp.fsum(2 * j * cA[j] * z ** j for j in range(1, jt)) / xi
    Bp = mp.fsum(2 * j * cB[j] * z ** j for j in range(1, jt)) / xi
    be = mp.fsum(cbe[j] * z ** j for j in range(jt))
    return np.array([complex(v) for v in (A, Ap, B, Bp, be)]), jt


def prhs(xi, Y, kap, om, lam, npert=2, mutate=None):
    """d/dxi of (f, g, q, qg, b, [A, A', B, B', beta] x npert), the background as css_inner.rhs."""
    met = 0 if mutate == "nometric" else 1
    hs = -1 if mutate == "hamsign" else 1
    f, g, q, qg, b = Y[:5]
    P = Y[5:].reshape(npert, 5)
    A, Ap, B, Bp, be = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4]
    nu0, nug = -1j * om, 1j * om
    nuA, nuB = lam - 1j * om, lam + 1j * om
    rho = 1 - kap * f * g
    ir = 1 / rho
    h = ir * ir
    XF0 = xi * q + nu0 * f
    XG0 = xi * qg + nug * g
    bp = 2 * xi * h * XF0 * XG0 / b
    Xc0 = xi * bp
    T0 = Xc0 / b
    rest_f = (2 + 2 * nu0) * xi * q + (nu0 ** 2 + nu0) * f - T0 * XF0 + 2 * kap * g * ir * XF0 ** 2
    rest_g = (2 + 2 * nug) * xi * qg + (nug ** 2 + nug) * g - T0 * XG0 + 2 * kap * f * ir * XG0 ** 2
    fpp = (b * b * q / xi - rest_f) / (xi * xi)
    gpp = (b * b * qg / xi - rest_g) / (xi * xi)
    # perturbation
    rho1 = -kap * (f * B + g * A)
    ir1 = -rho1 * ir * ir
    h1 = 2 * ir * ir1
    XF1 = xi * Ap + nuA * A
    XG1 = xi * Bp + nuB * B
    bep = (hs * 2 * xi * (h1 * XF0 * XG0 + h * (XF1 * XG0 + XF0 * XG1)) - be * bp) / b
    bm = met * be
    Xc1 = xi * met * bep + lam * bm
    T1 = Xc1 / b - Xc0 * bm / (b * b)
    rest_A = (2 + 2 * nuA) * xi * Ap + (nuA ** 2 + nuA) * A - (T1 * XF0 + T0 * XF1) \
        + 2 * kap * ((B * ir + g * ir1) * XF0 ** 2 + 2 * g * ir * XF0 * XF1)
    rest_B = (2 + 2 * nuB) * xi * Bp + (nuB ** 2 + nuB) * B - (T1 * XG0 + T0 * XG1) \
        + 2 * kap * ((A * ir + f * ir1) * XG0 ** 2 + 2 * f * ir * XG0 * XG1)
    App = ((b * b * Ap + 2 * b * bm * q) / xi - rest_A) / (xi * xi)
    Bpp = ((b * b * Bp + 2 * b * bm * qg) / xi - rest_B) / (xi * xi)
    out = np.empty_like(Y)
    out[:5] = [q, qg, fpp, gpp, bp]
    Po = out[5:].reshape(npert, 5)
    Po[:, 0], Po[:, 1], Po[:, 2], Po[:, 3], Po[:, 4] = Ap, App, Bp, Bpp, bep
    return out


class Inner:
    """det M(lambda) for one inner background and one (r0, r1) setting."""

    def __init__(self, kap, f0, om, r0=0.1, r1=0.3, rtol=1e-12, atol=1e-15, mutate=None, extra=25):
        self.kap, self.f0, self.om, self.r0, self.r1 = kap, f0, om, r0, r1
        self.rtol, self.atol, self.mutate = rtol, atol, mutate
        self.J = int(1 / (2 * r0 ** 2)) + extra
        self.bg = Background(kap, f0, om, self.J + 40)
        self.bgv = {}
        for sgn in (1, -1):
            Z0, jt, err = ci.series_value(self.bg.coef, r0 * np.exp(sgn * 1j * PHI0))
            self.bgv[sgn] = Z0
        self.nev = 0

    def J_for(self, lam):
        return self.J + int(abs(lam)) + 10

    def lateral(self, lam, datas, sgn, series=None):
        """Upper (sgn = 1) or lower (sgn = -1) lateral solution at xi = r1: background and the perturbations."""
        r0, r1 = self.r0, self.r1
        u = np.exp(sgn * 1j * PHI0)
        x0 = r0 * u
        Zs = [self.bgv[sgn]]
        for k, (A0, B0) in enumerate(datas):
            cA, cB, cbe = series[k]
            v, jt = series_eval(cA, cB, cbe, x0)
            Zs.append(v)
        Z = np.concatenate(Zs)
        npert = len(datas)
        kap, om, mut = self.kap, self.om, self.mutate
        s = solve_ivp(lambda t, Y: prhs(t * u, Y, kap, om, lam, npert, mut) * u, (r0, r1), Z, method="DOP853",
                      rtol=self.rtol, atol=self.atol)
        Z = s.y[:, -1]
        s2 = solve_ivp(lambda t, Y: prhs(r1 * np.exp(1j * t), Y, kap, om, lam, npert, mut) * 1j * r1 * np.exp(1j * t),
                       (sgn * PHI0, 0.0), Z, method="DOP853", rtol=self.rtol, atol=self.atol)
        return s2.y[:, -1], (s.status, s2.status)

    def M(self, lam, datas=((1, 0), (0, 1))):
        lam = complex(lam)
        J = self.J_for(lam)
        ser = [pert_series(self.bg, lam, A0, B0, J, self.mutate) for (A0, B0) in datas]
        up, _ = self.lateral(lam, datas, 1, ser)
        lo, _ = self.lateral(lam, datas, -1, ser)
        self.nev += 1
        M = np.empty((2, len(datas)), complex)
        for k in range(len(datas)):
            o = 5 + 5 * k
            M[0, k] = up[o] - lo[o]
            M[1, k] = up[o + 2] - lo[o + 2]
        return M, up, lo

    def det(self, lam):
        M = self.M(lam)[0]
        return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def background(kap, path=os.path.join(HERE, "data", "css_inner_kpath.jsonl")):
    """(f0, omega) of the inner Stokes zero at kappa: the stored kpath point if present, else css_inner.solve from the
    interpolated stored points."""
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        if abs(r["kappa"] - kap) < 1e-12:
            return r["f0"], r["omega"]
    ks = np.array([r["kappa"] for r in rows])
    f0 = np.interp(kap, ks, [r["f0"] for r in rows])
    om = np.interp(kap, ks, [r["omega"] for r in rows])
    x, res, ok = ci.solve(kap, f0, om)
    if not ok or res > 1e-8:
        raise RuntimeError("inner background not converged at kappa %g (res %.1e)" % (kap, res))
    return x[0], x[1]


def exact_modes(bg):
    """The exact modes' series (lam, A0, B0, cA, cB, cbe) from the symmetries, to compare with pert_series."""
    a, ag, be = bg.coef
    I = mp.mpc(0, 1)
    om, kap = mp.mpf(bg.om), mp.mpf(bg.kap)
    N = len(a) - 2
    b1 = be[1]
    gg = [mp.fsum(ag[i] * ag[k - i] for i in range(k + 1)) for k in range(N)]
    ff = [mp.fsum(a[i] * a[k - i] for i in range(k + 1)) for k in range(N)]
    out = []
    out.append((0, [I * a[k] for k in range(N)], [-I * ag[k] for k in range(N)], [mp.mpc(0)] * N))
    out.append((1, [(2 * k - I * om) * a[k] for k in range(N)], [(2 * k + I * om) * ag[k] for k in range(N)],
                [2 * k * be[k] for k in range(N)]))
    out.append((2, [2 * (k + 1) * a[k + 1] - 2 * b1 * (2 * k - I * om) * a[k] for k in range(N)],
                [2 * (k + 1) * ag[k + 1] - 2 * b1 * (2 * k + I * om) * ag[k] for k in range(N)],
                [2 * (k + 1) * be[k + 1] - 2 * b1 * (2 * k + 1) * be[k] for k in range(N)]))
    out.append((I * om, [mp.mpc(1)] + [mp.mpc(0)] * (N - 1), [-kap * gg[k] for k in range(N)], [mp.mpc(0)] * N))
    out.append((-I * om, [-kap * ff[k] for k in range(N)], [mp.mpc(1)] + [mp.mpc(0)] * (N - 1), [mp.mpc(0)] * N))
    return out


def opts_of(argv):
    pos, o = [], {}
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            o[k] = v
        else:
            pos.append(a)
    return pos, o


def secant(F, lam0, tol=1e-11, maxit=40):
    l0 = complex(lam0)
    l1 = l0 * (1 + 1e-4) + 1e-5
    d0, d1 = F(l0), F(l1)
    for _ in range(maxit):
        l2 = l1 - d1 * (l1 - l0) / (d1 - d0)
        if abs(l2 - l1) < tol * max(1, abs(l2)):
            return l2, True
        l0, d0 = l1, d1
        l1, d1 = l2, F(l2)
    return l2, False


def main():
    pos, o = opts_of(sys.argv[1:])
    out = o.get("out")
    mutate = o.get("mutate") or None
    rtol = float(o.get("rtol", "1e-12"))
    if "dps" in o:
        mp.mp.dps = int(o["dps"])

    def emit(rec):
        print(json.dumps(rec), flush=True)
        if out:
            with open(out, "a") as fo:
                fo.write(json.dumps(rec) + "\n")

    if not pos:
        raise SystemExit(__doc__)
    cmd = pos[0]
    if cmd == "det":
        kap = float(pos[1]); lam = complex(pos[2].replace(" ", ""))
        f0, om = background(kap)
        r0, r1 = float(o.get("r0", 0.1)), float(o.get("r1", 0.3))
        I = Inner(kap, f0, om, r0, r1, rtol=rtol, mutate=mutate)
        t = time.time()
        M, up, lo = I.M(lam)
        sv = np.linalg.svd(M, compute_uv=False)
        emit(dict(kind="det", kappa=kap, lam=[lam.real, lam.imag], r0=r0, r1=r1, det=[complex(np.linalg.det(M)).real,
                  complex(np.linalg.det(M)).imag], sv=sv.tolist(), seconds=round(time.time() - t, 2), mutate=mutate))
        return
    if cmd == "check":
        # 1. background part of prhs against css_inner.rhs; 2. the series recursion against the exact modes' series
        # (and the control: the same data at a wrong lambda must fail); 3. the exact modes as null vectors of M
        kap = float(pos[1]) if len(pos) > 1 else 0.0
        f0, om = background(kap)
        ok = True
        Z = np.array([0.4 + 0.1j, 0.3 - 0.2j, -0.2 + 0.3j, 0.1 + 0.05j, 1.3 + 0.1j]); xi = 0.37 + 0.05j
        d = np.abs(prhs(xi, np.concatenate([Z, np.zeros(10, complex)]), kap, om, 1.7)[:5] - ci.rhs(xi, Z, kap, om)).max()
        ok &= d < 1e-12
        emit(dict(kind="check", what="background rhs vs css_inner.rhs", kappa=kap, maxdiff=d))
        bg = Background(kap, f0, om, 100)
        N = 60

        def relerr(x, y):
            sc = max(abs(v) for v in y[:N]) if any(abs(v) > 0 for v in y[:N]) else 1
            return max(float(abs(x[k] - y[k]) / max(abs(y[k]), 1e-30 * sc)) if abs(y[k]) > 0 else float(abs(x[k]) / sc)
                       for k in range(N))
        for lam, cA, cB, cbe in exact_modes(bg):
            A_, B_, be_ = pert_series(bg, lam, cA[0], cB[0], N)
            e = max(relerr(A_, cA), relerr(B_, cB))
            eb = max(float(abs(be_[k] - cbe[k]) / max(abs(v) for v in bg.coef[2][:k + 2])) for k in range(N))
            A2, B2, _ = pert_series(bg, lam + mp.mpf("0.01"), cA[0], cB[0], N)   # control: wrong lambda
            ectl = max(relerr(A2, cA), relerr(B2, cB))
            good = e < 1e-20 and eb < 1e-20 and ectl > 1e-6
            ok &= good
            emit(dict(kind="check", what="series of the exact mode", lam=[float(mp.re(lam)), float(mp.im(lam))],
                      relerr_AB=e, err_beta_scaled=eb, control_lam_plus_0p01_relerr=ectl, ok=bool(good)))
        I = Inner(kap, f0, om, 0.1, 0.3)
        for lam in (0, 1, 2, 1.5):
            M = I.M(lam)[0]
            sv = np.linalg.svd(M, compute_uv=False)
            r = sv[-1] / sv[0]
            good = (r < 1e-10) if lam != 1.5 else (r > 1e-3)
            ok &= good
            emit(dict(kind="check", what="smallest/largest singular value of M (1.5: control, must be large)",
                      lam=lam, ratio=r, ok=bool(good)))
        print("check", "PASS" if ok else "FAIL", flush=True)
        sys.exit(0 if ok else 1)
    sets = o.get("set", "A,B,C").split(",")
    plant = complex(o["plant"].replace(" ", "")) if "plant" in o else None
    if cmd == "root":
        kap = float(pos[1]); lam0 = complex(pos[2].replace(" ", ""))
        f0, om = background(kap)
        for s in sets:
            r0, r1 = SETTINGS[s]
            I = Inner(kap, f0, om, r0, r1, rtol=rtol, mutate=mutate)
            t = time.time()
            lam, ok = secant(I.det, lam0)
            M = I.M(lam)[0]
            sv = np.linalg.svd(M, compute_uv=False)
            emit(dict(kind="root", kappa=kap, f0=f0, omega=om, setting=s, r0=r0, r1=r1, start=[lam0.real, lam0.imag],
                      lam=[lam.real, lam.imag], converged=ok, im_over_omega=lam.imag / om, sv=sv.tolist(),
                      evaluations=I.nev, seconds=round(time.time() - t, 1), mutate=mutate, rtol=rtol, dps=mp.mp.dps))
        return
    if cmd == "count":
        sys.path.insert(0, HERE)
        import complex_css as cc
        kap = float(pos[1]); re0, re1, im0, im1 = [float(v) for v in pos[2:6]]
        f0, om = background(kap)
        for s in sets:
            r0, r1 = SETTINGS[s]
            I = Inner(kap, f0, om, r0, r1, rtol=rtol, mutate=mutate)
            cache = {}

            def F(lam):
                key = (round(lam.real, 13), round(lam.imag, 13))
                if key not in cache:
                    d = I.det(lam)
                    cache[key] = d * (lam - plant) if plant is not None else d
                return cache[key]
            t = time.time()
            w, big, ni = cc.winding(cc.detrended(F, (re0, re1, im0, im1)), cc.box(re0, re1, im0, im1), K=int(o.get("K", 6)))
            emit(dict(kind="count", kappa=kap, setting=s, r0=r0, r1=r1, box=[re0, re1, im0, im1], winding=w,
                      largest_step=big, intervals=ni, evaluations=I.nev, seconds=round(time.time() - t, 1),
                      plant=None if plant is None else [plant.real, plant.imag], mutate=mutate, omega=om))
        return
    if cmd == "kpath":
        # follow a root in kappa: kpath K0 K1 STEP LAM ; at each kappa the inner background is solved (css_inner.solve
        # from the stored kpath points), the root found by secant at each setting from the extrapolated guess
        k0, k1, st = float(pos[1]), float(pos[2]), float(pos[3])
        lam = complex(pos[4].replace(" ", ""))
        hist = []
        ks = np.arange(k0, k1 + 0.5 * st * np.sign(st), st) if st else [k0]
        for kap in ks:
            kap = float(round(kap, 12))
            f0, om = background(kap)
            if len(hist) >= 2:
                (ka, la), (kb, lb) = hist[-2], hist[-1]
                g = lb + (lb - la) * (kap - kb) / (kb - ka)
            elif hist:
                g = hist[-1][1]
            else:
                g = lam
            res = []
            for s in sets:
                r0, r1 = SETTINGS[s]
                I = Inner(kap, f0, om, r0, r1, rtol=rtol, mutate=mutate)
                l, ok = secant(I.det, g)
                res.append((s, l, ok))
            l = res[0][1]
            hist.append((kap, l))
            emit(dict(kind="kpath", kappa=kap, f0=f0, omega=om, guess=[g.real, g.imag], lam=[l.real, l.imag],
                      settings=[[s_, x.real, x.imag, ok_] for s_, x, ok_ in res],
                      spread=max(abs(x - l) for _, x, _ in res), im_over_omega=l.imag / om, mutate=mutate))
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
