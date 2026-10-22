#!/usr/bin/env python3
"""The D = infinity CSS problem of the hedgehog at fixed eta (inner variables): its solutions and their growth rates.

Inner problem (sigma_eqs.py section 6; matter/STRUCTURE.md section 6), s = r^2/(2n), physical time t:
    d_t(u_t/c) = c (u_s - sin(2u)/(4s)),   c_s = 2 eta u_t^2 / c .
In tau = -ln(-t), xi = rho/(-t), rho = sqrt(2s) = r/sqrt(n) (so xi = x/sqrt(n)), with Q = u_tau + xi u_xi:
    (1 + d_tau + xi d_xi)(Q/c) = c [u_xi/xi - sin(2u)/(2 xi^2)],     c_xi = 2 eta xi Q^2 / c .
CSS, u = W(xi), c = C(xi) (checked with sympy in `check`):
    xi^4 W'' + 2 xi^3 W' - xi^4 W' C'/C - C^2 xi W' + C^2 sin(2W)/2 = 0,    C C' = 2 eta xi^3 W'^2,    C(0) = 1.
xi = 0 is an irregular singular point: the slow (formal) solutions W = A xi + O(xi^3) form a one-parameter family
(A = sqrt(n) U'(0) of the finite-n solution), and the fast solution carries e^{-1/(2 xi^2)} = e^{-n/(2x^2)}.
At finite n the CSS solution must be analytic at the past light cone x = 1 (xi = 1/sqrt(n) -> 0).  As for the scalar
(README, "The inner problem at large x", Conjecture 1), the D = infinity form of that condition is TAKEN to be the
vanishing of the Stokes multiplier of the formal series; the finite-D numbers in SIGMA.md test it.

Lateral method: in pi/4 < arg xi < 3pi/4 the fast mode dominates as xi -> 0, so the solution asymptotic to the series
there is unique.  It is started from the (optimally truncated) series at xi0 = r0 e^{i theta0}, carried on an arc of
radius r0 down to the real axis, and integrated along the real axis to xi = R.  For real data W- = conj W+, so the
Stokes jump is 2i Im W+: the CSS solutions are the zeros of Im W+(R) in A.

Modes: u = W + e^{lam tau} v(xi), c = C (1 + e^{lam tau} g(xi)), g(0) = 0 (c normalised at the light cone; a constant g
is the time reparametrisation).  Linearised:
    xi^2 [(1+lam) dq + xi dq'] - C [xi v' - cos(2W) v] - C g [xi W' - sin(2W)/2] = 0,   dq = (lam v + xi v' - xi W' g)/C
    C' g + C g' = 2 eta xi [2 xi W' (lam v + xi v') - xi^2 W'^2 g] / C
with the slow solution v = xi + O(xi^3).  lam is a growth rate iff the perturbation's jump vanishes, Im v+(R) = 0
(same lateral path, background W+ on it).  lam = 1 (time translation, v = xi W') is exact.

    python3 sigma_inner.py check
    python3 sigma_inner.py scanA ETA AMIN AMAX N
    python3 sigma_inner.py zeros ETA AMIN AMAX N      # refine every sign change of Im W+ (zeros = CSS solutions)
    python3 sigma_inner.py modes ETA A0 LMIN LMAX N   # solution near A0, then Im v+(R) against lambda, roots
    python3 sigma_inner.py track ETA0 ETA1 NETA A0 LAM0   # follow a solution and its growing mode in eta
"""
import os
import sys
import json
import math
import cmath
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from sigma_css import TPS, sincos

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- residuals (TPS or scalars)
def inner_res(xi, W, W1, W2, C, C1, eta):
    s, co = sincos(W)
    R1 = xi ** 4 * W2 + 2 * xi ** 3 * W1 - xi ** 4 * W1 * C1 / C - C * C * xi * W1 + C * C * s * co
    R2 = C * C1 - 2 * eta * xi ** 3 * W1 * W1
    return R1, R2


def lin_res(xi, W, W1, W2, C, C1, v, v1, v2, g, g1, lam, eta):
    s, co = sincos(W)
    iC = 1 / C
    dq = (lam * v + xi * v1 - xi * W1 * g) * iC
    dq1 = ((lam + 1) * v1 + xi * v2 - W1 * g - xi * W2 * g - xi * W1 * g1) * iC \
        - (lam * v + xi * v1 - xi * W1 * g) * C1 * iC * iC
    RW = xi * xi * ((1 + lam) * dq + xi * dq1) - C * (xi * v1 - (co * co - s * s) * v) - C * g * (xi * W1 - s * co)
    RC = C1 * g + C * g1 - 2 * eta * xi * (2 * xi * W1 * (lam * v + xi * v1) - xi * xi * W1 * W1 * g) * iC
    return RW, RC


# ----------------------------------------------------------------------------- formal series at xi = 0
def _solve_order(res, arrs, k, unknowns):
    r0 = res(k)
    h = 1.0 + np.max(np.abs(r0))
    M = np.zeros((len(r0), len(unknowns)), dtype=complex)
    for j, (a, idx) in enumerate(unknowns):
        arrs[a][idx] = h
        M[:, j] = (res(k) - r0) / h
        arrs[a][idx] = 0.0
    sol = np.linalg.lstsq(M, -r0, rcond=None)[0]
    for j, (a, idx) in enumerate(unknowns):
        arrs[a][idx] = sol[j]


def formal_series(A, eta, K=40):
    """W (odd) and C (even) in xi, regular at xi = 0, W = A xi + ..., C(0) = 1."""
    w = np.zeros(K + 3, complex)
    c = np.zeros(K + 3, complex)
    c[0] = 1.0
    w[1] = A
    x = TPS(np.r_[0.0, 1.0, np.zeros(K + 1)].astype(complex))

    def res(k):
        Wt, Ct = TPS(w), TPS(c)
        W1 = Wt.deriv()
        R = inner_res(x, Wt, W1, W1.deriv(), Ct, Ct.deriv(), eta)
        return np.array([R[0].c[k], R[1].c[k]])
    arrs = {'w': w, 'c': c}
    for k in range(K + 1):
        unk = [('c', k + 1)] + ([] if k == 1 else [('w', k)])
        _solve_order(res, arrs, k, unk)
    return TPS(w[:K + 1].real), TPS(c[:K + 1].real)


def pert_series(Ws, Cs, lam, eta, K=40):
    """Slow perturbation v = xi + O(xi^3) (odd), g even with g(0) = 0."""
    v = np.zeros(K + 3, complex)
    g = np.zeros(K + 3, complex)
    v[1] = 1.0
    x = TPS(np.r_[0.0, 1.0, np.zeros(K + 1)].astype(complex))
    W = TPS(Ws.c[:K + 3].astype(complex) if len(Ws.c) >= K + 3 else np.r_[Ws.c, np.zeros(K + 3 - len(Ws.c))].astype(complex))
    C = TPS(Cs.c[:K + 3].astype(complex) if len(Cs.c) >= K + 3 else np.r_[Cs.c, np.zeros(K + 3 - len(Cs.c))].astype(complex))
    W1 = W.deriv(); W2 = W1.deriv(); C1 = C.deriv()

    def res(k):
        V, G = TPS(v), TPS(g)
        V1 = V.deriv()
        R = lin_res(x, W, W1, W2, C, C1, V, V1, V1.deriv(), G, G.deriv(), lam, eta)
        return np.array([R[0].c[k], R[1].c[k]])
    arrs = {'v': v, 'g': g}
    for k in range(K + 1):
        unk = [('g', k + 1)] + ([] if k == 1 else [('v', k)])
        _solve_order(res, arrs, k, unk)
    return TPS(v[:K + 1]), TPS(g[:K + 1])


def optimal_eval(S, xi):
    """Evaluate a divergent series at xi, truncated before its smallest (nonzero) term."""
    terms = S.c * xi ** np.arange(len(S.c))
    nz = np.nonzero(S.c)[0]
    if len(nz) < 4:
        return terms.sum(), 0.0
    mags = np.abs(terms)
    km = nz[2 + np.argmin(mags[nz[2:]])]
    return terms[:km].sum(), mags[km]


# ----------------------------------------------------------------------------- ODEs along a path xi(p)
def rhs_bg(xi, W, W1, C, eta):
    s2 = cmath.sin(2 * W) / 2
    C1 = 2 * eta * xi ** 3 * W1 * W1 / C
    W2 = -(2 * xi ** 3 * W1 - xi ** 4 * W1 * C1 / C - C * C * xi * W1 + C * C * s2) / xi ** 4
    return W2, C1


def rhs_full(xi, y, eta, lam, with_pert):
    W, W1, C = y[0], y[1], y[2]
    W2, C1 = rhs_bg(xi, W, W1, C, eta)
    if not with_pert:
        return [W1, W2, C1]
    v, v1, g = y[3], y[4], y[5]
    g1 = (2 * eta * xi * (2 * xi * W1 * (lam * v + xi * v1) - xi * xi * W1 * W1 * g) / C - C1 * g) / C
    RW0, _ = lin_res(xi, W, W1, W2, C, C1, v, v1, 0.0, g, g1, lam, eta)
    v2 = -RW0 * C / xi ** 4
    return [W1, W2, C1, v1, v2, g1]


class Inner:
    def __init__(self, eta, r0=None, theta0=3 * math.pi / 8, R=0.4, K=40, rtol=1e-11):
        self.eta, self.theta0, self.R, self.K, self.rtol = float(eta), theta0, R, K, rtol
        self.r0 = r0

    def _r0(self, A):
        return self.r0 if self.r0 is not None else min(0.15, 0.35 / max(abs(A), 1e-9))

    def lateral(self, A, lam=None, R=None, dense=False):
        """Upper-sector solution (and perturbation if lam is given) at xi = R on the real axis."""
        R = R or self.R
        eta = self.eta
        r0 = self._r0(A)
        Ws, Cs = formal_series(A, eta, self.K)
        z0 = r0 * cmath.exp(1j * self.theta0)
        y0 = [optimal_eval(Ws, z0)[0], optimal_eval(Ws.deriv(), z0)[0], optimal_eval(Cs, z0)[0]]
        self.trunc = optimal_eval(Ws, z0)[1]
        wp = lam is not None
        if wp:
            Vs, Gs = pert_series(Ws, Cs, lam, eta, self.K)
            y0 += [optimal_eval(Vs, z0)[0], optimal_eval(Vs.deriv(), z0)[0], optimal_eval(Gs, z0)[0]]
        y0 = np.array(y0, complex)

        def f_arc(th, y):
            xi = r0 * cmath.exp(1j * th)
            return np.array(rhs_full(xi, y, eta, lam, wp)) * 1j * xi
        s1 = solve_ivp(f_arc, (self.theta0, 0.0), y0, method='DOP853', rtol=self.rtol, atol=1e-14)

        def f_real(x, y):
            return np.array(rhs_full(complex(x), y, eta, lam, wp))

        def blow(x, y):                      # stop on a singularity of the real-axis solution
            return 1e6 - abs(y[1])
        blow.terminal = True
        s2 = solve_ivp(f_real, (r0, R), s1.y[:, -1], method='DOP853', rtol=self.rtol, atol=1e-14, dense_output=dense,
                       events=blow)
        self.status = (s1.status, s2.status)
        if s2.status != 0:
            nanv = np.full(len(y0), np.nan + 0j)
            return s2 if dense else nanv
        if dense:
            return s2
        return s2.y[:, -1]

    def jump(self, A, R=None):
        y = self.lateral(A, R=R)
        return y[0].imag

    def _roots(self, fun, a0, a1, N, R2fac=0.7, tol=1e-7):
        """Sign changes of fun(p, R) on a grid, refined; kept only if also a root at R2 = R2fac R (a genuine Stokes
        zero does not depend on where the jump is read; a zero of the fast solution at R does)."""
        ps = np.linspace(a0, a1, N) + 1.3e-4 * (a1 - a0) / N
        js = []
        for p in ps:
            js.append(fun(p, self.R))
            if getattr(self, 'verbose', False):
                print("   %.6f  %+.5e" % (p, js[-1]), flush=True)
        out = []
        for i in range(N - 1):
            if not (np.isfinite(js[i]) and np.isfinite(js[i + 1])):
                continue
            if np.sign(js[i]) != np.sign(js[i + 1]):
                p = brentq(lambda q: fun(q, self.R), ps[i], ps[i + 1], xtol=1e-13, rtol=1e-13)
                try:
                    lo, hi = p - 0.02 * (a1 - a0) / N, p + 0.02 * (a1 - a0) / N
                    p2 = brentq(lambda q: fun(q, self.R * R2fac), lo, hi, xtol=1e-13, rtol=1e-13)
                except ValueError:
                    continue
                if abs(p2 - p) < tol * (1 + abs(p)):
                    out.append(p)
        return out, ps, js

    def zeros(self, a0, a1, N):
        return self._roots(lambda A, R: self.jump(A, R), a0, a1, N)

    def pjump(self, A, lam, R=None):
        y = self.lateral(A, lam=lam, R=R)
        return y[3].imag / abs(y[3])

    def modes(self, A, l0, l1, N):
        return self._roots(lambda l, R: self.pjump(A, l, R), l0, l1, N)


def check():
    import sympy as sp
    rho, t, eta = sp.symbols('rho t eta', positive=True)
    s = rho ** 2 / 2
    xi = sp.Symbol('xi', positive=True)
    Wf, Cf = sp.Function('W'), sp.Function('C')
    X = rho / (-t)
    u, c = Wf(X), Cf(X)
    ds = lambda F: sp.diff(F, rho) / rho
    E1 = sp.diff(sp.diff(u, t) / c, t) - c * (ds(u) - sp.sin(2 * u) / (4 * s))
    E2 = ds(c) - 2 * eta * sp.diff(u, t) ** 2 / c
    Wx, Cx = sp.Function('Wx')(xi), sp.Function('Cx')(xi)
    sub = {Wf(xi): Wx, Cf(xi): Cx}
    E1x = sp.simplify(sp.simplify(E1.subs(rho, -xi * t)).doit().subs(sub).doit())
    E2x = sp.simplify(sp.simplify(E2.subs(rho, -xi * t)).doit().subs(sub).doit())
    R1, R2 = inner_res(xi, Wx, sp.diff(Wx, xi), sp.diff(Wx, xi, 2), Cx, sp.diff(Cx, xi), eta) \
        if False else (xi ** 4 * sp.diff(Wx, xi, 2) + 2 * xi ** 3 * sp.diff(Wx, xi)
                       - xi ** 4 * sp.diff(Wx, xi) * sp.diff(Cx, xi) / Cx - Cx ** 2 * xi * sp.diff(Wx, xi)
                       + Cx ** 2 * sp.sin(2 * Wx) / 2,
                       Cx * sp.diff(Cx, xi) - 2 * eta * xi ** 3 * sp.diff(Wx, xi) ** 2)
    print("[%s] CSS reduction of the inner problem (wave)" % ("ok" if sp.simplify(sp.expand(E1x * Cx * xi ** 2 * t ** 2 - R1)) == 0 else "FAIL"))
    print("[%s] CSS reduction of the inner problem (c)" % ("ok" if sp.simplify(sp.expand(E2x * Cx * xi * t ** 2 - R2)) == 0 else "FAIL"))
    # linearisation: u = W + e^{lam tau} v, c = C(1 + e^{lam tau} g), tau = -ln(-t)
    lam, e = sp.symbols('lam epsilon')
    vf, gf = sp.Function('v'), sp.Function('g')
    up = Wf(X) + e * (-t) ** (-lam) * vf(X)
    cp = Cf(X) * (1 + e * (-t) ** (-lam) * gf(X))
    E1p = sp.diff(sp.diff(up, t) / cp, t) - cp * (ds(up) - sp.sin(2 * up) / (4 * s))
    E2p = ds(cp) - 2 * eta * sp.diff(up, t) ** 2 / cp
    L1 = sp.diff(E1p, e).subs(e, 0)
    L2 = sp.diff(E2p, e).subs(e, 0)
    vx, gx = sp.Function('vx')(xi), sp.Function('gx')(xi)
    sub2 = {Wf(xi): Wx, Cf(xi): Cx, vf(xi): vx, gf(xi): gx}
    L1x = L1.subs(rho, -xi * t).doit().subs(sub2).doit()
    L2x = L2.subs(rho, -xi * t).doit().subs(sub2).doit()
    RW, RC = lin_res(xi, Wx, sp.diff(Wx, xi), sp.diff(Wx, xi, 2), Cx, sp.diff(Cx, xi), vx, sp.diff(vx, xi),
                     sp.diff(vx, xi, 2), gx, sp.diff(gx, xi), lam, eta)
    # compare at random rational values of the jet (exact arithmetic): the ratios must be the same at every sample
    import random
    random.seed(1)
    ratios = []
    for trial in range(3):
        val = {}
        for F in (Wx, Cx, vx, gx):
            for k in (2, 1, 0):
                val[sp.diff(F, xi, k) if k else F] = sp.Rational(random.randint(1, 9), random.randint(2, 7))
        val.update({xi: sp.Rational(3, 5 + trial), t: -sp.Rational(7, 3), eta: sp.Rational(2, 7), lam: sp.Rational(5, 3)})
        ev = lambda E: sp.nsimplify(sp.N(E.subs(val), 40), rational=False)
        r1 = sp.N((L1x * (-t) ** lam * t ** 2 * xi ** 2).subs(val) / RW.subs(val), 30)
        r2 = sp.N((L2x * (-t) ** lam * t ** 2 * xi).subs(val) / RC.subs(val), 30)
        ratios.append((r1, r2))
    good1 = all(abs(r[0] - ratios[0][0]) < 1e-25 for r in ratios)
    good2 = all(abs(r[1] - ratios[0][1]) < 1e-25 for r in ratios)
    print("[%s] linearised wave equation (constant ratio %s at 3 random jets)" % ("ok" if good1 else "FAIL", ratios[0][0]))
    print("[%s] linearised c equation (constant ratio %s at 3 random jets)" % ("ok" if good2 else "FAIL", ratios[0][1]))
    # near the equator: W = pi/2 - e w, C = 1 + O(e^2); linear order, and w = 1/xi exact
    e = sp.Symbol('e')
    wx = sp.Function('w')(xi)
    Wn = sp.pi / 2 - e * wx
    R1n = xi ** 4 * sp.diff(Wn, xi, 2) + 2 * xi ** 3 * sp.diff(Wn, xi) - xi * sp.diff(Wn, xi) + sp.sin(2 * Wn) / 2
    lin = sp.simplify(sp.diff(R1n, e).subs(e, 0))
    target = -(xi ** 4 * sp.diff(wx, xi, 2) + 2 * xi ** 3 * sp.diff(wx, xi) - xi * sp.diff(wx, xi) - wx)
    print("[%s] about the equator: xi^4 w'' + 2 xi^3 w' - xi w' - w = 0 (gravity enters at O(w^2))"
          % ("ok" if sp.simplify(lin - target) == 0 else "FAIL"))
    print("[%s] w = 1/xi solves it exactly (no Stokes jump)"
          % ("ok" if sp.simplify(target.subs(wx, 1 / xi).doit()) == 0 else "FAIL"))
    ss, tt = sp.symbols('s t', positive=True)
    g = sp.Function('g')(ss, tt)
    w2 = g / sp.sqrt(ss)
    heat = sp.simplify((sp.diff(w2, tt, 2) - sp.diff(w2, ss) - w2 / (2 * ss)) * sp.sqrt(ss) - (sp.diff(g, tt, 2) - sp.diff(g, ss)))
    print("[%s] w_tt = w_s + w/(2s) with w = s^(-1/2) g  <=>  g_tt = g_s (free heat equation)" % ("ok" if heat == 0 else "FAIL"))
    Ws, Cs = formal_series(2.0, 0.0, 20)
    ex = [0, 2, 0, -2 / 3, 0, 2 / 5, 0, -2 / 7]
    print("[%s] flat A = 2 series = 2 atan(xi)  (max dev %.1e)" % ("ok" if np.max(np.abs(Ws.c[:8] - ex)) < 1e-12 else "FAIL",
                                                                   np.max(np.abs(Ws.c[:8] - ex))))
    I = Inner(0.0)
    for A in (1.9, 2.0, 2.1):
        y = I.lateral(A)
        print("     eta = 0, A = %.2f:  W+(R=%.2f) = %.12f %+.3e i   (2 atan R = %.12f)" % (A, I.R, y[0].real, y[0].imag,
                                                                                    2 * math.atan(I.R)))
    # time translation lam = 1 at the flat ground state
    print("     perturbation jump at A = 2, eta = 0: lam = 0.9, 1, 1.1:",
          ["%+.2e" % I.pjump(2.0, l) for l in (0.9, 1.0, 1.1)])


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'check'
    if what == 'check':
        check()
    elif what == 'scanA':
        eta, a0, a1, N = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
        I = Inner(eta)
        for A in np.linspace(a0, a1, N):
            y = I.lateral(A)
            print("%.6f  Im W+ %+.6e  Re W+ %+.6e  trunc %.1e %s" % (A, y[0].imag, y[0].real, I.trunc, I.status), flush=True)
    elif what == 'zeros':
        eta, a0, a1, N = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
        I = Inner(eta)
        I.verbose = True
        z, As, js = I.zeros(a0, a1, N)
        print(json.dumps(dict(eta=eta, zeros=z)), flush=True)
    elif what == 'modes':
        eta, A0, l0, l1, N = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
        I = Inner(eta)
        I.verbose = True
        A = brentq(I.jump, A0 - 0.05, A0 + 0.05, xtol=1e-13)
        print("A =", A, flush=True)
        m, ls, js = I.modes(A, l0, l1, N)
        print(json.dumps(dict(eta=eta, A=A, modes=m)), flush=True)
