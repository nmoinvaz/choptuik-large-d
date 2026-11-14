#!/usr/bin/env python3
"""Growth rates of the D = infinity CSS solution (kappa = 0) as zeros of the perturbation's Stokes jump (#771).

A second implementation, written without reading css_inner_pert.py or css_inner_pert_eqs.py. Everything below is
derived here from the inner PDE of matter/CSS.md section 8 (kappa = 0, the free complex scalar):

    c F_s = d_t(F_t / c),    c c_s = 2 F_t Fbar_t .

Background: F = T^{i w} f(xi), Fbar = T^{-i w} ft(xi), c = b(xi), with T = -t, xi = sqrt(2 s) / T (CSS.md's ansatz).
Perturbation: F += eps T^{i w - lam} u(xi), Fbar += eps T^{-i w - lam} v(xi), c += eps T^{-lam} w(xi): growth e^{lam tau},
tau = -ln(-t). F and Fbar are independent functions, so complex lam is allowed; on the real axis ft = conj(f).
sympy derives the ODEs (equations()); the exact solutions of the linear system are checked symbolically in `derive`:
s-translation (lam = 2), t-translation (lam = 1), the phase (lam = 0), the shift F -> F + const (lam = i w) and the time
reparametrizations t -> T(t), c -> c / T' at every lam, (u, v, w) = (g, gt, xi b' + (lam - 1) b). The last are gauge:
the condition w(0) = 0 (c = 1 at s = 0, the background's normalization) removes them except at lam = 1.

xi = 0 is an irregular singular point with the fast mode exp(-b^2 / (2 xi^2)) (the heat kernel). The selection rule asks
that the Stokes jump of the regular formal solution vanish (STRUCTURE.md section 7). Here:
  - formal power series at xi = 0, solved order by order from the polynomial forms of the ODEs (series_solve);
  - lateral solutions: the optimally truncated series at xi0 = r0 e^{+-i alpha}, integrated (DOP853) along the ray to
    r1 e^{+-i alpha} and the arc to r1; the jump is upper minus lower at xi = r1;
  - background: Newton on (|f(0)|, w), with f(0) real and b(0) = 1, for a zero jump of f;
  - perturbation: data (u(0), v(0)) = (1, 0) and (0, 1), w(0) = 0; the growth rates are the zeros of the 2 x 2
    determinant of the jumps of (u, v), D(lam), normalized by exp(-1/r1^2).
Planted check (`planted`): y'' of 2F0(a, beta;; 2 xi^2), whose ODE has the same singular point and fast mode; the
jump is compared with its exact value (2 i sin(pi beta) / Gamma(a)) x^-beta int_{1/x}^inf e^-t t^{a-1} (t - 1/x)^-beta dt,
x = 2 r1^2, and its zeros in a must be a = 0, -1, -2, ...

    python3 css_inf_jump.py derive                     the ODEs from the PDE and the exact modes (sympy)
    python3 css_inf_jump.py planted [--mutate=coef]    the 2F0 test (jump against the exact value, zeros in a);
                                                       the control (one coefficient off by 0.1%) must fail
    python3 css_inf_jump.py bg [--r1=0.3]              the background selection, from CSS.md's (|f(0)|, w)
    python3 css_inf_jump.py det LAM [LAM ...]          D(lam)
    python3 css_inf_jump.py root LAM0 [--r1=...]       a zero of D by the secant method from LAM0
Options: --r0=0.1 --r1=0.3 --alpha=0.375 (in units of pi) --rtol=1e-13 --N=140 (series orders) --out=FILE
"""
import functools
import json
import math
import sys
import time

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

F0_CSS, OM_CSS = 0.5019081944, 3.9789695074            # CSS.md section 8, kappa = 0, ten digits


# ------------------------------------------------------------------ equations
@functools.lru_cache(None)
def equations():
    T, xi, eps = sp.symbols("T xi epsilon", positive=True)
    om, lam = sp.symbols("omega lambda")
    f, ft, b, u, v, w = [sp.Function(n)(xi) for n in ("f", "ft", "b", "u", "v", "w")]

    def ds(H):                                  # d/ds at fixed t:  dxi/ds = 1 / (xi T^2)
        return sp.diff(H, xi) / (xi * T ** 2)

    def dt(H):                                  # d/dt at fixed s = -(d/dT|xi - (xi/T) d/dxi)
        return -(sp.diff(H, T) - xi / T * sp.diff(H, xi))
    F = T ** (sp.I * om) * f + eps * T ** (sp.I * om - lam) * u
    G = T ** (-sp.I * om) * ft + eps * T ** (-sp.I * om - lam) * v
    c = b + eps * T ** (-lam) * w
    res = [(c * ds(F) - dt(dt(F) / c), sp.I * om - 2), (c * ds(G) - dt(dt(G) / c), -sp.I * om - 2),
           (c * ds(c) - 2 * dt(F) * dt(G), -2)]
    jet = {}
    syms = {}
    for name, fn, order in (("F", f, 2), ("G", ft, 2), ("B", b, 1), ("U", u, 2), ("V", v, 2), ("W", w, 1)):
        for k in range(order, -1, -1):
            s = sp.Symbol("%s%d" % (name, k))
            syms[s.name] = s
            jet[fn.diff(xi, k) if k else fn] = s
    bg, lin = [], []
    for R, pw in res:
        r0 = sp.expand(R.subs(eps, 0) / T ** pw)
        r1 = sp.expand(sp.diff(R, eps).subs(eps, 0) / T ** (pw - lam))
        assert not r0.has(T) and not r1.has(T)
        bg.append(sp.together(r0.subs(jet)))
        lin.append(sp.together(r1.subs(jet)))
    S = syms

    def numer(e):
        n, d = sp.fraction(e)
        assert d.free_symbols <= {xi, S["B0"]}, d      # denominators are powers of xi and b only
        return sp.expand(n)
    bg_num = [numer(e) for e in bg]
    lin_num = [numer(e) for e in lin]
    # solved forms for the integration
    bsol = sp.solve(bg_num, [S["F2"], S["G2"], S["B1"]], dict=True)[0]
    psol = sp.solve(lin_num, [S["U2"], S["V2"], S["W1"]], dict=True)[0]
    jets_bg = [xi, S["F0"], S["F1"], S["G0"], S["G1"], S["B0"], om]
    bg_rhs = sp.lambdify(jets_bg, [bsol[S["F2"]], bsol[S["G2"]], bsol[S["B1"]]], "numpy")
    X = [S["U0"], S["U1"], S["V0"], S["V1"], S["W0"]]
    rows = [S["U1"], psol[S["U2"]], S["V1"], psol[S["V2"]], psol[S["W1"]]]
    A = [[sp.diff(r, x) for x in X] for r in rows]
    for r in rows:                                      # linear and homogeneous in the perturbation
        assert sp.simplify(r.subs({x: 0 for x in X})) == 0
    jets_p = [xi, S["F0"], S["F1"], S["F2"], S["G0"], S["G1"], S["G2"], S["B0"], S["B1"], om, lam]
    A_fn = sp.lambdify(jets_p, A, "numpy")
    ser_args = [xi] + [S[n] for n in ("F0", "F1", "F2", "G0", "G1", "G2", "B0", "B1")]
    pargs = [S[n] for n in ("U0", "U1", "U2", "V0", "V1", "V2", "W0", "W1")]
    bg_ser = sp.lambdify(ser_args + [om], bg_num, "math")
    lin_ser = sp.lambdify(ser_args + pargs + [om, lam], lin_num, "math")
    return dict(bg_rhs=bg_rhs, A=A_fn, bg_ser=bg_ser, lin_ser=lin_ser, bg_num=bg_num, lin_num=lin_num, S=S, xi=xi,
                om=om, lam=lam, sym=dict(f=f, ft=ft, b=b, u=u, v=v, w=w, xi=xi, om=om, lam=lam), lin=lin, bg=bg,
                bsol=bsol, psol=psol)


# ------------------------------------------------------------------ truncated power series
class Ser:
    """a truncated power series (complex coefficients); all operands have the same length"""
    __slots__ = ("c",)

    def __init__(self, c):
        self.c = np.asarray(c, dtype=complex)

    def __add__(self, o):
        if isinstance(o, Ser):
            return Ser(self.c + o.c)
        c = self.c.copy()
        c[0] += o
        return Ser(c)
    __radd__ = __add__

    def __neg__(self):
        return Ser(-self.c)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, Ser):
            return Ser(np.convolve(self.c, o.c)[:len(self.c)])
        return Ser(self.c * o)
    __rmul__ = __mul__

    def __pow__(self, n):
        assert isinstance(n, int) and n >= 0
        out = Ser(np.zeros_like(self.c))
        out.c[0] = 1
        for _ in range(n):
            out = out * self
        return out


def jets_of(c, L, nder):
    """Ser objects of length L for a function with coefficients c and its first nder derivatives"""
    out = []
    for d in range(nder + 1):
        k = np.arange(L)
        idx = k + d
        vals = np.zeros(L, complex)
        ok = idx < len(c)
        fac = np.ones(L)
        for j in range(d):
            fac = fac * (k + d - j)
        vals[ok] = c[idx[ok]] * fac[ok]
        out.append(Ser(vals))
    return out


def series_solve(resid_at, unknowns0, N):
    """order-by-order solution: resid_at(k, x) returns the order-k coefficients of the residual numerators with the
    order-(k+1) unknowns set to x (a vector); they are affine in x, so four evaluations give them"""
    coeffs = [np.zeros(N + 3, complex) for _ in unknowns0]
    for i, v0 in enumerate(unknowns0):
        coeffs[i][0] = v0
    worst = 0.0
    n = len(unknowns0)
    for k in range(N):
        def r(x):
            for i in range(n):
                coeffs[i][k + 1] = x[i]
            return np.array(resid_at(k, coeffs))
        r0 = r(np.zeros(n))
        sc = max(1.0, float(np.max(np.abs(r0))))         # trial values at the residual's scale: no cancellation
        Jm = np.zeros((n, n), complex)
        for i in range(n):
            e = np.zeros(n)
            e[i] = sc
            Jm[:, i] = (r(e) - r0) / sc
        x = np.linalg.solve(Jm, -r0)
        rr = r(x)
        worst = max(worst, float(np.max(np.abs(rr)) / max(1.0, np.max(np.abs(r0)))))
    return coeffs, worst


def bg_series(f0, om, N):
    E = equations()

    def resid_at(k, cf):
        L = k + 1
        F = jets_of(cf[0], L, 2)
        G = jets_of(cf[1], L, 2)
        B = jets_of(cf[2], L, 1)
        x = Ser(np.r_[0, 1, np.zeros(L - 2)] if L > 1 else [0])
        vals = E["bg_ser"](x, F[0], F[1], F[2], G[0], G[1], G[2], B[0], B[1], om)
        return [vv.c[k] if isinstance(vv, Ser) else vv for vv in vals]
    return series_solve(resid_at, [f0, np.conj(f0), 1.0], N)


def pert_series(bgc, om, lam, u0, v0, N):
    E = equations()

    def resid_at(k, cf):
        L = k + 1
        F = jets_of(bgc[0], L, 2)
        G = jets_of(bgc[1], L, 2)
        B = jets_of(bgc[2], L, 1)
        U = jets_of(cf[0], L, 2)
        V = jets_of(cf[1], L, 2)
        W = jets_of(cf[2], L, 1)
        x = Ser(np.r_[0, 1, np.zeros(L - 2)] if L > 1 else [0])
        vals = E["lin_ser"](x, F[0], F[1], F[2], G[0], G[1], G[2], B[0], B[1], U[0], U[1], U[2], V[0], V[1], V[2],
                            W[0], W[1], om, lam)
        return [vv.c[k] if isinstance(vv, Ser) else vv for vv in vals]
    return series_solve(resid_at, [u0, v0, 0.0], N)


def opt_sum(c, z, nder=1):
    """optimally truncated sum of sum c_k z^k and of its first derivative: stop before the smallest term (within the
    parity class when the odd coefficients vanish); a series that terminates is summed whole"""
    k = np.arange(len(c))
    terms = c * z ** k
    mag = np.abs(terms)
    cand = np.arange(2, len(c), 2) if np.all(c[1::2] == 0) else np.arange(1, len(c))
    mc = mag[cand]
    if np.all(mc[-6:] == 0):                        # terminates: no truncation
        kmin = len(c)
        tmin = 0.0
    else:
        pos = np.nonzero(mc > 0)[0]
        j = pos[np.argmin(mc[pos])]
        kmin = int(cand[j])
        tmin = float(mc[j])
        if kmin >= len(c) - 6:
            if tmin < 1e-28 * max(float(np.max(mag[:5])), 1e-300):
                kmin = len(c)                       # every term past the first few is negligible
            else:
                raise RuntimeError("series too short for optimal truncation (smallest term at the end)")
    s = np.sum(terms[:kmin])
    d = np.sum((k * c * z ** np.maximum(k - 1, 0))[1:kmin])
    return s, d, int(kmin), tmin


# ------------------------------------------------------------------ lateral solutions
def path_integrate(rhs, y0, r0, r1, alpha, rtol):
    """integrate dy/dxi = rhs(xi, y) from xi0 = r0 e^{i alpha} along the ray to r1 e^{i alpha} and the arc to r1"""
    e = np.exp(1j * alpha)
    s1 = solve_ivp(lambda r, y: e * rhs(r * e, y), (r0, r1), y0, method="DOP853", rtol=rtol, atol=1e-30 + rtol * 1e-6)
    y1 = s1.y[:, -1]
    s2 = solve_ivp(lambda p, y: 1j * r1 * np.exp(1j * p) * rhs(r1 * np.exp(1j * p), y), (alpha, 0.0), y1,
                   method="DOP853", rtol=rtol, atol=1e-30 + rtol * 1e-6)
    return s2.y[:, -1], s1.nfev + s2.nfev


class Problem:
    def __init__(self, f0=F0_CSS, om=OM_CSS, r0=0.1, r1=0.3, alpha=0.375 * math.pi, rtol=1e-13, N=140):
        self.f0, self.om, self.r0, self.r1, self.alpha, self.rtol, self.N = f0, om, r0, r1, alpha, rtol, N
        self.E = equations()
        self.set_bg(f0, om)

    def set_bg(self, f0, om):
        self.f0, self.om = f0, om
        self.bgc, self.bg_series_resid = bg_series(f0, om, self.N)

    def bg_rhs(self, xi, y):
        f, f1, g, g1, b = y[:5]
        f2, g2, b1 = self.E["bg_rhs"](xi, f, f1, g, g1, b, self.om)
        return np.array([f1, f2, g1, g2, b1]), (f2, g2, b1)

    def bg_init(self, xi0):
        out = []
        info = []
        for c in self.bgc:
            s, d, kmin, tmin = opt_sum(c, xi0)
            out += [s, d]
            info.append((kmin, tmin))
        return np.array([out[0], out[1], out[2], out[3], out[4]]), info

    def bg_jump(self):
        """f(r1) on the upper minus the lower lateral solution, normalized by exp(-1/(2 r1^2))"""
        vals = []
        for sg in (1, -1):
            xi0 = self.r0 * np.exp(1j * sg * self.alpha)
            y0, _ = self.bg_init(xi0)
            y1, _ = path_integrate(lambda x, y: self.bg_rhs(x, y)[0], y0, self.r0, self.r1, sg * self.alpha,
                                   self.rtol)
            vals.append(y1)
        return (vals[0] - vals[1]) / math.exp(-0.5 / self.r1 ** 2), vals

    def select(self, tol=1e-12, maxit=12, log=print):
        """Newton on (f0, omega) for a zero jump of f, until the jump is below tol or stops falling (the floor of the
        double-precision integration, about 2e-13 normalized)"""
        x = np.array([self.f0, self.om])
        best = (np.inf, x.copy())
        for it in range(maxit):
            self.set_bg(*x)
            J0, _ = self.bg_jump()
            r = np.array([J0[0].real, J0[0].imag])
            log("  newton %d: f0 %.15f omega %.15f |jump f| %.2e (series residual %.1e)" % (
                it, x[0], x[1], abs(J0[0]), self.bg_series_resid))
            if abs(J0[0]) >= 0.1 * best[0]:
                x = best[1]
                break
            best = (abs(J0[0]), x.copy())
            if abs(J0[0]) < tol:
                break
            Jm = np.zeros((2, 2))
            for i, h in enumerate((1e-7, 1e-7)):
                xp = x.copy()
                xp[i] += h
                self.set_bg(*xp)
                Jp, _ = self.bg_jump()
                Jm[:, i] = [(Jp[0].real - r[0]) / h, (Jp[0].imag - r[1]) / h]
            x = x - np.linalg.solve(Jm, r)
        self.set_bg(*x)
        return x, abs(self.bg_jump()[0][0])

    def det(self, lam):
        """D(lam): the 2 x 2 determinant of the jumps of (u, v) for the data (1, 0) and (0, 1), w(0) = 0"""
        lam = complex(lam)
        pcs = [pert_series(self.bgc, self.om, lam, u0, v0, self.N)[0] for u0, v0 in ((1.0, 0.0), (0.0, 1.0))]
        A = self.E["A"]
        jumps = []
        nfe = 0
        for sg in (1, -1):
            xi0 = self.r0 * np.exp(1j * sg * self.alpha)
            yb, _ = self.bg_init(xi0)
            cols = []
            for pc in pcs:
                col = []
                for cc in pc:
                    s, d, _, _ = opt_sum(cc, xi0)
                    col += [s, d]
                cols.append(np.array([col[0], col[1], col[2], col[3], col[4]]))
            y0 = np.concatenate([yb] + cols)

            def rhs(xi, y):
                dyb, (f2, g2, b1) = self.bg_rhs(xi, y)
                Am = np.array(A(xi, y[0], y[1], f2, y[2], y[3], g2, y[4], b1, self.om, lam), dtype=complex)
                return np.concatenate([dyb, Am @ y[5:10], Am @ y[10:15]])
            y1, nf = path_integrate(rhs, y0, self.r0, self.r1, sg * self.alpha, self.rtol)
            nfe += nf
            jumps.append(y1)
        d = (jumps[0] - jumps[1]) / math.exp(-0.5 / self.r1 ** 2)
        M = np.array([[d[5], d[10]], [d[7], d[12]]])              # rows: jump of u, of v; columns: the two data
        self.last = dict(nfev=nfe, jump_bg=abs(d[0]))
        return complex(np.linalg.det(M))

    def root(self, lam0, tol=1e-13, maxit=40, log=print):
        l0, l1 = complex(lam0), complex(lam0) + 1e-3
        d0, d1 = self.det(l0), self.det(l1)
        for it in range(maxit):
            l2 = l1 - d1 * (l1 - l0) / (d1 - d0)
            log("  secant %d: lambda %.13f %+.13fi |D| %.2e" % (it, l2.real, l2.imag, abs(d1)))
            if abs(l2 - l1) < tol:
                return l2, it
            l0, d0, l1, d1 = l1, d1, l2, self.det(l2)
        return l2, maxit


# ------------------------------------------------------------------ the planted check
def planted(r1=0.3, r0=0.1, alpha=0.375 * math.pi, rtol=1e-13, N=140, beta=0.7, log=print, mutate=False):
    """y(xi) = 2F0(a, beta;; 2 xi^2): x^2 y_xx + ((a + beta + 1) x - 1) y_x + a beta y = 0 with x = 2 xi^2, i.e.
    xi^3 y'' - y' + (2 a + 2 beta + 1) xi^2 y' + 4 a beta xi y = 0; same series machinery and paths as the CSS problem"""
    xi, Y0, Y1, Y2, a_, b_ = sp.symbols("xi Y0 Y1 Y2 a b")
    mu = 1.001 if mutate else 1.0                 # the control: the y' coefficient off by 0.1%, series and integration
    num = xi ** 3 * Y2 - Y1 + mu * (2 * a_ + 2 * b_ + 1) * xi ** 2 * Y1 + 4 * a_ * b_ * xi * Y0
    ser_fn = sp.lambdify([xi, Y0, Y1, Y2, a_, b_], [num], "math")

    def jump(a):
        def resid_at(k, cf):
            L = k + 1
            Yj = jets_of(cf[0], L, 2)
            x = Ser(np.r_[0, 1, np.zeros(L - 2)] if L > 1 else [0])
            vv = ser_fn(x, Yj[0], Yj[1], Yj[2], a, beta)[0]
            return [vv.c[k] if isinstance(vv, Ser) else vv]
        (c,), res = series_solve(resid_at, [1.0], N)
        exact_c = [float(mp.rf(a, m) * mp.rf(beta, m) * 2 ** m / mp.factorial(m)) for m in range(6)]
        cerr = max(abs(c[2 * m] - exact_c[m]) / max(1, abs(exact_c[m])) for m in range(6))
        rhs = lambda z, y: np.array([y[1], (y[1] - mu * (2 * a + 2 * beta + 1) * z ** 2 * y[1]
                                            - 4 * a * beta * z * y[0]) / z ** 3])
        vals = []
        for sg in (1, -1):
            z0 = r0 * np.exp(1j * sg * alpha)
            s, d, kmin, tmin = opt_sum(c, z0)
            y1, _ = path_integrate(rhs, np.array([s, d]), r0, r1, sg * alpha, rtol)
            vals.append(y1[0])
        return vals[0] - vals[1], cerr, res

    def exact(a):
        mp.mp.dps = 30
        x = mp.mpf(2) * mp.mpf(r1) ** 2
        if a <= 0 and abs(a - round(a)) < 1e-15:
            return 0j
        I = mp.quad(lambda tau: mp.exp(-1 / x - tau) * (1 / x + tau) ** (a - 1) * tau ** (-beta), [0, 1, mp.inf])
        return complex(2j * mp.sin(mp.pi * beta) / mp.gamma(a) * x ** (-beta) * I)
    ok = True
    rows = []
    for a in (0.5, 1.3, -0.4, -1.0, -2.0):
        j, cerr, res = jump(a)
        ex = exact(a)
        err = abs(j - ex) / max(abs(ex), 1e-300) if ex != 0 else abs(j)
        rows.append((a, j, ex, err, cerr))
        log("  a = %5.2f: jump %s exact %s %s %.1e (coefficients vs Pochhammer %.1e)" % (
            a, "%.12e%+.12ei" % (j.real, j.imag), "%.12e%+.12ei" % (ex.real, ex.imag),
            "rel err" if ex != 0 else "|jump|", err, cerr))
        ok &= (err < 1e-9 if ex != 0 else err < 1e-12) and cerr < 1e-12
    # the zero in a, by the secant method from -0.8 and from -2.3
    for a0, want in ((-0.8, -1.0), (-2.3, -2.0)):
        x0, x1 = a0, a0 + 1e-3
        j0, j1 = jump(x0)[0], jump(x1)[0]
        for it in range(40):
            x2 = x1 - (j1 * (x1 - x0) / (j1 - j0)).real
            if abs(x2 - x1) < 1e-14:
                break
            x0, j0, x1, j1 = x1, j1, x2, jump(x2)[0]
        log("  zero in a from %.1f: %.15f (exact %.1f), |error| %.1e" % (a0, x2, want, abs(x2 - want)))
        ok &= abs(x2 - want) < 1e-12
    log("planted %s" % ("PASS" if ok else "FAIL"))
    return ok


# ------------------------------------------------------------------ derive
def derive(log=print):
    t = time.time()
    E = equations()
    S, xi, om, lam = E["S"], E["xi"], E["om"], E["lam"]
    log("equations derived from the PDE in %.1f s" % (time.time() - t))
    bs = E["bsol"]
    F0, F1, G0, G1, B0 = S["F0"], S["F1"], S["G0"], S["G1"], S["B0"]
    g = xi * F1 - sp.I * om * F0
    gt = xi * G1 + sp.I * om * G0
    # the background in compact form: b' = 2 xi g gt / b, f'' = (b^4 f' + 2(i w - 1) xi^2 b^2 f' + w(w + i) xi b^2 f
    # + 2 xi^3 g^2 gt) / (xi^3 b^2), ft'' the same with w -> -w and (f, g) <-> (ft, gt)
    chk = [sp.simplify(bs[S["B1"]] - 2 * xi * g * gt / B0),
           sp.simplify(bs[S["F2"]] - (B0 ** 4 * F1 + 2 * (sp.I * om - 1) * xi ** 2 * B0 ** 2 * F1
                                      + om * (om + sp.I) * xi * B0 ** 2 * F0 + 2 * xi ** 3 * g ** 2 * gt) / (xi ** 3 * B0 ** 2)),
           sp.simplify(bs[S["G2"]] - (B0 ** 4 * G1 - 2 * (sp.I * om + 1) * xi ** 2 * B0 ** 2 * G1
                                      + om * (om - sp.I) * xi * B0 ** 2 * G0 + 2 * xi ** 3 * gt ** 2 * g) / (xi ** 3 * B0 ** 2))]
    log("background in compact form: %s" % ("matches" if all(c == 0 for c in chk) else "DIFFERS %s" % chk))
    ok = all(c == 0 for c in chk)
    # exact solutions of the linear system, in jet space: Dx is the total xi-derivative along background solutions
    G2s, F2s, B1s = bs[S["G2"]], bs[S["F2"]], bs[S["B1"]]
    G1s = S["G1"]

    def Dx(e):
        return (sp.diff(e, xi) + sp.diff(e, F0) * F1 + sp.diff(e, F1) * F2s + sp.diff(e, G0) * G1s
                + sp.diff(e, G1) * G2s + sp.diff(e, B0) * B1s)
    psol = E["psol"]
    modes = [("s-translation, lambda = 2", F1 / xi, G1 / xi, B1s / xi, 2),
             ("t-translation, lambda = 1", g, gt, xi * B1s, 1),
             ("reparametrization, any lambda", g, gt, (lam - 1) * B0 + xi * B1s, lam),
             ("phase, lambda = 0", sp.I * F0, -sp.I * G0, sp.Integer(0), 0),
             ("shift of F, lambda = i omega", sp.Integer(1), sp.Integer(0), sp.Integer(0), sp.I * om),
             ("control: s-translation at lambda = 2.01 (must fail)", F1 / xi, G1 / xi, B1s / xi, sp.Rational(201, 100))]
    pt = {xi: 0.37, F0: 0.3 + 0.2j, F1: -0.4 + 0.7j, G0: 0.25 - 0.1j, G1: 0.6 - 0.3j, B0: 1.3 + 0.1j, om: 3.1,
          lam: 0.77 + 0.4j}
    for name, U, V, W, lv in modes:
        sub = {S["U0"]: U, S["U1"]: Dx(U), S["V0"]: V, S["V1"]: Dx(V), S["W0"]: W, S["F2"]: F2s, S["G2"]: G2s,
               S["B1"]: B1s, lam: lv}
        res = []
        for key, target in ((S["U2"], Dx(Dx(U))), (S["V2"], Dx(Dx(V))), (S["W1"], Dx(W))):
            r = (target - psol[key].subs(sub, simultaneous=True)).subs(lam, lv)
            res.append(abs(complex(sp.N(r.subs(pt)))))
        good = max(res) < 1e-10
        expect = not name.startswith("control")
        ok &= good == expect
        log("  %-52s residuals %s -> %s" % (name, " ".join("%.1e" % x for x in res),
                                            "exact" if good else "not a solution"))
    log("derive %s" % ("PASS" if ok else "FAIL"))
    return ok


# ------------------------------------------------------------------ main
def main():
    args = sys.argv[1:]
    opts = {a[2:].split("=", 1)[0]: a.split("=", 1)[1] for a in args if a.startswith("--")}
    pos = [a for a in args if not a.startswith("--")]
    out = opts.get("out")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        if out:
            with open(out, "a") as fo:
                fo.write(json.dumps(rec) + "\n")
    kw = dict(r0=float(opts.get("r0", 0.1)), r1=float(opts.get("r1", 0.3)),
              alpha=float(opts.get("alpha", 0.375)) * math.pi, rtol=float(opts.get("rtol", 1e-13)),
              N=int(opts.get("N", 140)))
    if not pos:
        raise SystemExit(__doc__)
    cmd = pos[0]
    if cmd == "derive":
        sys.exit(0 if derive() else 1)
    if cmd == "planted":
        sys.exit(0 if planted(**{k: kw[k] for k in ("r0", "r1", "alpha", "rtol", "N")},
                              mutate=opts.get("mutate") == "coef") else 1)
    P = Problem(f0=float(opts.get("f0", F0_CSS)), om=float(opts.get("omega", OM_CSS)), **kw)
    if "select" in opts or cmd == "bg":
        t = time.time()
        x, j = P.select()
        emit(dict(kind="bg", f0=x[0], omega=x[1], A=x[0] * x[1], jump=j, seconds=round(time.time() - t, 1),
                  series_resid=P.bg_series_resid, **{k: (v / math.pi if k == "alpha" else v) for k, v in kw.items()}))
        if cmd == "bg":
            return
    if cmd == "det":
        for s in pos[1:]:
            lam = complex(s)
            t = time.time()
            d = P.det(lam)
            emit(dict(kind="det", lam=[lam.real, lam.imag], D=[d.real, d.imag], f0=P.f0, omega=P.om,
                      seconds=round(time.time() - t, 1), **P.last))
        return
    if cmd == "root":
        t = time.time()
        lam, it = P.root(complex(pos[1]))
        emit(dict(kind="root", start=pos[1], lam=[lam.real, lam.imag], iterations=it, f0=P.f0, omega=P.om,
                  seconds=round(time.time() - t, 1), **{k: (v / math.pi if k == "alpha" else v) for k, v in kw.items()}))
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
