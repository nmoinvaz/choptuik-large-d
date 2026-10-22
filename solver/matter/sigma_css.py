#!/usr/bin/env python3
"""CSS solutions of the self-gravitating hedgehog sigma model in D = n + 2 dimensions, and their growing modes.

Equations (derived and checked in sigma_eqs.py), conventions of solver/dss.py: x = r/(-t), tau = -ln(-t), polar-areal
gauge, c = alpha/a, SSH at x = 1 (c(1) = 1).  CSS: u = U(x),

  (C^2 - x^2) U'' + U' [C C' + n C^2/x - 2x + x^2 C'/C] = n C^2 a^2 sin(2U) / (2x^2)
  x C'/C = (n-1)(a^2-1) - 2 eta a^2 sin^2 U
  a^2 = [(n-1)/2 - (eta/n) x^2 U'^2 (1 - x^2/C^2)] / [(n-1)/2 - eta sin^2 U]

regular at x = 0 (U = q x + ..., C = c0 + ...) and analytic at x = 1 (C = 1, free U(1); for D = 4, U(1) = pi/2 and
U'(1) is free).  Shooting from Taylor series at both ends (generic truncated-power-series arithmetic, any order),
DOP853 to a matching point, Newton on (q, c0, U(1) or U'(1)).

Linear modes u = U + e^{lambda tau} v(x), ln a -> ln a + e^{lambda tau} al(x), ln c -> ln c + e^{lambda tau} ga(x):
  RW = x^2 (1+lam) dq + x^3 dq' - x^2 d(cu_x)' - n x d(cu_x) + (n/2) C a^2 [(ga + 2 al) sin2U + 2 cos2U v]
  RH = x al' - (eta/n)(2x^2 U' v' + 2x^2 q0 dq) - eta a^2 (2 al sin^2U + sin2U v) + (n-1) a^2 al
  RS = x ga' - 2(n-1) a^2 al + 2 eta a^2 (2 al sin^2U + sin2U v)
  dq = (lam v + x v')/C - x U' ga/C,  q0 = x U'/C,  d(cu_x) = C (v' + ga U')
with gauge ga(1) = 0 (the SSH stays at x = 1), v regular at 0 and analytic at 1.  lambda is a root of the 4x4
matching determinant.  The momentum constraint RM = lam al + x al' - (2 eta/n) x [x U' v' + U'(lam v + x v')] is not
imposed and is reported as a check.  The gauge mode (shift of the SSH ray) has lambda = 1 + c_1 = 1 since C'(1) = 0.

    python3 sigma_css.py flat        # eta = 0, D = 4..7 against Bizon-Chmaj-Tabor 2000 / Biernat-Bizon-Maliborski 2017
    python3 sigma_css.py d4          # D = 4 with gravity, eta scan
"""
import sys
import math
import json
import numpy as np
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------------- truncated power series
class TPS:
    __array_priority__ = 1000

    def __init__(self, c):
        self.c = np.array(c)

    @property
    def K(self):
        return len(self.c) - 1

    def _co(self, o):
        if isinstance(o, TPS):
            return o.c
        z = np.zeros(len(self.c), dtype=np.result_type(self.c, np.asarray(o)))
        z[0] = o
        return z

    def __add__(self, o):
        return TPS(self.c + self._co(o))
    __radd__ = __add__

    def __sub__(self, o):
        return TPS(self.c - self._co(o))

    def __rsub__(self, o):
        return TPS(self._co(o) - self.c)

    def __neg__(self):
        return TPS(-self.c)

    def __mul__(self, o):
        if not isinstance(o, TPS):
            return TPS(self.c * o)
        return TPS(np.convolve(self.c, o.c)[:len(self.c)])
    __rmul__ = __mul__

    def __truediv__(self, o):
        if not isinstance(o, TPS):
            return TPS(self.c / o)
        return self * o.inv()

    def __rtruediv__(self, o):
        return self.inv() * o

    def inv(self):
        b = self.c
        q = np.zeros_like(b, dtype=np.result_type(b, 1.0))
        q[0] = 1.0 / b[0]
        for k in range(1, len(b)):
            q[k] = -np.dot(b[1:k + 1], q[k - 1::-1][:k]) / b[0]
        return TPS(q)

    def __pow__(self, p):
        assert isinstance(p, int) and p >= 0
        out = TPS(self._co(1.0))
        for _ in range(p):
            out = out * self
        return out

    def deriv(self):
        d = np.zeros_like(self.c)
        d[:-1] = self.c[1:] * np.arange(1, len(self.c))
        return TPS(d)

    def sincos(self):
        u = self.c
        K = len(u)
        s = np.zeros(K, dtype=np.result_type(u, 1.0))
        c = np.zeros_like(s)
        s[0], c[0] = np.sin(u[0]), np.cos(u[0])
        ju = np.arange(K) * u
        for k in range(1, K):
            s[k] = np.dot(ju[1:k + 1], c[k - 1::-1][:k]) / k
            c[k] = -np.dot(ju[1:k + 1], s[k - 1::-1][:k]) / k
        return TPS(s), TPS(c)

    def __call__(self, e):
        out = 0.0
        for k in range(len(self.c) - 1, -1, -1):
            out = out * e + self.c[k]
        return out


def sincos(U):
    if isinstance(U, TPS):
        return U.sincos()
    if hasattr(U, 'free_symbols'):          # sympy expression (symbolic checks)
        import sympy as sp
        return sp.sin(U), sp.cos(U)
    return np.sin(U), np.cos(U)


# ----------------------------------------------------------------------------- residuals (floats or TPS)
def bg_res(x, U, V, W2, C, Cp, n, eta):
    """x^2 * (wave equation), slicing residual, a^2."""
    s, co = sincos(U)
    a2 = ((n - 1) / 2 - (eta / n) * x * x * V * V * (1 - x * x / (C * C))) / ((n - 1) / 2 - eta * s * s)
    RS = x * Cp - C * ((n - 1) * (a2 - 1) - 2 * eta * a2 * s * s)
    RW = x * x * (C * C - x * x) * W2 + V * (x * x * C * Cp + n * x * C * C - 2 * x * x * x + x * x * x * x * Cp / C) \
        - n * C * C * a2 * s * co
    return RW, RS, a2


def lin_res(x, U, U1, U2, C, Cp, a2, v, vp, vpp, al, alp, ga, gap, lam, n, eta):
    s, co = sincos(U)
    s2U = 2 * s * co
    c2U = co * co - s * s
    q0 = x * U1 / C
    dq = (lam * v + x * vp) / C - x * U1 * ga / C
    dqp = ((lam + 1) * vp + x * vpp) / C - (lam * v + x * vp) * Cp / (C * C) \
        - (U1 * ga + x * U2 * ga + x * U1 * gap) / C + x * U1 * ga * Cp / (C * C)
    dcu = C * (vp + ga * U1)
    dcup = Cp * (vp + ga * U1) + C * (vpp + gap * U1 + ga * U2)
    RW = x * x * (1 + lam) * dq + x * x * x * dqp - x * x * dcup - n * x * dcu \
        + (n / 2) * C * a2 * ((ga + 2 * al) * s2U + 2 * c2U * v)
    RH = x * alp - (eta / n) * (2 * x * x * U1 * vp + 2 * x * x * q0 * dq) - eta * a2 * (2 * al * s * s + s2U * v) \
        + (n - 1) * a2 * al
    RS = x * gap - 2 * (n - 1) * a2 * al + 2 * eta * a2 * (2 * al * s * s + s2U * v)
    RM = lam * al + x * alp - (2 * eta / n) * x * (x * U1 * vp + U1 * (lam * v + x * vp))
    return RW, RH, RS, RM


def lin_pre(x, U, C, n, eta):
    """Background quantities (TPS) used by lin_res_pre."""
    U1 = U.deriv(); U2 = U1.deriv(); Cp = C.deriv()
    a2 = bg_res(x, U, U1, U2, C, Cp, n, eta)[2]
    s, co = sincos(U)
    iC = C.inv()
    return dict(x=x, U1=U1, U2=U2, C=C, Cp=Cp, iC=iC, iC2=iC * iC, a2=a2, ss=s * s, s2U=2 * s * co, c2U=co * co - s * s)


def lin_res_pre(B, v, vp, vpp, al, alp, ga, gap, lam, n, eta):
    """lin_res with precomputed background B (multiplications only)."""
    x, U1, U2, C, Cp, iC, iC2, a2 = B['x'], B['U1'], B['U2'], B['C'], B['Cp'], B['iC'], B['iC2'], B['a2']
    ss, s2U, c2U = B['ss'], B['s2U'], B['c2U']
    q0 = x * U1 * iC
    dq = (lam * v + x * vp) * iC - x * U1 * ga * iC
    dqp = ((lam + 1) * vp + x * vpp) * iC - (lam * v + x * vp) * Cp * iC2 \
        - (U1 * ga + x * U2 * ga + x * U1 * gap) * iC + x * U1 * ga * Cp * iC2
    dcu = C * (vp + ga * U1)
    dcup = Cp * (vp + ga * U1) + C * (vpp + gap * U1 + ga * U2)
    src = 2 * al * ss + s2U * v
    RW = x * x * (1 + lam) * dq + x * x * x * dqp - x * x * dcup - n * x * dcu \
        + (n / 2) * C * a2 * ((ga + 2 * al) * s2U + 2 * c2U * v)
    RH = x * alp - (eta / n) * (2 * x * x * U1 * vp + 2 * x * x * q0 * dq) - eta * a2 * src + (n - 1) * a2 * al
    RS = x * gap - 2 * (n - 1) * a2 * al + 2 * eta * a2 * src
    return RW, RH, RS


# ----------------------------------------------------------------------------- background series
def series_ssh(n, eta, U1, b=None, K=30):
    """Regular Taylor series at x = 1 (eps = x - 1).  Free data U(1) (or, at resonance n = 2k+2, the coefficient
    u_{k+1} = b and U(1) must satisfy the solvability condition)."""
    u = np.zeros(K + 3)
    c = np.zeros(K + 3)
    u[0], c[0] = U1, 1.0
    x = TPS(np.r_[1.0, 1.0, np.zeros(K + 1)])

    def res():
        Ut = TPS(u)
        Ct = TPS(c)
        V = Ut.deriv()
        return bg_res(x, Ut, V, V.deriv(), Ct, Ct.deriv(), n, eta)
    for k in range(K + 1):
        # c_{k+1} from slicing at order k
        c[k + 1] = 0.0
        r0 = res()[1].c[k]
        h = 1.0 + abs(r0)
        c[k + 1] = h
        r1 = res()[1].c[k]
        c[k + 1] = -h * r0 / (r1 - r0)
        u[k + 1] = 0.0
        r0 = res()[0].c[k]
        h = 1.0 + abs(r0)
        u[k + 1] = h
        r1 = res()[0].c[k]
        if abs(r1 - r0) < 1e-12 * h:        # resonance
            if abs(r0) > 1e-9:
                raise ValueError("resonant order %d not solvable (r = %g)" % (k, r0))
            u[k + 1] = b
        else:
            u[k + 1] = -h * r0 / (r1 - r0)
    return TPS(u[:K + 1]), TPS(c[:K + 1])


def series_centre(n, eta, q, c0, K=24):
    u = np.zeros(K + 3)
    c = np.zeros(K + 3)
    c[0] = c0
    x = TPS(np.r_[0.0, 1.0, np.zeros(K + 1)])

    def res():
        Ut = TPS(u)
        Ct = TPS(c)
        V = Ut.deriv()
        return bg_res(x, Ut, V, V.deriv(), Ct, Ct.deriv(), n, eta)
    for k in range(K + 1):
        if k >= 1:
            c[k] = 0.0
            r0 = res()[1].c[k]
            h = 1.0 + abs(r0)
            c[k] = h
            r1 = res()[1].c[k]
            c[k] = -h * r0 / (r1 - r0)
        if k == 1:
            u[1] = q
            continue
        u[k] = 0.0
        r0 = res()[0].c[k]
        h = 1.0 + abs(r0)
        u[k] = h
        r1 = res()[0].c[k]
        u[k] = -h * r0 / (r1 - r0)
    return TPS(u[:K + 1]), TPS(c[:K + 1])


def radius(S, kmin=6):
    a = np.abs(S.c[kmin:])
    k = np.arange(kmin, len(S.c))
    good = a > 0
    if good.sum() < 3:
        return np.inf
    return float(np.min(a[good] ** (-1.0 / k[good])))


# ----------------------------------------------------------------------------- ODE right-hand sides
def bg_rhs(x, U, V, C, n, eta):
    x, U, V, C = float(x), float(U), float(V), float(C)
    s, co = math.sin(U), math.cos(U)
    a2 = ((n - 1) / 2 - (eta / n) * x * x * V * V * (1 - x * x / (C * C))) / ((n - 1) / 2 - eta * s * s)
    Cp = C * ((n - 1) * (a2 - 1) - 2 * eta * a2 * s * s) / x
    W2 = -(V * (x * x * C * Cp + n * x * C * C - 2 * x ** 3 + x ** 4 * Cp / C) - n * C * C * a2 * s * co) / (x * x * (C * C - x * x))
    return V, W2, Cp, a2


def full_rhs(x, y, n, eta, lam):
    """y = [U, V, C, v, vp, al, ga] (complex allowed for the last four).  Explicit form of lin_res solved for
    (v'', al', ga'); lin_res itself is the reference (mode_check evaluates it on the solutions)."""
    x = float(x)
    U, V, C = float(y[0].real), float(y[1].real), float(y[2].real)
    s, co = math.sin(U), math.cos(U)
    ss = s * s
    a2 = ((n - 1) / 2 - (eta / n) * x * x * V * V * (1 - x * x / (C * C))) / ((n - 1) / 2 - eta * ss)
    Cp = C * ((n - 1) * (a2 - 1) - 2 * eta * a2 * ss) / x
    W2 = -(V * (x * x * C * Cp + n * x * C * C - 2 * x ** 3 + x ** 4 * Cp / C) - n * C * C * a2 * s * co) / (x * x * (C * C - x * x))
    v, vp, al, ga = complex(y[3]), complex(y[4]), complex(y[5]), complex(y[6])
    s2U = 2 * s * co
    c2U = co * co - ss
    src = 2 * al * ss + s2U * v
    gap = (2 * (n - 1) * a2 * al - 2 * eta * a2 * src) / x
    q0 = x * V / C
    dq = (lam * v + x * vp) / C - x * V * ga / C
    alp = ((eta / n) * (2 * x * x * V * vp + 2 * x * x * q0 * dq) + eta * a2 * src - (n - 1) * a2 * al) / x
    dqp0 = (lam + 1) * vp / C - (lam * v + x * vp) * Cp / (C * C) - (V * ga + x * W2 * ga + x * V * gap) / C \
        + x * V * ga * Cp / (C * C)
    dcu = C * (vp + ga * V)
    dcup0 = Cp * (vp + ga * V) + C * (gap * V + ga * W2)
    RW0 = x * x * (1 + lam) * dq + x ** 3 * dqp0 - x * x * dcup0 - n * x * dcu + (n / 2) * C * a2 * ((ga + 2 * al) * s2U + 2 * c2U * v)
    vpp = -RW0 * C / (x * x * (x * x - C * C))
    return np.array([V, W2, Cp, vp, vpp, alp, gap])


# ----------------------------------------------------------------------------- the solver
class CSS:
    def __init__(self, D, eta, xm=0.5, Kc=24, Ks=30, rtol=1e-12, atol=1e-13):
        self.D, self.n, self.eta = float(D), float(D) - 2.0, float(eta)
        self.xm, self.Kc, self.Ks = xm, Kc, Ks
        self.rtol, self.atol = rtol, atol
        k = (self.n - 2) / 2
        self.resonant = abs(k - round(k)) < 1e-12 and round(k) >= 0
        self.kres = int(round(k)) if self.resonant else None
        self._pre = {}

    # ---- background
    def ends(self, z):
        """z = (q, c0, s1): s1 = U(1), or at resonance the free coefficient (U(1) fixed: pi/2 when n = 2)."""
        q, c0, s1 = z
        Uc, Cc = series_centre(self.n, self.eta, q, c0, self.Kc)
        if self.resonant:
            if self.kres != 0:
                raise NotImplementedError("even n >= 4: use a non-integer D next to it")
            Us, Cs = series_ssh(self.n, self.eta, np.pi / 2, s1, self.Ks)
        else:
            Us, Cs = series_ssh(self.n, self.eta, s1, None, self.Ks)
        if not getattr(self, 'frozen', False):
            Rc = min(radius(Uc), radius(Cc))
            Rs = min(radius(Us), radius(Cs))
            self.x0 = min(0.3 * Rc, 0.25 * self.xm, 0.05)
            self.d1 = min(0.3 * Rs, 0.5 * (1 - self.xm), 0.3)
        return Uc, Cc, Us, Cs

    def start_states(self, z):
        key = tuple(np.asarray(z, float))
        if getattr(self, '_ss_key', None) == key:
            return self._ss_val
        Uc, Cc, Us, Cs = self.ends(z)
        x0, d1 = self.x0, self.d1
        yc = np.array([Uc(x0), Uc.deriv()(x0), Cc(x0)])
        ys = np.array([Us(-d1), Us.deriv()(-d1), Cs(-d1)])
        self._ss_key, self._ss_val = key, (yc, ys, (Uc, Cc, Us, Cs))
        return self._ss_val

    def bg_integrate(self, y0, xa, xb, dense=False):
        f = lambda x, y: np.array(bg_rhs(x, y[0], y[1], y[2], self.n, self.eta)[:3])
        sol = solve_ivp(f, (xa, xb), y0, method='DOP853', rtol=self.rtol, atol=self.atol, dense_output=dense)
        return sol

    def mismatch(self, z):
        yc, ys, _ = self.start_states(z)
        a = self.bg_integrate(yc, self.x0, self.xm).y[:, -1]
        b = self.bg_integrate(ys, 1 - self.d1, self.xm).y[:, -1]
        return a - b

    def solve(self, z0, tol=1e-12, maxit=40, verbose=False):
        z = np.array(z0, float)
        self.frozen = False
        self._ss_key = None
        F = self.mismatch(z)
        self.frozen = True                 # keep x0, d1 fixed during the iteration
        for it in range(maxit):
            J = np.empty((3, 3))
            for j in range(3):
                h = 1e-7 * max(1.0, abs(z[j]))
                zp = z.copy(); zp[j] += h
                zm = z.copy(); zm[j] -= h
                J[:, j] = (self.mismatch(zp) - self.mismatch(zm)) / (2 * h)
            dz = np.linalg.solve(J, -F)
            lam = 1.0
            while True:
                Fn = self.mismatch(z + lam * dz)
                if np.linalg.norm(Fn) < np.linalg.norm(F) or lam < 1e-3:
                    break
                lam /= 2
            z = z + lam * dz
            F = Fn
            if verbose:
                print("   it %d  |F| = %.3e  z = %s" % (it, np.linalg.norm(F), z))
            if np.linalg.norm(F) < tol or np.linalg.norm(lam * dz) < 1e-14 * (1 + np.linalg.norm(z)):
                break
        self.z = z
        self._pre = {}
        self.F = np.linalg.norm(F)
        _, _, (Uc, Cc, Us, Cs) = self.start_states(z)
        self.U1, self.V1 = Us.c[0], Us.c[1]
        self.sR = (radius(Uc), radius(Us))
        self.cond = np.linalg.cond(J)
        return z

    def profile(self, xs=None):
        """Background on [0, 1]: returns x, U, U', C, a^2."""
        yc, ys, (Uc, Cc, Us, Cs) = self.start_states(self.z)
        sa = self.bg_integrate(yc, self.x0, self.xm, dense=True)
        sb = self.bg_integrate(ys, 1 - self.d1, self.xm, dense=True)
        if xs is None:
            xs = np.linspace(0, 1, 201)
        out = []
        for x in xs:
            if x <= self.x0:
                y = [Uc(x), Uc.deriv()(x), Cc(x)]
            elif x >= 1 - self.d1:
                y = [Us(x - 1), Us.deriv()(x - 1), Cs(x - 1)]
            elif x <= self.xm:
                y = sa.sol(x)
            else:
                y = sb.sol(x)
            out.append(y)
        out = np.array(out)
        a2 = np.array([bg_rhs(max(x, 1e-12), *y, self.n, self.eta)[3] for x, y in zip(xs, out)])
        return xs, out[:, 0], out[:, 1], out[:, 2], a2

    def beyond(self, xmax=1e8):
        """Continue the background past the SSH (x > 1, still t < 0) in ln x.  Returns x_B where a^2 -> infinity
        (A = 1/a^2 -> 0 together with x/c -> 1: an apparent horizon, Bizon-Wasserman's type B), or None if the solution
        reaches x = xmax (t -> 0-) without one."""
        _, _, (Uc, Cc, Us, Cs) = self.start_states(self.z)
        d = min(self.d1, 0.2)
        y0 = np.array([Us(d), Us.deriv()(d), Cs(d)])
        n, eta = self.n, self.eta

        def f(lx, y):
            x = math.exp(lx)
            V, W2, Cp, a2 = bg_rhs(x, y[0], y[1], y[2], n, eta)
            return np.array([x * V, x * W2, x * Cp])

        def ev(lx, y):
            return bg_rhs(math.exp(lx), y[0], y[1], y[2], n, eta)[3] - 1e6
        ev.terminal = True
        sol = solve_ivp(f, (math.log(1 + d), math.log(xmax)), y0, method='DOP853', rtol=1e-11, atol=1e-13, events=ev)
        if sol.status == 1:
            return math.exp(sol.t_events[0][0])
        if sol.status < 0:
            x = math.exp(sol.t[-1]); y = sol.y[:, -1]
            if bg_rhs(x, y[0], y[1], y[2], n, eta)[3] > 1e3 or abs(x / y[2] - 1) < 1e-3:
                return x
            return float('nan')
        return None

    # ---- modes
    def pert_series_centre(self, lam, Uc, Cc, v1, ga0, K=None):
        K = K or self.Kc
        n, eta = self.n, self.eta
        key = ('c', K)
        if key not in self._pre:
            x = TPS(np.r_[0.0, 1.0, np.zeros(K - 1)].astype(complex))
            self._pre[key] = lin_pre(x, TPS(Uc.c[:K + 1].astype(complex)), TPS(Cc.c[:K + 1].astype(complex)), n, eta)
        B = self._pre[key]
        v = np.zeros(K + 1, complex); al = np.zeros(K + 1, complex); ga = np.zeros(K + 1, complex)

        def res(k):
            V = TPS(v); A = TPS(al); G = TPS(ga)
            Vp = V.deriv()
            R = lin_res_pre(B, V, Vp, Vp.deriv(), A, A.deriv(), G, G.deriv(), lam, n, eta)
            return np.array([R[0].c[k], R[1].c[k], R[2].c[k]])
        arrs = [v, al, ga]
        for k in range(K + 1):
            free = []
            if k == 0:
                ga[0] = ga0; free = [2]
            if k == 1:
                v[1] = v1; free = [0]
            unk = [j for j in range(3) if j not in free]
            for j in unk:
                arrs[j][k] = 0.0
            r0 = res(k)
            h = 1.0 + np.linalg.norm(r0)
            M = np.zeros((3, len(unk)), complex)
            for i, j in enumerate(unk):
                arrs[j][k] = h
                M[:, i] = (res(k) - r0) / h
                arrs[j][k] = 0.0
            sol = np.linalg.lstsq(M, -r0, rcond=None)[0]
            for i, j in enumerate(unk):
                arrs[j][k] = sol[i]
        return TPS(v), TPS(al), TPS(ga)

    def pert_series_ssh(self, lam, Us, Cs, v0, al0, K=None):
        K = K or self.Ks
        n, eta = self.n, self.eta
        key = ('s', K)
        if key not in self._pre:
            x = TPS(np.r_[1.0, 1.0, np.zeros(K - 1)].astype(complex))
            self._pre[key] = lin_pre(x, TPS(Us.c[:K + 1].astype(complex)), TPS(Cs.c[:K + 1].astype(complex)), n, eta)
        B = self._pre[key]
        v = np.zeros(K + 1, complex); al = np.zeros(K + 1, complex); ga = np.zeros(K + 1, complex)
        v[0], al[0] = v0, al0

        def res(k):
            V = TPS(v); A = TPS(al); G = TPS(ga)
            Vp = V.deriv()
            R = lin_res_pre(B, V, Vp, Vp.deriv(), A, A.deriv(), G, G.deriv(), lam, n, eta)
            return np.array([R[0].c[k], R[1].c[k], R[2].c[k]])
        arrs = [v, al, ga]
        for k in range(K):
            for j in range(3):
                arrs[j][k + 1] = 0.0
            r0 = res(k)
            h = 1.0 + np.linalg.norm(r0)
            M = np.zeros((3, 3), complex)
            for j in range(3):
                arrs[j][k + 1] = h
                M[:, j] = (res(k) - r0) / h
                arrs[j][k + 1] = 0.0
            sol = np.linalg.lstsq(M, -r0, rcond=None)[0]       # exact resonance: least-squares (det has a pole there)
            for j in range(3):
                arrs[j][k + 1] = sol[j]
        return TPS(v), TPS(al), TPS(ga)

    def mode_matrix(self, lam, return_sols=False):
        lam = complex(lam)
        yc, ys, (Uc, Cc, Us, Cs) = self.start_states(self.z)
        x0, x1 = self.x0, 1 - self.d1
        cols = []
        sols = []
        f = lambda x, y: full_rhs(x, y, self.n, self.eta, lam)
        for (v1, ga0) in [(1.0, 0.0), (0.0, 1.0)]:
            V, A, G = self.pert_series_centre(lam, Uc, Cc, v1, ga0)
            y0 = np.r_[yc, V(x0), V.deriv()(x0), A(x0), G(x0)].astype(complex)
            s = solve_ivp(f, (x0, self.xm), y0, method='DOP853', rtol=self.rtol, atol=self.atol, dense_output=return_sols)
            cols.append(s.y[3:, -1]); sols.append(s)
        for (v0, al0) in [(1.0, 0.0), (0.0, 1.0)]:
            V, A, G = self.pert_series_ssh(lam, Us, Cs, v0, al0)
            e = -self.d1
            y0 = np.r_[ys, V(e), V.deriv()(e), A(e), G(e)].astype(complex)
            s = solve_ivp(f, (x1, self.xm), y0, method='DOP853', rtol=self.rtol, atol=self.atol, dense_output=return_sols)
            cols.append(-s.y[3:, -1]); sols.append(s)
        M = np.array(cols).T
        if return_sols:
            return M, sols
        return M

    def det(self, lam):
        M = self.mode_matrix(lam)
        # orthonormalise each end's pair (span-preserving, positive factor) to keep the scale sane
        Qc, Rc = np.linalg.qr(M[:, :2])
        Qs, Rs = np.linalg.qr(M[:, 2:])
        return np.linalg.det(np.c_[Qc, Qs]) * np.sign(np.real(np.prod(np.diag(Rc)) * np.prod(np.diag(Rs))) or 1)

    def detraw(self, lam):
        M = self.mode_matrix(lam)
        return np.linalg.det(M / np.linalg.norm(M, axis=0))

    def find_mode(self, lam0, tol=1e-11, maxit=40, width=0.6):
        """Real growing mode near lam0: bracket a sign change of the (real) determinant in lam0 +- width, avoiding the
        resonance poles of the SSH parametrisation, then refine by Illinois regula falsi.  Falls back to a complex
        secant if no real bracket is found."""
        poles = resonances(self.n, lam0 - 2 * width, lam0 + 2 * width)
        f = lambda l: self.detraw(l).real
        grid = lam0 + width * np.array([0, -0.05, 0.05, -0.15, 0.15, -0.35, 0.35, -0.7, 0.7, -1.0, 1.0]) + 1.1e-5
        vals = {}
        order = sorted(grid)
        for l in order:
            vals[l] = f(l)
        best = None
        for a, b in zip(order[:-1], order[1:]):
            if any(a <= p <= b for p in poles):
                continue
            if np.sign(vals[a]) != np.sign(vals[b]):
                d = min(abs(a - lam0), abs(b - lam0))
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best is not None:
            _, a, b = best
            fa, fb = vals[a], vals[b]
            side = 0
            for _ in range(maxit):
                m = (a * fb - b * fa) / (fb - fa)
                fm = f(m)
                if np.sign(fm) == np.sign(fb):
                    b, fb = m, fm
                    if side == -1:
                        fa /= 2
                    side = -1
                else:
                    a, fa = m, fm
                    if side == 1:
                        fb /= 2
                    side = 1
                if abs(b - a) < tol * (1 + abs(m)) or fm == 0:
                    break
            return complex(m)
        l0 = complex(lam0)
        l1 = l0 * (1 + 1e-4) + 1e-4
        f0, f1 = self.detraw(l0), self.detraw(l1)
        for _ in range(maxit):
            l2 = l1 - f1 * (l1 - l0) / (f1 - f0)
            l0, f0 = l1, f1
            l1, f1 = l2, self.detraw(l2)
            if abs(l1 - l0) < tol * (1 + abs(l1)):
                break
        return l1

    def mode_check(self, lam):
        """Null vector of the matching matrix; returns the max |RM| (momentum constraint) relative to max|al'|."""
        M, sols = self.mode_matrix(lam, return_sols=True)
        Mn = M / np.linalg.norm(M, axis=0)
        w = np.linalg.svd(Mn)[2][-1].conj() / np.linalg.norm(M, axis=0)
        sv = np.linalg.svd(Mn, compute_uv=False)
        lam = complex(lam)
        worst, scale = 0.0, 0.0
        for side, idx in [(0, (0, 1)), (1, (2, 3))]:
            for xx in np.linspace(sols[idx[0]].t[0], sols[idx[0]].t[-1], 9)[1:-1]:
                y = sum((w[i] if side == 0 else -w[i]) * sols[i].sol(xx) for i in idx)
                y[:3] = sols[idx[0]].sol(xx)[:3]
                d = full_rhs(xx, y, self.n, self.eta, lam)
                U, V, C = y[0].real, y[1].real, y[2].real
                _, W2, Cp, a2 = bg_rhs(xx, U, V, C, self.n, self.eta)
                RM = lin_res(xx, U, V, W2, C, Cp, a2, y[3], y[4], d[4], y[5], d[5], y[6], d[6], lam, self.n, self.eta)[3]
                worst = max(worst, abs(RM))
                scale = max(scale, abs(xx * d[5]), abs(lam * y[5]), abs(y[3]), abs(xx * y[4]))
        return worst / max(scale, 1e-300), sv[-1] / sv[0]


def count_modes(S, re_lo=0.05, re_hi=20.0, im_hi=12.0, h0=0.5):
    """Number of zeros of the matching determinant in the rectangle (argument principle; resonance poles of the SSH
    parametrisation inside the rectangle are added back).  Returns (zeros, winding, poles)."""
    corners = [complex(re_lo, -im_hi), complex(re_hi, -im_hi), complex(re_hi, im_hi), complex(re_lo, im_hi)]
    total = 0.0
    for a, b in zip(corners, corners[1:] + corners[:1]):
        L = abs(b - a)
        ts = list(np.linspace(0, 1, int(L / h0) + 2))
        vals = [S.detraw(a + t * (b - a)) for t in ts]
        i = 0
        while i < len(ts) - 1:
            dphi = np.angle(vals[i + 1] / vals[i])
            if abs(dphi) > 0.6 and (ts[i + 1] - ts[i]) * L > 1e-4:
                tm = 0.5 * (ts[i] + ts[i + 1])
                ts.insert(i + 1, tm)
                vals.insert(i + 1, S.detraw(a + tm * (b - a)))
                continue
            total += dphi
            i += 1
    wind = total / (2 * np.pi)
    poles = [p for p in resonances(S.n, re_lo, re_hi)]
    return int(round(wind)) + len(poles), wind, poles


def scan_real(S, lams):
    return np.array([S.detraw(l).real for l in lams])


def resonances(n, lo, hi):
    """lambda where the SSH series of the modes is resonant (n/2 - lambda a positive integer): poles of det."""
    return [n / 2 - 1 - k for k in range(0, 200) if lo <= n / 2 - 1 - k <= hi]


def real_roots(S, lo, hi, N=60):
    """Real roots of the matching determinant in [lo, hi] by sign changes + secant refinement (pole intervals
    excluded)."""
    ls = np.linspace(lo, hi, N) + 1.234567e-3
    ds = scan_real(S, ls)
    poles = resonances(S.n, lo - 1, hi + 1)
    roots = []
    for i in range(N - 1):
        if any(ls[i] <= p <= ls[i + 1] for p in poles):
            continue
        if np.sign(ds[i]) != np.sign(ds[i + 1]):
            a, b, fa, fb = ls[i], ls[i + 1], ds[i], ds[i + 1]
            for _ in range(60):
                m = b - fb * (b - a) / (fb - fa)
                fm = S.detraw(m).real
                if np.sign(fm) == np.sign(fa):
                    a, fa = m, fm
                else:
                    b, fb = m, fm
                if abs(b - a) < 1e-12 or fm == 0:
                    break
            roots.append(m)
    return roots, ls, ds


# ----------------------------------------------------------------------------- drivers
def flat_validation():
    print("Flat space (eta = 0): f_1 and its growing mode, against BCT00 (d = 3) and BBM17 Tables 1-2 (d = 3..6)")
    ref = {4: (21.75741, np.pi / 2, -0.30566, 6.33363), 5: (10.9953, 1.60634, -0.10654, 3.99883),
           6: (7.82119, np.pi / 2, 0.0, 3.0), 7: (6.71508, 1.53534, 0.059052, 2.42624)}
    out = []
    for D in (4, 5, 6.001, 5.999, 7):
        q, U1, V1, lref = ref[int(round(D))]
        S = CSS(D, 0.0)
        z0 = (q, 1.0, V1) if S.resonant else (q, 1.0, U1)
        z = S.solve(z0)
        _, _, (Uc, Cc, Us, Cs) = S.start_states(z)
        roots, _, _ = real_roots(S, 1.5, 8.0, 40)
        lam = [S.find_mode(r0) .real for r0 in roots]
        print("D = %-6g q = f'(0) = %.7f (ref %.7f)  U(1) = %.6f (ref %.6f)  U'(1) = %.6f (ref %.6f)  |F| = %.1e"
              % (D, z[0], q, Us.c[0], U1, Us.c[1], V1, S.F))
        print("          growing modes: %s   (ref lambda_1 = %.5f and the gauge mode 1)" % (np.round(lam, 8), lref))
        out.append(dict(D=D, eta=0.0, q=z[0], c0=z[1], U1=Us.c[0], V1=Us.c[1], lam=lam))
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "flat"
    if what == "flat":
        flat_validation()
