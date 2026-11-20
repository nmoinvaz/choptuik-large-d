#!/usr/bin/env python3
"""The l = 1 mode matrix M(lambda) of dss_ell.DSSEll in double-double (#1025, route (c)).

Same discrete operator as DSSEll.mode_matrix (frozen RK4 step counts, the stored double state z, the same x grid), built
in double-double so that M's small singular values (the decaying-mode ladder of matter/ELL.md section 6) are resolved
far below double rounding. Differences of method, all exact in exact arithmetic:
  - the state is carried in parity-restricted Fourier coefficients (p, m, psi odd; a, c even), so d/dtau is elementwise
    (k omega) and the truncation to k <= kc is the projection back to coefficients; products are formed on the same
    N-point grid;
  - derivatives are carried as dual numbers (value plus one tangent per input direction) instead of a complex step;
  - the tau operators (coefficients <-> grid values, d/dtau on the grid) are built exactly in mpmath and rounded to
    double-double; products with them use Ozaki's error-free splitting into float64 BLAS products;
  - the horizon series' periodic solves are a float64 LU refined with double-double residuals.

    python ell_ddm.py selftest [--mutate=slices|ops]
    python ell_ddm.py check|sshres|det|root ... [--mutate=lam|hionly]   (lam: tangents without lambda; hionly: every
                                                 double-double result rounded to double, the build's precision control)
        operators against DSSEll's float64 ones (1e-14), the Ozaki product against ddnp.matmul (1e-28 relative), and
        dual-number derivatives against the analytic ones in double-double (1e-28); --mutate=slices (2 slices) and
        --mutate=ops (one harmonic off) must fail.
    check and sshres take --Ks K --u0 U (default 30 and 0.04, the records' horizon series)
References: Ozaki, Ogita, Oishi and Rump, Numer. Algorithms 59, 95 (2012) (error-free splitting of matrix products);
double-double arithmetic as in ddnp.py (Dekker 1971; Hida, Li and Bailey 2001)."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ddnp as dd                                    # noqa: E402

MUTATE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")
NSLICE = 2 if MUTATE == "slices" else 5
if MUTATE == "hionly":                       # control: every double-double result keeps its high part only (double precision)
    for _nm in ("add", "sub", "mul", "div", "mul_d", "add_d", "neg"):
        setattr(dd, _nm, (lambda f: lambda *a: (lambda r: dd.R(r.h, np.zeros_like(r.h)))(f(*a)))(getattr(dd, _nm)))


# ---------------------------------------------------------------- exact tau operators
def tau_operators(N, kodd, keven):
    """Exact (mpmath, 40 digits) operators, rounded to double-double, in DSSEll's conventions:
    Bodd (n_odd x N), Beven (n_even x N): coefficients -> grid values (c @ B, DSSSpec.phys);
    Sodd (N x n_odd), Seven (N x n_even): grid values -> coefficients (f @ S, DSSSpec.spec);
    Dm (N x N): d/dtheta on the grid, f @ Dm (DSSEll._Dm: rfft, times i k, Nyquist zeroed, irfft)."""
    import mpmath as mp
    mp.mp.dps = 40
    kodd = [int(k) + (1 if MUTATE == "ops" else 0) for k in kodd]
    keven = [int(k) for k in keven]
    th = [2 * mp.pi * j / N for j in range(N)]

    def todd(rows):
        a = np.array([[float(v) for v in r] for r in rows])
        b = np.array([[float(v - float(v)) for v in r] for r in rows])
        return dd.R(a, b)
    Bo = [[2 * mp.cos(k * t) for t in th] for k in kodd] + [[-2 * mp.sin(k * t) for t in th] for k in kodd]
    Be = [[mp.mpf(1)] * N] + [[2 * mp.cos(k * t) for t in th] for k in keven[1:]] + \
         [[-2 * mp.sin(k * t) for t in th] for k in keven[1:]]
    So = [[mp.cos(k * t) / N for k in kodd] + [-mp.sin(k * t) / N for k in kodd] for t in th]
    Se = [[mp.mpf(1) / N] + [mp.cos(k * t) / N for k in keven[1:]] + [-mp.sin(k * t) / N for k in keven[1:]] for t in th]
    d = [-(mp.mpf(2) / N) * mp.fsum(k * mp.sin(2 * mp.pi * k * r / N) for k in range(1, N // 2)) for r in range(N)]
    Dm = [[d[(j - i) % N] for j in range(N)] for i in range(N)]
    return dict(Bodd=todd(Bo), Beven=todd(Be), Sodd=todd(So), Seven=todd(Se), Dm=todd(Dm))


# ---------------------------------------------------------------- Ozaki products
def _slices(h, l, axis, n):
    """exact split of the double-double array h + l into NSLICE float64 slices along rows (axis=-1 scales per row) or
    columns (axis=-2), each with few enough bits that products of slices summed over n terms are exact in float64"""
    rho = math.ceil((53 + math.log2(max(n, 2))) / 2)
    rh, rl = h.copy(), l.copy()
    out = []
    for _ in range(NSLICE):
        mx = np.max(np.abs(rh), axis=axis, keepdims=True)
        tau = np.where(mx > 0, np.ceil(np.log2(np.where(mx > 0, mx, 1.0))), 0.0)
        sig = 0.75 * np.exp2(rho + tau)
        s = (rh + sig) - sig
        out.append(s)
        rh = rh - s                                          # exact
        rh, rl = dd.two_sum(rh, rl)
    return out


class Op:
    """a fixed double-double operator W (n x m), pre-split for X @ W"""

    def __init__(self, W):
        self.W = W
        self.n = W.shape[0]
        self.sl = _slices(W.h, W.l, axis=-2, n=self.n)

    def rmul(self, X):
        """X @ W for a double-double X (..., n); exact slice products accumulated in double-double"""
        shp = X.shape
        Xh = X.h.reshape(-1, shp[-1]); Xl = X.l.reshape(-1, shp[-1])
        xs = _slices(Xh, Xl, axis=-1, n=self.n)
        acc = dd.R(xs[0] @ self.sl[0])                     # level 0, exact
        if NSLICE > 1:
            s, e = dd.two_sum(xs[0] @ self.sl[1], xs[1] @ self.sl[0])   # level 1 (about 2^-22 of level 0), exact
            acc = dd.add(acc, dd.R(s, e))
        rest = None                                          # levels >= 2 (2^-44 and below), summed in float64:
        for p in range(NSLICE):                              # their rounding is about 2^-97 of the result
            for q in range(max(0, 2 - p), NSLICE - p):
                P = xs[p] @ self.sl[q]
                rest = P if rest is None else rest + P
        if rest is not None:
            acc = dd.add_d(acc, rest)
        if MUTATE == "hionly":
            acc = dd.R(acc.h, np.zeros_like(acc.h))
        return dd.R(acc.h.reshape(shp[:-1] + (-1,)), acc.l.reshape(shp[:-1] + (-1,)))


# ---------------------------------------------------------------- dual double-double numbers
def _R(x):
    return x if isinstance(x, dd.R) else dd.R(np.asarray(x, dtype=float))


class DT:
    """value v (dd.R, broadcast over the member axis) and tangents t (dd.R with one row per input direction, or None)"""
    __slots__ = ("v", "t")

    def __init__(self, v, t=None):
        self.v = v
        self.t = t

    def __add__(self, o):
        if isinstance(o, DT):
            t = self.t if o.t is None else (o.t if self.t is None else dd.add(self.t, o.t))
            return DT(dd.add(self.v, o.v), t)
        return DT(dd.add_d(self.v, np.asarray(o, dtype=float)), self.t)
    __radd__ = __add__

    def __neg__(self):
        return DT(dd.neg(self.v), None if self.t is None else dd.neg(self.t))

    def __sub__(self, o):
        return self + (-o if isinstance(o, DT) else -np.asarray(o, dtype=float))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, DT):
            v = dd.mul(self.v, o.v)
            ts = []
            if o.t is not None:
                ts.append(dd.mul(self.v, o.t))
            if self.t is not None:
                ts.append(dd.mul(self.t, o.v))
            t = None if not ts else (ts[0] if len(ts) == 1 else dd.add(ts[0], ts[1]))
            return DT(v, t)
        if isinstance(o, dd.R):
            return DT(dd.mul(self.v, o), None if self.t is None else dd.mul(self.t, o))
        o = np.asarray(o, dtype=float)
        return DT(dd.mul_d(self.v, o), None if self.t is None else dd.mul_d(self.t, o))
    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, DT):
            q = dd.div(self.v, o.v)
            t = self.t
            if o.t is not None:
                qt = dd.mul(q, o.t)
                t = dd.neg(qt) if t is None else dd.sub(t, qt)
            return DT(q, None if t is None else dd.div(t, _bcast(o.v, t)))
        o = _R(o)
        return DT(dd.div(self.v, _bcast(o, self.v)), None if self.t is None else dd.div(self.t, _bcast(o, self.t)))

    def __rtruediv__(self, o):
        q = dd.div(_bcast(_R(o), self.v), self.v)
        return DT(q, None if self.t is None else dd.neg(dd.div(dd.mul(q, self.t), _bcast(self.v, self.t))))

    def lin(self, f):
        """a linear map f applied to value and tangents"""
        return DT(f(self.v), None if self.t is None else f(self.t))


def _bcast(b, a):
    """b broadcast to a's shape (ddnp.div reads b.h elementwise)"""
    return dd.R(np.broadcast_to(b.h, np.broadcast_shapes(b.h.shape, a.h.shape)).copy(),
                np.broadcast_to(b.l, np.broadcast_shapes(b.l.shape, a.l.shape)).copy())


def _cs(X, Y, k, j0=0):
    s = 0.0
    for j in range(j0, k + 1):
        s = s + X[j] * Y[k - j]
    return s


def _eval_series(Sx, x):
    out = 0.0
    for k in range(len(Sx) - 1, -1, -1):
        out = out * x + Sx[k]
    return out


class DDEll:
    """DSSEll.mode_matrix in double-double for a DSSEll S whose RK step counts are frozen (S.freeze_from)."""
    PAR = ("odd", "odd", "even", "even", "odd")

    def __init__(self, S):
        if S.ell < 1:
            raise ValueError("ell_ddm covers l >= 1")
        self.S = S
        ops = tau_operators(S.N, S.kodd, S.keven)
        self.B = {"odd": Op(ops["Bodd"]), "even": Op(ops["Beven"])}
        self.Sp = {"odd": Op(ops["Sodd"]), "even": Op(ops["Seven"])}
        self.Dm = Op(ops["Dm"])
        self.Dm_h = ops["Dm"].h
        mo, me = len(S.kodd), len(S.keven) - 1
        self.kvec = {"odd": np.asarray(S.kodd, float), "even": np.asarray(S.keven[1:], float)}
        self.m = {"odd": mo, "even": me}
        self.om = None
        self.lam = 0.0

    # ---- tau maps on dual numbers ----
    def phys(self, c, par):
        return c.lin(self.B[par].rmul)

    def spec(self, f, par):
        return f.lin(self.Sp[par].rmul)

    def _dcoef(self, X, par):
        """d/dtheta of coefficient arrays (..., n_par): (re, im) of harmonic k -> (-k im, k re); the mean -> 0"""
        k, m = self.kvec[par], self.m[par]
        o = 0 if par == "odd" else 1
        re = dd.R(X.h[..., o:o + m], X.l[..., o:o + m]); im = dd.R(X.h[..., o + m:], X.l[..., o + m:])
        nre = dd.mul_d(dd.neg(im), k); nim = dd.mul_d(re, k)
        parts_h = ([np.zeros_like(X.h[..., :1])] if o else []) + [nre.h, nim.h]
        parts_l = ([np.zeros_like(X.l[..., :1])] if o else []) + [nre.l, nim.l]
        return dd.R(np.concatenate(parts_h, axis=-1), np.concatenate(parts_l, axis=-1))

    def Dspec(self, c, par):
        """omega d/dtau on coefficients, + lambda on the tangents (perturbations ~ exp(lambda tau))"""
        v = dd.mul_d(self._dcoef(c.v, par), self.om)
        t = None
        if c.t is not None:
            t = dd.mul_d(self._dcoef(c.t, par), self.om)
            if self.lam != 0.0:
                t = dd.add(t, dd.mul_d(c.t, self.lam))
        return DT(v, t)

    def Dphys(self, f):
        """omega d/dtau on grid values (f @ Dm, DSSEll._D_unit), + lambda on the tangents"""
        v = dd.mul_d(self.Dm.rmul(f.v), self.om)
        t = None
        if f.t is not None:
            t = dd.mul_d(self.Dm.rmul(f.t), self.om)
            if self.lam != 0.0:
                t = dd.add(t, dd.mul_d(f.t, self.lam))
        return DT(v, t)

    # ---- the x-ODE in coefficients ----
    def rhs(self, x, s):
        S = self.S
        n, L = S.n, S.L
        p, m, a, c, q = (self.phys(s[f], self.PAR[f]) for f in range(5))
        Dp = self.phys(self.Dspec(s[0], "odd"), "odd")
        Dmm = self.phys(self.Dspec(s[1], "odd"), "odd")
        a2 = a * a
        g = a2 * q * q
        cx = c * ((n - 1) * (a2 - 1.0) - (2 * L) * g) / x
        ax = a * ((p * p + m * m) + (2 * L) * g + (n - 1) * (1.0 - a2)) / (2 * x)
        qx = (p + m) / (2 * x)
        pc = n * c * (p + m) / (2 * x)
        lt = L * c * a2 * q / x
        px = (-Dp + cx * p - c * p / x + pc - lt) / (x - c)
        mx = (-Dmm - cx * m + c * m / x - pc + lt) / (x + c)
        return [self.spec(px, "odd"), self.spec(mx, "odd"), self.spec(ax, "even"), self.spec(cx, "even"),
                self.spec(qx, "odd")]

    @staticmethod
    def _axpy(s, k, a):
        return [si + ki * a for si, ki in zip(s, k)]

    def _rk4(self, s, y0, h, steps, f):
        """RK4 in a variable y with f(y, s) the derivative; y advances in float64 as in DSSEll"""
        h6 = dd.div(dd.R(np.array(h)), dd.R(np.array(6.0)))
        y = y0
        for _ in range(steps):
            k1 = f(y, s)
            k2 = f(y + h / 2, self._axpy(s, k1, h / 2))
            k3 = f(y + h / 2, self._axpy(s, k2, h / 2))
            k4 = f(y + h, self._axpy(s, k3, h))
            inc = [a + b * 2.0 + c * 2.0 + d for a, b, c, d in zip(k1, k2, k3, k4)]
            s = [si + ii * h6 for si, ii in zip(s, inc)]
            y += h
        return s

    def integrate(self, s, xa, xb, steps):
        return self._rk4(s, xa, (xb - xa) / steps, steps, lambda x, st: self.rhs(x, st))

    def integrate_u(self, s, xa, xb, steps):
        ya, yb = np.log(1 - xa), np.log(1 - xb)

        def f(y, st):
            e = float(np.exp(y))
            return [r * (-e) for r in self.rhs(1 - e, st)]
        return self._rk4(s, ya, (yb - ya) / steps, steps, f)

    def integrate_log(self, s, xa, xb, steps):
        ya = np.log(xa)

        def f(y, st):
            e = float(np.exp(y))
            return [r * e for r in self.rhs(e, st)]
        return self._rk4(s, ya, (np.log(xb) - np.log(xa)) / steps, steps, f)

    def seg(self, s, xa, xb):
        k = self.S._steps(xa, xb)
        if k is None:
            raise ValueError("ell_ddm needs frozen RK step counts (S.freeze_from)")
        if min(xa, xb) >= 0.4:
            return self.integrate_u(s, xa, xb, k)
        return self.integrate(s, xa, xb, k)

    # ---- series ----
    def _zero(self):
        return DT(dd.R(np.zeros((1, self.S.N))))

    def centre_series(self, d, c0):
        """DSSEll.centre_series (l >= 1) on dual double-double grid values"""
        S = self.S
        n, L, l, K = S.n, S.L, S.ell, S.Kc
        Z = self._zero()
        P = [Z] * (K + 1); M = [Z] * (K + 1); A = [Z] * (K + 1); C = [Z] * (K + 1); Q = [Z] * (K + 1)
        A[0] = DT(dd.R(np.ones((1, S.N)))); C[0] = c0
        P[l] = d * float(l); M[l] = d * float(l); Q[l] = d
        cs = _cs
        for k in range(1, K + 1):
            Q2 = [cs(P, P, j) + cs(M, M, j) for j in range(k + 1)]
            QQ = [cs(Q, Q, j) for j in range(k + 1)]
            s = cs(A, Q2, k)
            A2 = [cs(A, A, j) for j in range(k + 1)]
            t3 = cs(A, A2, k)
            A2Q = [cs(A2, QQ, j) for j in range(k + 1)]
            u = cs(A, A2Q, k)
            A[k] = (s * 0.5 + u * float(L) - t3 * (0.5 * (n - 1))) / float(k + n - 1)
            A2 = [cs(A, A, j) for j in range(k + 1)]
            A2m1 = [A2[j] - (1.0 if j == 0 else 0.0) for j in range(k + 1)]
            G = [cs(A2, QQ, j) for j in range(k + 1)]
            C[k] = (cs(C, A2m1, k) * float(n - 1) - cs(C, G, k) * float(2 * L)) / float(k)
            if k <= l:
                continue
            PM = [P[j] + M[j] for j in range(K + 1)]
            Dp = self.Dphys(P[k - 1]); Dmm = self.Dphys(M[k - 1])
            X = [Z, -C[0]] + [(1.0 if j == 2 else 0.0) - C[j - 1] for j in range(2, k + 2)]
            Y = [Z, C[0]] + [(1.0 if j == 2 else 0.0) + C[j - 1] for j in range(2, k + 2)]
            xpx = sum(X[j] * float(k - j + 1) * P[k - j + 1] for j in range(1, k + 1))
            xmx = sum(Y[j] * float(k - j + 1) * M[k - j + 1] for j in range(1, k + 1))
            xcxp = sum(C[j] * float(j) * P[k - j] for j in range(1, k + 1))
            xcxm = sum(C[j] * float(j) * M[k - j] for j in range(1, k + 1))
            cp = cs(C, P, k); cm = cs(C, M, k); cpm = cs(C, PM, k)
            H = [cs(A2, Q, j) for j in range(k + 1)]
            ch = cs(C, H, k)
            r3 = Dp + xpx - xcxp + cp - cpm * (0.5 * n) + ch * float(L)
            r4 = Dmm + xmx + xcxm - cm + cpm * (0.5 * n) - ch * float(L)
            c = C[0]
            al = 0.5 * n - L / (2.0 * k)
            m11 = c * (1 - k - al); m12 = c * (-al)
            m21 = c * al;           m22 = c * (k - 1 + al)
            det = m11 * m22 - m12 * m21
            P[k] = (-r3 * m22 + r4 * m12) / det
            M[k] = (-r4 * m11 + r3 * m21) / det
            Q[k] = (P[k] + M[k]) / (2.0 * k)
        return P, M, A, C, Q

    def _solve(self, kap, lam, rhs):
        """(omega d/dtau + kappa + lam) f = rhs on the grid, rows of rhs (dd.R (B, N)); kap a dd.R (1, N): a float64 LU
        refined with double-double residuals"""
        import scipy.linalg as sla
        N = self.S.N
        A = self.om * self.Dm_h.T + np.diag(kap.h[0] + lam)
        lu = sla.lu_factor(A)
        f = dd.R(sla.lu_solve(lu, rhs.h.T).T)
        for _ in range(8):
            Af = dd.add(dd.mul_d(self.Dm.rmul(f), self.om), dd.mul(_bcast(kap, f), f))
            if lam != 0.0:
                Af = dd.add(Af, dd.mul_d(f, lam))
            r = dd.sub(rhs, Af)
            df = sla.lu_solve(lu, r.h.T).T
            f = dd.add_d(f, df) if False else dd.add(f, dd.R(df))
            if np.max(np.abs(df)) <= 1e-33 * max(np.max(np.abs(f.h)), 1e-300):
                break
        return f

    def solve_periodic(self, kappa, rhs):
        """DSSEll.solve_periodic on dual numbers: value with kappa's value; tangents with + lam and the kappa tangent"""
        fv = self._solve(kappa.v, 0.0, rhs.v)
        ft = None
        if rhs.t is not None or kappa.t is not None:
            rt = rhs.t if rhs.t is not None else dd.R(np.zeros((kappa.t.shape[0], self.S.N)))
            if kappa.t is not None:
                rt = dd.sub(rt, dd.mul(kappa.t, _bcast(fv, kappa.t)))
            ft = self._solve(kappa.v, self.lam, rt)
        return DT(fv, ft)

    def ssh_series(self, m0, a0, q0):
        """DSSEll.ssh_series on dual double-double grid values (orders 0..Ks)"""
        S = self.S
        n, L, K = S.n, S.L, S.Ks
        Z = self._zero()
        one = DT(dd.R(np.ones((1, S.N))))
        P = [Z] * (K + 2); M = [Z] * (K + 2); A = [Z] * (K + 2); C = [Z] * (K + 2); Q = [Z] * (K + 2)
        A2 = [Z] * (K + 2); Q2 = [Z] * (K + 2); QQ = [Z] * (K + 2); G = [Z] * (K + 2); H = [Z] * (K + 2)
        DP = [Z] * (K + 2); DM = [Z] * (K + 2)
        cs = _cs
        A[0] = a0; C[0] = one; M[0] = m0; Q[0] = q0
        A2[0] = A[0] * A[0]; QQ[0] = Q[0] * Q[0]; G[0] = A2[0] * QQ[0]; H[0] = A2[0] * Q[0]
        DM[0] = self.Dphys(M[0])
        for Kk in range(K + 1):
            C[Kk + 1] = (C[Kk] * float(Kk) - (cs(C, A2, Kk) - C[Kk]) * float(n - 1) + cs(C, G, Kk) * float(2 * L)) / float(Kk + 1)
            w = [-C[i] for i in range(Kk + 2)]
            w[0] = w[0] + 1.0; w[1] = w[1] - 1.0
            W = [w[0]] + [w[i] - w[i - 1] for i in range(1, Kk + 2)]
            V = [C[i + 1] * float(i + 1) - C[i] * float(i) for i in range(Kk + 1)]
            PM = [P[i] + M[i] for i in range(Kk + 1)]
            tau_term = -DP[Kk - 1] if Kk >= 1 else Z
            wpu = sum(P[i] * float(i) * W[Kk + 1 - i] for i in range(Kk + 1))
            r = tau_term - wpu + cs(V, P, Kk) + cs(C, P, Kk) - cs(C, PM, Kk) * (0.5 * n) + cs(C, H, Kk) * float(L)
            kappa = (1.0 + C[1]) * float(Kk) + C[1] + (1.0 - 0.5 * n)
            P[Kk] = self.solve_periodic(kappa, -r)
            DP[Kk] = self.Dphys(P[Kk])
            Q2[Kk] = cs(P, P, Kk) + cs(M, M, Kk)
            if Kk == K:
                break
            aG = cs(A, Q2, Kk) + (A[Kk] - cs(A, A2, Kk)) * float(n - 1) + cs(A, G, Kk) * float(2 * L)
            A[Kk + 1] = (A[Kk] * float(Kk) - aG * 0.5) / float(Kk + 1)
            A2[Kk + 1] = cs(A, A, Kk + 1)
            zc = [C[i] for i in range(Kk + 2)]
            zc[0] = zc[0] + 1.0; zc[1] = zc[1] - 1.0
            Zc = [zc[0]] + [zc[i] - zc[i - 1] for i in range(1, Kk + 2)]
            tau_term = DM[Kk] - (DM[Kk - 1] if Kk >= 1 else Z)
            PM = [P[i] + M[i] for i in range(Kk + 1)]
            zmu = sum(M[i] * float(i) * Zc[Kk + 1 - i] for i in range(Kk + 1))
            rest = tau_term - zmu - cs(V, M, Kk) - cs(C, M, Kk) + cs(C, PM, Kk) * (0.5 * n) - cs(C, H, Kk) * float(L)
            M[Kk + 1] = rest / (Zc[0] * float(Kk + 1))
            DM[Kk + 1] = self.Dphys(M[Kk + 1])
            Q[Kk + 1] = (Q[Kk] * float(Kk) - (P[Kk] + M[Kk]) * 0.5) / float(Kk + 1)
            QQ[Kk + 1] = cs(Q, Q, Kk + 1)
            G[Kk + 1] = cs(A2, QQ, Kk + 1)
            H[Kk + 1] = cs(A2, Q, Kk + 1)
        return P[:K + 1], M[:K + 1], A[:K + 1], C[:K + 1], Q[:K + 1]

    # ---- blocks ----
    def _inputs(self, X, sizes, pars):
        """dual numbers for a block's input vector X (float64): value X, tangents the identity columns"""
        nf = len(X)
        out, o = [], 0
        for sz in sizes:
            v = dd.R(X[None, o:o + sz].copy())
            t = np.zeros((nf, sz)); t[np.arange(o, o + sz), np.arange(sz)] = 1.0
            out.append(DT(v, dd.R(t)))
            o += sz
        return out

    def block(self, b, X):
        S = self.S
        no, ne = S.n_odd, S.n_even
        K = len(S.nodes)
        if b == 0:
            d, c0 = self._inputs(X, [no, ne], ["odd", "even"])
            Sx = self.centre_series(self.phys(d, "odd"), self.phys(c0, "even"))
            s = [self.spec(_eval_series(Sf, S.x0), self.PAR[f]) for f, Sf in enumerate(Sx)]
            return self.integrate_log(s, S.x0, S.nodes[0], S.steps_map["log"])
        if b < K:
            s = self._inputs(X, [no, no, ne, ne, no], self.PAR)
            return self.seg(s, S.nodes[b - 1], S.nodes[b])
        mS, aS, qS = self._inputs(X, [no, ne, no], ["odd", "even", "odd"])
        Sx = self.ssh_series(self.phys(mS, "odd"), self.phys(aS, "even"), self.phys(qS, "odd"))
        s = [self.spec(_eval_series(Sf, S.u0), self.PAR[f]) for f, Sf in enumerate(Sx)]
        if S.long:
            return s
        return self.seg(s, 1 - S.u0, S.nodes[-1])

    def mode_matrix(self, z, lam, only=None, log=None):
        """M(lambda) in double-double (dd.R), the layout of DSSEll.mode_matrix; only: block indices to build (the others
        are left zero, for block tests)"""
        S = self.S
        sizes, K, L = S.layout()
        nv = L - 1
        Mh = np.zeros((nv, nv)); Ml = np.zeros((nv, nv))
        blocks, offs = S._block_list()
        blk = 3 * S.n_odd + 2 * S.n_even
        node0 = offs[2]
        self.om = 2 * np.pi / z[-1]
        self.lam = 0.0 if MUTATE == "lam" else float(lam)
        import time
        try:
            for b, (c0, nf, fun, tgt) in enumerate(blocks):
                if only is not None and b not in only:
                    continue
                t0 = time.time()
                out = self.block(b, z[c0:c0 + nf])
                Rh = np.concatenate([o.t.h for o in out], axis=1); Rl = np.concatenate([o.t.l for o in out], axis=1)
                r0 = b * blk
                Mh[r0:r0 + blk, c0:c0 + nf] = Rh.T; Ml[r0:r0 + blk, c0:c0 + nf] = Rl.T
                t0_ = node0 + blk * tgt
                Mh[r0:r0 + blk, t0_:t0_ + blk] -= np.eye(blk)
                if log:
                    log(f"  block {b}: {time.time() - t0:.0f} s")
        finally:
            self.lam = 0.0
        return dd.R(Mh, Ml)


def selftest():
    sys.path.insert(0, HERE)
    from dss_ell import DSSEll, NODES_STD
    ok = True
    S = DSSEll(kc=15, nodes=NODES_STD, n=11.0, ell=1, Delta_ref=1.0, Ks=30, u0=0.04)
    ops = tau_operators(S.N, S.kodd, S.keven)
    chk = dict(Bodd=S._B["odd"], Beven=S._B["even"], Sodd=S._S["odd"], Seven=S._S["even"], Dm=S._Dm)
    errs = {k: float(np.max(np.abs(ops[k].h + ops[k].l - v)) / max(1.0, np.max(np.abs(v)))) for k, v in chk.items()}
    good = max(errs.values()) < 1e-14
    print(f"selftest operators vs DSSEll's float64 ones (k_c 15, N {S.N}): " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items())
          + f" (bar 1e-14): {'OK' if good else 'BAD'}", flush=True)
    ok &= good
    g = np.random.default_rng(1)
    X = dd.R(g.standard_normal((7, S.N)) * np.exp2(g.integers(-30, 30, (7, 1))))
    X = dd.add(X, dd.R(X.h * 1e-17 * g.standard_normal(X.shape)))
    ref = dd.matmul(X, ops["Dm"])
    got = Op(ops["Dm"]).rmul(X)
    rel = float(np.max(np.abs((got.h - ref.h) + (got.l - ref.l)) / np.max(np.abs(ref.h), axis=1, keepdims=True)))
    good = rel < 1e-28
    print(f"selftest Ozaki product ({NSLICE} slices, levels >= 2 summed in float64) vs ddnp.matmul: largest error {rel:.1e} "
          f"of the row's largest entry (bar 1e-28): {'OK' if good else 'BAD'}", flush=True)
    ok &= good
    # dual numbers: f(u) = u^3 / (1 + u) - 2 u; its tangent against f'(u) t = ((2u^3 + 3u^2) / (1 + u)^2 - 2) t in dd
    u0 = dd.R(g.uniform(0.2, 2.0, (1, 5)))
    tt = dd.R(g.standard_normal((3, 5)))
    f = lambda u: (u * u * u) / (1.0 + u) - u * 2.0
    D = f(DT(u0, tt))
    u2 = dd.mul(u0, u0); u3 = dd.mul(u2, u0); w = dd.add_d(u0, 1.0)
    fp = dd.add_d(dd.div(dd.add(dd.mul_d(u3, 2.0), dd.mul_d(u2, 3.0)), dd.mul(w, w)), -2.0)
    ex = dd.mul(_bcast(fp, tt), tt)
    rel = float(np.max(np.abs((D.t.h - ex.h) + (D.t.l - ex.l))) / np.max(np.abs(ex.h)))
    good = rel < 1e-28
    print(f"selftest dual derivative vs the analytic one in double-double: {rel:.1e} (bar 1e-28): {'OK' if good else 'BAD'}",
          flush=True)
    ok &= good
    print(f"selftest: {'PASS' if ok else 'FAIL'}" + (f" [mutate {MUTATE}]" if MUTATE else ""))
    return 0 if ok else 1


def check(argv):
    """check A: the double-double blocks of M against DSSEll.mode_matrix (float64, complex step), entrywise relative
    to the block's largest entry"""
    import argparse, json, time
    from dss_ell import DSSEll, NODES_STD
    ap = argparse.ArgumentParser()
    ap.add_argument("state"); ap.add_argument("D", type=float); ap.add_argument("kc", type=int); ap.add_argument("lam", type=float)
    ap.add_argument("--blocks", default=None); ap.add_argument("--bar", type=float, default=1e-12)
    ap.add_argument("--Ks", type=int, default=30); ap.add_argument("--u0", type=float, default=0.04)   # the horizon series
    ap.add_argument("--out", default=None)
    o = ap.parse_args([a for a in argv if not a.startswith("--mutate=")])
    z = np.load(o.state)
    S = DSSEll(kc=o.kc, nodes=NODES_STD, n=o.D - 2, ell=1, Delta_ref=z[-1], Ks=o.Ks, u0=o.u0)
    S.freeze_from(z, 0.7)
    t0 = time.time()
    with np.errstate(all="ignore"):
        Mf = S.mode_matrix(z, o.lam)
    tf = time.time() - t0
    E = DDEll(S)
    only = None if o.blocks is None else [int(v) for v in o.blocks.split(",")]
    t0 = time.time()
    Md = E.mode_matrix(z, o.lam, only=only, log=lambda m: print(m, flush=True))
    td = time.time() - t0
    blocks, offs = S._block_list()
    blk = 3 * S.n_odd + 2 * S.n_even
    worst, rows = 0.0, []
    for b, (c0, nf, fun, tgt) in enumerate(blocks):
        if only is not None and b not in only:
            continue
        r0 = b * blk
        F = Mf[r0:r0 + blk, c0:c0 + nf]
        G = Md.h[r0:r0 + blk, c0:c0 + nf] + Md.l[r0:r0 + blk, c0:c0 + nf]
        e = float(np.max(np.abs(G - F)) / np.max(np.abs(F)))
        worst = max(worst, e)
        rows.append(dict(block=b, rel=e, max_entry=float(np.max(np.abs(F)))))
        print(f"  block {b}: max |M_dd - M_f64| / max |M| = {e:.2e}", flush=True)
    good = worst < o.bar
    rec = dict(D=o.D, kc=o.kc, lam=o.lam, state=os.path.basename(o.state), mutate=MUTATE or None, blocks=rows, worst=worst,
               bar=o.bar, sec_f64=round(tf), sec_dd=round(td), verdict="PASS" if good else "FAIL")
    print(f"check A D {o.D} k_c {o.kc} lambda {o.lam}: worst {worst:.2e} (bar {o.bar:g}) -> {rec['verdict']}; float64 "
          f"{tf:.0f} s, double-double {td:.0f} s" + (f" [mutate {MUTATE}]" if MUTATE else ""), flush=True)
    if o.out:
        open(o.out, "a").write(json.dumps(rec) + "\n")
    return 0 if good else 1


def sshres(argv):
    """check A2: the double-double horizon series (value and tangents) against the x-ODE it solves, the rhs transcribed
    again and evaluated in mpmath (60 digits; tangents by a complex step of 1e-40), untruncated (the series is formed
    on the N-point grid), at u in --us"""
    import argparse, json
    from dss_ell import DSSEll, NODES_STD
    ap = argparse.ArgumentParser()
    ap.add_argument("state"); ap.add_argument("D", type=float); ap.add_argument("kc", type=int); ap.add_argument("lam", type=float)
    ap.add_argument("--us", default="0.04,0.02"); ap.add_argument("--bar", type=float, default=1e-18)
    ap.add_argument("--Ks", type=int, default=30); ap.add_argument("--u0", type=float, default=0.04)   # the horizon series
    ap.add_argument("--ncols", type=int, default=4); ap.add_argument("--out", default=None)
    o = ap.parse_args([a for a in argv if not a.startswith("--mutate=")])
    z = np.load(o.state)
    S = DSSEll(kc=o.kc, nodes=NODES_STD, n=o.D - 2, ell=1, Delta_ref=z[-1], Ks=o.Ks, u0=o.u0)
    S.freeze_from(z, 0.7)
    E = DDEll(S)
    blocks, offs = S._block_list()
    c0, nf, fun, tgt = blocks[-1]
    no, ne = S.n_odd, S.n_even
    X = z[c0:c0 + nf]
    om = 2 * np.pi / z[-1]
    E.om = om; E.lam = 0.0 if MUTATE == "lam" else o.lam
    d = E._inputs(X, [no, ne, no], ["odd", "even", "odd"])
    cols = np.linspace(0, nf - 1, o.ncols).astype(int)           # a sample of tangent directions
    d = [DT(x.v, dd.R(x.t.h[cols], x.t.l[cols])) for x in d]
    Sx = E.ssh_series(E.phys(d[0], "odd"), E.phys(d[1], "even"), E.phys(d[2], "odd"))
    K = len(Sx[0]) - 1
    import mpmath as mp
    mp.mp.dps = 60
    n, L, N = S.n, S.L, S.N
    hc = mp.mpf("1e-40")
    dvec = [-(mp.mpf(2) / N) * mp.fsum(k * mp.sin(2 * mp.pi * k * r / N) for k in range(1, N // 2)) for r in range(N)]
    omm, lamm = mp.mpf(om), mp.mpf(o.lam)             # the ODE always carries lambda; the mutation drops it from the series
    toh = lambda R_, i_, j_: mp.mpf(float(R_.h[i_, j_])) + mp.mpf(float(R_.l[i_, j_]))
    worst, rows = 0.0, []
    for u in (float(v) for v in o.us.split(",")):
        um, x = mp.mpf(u), 1 - mp.mpf(u)
        def ser(f, deriv, member):
            out = []
            for j in range(N):
                acc = mp.mpf(0)
                for k in range(K + 1):
                    e = Sx[f][k]
                    if member < 0:
                        c_ = toh(e.v, 0, j)
                    else:
                        c_ = toh(e.t, member, j) if e.t is not None else mp.mpf(0)
                    acc += (k * c_ * um ** (k - 1) if k else 0) if deriv else c_ * um ** k
                out.append(-acc if deriv else acc)
            return out
        nc = len(cols)
        V = [ser(f, False, -1) for f in range(5)]; dV = [ser(f, True, -1) for f in range(5)]
        for f in range(5):
            pass
        res_v = [0.0] * 5; res_t = [0.0] * 5; scl_v = [0.0] * 5; scl_t = [0.0] * 5
        for b in range(-1, nc):
            if b < 0:
                st = V; dst = dV
            else:
                T = [ser(f, False, b) for f in range(5)]; dT = [ser(f, True, b) for f in range(5)]
                st = [[V[f][j] + 1j * hc * T[f][j] for j in range(N)] for f in range(5)]
                dst = dT
            p, m, a, c, q = st
            def Dl(fv, f0):
                d_ = [omm * mp.fsum(fv[i_] * dvec[(j - i_) % N] for i_ in range(N)) for j in range(N)]
                if b >= 0:
                    d_ = [d_[j] + lamm * (fv[j] - f0[j]) for j in range(N)]
                return d_
            Dp, Dmm = Dl(p, V[0]), Dl(m, V[1])
            r = [[None] * N for _ in range(5)]
            for j in range(N):
                a2 = a[j] * a[j]; g = a2 * q[j] * q[j]
                cx = c[j] * ((n - 1) * (a2 - 1) - 2 * L * g) / x
                ax = a[j] * ((p[j] * p[j] + m[j] * m[j]) + 2 * L * g + (n - 1) * (1 - a2)) / (2 * x)
                qx = (p[j] + m[j]) / (2 * x); pc = n * c[j] * (p[j] + m[j]) / (2 * x); lt = L * c[j] * a2 * q[j] / x
                px = (-Dp[j] + cx * p[j] - c[j] * p[j] / x + pc - lt) / (x - c[j])
                mx = (-Dmm[j] - cx * m[j] + c[j] * m[j] / x - pc + lt) / (x + c[j])
                r[0][j], r[1][j], r[2][j], r[3][j], r[4][j] = px, mx, ax, cx, qx
            for f in range(5):
                if b < 0:
                    res_v[f] = max(float(abs(r[f][j] - dst[f][j])) for j in range(N))
                    scl_v[f] = max(float(abs(r[f][j])) for j in range(N))
                else:
                    res_t[f] = max(res_t[f], max(float(abs(mp.im(r[f][j]) / hc - dst[f][j])) for j in range(N)))
                    scl_t[f] = max(scl_t[f], max(float(abs(mp.im(r[f][j]) / hc)) for j in range(N)))
        for f, nm in enumerate("pmacq"):
            ev, et = res_v[f] / scl_v[f], res_t[f] / max(scl_t[f], 1e-300)
            worst = max(worst, ev, et)
            rows.append(dict(u=u, field=nm, value=ev, tangent=et))
            print(f"  u {u}: {nm}' residual (mpmath, 60 digits), value {ev:.1e}, tangents {et:.1e}", flush=True)
    good = worst < o.bar
    print(f"check A2 D {o.D} k_c {o.kc} lambda {o.lam}: horizon series against the x-ODE, worst {worst:.1e} (bar {o.bar:g}) "
          f"-> {'PASS' if good else 'FAIL'}" + (f" [mutate {MUTATE}]" if MUTATE else ""), flush=True)
    if o.out:
        open(o.out, "a").write(json.dumps(dict(D=o.D, kc=o.kc, lam=o.lam, mutate=MUTATE or None, rows=rows, worst=worst,
                                               verdict="PASS" if good else "FAIL")) + "\n")
    return 0 if good else 1


def _setup(state, D, kc):
    from dss_ell import DSSEll, NODES_STD
    z = np.load(state)
    S = DSSEll(kc=kc, nodes=NODES_STD, n=D - 2, ell=1, Delta_ref=z[-1], Ks=30, u0=0.04)
    S.freeze_from(z, 0.7)
    return S, z


def det_dd(E, z, lam, nvar=5, log=None, noise_eps=None, seeds=2):
    """sign and ln|det| of the double-double M(lambda) by ell_dd.lu_dd over the exact variants (dd entries kept)"""
    import time
    from ell_dd import lu_dd, bandwidths
    t0 = time.time()
    M = E.mode_matrix(z, lam)
    tb = time.time() - t0
    lb, ub = bandwidths(M.h)
    out = []
    g = np.random.default_rng(7)
    for i in range(nvar):
        if i == 0:
            A, a, b, lsc = M.copy(), lb, ub, 0.0
        elif i == 1:
            A, a, b, lsc = M.T.copy(), ub, lb, 0.0
        else:
            er = g.integers(-8, 9, M.shape[0]); ec = g.integers(-8, 9, M.shape[1])
            A = dd.R(np.ldexp(np.ldexp(M.h, er[:, None]), ec[None, :]), np.ldexp(np.ldexp(M.l, er[:, None]), ec[None, :]))
            a, b, lsc = lb, ub, float(er.sum() + ec.sum()) * math.log(2.0)
        t1 = time.time()
        s, l = lu_dd(A, a, b)
        out.append(dict(sign=s, ln=l - lsc, sec=round(time.time() - t1, 1)))
    if log:
        log(f"  lambda {lam!r}: dd M built in {tb:.0f} s; dd LU signs {[r['sign'] for r in out]}, ln|det| {out[0]['ln']:.12f} "
            f"(spread over variants {max(r['ln'] for r in out) - min(r['ln'] for r in out):.1e}), {out[0]['sec']} s per LU")
    noise = []
    for eps in (noise_eps or []):
        S = E.S
        blocks, offs = S._block_list()
        blk = 3 * S.n_odd + 2 * S.n_even
        mask = np.zeros(M.shape, bool)
        for b_, (c0, nf, fun, tgt) in enumerate(blocks):
            mask[b_ * blk:(b_ + 1) * blk, c0:c0 + nf] = True
        sg = []
        for sd in range(seeds):
            xi = np.random.default_rng(100 + sd).standard_normal(M.shape) * mask
            A = dd.add(M, dd.mul_d(M, eps * xi))
            sg.append(lu_dd(A, lb, ub)[0])
        noise.append(dict(eps=eps, signs=sg))
        if log:
            log(f"  lambda {lam!r}: relative noise {eps:g} on the dd propagator entries: dd signs {sg}")
    if noise_eps is not None:
        return out, noise
    return out


def detcmd(argv):
    """check B1: the double-double det sign at the given lambda over 5 variants, beside the float64 M's LAPACK sign"""
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("state"); ap.add_argument("D", type=float); ap.add_argument("kc", type=int); ap.add_argument("lams")
    ap.add_argument("--out", default=None); ap.add_argument("--noise", default=None)
    o = ap.parse_args([a for a in argv if not a.startswith("--mutate=")])
    S, z = _setup(o.state, o.D, o.kc)
    E = DDEll(S)
    recs = []
    neps = [float(v) for v in o.noise.split(",")] if o.noise else None
    for lam in (float(v) for v in o.lams.split(",")):
        r = det_dd(E, z, lam, log=lambda m: print(m, flush=True), noise_eps=neps)
        noise = None
        if neps is not None:
            r, noise = r
        with np.errstate(all="ignore"):
            sf, lf = np.linalg.slogdet(S.mode_matrix(z, lam))
        rec = dict(D=o.D, kc=o.kc, lam=lam, mutate=MUTATE or None, dd=r, dd_agree=len({x["sign"] for x in r}) == 1,
                   f64_sign=float(sf), f64_ln=float(lf), noise=noise,
                   noise_robust=None if noise is None else all(s == r[0]["sign"] for e in noise for s in e["signs"]))
        print(f"  lambda {lam!r}: dd signs agree {rec['dd_agree']}, dd {r[0]['sign']:+.0f} vs float64 {sf:+.0f}; ln|det| dd "
              f"{r[0]['ln']:.10f}, float64 {lf:.10f}", flush=True)
        recs.append(rec)
        if o.out:
            open(o.out, "a").write(json.dumps(rec) + "\n")
    return 0


def rootcmd(argv):
    """check B2: lambda_1 as the sign change of the double-double det M between lo and hi (brentq on sign * exp(ln - ref)),
    against a record"""
    import argparse, json, time
    from scipy.optimize import brentq
    from ell_dd import lu_dd, bandwidths
    ap = argparse.ArgumentParser()
    ap.add_argument("state"); ap.add_argument("D", type=float); ap.add_argument("kc", type=int)
    ap.add_argument("lo", type=float); ap.add_argument("hi", type=float); ap.add_argument("--record", type=float, default=None)
    ap.add_argument("--bar", type=float, default=1e-7); ap.add_argument("--xtol", type=float, default=1e-10)
    ap.add_argument("--out", default=None)
    o = ap.parse_args([a for a in argv if not a.startswith("--mutate=")])
    S, z = _setup(o.state, o.D, o.kc)
    E = DDEll(S)
    t0 = time.time()
    cache = {}

    def sd(lam):
        if lam not in cache:
            M = E.mode_matrix(z, lam)
            lb, ub = bandwidths(M.h)
            cache[lam] = lu_dd(M, lb, ub)
            print(f"  lambda {lam!r}: dd sign {cache[lam][0]:+.0f}, ln|det| {cache[lam][1]:.10f} ({time.time() - t0:.0f} s)", flush=True)
        return cache[lam]
    slo, shi = sd(o.lo), sd(o.hi)
    rec = dict(D=o.D, kc=o.kc, lo=o.lo, hi=o.hi, record=o.record, mutate=MUTATE or None, sign_lo=slo[0], sign_hi=shi[0])
    if slo[0] == shi[0]:
        rec.update(root=None, verdict="FAIL (no sign change)")
    else:
        ref = 0.5 * (slo[1] + shi[1])
        root = brentq(lambda l: sd(l)[0] * math.exp(sd(l)[1] - ref), o.lo, o.hi, xtol=o.xtol, rtol=1e-15)
        rec.update(root=root, n_eval=len(cache), minus_record=(root - o.record) if o.record is not None else None)
        good = o.record is None or abs(root - o.record) <= o.bar
        rec["verdict"] = "PASS" if good else "FAIL"
    rec["seconds"] = round(time.time() - t0)
    print(f"check B2 D {o.D} k_c {o.kc}: dd root {rec.get('root')!r}" + (f", minus the record {rec['minus_record']:+.2e} (bar "
          f"{o.bar:g})" if rec.get("minus_record") is not None else "") + f" -> {rec['verdict']}", flush=True)
    if o.out:
        open(o.out, "a").write(json.dumps(rec) + "\n")
    return 0 if rec["verdict"] == "PASS" else 1


if __name__ == "__main__":
    if sys.argv[1] == "det":
        sys.exit(detcmd(sys.argv[2:]))
    if sys.argv[1] == "root":
        sys.exit(rootcmd(sys.argv[2:]))
    if sys.argv[1] == "selftest":
        sys.exit(selftest())
    if sys.argv[1] == "check":
        sys.exit(check(sys.argv[2:]))
    if sys.argv[1] == "sshres":
        sys.exit(sshres(sys.argv[2:]))
