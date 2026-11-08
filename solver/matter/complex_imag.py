#!/usr/bin/env python3
"""The real DSS solution's imaginary sector: growth rates of delta phi_I, a free test field on the frozen DSS metric,
as zeros of a multiple-shooting determinant det M(lambda), counted by winding numbers (matter/COMPLEX_CRITICAL.md).

    python3 complex_imag.py check STATE.npz                     background reconstruction, series, lambda = 0 mode
    python3 complex_imag.py det STATE.npz LAM [LAM ...]           ln det M at the given lambda
    python3 complex_imag.py count STATE.npz BOX [BOX ...] [--plant=Z] [--mutate=M] [--out=F] [--tol=0.785]
          BOX = NAME:RE0:RE1:IM0:IM1, or B0, U0, FULL (the standard boxes, below)
    python3 complex_imag.py root STATE.npz LAM                   secant root of det M from LAM
    python3 complex_imag.py selftest                             the winding code on a synthetic g

Shares with hp only the stored state (p1, c0, node values, m_h, a_h, Delta of an hp .npz, hi parts) and the equations
of hp/src/model.rs, which complex_imag_eqs.py derives from Box psi = 0.  Everything else is separate code: the
d/dtheta matrix (from Fourier modes), the background's centre and light-cone series and its integration between the
nodes (scipy DOP853), and the test field's series, propagators and determinant (numpy, double precision).

Test field (psi = delta phi_I ~ e^{lam tau} x periodic; a, c, G = (n-1) c (a^2 - 1) frozen at the background):
    x (x - c) P_x = -x (lam + omega d_th) P + (G - c) P + (n/2) c (P + Q),
    x (x + c) Q_x = -x (lam + omega d_th) Q - (G - c) Q - (n/2) c (P + Q).
Folded layout as hp's: P, Q odd in theta (antiperiodic over half a period), M points on the half grid.  Unknowns:
P1 (centre data, P = P1 x + ..., Q = -P1 x + ...), the node states (P, Q) at the K nodes, Q_h at x = 1 (the regular
solution there, P_h fixed by (omega d_th + lam + (1 + c1) - n/2) P_0 = (n/2) Q_0).  Rows: centre series (evaluated at
x_s) propagated to node 0 minus node 0; each segment's propagator minus the next node; the light-cone series at the
last node minus that node.  2M(K+1) unknowns, as hp's test matrix.

Zeros known in advance: lam = 0 (psi = phi_R, the U(1) phase rotation; (P, Q) = the background's (p, m)) and its
copies 2 i k omega.  Poles: where omega d_th + lam + (1 + c1)(k+1) - n/2 is singular (k = 0..ks), on Im lam = odd
multiples of omega; their candidates are multiplied out, g = det M prod (lam - p_j) e^{-trend}, so a box's winding of g
bounds its number of zeros from above (equality when every candidate inside is a pole; notes/MODE_COUNT.md section 2).

Standard boxes (omega = 2 pi / Delta; together they hold every growth rate of both theta-parity sectors with
-0.05 < Re lam < 6, by the 2 i omega periodicity of the folded layout):
    B0   -0.05 < Re < 0.3,  |Im| < omega/2         expected: lam = 0 only
    U0   -0.05 < Re < 0.3,  omega/2 < Im < 3 omega/2  expected: none
    FULL  0.3 < Re < 6,     |Im| < 3 omega/2         expected: none (no growing mode)
Symmetric boxes use det M(conj lam) = conj det M(lam): winding = (phase change along the upper half) / pi.

Controls: --plant=Z multiplies g by (lam - Z)(lam - conj Z) (a box holding Z counts one or two more);
--mutate=coupling scales the P equation's (n/2) c (P + Q) by 1.1, --mutate=noG drops G from both equations: either
breaks the lam = 0 mode, so B0 must change its count; selftest's control (no refinement) must fail.
"""
import json
import math
import os
import sys
import time

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
import scipy.linalg as sl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ------------------------------------------------------------------ theta grid
def dmat_odd(M):
    """d/dtheta on the half grid theta_j = pi j / M for fields with f(theta + pi) = -f(theta): built from the odd
    harmonics k = +-1, +-3, ..., +-(M-1) (M of them), D = V diag(i k) V^-1."""
    th = np.pi * np.arange(M) / M
    ks = np.concatenate([-np.arange(M - 1, 0, -2), np.arange(1, M, 2)])
    V = np.exp(1j * np.outer(th, ks))
    Dm = V @ np.diag(1j * ks) @ np.linalg.inv(V)
    return Dm.real.copy()


def dmat_odd_mp(M):
    """dmat_odd in mpmath (object array): D = V diag(i k) V^-1 at the working precision, real part"""
    th = [mp.pi * j / M for j in range(M)]
    ks = list(range(-(M - 1), 0, 2)) + list(range(1, M, 2))
    V = mp.matrix(M, M)
    for i in range(M):
        for j, k in enumerate(ks):
            V[i, j] = mp.expj(k * th[i])
    Vi = V ** -1
    out = np.empty((M, M), dtype=object)
    for i in range(M):
        for j in range(M):
            out[i, j] = mp.re(sum(V[i, l] * (1j * ks[l]) * Vi[l, j] for l in range(M)))
    return out


# ------------------------------------------------------------------ background
class Background:
    def __init__(self, path, mutate=None, kc_series=50, xs_max=0.05, us=0.0, ks_series=100):
        zf = np.load(path, allow_pickle=True)
        self.meta = json.loads(str(zf["meta"]))
        z = zf["zh"] + zf["zl"]
        self.dps = 34
        mp.mp.dps = self.dps
        zmp = [mp.mpf(float(h)) + mp.mpf(float(l)) for h, l in zip(zf["zh"], zf["zl"])]
        self.path = path
        self.D = float(self.meta["D"])
        self.n = self.D - 2
        M = self.M = int(self.meta["M"])
        self.nodes = list(self.meta["nodes"])
        K = self.K = len(self.nodes)
        assert len(z) == 4 * M * (K + 1) + 1, "unexpected state length"
        self.p1, self.c0 = z[0:M], z[M:2 * M]
        self.node = []
        for i in range(K):
            o = 2 * M + 4 * M * i
            self.node.append(z[o:o + 4 * M].reshape(4, M))     # p, m, a, c
        o = 2 * M + 4 * M * K
        self.mh, self.ah = z[o:o + M], z[o + M:o + 2 * M]
        self.mh_mp = np.array(zmp[o:o + M], dtype=object)
        self.ah_mp = np.array(zmp[o + M:o + 2 * M], dtype=object)
        self.D_mp = mp.mpf(self.meta["D"])
        self.Delta = z[-1]
        self.oD_mp = (2 * mp.pi / zmp[-1]) * dmat_odd_mp(M)
        self.omega = 2 * np.pi / self.Delta
        self.Dm = dmat_odd(M)
        self.oD = self.omega * self.Dm
        self.ks = ks_series
        self.kc = kc_series
        self.xs = min(self.nodes[0], xs_max)
        # the light-cone series is evaluated at x_e = max(1 - us, last node), by default the last node.  A matching point
        # nearer the light cone is worse at large D: the propagator from the last node to x_e damps the irregular
        # branch u^sigma by (u_e/u_last)^sigma, 3e-19 at D = 60 for x_e = 0.9, and det M is then rounding noise
        self.xe = max(1 - us, self.nodes[-1]) if us > 0 else self.nodes[-1]
        self.X = self.nodes + ([self.xe] if self.xe > self.nodes[-1] + 1e-14 else [])
        self.mutate = mutate
        self._centre_series()
        self._horizon_series()
        self._segments()

    # model.rs right-hand sides (background), x real
    def rhs(self, x, y):
        M, n = self.M, self.n
        p, m, a, c = y.reshape(4, M)
        G = (n - 1) * c * (a * a - 1)
        H = a * (p * p + m * m) + (n - 1) * (a - a ** 3)
        px = (-x * (self.oD @ p) + G * p - c * p + 0.5 * n * c * (p + m)) / (x * (x - c))
        mx = (-x * (self.oD @ m) - G * m + c * m - 0.5 * n * c * (p + m)) / (x * (x + c))
        return np.concatenate([px, mx, H / (2 * x), G / x])

    def _centre_series(self):
        M, n, K = self.M, self.n, self.kc
        p = np.zeros((K + 1, M)); m = np.zeros((K + 1, M)); a = np.zeros((K + 1, M)); c = np.zeros((K + 1, M))
        a[0] = 1.0; c[0] = self.c0; p[1] = self.p1; m[1] = -self.p1
        G = np.zeros((K + 1, M))

        def cauchy(F, H, k, lo=0):
            return sum(F[j] * H[k - j] for j in range(lo, k + 1))
        for k in range(2, K + 1):
            pm2 = np.array([cauchy(p, p, l) + cauchy(m, m, l) for l in range(k + 1)])
            ap = cauchy(a, pm2, k)                            # a_k = 0 so far
            a3 = np.array([cauchy(a, a, l) for l in range(k + 1)])
            R3 = cauchy(a, a3, k)                             # [a^3]_k without its 3 a_k part
            a[k] = (ap - (n - 1) * R3) / (2 * k + 2 * (n - 1))
            a2m1 = np.array([cauchy(a, a, l) for l in range(k + 1)]); a2m1[0] -= 1
            c[k] = (n - 1) * cauchy(c, a2m1, k) / k
            G[k] = k * c[k]
            s, h = k - 1 + n / 2, n / 2
            rp = (k - 1) * p[k - 1] + self.oD @ p[k - 1]
            rm = (k - 1) * m[k - 1] + self.oD @ m[k - 1]
            for j in range(1, k):
                rp += -c[j] * (k - j) * p[k - j] - G[j] * p[k - j] + c[j] * p[k - j] - h * c[j] * (p[k - j] + m[k - j])
                rm += c[j] * (k - j) * m[k - j] + G[j] * m[k - j] - c[j] * m[k - j] + h * c[j] * (p[k - j] + m[k - j])
            det = -s * s + h * h
            p[k] = (s * (-rp) + h * (-rm)) / (det * c[0])
            m[k] = (-h * (-rp) - s * (-rm)) / (det * c[0])
        self.cs = dict(p=p, m=m, a=a, c=c, G=G)

    def centre_eval(self, x):
        pw = x ** np.arange(self.kc + 1)
        return np.concatenate([pw @ self.cs[f] for f in ("p", "m", "a", "c")])

    def _horizon_series(self):
        """light-cone series in u = 1 - x: c_0 = 1, a_0 = a_h, m_0 = m_h; at order k, G_k, then c_{k+1}, p_k from
        (omega D + (1 + c1)(k + 1) - n/2) p_k = rhs, then a_{k+1}, m_{k+1} (complex_imag_eqs.py check 5).
        In mpmath at self.dps digits (from the state's double-double values): in double, rounding excites the mass
        mode, whose coefficients grow like binomial(n - 2 + k, k), and at D = 60 the coefficients past order 15 are
        that mode (|c_100| ~ 1e29), which then corrupts the test field's series.  The result is rounded to double."""
        M, K = self.M, self.ks
        mp.mp.dps = self.dps
        n = mp.mpf(self.D_mp) - 2
        Z = lambda *shape: np.array([mp.mpf(0)] * int(np.prod(shape)), dtype=object).reshape(shape)
        p = Z(K + 1, M); m = Z(K + 2, M); a = Z(K + 2, M); c = Z(K + 2, M)
        G = Z(K + 1, M); A2 = Z(K + 2, M); S = Z(K + 1, M); e = Z(K + 2, M); v = Z(K + 2, M)
        cp = lambda F, H, k: (F[:k + 1] * H[k::-1]).sum(0)
        oD = self.oD_mp
        oDm = mp.matrix(oD.tolist())
        c[0] = np.array([mp.mpf(1)] * M, dtype=object); a[0] = self.ah_mp.copy(); m[0] = self.mh_mp.copy()
        A2[0] = a[0] * a[0]
        one = mp.mpf(1)
        half_n = n / 2
        for k in range(0, K + 1):
            A2m = A2[:k + 1].copy(); A2m[0] = A2m[0] - one
            G[k] = (n - 1) * cp(c, A2m, k)
            c[k + 1] = (k * c[k] - G[k]) / (k + 1)
            for l in (k, k + 1):
                wl = (1 if l == 0 else 0) - (1 if l == 1 else 0)
                wl1 = (1 if l == 1 else 0) - (1 if l == 2 else 0)
                e[l] = (wl - c[l]) - ((wl1 - c[l - 1]) if l >= 1 else 0)
                v[l] = (wl + c[l]) - ((wl1 + c[l - 1]) if l >= 1 else 0)
            kap = (1 + c[1]) * (k + 1) - half_n
            rhs = half_n * m[k] + (oD.dot(p[k - 1]) if k >= 1 else 0)
            for j in range(2, k + 1):
                rhs = rhs + e[j] * (k + 1 - j) * p[k + 1 - j]
            for j in range(1, k + 1):
                rhs = rhs + G[j] * p[k - j] - c[j] * p[k - j] + half_n * c[j] * (p[k - j] + m[k - j])
            L = oDm.copy()
            for i in range(M):
                L[i, i] += kap[i]
            sol = mp.lu_solve(L, mp.matrix(list(rhs)))
            p[k] = np.array([sol[i] for i in range(M)], dtype=object)
            S[k] = cp(p, p, k) + cp(m, m, k)
            Ha = cp(a, S, k) + (n - 1) * (a[k] - cp(a, A2, k))
            a[k + 1] = (2 * k * a[k] - Ha) / (2 * (k + 1))
            A2[k + 1] = cp(a, a, k + 1)
            rm = oD.dot(m[k] - (m[k - 1] if k >= 1 else 0))
            for j in range(1, k + 1):
                rm = rm - v[j] * (k + 1 - j) * m[k + 1 - j]
            for j in range(0, k + 1):
                rm = rm + G[j] * m[k - j] - c[j] * m[k - j] + half_n * c[j] * (p[k - j] + m[k - j])
            m[k + 1] = rm / (v[0] * (k + 1))
        fl = lambda X: np.array(X.tolist(), dtype=float)
        self.hs = dict(p=fl(p), m=fl(m[:K + 1]), a=fl(a[:K + 1]), c=fl(c[:K + 2]), G=fl(G), e=fl(e), v=fl(v))
        self.c1 = self.hs["c"][1]

    def horizon_eval(self, x):
        u = 1 - x
        pw = u ** np.arange(self.ks + 1)
        return np.concatenate([pw @ self.hs[f][:self.ks + 1] for f in ("p", "m", "a", "c")])

    def _segments(self):
        """background on [x_s, node 0] (from the centre series), on each [node b-1, node b] (from node b-1) and on
        [last node, x_e] (from the last node); each segment's end is compared with the stored node, or at x_e with the
        light-cone series"""
        self.segs = []
        self.mismatch = []
        pre = self.xs < self.nodes[0] - 1e-14
        ends = ([self.xs] if pre else []) + self.X
        self.Xvals = {}
        for i in range(len(ends) - 1):
            x0, x1 = ends[i], ends[i + 1]
            y0 = self.centre_eval(x0) if i == 0 and pre else self.node[self.nodes.index(x0)].ravel()
            sol = solve_ivp(self.rhs, (x0, x1), y0, method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True)
            yb = self.node[self.nodes.index(x1)].ravel() if x1 in self.nodes else self.horizon_eval(x1)
            self.mismatch.append((x0, x1, float(np.max(np.abs(sol.y[:, -1] - yb))), float(np.max(np.abs(yb)))))
            self.segs.append((x0, x1, sol.sol))
            self.Xvals[x1] = sol.y[:, -1].reshape(4, self.M)
        if not pre:
            y = self.centre_eval(self.nodes[0])
            self.mismatch.insert(0, (0.0, self.nodes[0], float(np.max(np.abs(y - self.node[0].ravel()))),
                                     float(np.max(np.abs(self.node[0])))))
        self.pre = pre
        # the light-cone series at the last node as well (not used by the matrix when x_e > last node)
        yh = self.horizon_eval(self.nodes[-1])
        self.mismatch_series_last = float(np.max(np.abs(yh - self.node[-1].ravel())))
        for i, x in enumerate(self.nodes):
            self.Xvals[x] = self.node[i]

    def bg_at(self, x, seg):
        y = seg(x)
        return y.reshape(4, self.M)


# ------------------------------------------------------------------ test field
class TestField:
    def __init__(self, bg, mutate=None, rtol=1e-11, kf=None):
        self.bg = bg
        self.rtol = rtol
        self.kf = kf
        self.mutate = mutate
        self.kcoup = 1.1 if mutate == "coupling" else 1.0
        self.gG = 0.0 if mutate == "noG" else 1.0
        M = bg.M
        self.I = np.eye(M)
        self.poles_cache = None

    def amat(self, x, seg, lam):
        bg, M, n = self.bg, self.bg.M, self.bg.n
        p, m, a, c = bg.bg_at(x, seg)
        G = self.gG * (n - 1) * c * (a * a - 1)
        L = lam * self.I + bg.oD
        d1 = 1.0 / (x * (x - c)); d2 = 1.0 / (x * (x + c))
        A = np.empty((2 * M, 2 * M), complex)
        A[:M, :M] = (-x * L + np.diag(G - c + self.kcoup * 0.5 * n * c)) * d1[:, None]
        A[:M, M:] = np.diag(self.kcoup * 0.5 * n * c * d1)
        A[M:, :M] = np.diag(-0.5 * n * c * d2)
        A[M:, M:] = (-x * L - np.diag(G - c + 0.5 * n * c)) * d2[:, None]
        return A

    def propagate(self, seg, x0, x1, Y0, lam, rtol=None):
        rtol = rtol or self.rtol
        shp = Y0.shape

        def f(x, y):
            return (self.amat(x, seg, lam) @ y.reshape(shp)).ravel()
        sol = solve_ivp(f, (x0, x1), Y0.ravel().astype(complex), method="DOP853", rtol=rtol, atol=1e-14)
        return sol.y[:, -1].reshape(shp)

    def centre(self, lam, x):
        """the test field's centre series for P1 = unit vectors (columns), evaluated at x: (2M, M)"""
        bg, M, n, K = self.bg, self.bg.M, self.bg.n, self.bg.kc
        c, G = bg.cs["c"], self.gG * bg.cs["G"]
        P = np.zeros((K + 1, M, M), complex); Q = np.zeros((K + 1, M, M), complex)
        P[1] = self.I; Q[1] = -self.I
        L = lam * self.I + bg.oD
        h = n / 2
        hp_ = self.kcoup * h
        for k in range(2, K + 1):
            s = k - 1 + n / 2
            rp = (k - 1) * P[k - 1] + L @ P[k - 1]
            rq = (k - 1) * Q[k - 1] + L @ Q[k - 1]
            for j in range(1, k):
                cj, Gj = c[j][:, None], G[j][:, None]
                rp = rp - cj * (k - j) * P[k - j] - Gj * P[k - j] + cj * P[k - j] - hp_ * cj * (P[k - j] + Q[k - j])
                rq = rq + cj * (k - j) * Q[k - j] + Gj * Q[k - j] - cj * Q[k - j] + h * cj * (P[k - j] + Q[k - j])
            # c0 [[-s - (hp_-h), -hp_], [h, s]] (P_k, Q_k) = -(rp, rq): the j = 0 terms with the mutated coupling
            a11, a12, a21, a22 = -(k - 1) - hp_, -hp_, h, s
            det = a11 * a22 - a12 * a21
            c0 = c[0][:, None]
            P[k] = (a22 * (-rp) - a12 * (-rq)) / (det * c0)
            Q[k] = (-a21 * (-rp) + a11 * (-rq)) / (det * c0)
        pw = x ** np.arange(K + 1)
        return np.concatenate([np.tensordot(pw, P, 1), np.tensordot(pw, Q, 1)])

    def horizon(self, lam, x):
        """the light-cone series for Q_h = unit vectors, evaluated at x: (2M, M)"""
        bg, M, n, K = self.bg, self.bg.M, self.bg.n, self.bg.ks
        cc = bg.hs["c"]
        G = self.gG * bg.hs["G"]
        e, v = bg.hs["e"], bg.hs["v"]
        P = np.zeros((K + 1, M, M), complex); Q = np.zeros((K + 2, M, M), complex)
        Q[0] = self.I
        L = lam * self.I + bg.oD
        h = n / 2
        hp_ = self.kcoup * h
        # -(G0 - c0) = 1 + c1 (unmutated); with noG the p_k coefficient is -e1 k - (G0 - 1) - hp_
        G0 = G[0]
        for k in range(0, K + 1):
            kap = -e[1] * k - G0 + 1 - hp_
            rhs = hp_ * Q[k] + (L @ P[k - 1] if k >= 1 else 0)
            for j in range(2, k + 1):
                rhs = rhs + e[j][:, None] * (k + 1 - j) * P[k + 1 - j]
            for j in range(1, k + 1):
                rhs = rhs + G[j][:, None] * P[k - j] - cc[j][:, None] * P[k - j] + hp_ * cc[j][:, None] * (P[k - j] + Q[k - j])
            P[k] = np.linalg.solve(L + np.diag(kap), rhs)
            rq = L @ (Q[k] - (Q[k - 1] if k >= 1 else 0))
            for j in range(1, k + 1):
                rq = rq - v[j][:, None] * (k + 1 - j) * Q[k + 1 - j]
            for j in range(0, k + 1):
                rq = rq + G[j][:, None] * Q[k - j] - cc[j][:, None] * Q[k - j] + h * cc[j][:, None] * (P[k - j] + Q[k - j])
            Q[k + 1] = rq / (v[0][:, None] * (k + 1))
        u = 1 - x
        pw = u ** np.arange(K + 1)
        self.last_tail = float(max(np.max(np.abs(P[-1])) * u ** K, np.max(np.abs(Q[K])) * u ** K))
        return np.concatenate([np.tensordot(pw, P, 1), np.tensordot(pw, Q[:K + 1], 1)])

    def matrix(self, lam):
        """unknowns P1 | Y at each matching point X_0..X_{L-1} | Q_h; rows: centre, L - 1 segments, light cone"""
        bg, M = self.bg, self.bg.M
        L = len(bg.X)
        N = 2 * M * (L + 1)
        A = np.zeros((N, N), complex)
        I2 = np.eye(2 * M)
        C = self.centre(lam, bg.xs)
        si = 0
        if bg.pre:
            x0, x1, seg = bg.segs[0]
            C = self.propagate(seg, x0, x1, C, lam)
            si = 1
        A[0:2 * M, 0:M] = C
        A[0:2 * M, M:3 * M] = -I2
        for b in range(1, L):
            x0, x1, seg = bg.segs[si + b - 1]
            Phi = self.propagate(seg, x0, x1, I2.astype(complex), lam)
            r0 = 2 * M * b
            A[r0:r0 + 2 * M, M + 2 * M * (b - 1):M + 2 * M * b] = Phi
            A[r0:r0 + 2 * M, M + 2 * M * b:M + 2 * M * (b + 1)] = -I2
        H = self.horizon(lam, bg.X[-1])
        r0 = 2 * M * L
        A[r0:r0 + 2 * M, M + 2 * M * (L - 1):M + 2 * M * L] = -I2
        A[r0:r0 + 2 * M, M + 2 * M * L:] = H
        if self.kf is not None:
            # the cut (hp's kfreeze): P1 restricted to the odd harmonics <= kf, and the light-cone block's P rows
            # projected onto the same harmonics; the matrix stays square
            B, Bp = self.low_basis()
            nl = B.shape[1]
            keep = np.r_[0:2 * M * L, 2 * M * L + M:N]
            A2 = np.zeros((N - (M - nl), N - (M - nl)), complex)
            cols = np.r_[M:N]
            A2[:, :nl] = np.concatenate([A[:r0, :M] @ B, np.zeros((nl + M, nl))])
            A2[:r0, nl:] = A[:r0, cols]
            A2[r0:r0 + nl, nl:] = Bp @ A[r0:r0 + M, cols]
            A2[r0 + nl:, nl:] = A[r0 + M:, cols]
            return A2
        return A

    def low_basis(self):
        """real basis cos k th, sin k th (odd k <= kf) on the half grid, and its least-squares inverse"""
        M = self.bg.M
        th = np.pi * np.arange(M) / M
        B = np.stack([f(k * th) for k in range(1, self.kf + 1, 2) for f in (np.cos, np.sin)], 1)
        return B, np.linalg.pinv(B)

    def lndet(self, lam):
        """ln det M from an LU with partial pivoting (scipy getrf): sum of ln u_ii, plus i pi per row swap"""
        lu, piv = sl.lu_factor(self.matrix(complex(lam)), check_finite=True)
        d = np.diag(lu)
        swaps = int(np.sum(piv != np.arange(len(piv))))
        v = np.sum(np.log(np.abs(d))) + 1j * (np.sum(np.angle(d)) + np.pi * swaps)
        return complex(v.real, math.remainder(v.imag, 2 * np.pi))

    def poles(self):
        """candidate poles: lam with omega D + lam + kappa_k singular, k = 0..ks"""
        if self.poles_cache is None:
            bg, n = self.bg, self.bg.n
            out = []
            cc = bg.hs["c"]
            w = -cc.copy(); w[0] += 1; w[1] += -1
            e1 = w[1] - w[0]
            G0 = self.gG * (n - 1) * (bg.ah ** 2 - 1)
            for k in range(bg.ks + 1):
                kap = -e1 * k - G0 + 1 - self.kcoup * n / 2
                out.extend(list(-np.linalg.eigvals(bg.oD + np.diag(kap))))
            self.poles_cache = np.array(out)
        return self.poles_cache


# ------------------------------------------------------------------ counting
def wrap(x):
    return math.remainder(x, 2 * math.pi)


def track(lng, a, c, K, tol, maxdepth=14):
    """unwrapped change of Im ln g from a to c, K initial intervals, bisected until each step <= tol and the halves
    agree with the whole; returns (change, largest step, intervals, refinement-limit hits)"""
    total, big, nint, hits = 0.0, 0.0, 0, 0
    ts = np.linspace(0, 1, K + 1)
    val = lambda t: lng(a + (c - a) * t)

    def incr(t0, t1, depth):
        nonlocal big, nint, hits
        d = wrap(val(t1).imag - val(t0).imag)
        tm = 0.5 * (t0 + t1)
        d1 = wrap(val(tm).imag - val(t0).imag)
        d2 = wrap(val(t1).imag - val(tm).imag)
        if abs(d1 + d2 - d) > 1e-6 or abs(d) > tol:
            if depth < maxdepth:
                return incr(t0, tm, depth + 1) + incr(tm, t1, depth + 1)
            hits += 1
        big = max(big, abs(d1 + d2))
        nint += 1
        return d1 + d2
    for t0, t1 in zip(ts[:-1], ts[1:]):
        total += incr(t0, t1, 0)
    return total, big, nint, hits


LNDET_CACHE = {}


def load_cache(path, mutate):
    """ln det M values of an earlier run with the same state and mutation (the --evlog file)"""
    if path and os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if r.get("kind") == "eval" and r.get("mutate") == mutate:
                LNDET_CACHE[(round(r["lam"][0], 13), round(r["lam"][1], 13))] = complex(*r["lndet"])


class G:
    """ln g(lam) = ln det M + sum ln(lam - p_j) + ln plant - trend, cached by lam"""

    def __init__(self, tf, box, plant=None, log=None):
        self.tf, self.plant = tf, plant
        re0, re1, im0, im1 = box
        ph = tf.poles()
        mar = 2.0
        self.pl = ph[(ph.real > re0 - mar) & (ph.real < re1 + mar) & (np.abs(ph.imag) < max(abs(im0), abs(im1)) + 2 * tf.bg.omega)]
        self.cache = {}
        self.nev = 0
        self.trend = (0, 0, 0)
        self.c = complex(0.5 * (re0 + re1), 0.0)
        self.log = log

    def raw(self, lam):
        key = (round(lam.real, 13), round(lam.imag, 13))
        if key not in self.cache:
            t = time.time()
            if key in LNDET_CACHE:
                ld = LNDET_CACHE[key]
            else:
                ld = self.tf.lndet(lam)
                LNDET_CACHE[key] = ld
                if self.log:
                    self.log(dict(kind="eval", lam=[lam.real, lam.imag], lndet=[ld.real, ld.imag],
                                  s=round(time.time() - t, 2)))
            v = ld + np.sum(np.log(lam - self.pl))
            if self.plant is not None:
                v += np.log(lam - self.plant) + np.log(lam - np.conj(self.plant))
            self.cache[key] = v
            self.nev += 1
        return self.cache[key]

    def fit_trend(self, pts):
        """real quadratic trend c0 + c1 mu + c2 mu^2 (mu = lam - c, real coefficients: symmetric under conjugation)
        fitted to ln|g| at pts"""
        mu = np.array(pts) - self.c
        A = np.stack([np.ones(len(mu)), mu.real, (mu * mu).real], 1)
        y = np.array([self.raw(p).real for p in pts])
        self.trend = tuple(np.linalg.lstsq(A, y, rcond=None)[0])

    def __call__(self, lam):
        mu = lam - self.c
        c0, c1, c2 = self.trend
        return self.raw(lam) - (c0 + c1 * mu + c2 * mu * mu)


def count_box(tf, box, plant=None, h0=0.25, tol=math.pi / 4, log=None, maxdepth=14):
    """winding of g around the box: edges sampled at spacing <= h0, then bisected (track); a quadratic trend fitted
    to ln|g| at the corners and edge midpoints is taken out first (exp has no zeros: the winding is unchanged)"""
    re0, re1, im0, im1 = box
    g = G(tf, box, plant, log)
    sym = abs(im0 + im1) < 1e-12
    lo = 0.0 if sym else im0
    g.fit_trend([complex(re0, lo), complex(re1, lo), complex(re1, im1), complex(re0, im1),
                 complex(0.5 * (re0 + re1), im1), complex(re1, 0.5 * (lo + im1)), complex(re0, 0.5 * (lo + im1))])
    if sym:
        path = [complex(re1, 0), complex(re1, im1), complex(re0, im1), complex(re0, 0)]
    else:
        path = [complex(re0, im0), complex(re1, im0), complex(re1, im1), complex(re0, im1), complex(re0, im0)]
    tot, big, nint, hits = 0.0, 0.0, 0, 0
    for a, c in zip(path[:-1], path[1:]):
        Ke = max(1, int(math.ceil(abs(c - a) / h0)))
        d, b, ni, h = track(g, a, c, Ke, tol, maxdepth=maxdepth)
        tot += d; big = max(big, b); nint += ni; hits += h
    wind = tot / math.pi if sym else tot / (2 * math.pi)
    inside = [p for p in g.pl if re0 < p.real < re1 and im0 < p.imag < im1]
    return dict(winding_g=float(wind), largest_step=float(big), intervals=int(nint), refinement_limit_hits=int(hits),
                evaluations=int(g.nev), candidate_poles_inside=len(inside), symmetric=bool(sym),
                trend=[float(t) for t in g.trend])


def std_boxes(omega):
    return {"B0": (-0.05, 0.3, -omega / 2, omega / 2), "U0": (-0.05, 0.3, omega / 2, 1.5 * omega),
            "FULL": (0.3, 6.0, -1.5 * omega, 1.5 * omega)}


def opts_of(argv):
    pos, o = [], {}
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            o[k] = v
        else:
            pos.append(a)
    return pos, o


def main():
    pos, o = opts_of(sys.argv[1:])
    mutate = o.get("mutate") or None
    out = o.get("out")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        if out:
            with open(out, "a") as fo:
                fo.write(json.dumps(rec) + "\n")

    if pos[0] == "selftest":
        # synthetic ln det: trend 40 lam + 1.5 lam^2 (as det M's growth), zeros at 0 (B0), 1.2 +- 0.7i and
        # 5.98 +- 0.2i (FULL, the latter 0.02 inside its right edge), 3 +- 9i (outside); omega = 1
        zs = [0.0, 1.2 + 0.7j, 1.2 - 0.7j, 5.98 + 0.2j, 5.98 - 0.2j, 3.0 + 9.0j, 3.0 - 9.0j]

        class FakeBg:
            omega = 1.0

        class Fake:
            bg = FakeBg()

            def lndet(self, l):
                return 40 * l + 1.5 * l * l + sum(np.log(l - z) for z in zs)

            def poles(self):
                return np.array([], complex)
        ok = True
        for plant, want in ((None, (1, 0, 4)), (2.0 + 1.0j, (1, 0, 6))):
            res, resc = [], []
            for name, bx in std_boxes(1.0).items():
                r = count_box(Fake(), bx, plant=plant)
                rc = count_box(Fake(), bx, plant=plant, h0=10.0, tol=100.0, maxdepth=0)
                res.append(round(r["winding_g"], 6)); resc.append(round(rc["winding_g"], 6))
                ok = ok and r["largest_step"] <= math.pi / 4 and r["refinement_limit_hits"] == 0
            got = tuple(int(round(w)) for w in res)
            ctl = any(abs(w - e) > 1e-6 for w, e in zip(resc, want))
            print("selftest plant=%s: B0, U0, FULL = %s, expected %s; control (one sample per edge) %s -> %s"
                  % (plant, res, want, resc, "fires" if ctl else "DOES NOT FIRE"))
            ok = ok and got == want and all(abs(w - e) < 1e-6 for w, e in zip(res, want)) and ctl
        print("selftest", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    bg = Background(pos[1], us=float(o.get("us", "0")), ks_series=int(o.get("ks", "100")))
    tf = TestField(bg, mutate, kf=int(o["kf"]) if o.get("kf") else None)
    info = dict(state=os.path.basename(bg.path), D=bg.D, M=bg.M, K=bg.K, nodes=bg.nodes, omega=bg.omega, kf=tf.kf,
                Delta=bg.Delta, ks_series=bg.ks, kc_series=bg.kc, xs=bg.xs, xe=bg.xe, mutate=mutate)

    if pos[0] == "check":
        mm = [dict(x0=a, x1=b, maxdiff=d, maxval=v) for a, b, d, v in bg.mismatch]
        sym = float(np.max(np.abs(bg.cs["m"] - bg.cs["p"] * ((-1.0) ** np.arange(bg.kc + 1))[:, None])))
        try:
            sys.path.insert(0, os.path.join(HERE, "..", "..", "hp"))
            from hpkernel.driver import dmat_odd as hp_dmat
            dh, dl = hp_dmat(bg.M)
            ddiff = float(np.max(np.abs(dh + dl - bg.Dm)))
        except Exception as ex:
            ddiff = "hpkernel not available: %s" % ex
        # lambda = 0: the background (p, m) is a null vector of M(0)
        A0 = tf.matrix(0.0)
        zb = np.concatenate([bg.p1] + [bg.Xvals[x][:2].ravel() for x in bg.X] + [bg.mh])
        r0 = float(np.max(np.abs(A0 @ zb)) / np.max(np.abs(zb)))
        sv = np.linalg.svd(A0, compute_uv=False)
        rec = dict(kind="check", **info, background_mismatch=mm, series_at_last_node=bg.mismatch_series_last, centre_symmetry=sym, dmat_vs_hpkernel=ddiff,
                   null_residual_bg=r0, sv_min=float(sv[-1]), sv_next=float(sv[-2]), sv_max=float(sv[0]),
                   horizon_tail=tf.last_tail)
        emit(rec)
        return

    if pos[0] == "det":
        for s in pos[2:]:
            lam = complex(s.replace(" ", ""))
            t = time.time()
            v = tf.lndet(lam)
            emit(dict(kind="det", **info, lam=[lam.real, lam.imag], lndet=[v.real, v.imag], seconds=round(time.time() - t, 2)))
        return

    if pos[0] == "root":
        lam0 = complex(pos[2].replace(" ", ""))
        f = lambda l: np.exp(tf.lndet(l) - tf.lndet(lam0).real)
        l0, l1 = lam0, lam0 + 1e-4
        d0, d1 = f(l0), f(l1)
        for it in range(30):
            l2 = l1 - d1 * (l1 - l0) / (d1 - d0)
            if abs(l2 - l1) < 1e-11:
                break
            l0, d0, l1, d1 = l1, d1, l2, f(l2)
        emit(dict(kind="root", **info, start=[lam0.real, lam0.imag], lam=[l2.real, l2.imag], iterations=it))
        return

    if pos[0] == "count":
        plant = complex(o["plant"].replace(" ", "")) if "plant" in o else None
        tol = float(o.get("tol", math.pi / 4))
        boxes = std_boxes(bg.omega)
        evlog = o.get("evlog")

        load_cache(evlog, mutate)

        def log(r):
            r["mutate"] = mutate
            if evlog:
                with open(evlog, "a") as fo:
                    fo.write(json.dumps(r) + "\n")
        for spec in pos[2:]:
            if spec in boxes:
                name, bx = spec, boxes[spec]
            else:
                parts = spec.split(":")
                name, bx = parts[0], tuple(float(v) for v in parts[1:5])
            t = time.time()
            res = count_box(tf, bx, plant=plant, tol=tol, log=log, h0=float(o.get("h0", "0.25")))
            emit(dict(kind="count", **info, box_name=name, box=list(bx), plant=None if plant is None else
                      [plant.real, plant.imag], seconds=round(time.time() - t, 1), **res))
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
