"""Form S of the DSS problem with its horizon series in a Moebius variable, whose rounding does not grow with D (#1090).

Equations (form S in the parity variables of notes/LARGE_D_FACTORED.md, checked symbolically there by
solver/factored/derive_factored.py; U = p + m, V = (p - m)/x, z = x^2, n = D - 2, theta = omega tau):
    A = 2z U_z - 2cz V_z + w U_th - G V = 0
    B = 2z^2 V_z - 2cz U_z + w z V_th + z V - (G + (n-1) c) U = 0
    G = 2z c_z,   (w d_th - (n-1) - q) X = q,  q = (U^2 + z V^2)/2 - c U V,  X = 1/a^2 - 1,  G = -(n-1) c X/(1 + X)
The last relation is the momentum constraint with a_x eliminated (gr-qc/9604019 Eq. 19; #1089 slice 3): a is slaved,
so there is no mass mode.

Layout: Track B's centre series in z, data p1 (V(0) = 2 p1) and c0, summed at x* = 1/sqrt 2 (solver/factored/
fseries_s.py, FactoredS.centre, copied here); the horizon series at the SSH with c = 1 and Q0 = U - V = 2 m_h; rows U,
V, c matched at x* and the phase row. Unknowns p1, c0, Q0 (M collocation points each) and Delta.

The horizon variable. In any variable affine in z (Track B's s = (1 - z)/(1 - z*)), x* lies halfway between the SSH and
the centre z = 0, where every SSH-analytic solution except the critical one is singular. The critical solution's
horizon series is then the minimal solution of its recurrence, and forward recurrence amplifies rounding by a factor
that grows like 2^(n/2) (logs in I1090-mac/). Here the series runs in
    t = (1 - z)/(1 + a z),   tau = t/t*,   t* = 1/(2 + a)   (x* at tau = 1, the centre at tau = 2 + a, z = inf at
                                                             tau = -(2 + a)/a),
with z d/dz = -h d/dt, h = (1 - t)(1 + a t)/(1 + a), z^2 d/dz = -e d/dt, e = (1 - t)^2/(1 + a), and
    A + B = 2h(c - 1) U_t + 2(hc - e) V_t + w (U + z V)_th - G (U + V) + z V - (n-1) c U
    A     = -2h (U_t - c V_t) + w U_th - G V,          c_t = -G/(2h).
Both derivative terms of A + B carry a factor t at the SSH (c = 1, z = 1), so order k of A + B fixes P_k = U_k + V_k
through L_k = w D + k(1 + a + 2c_1)/(1 + a) - G_0 - (n/2 - 1) (resonant at k = sigma_R, the SSH exponent), and order
k of A fixes Q_{k+1} = U_{k+1} - V_{k+1}. q_k needs only Q_k (z_0 = c_0 = 1). a = 0 is Track B's recurrence (in w);
both reproduce Track B's sums at x* to 1e-15 at its D-100 solution (check_mobius.py).

Generic over numpy float64 and complex128 (complex-step Jacobian).
"""
import numpy as np


# ---------------------------------------------------------------- collocation in theta (solver/factored/fseries.py)
def dmat_odd(M):
    """Folded d/dtheta for odd fields on the half grid of N = 2M points (hp/python/hpkernel/driver.py)."""
    N = 2 * M
    D = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            v = 0.0
            for jj, sgn in ((j, 1.0), (j + M, -1.0)):
                if jj != i:
                    v += sgn * 0.5 * (-1.0) ** (i - jj) / np.tan((i - jj) * np.pi / N)
            D[i, j] = v
    return D


def dmat_even(M):
    """d/dtheta for even fields (period pi) on the half grid theta_j = pi j / M."""
    D = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i != j:
                D[i, j] = (-1.0) ** (i - j) / np.tan((i - j) * np.pi / M)
    return D


def phase_weights(M):
    return np.sin(np.pi * np.arange(M) / M)


def cauchy(A, B, k, lo, hi):
    """sum_{j=lo}^{hi} A[j] B[k-j] (pointwise in theta)."""
    if hi < lo:
        return 0.0 * A[0]
    return np.einsum("jm,jm->m", A[lo:hi + 1], B[k - hi:k - lo + 1][::-1])


class FormS:
    def __init__(self, D, M, a=1.0, kc=None, ks=None, kb=150, zK=0.5):
        self.D = float(D)
        self.n = n = self.D - 2.0
        self.M = M
        self.a = float(a)
        self.kc = kc if kc is not None else 160
        self.ks = ks if ks is not None else int(n + 14 * np.sqrt(n) + 80)
        self.kb = kb
        self.zK = zK
        self.Do = dmat_odd(M)
        self.Dt = self.Do.T
        self.De = dmat_even(M)
        self.phase = phase_weights(M)
        a = self.a
        ts = (1.0 - zK) / (1.0 + a * zK)  # t at x*
        self.ts = ts
        self.iota = 1.0 / ts
        K = self.ks + 4
        h = np.zeros(K)
        e = np.zeros(K)
        zs = np.zeros(K)
        h[0], h[1], h[2] = 1 / (1 + a), (a - 1) / (1 + a) * ts, -a / (1 + a) * ts ** 2
        e[0], e[1], e[2] = 1 / (1 + a), -2 / (1 + a) * ts, 1 / (1 + a) * ts ** 2
        zs[0] = 1.0
        if a == 0:
            zs[1] = -ts
        else:
            for j in range(1, K):
                zs[j] = (1 + a) / a * (-a * ts) ** j  # = -(1 + a)(-a)^(j-1) t*^j, without overflow
        self.h, self.e, self.zs = h, e, zs
        self.noise = None  # (names, eta, rng): relative noise on stored arrays after each horizon order (a test)
        self.mutate = ""

    # ------------------------------------------------------------------ helpers
    def J(self, k, lo, hi):
        if hi < lo:
            return np.arange(0)
        if self.kb is None or hi - lo + 1 <= 2 * self.kb + 2:
            return np.arange(lo, hi + 1)
        return np.concatenate([np.arange(lo, lo + self.kb + 1), np.arange(max(lo + self.kb + 1, k - self.kb), hi + 1)])

    def cv(self, A, B, k, lo=0, hi=None):
        hi = k if hi is None else hi
        J = self.J(k, lo, hi)
        if len(J) == 0:
            return 0.0 * A[0]
        return np.einsum("jm,jm->m", A[J], B[k - J])

    def cvs(self, s, B, k, lo=0, hi=None):
        hi = k if hi is None else hi
        J = self.J(k, lo, hi)
        if len(J) == 0:
            return 0.0 * B[0]
        return np.einsum("j,jm->m", s[J], B[k - J])

    def xsolve(self, om, q0, rhs):
        A = om * self.De - np.diag((self.n - 1) + q0)
        return np.linalg.solve(A, rhs)

    # ------------------------------------------------------------------ centre (Track B, FactoredS.centre)
    def centre(self, p1, c0, om):
        n, K, M = self.n, self.kc, self.M
        dt = np.result_type(p1, c0, om)
        Z = lambda: np.zeros((K + 1, M), dt)  # noqa: E731
        U, V, X, beta, c, G, W, qs = Z(), Z(), Z(), Z(), Z(), Z(), Z(), Z()
        c[0] = c0
        V[0] = 2.0 * p1
        for k in range(1, K + 1):
            j = np.arange(1, k)[:, None]
            wts = 2 * (k - j) * c[1:k] + G[1:k] + (n - 1) * c[1:k]
            num = 2 * (k - 1) * V[k - 1] + om * (V[k - 1] @ self.Dt) + V[k - 1]
            if k > 1:
                num = num - np.einsum("jm,jm->m", wts, U[k - 1:0:-1])
            U[k] = num / (c0 * (2 * k + n - 1))
            W[k] = cauchy(U, V, k, 1, k)
            cW = cauchy(c, W, k, 0, k - 1)
            qs[k] = 0.5 * (cauchy(U, U, k, 1, k - 1) + cauchy(V, V, k - 1, 0, k - 1)) - cW
            rhs = qs[k] + cauchy(qs, X, k, 1, k)
            X[k] = self.xsolve(om, qs[0], rhs)
            beta[k] = -(n - 1) * X[k]
            G[k] = cauchy(c, beta, k, 0, k - 1) - cauchy(X, G, k, 1, k - 1)
            c[k] = G[k] / (2 * k)
            num = 2 * k * U[k] + om * (U[k] @ self.Dt)
            if k > 1:
                num = num - np.einsum("jm,jm->m", 2 * c[1:k] * (k - j), V[k - 1:0:-1])
            num = num - cauchy(G, V, k, 1, k)
            V[k] = num / (2 * c0 * k)
        zp = self.zK ** np.arange(K + 1)
        out = {nm: zp @ Xs for nm, Xs in (("U", U), ("V", V), ("c", c))}
        out["tail"] = max(np.max(np.abs(Y[-1] * zp[-1])) for Y in (U, V, c, X))
        return out

    # ------------------------------------------------------------------ horizon (Moebius variable, tau-units)
    def horizon(self, Q0, om, keep=False):
        n, ks, M = self.n, self.ks, self.M
        h, e, zs, io = self.h, self.e, self.zs, self.iota
        dt = np.result_type(Q0, om)
        K = ks + 3
        Z = lambda: np.zeros((K, M), dt)  # noqa: E731
        U, V, X, G, c, q, dU, dV, Qn, VVc, UVc, hc, zV, zVV = (Z() for _ in range(14))
        ats = self.a * self.ts  # (1 + a t* tau) zW = (1 - t* tau) W: zW_k = W_k - t* W_{k-1} - a t* zW_{k-1}
        c[0] = 1.0
        Qn[0] = Q0
        Do, Dt = self.Do, self.Dt
        Ax = om * self.De - np.diag((n - 1) + 0.5 * Q0 * Q0)  # q_0 = Q0^2/2 at the SSH
        for k in range(0, ks + 1):
            U[k] = 0.5 * Qn[k]
            V[k] = -0.5 * Qn[k]
            VVk = self.cv(V, V, k)
            UVk = self.cv(U, V, k)
            VVc[k] = VVk
            UVc[k] = UVk
            zVV[k] = VVc[k] - (self.ts * VVc[k - 1] + ats * zVV[k - 1] if k >= 1 else 0.0)
            q[k] = 0.5 * (self.cv(U, U, k) + zVV[k]) - self.cv(c, UVc, k)
            rhs = q[k] + (self.cv(q, X, k, 1, k) if k >= 1 else 0.0 * q[k])
            X[k] = np.linalg.solve(Ax, rhs)
            G[k] = (-(n - 1) * self.cv(c, X, k) - (self.cv(G, X, k, 0, k - 1) if k >= 1 else 0.0)) / (1.0 + X[0])
            s = -0.5 * G[k]
            for i in range(1, min(k, 2) + 1):
                s = s - io * h[i] * (k - i + 1) * c[k - i + 1]
            c[k + 1] = s / (io * h[0] * (k + 1))
            hc[k] = h[0] * c[k] + (h[1] * c[k - 1] if k >= 1 else 0.0) + (h[2] * c[k - 2] if k >= 2 else 0.0)
            if k >= 1:
                dU[k - 1] = k * U[k]
                dV[k - 1] = k * V[k]
                m = self.J(k, 0, k - 1)
                hc1 = hc[k - m] - h[k - m][:, None]
                hce = hc[k - m] - e[k - m][:, None]
                deriv = 2.0 * io * (np.einsum("jm,jm->m", hc1, dU[m]) + np.einsum("jm,jm->m", hce, dV[m]))
            else:
                deriv = 0.0 * Q0
            zVk = V[k] - (self.ts * V[k - 1] + ats * zV[k - 1] if k >= 1 else 0.0)
            r = (deriv + om * ((U[k] + zVk) @ Dt) - self.cv(G, U, k) - self.cv(G, V, k) + zVk
                 - (n - 1) * self.cv(c, U, k))
            Lk = om * Do + np.diag(io * k * (2 * h[0] * c[1] + h[1] - e[1]) - G[0] - (n / 2 - 1))
            Pk = np.linalg.solve(Lk, -r)
            U[k] = U[k] + 0.5 * Pk
            V[k] = V[k] + 0.5 * Pk
            if self.noise is not None:
                names, eta, rng = self.noise
                for nm in names:
                    arr = {"U": U, "V": V, "X": X, "G": G, "c": c}[nm]
                    kk = k + 1 if nm == "c" else k
                    arr[kk] = arr[kk] * (1 + eta * rng.standard_normal(M))
            if k >= 1:
                dU[k - 1] = k * U[k]
                dV[k - 1] = k * V[k]
            VVc[k] = self.cv(V, V, k)
            UVc[k] = self.cv(U, V, k)
            zV[k] = V[k] - (self.ts * V[k - 1] + ats * zV[k - 1] if k >= 1 else 0.0)
            zVV[k] = VVc[k] - (self.ts * VVc[k - 1] + ats * zVV[k - 1] if k >= 1 else 0.0)
            dU[k] = 0.0
            dV[k] = 0.0
            Ak = 0.0 * Q0
            for i in range(0, min(k, 2) + 1):
                mm = k - i
                Ak = Ak + h[i] * (dU[mm] - self.cv(c, dV, mm))
            Ak = -2.0 * io * Ak + om * (U[k] @ Dt) - self.cv(G, V, k)
            Qn[k + 1] = Ak / (2.0 * io * h[0] * (k + 1))
        out = {"U": U[:ks + 1].sum(0), "V": V[:ks + 1].sum(0), "c": c[:ks + 1].sum(0)}
        out["tail"] = max(np.max(np.abs(Y[ks])) for Y in (U, V, c))
        if keep:
            out["series"] = {"U": U[:ks + 1], "V": V[:ks + 1], "c": c[:ks + 1], "X": X[:ks + 1]}
        return out

    # ------------------------------------------------------------------ residual
    def unpack(self, y):
        M = self.M
        return y[:M], y[M:2 * M], y[2 * M:3 * M], y[3 * M]

    def residual(self, y):
        p1, c0, Q0, Delta = self.unpack(y)
        om = 2 * np.pi / Delta
        ce = self.centre(p1, c0, om)
        ho = self.horizon(Q0, om)
        self.last = (ce, ho)
        return np.concatenate([ce["U"] - ho["U"], ce["V"] - ho["V"], ce["c"] - ho["c"], [self.phase @ p1]])

    def jacobian(self, y, h=1e-30):
        L = len(y)
        J = np.zeros((L, L))
        for i in range(L):
            yc = y.astype(complex)
            yc[i] += 1j * h
            J[:, i] = self.residual(yc).imag / h
        return J


# ---------------------------------------------------------------- resampling between M (solver/factored/fseries.py)
def resample_odd(f, M):
    """Odd field on the half grid of len(f) points to M points (odd harmonics below M kept)."""
    M0 = len(f)
    F = np.fft.rfft(np.concatenate([f, -f])) / (2 * M0)
    G = np.zeros(M + 1, dtype=complex)
    kmax = min(M, M0)
    G[1:kmax:2] = F[1:kmax:2]
    return np.fft.irfft(G * (2 * M), n=2 * M)[:M]


def resample_even(f, M):
    M0 = len(f)
    F = np.fft.rfft(np.concatenate([f, f])) / (2 * M0)
    G = np.zeros(M + 1, dtype=complex)
    kmax = min(M, M0)
    G[0:kmax:2] = F[0:kmax:2]
    if kmax == M and M % 2 == 0:
        G[M] = F[M].real if M < M0 else F[M]
    return np.fft.irfft(G * (2 * M), n=2 * M)[:M]


def harmonics_odd(f):
    M = len(f)
    F = np.fft.rfft(np.concatenate([f, -f])) / (2 * M)
    return np.abs(F[1:M:2]) * 2
