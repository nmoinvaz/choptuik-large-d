"""Reference implementation of the Moebius horizon series (mobius_s.py) in t-units, element by element, generic over
float64 and mpmath object arrays ("mp" backend). Written first (scratch hseries_t.py), kept as the independent check of
the vectorized float64 and double-double versions: it shares the equations with them, not the code."""
import numpy as np
import mpmath as mp


def dmat_odd(M):
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
    D = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i != j:
                D[i, j] = (-1.0) ** (i - j) / np.tan((i - j) * np.pi / M)
    return D


def dmat_odd_mp(M):
    N = 2 * M
    D = np.empty((M, M), dtype=object)
    for i in range(M):
        for j in range(M):
            v = mp.mpf(0)
            for jj, sgn in ((j, 1), (j + M, -1)):
                if jj != i:
                    v += sgn * mp.mpf(1) / 2 * (-1) ** (i - jj) * mp.cot((i - jj) * mp.pi / N)
            D[i, j] = v
    return D


def dmat_even_mp(M):
    D = np.empty((M, M), dtype=object)
    for i in range(M):
        for j in range(M):
            D[i, j] = (-1) ** (i - j) * mp.cot((i - j) * mp.pi / M) if i != j else mp.mpf(0)
    return D


class HSeriesT:
    def __init__(self, D, M, a=0.0, ks=None, backend="f64", kb=None):
        self.D, self.n, self.M, self.backend = D, D - 2, M, backend
        n = self.n
        self.ks = ks if ks is not None else int(n + 14 * np.sqrt(n) + 80)
        self.kb = kb
        if backend == "mp":
            self.a = mp.mpf(a)
            self.Do, self.De = dmat_odd_mp(M), dmat_even_mp(M)
            self.one = mp.mpf(1)
        else:
            self.a = float(a)
            self.Do, self.De = dmat_odd(M), dmat_even(M)
            self.one = 1.0
        a = self.a
        # h, e, z as polynomials / series in t
        K = self.ks + 3
        self.h = [(1 * self.one) / (1 + a), (a - 1) / (1 + a), -a / (1 + a)] + [0 * self.one] * K
        self.e = [(1 * self.one) / (1 + a), -2 * self.one / (1 + a), self.one / (1 + a)] + [0 * self.one] * K
        # z = (1 - t)/(1 + a t) = (1 - t) sum (-a t)^j
        z = [self.one]
        for j in range(1, K):
            z.append((-a) ** j - (-a) ** (j - 1))
        self.z = z
        self.noise = None

    def zeros(self, K):
        if self.backend == "mp":
            Z = np.empty((K, self.M), dtype=object)
            Z[:] = mp.mpf(0)
            return Z
        return np.zeros((K, self.M))

    def solve(self, A, b):
        if self.backend == "mp":
            x = mp.lu_solve(mp.matrix(A.tolist()), mp.matrix(list(b)))
            return np.array([x[i] for i in range(self.M)], dtype=object)
        return np.linalg.solve(A, b)

    def eye_diag(self, v):
        if self.backend == "mp":
            A = np.empty((self.M, self.M), dtype=object)
            A[:] = mp.mpf(0)
            for i in range(self.M):
                A[i, i] = v[i]
            return A
        return np.diag(v)

    def conv(self, A, B, k, lo=0, hi=None):
        """sum_{j=lo}^{hi} A[j] B[k-j] pointwise in theta; A, B arrays (or lists of scalars for A)."""
        hi = k if hi is None else hi
        if hi < lo:
            return 0 * self.one * np.ones(self.M) if self.backend != "mp" else self.zeros(1)[0]
        if self.kb is not None and hi - lo + 1 > 2 * self.kb + 2:
            js = list(range(lo, lo + self.kb + 1)) + list(range(max(lo + self.kb + 1, k - self.kb), hi + 1))
        else:
            js = range(lo, hi + 1)
        s = None
        for j in js:
            t = A[j] * B[k - j]
            s = t if s is None else s + t
        return s

    def horizon(self, Q0, om):
        n, ks, M, a = self.n, self.ks, self.M, self.a
        h, e, z = self.h, self.e, self.z
        K = ks + 2
        U, V, X, G, c, q, dU, dV = (self.zeros(K + 1) for _ in range(8))
        Qn = self.zeros(K + 1)  # Q_k = U_k - V_k, known one order ahead
        c[0] = self.one
        Qn[0] = Q0
        Dt = self.Do.T
        DeT = self.De
        for k in range(0, ks + 1):
            # provisional U_k, V_k (P_k = 0)
            U[k] = Qn[k] / 2
            V[k] = -Qn[k] / 2
            # q_k (independent of P_k)
            zV2 = self.conv(z, self.conv_series(V, V, k), k) if False else None  # placeholder (unused)
            UU = self.conv(U, U, k)
            VV = self.conv(V, V, k)
            zVV = self.conv_list(z, self.cache_VV, k, VV)
            cUV = self.conv_list_arr(c, self.cache_UV, k, self.conv(U, V, k))
            q[k] = (UU + zVV) / 2 - cUV
            # X_k: (w D - (n-1) - q_0) X_k = q_k + sum_{j>=1} q_j X_{k-j}
            rhs = q[k] + (self.conv(q, X, k, 1, k) if k >= 1 else 0 * q[k])
            Ak = om * DeT - self.eye_diag((n - 1) + q[0])
            X[k] = self.solve(Ak, rhs)
            # G_k (1 + X_0) = -(n-1) [c X]_k - sum_{j<k} G_j X_{k-j}
            cX = self.conv(c, X, k)
            GX = self.conv(G, X, k, 0, k - 1) if k >= 1 else 0 * cX
            G[k] = (-(n - 1) * cX - GX) / (1 + X[0])
            # c_{k+1}: h_0 (k+1) c_{k+1} = -G_k/2 - sum_{i>=1} h_i (k-i+1) c_{k-i+1}
            s = -G[k] / 2
            for i in range(1, min(k, 2) + 1):
                s = s - h[i] * (k - i + 1) * c[k - i + 1]
            c[k + 1] = s / (h[0] * (k + 1))
            # (A + B)_k with provisional U_k, V_k
            dU[k - 1 if k >= 1 else 0] = k * U[k] if k >= 1 else dU[0]
            dV[k - 1 if k >= 1 else 0] = k * V[k] if k >= 1 else dV[0]
            r = self.apb(U, V, c, G, dU, dV, k, om, Dt)
            c1 = c[1]
            Lk = om * self.Do + self.eye_diag(k * (1 + a + 2 * c1) / (1 + a) - G[0] - (n / 2 - 1))
            Pk = self.solve(Lk, -r)
            U[k] = U[k] + Pk / 2
            V[k] = V[k] + Pk / 2
            if self.noise is not None:
                for nm in self.noise[0]:
                    arr = {"U": U, "V": V, "X": X, "G": G, "c": c}[nm]
                    kk = k + 1 if nm == "c" else k
                    arr[kk] = arr[kk] * (1 + self.noise[1] * self.noise[2].standard_normal(M))
            if k >= 1:
                dU[k - 1] = k * U[k]
                dV[k - 1] = k * V[k]
            self.cache_VV[k] = self.conv(V, V, k)
            self.cache_UV[k] = self.conv(U, V, k)
            # Q_{k+1} from A_k with U_{k+1} = V_{k+1} = 0
            dU[k] = 0 * U[k]
            dV[k] = 0 * V[k]
            A0 = self.a_k(U, V, c, G, dU, dV, k, om, Dt)
            Qn[k + 1] = A0 / (2 * h[0] * (k + 1))
        self.series = dict(U=U[:ks + 1], V=V[:ks + 1], X=X[:ks + 1], c=c[:ks + 1], G=G[:ks + 1])
        ts = (1 * self.one) / (2 + a)
        tp = [ts ** k for k in range(ks + 1)]
        out = {}
        for nm in ("U", "V", "c", "X"):
            A = self.series[nm]
            s = None
            for k in range(ks + 1):
                t = A[k] * tp[k]
                s = t if s is None else s + t
            out[nm] = s
        return out

    # convolution helpers that reuse cached order-k products for k-1, ... (cache_* set per order)
    def conv_list(self, zl, cache, k, cur):
        """sum_j z_j P_{k-j}, P_m = (V V)_m; cache[m] for m < k, cur = (V V)_k."""
        s = zl[0] * cur
        for j in range(1, k + 1):
            if self.kb is not None and j > self.kb and k - j > self.kb:
                continue
            s = s + zl[j] * cache[k - j]
        return s

    def conv_list_arr(self, c, cache, k, cur):
        s = c[0] * cur
        for j in range(1, k + 1):
            if self.kb is not None and j > self.kb and k - j > self.kb:
                continue
            s = s + c[j] * cache[k - j]
        return s

    def conv_series(self, A, B, k):
        return self.conv(A, B, k)

    def apb(self, U, V, c, G, dU, dV, k, om, Dt):
        """(A + B)_k: 2[h(c-1)U_t]_k + 2[(hc - e)V_t]_k + w D (U + zV)_k - [G(U+V)]_k + [zV]_k - (n-1)[cU]_k"""
        n, h, e, z = self.n, self.h, self.e, self.z
        # hc1 = h (c - 1), hce = h c - e as series up to order k
        out = None
        cm1 = [c[j] if j > 0 else 0 * c[0] for j in range(k + 1)]
        hc1 = [self.conv_scal(h, cm1, j) for j in range(k + 1)]
        hce = [self.conv_scal(h, c, j) - e[j] for j in range(k + 1)]
        t1 = None
        for m in range(0, k):  # [hc1 U_t]_k = sum_m hc1_{k-m} dU_m, hc1_0 = 0
            if self.kb is not None and m > self.kb and k - m > self.kb:
                continue
            term = 2 * (hc1[k - m] * dU[m] + hce[k - m] * dV[m])
            t1 = term if t1 is None else t1 + term
        if t1 is None:
            t1 = 0 * U[0]
        zV = self.conv_list(z, self.cache_V, k, V[k]) if False else self.convz(V, k)
        UzV = U[k] + zV
        out = t1 + om * (UzV @ Dt) - self.conv(G, U, k) - self.conv(G, V, k) + zV - (n - 1) * self.conv(c, U, k)
        return out

    def a_k(self, U, V, c, G, dU, dV, k, om, Dt):
        """A_k = -2[h (U_t - c V_t)]_k + w D U_k - [G V]_k, with the order-(k+1) terms zeroed in dU[k], dV[k]"""
        h = self.h
        s = None
        for i in range(0, min(k, 2) + 1):
            m = k - i
            cVt = self.conv(c, dV, m)
            term = h[i] * (dU[m] - cVt)
            s = term if s is None else s + term
        return -2 * s + om * (U[k] @ Dt) - self.conv(G, V, k)

    def convz(self, V, k):
        z = self.z
        s = z[0] * V[k]
        for j in range(1, k + 1):
            if self.kb is not None and j > self.kb and k - j > self.kb:
                continue
            s = s + z[j] * V[k - j]
        return s

    def conv_scal(self, h, c, j):
        """[h c]_j for h a short scalar polynomial (degree <= 2) and c a series"""
        s = h[0] * c[j]
        for i in range(1, min(j, 2) + 1):
            s = s + h[i] * c[j - i]
        return s


def make(D, M, a, backend="f64", ks=None, kb=None):
    o = HSeriesT(D, M, a, ks=ks, backend=backend, kb=kb)
    K = o.ks + 3
    o.cache_VV = [None] * (K + 1)
    o.cache_UV = [None] * (K + 1)
    return o
