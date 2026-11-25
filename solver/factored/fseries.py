"""The factored large-D formulation of the DSS problem (#1088, notes/LARGE_D_FACTORED.md): residual and helpers.

Fields in parity variables, all even functions of x, hence series in z = x^2 (derive_factored.py, checks 1-5):

  U = p + m,  V = (p - m)/x,  a,  c;   q = (U^2 + z V^2)/2,  G = (n-1) c (a^2 - 1) = 2 z c_z
  A = 2z U_z - 2cz V_z + omega U_th - G V = 0
  B = 2z^2 V_z - 2cz U_z + omega z V_th + z V - (G + (n-1)c) U = 0
  4z a_z = a q + (n-1) a (1 - a^2)

Layout (no Taylor steps, no node unknowns):
  - centre series in z, data p1 (V(0) = 2 p1) and c0, summed at z_K = 1/2 (x_K = 1/sqrt 2, where the fast scalar mode
    F = x^-n (1-x^2)^(n(1+K)/2) has modulus 2^(-kappa/2), independent of n; check 7);
  - horizon series in w = 1 - z, data Q0 = U - V = 2 m_h at w = 0 and c = 1 there (the gauge); P0 = U + V from the
    order-0 relation (omega D - (n/2 - 1 + g)) P0 = (n/2) Q0; at order k, Q_k pointwise and P_k by one M x M solve;
  - the mass mode factored: beta = (n-1)(1 - 1/a^2) on the horizon side is the minimal (Miller) solution of its
    recurrence plus mu times the normalized mass mode chi, with mu fixed by beta(z_K) = the centre side's beta(z_K),
    so the horizon datum a_h is an output with sensitivity 1/G to the matched value, never an unknown;
  - rows: U, V, c matched at z_K (3M) and the phase row sum_j sin(pi j / M) p1_j; unknowns p1, c0, Q0, Delta.

The arithmetic is numpy's: float64 or complex128 (complex-step Jacobian). tau grid: theta_j = pi j / M, j < M (half
grid; odd fields antiperiodic under theta -> theta + pi), as hp's driver (dmat_odd, phase_weights).
"""
import numpy as np


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


def phase_weights(M):
    return np.sin(np.pi * np.arange(M) / M)


def resample_odd(f, M):
    """Odd field on the half grid of len(f) points to M points (odd harmonics below M kept)."""
    M0 = len(f)
    full = np.concatenate([f, -f])
    F = np.fft.rfft(full) / (2 * M0)
    G = np.zeros(M + 1, dtype=complex)
    kmax = min(M, M0)
    G[1:kmax:2] = F[1:kmax:2]
    out = np.fft.irfft(G * (2 * M), n=2 * M)
    return out[:M]


def resample_even(f, M):
    M0 = len(f)
    full = np.concatenate([f, f])
    F = np.fft.rfft(full) / (2 * M0)
    G = np.zeros(M + 1, dtype=complex)
    kmax = min(M, M0)
    G[0:kmax:2] = F[0:kmax:2]
    if kmax == M and M % 2 == 0:
        G[M] = F[M].real if M < M0 else F[M]
    out = np.fft.irfft(G * (2 * M), n=2 * M)
    return out[:M]


def harmonics_odd(f):
    """|complex amplitude| of each odd harmonic of an odd half-grid field."""
    M = len(f)
    F = np.fft.rfft(np.concatenate([f, -f])) / (2 * M)
    return np.abs(F[1:M:2]) * 2


def cauchy(A, B, k, lo, hi):
    """sum_{j=lo}^{hi} A[j] B[k-j] (pointwise in theta)."""
    if hi < lo:
        return 0.0 * A[0]
    return np.einsum("jm,jm->m", A[lo:hi + 1], B[k - hi:k - lo + 1][::-1])


class Factored:
    def __init__(self, D, M, kc=160, ks=None, zK=0.5, beta_tol=None, max_sweeps=200, mutate=""):
        self.Dd = float(D)
        self.n = self.Dd - 2.0
        self.M = M
        self.kc = kc
        n = self.n
        self.ks = ks if ks is not None else int(n + 14 * np.sqrt(n) + 80)
        self.zK = zK
        self.wK = 1.0 - zK
        self.Dm = dmat_odd(M)
        self.Dt = self.Dm.T
        self.phase = phase_weights(M)
        self.beta_tol = beta_tol
        self.max_sweeps = max_sweeps
        self.mutate = mutate
        self.beta_cache = None
        self.stats = {}
        self.block_sweeps = 18
        self.noise = 0.0  # relative noise put on every series coefficient (the rounding-amplification test)
        self.noise_where = "both"
        self.rng = np.random.default_rng(1088)

    # ------------------------------------------------------------------ centre
    def centre(self, p1, c0, om):
        n, K, M = self.n, self.kc, self.M
        dt = np.result_type(p1, c0, om)
        U = np.zeros((K + 1, M), dt)
        V = np.zeros((K + 1, M), dt)
        a = np.zeros((K + 1, M), dt)
        c = np.zeros((K + 1, M), dt)
        a2 = np.zeros((K + 1, M), dt)
        G = np.zeros((K + 1, M), dt)
        q = np.zeros((K + 1, M), dt)
        a[0] = 1.0
        a2[0] = 1.0
        c[0] = c0
        V[0] = 2.0 * p1
        for k in range(1, K + 1):
            q[k] = 0.5 * (cauchy(U, U, k, 1, k - 1) + cauchy(V, V, k - 1, 0, k - 1))
            aq = cauchy(a, q, k, 0, k - 1)
            S2 = cauchy(a, a, k, 1, k - 1)
            S3 = cauchy(a, a2, k, 1, k - 1)
            a[k] = (aq - (n - 1) * (S2 + S3)) / (4 * k + 2 * (n - 1))
            a2[k] = 2 * a[k] + S2
            c[k] = (n - 1) * cauchy(c, a2, k, 0, k - 1) / (2 * k)
            G[k] = 2 * k * c[k]
            j = np.arange(1, k)[:, None]
            wts = 2 * (k - j) * c[1:k] + G[1:k] + (n - 1) * c[1:k]
            num = 2 * (k - 1) * V[k - 1] + om * (V[k - 1] @ self.Dt) + V[k - 1]
            if k > 1:
                num = num - np.einsum("jm,jm->m", wts, U[k - 1:0:-1])
            U[k] = num / (c0 * (2 * k + n - 1))
            num = 2 * k * U[k] + om * (U[k] @ self.Dt)
            if k > 1:
                num = num - np.einsum("jm,jm->m", 2 * c[1:k] * (k - j), V[k - 1:0:-1])
            num = num - cauchy(G, V, k, 1, k)
            V[k] = num / (2 * c0 * k)
            if self.noise and self.noise_where in ("both", "centre"):
                U[k] = U[k] * (1 + self.noise * self.rng.standard_normal(M))
                V[k] = V[k] * (1 + self.noise * self.rng.standard_normal(M))
        zp = self.zK ** np.arange(K + 1)
        out = {}
        for nm, X in (("U", U), ("V", V), ("c", c)):
            out[nm] = zp @ X
        alpha = zp[1:] @ a[1:]
        out["alpha"] = alpha
        out["beta"] = (n - 1) * alpha * (2 + alpha) / (1 + alpha) ** 2
        tails = [np.max(np.abs((X[-1] * zp[-1]).real)) for X in (U, V, a, c)]
        out["tail"] = max(tails)
        out["series"] = (U, V, a, c)
        return out

    # ------------------------------------------------------------------ horizon
    def beta_solve(self, q, betaK):
        """beta's coefficients in s = w/w_K: rows k = 0..ks-1 of its recurrence and the match beta(z_K) = betaK.

        Row k: d_k beta_k - (2/lam)(k+1) beta_{k+1} + sum_{j>=1} q_j beta_{k-j} = (n-1) q_k, d_k = 2k + n - 1 + q_0,
        lam = w_K. The mass mode (the homogeneous solution) has s-coefficients that rise to k* = (n-1-2/lam)/(2/lam-2)
        and fall after, so a particular solution is swept outward from k* in both directions (backward below k*,
        forward above it), each direction its stable one; chi is the homogeneous solution swept the same way with
        chi_{k*} = 1, and beta = particular + mu chi with mu fixed by the match. The convolution is iterated.
        """
        n, ks = self.n, self.ks
        lam = self.wK
        kk = np.arange(ks + 1)[:, None]
        d = 2 * kk + (n - 1) + q[0][None, :]
        rhs = (n - 1) * q
        kst = int(np.clip(np.floor((n - 1 - 2 / lam) / (2 / lam - 2)), 0, ks))
        if self.mutate == "kstar0":
            kst = 0
        self.stats["kstar"] = kst

        def sweep(r, anchor):
            b = np.zeros_like(q)
            b[kst] = anchor
            for k in range(kst - 1, -1, -1):
                b[k] = (r[k] + (2 / lam) * (k + 1) * b[k + 1]) / d[k]
            for k in range(kst, ks):
                b[k + 1] = (d[k] * b[k] - r[k]) * lam / (2 * (k + 1))
            return b

        def conv_of(b):
            cv = np.zeros_like(q)
            for k in range(1, ks + 1):
                cv[k] = cauchy(q, b, k, 1, k)
            return cv

        zero = 0.0 * q[0]
        one = zero + 1.0
        bp = sweep(rhs, zero)
        chi = sweep(0.0 * rhs, one)
        for it in range(40):
            nbp = sweep(rhs - conv_of(bp), zero)
            nchi = sweep(-conv_of(chi), one)
            dif = max(np.max(np.abs((nbp - bp).real)) / max(1e-300, np.max(np.abs(nbp.real))),
                      np.max(np.abs((nchi - chi).real)))
            bp, chi = nbp, nchi
            if dif < 1e-17:
                break
        self.stats["beta_iters"] = it + 1
        mu = (betaK - bp.sum(axis=0)) / chi.sum(axis=0)
        out = bp + mu[None, :] * chi
        if self.noise and self.noise_where in ("both", "horizon"):
            out = out * (1 + self.noise * self.rng.standard_normal(out.shape))
        return out, mu

    def horizon_fields(self, Q0, om, beta, kmax=None, kst=None, want_q=False):
        """Forward recurrence of U, V, c on the horizon side.

        beta holds the s-coefficients of beta; with kst given, the orders above kst are not read but computed in the
        same pass, beta_{k+1} = (d_k beta_k + sum_{j>=1} q_j beta_{k-j} - (n-1) q_k) lam / (2(k+1)), the stable
        direction of the beta recurrence there; beta is then updated in place. kmax stops the pass early.
        """
        n, ks, M = self.n, self.ks, self.M
        kmax = ks if kmax is None else kmax
        dt = np.result_type(Q0, om, beta)
        q = np.zeros((ks + 1, M), dt)
        V2 = np.zeros((ks + 1, M), dt)
        U = np.zeros((ks + 1, M), dt)
        V = np.zeros((ks + 1, M), dt)
        DU = np.zeros((ks + 1, M), dt)
        DV = np.zeros((ks + 1, M), dt)
        c = np.zeros((ks + 2, M), dt)
        c2 = np.zeros((ks + 2, M), dt)
        G = np.zeros((ks + 1, M), dt)
        cG = np.zeros((ks + 1, M), dt)
        H = np.zeros((ks + 1, M), dt)
        f = np.zeros((ks + 1, M), dt)
        Bs = -beta / (n - 1)
        B0 = 1.0 + Bs[0]
        c[0] = 1.0
        c2[0] = 1.0
        eprev = None
        g = None
        lam = self.wK
        for k in range(0, kmax + 1):
            if kst is not None and k > kst:
                Bs[k] = -beta[k] / (n - 1)
            G[k] = (cauchy(c, beta, k, 0, k) - cauchy(Bs, G, k, 1, k)) / B0
            c[k + 1] = lam * (2 * k * c[k] - G[k]) / (2 * (k + 1))
            c2[k + 1] = cauchy(c, c, k + 1, 0, k + 1)
            e = -c2[k + 1] - (lam if k == 0 else 0.0)
            f[k] = e / lam if k == 0 else (e - lam * eprev) / lam
            eprev = e
            cG[k] = cauchy(c, G, k, 0, k)
            H[k] = cG[k] + (n - 1) * c2[k]
            if k == 0:
                g = G[0]
                A0 = om * self.Dm - np.diag(n / 2 - 1 + g)
                P = np.linalg.solve(A0, (n / 2) * Q0)
                U[0] = (P + Q0) / 2
                V[0] = (P - Q0) / 2
                DU[0] = U[0] @ self.Dt
                DV[0] = V[0] @ self.Dt
                if want_q or kst is not None:
                    self._q_step(q, V2, U, V, 0)
                    if kst is not None and kst == 0 and k < ks:
                        self._beta_step(beta, q, 0)
                continue
            jj = np.arange(1, k + 1)[:, None]
            Ur = U[k - 1::-1]  # U_{k-1}, ..., U_0  (pairs with j = 1..k)
            Vr = V[k - 1::-1]
            DUr = DU[k - 1::-1]
            DVr = DV[k - 1::-1]
            fk = f[1:k + 1] * (k - jj)
            r1 = (-2 * np.einsum("jm,jm->m", fk, Vr) + om * np.einsum("jm,jm->m", c[1:k + 1], DUr)
                  - lam * om * DV[k - 1] - np.einsum("jm,jm->m", cG[1:k + 1], Vr) - lam * V[k - 1]
                  - np.einsum("jm,jm->m", G[1:k + 1] + (n - 1) * c[1:k + 1], Ur))
            cz = c[1:k + 1] - lam * c[0:k]
            cgz = (c[1:k + 1] - G[1:k + 1]) - lam * (c[0:k] - G[0:k])
            r2 = (-2 * np.einsum("jm,jm->m", fk, Ur) - lam * om * DU[k - 1] + om * np.einsum("jm,jm->m", cz, DVr)
                  + np.einsum("jm,jm->m", cgz, Vr) - np.einsum("jm,jm->m", H[1:k + 1], Ur))
            Qk = (r1 - r2) / (2 * k * (1 - g))
            Ak = om * self.Dm + np.diag(k * (1 - g) - (n / 2 - 1 + g))
            if self.mutate == "sigma":
                Ak = om * self.Dm + np.diag(k * (1 - g) - (n / 2 - 1))
            Pk = np.linalg.solve(Ak, -r2 - (k * (1 - g) - n / 2) * Qk)
            U[k] = (Pk + Qk) / 2
            V[k] = (Pk - Qk) / 2
            if self.noise and self.noise_where in ("both", "horizon"):
                U[k] = U[k] * (1 + self.noise * self.rng.standard_normal(M))
                V[k] = V[k] * (1 + self.noise * self.rng.standard_normal(M))
            DU[k] = U[k] @ self.Dt
            DV[k] = V[k] @ self.Dt
            if want_q or kst is not None:
                self._q_step(q, V2, U, V, k)
                if kst is not None and k >= kst and k < ks:
                    self._beta_step(beta, q, k)
        if want_q or kst is not None:
            return U, V, c[:ks + 1], q
        return U, V, c[:ks + 1]

    def _q_step(self, q, V2, U, V, k):
        V2[k] = cauchy(V, V, k, 0, k)
        q[k] = 0.5 * (cauchy(U, U, k, 0, k) + V2[k] - (self.wK * V2[k - 1] if k > 0 else 0.0))

    def _beta_step(self, beta, q, k):
        n, lam = self.n, self.wK
        d = 2 * k + (n - 1) + q[0]
        beta[k + 1] = (d * beta[k] + cauchy(q, beta, k, 1, k) - (n - 1) * q[k]) * lam / (2 * (k + 1))

    def kstar(self):
        n, lam = self.n, self.wK
        k = int(np.clip(np.floor((n - 1 - 2 / lam) / (2 / lam - 2)), 0, self.ks))
        return 0 if self.mutate == "kstar0" else k

    def horizon_anchor(self, Q0, om, anchor):
        """Horizon side with the mass mode's coefficient at its s-peak k* as datum (the anchor).

        Orders 0..k* of beta come from the backward sweep from the anchor (their stable direction), iterated with the
        forward U, V, c pass over the same orders until they stop changing; orders above k* are swept forward in one
        final pass. There is no fixed point over the orders above k*, where the fast scalar mode's shadow lives.
        """
        n, ks, M = self.n, self.ks, self.M
        lam = self.wK
        kst = self.kstar()
        self.stats["kstar"] = kst
        dt = np.result_type(Q0, om, anchor)
        beta = np.zeros((ks + 1, M), dt)
        beta[kst] = anchor
        hist = []
        # a fixed number of block sweeps from zero, so that the map is the same smooth function for every input
        # (an early exit or a warm start makes complex-step and difference derivatives inconsistent)
        for it in range(self.block_sweeps):
            U, V, c, q = self.horizon_fields(Q0, om, beta, kmax=kst, want_q=True)
            new = beta.copy()
            for k in range(kst - 1, -1, -1):
                d = 2 * k + (n - 1) + q[0]
                new[k] = ((n - 1) * q[k] - cauchy(q, beta, k, 1, k) + (2 / lam) * (k + 1) * new[k + 1]) / d
            scale = max(1e-300, np.max(np.abs(new[:kst + 1].real)))
            dif = np.max(np.abs(new[:kst] - beta[:kst])) / scale if kst > 0 else 0.0
            hist.append(dif)
            beta = new
        U, V, c, q = self.horizon_fields(Q0, om, beta, kst=kst)
        self.beta_cache = beta.real.copy()
        self.stats["sweeps"] = len(hist)
        self.stats["beta_dif"] = float(hist[-1])
        out = {"U": U.sum(0), "V": V.sum(0), "c": c.sum(0), "beta": beta.sum(0), "series": (U, V, c, beta)}
        out["tail"] = max(np.max(np.abs(X[-1].real)) for X in (U, V, c, beta))
        out["a_h"] = 1.0 / np.sqrt(1.0 - beta[0] / (n - 1))
        return out

    def q_of(self, U, V):
        ks = self.ks
        q = np.zeros_like(U)
        V2 = np.zeros_like(V)
        for k in range(ks + 1):
            V2[k] = cauchy(V, V, k, 0, k)
            q[k] = 0.5 * (cauchy(U, U, k, 0, k) + V2[k] - (self.wK * V2[k - 1] if k > 0 else 0.0))
        return q

    def horizon(self, Q0, om, betaK):
        n, ks = self.n, self.ks
        dt = np.result_type(Q0, om, betaK)
        if self.beta_cache is not None and self.beta_cache.shape[1] == self.M:
            beta = self.beta_cache.astype(dt)
        else:
            beta = np.zeros((ks + 1, self.M), dt)
            beta[0] = betaK
        wp = np.ones(ks + 1)
        hist = []
        for sweep in range(self.max_sweeps):
            U, V, c = self.horizon_fields(Q0, om, beta)
            q = self.q_of(U, V)
            nb, mu = self.beta_solve(q, betaK)
            dif = np.max(np.abs((nb - beta).real * wp[:, None]))
            hist.append(dif)
            beta = nb
            if dif < 1e-16 * max(1e-300, np.max(np.abs(betaK.real))) or (sweep > 3 and dif >= hist[-2] and dif < 1e-12):
                break
        U, V, c = self.horizon_fields(Q0, om, beta)
        self.beta_cache = beta.real.copy()
        self.stats["sweeps"] = len(hist)
        self.stats["beta_dif"] = hist[-1]
        out = {"U": wp @ U, "V": wp @ V, "c": wp @ c, "beta_series": beta, "series": (U, V, c), "mu": mu}
        out["tail"] = max(np.max(np.abs((X[-1] * wp[-1]).real)) for X in (U, V, c))
        Bs0 = 1.0 - beta[0] / (n - 1)
        out["a_h"] = 1.0 / np.sqrt(Bs0)
        return out

    # ------------------------------------------------------------------ residual
    def unpack(self, y):
        M = self.M
        return y[0:M], y[M:2 * M], y[2 * M:3 * M], y[3 * M]

    def residual(self, y):
        p1, c0, Q0, Delta = self.unpack(y)
        om = 2 * np.pi / Delta
        ce = self.centre(p1, c0, om)
        ho = self.horizon(Q0, om, ce["beta"])
        R = np.concatenate([ce["U"] - ho["U"], ce["V"] - ho["V"], ce["c"] - ho["c"], [self.phase @ p1]])
        self.last = (ce, ho)
        return R

    def unpack4(self, y):
        M = self.M
        return y[0:M], y[M:2 * M], y[2 * M:3 * M], y[3 * M:4 * M], y[4 * M]

    def residual4(self, y):
        p1, c0, Q0, anc, Delta = self.unpack4(y)
        om = 2 * np.pi / Delta
        ce = self.centre(p1, c0, om)
        ho = self.horizon_anchor(Q0, om, anc)
        R = np.concatenate([ce["U"] - ho["U"], ce["V"] - ho["V"], ce["c"] - ho["c"], ce["beta"] - ho["beta"],
                            [self.phase @ p1]])
        self.last = (ce, ho)
        return R

    def jacobian4(self, y, h=1e-30):
        L = len(y)
        cache = None if self.beta_cache is None else self.beta_cache.copy()
        J = np.zeros((L, L))
        for i in range(L):
            yc = y.astype(complex)
            yc[i] += 1j * h
            self.beta_cache = cache
            J[:, i] = self.residual4(yc).imag / h
        self.beta_cache = cache
        return J

    def jacobian(self, y, h=1e-30):
        """Complex-step Jacobian: exact derivatives to float64 rounding."""
        L = len(y)
        cache = None if self.beta_cache is None else self.beta_cache.copy()
        J = np.zeros((L, L))
        for i in range(L):
            yc = y.astype(complex)
            yc[i] += 1j * h
            self.beta_cache = cache
            J[:, i] = self.residual(yc).imag / h
        self.beta_cache = cache
        return J


def from_hp_state(path, M):
    """Initial unknowns (p1, c0, Q0 = 2 m_h, Delta) from an hp npz state, resampled to M points."""
    z = np.load(path, allow_pickle=False)
    import json
    meta = json.loads(str(z["meta"]))
    M0 = meta["M"]
    K = len(meta["nodes"])
    zz = z["zh"] + z["zl"]
    p1 = zz[0:M0]
    c0 = zz[M0:2 * M0]
    off_mh = 2 * M0 + 4 * M0 * K
    mh = zz[off_mh:off_mh + M0]
    ah = zz[off_mh + M0:off_mh + 2 * M0]
    Delta = zz[-1]
    y = np.concatenate([resample_odd(p1, M), resample_even(c0, M), 2 * resample_odd(mh, M), [Delta]])
    return y, meta, {"p1": p1, "c0": c0, "mh": mh, "ah": ah, "Delta": Delta, "zz": zz, "M0": M0, "K": K}
