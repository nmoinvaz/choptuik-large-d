"""Form S of the factored formulation: a slaved at every x by the momentum constraint (#1088, #1089 slice 3).

Gundlach's form (gr-qc/9604019, Eq. 19 and App. C) replaces the Hamiltonian equation for a by the combination of the
Hamiltonian and momentum constraints that has no x-derivative. With B = 1/a^2, X = B - 1 (derive_s.py, check 1):

    (omega d/dtheta - (n-1) - q_S) X = q_S,     q_S = ((x - c) p^2 + (x + c) m^2)/x = (U^2 + z V^2)/2 - c U V

a periodic linear solve in theta at each x, and in each series at each order: no x-recurrence for a, hence no mass
mode x^-(n-1) and no a datum anywhere. beta = (n-1)(1 - B) = -(n-1) X, G = (n-1) c (a^2 - 1) = c beta / B.

The Hamiltonian residual then obeys H_tau = (n-1) a^2 H (slice 3), whose only periodic solution is H = 0.

Order by order:
  centre (z-series): U_k (needs no G_k), then q_S,k, X_k, G_k (G B = c beta), c_k = G_k / (2k), then V_k;
  horizon (s = w / w_K): Q_k = U_k - V_k from S1 - S2 (where G_k cancels), then q_S,k = Q_0 Q_k + (lower orders),
  X_k, G_k, c_{k+1}, then P_k = U_k + V_k from S2 with the M x M solve of fseries.py.
Unknowns p1, c0, Q0 = 2 m_h (M each) and Delta; rows U, V, c matched at z_K = 1/2 and the phase row.
"""
import numpy as np

from fseries import Factored, cauchy


def dmat_even(M):
    """d/dtheta for even fields (period pi) on the half grid theta_j = pi j / M."""
    D = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i != j:
                D[i, j] = (-1.0) ** (i - j) / np.tan((i - j) * np.pi / M)
    return D


class FactoredS(Factored):
    def __init__(self, D, M, **kw):
        super().__init__(D, M, **kw)
        self.De = dmat_even(M)
        if self.mutate == "evenD":
            self.De = 0.5 * self.De

    def xsolve(self, om, qs0, rhs):
        n = self.n
        A = om * self.De - np.diag((n - 1) + qs0)
        return np.linalg.solve(A, rhs)

    # ------------------------------------------------------------------ centre
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
            W[k] = cauchy(U, V, k, 1, k)  # U_0 = 0, so V_k is not needed
            cW = cauchy(c, W, k, 0, k - 1)
            qs[k] = 0.5 * (cauchy(U, U, k, 1, k - 1) + cauchy(V, V, k - 1, 0, k - 1)) - cW
            rhs = qs[k] + cauchy(qs, X, k, 1, k)
            X[k] = self.xsolve(om, qs[0], rhs)
            beta[k] = -(n - 1) * X[k]
            # G B = c beta, B = 1 + X, X_0 = 0, beta_0 = 0
            G[k] = cauchy(c, beta, k, 0, k - 1) - cauchy(X, G, k, 1, k - 1)
            c[k] = G[k] / (2 * k)
            num = 2 * k * U[k] + om * (U[k] @ self.Dt)
            if k > 1:
                num = num - np.einsum("jm,jm->m", 2 * c[1:k] * (k - j), V[k - 1:0:-1])
            num = num - cauchy(G, V, k, 1, k)
            V[k] = num / (2 * c0 * k)
            if self.noise and self.noise_where in ("both", "centre"):
                U[k] = U[k] * (1 + self.noise * self.rng.standard_normal(M))
                V[k] = V[k] * (1 + self.noise * self.rng.standard_normal(M))
        zp = self.zK ** np.arange(K + 1)
        out = {nm: zp @ Xs for nm, Xs in (("U", U), ("V", V), ("c", c), ("beta", beta))}
        Xs = zp @ X
        out["alpha"] = 1 / np.sqrt(1 + Xs) - 1
        out["tail"] = max(np.max(np.abs((Y[-1] * zp[-1]).real)) for Y in (U, V, c, X))
        out["series"] = (U, V, c, X)
        return out

    # ------------------------------------------------------------------ horizon
    def horizon(self, Q0, om):
        n, ks, M = self.n, self.ks, self.M
        lam = self.wK
        dt = np.result_type(Q0, om)
        Z = lambda r=ks + 1: np.zeros((r, M), dt)  # noqa: E731
        U, V, DU, DV, X, beta, G, cG, H, f, W, V2, U2, qs = (Z() for _ in range(14))
        c, c2 = Z(ks + 2), Z(ks + 2)
        c[0] = 1.0
        c2[0] = 1.0
        # order 0: q_S(0) = Q0^2 / 2, X_0, beta_0, g = G_0; P_0 from the order-0 relation
        qs[0] = 0.5 * Q0 * Q0
        X[0] = self.xsolve(om, qs[0], qs[0])
        beta[0] = -(n - 1) * X[0]
        B0 = 1.0 + X[0]
        G[0] = beta[0] / B0
        g = G[0]
        c[1] = lam * (-G[0]) / 2
        c2[1] = 2 * c[1]
        e = -c2[1] - lam
        f[0] = e / lam
        eprev = e
        cG[0] = G[0]
        H[0] = cG[0] + (n - 1)
        A0 = om * self.Dm - np.diag(n / 2 - 1 + g)
        P = np.linalg.solve(A0, (n / 2) * Q0)
        U[0] = (P + Q0) / 2
        V[0] = (P - Q0) / 2
        DU[0] = U[0] @ self.Dt
        DV[0] = V[0] @ self.Dt
        U2[0] = U[0] * U[0]
        V2[0] = V[0] * V[0]
        W[0] = U[0] * V[0]
        for k in range(1, ks + 1):
            jj = np.arange(1, k + 1)[:, None]
            Ur = U[k - 1::-1]
            Vr = V[k - 1::-1]
            DUr = DU[k - 1::-1]
            DVr = DV[k - 1::-1]
            fk = f[1:k + 1] * (k - jj)
            # provisional G_k = 0 (it cancels from S1 - S2)
            cGk0 = cauchy(c, G, k, 1, k)  # sum_{j>=1} c_j G_{k-j}
            Hk0 = cGk0 + (n - 1) * c2[k]
            cGc = cG.copy() if False else None  # noqa: F841
            cG[k] = cGk0
            H[k] = Hk0
            G[k] = 0.0
            r1 = (-2 * np.einsum("jm,jm->m", fk, Vr) + om * np.einsum("jm,jm->m", c[1:k + 1], DUr)
                  - lam * om * DV[k - 1] - np.einsum("jm,jm->m", cG[1:k + 1], Vr) - lam * V[k - 1]
                  - np.einsum("jm,jm->m", G[1:k + 1] + (n - 1) * c[1:k + 1], Ur))
            cz = c[1:k + 1] - lam * c[0:k]
            cgz = (c[1:k + 1] - G[1:k + 1]) - lam * (c[0:k] - G[0:k])
            r2 = (-2 * np.einsum("jm,jm->m", fk, Ur) - lam * om * DU[k - 1] + om * np.einsum("jm,jm->m", cz, DVr)
                  + np.einsum("jm,jm->m", cgz, Vr) - np.einsum("jm,jm->m", H[1:k + 1], Ur))
            Qk = (r1 - r2) / (2 * k * (1 - g))
            # q_S at order k: Q_0 Q_k plus the terms without U_k, V_k
            U2p = cauchy(U, U, k, 1, k - 1)
            V2p = cauchy(V, V, k, 1, k - 1)
            Wp = cauchy(U, V, k, 1, k - 1)
            cWlow = cauchy(c, W, k, 1, k)
            qs[k] = Q0 * Qk + 0.5 * (U2p + V2p - lam * V2[k - 1]) - Wp - cWlow
            if self.mutate == "noslave":
                qs[k] = 0.0 * qs[k]
            rhs = qs[k] + cauchy(qs, X, k, 1, k)
            X[k] = self.xsolve(om, qs[0], rhs)
            beta[k] = -(n - 1) * X[k]
            G[k] = (cauchy(c, beta, k, 0, k) - cauchy(X, G, k, 1, k)) / B0
            cG[k] = cGk0 + G[k]
            H[k] = cG[k] + (n - 1) * c2[k]
            c[k + 1] = lam * (2 * k * c[k] - G[k]) / (2 * (k + 1))
            c2[k + 1] = cauchy(c, c, k + 1, 0, k + 1)
            e = -c2[k + 1]
            f[k] = (e - lam * eprev) / lam
            eprev = e
            r2 = r2 - G[k] * (V[0] + U[0])
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
            U2[k] = cauchy(U, U, k, 0, k)
            V2[k] = cauchy(V, V, k, 0, k)
            W[k] = cauchy(U, V, k, 0, k)
        out = {"U": U.sum(0), "V": V.sum(0), "c": c[:ks + 1].sum(0), "beta": beta.sum(0),
               "series": (U, V, c[:ks + 1], X)}
        out["tail"] = max(np.max(np.abs(Y[-1].real)) for Y in (U, V, c[:ks + 1], X))
        out["a_h"] = 1.0 / np.sqrt(1.0 + X[0])
        return out

    # ------------------------------------------------------------------ residual
    def residual(self, y):
        p1, c0, Q0, Delta = self.unpack(y)
        om = 2 * np.pi / Delta
        ce = self.centre(p1, c0, om)
        ho = self.horizon(Q0, om)
        R = np.concatenate([ce["U"] - ho["U"], ce["V"] - ho["V"], ce["c"] - ho["c"], [self.phase @ p1]])
        self.last = (ce, ho)
        return R

    def jacobian(self, y, h=1e-30):
        L = len(y)
        J = np.zeros((L, L))
        for i in range(L):
            yc = y.astype(complex)
            yc[i] += 1j * h
            J[:, i] = self.residual(yc).imag / h
        return J
