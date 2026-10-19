"""Faster drop-in replacements (same mathematics, fewer Python-level operations).

* ssh_series: O(K^2) instead of O(K^3) -- the squared-field series A2 = A*A and Q2 = P*P + M*M are cached
  order by order instead of being rebuilt at every order; Cauchy products and the weighted sums are single
  vectorised numpy reductions over a (K, B, N) coefficient array; tau-derivatives of past orders are cached.
* solve_periodic: one batched LAPACK call instead of a Python loop over batch members.
* lm_svd3: lm_svd2 with all damping trials (and their chord corrections) evaluated as one batched residual call.

install() patches the DSS class so every existing script picks these up.
"""
import time
import numpy as np
import dss as _dss


def _cs(X, Y, k):
    """k-th Cauchy coefficient sum_{j=0..k} X[j] Y[k-j] for coefficient arrays X, Y of shape (>=k+1, B, N)."""
    return np.einsum("jbn,jbn->bn", X[:k + 1], Y[k::-1] if k > 0 else Y[:1])


def solve_periodic_fast(self, kappa, rhs, omega, bg=None):
    """Solve omega D f + kappa f = rhs (+ lam terms for perturbation members), batchwise in one LAPACK call."""
    B, N = rhs.shape
    Dm = self.Dmat1.T
    dt = np.result_type(kappa, rhs, omega)
    Mt = omega[:, None, None] * Dm[None, :, :].astype(dt)
    idx = np.arange(N); Mt[:, idx, idx] += kappa
    if self.lam == 0.0:
        return np.linalg.solve(Mt, rhs[..., None])[..., 0]
    out = np.empty(rhs.shape, dtype=dt)
    out[0] = np.linalg.solve(Mt[0], rhs[0])
    if B > 1:
        Mt1 = Mt[1:].copy(); Mt1[:, idx, idx] += self.lam
        out[1:] = np.linalg.solve(Mt1, (rhs[1:] + self.lam * out[0])[..., None])[..., 0]
    return out


def ssh_series_fast(self, m0, a0, omega):
    n, K = self.n, self.Ks
    B, N = m0.shape
    dt = np.result_type(m0, a0, omega, float)
    shp = (K + 2, B, N)
    P = np.zeros(shp, dt); M = np.zeros(shp, dt); A = np.zeros(shp, dt); C = np.zeros(shp, dt)
    A2 = np.zeros(shp, dt); Q2 = np.zeros(shp, dt)
    DP = np.zeros(shp, dt); DM = np.zeros(shp, dt)        # cached tau-derivatives
    A[0] = a0; C[0] = 1.0; M[0] = m0
    A2[0] = A[0] * A[0]
    DM[0] = self.D(M[0], omega)
    for Kk in range(K + 1):
        # c_{K+1}  (A2m1 = A2 - delta_{j0})
        C[Kk + 1] = (Kk * C[Kk] - (n - 1) * (_cs(C, A2, Kk) - C[Kk])) / (Kk + 1)
        # p_K
        j = np.arange(Kk + 2)
        w = -C[:Kk + 2].copy(); w[0] += 1.0; w[1] -= 1.0
        W = np.empty_like(w); W[0] = w[0]; W[1:] = w[1:] - w[:-1]
        V = (j[1:, None, None] * C[1:Kk + 2]) - (j[:-1, None, None] * C[:Kk + 1])
        PM = P[:Kk + 1] + M[:Kk + 1]
        tau_term = -DP[Kk - 1] if Kk >= 1 else 0.0
        i = j[:-1]
        wpu = np.einsum("ibn,ibn->bn", i[:, None, None] * P[:Kk + 1], W[Kk + 1:0:-1])
        r = tau_term - wpu + _cs(V, P, Kk) + _cs(C, P, Kk) - 0.5 * n * _cs(C, PM, Kk)
        kappa = Kk * (1.0 + C[1]) + C[1] + 1.0 - 0.5 * n
        P[Kk] = self.solve_periodic(kappa + 0 * r, -r, omega)
        DP[Kk] = self.D(P[Kk], omega)
        Q2[Kk] = _cs(P, P, Kk) + _cs(M, M, Kk)
        if Kk == K:
            break
        # a_{K+1}
        A3 = _cs(A, A2, Kk)
        aG = _cs(A, Q2, Kk) + (n - 1) * (A[Kk] - A3)
        A[Kk + 1] = (Kk * A[Kk] - 0.5 * aG) / (Kk + 1)
        A2[Kk + 1] = _cs(A, A, Kk + 1)
        # m_{K+1}
        zc = C[:Kk + 2].copy(); zc[0] += 1.0; zc[1] -= 1.0
        Zc = np.empty_like(zc); Zc[0] = zc[0]; Zc[1:] = zc[1:] - zc[:-1]
        tau_term = DM[Kk] - (DM[Kk - 1] if Kk >= 1 else 0.0)
        PM = P[:Kk + 1] + M[:Kk + 1]                    # now includes the new P[Kk]
        zmu = np.einsum("ibn,ibn->bn", i[:, None, None] * M[:Kk + 1], Zc[Kk + 1:0:-1])
        rest = tau_term - zmu - _cs(V, M, Kk) - _cs(C, M, Kk) + 0.5 * n * _cs(C, PM, Kk)
        M[Kk + 1] = rest / (Zc[0] * (Kk + 1))
        DM[Kk + 1] = self.D(M[Kk + 1], omega)
    return P[:K + 1], M[:K + 1], A[:K + 1], C[:K + 1]


def _D_unit_mat(self, f):
    """d/dtau (omega = 1) as one dense 72x72 GEMM instead of rfft/irfft (BLAS, SIMD)."""
    Dm = self.__dict__.get("_Dm_unit")
    if Dm is None:
        Dm = _orig["_D_unit"](self, np.eye(self.N)); self._Dm_unit = Dm
    return f @ Dm

def _trunc_mat(self, f):
    """Spectral truncation to k <= kc as one dense GEMM."""
    T = self.__dict__.get("_Tmat")
    if T is None:
        T = _orig["trunc"](self, np.eye(self.N)); self._Tmat = T
    return f @ T


_orig = {}
def install():
    if _orig: return
    import spec as _spec
    _orig["ssh_series"] = _dss.DSS.ssh_series; _orig["solve_periodic"] = _dss.DSS.solve_periodic
    _orig["_D_unit"] = _dss.DSS._D_unit; _orig["trunc"] = _spec.DSSSpec.trunc
    _dss.DSS.ssh_series = ssh_series_fast
    _dss.DSS.solve_periodic = solve_periodic_fast
    _dss.DSS._D_unit = _D_unit_mat
    _spec.DSSSpec.trunc = _trunc_mat

def uninstall():
    if not _orig: return
    import spec as _spec
    _dss.DSS.ssh_series = _orig.pop("ssh_series"); _dss.DSS.solve_periodic = _orig.pop("solve_periodic")
    _dss.DSS._D_unit = _orig.pop("_D_unit"); _spec.DSSSpec.trunc = _orig.pop("trunc")


def lm_svd3(S, z, tol=1e-13, maxjac=30, ncorr=3, verbose=False, jac=None, eps=3e-7, mus=None, res=None):
    """lm_svd2 with batched trials: all damping values are tried in one residual call, then each round of chord
    corrections for all of them in one more call. Returns best z, |F|, seconds."""
    from spec_c import jacobian_cs
    jac = jac or jacobian_cs
    mus = mus if mus is not None else 10.0 ** np.arange(-3, -13.5, -0.5)
    t0 = time.time()
    R = res or (lambda Zb: S.residual(Zb))
    F = R(z[None, :])[0]; nF = np.linalg.norm(F); stall = 0
    for it in range(maxjac):
        if nF < tol: break
        _, J = jac(S, z, eps)
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        filt = np.stack([s / (s * s + (mu * s[0]) ** 2) for mu in mus])           # (nmu, L)
        step = lambda Fb: -((U.T @ Fb.T).T * filt) @ Vt                           # rows: one step per mu
        Zt = z[None, :] + step(np.repeat(F[None, :], len(mus), 0))
        best = (nF, None, None, 0)
        for c in range(ncorr + 1):
            Ft = R(Zt); nt = np.linalg.norm(Ft, axis=1)
            for q in range(len(mus)):
                if np.isfinite(nt[q]) and nt[q] < best[0]: best = (nt[q], Zt[q].copy(), mus[q], c)
            if c == ncorr: break
            ok = np.isfinite(nt)
            Ft[~ok] = 0.0
            Zt = Zt + step(Ft)
        if best[1] is None: break
        prev = nF
        z = best[1]; F = R(z[None, :])[0]; nF = np.linalg.norm(F)
        stall = stall + 1 if (nF > 0.7 * prev and nF < 1e-8) else 0   # stagnation only counts near the floor
        if verbose: print(f"   jac {it}: |F| {nF:.3e} Delta {z[-1]:.12f} mu {best[2]:.1e} corr {best[3]}", flush=True)
        if stall >= 2: break
    return z, nF, time.time() - t0
