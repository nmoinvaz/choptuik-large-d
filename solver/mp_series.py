"""The horizon (SSH) Taylor series of dss.DSSMulti.ssh_series, in arbitrary precision (mpmath).

Question it answers: is the apparent radius of convergence of the horizon series, ~2.5/sqrt(n) in double precision,
a real singularity of the horizon-regular solution or rounding error amplified by the recursion? Same recursion,
with the tau functions as object arrays of mpf on the N-point grid, the unit derivative as an mp collocation matrix,
and the periodic solves by mp LU; the products that do not change between orders are cached.

    from mp_series import mp_ssh_series
    P, M, A, C = mp_ssh_series(n, m0, a0, omega, K, dps=50)       # m0, a0: tau grids (float or mp)
"""
import mpmath as mp
import numpy as np


def _dmat(N):
    """Unit Fourier differentiation matrix on N points (Nyquist dropped), mp."""
    th = [2 * mp.pi * j / N for j in range(N)]
    D = mp.matrix(N, N)
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = -2 * sum(k * mp.sin(k * (th[i] - th[j])) for k in range(1, N // 2)) / N
    return D


def _mv(Mt, f):
    N = len(f)
    return np.array([mp.fsum(Mt[i, j] * f[j] for j in range(N)) for i in range(N)], dtype=object)


def mp_ssh_series(n, m0, a0, omega, K, dps=50):
    mp.mp.dps = dps
    N = len(m0)
    n = mp.mpf(n)
    omega = mp.mpf(omega)
    Dm = _dmat(N)
    Dw = Dm * omega
    to = lambda v: np.array([mp.mpf(float(t)) if not isinstance(t, mp.mpf) else t for t in v], dtype=object)
    Z = np.array([mp.mpf(0)] * N, dtype=object)
    one = np.array([mp.mpf(1)] * N, dtype=object)
    P = [Z.copy() for _ in range(K + 2)]
    M = [Z.copy() for _ in range(K + 2)]
    A = [Z.copy() for _ in range(K + 2)]
    C = [Z.copy() for _ in range(K + 2)]
    A[0] = to(a0); C[0] = one.copy(); M[0] = to(m0)

    def cs(X, Y, k):
        s = Z.copy()
        for j in range(k + 1):
            s = s + X[j] * Y[k - j]
        return s

    def D(f):
        return _mv(Dw, f)

    def solve_periodic(kappa, rhs):
        Mt = Dw.copy()
        for i in range(N):
            Mt[i, i] += kappa[i]
        sol = mp.lu_solve(Mt, mp.matrix(list(rhs)))
        return np.array([sol[i] for i in range(N)], dtype=object)

    AA, QQ, A2m1 = [], [], []          # caches: (A A)_j, (P P + M M)_j, (A A)_j - delta_j0
    for Kk in range(K + 1):
        # c_{K+1}
        AA.append(cs(A, A, Kk))
        A2m1.append(AA[Kk] - (one if Kk == 0 else Z))
        C[Kk + 1] = (Kk * C[Kk] - (n - 1) * cs(C, A2m1, Kk)) / (Kk + 1)
        # p_K
        w = [one - C[0]] + [-one - C[1]] + [-C[j] for j in range(2, Kk + 2)]
        W = [w[0]] + [w[j] - w[j - 1] for j in range(1, Kk + 2)]
        V = [(j + 1) * C[j + 1] - j * C[j] for j in range(Kk + 1)]
        PM = [P[j] + M[j] for j in range(Kk + 1)]
        tau_term = -D(P[Kk - 1]) if Kk >= 1 else Z
        wpu = Z.copy()
        for j in range(1, Kk + 2):
            if Kk - j + 1 >= 0:
                wpu = wpu + W[j] * (Kk - j + 1) * P[Kk - j + 1]
        r = tau_term - wpu + cs(V, P, Kk) + cs(C, P, Kk) - n / 2 * cs(C, PM, Kk)
        kappa = Kk * (one + C[1]) + C[1] + one - n / 2 * one
        P[Kk] = solve_periodic(kappa, -r)
        if Kk == K:
            break
        # a_{K+1}
        QQ.append(cs(P, P, Kk) + cs(M, M, Kk))
        A3 = cs(A, AA, Kk)
        aG = cs(A, QQ, Kk) + (n - 1) * (A[Kk] - A3)
        A[Kk + 1] = (Kk * A[Kk] - aG / 2) / (Kk + 1)
        # m_{K+1}
        z = [one + C[0], -one + C[1]] + [C[j] for j in range(2, Kk + 2)]
        Zc = [z[0]] + [z[j] - z[j - 1] for j in range(1, Kk + 2)]
        PM = [P[j] + M[j] for j in range(Kk + 1)]
        tau_term = D(M[Kk]) - (D(M[Kk - 1]) if Kk >= 1 else Z)
        zmu = Z.copy()
        for j in range(1, Kk + 2):
            if Kk - j + 1 >= 0:
                zmu = zmu + Zc[j] * (Kk - j + 1) * M[Kk - j + 1]
        rest = tau_term - zmu - cs(V, M, Kk) - cs(C, M, Kk) + n / 2 * cs(C, PM, Kk)
        M[Kk + 1] = rest / (Zc[0] * (Kk + 1))
    return P[:K + 1], M[:K + 1], A[:K + 1], C[:K + 1]


def harmonic(f, k=1):
    """k-th Fourier coefficient (complex) of an mp grid function."""
    N = len(f)
    return mp.fsum(f[j] * mp.exp(-2j * mp.pi * k * j / N) for j in range(N)) / N


def evaluate(S, u):
    out = np.array([mp.mpf(0)] * len(S[0]), dtype=object)
    for k in range(len(S) - 1, -1, -1):
        out = out * u + S[k]
    return out
