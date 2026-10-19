"""f64 prototype of the high-precision solver's building blocks (checks the recursions before the Rust port).

Half-grid collocation: N = 2M points theta_j = 2 pi j / N, only j < M stored; p, m odd (f(theta + pi) = -f(theta)),
a, c even. d/dtau = omega d/dtheta, omega = 2 pi / Delta. Taylor recursion in x around a regular point x0 for
    x c_x = (n-1)(c a^2 - c)                       = G
    2 x a_x = a (p^2 + m^2) + (n-1)(a - a a^2)     = H
    x (x - c) p_x = -omega x p_th + G p - c p + (n/2) c (p + m)
    x (x + c) m_x = -omega x m_th - G m + c m - (n/2) c (p + m)
"""
import numpy as np


def dmat_odd(M):
    """Folded d/dtheta for odd (antiperiodic) fields on the half grid: full spectral D, f_{j+M} = -f_j."""
    N = 2 * M
    th = 2 * np.pi * np.arange(N) / N
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = 0.5 * (-1) ** (i - j) / np.tan((th[i] - th[j]) / 2)
    return D[:M, :M] - D[:M, M:]


def cs(X, Y, k):
    return sum(X[j] * Y[k - j] for j in range(k + 1))


def taylor_coeffs(s, x0, n, omega, Dm, K):
    p0, m0, a0, c0 = s
    P, Mm, A, C = [p0], [m0], [a0], [c0]
    A2, Q2, G, H = [], [], [], []
    Pth, Mth = [], []
    for k in range(K):
        A2.append(cs(A, A, k))
        Q2.append(cs(P, P, k) + cs(Mm, Mm, k))
        G.append((n - 1) * (cs(C, A2, k) - C[k]))
        H.append(cs(A, Q2, k) + (n - 1) * (A[k] - cs(A, A2, k)))
        Pth.append(Dm @ P[k]); Mth.append(Dm @ Mm[k])
        xp = x0 * Pth[k] + (Pth[k - 1] if k else 0)
        xm = x0 * Mth[k] + (Mth[k - 1] if k else 0)
        PM = [P[j] + Mm[j] for j in range(k + 1)]
        Rp = -omega * xp + cs(G, P, k) - cs(C, P, k) + n / 2 * cs(C, PM, k)
        Rm = -omega * xm - cs(G, Mm, k) + cs(C, Mm, k) - n / 2 * cs(C, PM, k)
        c_new = (G[k] - k * C[k]) / ((k + 1) * x0)
        a_new = (H[k] / 2 - k * A[k]) / ((k + 1) * x0)
        C.append(c_new); A.append(a_new)
        # W = x (x - c), V = x (x + c): coefficients j = 0..k (need c_0..c_k)
        def xc(j):
            return x0 * C[j] + (C[j - 1] if j else 0)
        def x2(j):
            return [x0 * x0, 2 * x0, 1.0][j] if j <= 2 else 0.0
        W = [x2(j) - xc(j) for j in range(k + 1)]
        V = [x2(j) + xc(j) for j in range(k + 1)]
        sp = sum(W[j] * (k - j + 1) * P[k - j + 1] for j in range(1, k + 1))
        sm = sum(V[j] * (k - j + 1) * Mm[k - j + 1] for j in range(1, k + 1))
        P.append((Rp - sp) / (W[0] * (k + 1)))
        Mm.append((Rm - sm) / (V[0] * (k + 1)))
    return P, Mm, A, C


def horner(S, h):
    out = 0 * S[0]
    for c in S[::-1]:
        out = out * h + c
    return out


def taylor_step(s, x0, h, n, omega, Dm, K):
    P, Mm, A, C = taylor_coeffs(s, x0, n, omega, Dm, K)
    return np.stack([horner(Q, h) for Q in (P, Mm, A, C)]), (P, Mm, A, C)


def rhs(x, s, n, omega, Dm):
    p, m, a, c = s
    a2 = a * a
    cx = c * (n - 1) * (a2 - 1) / x
    ax = a * ((p * p + m * m) + (n - 1) * (1 - a2)) / (2 * x)
    pc = n * c * (p + m) / (2 * x)
    px = (-omega * (Dm @ p) + cx * p - c * p / x + pc) / (x - c)
    mx = (-omega * (Dm @ m) - cx * m + c * m / x - pc) / (x + c)
    return np.stack([px, mx, ax, cx])


def rk4(s, xa, xb, steps, n, omega, Dm):
    h = (xb - xa) / steps
    x = xa
    for _ in range(steps):
        k1 = rhs(x, s, n, omega, Dm)
        k2 = rhs(x + h / 2, s + h / 2 * k1, n, omega, Dm)
        k3 = rhs(x + h / 2, s + h / 2 * k2, n, omega, Dm)
        k4 = rhs(x + h, s + h * k3, n, omega, Dm)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x += h
    return s


def resample(F, M):
    """Full-grid samples F (any N0) -> half-grid samples on the N = 2M grid (Fourier interpolation)."""
    N0 = len(F)
    c = np.fft.rfft(F) / N0
    N = 2 * M
    th = 2 * np.pi * np.arange(M) / N
    k = np.arange(len(c))
    w = np.where((k == 0) | ((N0 % 2 == 0) & (k == N0 // 2)), 1.0, 2.0)
    return (w * c[None, :] * np.exp(1j * k[None, :] * th[:, None])).real.sum(1)


def wsum(X, Y, k, off):
    return sum((i * X[i]) * Y[off - i] for i in range(k + 1))


def ssh_series(m0, a0, n, omega, Dm, Ks):
    """Horizon series in u = 1 - x (transliteration of the Rust kernel's ssh_series)."""
    M = len(m0)
    z = np.zeros(M)
    L = Ks + 2
    pp = [z.copy() for _ in range(L)]; mm = [z.copy() for _ in range(L)]
    aa = [z.copy() for _ in range(L)]; cc = [z.copy() for _ in range(L)]
    a2 = [z.copy() for _ in range(L)]; q2 = [z.copy() for _ in range(L)]
    dp = [z.copy() for _ in range(L)]; dm = [z.copy() for _ in range(L)]
    pm = [z.copy() for _ in range(L)]; vv = [z.copy() for _ in range(L)]
    ww = [z.copy() for _ in range(L)]; zc = [z.copy() for _ in range(L)]
    aa[0] = a0.copy(); cc[0] = np.ones(M); mm[0] = m0.copy(); a2[0] = aa[0] * aa[0]
    dm[0] = omega * (Dm @ mm[0])
    wf = lambda i: -cc[0] + 1 if i == 0 else (-cc[1] - 1 if i == 1 else -cc[i])
    zf = lambda i: cc[0] + 1 if i == 0 else (cc[1] - 1 if i == 1 else cc[i])
    ww[0] = wf(0); zc[0] = zf(0)
    for kk in range(Ks + 1):
        t = cs(cc, a2, kk)
        cc[kk + 1] = (cc[kk] * kk - (t - cc[kk]) * (n - 1)) / (kk + 1)
        ww[kk + 1] = wf(kk + 1) - wf(kk)
        vv[kk] = cc[kk + 1] * (kk + 1) - cc[kk] * kk
        zc[kk + 1] = zf(kk + 1) - zf(kk)
        pm[kk] = pp[kk] + mm[kk]
        wpu = wsum(pp, ww, kk, kk + 1)
        tau = -dp[kk - 1] if kk >= 1 else z
        r = tau - wpu + cs(vv, pp, kk) + cs(cc, pp, kk) - cs(cc, pm, kk) * (0.5 * n)
        kappa = (cc[1] + 1) * kk + cc[1] + 1 - 0.5 * n
        pp[kk] = np.linalg.solve(omega * Dm + np.diag(kappa), -r)
        dp[kk] = omega * (Dm @ pp[kk])
        q2[kk] = cs(pp, pp, kk) + cs(mm, mm, kk)
        if kk == Ks:
            break
        a3 = cs(aa, a2, kk)
        ag = cs(aa, q2, kk) + (aa[kk] - a3) * (n - 1)
        aa[kk + 1] = (aa[kk] * kk - ag * 0.5) / (kk + 1)
        a2[kk + 1] = cs(aa, aa, kk + 1)
        tau = dm[kk] - dm[kk - 1] if kk >= 1 else dm[kk]
        pm[kk] = pp[kk] + mm[kk]
        zmu = wsum(mm, zc, kk, kk + 1)
        rest = tau - zmu - cs(vv, mm, kk) - cs(cc, mm, kk) + cs(cc, pm, kk) * (0.5 * n)
        mm[kk + 1] = rest / (zc[0] * (kk + 1))
        dm[kk + 1] = omega * (Dm @ mm[kk + 1])
    k1 = Ks + 1
    return pp[:k1], mm[:k1], aa[:k1], cc[:k1]


def centre_series(p1, c0, n, omega, Dm, K):
    """Centre series in x (transliteration of the Rust kernel's centre_state)."""
    M = len(p1)
    z = np.zeros(M)
    pp = [z.copy() for _ in range(K + 1)]; mm = [z.copy() for _ in range(K + 1)]
    aa = [z.copy() for _ in range(K + 1)]; cc = [z.copy() for _ in range(K + 1)]
    aa[0] = np.ones(M); cc[0] = c0.copy(); pp[1] = p1.copy(); mm[1] = -p1
    for k in range(2, K + 1):
        pm = [pp[j] + mm[j] for j in range(K + 1)]
        q2 = [cs(pp, pp, j) + cs(mm, mm, j) for j in range(k + 1)]
        s = cs(aa, q2, k)
        a2 = [cs(aa, aa, j) for j in range(k + 1)]
        t3 = cs(aa, a2, k)
        aa[k] = (s * 0.5 - t3 * (0.5 * (n - 1))) / (k + n - 1)
        a2m1 = [cs(aa, aa, j) for j in range(k + 1)]
        a2m1[0] = a2m1[0] - 1
        cc[k] = cs(cc, a2m1, k) * (n - 1) / k
        dpk = omega * (Dm @ pp[k - 1]); dmk = omega * (Dm @ mm[k - 1])
        xj = lambda j: -cc[0] if j == 1 else -cc[j - 1] + (1.0 if j == 2 else 0.0)
        yj = lambda j: cc[0] if j == 1 else cc[j - 1] + (1.0 if j == 2 else 0.0)
        xpx = sum(xj(j) * (k - j + 1) * pp[k - j + 1] for j in range(1, k + 1))
        xmx = sum(yj(j) * (k - j + 1) * mm[k - j + 1] for j in range(1, k + 1))
        xcxp = sum(cc[j] * j * pp[k - j] for j in range(1, k + 1))
        xcxm = sum(cc[j] * j * mm[k - j] for j in range(1, k + 1))
        cp, cm, cpm = cs(cc, pp, k), cs(cc, mm, k), cs(cc, pm, k)
        r3 = dpk + xpx - xcxp + cp - cpm * (0.5 * n)
        r4 = dmk + xmx + xcxm - cm + cpm * (0.5 * n)
        c = cc[0]
        m11 = c * (1 - k) - c * 0.5 * n; m12 = -c * 0.5 * n; m21 = c * 0.5 * n; m22 = c * (k - 1) + c * 0.5 * n
        det = m11 * m22 - m12 * m21
        pp[k] = (-r3 * m22 + r4 * m12) / det
        mm[k] = (-r4 * m11 + r3 * m21) / det
    return pp, mm, aa, cc
