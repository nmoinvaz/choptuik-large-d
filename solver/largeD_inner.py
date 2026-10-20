"""Inner problem at x ~ sqrt(n): xi = x / sqrt(n), Dd = p - m = O(1), sigma = sqrt(n) (p + m), c = O(1), b - 1 = O(1/n).

Leading-order inner equations (L = xi^2 d_xi + omega xi d_theta):
    (I1)  L Dd = c sigma
    (I2)  L sigma = c xi Dd_xi + c (Dd^2/2 - 1) Dd
    (I3)  xi c_xi = c Dd^2 / 2           (beta = n (b-1) = p^2 + m^2 = Dd^2/2, G = c beta)
Formal solution regular at xi = 0 matching the outer P0 = x F:
    Dd = sum_j d_j xi^(2j+1), sigma = sum_j s_j xi^(2j+2), c = sum_j g_j xi^(2j), d_0 = 2F, g_0 = 1.
Run as a script: checks d_k, g_k, s_(k-1) against the top-degree coefficients of the outer D_k, G_k, S_k
(largeD_late_orders.py) for F = 2.72 sin(theta), orders 1-10.
Recursion:  s_j = (2j+1) d_j + w d_j' - sum_{i>=1} g_i s_{j-i}
            g_{j+1} = (1/(2j+2)) sum_{i<=j} g_i e_{j+1-i},  e_m = (1/2) sum_{i+l=m-1} d_i d_l
            d_{j+1} = [(2j+2) s_j + w s_j' - sum_{i>=1} g_i 2(j+1-i) d_{j+1-i} - [c E d]_{j+1}] / (2j+2)
The theta functions are sampled on NT points (spectral derivative); `keep` optionally projects every order onto
the harmonics |mu| <= keep (Galerkin truncation).
"""
import numpy as np

NT = 256
KK = np.fft.fftfreq(NT, 1 / NT)
TH = 2 * np.pi * np.arange(NT) / NT


def dth(f):
    return np.fft.ifft(np.fft.fft(f) * (1j * KK))


def proj(f, keep=None, parity=1):
    fh = np.fft.fft(f)
    mask = (np.abs(KK) % 2 == parity)
    if keep is not None:
        mask &= np.abs(KK) <= keep
    return np.fft.ifft(fh * mask)


def harm(f, mu):
    return np.fft.fft(f)[mu % NT] / NT


def inner(Fg, w, J, keep=None, gravity=True, fmax=1):
    """fmax: highest harmonic of F; order j then holds harmonics up to fmax (2j+1), and higher ones (roundoff, amplified by
    w d_theta at every order) are projected out"""
    kp = (lambda j: min(keep, fmax * (2 * j + 1)) if keep is not None else fmax * (2 * j + 1))
    kpe = (lambda j: min(keep + 1, fmax * (2 * j + 1)) if keep is not None else fmax * (2 * j + 1))
    d = [2 * Fg.astype(complex)]
    s, g, e = [], [np.ones(NT, complex)], [np.zeros(NT, complex)]
    for j in range(J):
        sj = (2 * j + 1) * d[j] + w * dth(d[j])
        for i in range(1, j + 1):
            sj = sj - g[i] * s[j - i]
        s.append(proj(sj, kp(j)))
        # e_{j+1}, g_{j+1}
        m = j + 1
        em = 0.5 * sum(d[i] * d[m - 1 - i] for i in range(m)) if gravity else np.zeros(NT, complex)
        e.append(proj(em, kpe(m), 0))
        gm = sum(g[i] * e[m - i] for i in range(m)) / (2 * m)
        g.append(proj(gm, kpe(m), 0))
        if not gravity:
            g[-1] = np.zeros(NT, complex)
        num = (2 * j + 2) * s[j] + w * dth(s[j])
        for i in range(1, j + 2):
            num = num - g[i] * 2 * (j + 1 - i) * d[j + 1 - i]
        # [c E d]_{j+1} = sum g_i e_l d_q, i + l + q = j+1, l >= 1
        ced = np.zeros(NT, complex)
        for l in range(1, j + 2):
            for i in range(0, j + 2 - l):
                q = j + 1 - l - i
                ced = ced + g[i] * e[l] * d[q]
        num = num - ced
        d.append(proj(num / (2 * j + 2), kp(j + 1)))
    return d, s, g


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import largeD_late_orders as lo
    w = 2 * np.pi / 1.64389375
    th = 2 * np.pi * np.arange(lo.NT) / lo.NT
    Fg = 2.72 * np.sin(th)
    K = 10
    S, D, G, B, _ = lo.run(Fg, w, K, verbose=False)
    # inner on the same grid
    global_NT = lo.NT
    d, s, g = inner(np.interp(TH, np.r_[th, 2 * np.pi], np.r_[Fg, Fg[0]]) if NT != lo.NT else Fg, w, K)
    for k in range(1, K + 1):
        top_D = D[k][2 * k + 1]
        top_G = G[k][2 * k]
        top_S = S[k][2 * k]
        rD = np.abs(top_D - d[k].real).max() / np.abs(d[k]).max()
        rG = np.abs(-top_G - g[k].real).max() / np.abs(g[k]).max()
        rS = np.abs(top_S - s[k - 1].real).max() / np.abs(s[k - 1]).max()
        print(f"order {k:2d}: top-degree of outer D_k vs inner d_k {rD:.1e}; -G_k vs g_k {rG:.1e}; S_k vs s_(k-1) {rS:.1e}"
              f"   |d_k| {np.abs(d[k]).max():.3e}", flush=True)
