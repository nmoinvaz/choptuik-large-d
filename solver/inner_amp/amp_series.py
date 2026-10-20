"""Inner problem at x ~ sqrt(n), expanded in the amplitude eps of F, with every harmonic kept exactly.

Inner equations (README, "The inner problem at large x"), L = xi^2 d_xi + omega xi d_theta:
    L y = c sigma,   L sigma = c xi y_xi + c (y^2/2 - 1) y,   xi c_xi = c y^2/2,     y = Delta_P = p - m.
Formal solution y = sum_j d_j xi^(2j+1), sigma = sum_j s_j xi^(2j+2), c = sum_j g_j xi^(2j), d_0 = 2F, g_0 = 1:
    s_j     = (2j+1+w d) d_j - sum_{i>=1} g_i s_{j-i}
    e_m     = 1/2 sum_{i+l=m-1} d_i d_l,      g_m = 1/(2m) sum_{i<m} g_i e_{m-i}
    d_{j+1} = (2j+2+w d) s_j / (2j+2) - sum_{i>=1} g_i d_{j+1-i}          (d = d/dtheta)
F = eps sin(theta) + sum_{nu>=3} f_nu(eps) e^{i nu theta} (f_nu = O(eps^nu)). Every theta function is an array
X[k, mu], k = power of eps (0..K), mu = harmonic (-K..K): harmonic mu first appears at order eps^|mu|, so cropping
products at |mu| <= K is exact through eps^K -- no Galerkin truncation error at any order kept.
Coefficients are stored scaled, X^_j = X_j / (2^j j!), so products carry binomial weights and nothing overflows.

    from amp_series import series;  d, s, g = series(w, K, J, fextra)
    python3 amp_series.py      check against solver/largeD_inner.py at eps = 0.3 (exact through the kept orders)
"""
import numpy as np
from scipy.signal import convolve2d
from math import comb


def mul(X, Y, K):
    Z = convolve2d(X, Y)
    return Z[:K + 1, K:3 * K + 1]


def series(w, K, J, fextra=None):
    """fextra: dict {(k, mu): value} of extra coefficients of F (k >= 3, |mu| >= 3); returns scaled d, s, g lists."""
    H = K
    mus = np.arange(-H, H + 1)
    Dth = 1j * w * mus[None, :]                         # w d/dtheta
    d0 = np.zeros((K + 1, 2 * H + 1), complex)
    d0[1, H + 1] = -1j                                   # 2 * (eps sin theta): e^{i theta} coefficient -i
    d0[1, H - 1] = 1j
    for (k, mu), v in (fextra or {}).items():
        d0[k, H + mu] += 2 * v
    one = np.zeros_like(d0); one[0, H] = 1.0
    zero = np.zeros_like(d0)
    d, s, g, e = [d0], [], [one], [zero]
    for j in range(J):
        sj = (2 * j + 1 + Dth) * d[j]
        for i in range(1, j + 1):
            sj = sj - mul(g[i], s[j - i], K) / comb(j, i)
        s.append(sj)
        m = j + 1
        em = zero.copy()
        for i in range(0, m):                          # e_m = 1/2 sum_{i+l=m-1} d_i d_l -> scaled 1/4 sum d^_i d^_l i! l!/m!
            em = em + mul(d[i], d[m - 1 - i], K) / (m * comb(m - 1, i))
        e.append(em / 4)
        gm = zero.copy()
        for i in range(0, m):
            gm = gm + mul(g[i], e[m - i], K) / comb(m, i)
        g.append(gm / (2 * m))
        dn = (2 * j + 2 + Dth) * s[j] / (4 * (j + 1) ** 2)
        for i in range(1, j + 2):
            dn = dn - mul(g[i], d[j + 1 - i], K) / comb(j + 1, i)
        d.append(dn)
    return d, s, g


if __name__ == "__main__":
    # unscaled check against solver/largeD_inner.py-style recursion at a finite eps (orders 0..6)
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # solver/ (largeD_inner.py)
    import largeD_inner as LI
    from math import factorial
    w = 2 * np.pi / 1.64389375
    K, J = 7, 6
    d, s, g = series(w, K, J)
    eps = 0.3
    dN, sN, gN = LI.inner(eps * np.sin(LI.TH), w, J)
    for j in range(J + 1):
        dj = d[j] * 2 ** j * factorial(j)
        val1 = sum(dj[k, K + 1] * eps ** k for k in range(K + 1))
        ref = LI.harm(dN[j], 1)
        print(j, val1, ref, abs(val1 - ref) / abs(ref))
