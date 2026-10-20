"""Inner problem at x ~ sqrt(n), exact harmonic bookkeeping in ball arithmetic (python-flint acb).

Equations (README, "The inner problem at x ~ sqrt n"), L = xi^2 d_xi + omega xi d_theta:
    L Dd = c sigma,   L sigma = c xi Dd_xi + c (Dd^2/2 - 1) Dd,   xi c_xi = c Dd^2/2.
Formal solution Dd = sum_j d_j xi^(2j+1), sigma = sum_j s_j xi^(2j+2), c = sum_j g_j xi^(2j), d_0 = 2F, g_0 = 1.
With E = Dd^2/2 = sum_m e_m xi^(2m) and [c E]_m = 2m g_m (from the c equation) the recursion is
    s_j     = (2j+1) d_j + w d_j' - sum_{i=1..j} g_i s_{j-i}
    e_m     = (1/2) sum_{i+l=m-1} d_i d_l,      g_m = (1/(2m)) sum_{i=0..m-1} g_i e_{m-i}
    d_{j+1} = s_j + w s_j' / (2j+2) - sum_{i=1..j+1} g_i d_{j+1-i}.
Every theta function is a list of harmonics mu = -M..M (Galerkin truncation |mu| <= M, products exact inside it).
gravity=False sets c = 1 and drops the cubic term (then d_j = 2^j (alpha)_j (beta)_j / j! d_0 per harmonic).

    from inner_series import series
    d = series(F={1: -1j*A/2, -1: 1j*A/2}, w=..., J=200, M=15, prec=600)   # d[j][mu + M]
"""
import os
from flint import acb, acb_poly, ctx


def series(F, w, J, M, prec=600, gravity=True, verbose=False):
    ctx.prec = prec
    N = 2 * M + 1
    w = acb(w) if not isinstance(w, acb) else w
    I = acb(0, 1)
    dfac = [I * w * (mu - M) for mu in range(N)]          # d/dtheta -> i mu, times w

    def lst(dct):
        out = [acb(0)] * N
        for mu, v in dct.items():
            if abs(mu) <= M:
                out[mu + M] = acb(complex(v).real, complex(v).imag) if not isinstance(v, acb) else v
        return out

    def mul(pa, pb):
        c = (pa * pb).coeffs()
        c = c + [acb(0)] * (4 * M + 1 - len(c))
        return c[M:3 * M + 1]

    def add(a, b, fb=1):
        return [x + fb * y for x, y in zip(a, b)]

    d = [[2 * v for v in lst(F)]]
    dp = [acb_poly(d[0])]
    g = [lst({0: 1})]
    gp = [acb_poly(g[0])]
    s, sp, e = [], [], [None]
    for j in range(J):
        sj = [(2 * j + 1) * d[j][k] + dfac[k] * d[j][k] for k in range(N)]
        if gravity:
            for i in range(1, j + 1):
                sj = add(sj, mul(gp[i], sp[j - i]), -1)
        s.append(sj); sp.append(acb_poly(sj))
        m = j + 1
        if gravity:
            em = [acb(0)] * N
            for i in range(0, m):
                l = m - 1 - i
                if i > l:
                    break
                t = mul(dp[i], dp[l])
                em = add(em, t, 1 if i == l else 2)
            em = [v / 2 for v in em]
            e.append(em)
            ep = [None] + [acb_poly(x) for x in e[1:]]
            gm = [acb(0)] * N
            for i in range(0, m):
                gm = add(gm, mul(gp[i], acb_poly(e[m - i])))
            gm = [v / (2 * m) for v in gm]
        else:
            gm = [acb(0)] * N
        g.append(gm); gp.append(acb_poly(gm))
        dn = [sj[k] + dfac[k] * sj[k] / (2 * j + 2) for k in range(N)]
        if gravity:
            for i in range(1, j + 2):
                dn = add(dn, mul(gp[i], dp[j + 1 - i]), -1)
        d.append(dn); dp.append(acb_poly(dn))
        if verbose and j % 25 == 0:
            print(f"   order {j}", flush=True)
    return d, s, g


if __name__ == "__main__":
    # cross-check against the double-precision recursion in solver/largeD_inner.py (gravity on, F = 2.72 sin theta)
    import sys
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import largeD_inner as LI
    w = 2 * np.pi / 1.64389375
    A = 2.72
    K = 8
    dN, sN, gN = LI.inner(A * np.sin(LI.TH), w, K)
    d, s, g = series({1: -0.5j * A, -1: 0.5j * A}, w, K, M=2 * K + 3, prec=200)
    M = 2 * K + 3
    for k in range(K + 1):
        err = max(abs(complex(d[k][mu + M].mid()) - LI.harm(dN[k], mu)) for mu in range(-2 * k - 1, 2 * k + 2, 2))
        print(f"order {k}: max harmonic difference vs double-precision recursion {err:.2e} (|d_k| ~ {np.abs(dN[k]).max():.2e})")
