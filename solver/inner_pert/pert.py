"""Growing modes of the D = infinity inner problem (README, "The inner problem at large x"), Galerkin-truncated.

Background (B's recursion, solver/inner_borel/inner_series.py): with L = xi^2 d_xi + omega xi d_theta,
    L Dd = c sigma,   L sigma = (xi d_xi - 1)(c Dd),   xi c_xi = c Dd^2/2.
Perturbations ~ e^{lambda tau}: L -> L_lam = xi^2 d_xi + xi (lambda + omega d_theta) on the perturbation fields
(there is no tau-derivative in the c equation):
    L_lam dDd = dc sigma + c dsig,   L_lam dsig = (xi d_xi - 1)(dc Dd + c dDd),   xi dc_xi = dc Dd^2/2 + c Dd dDd.
Series dDd = sum a_j xi^(2j+1), dsig = sum b_j xi^(2j+2), dc = sum h_j xi^(2j), a_0 = 2 f, h_0 = 0:
    b_j     = (2j+1+lam) a_j + w a_j' - sum_{i>=1} (g_i b_{j-i} + h_i s_{j-i})
    h_m     = (1/2m) (sum_{i<m} h_i e_{m-i} + sum_i g_i q_{m-i}),   q_k = sum_{i+l=k-1} d_i a_l
    a_{j+1} = [(2j+2+lam) b_j + w b_j'] / (2j+2) - sum_{i>=1} g_i a_{j+1-i} - sum_{i>=1} h_i d_{j+1-i}.
Fast modes of the perturbation: the background transport exponents shifted, gamma_k + lam (omega d_theta enters the
fast transport twice, xi a_xi + 2 (lam + omega d_theta) a = 2 F^2 a, so the exponent shifts by lam; this matches the
exact lam = 1 mode X Z, whose fast part is X applied to the background's, one power of u lower). The growth rate of a mode is a lam for which a real profile f has vanishing
fast constants.

    from pert import series_pert
"""
from flint import acb, acb_poly, ctx


def series_pert(F, f, w, lam, J, M, prec=900):
    ctx.prec = prec
    N = 2 * M + 1
    w = acb(w) if not isinstance(w, acb) else w
    lam = acb(lam) if not isinstance(lam, acb) else lam
    I = acb(0, 1)
    dfac = [I * w * (mu - M) for mu in range(N)]

    def lst(dct):
        out = [acb(0)] * N
        for mu, v in dct.items():
            if abs(mu) <= M:
                out[mu + M] = v if isinstance(v, acb) else acb(complex(v).real, complex(v).imag)
        return out

    def mul(pa, pb):
        c = (pa * pb).coeffs()
        c = c + [acb(0)] * (4 * M + 1 - len(c))
        return c[M:3 * M + 1]

    def add(a, b, fb=1):
        return [x + fb * y for x, y in zip(a, b)]

    Z = [acb(0)] * N
    d = [[2 * v for v in lst(F)]]; dp = [acb_poly(d[0])]
    g = [lst({0: 1})]; gp = [acb_poly(g[0])]
    a = [[2 * v for v in lst(f)]]; ap = [acb_poly(a[0])]
    h = [list(Z)]; hp = [acb_poly(h[0])]
    s, sp, e, b, bp = [], [], [None], [], []
    for j in range(J):
        # background
        sj = [(2 * j + 1) * d[j][k] + dfac[k] * d[j][k] for k in range(N)]
        for i in range(1, j + 1):
            sj = add(sj, mul(gp[i], sp[j - i]), -1)
        s.append(sj); sp.append(acb_poly(sj))
        # perturbation b_j
        bj = [(2 * j + 1 + lam) * a[j][k] + dfac[k] * a[j][k] for k in range(N)]
        for i in range(1, j + 1):
            bj = add(bj, mul(gp[i], bp[j - i]), -1)
            bj = add(bj, mul(hp[i], sp[j - i]), -1)
        b.append(bj); bp.append(acb_poly(bj))
        m = j + 1
        em = list(Z)
        for i in range(0, m):
            l = m - 1 - i
            if i > l:
                break
            t = mul(dp[i], dp[l])
            em = add(em, t, 1 if i == l else 2)
        em = [v / 2 for v in em]
        e.append(em)
        gm = list(Z)
        for i in range(0, m):
            gm = add(gm, mul(gp[i], acb_poly(e[m - i])))
        gm = [v / (2 * m) for v in gm]
        g.append(gm); gp.append(acb_poly(gm))
        # perturbation h_m: q_k = sum_{i+l=k-1} d_i a_l
        hm = list(Z)
        for i in range(1, m):
            hm = add(hm, mul(hp[i], acb_poly(e[m - i])))
        for i in range(0, m):
            k = m - i
            qk = list(Z)
            for i2 in range(0, k):
                qk = add(qk, mul(dp[i2], ap[k - 1 - i2]))
            hm = add(hm, mul(gp[i], acb_poly(qk)))
        hm = [v / (2 * m) for v in hm]
        h.append(hm); hp.append(acb_poly(hm))
        # background d_{j+1}
        dn = [sj[k] + dfac[k] * sj[k] / (2 * j + 2) for k in range(N)]
        for i in range(1, j + 2):
            dn = add(dn, mul(gp[i], dp[j + 1 - i]), -1)
        d.append(dn); dp.append(acb_poly(dn))
        # perturbation a_{j+1}
        an = [((2 * j + 2 + lam) * bj[k] + dfac[k] * bj[k]) / (2 * j + 2) for k in range(N)]
        for i in range(1, j + 2):
            an = add(an, mul(gp[i], ap[j + 1 - i]), -1)
            an = add(an, mul(hp[i], dp[j + 1 - i]), -1)
        a.append(an); ap.append(acb_poly(an))
    return d, a, g, h
