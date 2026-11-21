"""The interior bound Cr, and the weights P and W, re-derived for the #1062 review (notes/P6_route3_review.md 2, 5).

Route: B3 reduces each interior term to the nu = 1 sequences t_n = |T^1_n|, with P_mu in front. Instead of the
log-convexity grouping of D_entire.md 6.3, the termwise majorant is summed directly for 10 <= j <= J1, in mpmath.iv:

  per triple:  c * sum_{i=3}^{j} G_i X_{j-i} t_{j-i}/t_{j+1} + sum_{i=3}^{j} G_i t_{j+1-i}/t_{j+1}
               + 1/(8(j+1)) sum_{l=1}^{j-1} t_l t_{j-l}/t_{j+1},          G_i = 1/(8i) sum_{l<i} t_l t_{i-1-l},

with X_l the prefactor bound of |2l + 1 + i lam3 w| and c that of |c_j|. Two modes:
  B4:    c = (2 + 2x + w/kappa)/(4(1 + x)), x = 1/j, kappa = 2;  X_l = 2l + 1 + (nu + 2) w for every outer label.
         The bound is A_j + nu B_j, and j^2 (A_j + nu B_j)/nu <= j^2 (A_j/3 + B_j) for nu >= 3.
  tight: |c_j| <= 1/2 + nu w/(4(j + 1));  X_l = 2l + 1 + |mu| w for the perturbation outside, sqrt((2l+1)^2 + w^2) for
         a background label outside; the sup over nu in [3, j/2] on a grid of nu intervals (the bound increases in nu).
Above J1 a log-convexity bound of my own (notes, 5.3) caps both modes.

Checks: r_m increasing on m >= 1; U_n <= 2 + t_1; the nu = 1 equality test (the majorant with exact prefactors
equals the sum of the moduli of every interior term, from model.py); B4's prefactor bounds pointwise.

    python3 interior.py [--json=PATH]      Cr (both modes), P, W (about 2 min); PATH gets the upper ends, rounded up
    python3 interior.py --mutate=nuprefac  M3, 3w for (nu + 2)w: the prefactor check fails, exit 1
    python3 interior.py --mutate=dropii    M5, the S^s term i = j dropped: the nu = 1 equality test fails, exit 1
"""
import sys
import time
from mpmath import mp, iv
import model

J0 = 10
KAPPA = 2


def W_iv():
    return 2 * iv.pi * 160000 / iv.mpf(263023)


class Tseq:
    def __init__(self, L, W):
        self.W = W
        w4 = W ** 2 / 4
        self.r2 = [((m + iv.mpf(1) / 2) ** 2 + w4) * ((m + 1) ** 2 + w4) / (m + 1) ** 2 for m in range(L + 1)]
        self.r = [iv.sqrt(x) for x in self.r2]
        t = [iv.mpf(1)]
        for m in range(L):
            t.append(t[-1] * self.r[m])
        self.t = t
        self.G = [iv.mpf(0)]
        for i in range(1, L + 1):
            acc = iv.mpf(0)
            for l in range(i):
                acc += t[l] * t[i - 1 - l]
            self.G.append(acc / (8 * i))


def sums(T, j, drop_ii=False):
    """the j-dependent sums of the majorant, each divided by t_{j+1}"""
    t, G, W = T.t, T.G, T.W
    N = t[j + 1]
    SsA = Ss1 = SsB = iv.mpf(0)
    top = j if not drop_ii else j - 1
    for i in range(3, top + 1):
        l = j - i
        x = G[i] * t[l]
        SsA += x * (2 * l + 1)
        Ss1 += x
        SsB += x * iv.sqrt((2 * l + 1) ** 2 + W ** 2)
    Sd = iv.mpf(0)
    for i in range(3, j + 1):
        Sd += G[i] * t[j + 1 - i]
    far = iv.mpf(0)
    for l in range(1, j):
        far += t[l] * t[j - l]
    return {'SsA': SsA / N, 'Ss1': Ss1 / N, 'SsB': SsB / N, 'Sd': Sd / N, 'far': far / (8 * (j + 1)) / N}


def b4_per_triple(S, j, W, nuprefac=False):
    """(A_j, B_j): the B4-mode majorant per triple is A_j + nu B_j"""
    x = iv.mpf(1) / j
    cbar = (2 + 2 * x + W / KAPPA) / (4 * (1 + x))
    lab = 3 * W if nuprefac else 2 * W                      # (nu + 2) w = 2w + nu w; M3 uses 3w and no nu w
    A = cbar * (S['SsA'] + lab * S['Ss1']) + S['Sd'] + S['far']
    B = cbar * W * S['Ss1'] if not nuprefac else iv.mpf(0)
    return A, B


def tight(S, j, W, k, n_out, N_tr, nus):
    """sup over nu intervals of j^2 M(nu)/nu, tight mode, all N_tr triples; n_out have the perturbation outside"""
    best = iv.mpf(0)
    for na, nb in nus:
        c = iv.mpf(1) / 2 + nb * W / (4 * (j + 1))
        pert = c * (S['SsA'] + (nb + k) * W * S['Ss1'])
        bg = c * S['SsB']
        M = n_out * pert + (N_tr - n_out) * bg + N_tr * (S['Sd'] + S['far'])
        v = j * j * M / na
        best = iv.mpf([0, max(best.b, v.b)])
    return best


def nu_grid(j, n=24):
    """intervals covering [3, j/2]"""
    lo, hi = mp.mpf(3), mp.mpf(j) / 2
    if hi <= lo:
        return [(iv.mpf(3), iv.mpf(3))]
    pts = [lo * (hi / lo) ** (mp.mpf(q) / n) for q in range(n + 1)]
    return [(iv.mpf(pts[q]), iv.mpf(pts[q + 1])) for q in range(n)]


def tail_bound(T, J1, W):
    """per triple, B4 mode, an upper bound of j^2 (A_j/3 + B_j) for every j > J1 (log-convexity, notes 5.3)"""
    t1, t2 = T.t[1], T.t[2]
    gam = (2 + t1) / 8
    X = iv.mpf([0, 1]) / (J1 + 1)                          # x = 1/j over j >= J1 + 1
    lj = iv.log(iv.mpf(J1 + 1)) / (J1 + 1)                 # ln j / j decreases for j >= 3
    q = lambda c: 1 / (1 - c * X)                          # j/(j - c) = 1/(1 - c x)
    h = iv.mpf(1) / 2
    Q1 = q(h) * q(-h)                                      # j t_{j-1}/t_{j+1} <= x Q1  (r_m >= m + 1/2)
    Q2 = q(3 * h) * Q1                                     # j^2 t_{j-2}/t_{j+1} <= x Q2
    Q3 = q(5 * h) * Q2                                     # j^3 t_{j-3}/t_{j+1} <= x Q3
    jj1 = q(iv.mpf(1))                                     # j/(j - 1)
    # each line is j^2 times the sum (notes 5.3): ends i = j, j - 1 exactly, the rest by log-convexity
    SsA = gam * (X * Q1 + 3 * t1 * X ** 2 * Q2 * jj1 + t2 * (2 + X) * Q3 * lj)
    Ss1 = gam * (X * Q1 + t1 * X ** 2 * Q2 * jj1 + t2 * Q3 * X * lj)
    Sd = gam * (t1 * X * Q1 + t2 * Q2 * lj)
    far = (2 * t1 * X * Q1 + t2 * X * Q2) / (8 * (1 + X))
    cbar = (2 + 2 * X + W / KAPPA) / (4 * (1 + X))
    A = cbar * (SsA + 2 * W * Ss1) + Sd + far
    B = cbar * W * Ss1
    return (A / 3 + B).b


def checks_monotone(T, W):
    ok = True
    mstar = W ** 2 / 2 - 1                                  # r_{m+1} > r_m for m > w^2/2 - 1 (notes 3)
    top = int(mp.ceil(mstar.b)) + 1
    for m in range(1, top + 1):
        if not (T.r2[m + 1] > T.r2[m]):
            ok = False
            print(f"  r_m not increasing at m = {m}")
    return ok, top, mstar


def check_conv(T, L):
    """U_n = sum_l t_l t_{n-l}/t_n <= 2 + t_1 for n <= L (proved in notes 3; checked here)"""
    worst = iv.mpf(0)
    for n in range(L + 1):
        Un = T.G[n + 1] * 8 * (n + 1) / T.t[n]
        v = Un / (2 + T.t[1])
        worst = iv.mpf([0, max(worst.b, v.b)])
    return worst


def check_prefactors(W, nuprefac):
    """B4's bounds against the exact prefactors, nu = 3..41 odd, j >= 2 nu"""
    ok = True
    for nv in range(3, 43, 2):
        for jj in (2 * nv, 2 * nv + 1, 3 * nv, 10 * nv):
            x = iv.mpf(1) / jj
            cbar = (2 + 2 * x + W / KAPPA) / (4 * (1 + x))
            cex = iv.sqrt((2 * jj + 2) ** 2 + (nv * W) ** 2) / (4 * (jj + 1))
            if not (cbar >= cex):
                ok = False
            for l in (0, 1, jj // 2, jj):
                bound = 2 * l + 1 + (3 * W if nuprefac else (nv + 2) * W)
                for lam in (nv - 2, nv, nv + 2, 1, -1):
                    ex = iv.sqrt((2 * l + 1) ** 2 + (lam * W) ** 2)
                    if not (bound >= ex):
                        ok = False
    return ok


def equality_test(T, drop_ii):
    """nu = 1, background-like labels: majorant with exact prefactors = sum of |interior terms| (model.py, mp)"""
    C, w = model.ctx('mp', 320)
    worst = mp.mpf(0)
    for mu, ntr in ((1, 6), (-1, 3)):
        for jj in (6, 12, 25):
            exact = model.interior_abs_terms(1, mu, jj, C, w)
            S = sums(T, jj, drop_ii)
            cj = iv.sqrt((2 * jj + 2) ** 2 + T.W ** 2) / (4 * (jj + 1))
            maj = ntr * (cj * S['SsB'] + S['Sd'] + S['far'])
            rel = abs(mp.mpf(maj.mid) - exact) / exact
            worst = max(worst, rel)
    return worst


def P_bound():
    """P = sup P_{nu+2} <= e^{pi w}; the closed form against mpmath's Gamma ratio, and the product below it"""
    W = W_iv()
    P = iv.exp(iv.pi * W)
    C, w = model.ctx('mp', 320)
    ok = True
    for nv in (3, 5, 9, 41):
        a, b = model.ab(C, w, nv)
        a2, b2 = model.ab(C, w, nv + 2)
        gr = abs(mp.gamma(a) * mp.gamma(b) / (mp.gamma(a2) * mp.gamma(b2)))
        cf = mp.sqrt(nv * mp.sinh(mp.pi * (nv + 2) * w) / ((nv + 2) * mp.sinh(mp.pi * nv * w)))
        Tn = model.kummer(C, w, nv, 3000)
        Tp = model.kummer(C, w, nv + 2, 3000)
        prod = abs(Tp[3000] / Tn[3000])
        if not (prod < cf and cf < P.a and abs(gr / cf - 1) < mp.mpf('1e-60')):
            ok = False
        print(f"  P_(nu+2) at nu = {nv}: closed form {mp.nstr(cf, 10)}, Gamma ratio rel. diff {mp.nstr(abs(gr / cf - 1), 3)},"
              f" |rho_3000| {mp.nstr(prod, 8)} (increasing toward it)")
    return P, ok


def W_bound(T):
    """W = sup over odd nu >= 3 of nu |rho^{+-1}_{j0}|, j0 = max(2 nu, 10): iv product to 99, hand bound above"""
    W = T.W
    w4 = W ** 2 / 4
    best, at = iv.mpf(0), None
    for nv in range(3, 100, 2):
        j0 = max(2 * nv, J0)
        rho2 = iv.mpf(1)
        for m in range(j0):
            an2 = ((m + iv.mpf(1) / 2) ** 2 + nv * nv * w4) * ((m + 1) ** 2 + nv * nv * w4) / (m + 1) ** 2
            rho2 *= T.r2[m] / an2
        v = nv * iv.sqrt(rho2)
        if v.b > best.b:
            best, at = v, nv
    nv = 101
    q = ((nv * W / 4 + 1) ** 2 + w4) / (nv * nv * w4)
    M = int(mp.floor((nv * W / 4).a))
    assert int(mp.floor((nv * W / 4).b)) == M
    hand = nv * q ** M
    ok = q.b < mp.mpf(101) / 103 and M <= 2 * nv
    return best, at, hand, q, M, ok


def main():
    mut = {a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--mutate=')}
    J1 = 2000
    for a in sys.argv[1:]:
        if a.startswith('--J1='):
            J1 = int(a.split('=')[1])
    iv.prec = 120
    W = W_iv()
    t0 = time.time()
    T = Tseq(max(J1 + 2, 2 * 99 + 2), W)                    # W_bound needs r_m for m < j0(99) = 198
    print(f"t_n, G_i to {J1 + 2} [{time.time() - t0:.0f}s]; t_1 <= {model.up(T.t[1], 8)}, w in [{model.dn(W, 12)}, {model.up(W, 12)}]", flush=True)
    ok = True
    m_ok, top, mstar = checks_monotone(T, W)
    print(f"r_m increasing for 1 <= m <= {top} (checked), above m > w^2/2 - 1 <= {model.up(mstar, 5)} (proved): "
          f"{'yes' if m_ok else 'NO'}")
    ok &= m_ok
    cw = check_conv(T, J1)
    print(f"max_n<={J1} U_n/(2 + t_1) <= {model.up(cw, 4)}: {'yes' if cw.b <= 1 else 'NO'}")
    ok &= cw.b <= 1
    pf = check_prefactors(W, 'nuprefac' in mut)
    print(f"B4 prefactor bounds >= exact prefactors (nu = 3..41, j >= 2 nu): {'yes' if pf else 'NO'}")
    ok &= pf
    eq = equality_test(T, 'dropii' in mut)
    print(f"nu = 1 equality test, majorant vs sum of |interior terms|: worst relative difference {mp.nstr(eq, 3)}"
          f" (bar 1e-25): {'yes' if eq < mp.mpf('1e-25') else 'NO'}")
    ok &= eq < mp.mpf('1e-25')
    P, p_ok = P_bound()
    print(f"P <= e^(pi w) <= {model.up(P, 7)}: closed form checks {'yes' if p_ok else 'NO'}")
    ok &= p_ok
    Wb, at, hand, q, M, w_ok = W_bound(T)
    print(f"W: max_nu<=99 nu|rho_j0| <= {model.up(Wb, 5)} at nu = {at}; nu >= 101: nu q^M <= {model.up(hand, 3)} "
          f"(q <= {model.up(q, 4)}, M = {M}): {'yes' if w_ok else 'NO'}")
    ok &= w_ok
    # the majorant sups
    b4 = iv.mpf(0)
    b4_at = None
    tgt = {-2: iv.mpf(0), 0: iv.mpf(0), 2: iv.mpf(0)}
    tgt_at = {}
    for jj in range(J0, J1 + 1):
        S = sums(T, jj, 'dropii' in mut)
        A, B = b4_per_triple(S, jj, W, 'nuprefac' in mut)
        v = jj * jj * (A / 3 + B)
        if v.b > b4.b:
            b4, b4_at = v, jj
        nus = nu_grid(jj)
        for k, (n_out, ntr) in {-2: (1, 3), 0: (2, 6), 2: (1, 3)}.items():
            vt = tight(S, jj, W, k, n_out, ntr, nus)
            if vt.b > tgt[k].b:
                tgt[k], tgt_at[k] = vt, jj
        if jj in (10, 11, 12, 20, 50, 100, 500, 1000, 2000):
            print(f"  j = {jj}: B4 per triple j^2(A/3 + B) <= {model.up(v, 5)} [{time.time() - t0:.0f}s]", flush=True)
    tb = tail_bound(T, J1, W)
    print(f"B4 per triple: max over 10 <= j <= {J1} <= {model.up(b4, 6)} at j = {b4_at}; every j > {J1}: <= {model.up(tb, 3)}")
    cr = max(b4.b, tb)
    print(f"Cr (B4 mode): 3 triples <= {model.up(3 * cr, 6)}, 6 triples <= {model.up(6 * cr, 6)}")
    U = lambda x: model.up(x, 45)                          # upper ends, rounded up
    res = {'Cr_B4': {'-2': U(3 * cr), '0': U(6 * cr), '2': U(3 * cr)}, 'Cr_tight': {},
           'P': U(P), 'W': U(max(Wb.b, hand.b)), 'W_at': at, 'W_hand101': U(hand), 'tail_per_triple': U(tb)}
    for k in (-2, 0, 2):
        ntr = 6 if k == 0 else 3
        ct = max(tgt[k].b, ntr * tb)
        res['Cr_tight'][str(k)] = U(ct)
        print(f"Cr (tight), mu = nu{k:+d}: <= {model.up(ct, 6)} (max at j = {tgt_at[k]})")
    out = [a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--json=')]
    if out and ok:
        import json
        json.dump(res, open(out[0], 'w'), indent=1)
    print(f"interior.py {'PASS' if ok else 'FAIL'} [{time.time() - t0:.0f}s]")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
