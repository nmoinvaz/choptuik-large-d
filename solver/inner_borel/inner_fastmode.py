"""WARNING: the correction terms m >= 3 are wrong for M > 1 (t5_check_fastmode.py: off by 3e-3 at m = 3, 1.6e-2 at m = 4,
probably the dc-coupling terms); use only for M = 1, or for the leading terms m <= 2.

Fast modes of the inner problem as formal series, and the Stokes constants they define (well-conditioned Darboux).

Linearize the inner equations about the formal solution (inner_series.py; Dd = xi D(u), sigma = u S(u), c = C(u), u = xi^2)
and put  dDd = E xi u^p a(u, th),  dsigma = E u^p b(u, th),  dc = E u^(p+2) g(u, th),  E = exp(-1/(2u)).
Order by order in u (a = sum a_m u^m, etc.; products are theta convolutions, K a = 2 C_1 a - 2 w a_th):
    (K - kappa) a_0 = 0,  2p + 1 = kappa         (transport; a_0 an eigenvector)
    delta_m = a_m - b_m = -(2p+2m-1) a_{m-1} - w a'_{m-1} + sum_{i>=1} C_i b_{m-i} + [S g]_{m-3}
    (K - kappa - 2m) a_m = C_1 delta_m - (2p+2m) delta_m - w delta_m' - rest_m
    rest_m = sum_{i>=2} C_i (a+b)_{m+1-i} + [S g]_{m-2} + sum_{i>=1} C_i ((2p+2m-2i) a_{m-i}) + 3/2 [C D^2 a]_{m-1}
             + [g (D + 2u D')]_{m-2} - [g D]_{m-2} + 1/2 [g D^3]_{m-3}
    g_m = -(2p+2m+2) g_{m-1} + 1/2 [g D^2]_{m-2} + [C D a]_m
Each mode k (eigenvalue kappa_k, gamma_k = (1 - kappa_k)/2) predicts the late orders
    d_{j,mu} ~ S_k sum_m (a_{k,m})_mu 2^(j-m) Gamma(j + gamma_k - m),
so a least-squares fit over j and mu has only the S_k as unknowns.

    python3 inner_fastmode.py A Mmax J [wf] [B3]
"""
import sys
import mpmath as mp
from flint import acb, acb_poly, acb_mat, ctx
from inner_series import series

W0 = 2 * mp.pi / mp.mpf("1.64389375")


def run(A, Mmax, J, wf=1, B3=0, mmax=40, prec=900, verbose=True, j_fit=None, mus=None):
    ctx.prec = prec
    Mst = Mmax + 1
    N = 2 * Mst + 1
    w = acb(mp.nstr(W0 * wf, 60)) if True else None
    Fh = {1: -0.5j * float(A), -1: 0.5j * float(A)}
    Fh_acb = {1: acb(0, -A / 2), -1: acb(0, A / 2)}
    if B3:
        Fh_acb.update({3: acb(0, -B3 / 2), -3: acb(0, B3 / 2)})
    d, s, g = series(Fh_acb, w, J, M=Mst, prec=prec)
    I = acb(0, 1)
    mu_of = lambda k: k - Mst

    def mul(x, y):
        c = (acb_poly(x) * acb_poly(y)).coeffs()
        c = c + [acb(0)] * (4 * Mst + 1 - len(c))
        return c[Mst:3 * Mst + 1]

    def add(x, y, f=1):
        return [p + f * q for p, q in zip(x, y)]

    def scal(x, f):
        return [f * p for p in x]

    def dth(x):
        return [I * mu_of(k) * x[k] for k in range(N)]

    def clean(x, parity):                   # keep the parity and the truncation
        return [x[k] if (abs(mu_of(k)) % 2 == parity and abs(mu_of(k)) <= (Mmax if parity else Mmax + 1)) else acb(0)
                for k in range(N)]

    Z = [acb(0)] * N
    C = g                                   # C_j = g_j
    D = d                                   # D_j = d_j
    S = s                                   # S_j = s_j
    D2 = [None] * (mmax + 3)
    for m in range(mmax + 3):
        acc = Z
        for i in range(m + 1):
            acc = add(acc, mul(D[i], D[m - i]))
        D2[m] = acc
    D3 = [None] * (mmax + 3)
    for m in range(mmax + 3):
        acc = Z
        for i in range(m + 1):
            acc = add(acc, mul(D2[i], D[m - i]))
        D3[m] = acc
    CD2 = [None] * (mmax + 3)
    CD = [None] * (mmax + 3)
    for m in range(mmax + 3):
        a1, a2 = Z, Z
        for i in range(m + 1):
            a1 = add(a1, mul(C[i], D2[m - i])); a2 = add(a2, mul(C[i], D[m - i]))
        CD2[m], CD[m] = a1, a2
    Dp = [scal(D[m], 1 + 2 * m) for m in range(mmax + 3)]           # D + 2u D'

    # transport matrix on odd harmonics
    odd = [k for k in range(N) if abs(mu_of(k)) % 2 == 1 and abs(mu_of(k)) <= Mmax]

    def Kop(x):
        return add(scal(mul(C[1], x), 2), scal(dth(x), -2 * w))

    nmode = len(odd)
    Kmat = acb_mat(nmode, nmode)
    for c_, kc in enumerate(odd):
        e = [acb(0)] * N; e[kc] = acb(1)
        col = Kop(e)
        for r_, kr in enumerate(odd):
            Kmat[r_, c_] = col[kr]
    # eigen-decomposition via mpmath (small)
    Km = mp.matrix(nmode, nmode)
    for r_ in range(nmode):
        for c_ in range(nmode):
            v = Kmat[r_, c_].mid()
            Km[r_, c_] = mp.mpc(mp.mpf(v.real.str(100, radius=False)), mp.mpf(v.imag.str(100, radius=False)))
    mp.mp.dps = 100
    Ev, Vv = mp.eig(Km)

    def solve(Mshift, rhs):
        """(K - Mshift) x = rhs on the odd harmonics"""
        Am = acb_mat(nmode, nmode)
        for r_ in range(nmode):
            for c_ in range(nmode):
                Am[r_, c_] = Kmat[r_, c_] - (Mshift if r_ == c_ else 0)
        bm = acb_mat(nmode, 1, [rhs[k] for k in odd])
        xs = Am.solve(bm)
        out = [acb(0)] * N
        for r_, k in enumerate(odd):
            out[k] = xs[r_, 0]
        return out

    modes = []
    for km in range(nmode):
        kap = acb(mp.nstr(mp.re(Ev[km]), 90), mp.nstr(mp.im(Ev[km]), 90))
        # refine eigenpair by inverse iteration in acb
        v = [acb(0)] * N
        for r_, k in enumerate(odd):
            v[k] = acb(mp.nstr(mp.re(Vv[r_, km]), 90), mp.nstr(mp.im(Vv[r_, km]), 90))
        p = (kap - 1) / 2
        a, b, gg = [v], [v], [mul(CD[0], v)]
        for m in range(1, mmax + 1):
            dl = add(scal(a[m - 1], -(2 * p + 2 * m - 1)), scal(dth(a[m - 1]), -w))
            for i in range(1, m + 1):
                dl = add(dl, mul(C[i], b[m - i]))
            for k_ in range(0, m - 2):
                dl = add(dl, mul(S[k_], gg[m - 3 - k_]))
            rest = Z
            for i in range(2, m + 2):
                rest = add(rest, mul(C[i], add(a[m + 1 - i], b[m + 1 - i])))
            for k_ in range(0, m - 1):
                rest = add(rest, mul(S[k_], gg[m - 2 - k_]))
            for i in range(1, m + 1):
                rest = add(rest, mul(C[i], scal(a[m - i], 2 * p + 2 * m - 2 * i)))
            acc = Z
            for i in range(0, m):
                acc = add(acc, mul(CD2[i], a[m - 1 - i]))
            rest = add(rest, scal(acc, acb(3) / 2))
            for i in range(0, m - 1):
                rest = add(rest, mul(gg[i], Dp[m - 2 - i]))
                rest = add(rest, mul(gg[i], D[m - 2 - i]), -1)
            for i in range(0, m - 2):
                rest = add(rest, scal(mul(gg[i], D3[m - 3 - i]), acb(1) / 2))
            rhs = add(add(mul(C[1], dl), scal(dl, -(2 * p + 2 * m))), add(scal(dth(dl), -w), rest, -1))
            am = solve(kap + 2 * m, rhs)
            am = clean(am, 1)
            bm_ = clean(add(am, dl, -1), 1)
            gm = scal(gg[m - 1], -(2 * p + 2 * m + 2))
            aa = a + [am]
            for i in range(0, m + 1):
                gm = add(gm, mul(CD[i], aa[m - i]))
            acc = Z
            for i in range(0, m - 1):
                acc = add(acc, mul(gg[i], D2[m - 2 - i]))
            gm = clean(add(gm, scal(acc, acb(1) / 2)), 0)
            a.append(am); b.append(bm_); gg.append(gm)
        modes.append((kap, (1 - kap) / 2, a))
    return d, modes, Mst, odd


def fit_S(d, modes, Mst, odd, js, mus, mtr, dps=60):
    """least squares over (j, mu) for S_k; mtr: number of fast-mode terms kept"""
    mp.mp.dps = dps
    tom = lambda z: mp.mpc(mp.mpf(z.mid().real.str(dps + 10, radius=False)), mp.mpf(z.mid().imag.str(dps + 10, radius=False)))
    rows, rhs = [], []
    for j in js:
        ref = None
        for mu in mus:
            row = []
            for kap, gam, a in modes:
                gm = tom(gam)
                val = mp.fsum(tom(a[m][mu + Mst]) * mp.mpf(2) ** (-m) * mp.gamma(j + gm - m) for m in range(min(mtr, len(a))))
                row.append(val)
            sc = 1 / mp.gamma(j + mp.re(tom(modes[0][1])))
            rows.append([x * sc for x in row]); rhs.append(tom(d[j][mu + Mst]) / mp.mpf(2) ** j * sc)
    Am = mp.matrix(rows); bv = mp.matrix(rhs)
    x, res = mp.qr_solve(Am, bv)
    rel = res / mp.norm(bv)
    return [x[i] for i in range(len(modes))], rel


if __name__ == "__main__":
    A = float(sys.argv[1]); Mmax = int(sys.argv[2]); J = int(sys.argv[3])
    wf = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    B3 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    d, modes, Mst, odd = run(A, Mmax, J, wf, B3, mmax=min(40, J // 3))
    mus = [mu for mu in range(-Mmax, Mmax + 1, 2)]
    for mtr in (10, 20, 40):
        for js in (list(range(J - 40, J + 1, 4)), list(range(J - 100, J + 1, 10))):
            S, rel = fit_S(d, modes, Mst, odd, js, mus, mtr)
            print(f"A {A} Mmax {Mmax} wf {wf} B3 {B3}: terms {mtr}, j {js[0]}..{js[-1]}: rel residual {mp.nstr(rel, 3)}")
            for (kap, gam, a), Sk in zip(modes, S):
                print(f"    gamma {mp.nstr(mp.mpc(float(gam.real.mid()), float(gam.imag.mid())), 8):>28s}   S {mp.nstr(Sk, 12)}")
