"""Inner Stokes multipliers to high order in the amplitude, in ball arithmetic (python-flint), all harmonics exact.

Same problem as amp_series.py / stokes_amp2.py (README, "The inner problem at large x"):
    L y = c sigma,  L sigma = c xi y_xi + c (y^2/2 - 1) y,  xi c_xi = c y^2/2,   L = xi^2 d_xi + w xi d_theta,
y = sum_j d_j xi^(2j+1), sigma = sum_j s_j xi^(2j+2), c = sum_j g_j xi^(2j); stored as X~_j = X_j / 2^j so that every
product is a pure convolution in j:
    s~_j = (2j+1+w d) d~_j - [g~ s~]_j,   e~_m = 1/4 [d~ d~]_{m-1},   g~_m = [g~ e~]_m / (2m),
    d~_{j+1} = (2j+2+w d) s~_j / (4(j+1)) - [g~ d~]_{j+1}        (products without the g_0 = 1 term).
Expanded in the amplitude eps (F = eps sin theta + sum_{nu>=3} f_nu e^{i nu theta}), order eps^k depends linearly on
itself only through g_0 = 1, so the orders are computed one at a time: every product at order k pairs LOWER orders and is
one Kronecker-substituted acb_poly multiplication over all (j, mu) at once. Harmonic mu first appears at order eps^|mu|,
so keeping |mu| <= K is exact through eps^K.

After each odd order k >= 3 the new coefficients f_nu^(k) (3 <= |nu| <= k) are tuned so that the fast-type multipliers
S_nu vanish at that order (the linear solution of harmonic nu is then added exactly). S_1(eps) is what remains:
    S_1 / S_1^(1) = 1 + sigma_1 eps^2 + sigma_2 eps^4 + ...,   selection: S_1 = 0.
Late orders: R_j = [e^{-P} d_j]_nu j! / (Gamma(a+j-h) Gamma(b+j-h)), h = <F^2>/2 (eps-series), fitted per order with
1 + Gamma ratios in 1/j + neighbour families (nu +- 2, 4, 6).

    python3 amp_flint.py wf K J [prec_bits]      writes sigF_w<wf>_K<K>_J<J>.json (sigma_k, tuned f_nu^(k))
"""
import sys
import json
import time
import mpmath as mp
from flint import acb, acb_poly, arb, ctx


def to_mp(z):
    def r(x):
        m, e = x.mid().man_exp()
        return mp.ldexp(mp.mpf(int(m)), int(e))
    return mp.mpc(r(z.real), r(z.imag))


class Amp:
    def __init__(self, wf, K, J, prec=320, jlo=None, M=12, fams=(2, 4, 6), Mf=4, dps=60):
        ctx.prec = prec
        mp.mp.dps = dps
        self.K, self.J, self.H = K, J, K
        self.W = 4 * K + 1
        self.wmp = mp.mpf(wf) * 2 * mp.pi / mp.mpf("1.64389375")
        self.w = acb(arb(str(mp.nstr(self.wmp, dps))))
        self.jlo = jlo or (2 * J) // 5
        self.M, self.fams, self.Mf = M, fams, Mf
        Z = acb(0)
        nz = 2 * self.H + 1
        self.zero_row = [Z] * nz
        self.d, self.s, self.g, self.e = {}, {}, {}, {}
        self.F = {}                       # F[k] = list over mu of acb (order eps^k)
        self.F[1] = [Z] * nz
        self.F[1][self.H + 1] = acb(0, -1) / 2
        self.F[1][self.H - 1] = acb(0, 1) / 2
        self.cache_psi = {}
        self.basis_cache = {}

    # --- convolutions over (j, mu) via Kronecker substitution
    def poly(self, X):
        W, H = self.W, self.H
        c = [acb(0)] * ((self.J + 1) * W)
        for j, row in enumerate(X):
            base = j * W
            for i, v in enumerate(row):
                c[base + i] = v
        return acb_poly(c)

    def conv(self, PX, PY):
        W, H, J = self.W, self.H, self.J
        cs = (PX * PY).coeffs()
        L = len(cs)
        out = []
        for j in range(J + 1):
            base = j * W + H                 # mu + 2H offset -> keep |mu| <= H
            row = cs[base: base + 2 * H + 1] if base + 2 * H + 1 <= L else (cs[base:] + [acb(0)] * (base + 2 * H + 1 - L))
            if len(row) < 2 * H + 1:
                row = row + [acb(0)] * (2 * H + 1 - len(row))
            out.append(row)
        return out

    def add_into(self, A, B, scale=None):
        for j in range(len(A)):
            a, b = A[j], B[j]
            if scale is None:
                A[j] = [x + y for x, y in zip(a, b)]
            else:
                A[j] = [x + scale * y for x, y in zip(a, b)]

    # --- passes
    def pass_even(self, k):
        H, J = self.H, self.J
        zero = [list(self.zero_row) for _ in range(J + 1)]
        D = [list(self.zero_row) for _ in range(J + 1)]
        for k1 in range(1, k, 2):
            self.add_into(D, self.conv(self.P[("d", k1)], self.P[("d", k - k1)]))
        e = [list(self.zero_row)] + [[x / 4 for x in D[m - 1]] for m in range(1, J + 1)]
        self.e[k] = e
        self.P[("e", k)] = self.poly(e)
        G = [list(row) for row in e]                      # g_0^(0) = 1 term
        for k1 in range(2, k - 1, 2):
            self.add_into(G, self.conv(self.P[("g", k1)], self.P[("e", k - k1)]))
        g = [list(self.zero_row)] + [[x / (2 * m) for x in G[m]] for m in range(1, J + 1)]
        self.g[k] = g
        self.P[("g", k)] = self.poly(g)

    def pass_odd(self, k):
        H, J = self.H, self.J
        Ss = [list(self.zero_row) for _ in range(J + 1)]
        Sd = [list(self.zero_row) for _ in range(J + 1)]
        for k1 in range(2, k, 2):
            self.add_into(Ss, self.conv(self.P[("g", k1)], self.P[("s", k - k1)]))
            self.add_into(Sd, self.conv(self.P[("g", k1)], self.P[("d", k - k1)]))
        Dth = [acb(0, mu) * self.w for mu in range(-H, H + 1)]
        d = [[2 * v for v in self.F.get(k, self.zero_row)]]
        s = []
        for j in range(J + 1):
            sj = [(2 * j + 1 + Dth[i]) * d[j][i] - Ss[j][i] for i in range(2 * H + 1)]
            s.append(sj)
            if j < J:
                d.append([(2 * j + 2 + Dth[i]) * sj[i] / (4 * (j + 1)) - Sd[j + 1][i] for i in range(2 * H + 1)])
        self.d[k], self.s[k] = d, s

    def add_linear(self, k, nu, f):
        """add the exact linear solution of harmonic nu with d_0 = 2 f at order k"""
        i = self.H + nu
        s_ = acb(0, nu) * self.w
        a, b = (1 + s_) / 2, (2 + s_) / 2
        t = 2 * f                                # d~_j = (a)_j (b)_j / j! * 2f
        for j in range(self.J + 1):
            self.d[k][j][i] += t
            self.s[k][j][i] += (2 * j + 1 + s_) * t
            t = t * (a + j) * (b + j) / (j + 1)
        F = list(self.F.get(k, self.zero_row))
        F[i] += f
        self.F[k] = F

    # --- theta-function helpers (lists over mu) and the e^{-P}, <F^2> series
    def tmul(self, A, B):
        H = self.H
        out = [acb(0)] * (2 * H + 1)
        for i, x in enumerate(A):
            if x == 0:
                continue
            for l, y in enumerate(B):
                if y == 0:
                    continue
                m = i + l - H
                if 0 <= m <= 2 * H:
                    out[m] += x * y
        return out

    def theta_series(self, upto):
        """E = exp(-P) and delta = <F^2> as eps-series up to order `upto`, from the current F"""
        H = self.H
        F2 = {}
        for k in range(2, upto + 1, 2):
            acc = [acb(0)] * (2 * H + 1)
            for k1 in range(1, k, 2):
                if k1 in self.F and (k - k1) in self.F:
                    acc = [x + y for x, y in zip(acc, self.tmul(self.F[k1], self.F[k - k1]))]
            F2[k] = acc
        P = {k: [(F2[k][i] / (acb(0, i - H) * self.w) if i != H else acb(0)) for i in range(2 * H + 1)] for k in F2}
        E = {0: [acb(0)] * (2 * H + 1)}
        E[0][H] = acb(1)
        for k in range(2, upto + 1, 2):         # k E_k = - sum_m m P_m E_{k-m}
            acc = [acb(0)] * (2 * H + 1)
            for m in range(2, k + 1, 2):
                acc = [x + m * y for x, y in zip(acc, self.tmul(P[m], E[k - m]))]
            E[k] = [-x / k for x in acc]
        delta = {k: F2[k][H] for k in F2}
        return E, delta

    def psi_tab(self, nu, nmax):
        key = (nu, nmax)
        if key in self.cache_psi:
            return self.cache_psi[key]
        s_ = 1j * nu * self.wmp
        a, b = (1 + s_) / 2, (2 + s_) / 2
        tab = []
        for j in range(self.jlo, self.J + 1):
            base = mp.factorial(j) * mp.rgamma(a + j) * mp.rgamma(b + j)
            ps = [(mp.psi(n - 1, a + j) + mp.psi(n - 1, b + j)) / mp.factorial(n) for n in range(1, nmax + 1)]
            tab.append((base, ps))
        self.cache_psi[key] = tab
        return tab

    def basis(self, nu):
        if nu in self.basis_cache:
            return self.basis_cache[nu]
        s_ = 1j * nu * self.wmp
        g0 = s_ + mp.mpf(1) / 2
        rows = []
        for j in range(self.jlo, self.J + 1):
            row = [mp.mpc(1)]
            t = mp.mpc(1)
            for m in range(1, self.M + 1):
                t = t / (j + g0 - m)
                row.append(t)
            for sh in self.fams:
                for sgn in (-1, 1):
                    gs = g0 + sgn * sh * 1j * self.wmp
                    r0 = mp.exp(mp.loggamma(j + gs) - mp.loggamma(j + g0))
                    t = r0
                    for m in range(1, self.Mf + 1):
                        t = t / (j + gs - m)
                        row.append(t)
            rows.append(row)
        A = mp.matrix(rows)
        sc = [max(abs(A[i, c]) for i in range(A.rows)) for c in range(A.cols)]
        for c in range(A.cols):
            for i in range(A.rows):
                A[i, c] /= sc[c]
        self.basis_cache[nu] = (A, sc)
        return A, sc

    def S_at(self, nu, k):
        """fitted multiplier of harmonic nu at order eps^k (data: orders <= k of d)"""
        E, delta = self.theta_series(k)
        H = self.H
        # z_j^(k) = sum_{k''} sum_mu E^(k'')[nu - mu] d_j^(k-k'')[mu]
        # normalization series: N_j = base * exp(-sum_n ps_n (-h)^n), h = delta/2
        nmax = k // 2 + 1
        tab = self.psi_tab(nu, nmax)
        hser = [mp.mpc(0)] * (k + 1)
        for kk, v in delta.items():
            if kk <= k:
                hser[kk] = to_mp(v) / 2
        def smul(x, y):
            out = [mp.mpc(0)] * (k + 1)
            for i, a in enumerate(x):
                if a == 0:
                    continue
                for l in range(k + 1 - i):
                    if y[l] != 0:
                        out[i + l] += a * y[l]
            return out
        hp = [[mp.mpc(1)] + [mp.mpc(0)] * k]
        for n in range(1, nmax + 1):
            hp.append(smul(hp[-1], [-x for x in hser]))
        Emp = {kk: [to_mp(x) for x in E[kk]] for kk in E}
        ys = []
        for idx, j in enumerate(range(self.jlo, self.J + 1)):
            base, ps = tab[idx]
            Lser = [mp.mpc(0)] * (k + 1)
            for n in range(1, nmax + 1):
                for q in range(k + 1):
                    Lser[q] -= ps[n - 1] * hp[n][q]
            # exp of Lser (Lser[0] = 0)
            Nser = [mp.mpc(1)] + [mp.mpc(0)] * k
            term = list(Nser)
            for n in range(1, k + 1):
                term = [x / n for x in smul(term, Lser)]
                if all(t == 0 for t in term):
                    break
                Nser = [x + y for x, y in zip(Nser, term)]
            # z series up to order k (odd orders only)
            zser = [mp.mpc(0)] * (k + 1)
            for kp in range(1, k + 1, 2):
                if kp not in self.d:
                    continue
                dj = self.d[kp][j]
                for kpp in range(0, k - kp + 1, 2):
                    if kpp not in Emp:
                        continue
                    Ek = Emp[kpp]
                    acc = mp.mpc(0)
                    for mu in range(-min(H, kp), min(H, kp) + 1, 2):
                        e_ = Ek[H + nu - mu] if 0 <= H + nu - mu <= 2 * H else 0
                        if e_ != 0:
                            acc += e_ * to_mp(dj[H + mu])
                    zser[kp + kpp] += acc
            ys.append(base * sum(zser[q] * Nser[k - q] for q in range(k + 1)))
        A, sc = self.basis(nu)
        y = mp.matrix(ys)
        c, res = mp.qr_solve(A, y)
        resid = res / mp.norm(y)
        return c[0] / sc[0], resid

    def run(self, verbose=True, tune=True):
        self.P = {}
        t0 = time.time()
        sig, f_out, S1 = [], {}, {}
        for k in range(1, self.K + 1):
            if k % 2 == 0:
                self.pass_even(k)
            else:
                self.pass_odd(k)
                if tune and k >= 3:
                    for nu in range(3, k + 1, 2):
                        S, rs = self.S_at(nu, k)
                        s_ = 1j * nu * self.wmp
                        linS = 1 / (mp.gamma((1 + s_) / 2) * mp.gamma((2 + s_) / 2))
                        f = -S / (2 * linS)
                        fa = acb(arb(mp.nstr(f.real, 50)), arb(mp.nstr(f.imag, 50)))
                        self.add_linear(k, nu, fa)
                        self.add_linear(k, -nu, fa.conjugate())
                        f_out[f"{k}_{nu}"] = [float(f.real), float(f.imag)]
                        if verbose:
                            print(f"   k {k} nu {nu}: |S_gen/linS| {float(abs(S / linS)):.3e} fit resid {float(rs):.1e} f {mp.nstr(f, 10)}",
                                  flush=True)
                self.P[("d", k)] = self.poly(self.d[k])
                self.P[("s", k)] = self.poly(self.s[k])
                S, rs = self.S_at(1, k)
                S1[k] = S
                if verbose:
                    print(f"k {k}: S_1^({k}) = {mp.nstr(S, 15)}  fit resid {float(rs):.1e}  sigma = {mp.nstr(S / S1[1], 15)}"
                          f"  [{time.time() - t0:.0f}s]", flush=True)
                sig.append(S / S1[1])
        return sig, f_out


if __name__ == "__main__":
    wf, K, J = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    prec = int(sys.argv[4]) if len(sys.argv) > 4 else 320
    A = Amp(wf, K, J, prec=prec)
    sig, f = A.run()
    json.dump({"wf": wf, "K": K, "J": J, "prec": prec, "sigma": [[float(z.real), float(z.imag)] for z in sig],
               "sigma_str": [[mp.nstr(z.real, 20), mp.nstr(z.imag, 20)] for z in sig], "f": f},
              open(f"sigF_w{wf:.4f}_K{K}_J{J}.json", "w"))
