"""Fast-type Stokes multipliers of the inner problem (x ~ sqrt(n)) by lateral solutions.

Inner equations (xi = x/sqrt(n), Dd = p - m, sigma = sqrt(n)(p + m)), Galerkin-truncated in theta:
    xi^2 Dd_xi  + i mu w xi Dd   = [c sigma]_mu                                  (I1)
    xi^2 sig_xi + i mu w xi sig  = [c xi Dd_xi]_mu + [c (Dd^2/2 - 1) Dd]_mu        (I2)
    xi c_xi = [c Dd^2/2]_nu                                                      (I3)
with Dd, sigma on odd harmonics |mu| <= M and c on even harmonics |nu| <= M + 1 (gravity=False: c = 1).

Method (route b):
  1. formal series Dd = sum d_j xi^(2j+1), sigma = sum s_j xi^(2j+2), c = sum g_j xi^(2j) in mpmath
     (harmonic-space version of inner_series.py's recursion);
  2. y+ : start on the ray arg(xi^2) = +3pi/4 at |xi| = r0 (every fast mode e^{-1/(2 xi^2)} xi^{-2 s_mu} is
     exponentially LARGE there, so the series data pin the solution: it is the lateral Borel sum from above),
     integrate out along the ray to |xi| = r1, then along the arc |xi| = r1 to the real axis;
  3. for real F the lateral sum from below is the conjugate: y-_mu = conj(y+_{-mu}) on the real axis, so the jump is
     J_mu = y+_mu - conj(y+_{-mu});
  4. linear check (M = 1, no gravity): J(Dd)_1 = S f(xi1), f = xi^{-2s} e^{-1/(2 xi^2)} 2F0(1-a, 1-b;; -2 xi^2)
     (Borel-summed by optimal truncation), and S should be 2 pi i K' (up to orientation sign),
     K' = 2^{-s-1/2} / (Gamma(a) Gamma(b)).

    python3 lateral.py linear            linear check
    python3 lateral.py jump A w [F3re F3im] [M]   jump at real xi for F = A sin(theta) + 2 Re(F3 e^{3 i theta})
"""
import sys
import mpmath as mp

mp.mp.dps = 30


class Model:
    def __init__(self, Fh, w, M, gravity=True):
        """Fh: dict mu -> complex harmonic of F (odd mu, both signs)"""
        self.w, self.M, self.grav = mp.mpf(w), M, gravity
        self.odd = [mu for mu in range(-M, M + 1) if mu % 2]
        self.even = [nu for nu in range(-(M + 1), M + 2) if nu % 2 == 0] if gravity else [0]
        self.Fh = {mu: mp.mpc(Fh.get(mu, 0)) for mu in self.odd}

    # products on harmonic dicts, projected
    def mul(self, A, B, keep):
        out = {k: mp.mpc(0) for k in keep}
        for i, ai in A.items():
            if ai == 0:
                continue
            for j, bj in B.items():
                k = i + j
                if k in out:
                    out[k] += ai * bj
        return out

    def series(self, J):
        """formal inner series; returns d[j], s[j], g[j] (harmonic dicts). Every product is projected at once onto the
        space of its result (odd |mu| <= M, even |nu| <= M+1), exactly as in rhs(), so the series is the formal
        solution of the truncated system that is integrated."""
        w, odd, even = self.w, self.odd, self.even
        d = [{mu: 2 * self.Fh[mu] for mu in odd}]
        s, g = [], [{nu: mp.mpc(1 if nu == 0 else 0) for nu in even}]
        e = [{nu: mp.mpc(0) for nu in even}]
        ce = [{nu: mp.mpc(0) for nu in even}]          # ce[m] = [c E]_m = sum_{i<m} g_i e_{m-i}
        for j in range(J):
            sj = {mu: (2 * j + 1 + 1j * mu * w) * d[j][mu] for mu in odd}
            for i in range(1, j + 1):
                t = self.mul(g[i], s[j - i], odd)
                for mu in odd:
                    sj[mu] -= t[mu]
            s.append(sj)
            m = j + 1
            if self.grav:
                em = {nu: mp.mpc(0) for nu in even}
                for i in range(m):
                    t = self.mul(d[i], d[m - 1 - i], even)
                    for nu in even:
                        em[nu] += t[nu] / 2
                e.append(em)
                cm = {nu: mp.mpc(0) for nu in even}
                for i in range(m):
                    t = self.mul(g[i], e[m - i], even)
                    for nu in even:
                        cm[nu] += t[nu]
                ce.append(cm)
                g.append({nu: cm[nu] / (2 * m) for nu in even})
            else:
                e.append({0: mp.mpc(0)}); ce.append({0: mp.mpc(0)}); g.append({0: mp.mpc(0)})
            num = {mu: (2 * j + 2 + 1j * mu * w) * s[j][mu] for mu in odd}
            for i in range(1, j + 2):
                t = self.mul(g[i], d[j + 1 - i], odd)
                for mu in odd:
                    num[mu] -= 2 * (j + 1 - i) * t[mu]
            if self.grav:
                for mm in range(1, j + 2):                 # [c E Dd]_{j+1} = sum_m ce_m d_{j+1-m}
                    t = self.mul(ce[mm], d[j + 1 - mm], odd)
                    for mu in odd:
                        num[mu] -= t[mu]
            d.append({mu: num[mu] / (2 * j + 2) for mu in odd})
        return d, s, g

    def series_value(self, d, s, g, xi, J=None):
        """optimally truncated sum (truncate at the smallest term of Dd)"""
        J = J or len(d) - 1
        x2 = xi * xi
        best = None
        mags = []
        for j in range(J):
            mags.append(max(abs(d[j][mu]) for mu in self.odd) * abs(x2) ** j)
        jt = min(range(1, J), key=lambda j: mags[j])
        D = {mu: sum(d[j][mu] * xi ** (2 * j + 1) for j in range(jt)) for mu in self.odd}
        S = {mu: sum(s[j][mu] * xi ** (2 * j + 2) for j in range(jt)) for mu in self.odd}
        C = {nu: sum(g[j][nu] * xi ** (2 * j) for j in range(jt)) for nu in self.even}
        return D, S, C, jt, mags[jt]

    def pack(self, D, S, C):
        return [D[mu] for mu in self.odd] + [S[mu] for mu in self.odd] + [C[nu] for nu in self.even]

    def unpack(self, Y):
        no, ne = len(self.odd), len(self.even)
        D = dict(zip(self.odd, Y[:no])); S = dict(zip(self.odd, Y[no:2 * no])); C = dict(zip(self.even, Y[2 * no:2 * no + ne]))
        return D, S, C

    def rhs(self, xi, Y):
        w, odd, even = self.w, self.odd, self.even
        D, S, C = self.unpack(Y)
        cs = self.mul(C, S, odd)
        dD = {mu: (cs[mu] - 1j * mu * w * xi * D[mu]) / (xi * xi) for mu in odd}
        if self.grav:
            E = {k: v / 2 for k, v in self.mul(D, D, even).items()}
            cE = self.mul(C, E, even)
            cEd = self.mul(cE, D, odd)
            cd = self.mul(C, D, odd)
            cxdD = self.mul(C, {mu: xi * dD[mu] for mu in odd}, odd)
            r2 = {mu: cxdD[mu] + cEd[mu] - cd[mu] for mu in odd}
            dC = {nu: cE[nu] / xi for nu in even}
        else:
            r2 = {mu: xi * dD[mu] - D[mu] for mu in odd}
            dC = {0: mp.mpc(0)}
        dS = {mu: (r2[mu] - 1j * mu * w * xi * S[mu]) / (xi * xi) for mu in odd}
        return self.pack(dD, dS, dC)

    def integrate_path(self, Y0, path):
        """path: list of (kind, params): ('ray', phi, r_a, r_b) or ('arc', r, phi_a, phi_b); xi = r e^{i phi}"""
        Y = Y0
        for kind, *p in path:
            if kind == "ray":
                phi, ra, rb = p
                u = mp.expj(phi)
                f = lambda r, Z: [v * u for v in self.rhs(r * u, Z)]
                sol = mp.odefun(f, ra, Y)
                Y = sol(rb)
            else:
                r, pa, pb = p                       # odefun integrates forward only: phi = pa + (pb - pa) t
                dph = pb - pa
                f = lambda t, Z: [v * 1j * r * mp.expj(pa + dph * t) * dph for v in self.rhs(r * mp.expj(pa + dph * t), Z)]
                sol = mp.odefun(f, 0, Y)
                Y = sol(1)
        return Y


def lateral_jump(model, J, r0, r1, verbose=True):
    d, s, g = model.series(J)
    phi0 = 3 * mp.pi / 8                       # arg xi on the ray: arg xi^2 = 3 pi / 4
    xi0 = r0 * mp.expj(phi0)
    D, S, C, jt, err = model.series_value(d, s, g, xi0)
    if verbose:
        print(f"   series at |xi| = {mp.nstr(r0, 4)} truncated at j = {jt}, smallest term {mp.nstr(err, 3)}", flush=True)
    Y = model.integrate_path(model.pack(D, S, C), [("ray", phi0, r0, r1), ("arc", r1, phi0, mp.mpf(0))])
    Dp, Sp, Cp = model.unpack(Y)
    jump = {mu: Dp[mu] - mp.conj(Dp[-mu]) for mu in model.odd}
    jumpS = {mu: Sp[mu] - mp.conj(Sp[-mu]) for mu in model.odd}
    return jump, jumpS, (Dp, Sp, Cp)


def fast_lin_series(s, xi, K=80):
    """xi^{-2s} e^{-1/(2 xi^2)} 2F0(1-a,1-b;;-2 xi^2), optimally truncated"""
    a, b = (1 + s) / 2, (2 + s) / 2
    z = 2 * xi * xi
    terms, t = [], mp.mpc(1)
    for k in range(K):
        terms.append(t)
        t = t * (k + 1 - a) * (k + 1 - b) / (k + 1) * (-z)
    kt = min(range(1, K), key=lambda k: abs(terms[k]))
    return xi ** (-2 * s) * mp.exp(-1 / (2 * xi * xi)) * sum(terms[:kt]), abs(terms[kt])


def fast_lin(s, xi1, xia=mp.mpf("0.09")):
    """linear fast solution at real xi1: series data at xia (truncation error ~e^{-1/(2 xia^2)}), integrated outward
    (the fast mode grows outward on the real axis, so this is stable). Returns Dd component."""
    w = mp.re(s / 1j)
    m = Model({1: 0, -1: 0}, w, 1, gravity=False)
    fa, err = fast_lin_series(s, xia)
    fd = mp.diff(lambda x: fast_lin_series(s, x)[0], xia)
    Y0 = [mp.mpc(0), fa, mp.mpc(0), xia * xia * fd + s * xia * fa, mp.mpc(1)]   # order: D_-1, D_1, S_-1, S_1, c_0
    sol = mp.odefun(lambda x, Z: m.rhs(x, Z), xia, Y0)
    return sol(xi1)[1], err


if __name__ == "__main__":
    w0 = 2 * mp.pi / mp.mpf("1.64389375")
    if sys.argv[1] == "linear":
        s = 1j * w0
        a, b = (1 + s) / 2, (2 + s) / 2
        Kp = mp.power(2, -s - mp.mpf(1) / 2) / (mp.gamma(a) * mp.gamma(b))
        m = Model({1: -0.5j, -1: 0.5j}, w0, 1, gravity=False)     # F = sin(theta): d_0 = 2F, harmonic 1 = -i
        for r0, r1, xi1 in ((mp.mpf("0.12"), mp.mpf("0.3"), None),):
            jump, jumpS, _ = lateral_jump(m, 70, r0, r1)
            f, ferr = fast_lin(s, r1)
            Smeas = jump[1] / f / (-1j)         # per unit d_0 harmonic (d_0,1 = -i)
            print(f"r0 {mp.nstr(r0,3)} r1 {mp.nstr(r1,3)}: S = {mp.nstr(Smeas, 12)}   2 pi i K' = {mp.nstr(2j * mp.pi * Kp, 12)}"
                  f"   ratio {mp.nstr(Smeas / (2j * mp.pi * Kp), 12)}  (fast-series trunc. {mp.nstr(ferr, 2)})", flush=True)
