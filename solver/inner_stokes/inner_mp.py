"""inner_np.py's inner Stokes model, in mpmath at a set precision (#1076): the same equations, start series, ray and
arc, and de-mixed residual, with a Taylor-series stepper in place of DOP853, so that the rounding floor of R_1
(argcount's sig0 = 1e-15 |Y(xi1)| in double precision) falls with mp.dps.

    xi^2 D' = [C S]_o - i mu w xi D,  xi^2 S' = r2 - i mu w xi S,  xi C' = [C E]_e,
    r2 = [C xi D']_o + [cE D]_o - [C D]_o,  E = [D D]_e / 2,  cE = [C E]_e          (inner_np.py's docstring)

Taylor stepper: at a centre xi_c the local series y_n of (D, S, C) in h = xi - xi_c follows from the equations by
Cauchy products (xi^2 = xi_c^2 + 2 xi_c h + h^2), to order p; the step |h| is set so that the last term is below
10^-dps of the first. The ray is a straight segment in xi; the arc r1 e^{it} runs as chords of at most ARC_DT rad
between its points (the solution is analytic between a chord and its arc, which max|Y| already guards in inner_np).

    python3 inner_mp.py compare A WF B3 C3 [--r1=0.17] [--dps=34] [--order=40] [--margin=12]
        R_1, r_3 here and from inner_np.residual_demixed at the same point, with this run's seconds
"""
import math
import os
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ARC_DT = 0.02
MUTATE = set()          # controls: "conv" drops [C S] from D's equation, "series" keeps only 8 terms of the start series


def W0():
    return 2 * mp.pi / mp.mpf("1.64389375")


def DELTA():
    return mp.pi / 8


def Kp(s):
    a, b = (1 + s) / 2, (2 + s) / 2
    return mp.power(2, -s - mp.mpf(1) / 2) / (mp.gamma(a) * mp.gamma(b))


class InnerMP:
    def __init__(self, M, w, Fh, M0=None, lam=1):
        self.M, self.w = M, mp.mpf(w) if not isinstance(w, mp.mpf) else w
        L = M + 2
        self.L, self.n = L, 2 * L + 1
        ks = range(-L, L + 1)
        self.k = list(ks)
        self.odd = [1 if (abs(k) <= M and k % 2 == 1) else 0 for k in ks]
        self.even = [1 if (abs(k) <= M + 1 and k % 2 == 0) else 0 for k in ks]
        M0 = M if M0 is None else M0
        self.wt = [mp.mpf(1) if abs(k) <= M0 + 1 else mp.mpf(lam) for k in ks]
        self.F = [mp.mpc(Fh.get(k, 0)) * self.odd[i] for i, k in enumerate(ks)]
        self.imw = [mp.mpc(0, k) * self.w * self.odd[i] for i, k in enumerate(ks)]
        self.oi = [i for i in range(self.n) if self.odd[i]]
        self.ei = [i for i in range(self.n) if self.even[i]]

    def zero(self):
        return [mp.mpc(0)] * self.n

    def mul(self, a, b, space):
        """the weighted convolution of inner_np.Inner.mul: c_k = w_k space_k sum_{i+j=k} w_i a_i w_j b_j"""
        L, n, wt = self.L, self.n, self.wt
        aw = [(i, wt[i] * a[i]) for i in range(n) if a[i] != 0]
        bw = [(j, wt[j] * b[j]) for j in range(n) if b[j] != 0]
        out = [mp.mpc(0)] * n
        for i, x in aw:
            for j, y in bw:
                k = i + j - L
                if 0 <= k < n and space[k]:
                    out[k] += x * y
        return [out[k] * wt[k] * space[k] if space[k] else mp.mpc(0) for k in range(n)]

    @staticmethod
    def add(*vs):
        return [sum(t) for t in zip(*vs)]

    @staticmethod
    def scal(c, v):
        return [c * x for x in v]

    def series_at(self, xi0, J):
        """inner_np.Inner.series_at at complex xi0, in mpmath: (Y, jt, the smallest term over the first)"""
        q = xi0 * xi0
        o, e, mul = self.odd, self.even, self.mul
        D = [self.scal(2, self.F)]
        S, G = [], [[mp.mpc(1) if (e[i] and self.k[i] == 0) else mp.mpc(0) for i in range(self.n)]]
        E, CE = [self.zero()], [self.zero()]
        mags = [max(abs(x) for x in D[0])]
        for j in range(J):
            sj = [(2 * j + 1) * D[j][i] + self.imw[i] * D[j][i] for i in range(self.n)]
            for i in range(1, j + 1):
                sj = self.add(sj, self.scal(-1, mul(G[i], S[j - i], o)))
            S.append([sj[i] * o[i] for i in range(self.n)])
            m = j + 1
            em = self.zero()
            for i in range(m):
                em = self.add(em, mul(D[i], D[m - 1 - i], e))
            E.append(self.scal(q / 2, em))
            cm = self.zero()
            for i in range(m):
                cm = self.add(cm, mul(G[i], E[m - i], e))
            CE.append(cm)
            G.append(self.scal(mp.mpf(1) / (2 * m), cm))
            num = [q * ((2 * j + 2) * S[j][i] + self.imw[i] * S[j][i]) for i in range(self.n)]
            for i in range(1, j + 2):
                num = self.add(num, self.scal(-2 * (j + 1 - i), mul(G[i], D[j + 1 - i], o)))
            for mm in range(1, j + 2):
                num = self.add(num, self.scal(-1, mul(CE[mm], D[j + 1 - mm], o)))
            D.append([num[i] * o[i] / (2 * j + 2) for i in range(self.n)])
            mags.append(max(abs(x) for x in D[-1]))
            if j > 10 and mags[-1] > 10 * min(mags):
                break
        jt = min(range(1, len(mags)), key=lambda t: mags[t])
        if "series" in MUTATE:                   # the control: only 8 terms (half the terms change Y by 3e-34 only)
            jt = min(jt, 8)
        Dd = self.scal(xi0, self.add(*D[:jt]))
        Sg = self.scal(q, self.add(*S[:jt]))
        Cc = self.add(*G[:jt])
        return (Dd, Sg, Cc), jt, mags[jt] / mags[0]

    def taylor(self, xc, Y, p):
        """the local series (d_n, s_n, c_n), n = 0..p, of the solution through Y = (D, S, C) at xi = xc"""
        o, e, mul, n_ = self.odd, self.even, self.mul, self.n
        d, s, c = [list(Y[0])], [list(Y[1])], [list(Y[2])]
        E = [self.scal(mp.mpf(1) / 2, mul(d[0], d[0], e))]
        cE = [mul(c[0], E[0], e)]
        qd = []                                   # coefficients of xi D'
        x2 = xc * xc
        for n in range(p):
            cs = self.zero()
            if "conv" not in MUTATE:
                for i in range(n + 1):
                    cs = self.add(cs, mul(c[i], s[n - i], o))
            dm1 = d[n - 1] if n >= 1 else self.zero()
            pn = [cs[i] - self.imw[i] * (xc * d[n][i] + dm1[i]) for i in range(n_)]
            dn1 = [(pn[i] - 2 * xc * n * d[n][i] - (n - 1) * dm1[i]) / (x2 * (n + 1)) * o[i] for i in range(n_)]
            d.append(dn1)
            cn1 = [(cE[n][i] - n * c[n][i]) / (xc * (n + 1)) * e[i] for i in range(n_)]
            c.append(cn1)
            qd.append([xc * (n + 1) * d[n + 1][i] + n * d[n][i] for i in range(n_)])
            r2 = self.zero()
            for i in range(n + 1):
                r2 = self.add(r2, mul(c[i], qd[n - i], o), mul(cE[i], d[n - i], o), self.scal(-1, mul(c[i], d[n - i], o)))
            sm1 = s[n - 1] if n >= 1 else self.zero()
            sn1 = [(r2[i] - self.imw[i] * (xc * s[n][i] + sm1[i]) - 2 * xc * n * s[n][i] - (n - 1) * sm1[i])
                   / (x2 * (n + 1)) * o[i] for i in range(n_)]
            s.append(sn1)
            En = self.zero()
            for i in range(n + 2):
                En = self.add(En, mul(d[i], d[n + 1 - i], e))
            E.append(self.scal(mp.mpf(1) / 2, En))
            cEn = self.zero()
            for i in range(n + 2):
                cEn = self.add(cEn, mul(c[i], E[n + 1 - i], e))
            cE.append(cEn)
        return d, s, c

    def advance(self, xa, xb, Y, p, tol):
        """Y at xb from Y at xa along the straight segment, by Taylor steps; returns (Y, steps, max|Y|)"""
        steps, ymax = 0, max(abs(v) for part in Y for v in part)
        x = xa
        while True:
            rem = xb - x
            if abs(rem) == 0:
                return Y, steps, ymax
            d, s, c = self.taylor(x, Y, p)
            norm = lambda v: max(abs(t) for t in v)
            a0 = max(norm(d[0]), norm(s[0]), norm(c[0]), mp.mpf(10) ** -mp.mp.dps)
            ap = max(norm(d[p]), norm(s[p]), norm(c[p]), norm(d[p - 1]), norm(s[p - 1]), norm(c[p - 1]))
            hmax = mp.mpf("0.5") * (tol * a0 / ap) ** (mp.mpf(1) / p) if ap > 0 else abs(rem)
            h = rem if abs(rem) <= hmax else rem / abs(rem) * hmax
            hp = [mp.mpc(1)]
            for _ in range(p):
                hp.append(hp[-1] * h)
            ev = lambda cs: [sum(cs[k][i] * hp[k] for k in range(p + 1)) for i in range(self.n)]
            Y = (ev(d), ev(s), ev(c))
            x = x + h
            steps += 1
            ymax = max(ymax, max(abs(v) for part in Y for v in part))

    def r0_for(self, margin=12):
        sd = mp.sin(DELTA())
        return mp.sqrt(sd / (2 * (self.M * self.w * (mp.pi / 2 + DELTA()) + margin)))

    def jumps(self, r1, p=40, margin=12):
        """J_k(r1) on the k grid, and info (r0, jt, terr, steps, max|Y|)"""
        tol = mp.mpf(10) ** (-mp.mp.dps)
        phi = (mp.pi / 2 + DELTA()) / 2
        r0 = self.r0_for(margin)
        u = mp.expj(phi)
        Y, jt, terr = self.series_at(r0 * u, J=int(mp.mpf("1.2") / (2 * r0 * r0)) + 40)
        Y, n1, y1 = self.advance(r0 * u, r1 * u, Y, p, tol)
        nseg = max(1, int(mp.ceil(phi / ARC_DT)))
        steps, ymax, t = n1, y1, phi
        for kseg in range(nseg):
            t1 = phi * (nseg - kseg - 1) / nseg
            Y, nk, yk = self.advance(r1 * mp.expj(t), r1 * mp.expj(t1) if kseg < nseg - 1 else mp.mpf(r1), Y, p, tol)
            steps, ymax, t = steps + nk, max(ymax, yk), t1
        D = Y[0]
        Jd = [D[i] - mp.conj(D[self.n - 1 - i]) for i in range(self.n)]
        return Jd, dict(r0=r0, jt=jt, terr=terr, steps=steps, ymax=ymax)


def Fh_from(x, M):
    A = x[0]
    h = {1: A / mp.mpc(0, 2), -1: -A / mp.mpc(0, 2)}
    for i, nu in enumerate(range(3, M + 1, 2)):
        B, C = x[2 + 2 * i], x[3 + 2 * i]
        h[nu] = B / mp.mpc(0, 2) + C / 2
        h[-nu] = mp.conj(h[nu])
    return h


def F2mean(Fh):
    return sum(abs(v) ** 2 for v in Fh.values())


def residual_demixed(x, M=3, r1="0.17", NTH=64, p=40, margin=12):
    """inner_np.residual_demixed in mpmath (its numpy FFTs written as sums over the same NTH points);
    returns ([Re r_1, Im r_1, Re r_3, Im r_3, ...], info)"""
    x = [mp.mpf(v) for v in x]
    r1 = mp.mpf(r1)
    w = W0() * x[1]
    Fh = Fh_from(x, M)
    m = InnerMP(M, w, Fh)
    Jd, info = m.jumps(r1, p, margin)
    th = [2 * mp.pi * j / NTH for j in range(NTH)]
    kk = [k if k < NTH // 2 else k - NTH for k in range(NTH)]
    Fth = [mp.re(sum(v * mp.expj(mu * t) for mu, v in Fh.items())) for t in th]
    F2h = [sum(Fth[j] ** 2 * mp.expj(-2 * mp.pi * j * k / NTH) for j in range(NTH)) / NTH for k in range(NTH)]
    Pi = [mp.re(sum((F2h[k] / mp.mpc(0, kk[k]) if kk[k] != 0 else 0) * mp.expj(2 * mp.pi * j * k / NTH)
                    for k in range(NTH))) / w for j in range(NTH)]
    L = m.L
    Jth = [sum(Jd[k + L] * mp.expj(k * t) for k in range(-M, M + 1) if k % 2) for t in th]
    g = [Jth[j] * mp.exp(-Pi[j]) for j in range(NTH)]
    sc = mp.exp(1 / (2 * r1 * r1)) * r1 ** (-2 * F2mean(Fh))
    out = []
    for nu in range(1, M + 1, 2):
        qn = sum(g[j] * mp.expj(-2 * mp.pi * j * (nu % NTH) / NTH) for j in range(NTH)) / NTH
        z = qn * r1 ** mp.mpc(0, 2 * nu * w) * sc / (2 * mp.pi * abs(Kp(mp.mpc(0, nu * w))))
        out += [mp.re(z), mp.im(z)]
    return out, info


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    for m_ in opts.get("mutate", "").split(","):
        if m_:
            MUTATE.add(m_)
    if args and args[0] == "compare":
        A, wf, B3, C3 = args[1:5]
        r1 = opts.get("r1", "0.17")
        mp.mp.dps = int(opts.get("dps", 34))
        t0 = time.time()
        r, info = residual_demixed([A, wf, B3, C3], 3, r1, p=int(opts.get("order", 40)), margin=int(opts.get("margin", 12)))
        tm = time.time() - t0
        sys.path.insert(0, HERE)
        import numpy as np
        import inner_np as I
        rf = I.residual_demixed(np.array([float(A), float(wf), float(B3), float(C3)]), 3, r1=float(r1))
        print(f"dps {mp.mp.dps} order {opts.get('order', 40)} margin {opts.get('margin', 12)}: {tm:.1f} s, "
              f"steps {info['steps']}, series jt {info['jt']} terr {mp.nstr(info['terr'], 3)}, "
              f"max|Y| {mp.nstr(info['ymax'], 4)}")
        for i, name in enumerate(("Re r1", "Im r1", "Re r3", "Im r3")):
            print(f"  {name}: mp {mp.nstr(r[i], 20)}   f64 {rf[i]!r}   diff {float(r[i] - rf[i]):.3e}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
