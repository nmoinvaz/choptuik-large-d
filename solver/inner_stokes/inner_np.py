"""Inner Stokes jump with gravity, all in double precision, Galerkin-truncated in theta with a homotopy weight.

Inner equations (xi = x/sqrt(n), Dd = p - m, sigma = sqrt(n)(p+m), c), as in solver/inner_stokes/lateral.py:
    xi^2 Dd_xi  + i mu w xi Dd  = [c sigma]_mu
    xi^2 sig_xi + i mu w xi sig = [c xi Dd_xi]_mu + [c E Dd]_mu - [c Dd]_mu,   E = [Dd^2/2]
    xi c_xi = [c E]_nu
Dd, sigma on odd |mu| <= M, c and E on even |nu| <= M + 1. Every product a*b -> k is weighted by
w_i w_j w_k with w = 1 on the harmonics of the previous truncation (|mu| <= M0, |nu| <= M0 + 1) and w = lam on the
new ones, so lam = 0 decouples the new harmonics and lam = 1 is the full M-harmonic model (homotopy in M).

Formal series: computed directly in scaled form at the start point xi0 (coefficients d_j xi0^(2j) etc., no overflow),
optimally truncated. y+ (lateral sum from above) is pinned on the ray arg xi^2 = pi/2 + delta, where every fast mode
e^{-1/(2xi^2)} xi^{-2 i nu w} is exponentially large; r0 is chosen so that the weakest one (nu = -M) still dominates by
e^{12}. Then: ray out to |xi| = r1, arc to the real axis; jump J_nu = y+_nu - conj(y+_{-nu}) for real F.

    python3 inner_np.py checklin M      linear (no gravity) check: R_nu = J_nu / (2 pi i K'_nu f_nu 2 F_nu) = 1
    python3 inner_np.py m1zero          reproduce the M = 1 zero (A = 2.54215, w/w_inf = 0.77699) on the new ray
"""
import os
import sys
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp

W0 = float(2 * mp.pi / mp.mpf("1.64389375"))
DELTA = np.pi / 8


def Kp(s):
    a, b = (1 + s) / 2, (2 + s) / 2
    return complex(mp.power(2, -s - mp.mpf(1) / 2) / (mp.gamma(a) * mp.gamma(b)))


class Inner:
    def __init__(self, M, w, Fh, gravity=True, M0=None, lam=1.0):
        self.M, self.w, self.grav = M, w, gravity
        self.L = M + 2
        k = np.arange(-self.L, self.L + 1)
        self.k = k
        self.odd = (np.abs(k) <= M) & (k % 2 == 1)
        self.even = (np.abs(k) <= M + 1) & (k % 2 == 0) if gravity else (k == 0)
        M0 = M if M0 is None else M0
        self.wt = np.where(np.abs(k) <= M0 + 1, 1.0, lam)
        self.F = np.zeros(2 * self.L + 1, complex)
        for mu, v in Fh.items():
            self.F[mu + self.L] = v
        self.F *= self.odd
        self.imw = 1j * k * w * self.odd

    def mul(self, a, b, space):
        L = self.L
        c = np.convolve(self.wt * a, self.wt * b)[L:3 * L + 1]
        return self.wt * c * space

    # ---- formal series, scaled, at complex xi0
    def series_at(self, xi0, J):
        q = xi0 * xi0
        o, e = self.odd, self.even
        D = [2 * self.F]
        S, G, E, CE = [], [e * (self.k == 0) * (1.0 + 0j)], [np.zeros_like(self.F)], [np.zeros_like(self.F)]
        mags = [np.abs(D[0]).max()]
        for j in range(J):
            sj = (2 * j + 1) * D[j] + self.imw * D[j]
            for i in range(1, j + 1):
                sj = sj - self.mul(G[i], S[j - i], o)
            S.append(sj * o)
            m = j + 1
            if self.grav:
                em = np.zeros_like(self.F)
                for i in range(m):
                    em = em + self.mul(D[i], D[m - 1 - i], e)
                em = q * em / 2
                E.append(em)
                cm = np.zeros_like(self.F)
                for i in range(m):
                    cm = cm + self.mul(G[i], E[m - i], e)
                CE.append(cm)
                G.append(cm / (2 * m))
            else:
                E.append(np.zeros_like(self.F)); CE.append(np.zeros_like(self.F)); G.append(np.zeros_like(self.F))
            num = q * ((2 * j + 2) * S[j] + self.imw * S[j])
            for i in range(1, j + 2):
                num = num - 2 * (j + 1 - i) * self.mul(G[i], D[j + 1 - i], o)
            if self.grav:
                for mm in range(1, j + 2):
                    num = num - self.mul(CE[mm], D[j + 1 - mm], o)
            D.append(num * o / (2 * j + 2))
            mags.append(np.abs(D[-1]).max())
            if j > 10 and mags[-1] > 10 * min(mags):
                break
        jt = int(np.argmin(mags[1:])) + 1
        Dd = xi0 * sum(D[:jt]); Sg = q * sum(S[:jt]); Cc = sum(G[:jt])
        return self.pack(Dd, Sg, Cc), jt, mags[jt] / mags[0]

    def pack(self, D, S, C):
        return np.concatenate([D[self.odd], S[self.odd], C[self.even]])

    def unpack(self, Y):
        no, ne = self.odd.sum(), self.even.sum()
        D = np.zeros_like(self.F); S = np.zeros_like(self.F); C = np.zeros_like(self.F)
        D[self.odd] = Y[:no]; S[self.odd] = Y[no:2 * no]; C[self.even] = Y[2 * no:2 * no + ne]
        return D, S, C

    def rhs(self, xi, Y):
        o, e = self.odd, self.even
        D, S, C = self.unpack(Y)
        dD = (self.mul(C, S, o) - self.imw * xi * D) / (xi * xi)
        if self.grav:
            E = self.mul(D, D, e) / 2
            cE = self.mul(C, E, e)
            r2 = self.mul(C, xi * dD, o) + self.mul(cE, D, o) - self.mul(C, D, o)
            dC = cE / xi
        else:
            r2 = xi * dD - D
            dC = np.zeros_like(C)
        dS = (r2 - self.imw * xi * S) / (xi * xi)
        return self.pack(dD, dS, dC)

    def integrate(self, Y, kind, a, b, par, rtol=1e-12, atol=1e-16):
        if kind == "ray":
            u = np.exp(1j * par)
            f = lambda t, Z: self.rhs(t * u, Z) * u
        else:
            r = par
            f = lambda t, Z: self.rhs(r * np.exp(1j * t), Z) * 1j * r * np.exp(1j * t)
        sol = solve_ivp(f, (a, b), Y, method="DOP853", rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[:, -1]

    def r0_for(self, margin=12.0):
        sd = np.sin(DELTA)
        return float(np.sqrt(sd / (2 * (self.M * self.w * (np.pi / 2 + DELTA) + margin))))

    def jumps(self, r1s=(0.2,), r0=None, rtol=1e-12):
        """J_nu(r1) for nu > 0, at each r1 (dict r1 -> array over k)"""
        phi = (np.pi / 2 + DELTA) / 2
        r0 = r0 or self.r0_for()
        Y, jt, terr = self.series_at(r0 * np.exp(1j * phi), J=int(1.2 / (2 * r0 * r0)) + 40)
        out = {}
        rprev = r0
        for r1 in sorted(r1s):
            Y = self.integrate(Y, "ray", rprev, r1, phi, rtol)
            rprev = r1
            Ye = self.integrate(Y, "arc", phi, 0.0, r1, rtol)
            D, S, C = self.unpack(Ye)
            Jd = D - np.conj(D[::-1])                  # index k <-> -k reversal on the symmetric grid
            out[r1] = Jd
        return out, (r0, jt, terr)


# ---------------- unknowns / residuals
def Fh_from(x, M):
    """x = [A, wf, B3, C3, B5, C5, ...]: F = A sin t + sum (B_nu sin nu t + C_nu cos nu t)"""
    A = x[0]
    h = {1: A / 2j, -1: -A / 2j}
    for i, nu in enumerate(range(3, M + 1, 2)):
        B, C = x[2 + 2 * i], x[3 + 2 * i]
        h[nu] = B / 2j + C / 2
        h[-nu] = np.conj(h[nu])
    return h


def F2mean(Fh):
    return sum(abs(v) ** 2 for v in Fh.values())


def residual(x, M, M0=None, lam=1.0, r1=0.2):
    """normalized jump: r_nu = J_nu(r1) e^{1/(2 r1^2)} r1^{-2<F^2>} / (2 pi |K'_nu|), nu = 1, 3, .., M; Re/Im stacked"""
    w = W0 * x[1]
    Fh = Fh_from(x, M)
    m = Inner(M, w, Fh, True, M0, lam)
    J, info = m.jumps((r1,))
    Jd = J[r1]
    sc = np.exp(1 / (2 * r1 * r1)) * r1 ** (-2 * F2mean(Fh))
    out = []
    for nu in range(1, M + 1, 2):
        z = Jd[nu + m.L] * sc / (2 * np.pi * abs(Kp(1j * nu * w)))
        out += [z.real, z.imag]
    return np.array(out)


if __name__ == "__main__":
    import time
    if sys.argv[1] == "checklin":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from lateral_np import fast_lin
        M = int(sys.argv[2])
        Fh = {}
        for nu in range(1, M + 1, 2):
            Fh[nu] = (0.3 + 0.1j) / nu; Fh[-nu] = np.conj(Fh[nu])
        t = time.time()
        m = Inner(M, W0, Fh, gravity=False)
        J, info = m.jumps((0.15, 0.2))
        print(f"M {M}, r0 {info[0]:.4f}, series truncated at j = {info[1]} (smallest/first {info[2]:.1e}), {time.time() - t:.1f} s")
        for r1, Jd in J.items():
            print(f"  r1 {r1}: " + "  ".join(
                f"R_{nu} {Jd[nu + m.L] / (2j * np.pi * Kp(1j * nu * W0) * fast_lin(1j * nu * W0, r1) * 2 * Fh[nu]):.8f}"
                for nu in range(1, M + 1, 2)))
    elif sys.argv[1] == "m1zero":
        for r1 in (0.2, 0.25):
            t = time.time()
            r = residual(np.array([2.54215, 0.77699]), 1, r1=r1)
            print(f"r1 {r1}: residual at the old M=1 zero {r}  ({time.time() - t:.1f} s)")


def residual_demixed(x, M, M0=None, lam=1.0, r1=0.2, NTH=64):
    """like residual(), but the jump is de-mixed first: S(theta0) = J(theta) e^{-Pi(theta)} shifted to the label
    theta0 = theta - 2 w ln xi (Pi = (1/w) int (F^2 - <F^2>)), so that harmonic nu of the result is the multiplier S_nu
    and the huge high-harmonic jumps (|K'_nu| ~ e^{pi |nu| w / 2}) do not leak into the low conditions.
    Same zero set as residual() (invertible linear map on the truncated jump)."""
    w = W0 * x[1]
    Fh = Fh_from(x, M)
    m = Inner(M, w, Fh, True, M0, lam)
    J, info = m.jumps((r1,))
    Jd = J[r1]
    th = 2 * np.pi * np.arange(NTH) / NTH
    kk = np.fft.fftfreq(NTH, 1 / NTH)
    Fth = sum(v * np.exp(1j * mu * th) for mu, v in Fh.items()).real
    F2h = np.fft.fft(Fth ** 2) / NTH
    Pi = np.real(np.fft.ifft(np.where(kk != 0, F2h / (1j * kk + (kk == 0)), 0)) * NTH) / w
    Jth = sum(Jd[k + m.L] * np.exp(1j * k * th) for k in range(-M, M + 1) if k % 2)
    q = np.fft.fft(Jth * np.exp(-Pi)) / NTH
    sc = np.exp(1 / (2 * r1 * r1)) * r1 ** (-2 * F2mean(Fh))
    out = []
    for nu in range(1, M + 1, 2):
        z = q[nu % NTH] * r1 ** (2j * nu * w) * sc / (2 * np.pi * abs(Kp(1j * nu * w)))
        out += [z.real, z.imag]
    return np.array(out)
