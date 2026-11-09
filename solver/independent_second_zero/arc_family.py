"""Planted family of issue #214: a Laplace integral over a curved arc, fitted to the data, with a known zero crossing.

    S(x; w) = int_0^1 e^{x gamma(s)} rho(s) ds / int_0^1 rho(s) ds,
    gamma(s) = za + s (zp - za) + kap s (1 - s),   rho(s) = s^{al-1} (1 - s)^{be-1} (e0 + e1 s),

with (za, zp, kap, al, be, e0, e1)(w) linear in w from fit_arc.py (arc_fit.json).  That fit matches the data's
sigma_1..sigma_22 to better than 1e-4 relative at each of w = 1.09, 1.10, 1.11, against 0.2-0.9% for the Kummer
family of planted.py.  The coefficients are the arc's moments, c_n = n! sigma_n = sum_k [s^k] gamma(s)^n
(e0 (al)_k/(al+be)_k + e1 (al)_{k+1}/(al+be)_{k+1}) / norm, exact at any precision.  F(t) = sum c_n t^n then has branch
points at t = 1/za and 1/zp with exponents set by al and be.  Unlike the Kummer family, the density, continued as a
function of zeta = gamma(s), has a square-root branch point at the critical value gamma(s*), gamma'(s*) = 0, and a
non-Beta factor e0 + e1 s.  So the discontinuity is not of Beta/Kummer form.
Planting.  The fitted family's own zero x1(w) near 13.5 has Im x1 = 0 at some w_nat.  The planted family at crossing
w_p is the fitted one shifted in w, S_p(x; w) = S(x; w - (w_p - w_nat)), so its zero crosses the real axis at w_p
exactly and its parameters move with w as the fit does.  x1 is the Newton zero of the series sum c_n x^n/n!, summed to
n = 200 at 160 digits (the last term is below 1e-45 of the largest at |x| = 14); `check_quad` checks S against
tanh-sinh quadrature of the integral.
"""
import json
import os

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["za", "zp", "kap", "al", "be", "e0", "e1"]
NSER = 200
SDPS = 160
X_START = mp.mpf("13.5")


def load_fit(path=os.path.join(HERE, "arc_fit.json")):
    return json.load(open(path))


class ArcBase:
    """The fitted family, unplanted."""

    def __init__(self, fit=None):
        self.fit = fit or load_fit()
        self._cache = {}

    def params(self, w):
        t = (mp.mpf(w) - mp.mpf("1.10")) / mp.mpf("0.01")
        return {k: mp.mpc(*self.fit["A"][k]) + t * mp.mpc(*self.fit["B"][k]) for k in NAMES}

    def coeffs(self, w, nmax=NSER):
        """c_0..c_nmax (c_0 = 1), at SDPS digits"""
        key = (mp.nstr(mp.mpf(w), 40), nmax)
        if key in self._cache:
            return self._cache[key]
        with mp.workdps(SDPS):
            p = self.params(w)
            za, zp, kap, al, be, e0, e1 = [p[k] for k in NAMES]
            path = [za, zp - za + kap, -kap]
            K = 2 * nmax + 3
            pr = [mp.mpc(1)]
            for k in range(1, K):
                pr.append(pr[-1] * (al + k - 1) / (al + be + k - 1))
            P = [mp.mpc(1)]
            c = []
            for n in range(nmax + 1):
                c.append(mp.fsum(P[k] * (e0 * pr[k] + e1 * pr[k + 1]) for k in range(len(P))))
                Q = [mp.mpc(0)] * (len(P) + 2)
                for k, v in enumerate(P):
                    Q[k] += v * path[0]
                    Q[k + 1] += v * path[1]
                    Q[k + 2] += v * path[2]
                P = Q
            c0 = c[0]
            c = [x / c0 for x in c]
        self._cache[key] = c
        return c

    def sigma(self, w, K=22):
        c = self.coeffs(w)
        return [+(c[n] / mp.factorial(n)) for n in range(K + 1)]

    def S(self, x, w, deriv=False):
        c = self.coeffs(w)
        with mp.workdps(SDPS):
            x = mp.mpc(x)
            s = ds = mp.mpc(0)
            tn = mp.mpc(1)
            big = last = mp.mpf(0)
            for n, cn in enumerate(c):
                term = cn * tn
                if n > 0:
                    ds += cn * tn * n / x
                s += term
                big = max(big, abs(term))
                last = abs(term)
                tn = tn * x / (n + 1)
            self.last_rel = last / big
        return (s, ds) if deriv else s

    def x1(self, w, x0=None, tol=mp.mpf("1e-40")):
        x = mp.mpc(X_START if x0 is None else x0)
        with mp.workdps(SDPS):
            for _ in range(80):
                s, ds = self.S(x, w, deriv=True)
                dx = s / ds
                x -= dx
                if abs(dx) < tol * abs(x):
                    return +x
        raise RuntimeError(f"x1 Newton did not converge at w {w}")

    def w_nat(self, lo="1.095", hi="1.105"):
        """the w at which Im x1(w) = 0 (secant)"""
        with mp.workdps(40):
            return mp.findroot(lambda w: self.x1(w).imag, (mp.mpf(lo), mp.mpf(hi)), solver="secant",
                               tol=mp.mpf("1e-30"), maxsteps=60)

    def check_quad(self, x, w):
        """S(x; w) by tanh-sinh quadrature of the arc integral, for comparison with the series"""
        with mp.workdps(50):
            p = self.params(w)
            za, zp, kap, al, be, e0, e1 = [p[k] for k in NAMES]

            def rho(s):
                return s ** (al - 1) * (1 - s) ** (be - 1) * (e0 + e1 * s)

            def gam(s):
                return za + s * (zp - za) + kap * s * (1 - s)
            num = mp.quad(lambda s: mp.exp(mp.mpc(x) * gam(s)) * rho(s), [0, mp.mpf(1) / 4, mp.mpf(1) / 2, mp.mpf(3) / 4, 1])
            den = mp.quad(rho, [0, mp.mpf(1) / 2, 1])
            return num / den


class Family:
    """The planted family: ArcBase shifted in w so that Im x1 = 0 at w = wp.  Same interface as planted.Family."""

    def __init__(self, wp, fit=None, eps=None):
        self.base = ArcBase(fit)
        self.wp = mp.mpf(wp)
        self.wnat = self.base.w_nat()
        self.shift = self.wp - self.wnat
        self.da = self.shift            # logged by second_zero.py as the planting constant

    def _w(self, w):
        return mp.mpf(w) - self.shift

    def sigma(self, w, K=22):
        return self.base.sigma(self._w(w), K)

    def S(self, x, w):
        return self.base.S(x, self._w(w))

    def x1(self, w, x0=None):
        return self.base.x1(self._w(w), x0)


def noisy(sig, seed, rel=1e-12):
    """each sigma_n times (1 + rel (g1 + i g2)), g standard normal, numpy seed (as planted.noisy)"""
    g = np.random.RandomState(seed).standard_normal((len(sig), 2))
    return [s * (1 + mp.mpf(rel) * mp.mpc(float(g[n, 0]), float(g[n, 1]))) for n, s in enumerate(sig)]


if __name__ == "__main__":
    # python3 arc_family.py <dir with I82-win2's sigF_*.json>: the family's exact zero, w_nat, checks, fit vs data
    import sys
    sys.path.insert(0, HERE)
    import lapl
    mp.mp.dps = 50
    D = sys.argv[1]
    b = ArcBase()
    for w in ["1.09", "1.10", "1.11"]:
        x = b.x1(mp.mpf(w))
        lr_tail = b.last_rel
        sd, _ = lapl.load_sigma(os.path.join(D, f"sigF_w{w}_K45_J1500_p1280.json"))
        sm = b.sigma(mp.mpf(w))
        lr = max(abs(mp.log(sm[n] / sd[n])) for n in range(1, 23))
        print(f"w {w}: x1 {mp.nstr(x, 15)} (last series term / largest {mp.nstr(lr_tail, 3)}); "
              f"max |log(sigma_family/sigma_data)|, n = 1..22: {mp.nstr(lr, 3)}", flush=True)
    xq = mp.mpc("13.47", "-0.01")
    q, s = b.check_quad(xq, mp.mpf("1.10")), b.S(xq, mp.mpf("1.10"))
    print(f"S(13.47 - 0.01i; 1.10): series {mp.nstr(s, 20)}, quadrature {mp.nstr(q, 20)}, rel diff "
          f"{mp.nstr(abs(q - s) / abs(s), 3)}")
    wn = b.w_nat()
    print(f"w_nat (Im x1 = 0) {mp.nstr(wn, 15)}, x1(w_nat) {mp.nstr(b.x1(wn), 15)}")
    print(f"next zero inward at 1.10 (Newton from 8 - 1i): {mp.nstr(b.x1(mp.mpf('1.10'), mp.mpc(8, -1)), 10)}")
    for wp in ["1.095", "1.15"]:
        f = Family(wp)
        f.wnat, f.shift = wn, mp.mpf(wp) - wn
        for w in ["1.09", "1.10", "1.11"]:
            sd, _ = lapl.load_sigma(os.path.join(D, f"sigF_w{w}_K45_J1500_p1280.json"))
            sm = f.sigma(mp.mpf(w))
            lr = max(abs(mp.log(sm[n] / sd[n])) for n in range(1, 23))
            print(f"planted w_p {wp} (shift {mp.nstr(f.shift, 6)}): x1({w}) {mp.nstr(f.x1(mp.mpf(w)), 12)}; "
                  f"max |log(sigma_planted/sigma_data)| {mp.nstr(lr, 3)}", flush=True)
