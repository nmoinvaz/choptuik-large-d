"""Planted family for the controls of issue #203: a zero of known position crossing the real axis at w = w_p.

S(x; w) = e^{za x} [ 1F1(a + da; b; dz x) + eps 1F1(a + da + 1; b + 3/2; dz x) ],   dz = zp - za,   normalised to S(0) = 1.

Its Laplace-conjugate function has one cut, from za to zp, with a discontinuity
    u^{a+da-1} (1-u)^{b-a-da-1} [1 + eps' u (1-u)^{1/2}],   zeta = za + u dz,
so c_n = n! sigma_n grows like zp^n n^{a-b}, and the zero near 13.5 comes from the interference of the two ends'
contributions (Kummer's connection formula, DLMF 13.7.2).  (za, zp, a, b)(w) are linear in w through the fits of
fit_kummer.py at w = 1.09 and 1.11 (that fit matches the data's sigma_1..sigma_22 to 0.2-0.9% each, so the structure
is the data's); eps = 0.05 makes the discontinuity non-Kummer.  The complex constant da is solved so that S(13.5; w_p)
= 0 exactly: the planted zero is x1(w_p) = 13.5, real, and x1(w) is followed exactly (mpmath hyp1f1, findroot) at
other w.
"""
import json
import os

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = mp.mpf("0.05")
X_PLANT = mp.mpf("13.5")


def _params(w, fit):
    def lin(key):
        v09 = mp.mpc(*fit["1.09"][key])
        v11 = mp.mpc(*fit["1.11"][key])
        return v09 + (v11 - v09) * (mp.mpf(w) - mp.mpf("1.09")) / mp.mpf("0.02")
    return lin("za"), lin("zp"), lin("a"), lin("b")


def load_fit(path=os.path.join(HERE, "kummer_fit.json")):
    return json.load(open(path))


class Family:
    def __init__(self, wp, fit=None):
        mp.mp.dps = 50
        self.fit = fit or load_fit()
        self.wp = mp.mpf(wp)
        self.da = mp.mpc(0)
        # da: S(13.5; w_p) = 0, from da = 0 by the secant method
        self.da = mp.findroot(lambda d: self._S(X_PLANT, self.wp, d), (mp.mpc(0), mp.mpc("0.01", "0.01")),
                              solver="secant", tol=mp.mpf("1e-40"), maxsteps=200)

    def _S(self, x, w, da=None):
        da = self.da if da is None else da
        za, zp, a, b = _params(w, self.fit)
        dz = zp - za
        a1 = a + da
        return mp.exp(za * x) * (mp.hyp1f1(a1, b, dz * x) + EPS * mp.hyp1f1(a1 + 1, b + mp.mpf(3) / 2, dz * x)) / (1 + EPS)

    def S(self, x, w):
        return self._S(x, w)

    def sigma(self, w, K=22):
        """sigma_0..sigma_K, exact (50 digits), normalised to sigma_0 = 1"""
        za, zp, a, b = _params(w, self.fit)
        dz = zp - za
        a1 = a + self.da
        k1 = [mp.rf(a1, m) * dz ** m / (mp.rf(b, m) * mp.factorial(m)) for m in range(K + 1)]
        k2 = [mp.rf(a1 + 1, m) * dz ** m / (mp.rf(b + mp.mpf(3) / 2, m) * mp.factorial(m)) for m in range(K + 1)]
        e = [za ** j / mp.factorial(j) for j in range(K + 1)]
        s = [mp.fsum(e[n - m] * (k1[m] + EPS * k2[m]) for m in range(n + 1)) / (1 + EPS) for n in range(K + 1)]
        return s

    def x1(self, w, x0=None):
        x0 = X_PLANT if x0 is None else x0
        return mp.findroot(lambda x: self.S(x, w), mp.mpc(x0), tol=mp.mpf("1e-35"))


def noisy(sig, seed, rel=1e-12):
    """each sigma_n times (1 + rel (g1 + i g2)), g standard normal, numpy seed"""
    g = np.random.RandomState(seed).standard_normal((len(sig), 2))
    return [s * (1 + mp.mpf(rel) * mp.mpc(float(g[n, 0]), float(g[n, 1]))) for n, s in enumerate(sig)]
