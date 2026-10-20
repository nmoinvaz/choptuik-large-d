"""Check of the fast-mode WKB with gravity (Proposition 3 draft).

Fast operator, accurate to O(1) relative to its O(n) part: b = 1, G = 0, c = 1 - F(theta)^2 (1 - x^2)/n,
    x(x-c) p_x = -w x p_th - c p + n/2 c (p+m),   x(x+c) m_x = -w x m_th + c m - n/2 c (p+m).
WKB: p = e^{-n Phi} A, m = e^{-n Phi} A (1-x)/(1+x), Phi = ln x - ln(1-x^2)/2, and
    x (1-x^2) A_x - 2 w x^2 A_th = (1 + x - 2 x^2 F^2) A
(characteristics theta = theta0 + w ln(1-x^2)). Integrate the exact linear system inward from x1 (data = WKB) to x2
in the rescaled variables a = p e^{n Phi}, bm = m e^{n Phi}, and compare with the transported A at x2.
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp, quad

NT = 64
th = 2 * np.pi * np.arange(NT) / NT
KK = np.fft.fftfreq(NT, 1 / NT)
w = 2 * np.pi / 1.64389375
Ff = lambda t: 2.72 * np.sin(t) + 0.22 * np.sin(3 * t)
Phi = lambda x: np.log(x) - 0.5 * np.log(1 - x * x)
dPhi = lambda x: 1 / (x * (1 - x * x))


def dth(f):
    return np.fft.ifft(np.fft.fft(f) * (1j * KK))


def A0(t):               # initial amplitude at x1 (any smooth periodic function)
    return np.exp(1j * t) + 0.3 * np.exp(-3j * t)


def transported(x1, x2, gravity=True):
    """A(x2, theta) from A(x1, .) = A0 along characteristics"""
    out = np.zeros(NT, complex)
    for i, t2 in enumerate(th):
        t1 = t2 + w * (np.log(1 - x1 ** 2) - np.log(1 - x2 ** 2))
        tc = lambda x: t1 + w * np.log((1 - x * x) / (1 - x1 ** 2))
        R = lambda x: (1 + x - (2 * x * x * Ff(tc(x)) ** 2 if gravity else 0)) / (x * (1 - x * x))
        val, _ = quad(R, x1, x2, limit=200, epsabs=1e-13, epsrel=1e-13)
        out[i] = A0(t1) * np.exp(val)
    return out


def exact(n, x1, x2, gravity=True):
    F2 = Ff(th) ** 2 if gravity else 0 * th
    def rhs(x, Y):
        a, bm = Y[:NT], Y[NT:]
        c = 1 - F2 * (1 - x * x) / n
        # p = e^{-n Phi} a: p_x = e^{-n Phi}(a_x - n Phi' a)
        pa = (-w * x * dth(a) - c * a + n / 2 * c * (a + bm)) / (x * (x - c)) + n * dPhi(x) * a
        pb = (-w * x * dth(bm) + c * bm - n / 2 * c * (a + bm)) / (x * (x + c)) + n * dPhi(x) * bm
        return np.r_[pa, pb]
    a1 = A0(th)
    Y0 = np.r_[a1, a1 * (1 - x1) / (1 + x1)]
    sol = solve_ivp(rhs, (x1, x2), Y0, method="DOP853", rtol=1e-11, atol=1e-13)
    return sol.y[:NT, -1]


if __name__ == "__main__":
    x1, x2 = 0.9, 0.72
    for grav in (False, True):
        At = transported(x1, x2, grav)
        print(f"gravity {grav}: |A(x2)| max {np.abs(At).max():.4f}", flush=True)
        for n in (100, 200, 400, 800, 1600):
            a = exact(n, x1, x2, grav)
            err = np.abs(a - At).max() / np.abs(At).max()
            print(f"   n {n:5d}: rel. error of WKB amplitude at x2 {err:.3e}   n * err {n * err:.3f}", flush=True)
