"""Fast-type Stokes multiplier of the inner problem with gravity: lateral solutions, double-precision integrator.

Same system and truncation as lateral.py (the formal series is still built there, in mpmath, then rounded):
    xi^2 Dd_xi  + i mu w xi Dd  = [c sigma]_mu
    xi^2 sig_xi + i mu w xi sig = [c xi Dd_xi]_mu + [c E Dd]_mu - [c Dd]_mu,   E = [Dd^2/2]_even
    xi c_xi = [c E]_nu
on odd |mu| <= M (Dd, sigma) and even |nu| <= M + 1 (c, E), every product projected onto its result's space.

y+ = lateral Borel sum from above: optimally truncated series at xi0 = r0 e^{3 i pi/8} (arg xi^2 = 3 pi/4, where every
fast mode e^{-1/(2 xi^2)} xi^{-2 s_mu} is exponentially large, so the series data select y+ uniquely), integrated along
the ray to |xi| = r1 and along the arc |xi| = r1 to the real axis. For real F, y- = conj(y+_{-mu}) on the real axis,
so the Stokes jump is J_mu = y+_mu - conj(y+_{-mu}) (computed for Dd and sigma).

Normalization: R_mu = J(Dd)_mu(r1) / (2 pi i K'(s_mu) f_mu(r1) 2 F_mu), with f_mu the linear fast solution
xi^{-2 s} e^{-1/(2 xi^2)} 2F0(1-a,1-b;;-2xi^2) (Borel-summed by integrating outward from its series at xi = 0.09).
R_mu = 1 without gravity (checked to 1e-10). The selection condition is J = 0.

    python3 lateral_np.py check                     linear check against 2 pi i K'
    python3 lateral_np.py scan M wf A [B3 C3]       R_mu for F = A sin t + B3 sin 3t + C3 cos 3t, w = wf w_inf
"""
import sys
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from lateral import Model

W0 = float(2 * mp.pi / mp.mpf("1.64389375"))


class NP:
    def __init__(self, M, w, gravity=True):
        self.M, self.w, self.grav = M, w, gravity
        self.odd = np.array([mu for mu in range(-M, M + 1) if mu % 2])
        self.even = np.array([nu for nu in range(-(M + 1), M + 2) if nu % 2 == 0]) if gravity else np.array([0])
        self.L = M + 2                       # dense index range -L..L
        self.no, self.ne = len(self.odd), len(self.even)

    def dense(self, idx, v):
        a = np.zeros(2 * self.L + 1, complex)
        a[idx + self.L] = v
        return a

    def mul(self, ia, a, ib, b, iout):
        A, B = self.dense(ia, a), self.dense(ib, b)
        c = np.convolve(A, B)                # index range -2L..2L
        return c[iout + 2 * self.L]

    def rhs(self, xi, Y):
        o, e, w = self.odd, self.even, self.w
        D, S, C = Y[:self.no], Y[self.no:2 * self.no], Y[2 * self.no:]
        cs = self.mul(e, C, o, S, o)
        dD = (cs - 1j * o * w * xi * D) / (xi * xi)
        if self.grav:
            E = self.mul(o, D, o, D, e) / 2
            cE = self.mul(e, C, e, E, e)
            r2 = self.mul(e, C, o, xi * dD, o) + self.mul(e, cE, o, D, o) - self.mul(e, C, o, D, o)
            dC = cE / xi
        else:
            r2 = xi * dD - D
            dC = np.zeros(1, complex)
        dS = (r2 - 1j * o * w * xi * S) / (xi * xi)
        return np.concatenate([dD, dS, dC])

    def path(self, Y0, segs, rtol=1e-12, atol=1e-15):
        Y = Y0
        for kind, *p in segs:
            if kind == "ray":
                phi, ra, rb = p
                u = np.exp(1j * phi)
                f = lambda t, Z: self.rhs(t * u, Z) * u
                sol = solve_ivp(f, (ra, rb), Y, method="DOP853", rtol=rtol, atol=atol)
            elif kind == "arc":
                r, pa, pb = p
                f = lambda t, Z: self.rhs(r * np.exp(1j * t), Z) * 1j * r * np.exp(1j * t)
                sol = solve_ivp(f, (pa, pb), Y, method="DOP853", rtol=rtol, atol=atol)
            else:                             # real axis
                xa, xb = p
                sol = solve_ivp(lambda t, Z: self.rhs(t, Z), (xa, xb), Y, method="DOP853", rtol=rtol, atol=atol)
            if not sol.success:
                raise RuntimeError(sol.message)
            Y = sol.y[:, -1]
        return Y


def series_data(Fh, w, M, xi0, gravity=True):
    mp.mp.dps = 30
    m = Model(Fh, w, M, gravity=gravity)
    J = int(1 / (2 * abs(xi0) ** 2)) + 25
    d, s, g = m.series(J)
    D, S, C, jt, err = m.series_value(d, s, g, mp.mpc(xi0))
    Y = np.array([complex(v) for v in m.pack(D, S, C)])
    return Y, jt, float(err)


_fcache = {}


def fast_lin(s, r1, xia=0.09):
    """linear fast solution (Dd component) at real r1, Borel-summed: series at xia, integrated outward"""
    key = (complex(s), float(r1))
    if key in _fcache:
        return _fcache[key]
    a, b = (1 + s) / 2, (2 + s) / 2
    def ser(x, K=100):
        """value and derivative of x^{-2s} e^{-1/(2x^2)} sum_k c_k (-2x^2)^k, optimally truncated"""
        z = 2 * x * x
        cs, c = [], 1 + 0j
        for k in range(K):
            cs.append(c)
            c = c * (k + 1 - a) * (k + 1 - b) / (k + 1)
        kt = min(range(1, K), key=lambda k: abs(cs[k] * z ** k))
        P = sum(cs[k] * (-z) ** k for k in range(kt))
        dP = sum(cs[k] * k * (-z) ** (k - 1) * (-4 * x) for k in range(1, kt))
        pre = x ** (-2 * s) * np.exp(-1 / (2 * x * x))
        return pre * P, pre * (dP + P * (-2 * s / x + 1 / x ** 3))
    fa, fd = ser(xia)
    w = (s / 1j).real
    nm = NP(1, w, gravity=False)
    sc = abs(fa)                              # linear: integrate at unit scale (atol would swallow ~1e-27 values)
    Y0 = np.array([0, fa / sc, 0, (xia * xia * fd + s * xia * fa) / sc, 1], complex)
    Y = nm.path(Y0, [("real", xia, r1)])
    _fcache[key] = Y[1] * sc
    return Y[1] * sc


def Kp(s):
    a, b = (1 + s) / 2, (2 + s) / 2
    return complex(mp.power(2, -s - mp.mpf(1) / 2) / (mp.gamma(a) * mp.gamma(b)))


def Fharm(A, B3=0.0, C3=0.0):
    h = {1: A / 2j, -1: -A / 2j}
    if B3 or C3:
        h[3] = B3 / 2j + C3 / 2
        h[-3] = -B3 / 2j + C3 / 2
    return h


def jump(M, w, Fh, r0=None, r1=0.3, gravity=True):
    r0 = r0 or (0.1 if M <= 1 else 0.085)
    phi0 = 3 * np.pi / 8
    Y0, jt, err = series_data(Fh, w, M, r0 * np.exp(1j * phi0), gravity)
    nm = NP(M, w, gravity)
    Y = nm.path(Y0, [("ray", phi0, r0, r1), ("arc", r1, phi0, 0.0)])
    no = nm.no
    D, S = Y[:no], Y[no:2 * no]
    idx = {mu: k for k, mu in enumerate(nm.odd)}
    JD = {mu: D[idx[mu]] - np.conj(D[idx[-mu]]) for mu in nm.odd}
    JS = {mu: S[idx[mu]] - np.conj(S[idx[-mu]]) for mu in nm.odd}
    R = {}
    for mu in nm.odd:
        if mu > 0:
            s = 1j * mu * w
            ref = 2j * np.pi * Kp(s) * fast_lin(s, r1) * (2 * Fh[mu] if abs(Fh.get(mu, 0)) > 0 else 1)
            R[mu] = JD[mu] / ref
    return R, JD, JS, err


if __name__ == "__main__":
    if sys.argv[1] == "check":
        for r1 in (0.3, 0.25):
            R, JD, JS, err = jump(1, W0, Fharm(1.0), r1=r1, gravity=False)
            print(f"linear, r1 {r1}: R_1 = {R[1]:.12f}  (series truncation at r0: {err:.1e})")
    elif sys.argv[1] == "scan":
        M, wf = int(sys.argv[2]), float(sys.argv[3])
        A = float(sys.argv[4]); B3 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        C3 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
        for r1 in (0.3, 0.25):
            R, JD, JS, err = jump(M, W0 * wf, Fharm(A, B3, C3), r1=r1)
            print(f"M {M} w/w_inf {wf} A {A} B3 {B3} C3 {C3} r1 {r1}: " + "  ".join(f"R_{k} = {v:.6f}" for k, v in R.items()))
