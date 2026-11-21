"""Sideways marching on warped curves through the sonic valley of the low-D states (notes/LOW_D_SONIC.md).

hp marches its equations (hp/src/model.rs) in x on curves of constant x,

    x (x - c) p_x = -omega x p_theta + G p - c p + (n/2) c (p + m),   x (x + c) m_x = -omega x m_theta - G m + c m - (n/2) c (p + m),
    2 x a_x = a (p^2 + m^2) + (n - 1)(a - a^3),                       x c_x = G = (n - 1) c (a^2 - 1),

which is singular where x = c, where a curve of constant x is tangent to a characteristic of p. Here the same fields of
the same gauge are marched in zeta on the curves x = h(theta, zeta) = zeta + b(zeta) phi(theta), with b(zeta) =
sin^2(pi (zeta - x_A) / (x_B - x_A)) on the band x_A <= zeta <= x_B (so the band's ends are curves of constant x) and phi
periodic with period pi in theta (so the parities of p, m, a, c are kept). With p_x = p_zeta / h_zeta and p_theta|_x =
p_theta|_zeta - (h_theta / h_zeta) p_zeta:

    p_zeta = -h_zeta (R_p - omega x p_theta) / (x g_p),   g_p = c - x + omega h_theta,
    m_zeta =  h_zeta (R_m - omega x m_theta) / (x g_m),   g_m = c + x - omega h_theta,
    a_zeta = h_zeta [a (p^2 + m^2) + (n - 1)(a - a^3)] / (2 x),   c_zeta = h_zeta G / x,

with R_p, R_m the right-hand sides above. The marching is singular only where g_p or g_m vanishes, where the warped curve
is tangent to a characteristic, so a phi with omega b phi_theta > x - c wherever c - x is small keeps it regular through
a valley of c - x and through a patch where c < x. A curve of constant x is the case phi = 0. Changing the curves on
which one marches is the device of Gundlach's tau-dependent radial coordinate zeta = ln x - xi_0(tau) (PRD 55, 695 (1997)),
used there to place the self-similarity horizon on zeta = 0 (his Sec. II.3), and here to keep the curves off the
characteristics.

Fields are on the full grid of N = 2M points theta_j = 2 pi j / N (hp keeps the half grid, the first M points). The
integrator is the explicit Runge-Kutta method of order 8(5,3) of Dormand and Prince, DOP853 (Hairer, Norsett and
Wanner, Solving Ordinary Differential Equations I: Nonstiff Problems, Springer, Sec. II, the reference scipy names; its
edition not checked), its eighth-order weights only, with the coefficients from scipy and fixed steps from a plan.
--mutate nowarp drops omega h_theta from g_p and g_m (the warped curves treated as curves of constant x), the control
that must fail.
"""
import numpy as np
from scipy.integrate._ivp import dop853_coefficients as _dop

A8 = np.array(_dop.A[:12, :12], float)
B8 = np.array(_dop.B, float)
C8 = np.array(_dop.C[:12], float)


def dtheta(f):
    """d/dtheta along the last axis (full grid of N points over [0, 2 pi)); the Nyquist harmonic dropped (odd fields
    have none)."""
    N = f.shape[-1]
    F = np.fft.rfft(f, axis=-1)
    k = np.arange(F.shape[-1])
    F = F * (1j * k)
    if N % 2 == 0:
        F[..., -1] = 0.0
    return np.fft.irfft(F, n=N, axis=-1)


def integrate_theta(g):
    """Zero-mean periodic antiderivative of g along the last axis."""
    N = g.shape[-1]
    G = np.fft.rfft(g, axis=-1)
    k = np.arange(G.shape[-1]).astype(float)
    out = np.zeros_like(G)
    out[..., 1:] = G[..., 1:] / (1j * k[1:])
    if N % 2 == 0:
        out[..., -1] = 0.0
    return np.fft.irfft(out, n=N, axis=-1)


def lowpass(f, kmax):
    F = np.fft.rfft(f, axis=-1)
    F[..., kmax + 1:] = 0.0
    return np.fft.irfft(F, n=f.shape[-1], axis=-1)


class Warp:
    """x = zeta + b(zeta) phi(theta) on [xa, xb]; phi on the full grid (period pi in theta)."""

    def __init__(self, xa, xb, phi):
        self.xa, self.xb = float(xa), float(xb)
        self.phi = np.asarray(phi, float)
        self.phi_t = dtheta(self.phi)
        self.L = self.xb - self.xa

    def b(self, z):
        return np.sin(np.pi * (z - self.xa) / self.L) ** 2

    def bz(self, z):
        return (np.pi / self.L) * np.sin(2 * np.pi * (z - self.xa) / self.L)

    def at(self, z):
        """(x, h_zeta, h_theta) on the curve zeta = z"""
        b, bz = self.b(z), self.bz(z)
        return z + b * self.phi, 1.0 + bz * self.phi, b * self.phi_t

    @staticmethod
    def none(xa, xb, N):
        return Warp(xa, xb, np.zeros(N))


class Band:
    """The warped equations at D (n = D - 2) on the full theta grid of N points."""

    def __init__(self, D, N, warp, mutate="none"):
        self.n = float(D) - 2.0
        self.N = int(N)
        self.warp = warp
        self.mutate = mutate
        self.nevals = 0

    def gaps(self, z, s, omega):
        x, hz, ht = self.warp.at(z)
        c = s[..., 3, :]
        return c - x + omega * ht, c + x - omega * ht, x, hz, ht

    def rhs(self, z, s, omega):
        """s: (..., 4, N) = (p, m, a, c); omega scalar"""
        self.nevals += 1
        n = self.n
        p, m, a, c = s[..., 0, :], s[..., 1, :], s[..., 2, :], s[..., 3, :]
        x, hz, ht = self.warp.at(z)
        if self.mutate == "nowarp":
            gp, gm = c - x, c + x
        else:
            gp, gm = c - x + omega * ht, c + x - omega * ht
        a2 = a * a
        G = (n - 1) * c * (a2 - 1)
        pm = 0.5 * n * c * (p + m)
        Rp = G * p - c * p + pm
        Rm = -G * m + c * m - pm
        pt, mt = dtheta(p), dtheta(m)
        out = np.empty_like(s)
        out[..., 0, :] = -hz * (Rp - omega * x * pt) / (x * gp)
        out[..., 1, :] = hz * (Rm - omega * x * mt) / (x * gm)
        out[..., 2, :] = hz * (a * (p * p + m * m) + (n - 1) * (a - a * a2)) / (2 * x)
        out[..., 3, :] = hz * G / x
        return out

    def step(self, z, s, h, omega):
        k = []
        for i in range(12):
            si = s
            for j in range(i):
                if A8[i, j] != 0.0:
                    si = si + (h * A8[i, j]) * k[j]
            k.append(self.rhs(z + C8[i] * h, si, omega))
        out = s
        for i in range(12):
            if B8[i] != 0.0:
                out = out + (h * B8[i]) * k[i]
        return out

    def rate(self, z, s, omega):
        """the largest transport rate (M - 1) omega h_zeta / min(g_p, g_m) over theta, and min g_p, min g_m"""
        gp, gm, x, hz, ht = self.gaps(z, s, omega)
        M = self.N // 2
        g = np.minimum(gp, gm)
        return float((M - 1) * omega * np.max(hz / np.maximum(g, 1e-300))), float(gp.min()), float(gm.min())

    def plan(self, s0, omega, kappa=1.0, record=None, hmax=None, max_steps=2_000_000):
        """Step points from xa to xb, each step h = kappa / rate at its start (the rate from the state being
        integrated); returns (points, end state, min g_p, min g_m). record: list to append (zeta, state) to."""
        z, s = self.warp.xa, np.array(s0, float)
        pts = [z]
        gpmin = gmmin = np.inf
        hmax = hmax or (self.warp.xb - self.warp.xa) / 8
        while z < self.warp.xb:
            lam, gp, gm = self.rate(z, s, omega)
            gpmin, gmmin = min(gpmin, gp), min(gmmin, gm)
            if gp <= 0 or gm <= 0 or not np.isfinite(lam):
                raise FloatingPointError(f"characteristic tangency at zeta {z:.6g}: min g_p {gp:.3e}, min g_m {gm:.3e}")
            h = min(kappa / lam, hmax)
            last = self.warp.xb - z - h < 0.5 * h
            if last:
                h = self.warp.xb - z
            s = self.step(z, s, h, omega)
            z = self.warp.xb if last else z + h
            pts.append(z)
            if record is not None:
                record.append((z, s.copy()))
            if not np.all(np.isfinite(s)):
                raise FloatingPointError(f"non-finite state at zeta {z:.6g}")
            if len(pts) > max_steps:
                raise RuntimeError("too many steps")
        return pts, s, gpmin, gmmin

    def run(self, s0, omega, pts, record=None):
        """Integrate along the given step points; returns (end state, min g_p, min g_m over the step starts)."""
        s = np.array(s0, float)
        gpmin = gmmin = np.inf
        for i in range(len(pts) - 1):
            gp, gm, *_ = self.gaps(pts[i], s, omega)
            gpmin, gmmin = min(gpmin, float(gp.min())), min(gmmin, float(gm.min()))
            s = self.step(pts[i], s, pts[i + 1] - pts[i], omega)
            if record is not None:
                record.append((pts[i + 1], s.copy()))
        return s, gpmin, gmmin


def design_phi(profile, omega, eps0, kmax=None, bmin=0.85, width=None):
    """phi for a band from an unwarped profile [(x, state)] (state (4, N)): omega phi_theta = (S(eps0 - g*) -
    mean) / bmin, g*(theta) = min over the profile of c - x, S a softplus of width eps0/4, low-passed to kmax."""
    g = np.min(np.array([s[3] - x for x, s in profile]), axis=0)
    w = width or eps0 / 4
    u = (eps0 - g) / w
    S = w * np.where(u > 30, u, np.log1p(np.exp(np.minimum(u, 30))))
    pt = (S - S.mean()) / (omega * bmin)
    if kmax:
        pt = lowpass(pt, kmax)
    return integrate_theta(pt), g


def full_from_half(f, odd):
    return np.concatenate([f, -f if odd else f], axis=-1)


def node_full(zv, M, i):
    """full-grid (4, 2M) state of hp node i from a solution vector (half grid: p, m odd; a, c even)"""
    b = np.asarray(zv[2 * M + 4 * M * i:2 * M + 4 * M * (i + 1)], float).reshape(4, M)
    return np.stack([full_from_half(b[0], True), full_from_half(b[1], True), full_from_half(b[2], False),
                     full_from_half(b[3], False)])


def half_rows(s):
    """(4, 2M) full-grid state -> 4M half-grid rows (p, m, a, c)"""
    M = s.shape[-1] // 2
    return np.concatenate([s[0, :M], s[1, :M], s[2, :M], s[3, :M]])
