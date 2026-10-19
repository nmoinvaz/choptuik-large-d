"""Horizon regularity as a monodromy condition, for large D.

Why: the period is selected by the horizon-singular branch u^sigma (u = 1 - x, sigma ~ n/2). Matching at u_K = 0.3
sees it only through u_K^sigma, exponentially small in n; that is what limits the multiple-shooting solver above
D ~ 35. Here the centre-regular solution is integrated outward along the real axis to x = 1 - rho and then continued
around the horizon x = 1 on two semicircles of radius rho, above and below, to x = 1 + rho (and on to x_end). A part
analytic at x = 1 comes back the same on both paths; the singular part R u^sigma picks up exp(2 pi i sigma) between
them. So the jump between the two continuations measures R, at a distance |u| = rho ~ 1 from the horizon instead of
0.3, i.e. ~ (rho/0.3)^(n/2) better conditioned.

Fields: p, m (odd in tau -> tau + Delta/2), a, c (even), on N tau-points, complex for complex x; the same equations as
dss.DSSMulti.rhs, with spectral truncation to |k| <= kc.

    from monodromy import Mono
    M = Mono(D, kc=23)
    J, c1 = M.jump(p1, c0, Delta)       # jump of (p, m, a, c) at x_end between the two paths; c(tau, 1) by Cauchy mean
"""
import numpy as np

from spec_long import make_long


class Mono:
    def __init__(self, D, kc=23, rho=0.5, x_end=None, n_real=400, n_arc=600, n_tail=200, x_join=None):
        self.D, self.n = D, D - 2
        self.S = make_long(kc, D - 2, nodes=[0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6], Ks=60)
        self.N, self.kc = self.S.N, kc
        k = np.fft.fftfreq(self.N, 1.0 / self.N)
        self.k = k
        self.mask = (np.abs(k) <= kc).astype(float)
        self.rho = rho
        self.x_join = 1 - rho if x_join is None else x_join
        self.x_end = 1 + rho if x_end is None else x_end
        self.n_real, self.n_arc, self.n_tail = n_real, n_arc, n_tail

    # spectral helpers (complex-safe)
    def tr(self, f):
        return np.fft.ifft(np.fft.fft(f, axis=-1) * self.mask, axis=-1)

    def Dt(self, f, om):
        return np.fft.ifft(np.fft.fft(f, axis=-1) * (1j * self.k * self.mask), axis=-1) * om

    def rhs(self, x, s, om):
        p, m, a, c = s
        n = self.n
        a2 = a * a
        cx = c * (n - 1) * (a2 - 1) / x
        ax = a * ((p * p + m * m) + (n - 1) * (1 - a2)) / (2 * x)
        pc = n * c * (p + m) / (2 * x)
        px = (-self.Dt(p, om) + cx * p - c * p / x + pc) / (x - c)
        mx = (-self.Dt(m, om) - cx * m + c * m / x - pc) / (x + c)
        return self.tr(np.stack([px, mx, ax, cx]))

    def rk4_path(self, s, xs, om, record=None):
        """RK4 along the polyline xs (complex), dx = x_{i+1} - x_i."""
        for i in range(len(xs) - 1):
            x, h = xs[i], xs[i + 1] - xs[i]
            k1 = self.rhs(x, s, om)
            k2 = self.rhs(x + h / 2, s + h / 2 * k1, om)
            k3 = self.rhs(x + h / 2, s + h / 2 * k2, om)
            k4 = self.rhs(x + h, s + h * k3, om)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if record is not None:
                record.append(s)
        return s

    def centre_state(self, p1, c0, om):
        S = self.S
        P, M, A, C = S.centre_series(p1, c0, np.array([om]))
        s = np.stack([S.eval_series(Q, S.x0) for Q in (P, M, A, C)])[:, 0, :]
        return s.astype(complex), S.x0

    def jump(self, p1, c0, Delta):
        """p1, c0: centre data on the tau grid (real, N points). Returns (J, c1): J = state(upper) - state(lower) at x_end,
        shape (4, N); c1 = c(tau, 1) from the mean over the circle |x - 1| = rho."""
        om = 2 * np.pi / Delta
        s, x0 = self.centre_state(np.atleast_2d(p1), np.atleast_2d(c0), om)
        # centre -> x_join on the real axis, log-spaced near the centre then uniform
        xl = np.r_[np.geomspace(x0, 0.05, self.n_real // 2, endpoint=False), np.linspace(0.05, self.x_join, self.n_real // 2)]
        s = self.rk4_path(s, xl.astype(complex), om)
        th = np.linspace(0, np.pi, self.n_arc + 1)
        up = 1 - self.rho * np.exp(-1j * th)            # through the upper half-plane
        lo = 1 - self.rho * np.exp(1j * th)             # through the lower half-plane
        tail = np.linspace(1 + self.rho, self.x_end, self.n_tail + 1).astype(complex)
        rec_u, rec_l = [s], [s]
        su = self.rk4_path(s, up, om, rec_u)
        sl = self.rk4_path(s, lo, om, rec_l)
        if self.x_end > 1 + self.rho + 1e-12:
            su = self.rk4_path(su, tail, om)
            sl = self.rk4_path(sl, tail, om)
        # c(tau, 1): mean value over the circle (upper and lower arcs together, trapezoid in theta)
        cu = np.array([r[3] for r in rec_u]); cl = np.array([r[3] for r in rec_l])
        w = np.full(len(th), 1.0); w[0] = w[-1] = 0.5
        c1 = ((w[:, None] * cu).sum(0) + (w[:, None] * cl).sum(0)) / (2 * w.sum())
        return su - sl, c1


def centre_data(D, z, nodes, kc=23):
    """Centre data (p1, c0) on the tau grid and Delta from a converged solution vector z."""
    S = make_long(kc, D - 2, nodes=nodes, Ks=60)
    parts, _ = S.split(z[None, :])
    p1 = S.phys(parts[0][0][None, :], "odd")[0]
    c0 = S.phys(parts[1][0][None, :], "even")[0]
    return p1, c0, float(z[-1])
