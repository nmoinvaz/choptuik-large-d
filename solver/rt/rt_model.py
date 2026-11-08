"""A small prototype of the Reiterer-Trubowitz (RT) formulation of the DSS critical solution in D dimensions.

Equations: EEG (arXiv:2611.07007v1) Eq. (5), checked against the covariant equations by rt_eqs_check.py (they hold for
Ric = (D - 2) dphi dphi). Unknowns, as in RT (arXiv:1203.3766, Sec. 2.2): mu and four functions on (t, xi) in
[0, 2 pi) x [-1, 1], omega_i = omega_i^- (i = 2, 3, 4) and omega_1, with omega_i^+ = -P omega_i, P xi = -xi; omega_1 is odd
in xi; omega_1..3 are 2 pi periodic in t and omega_4 antiperiodic. Equations: (1 + P) Omega_1^+, Omega_2^+, Omega_3^+,
Omega_4^+, each divided by xi (RT's S), and the phase condition Re (omega_4)_{1,0} = 0.

Discretization (unlike RT's coefficient space, a collocation): N_t points in t on [0, 2 pi) with exact Fourier
derivatives (integer wavenumbers for the periodic fields, half-integer for omega_4); Chebyshev-Gauss-Lobatto points
xi_j = cos(pi j / N), N odd, so xi = 0 is not a point and xi = +-1 are. No boundary condition is imposed anywhere: at
xi = -1 (the self-similarity horizon of the + equations) D^+ has no xi component and the collocated equation is the
regularity condition. The residual is exactly quadratic in (mu, omega), so the Jacobian columns are exact central
differences with unit step (RT Sec. 1.2: C(x) = A x + B(x, x)).

Delta = 4 pi mu - 2 int_0^{2 pi} omega_3^-(t, 0) dt (RT Eq. (15b), Delta = 2K), and the same with xi = 1.

Horizon rows (issue #163, `horizon=K`, default 0 = the collocation above). At xi = -1 the equations Omega_{2,3,4}^+ / xi
are regular-singular in s = 1 + xi: D^+ = d_t + mu s d_s, and their linearization there is lower triangular in
(omega_2, omega_4, omega_3) with exponents lambda_2 = -<a_2>/mu, lambda_4 = -<a_4>/mu and lambda_3 = -1, each shifted by
-i k / mu for harmonic k (rt_horizon.py, sympy). A branch s^lambda with Re lambda > 0 is bounded and is excluded only
by analyticity, which degree-N polynomials see to N^(-2 Re lambda - 1). With K > 0 the collocated rows of Omega_{2,3,4}^+
at the K points next to xi = -1 (j = N - 1 .. N - K) are replaced by the Taylor coefficients of orders 1..K of the same
residuals at xi = -1 (order 0 is the row at xi = -1 itself), computed from the fields' Taylor coefficients there by
truncated series products. An analytic solution satisfies them; a branch with Re lambda < K violates the order
ceil(Re lambda) row. The residual stays exactly quadratic. Row n is scaled so that its largest entry is N^2.
"""
import numpy as np


def cheb(N):
    """Chebyshev-Gauss-Lobatto points cos(pi j / N) and the differentiation matrix (Trefethen, Spectral Methods in
    MATLAB, 2000, ch. 6)."""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.r_[2.0, np.ones(N - 1), 2.0] * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    Dm = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    Dm = Dm - np.diag(Dm.sum(axis=1))
    return x, Dm


def bary_weights(N):
    w = (-1.0) ** np.arange(N + 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def cheb_interp_row(xg, x0):
    """Row vector r with r . f = the Chebyshev interpolant of f (on the Lobatto points xg) at x0."""
    N = len(xg) - 1
    w = bary_weights(N)
    d = x0 - xg
    if np.any(d == 0):
        r = (d == 0).astype(float)
        return r
    r = w / d
    return r / r.sum()


def smul(a, b):
    """Cauchy product of two truncated Taylor series stored along the last axis (float or object arrays)."""
    K1 = a.shape[-1]
    return np.stack([sum(a[..., j] * b[..., n - j] for j in range(n + 1)) for n in range(K1)], axis=-1)


def xmul(a):
    """xi a = (s - 1) a for a series in s = 1 + xi."""
    out = -a
    out[..., 1:] = out[..., 1:] + a[..., :-1]
    return out


def horizon_series(m2, m3, m4, p2, p4, a1, mu, D, cm, Dp, tm=1.0):
    """Taylor coefficients in s = 1 + xi at xi = -1 of Omega_{2,3,4}^+ / xi, from those of the fields (last axis):
    m = omega^- (the unknowns), p = omega^+ = -P omega^-, a1 = omega_1. The formulas are rt_model.RT.omegas with
    products as series products; Dp(S, anti) is D^+ on a series (d_t + mu n on order n). tm scales the coefficient
    2 (D - 1) of Omega_2 (the planted check's control)."""
    X = xmul
    O2 = (X(Dp(m2) + mu * m2) + mu * ((D - 3) * a1 + p2 + m2)
          + X((D - 3) * (smul(a1, a1) + smul(p2, p2) + smul(m2, m2) - 2 * smul(a1, p2 + m2))
              - 2 * (D - 1) * tm * smul(p2, m2)) / 4)
    O3 = (X(Dp(m3) + mu * m3) - mu / 2 * (D - 3) * (D - 2) * cm * a1
          + X((D - 2) * ((D - 3) * (smul(p2 + m2, a1) - smul(a1, a1) / 2 - smul(p2 - m2, p2 - m2) / 2)
                         - 2 * smul(p4, m4))) / 4)
    O4 = (X(Dp(m4, True) + mu * m4) + mu / 2 * (D - 2) * (p4 + m4)
          - X((D - 2) * (smul(p4, m2) + smul(m4, p2))) / 2)
    # 1 / xi = -(1 + s + s^2 + ...)
    return [-np.cumsum(O, axis=-1) for O in (O2, O3, O4)]


class RT:
    def __init__(self, D, Nt, N, coef_mut=1.0, horizon=0, taylor_mut=1.0):
        assert Nt % 2 == 0 and N % 2 == 1
        self.D, self.Nt, self.N = float(D), Nt, N
        self.coef_mut = coef_mut           # the --mutate=coef control: one coefficient of Omega_3 scaled
        self.t = 2 * np.pi * np.arange(Nt) / Nt
        self.xi, self.Dx = cheb(N)
        self.rev = np.arange(N + 1)[::-1]
        self.pos = np.arange((N + 1) // 2)              # xi > 0
        k = np.fft.fftfreq(Nt, 1.0 / Nt)
        kp = 1j * k
        kp[Nt // 2] = 0.0                                # Nyquist of the periodic fields
        self.kper = kp
        self.kanti = 1j * (k + 0.5)                      # antiperiodic: e^{i (k + 1/2) t}
        self.shift = np.exp(-0.5j * self.t)
        self.n1 = self.Nt * len(self.pos)
        self.nf = self.Nt * (N + 1)
        self.L = self.n1 + 3 * self.nf + 1
        cw = np.full(N + 1, np.pi / N)
        cw[0] = cw[-1] = np.pi / (2 * N)
        self.phase_w = np.outer(np.cos(self.t / 2), cw)  # Re (omega_4)_{1,0} up to a constant factor
        self.r0 = cheb_interp_row(self.xi, 0.0)
        self.K = int(horizon)
        self.taylor_mut = taylor_mut
        if self.K:
            self._horizon_setup()

    def _horizon_setup(self):
        """TR[n] . f = f^(n)(-1) / n! for Lobatto values f (row N of Dx^n over n!), and the row scales. The rows are
        computed at 200 bits (mpmath) and rounded: repeated f64 products with Dx lose up to 1e-2 at order 7."""
        import mpmath
        K, N = self.K, self.N
        assert 1 <= K < N // 2, "horizon rows: 1 <= K < N / 2"
        with mpmath.workprec(200):
            x = [mpmath.cos(mpmath.pi * j / N) for j in range(N + 1)]
            c = [(2 if j in (0, N) else 1) * (-1) ** j for j in range(N + 1)]
            Dm = [[mpmath.mpf(c[i]) / c[j] / (x[i] - x[j]) if i != j else 0 for j in range(N + 1)] for i in range(N + 1)]
            for i in range(N + 1):
                Dm[i][i] = -mpmath.fsum(Dm[i][j] for j in range(N + 1) if j != i)
            r = [mpmath.mpf(0)] * N + [mpmath.mpf(1)]
            rows = [[float(v) for v in r]]
            for n in range(1, K + 1):
                r = [mpmath.fsum(r[i] * Dm[i][j] for i in range(N + 1)) / n for j in range(N + 1)]
                rows.append([float(v) for v in r])
        rows = [np.array(v) for v in rows]
        self.TR = np.array(rows)
        # the largest entry of the scaled row n is N^2, as for a collocated derivative near the ends
        self.tscale = np.array([1.0] + [float(N) ** 2 / np.abs(rows[n]).max() for n in range(1, K + 1)])
        self.hrows = N - np.arange(1, K + 1)             # collocation points replaced by orders 1..K

    def horizon_rows(self, z):
        """The Taylor coefficients R_n (n = 0..K) at xi = -1 of Omega_{2,3,4}^+ / xi, each (B, Nt, K + 1)."""
        w1, w2, w3, w4, mu = self.unpack(z)
        T = lambda f: f @ self.TR.T
        nn = np.arange(self.K + 1)[None, None, :]
        Dp = lambda S, anti=False: self.dt(S, anti) + mu * nn * S
        return horizon_series(T(w2), T(w3), T(w4), T(-self.P(w2)), T(-self.P(w4)), T(w1), mu, self.D,
                              self.coef_mut, Dp, self.taylor_mut)

    # ---------------------------------------------------------------- layout
    def unpack(self, z):
        """z (B, L) -> w1, w2, w3, w4 (B, Nt, N+1), mu (B, 1, 1)."""
        z = np.atleast_2d(z)
        B, Nt, N1, h = z.shape[0], self.Nt, self.N + 1, len(self.pos)
        o = 0
        half = z[:, o:o + self.n1].reshape(B, Nt, h)
        o += self.n1
        w1 = np.empty((B, Nt, N1))
        w1[:, :, self.pos] = half
        w1[:, :, self.rev[self.pos]] = -half
        w2 = z[:, o:o + self.nf].reshape(B, Nt, N1); o += self.nf
        w3 = z[:, o:o + self.nf].reshape(B, Nt, N1); o += self.nf
        w4 = z[:, o:o + self.nf].reshape(B, Nt, N1); o += self.nf
        mu = z[:, o][:, None, None]
        return w1, w2, w3, w4, mu

    def pack(self, w1, w2, w3, w4, mu):
        return np.r_[w1[:, self.pos].ravel(), w2.ravel(), w3.ravel(), w4.ravel(), mu]

    # ---------------------------------------------------------------- derivatives
    def dt(self, f, anti=False):
        if anti:
            g = f * self.shift[None, :, None]
            G = np.fft.fft(g, axis=1) * self.kanti[None, :, None]
            return (np.fft.ifft(G, axis=1) / self.shift[None, :, None]).real
        F = np.fft.fft(f, axis=1) * self.kper[None, :, None]
        return np.fft.ifft(F, axis=1).real

    def dxi(self, f):
        return f @ self.Dx.T

    def Dp(self, f, mu, anti=False):
        """D^+ f = d_t f + mu (1 + xi) d_xi f."""
        return self.dt(f, anti) + mu * (1 + self.xi)[None, None, :] * self.dxi(f)

    def P(self, f):
        return f[:, :, self.rev]

    # ---------------------------------------------------------------- residual
    def omegas(self, z):
        w1, w2, w3, w4, mu = self.unpack(z)
        D, X = self.D, self.xi[None, None, :]
        m2, m3, m4 = w2, w3, w4
        p2, p3, p4 = -self.P(w2), -self.P(w3), -self.P(w4)
        O1 = (X * (self.Dp(w1, mu) + mu * w1) + mu * (D - 2) * w1
              + X / 4 * ((D - 3) * (w1 ** 2 + p2 ** 2 + m2 ** 2 - 2 * m2 * (p2 + w1)) - 2 * (D - 1) * w1 * p2
                         + 4 * p3 * (w1 + p2 - m2) - 4 * p4 ** 2))
        O2 = (X * (self.Dp(m2, mu) + mu * m2) + mu * ((D - 3) * w1 + p2 + m2)
              + X / 4 * ((D - 3) * (w1 ** 2 + p2 ** 2 + m2 ** 2 - 2 * w1 * (p2 + m2)) - 2 * (D - 1) * p2 * m2))
        O3 = (X * (self.Dp(m3, mu) + mu * m3) - mu / 2 * (D - 3) * (D - 2) * self.coef_mut * w1
              + X / 4 * (D - 2) * ((D - 3) * ((p2 + m2) * w1 - 0.5 * w1 ** 2 - 0.5 * (p2 - m2) ** 2) - 2 * p4 * m4))
        O4 = (X * (self.Dp(m4, mu, anti=True) + mu * m4) + mu / 2 * (D - 2) * (p4 + m4)
              - X / 2 * (D - 2) * (p4 * m2 + m4 * p2))
        return O1, O2, O3, O4, w4

    def residual(self, z, forcing=None):
        """F(z) (B, L): (1 + P) Omega_1^+ / xi on xi > 0, Omega_{2,3,4}^+ / xi, phase row."""
        O1, O2, O3, O4, w4 = self.omegas(z)
        X = self.xi[None, None, :]
        E1 = ((O1 + self.P(O1)) / X)[:, :, self.pos]
        B = E1.shape[0]
        ph = np.einsum("bij,ij->b", w4, self.phase_w)
        E2, E3, E4 = O2 / X, O3 / X, O4 / X
        if self.K:
            for E, R in zip((E2, E3, E4), self.horizon_rows(z)):
                for n, j in enumerate(self.hrows, start=1):
                    E[:, :, j] = self.tscale[n] * R[:, :, n]
        F = np.concatenate([E1.reshape(B, -1), E2.reshape(B, -1), E3.reshape(B, -1),
                            E4.reshape(B, -1), ph[:, None]], axis=1)
        if forcing is not None:
            F = F - forcing[None, :]
        return F

    def jacobian(self, z, batch=256):
        """Exact for a quadratic residual: J e = (F(z + e) - F(z - e)) / 2."""
        L = self.L
        J = np.empty((L, L))
        for s in range(0, L, batch):
            idx = np.arange(s, min(L, s + batch))
            E = np.zeros((len(idx), L))
            E[np.arange(len(idx)), idx] = 1.0
            J[:, idx] = ((self.residual(z[None, :] + E) - self.residual(z[None, :] - E)) / 2).T
        return J

    def newton(self, z, tol=1e-12, it_max=20, forcing=None, verbose=True, log=None):
        hist = []
        for it in range(it_max):
            F = self.residual(z, forcing)[0]
            r = np.abs(F).max()
            hist.append(r)
            if verbose:
                msg = f"    newton it {it}: max|F| {r:.3e}  Delta {self.delta(z)[0]:.15f}"
                print(msg, flush=True)
                if log:
                    log.write(msg + "\n"); log.flush()
            if r < tol:
                return z, True, hist
            if it > 3 and r > 0.5 * hist[-2] and r < 1e3 * tol:
                return z, True, hist            # at the rounding floor
            J = self.jacobian(z)
            dz = np.linalg.solve(J, -F)
            lam = 1.0
            for _ in range(8):
                zn = z + lam * dz
                if np.abs(self.residual(zn, forcing)[0]).max() < r or lam < 0.02:
                    break
                lam /= 2
            z = zn
        F = self.residual(z, forcing)[0]
        hist.append(np.abs(F).max())
        return z, hist[-1] < tol, hist

    # ---------------------------------------------------------------- observables
    def delta(self, z):
        """(Delta from omega_3 at xi = 0, Delta from omega_3 at xi = 1)."""
        w1, w2, w3, w4, mu = self.unpack(z)
        mu = mu[0, 0, 0]
        w30 = w3[0] @ self.r0
        d0 = 4 * np.pi * mu - 2 * (2 * np.pi / self.Nt) * w30.sum()
        d1 = 4 * np.pi * mu - 2 * (2 * np.pi / self.Nt) * w3[0, :, 0].sum()
        return d0, d1

    def mu(self, z):
        return z[-1]

    # ---------------------------------------------------------------- resampling
    def resample_to(self, z, other):
        """Spectral interpolation of a state to another RT grid (Fourier in t, Chebyshev in xi)."""
        w1, w2, w3, w4, mu = self.unpack(z)
        out = []
        for f, anti in ((w1, False), (w2, False), (w3, False), (w4, True)):
            out.append(resample_field(f[0], self, other, anti))
        return other.pack(*out, mu[0, 0, 0])


def resample_field(f, a, b, anti):
    """f on grid a (Nt_a, N_a + 1) -> grid b: Chebyshev interpolation in xi, Fourier interpolation in t."""
    Ix = np.array([cheb_interp_row(a.xi, x0) for x0 in b.xi])
    g = (f @ Ix.T).astype(complex)
    if anti:
        g = g * np.exp(-0.5j * a.t)[:, None]
    G = np.fft.fft(g, axis=0) / a.Nt
    k = np.fft.fftfreq(a.Nt, 1.0 / a.Nt)
    if not anti:                                       # split the Nyquist mode of a periodic field
        ny = a.Nt // 2
        G = np.vstack([G, 0.5 * G[ny:ny + 1]])
        G[ny] *= 0.5
        k = np.r_[k, -k[ny]]
    h = np.exp(1j * np.outer(b.t, k)) @ G
    if anti:
        h = h * np.exp(0.5j * b.t)[:, None]
    return h.real
