"""The RT residual of rt_model.py in arbitrary precision (python-flint arb in numpy object arrays), and Newton with that
residual and the f64 Jacobian (iterative refinement: it converges while kappa(J) times 2^-53 stays below 1, and the
solution then carries the residual's precision, not f64's).

    from rt_mp import RTmp, newton_mixed
    rm = RTmp(D, Nt, N, prec=160); z = newton_mixed(rm, z_f64)       # z: object array of arb

The derivative matrices are those of rt_model.RT built in arb: the Chebyshev matrix from xi_j = cos(pi j / N), and the
Fourier ones as dense matrices (exactly rt_model's FFT derivatives: Nyquist mode zeroed for the periodic fields,
wavenumbers k + 1/2 for the antiperiodic one).
"""
import os
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
import numpy as np
import flint
import scipy.linalg as sl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_model import RT, horizon_series

arb = flint.arb


def obj(a):
    return np.vectorize(lambda v: arb(float(v)), otypes=[object])(np.asarray(a, dtype=float))


class RTmp(RT):
    def __init__(self, D, Nt, N, prec=160, coef_mut=1.0, horizon=0, taylor_mut=1.0):
        super().__init__(D, Nt, N, coef_mut=coef_mut, horizon=horizon, taylor_mut=taylor_mut)
        flint.ctx.prec = prec
        self.prec = prec
        pi = arb.pi()
        self.Dm = arb(repr(float(D))) if False else arb(D)
        self.cm = arb(coef_mut) if coef_mut == 1.0 else arb(coef_mut)
        # Chebyshev points and differentiation matrix in arb
        x = [(pi * j / N).cos() for j in range(N + 1)]
        c = [arb(2 if j in (0, N) else 1) * (-1) ** j for j in range(N + 1)]
        Dx = np.empty((N + 1, N + 1), dtype=object)
        for i in range(N + 1):
            s = arb(0)
            for j in range(N + 1):
                if i != j:
                    Dx[i, j] = c[i] / c[j] / (x[i] - x[j])
                    s += Dx[i, j]
            Dx[i, i] = -s
        self.xm = np.array(x, dtype=object)
        self.DxT = Dx.T.copy()
        # Fourier derivative matrices on t_a = 2 pi a / Nt
        t = [2 * pi * a / Nt for a in range(Nt)]
        self.tm = t
        kk = np.fft.fftfreq(Nt, 1.0 / Nt)
        Dper = np.empty((Nt, Nt), dtype=object)
        Dant = np.empty((Nt, Nt), dtype=object)
        for a in range(Nt):
            for b in range(Nt):
                sp, sa = arb(0), arb(0)
                for k in kk:
                    kr = int(k)
                    th = t[a] - t[b]
                    if kr != -Nt // 2:            # the Nyquist mode of the periodic fields is zeroed
                        sp += -kr * (kr * th).sin()
                    ka = arb(kr) + arb(1) / 2
                    sa += -ka * (ka * th).sin()
                Dper[a, b] = sp / Nt
                Dant[a, b] = sa / Nt
        self.Dper, self.Dant = Dper, Dant
        cw = [pi / N] * (N + 1)
        cw[0] = cw[-1] = pi / (2 * N)
        self.phase_wm = np.array([[(t[a] / 2).cos() * cw[j] for j in range(N + 1)] for a in range(Nt)], dtype=object)
        # interpolation row to xi = 0 (barycentric)
        w = [arb((-1) ** j) * (arb(1) / 2 if j in (0, N) else 1) for j in range(N + 1)]
        r = [w[j] / (0 - x[j]) for j in range(N + 1)]
        sr = sum(r, arb(0))
        self.r0m = np.array([v / sr for v in r], dtype=object)
        if self.K:
            # issue #163: the Taylor rows at xi = -1 in arb (row N of Dx^n over n!), the f64 path's scales exactly
            r = np.array([arb(0)] * N + [arb(1)], dtype=object)
            rows = [r]
            for n in range(1, self.K + 1):
                rows.append(self.DxT.dot(rows[-1]) / n)
            self.TRmT = np.array(rows, dtype=object).T.copy()
            self.tscale_m = [arb(float(v)) for v in self.tscale]
            self.tm_m = arb(taylor_mut)

    # ---------------------------------------------------------------- object-array versions
    def unpack_m(self, z):
        Nt, N1, h = self.Nt, self.N + 1, len(self.pos)
        o = 0
        half = z[o:o + self.n1].reshape(Nt, h)
        o += self.n1
        w1 = np.empty((Nt, N1), dtype=object)
        w1[:, self.pos] = half
        w1[:, self.rev[self.pos]] = -half
        w2 = z[o:o + self.nf].reshape(Nt, N1); o += self.nf
        w3 = z[o:o + self.nf].reshape(Nt, N1); o += self.nf
        w4 = z[o:o + self.nf].reshape(Nt, N1); o += self.nf
        return w1, w2, w3, w4, z[o]

    def residual_m(self, z):
        w1, w2, w3, w4, mu = self.unpack_m(z)
        D, X = self.Dm, self.xm[None, :]
        P = lambda f: f[:, self.rev]
        one_p_X = np.array([1 + v for v in self.xm], dtype=object)[None, :]
        Dp = lambda f, anti=False: (self.Dant if anti else self.Dper).dot(f) + mu * one_p_X * f.dot(self.DxT)
        m2, m3, m4 = w2, w3, w4
        p2, p3, p4 = -P(w2), -P(w3), -P(w4)
        q = arb(1) / 4
        h = arb(1) / 2
        O1 = (X * (Dp(w1) + mu * w1) + mu * (D - 2) * w1
              + X * q * ((D - 3) * (w1 * w1 + p2 * p2 + m2 * m2 - 2 * m2 * (p2 + w1)) - 2 * (D - 1) * w1 * p2
                         + 4 * p3 * (w1 + p2 - m2) - 4 * p4 * p4))
        O2 = (X * (Dp(m2) + mu * m2) + mu * ((D - 3) * w1 + p2 + m2)
              + X * q * ((D - 3) * (w1 * w1 + p2 * p2 + m2 * m2 - 2 * w1 * (p2 + m2)) - 2 * (D - 1) * p2 * m2))
        O3 = (X * (Dp(m3) + mu * m3) - mu * h * (D - 3) * (D - 2) * self.cm * w1
              + X * q * (D - 2) * ((D - 3) * ((p2 + m2) * w1 - h * w1 * w1 - h * (p2 - m2) * (p2 - m2))
                                   - 2 * p4 * m4))
        O4 = (X * (Dp(m4, True) + mu * m4) + mu * h * (D - 2) * (p4 + m4)
              - X * h * (D - 2) * (p4 * m2 + m4 * p2))
        E1 = ((O1 + P(O1)) / X)[:, self.pos]
        ph = (w4 * self.phase_wm).sum()
        E2, E3, E4 = O2 / X, O3 / X, O4 / X
        if self.K:
            for E, R in zip((E2, E3, E4), self.horizon_rows_m(w1, w2, w3, w4, mu)):
                for n, j in enumerate(self.hrows, start=1):
                    E[:, j] = self.tscale_m[n] * R[:, n]
        return np.concatenate([E1.ravel(), E2.ravel(), E3.ravel(), E4.ravel(), [ph]])

    def horizon_rows_m(self, w1, w2, w3, w4, mu):
        """rt_model.RT.horizon_rows in arb: each (Nt, K + 1)."""
        T = lambda f: f.dot(self.TRmT)
        P = lambda f: f[:, self.rev]
        nn = np.array([arb(n) for n in range(self.K + 1)], dtype=object)[None, :]
        Dp = lambda S, anti=False: (self.Dant if anti else self.Dper).dot(S) + mu * nn * S
        return horizon_series(T(w2), T(w3), T(w4), T(-P(w2)), T(-P(w4)), T(w1), mu, self.Dm, self.cm, Dp, self.tm_m)

    def delta_m(self, z):
        w1, w2, w3, w4, mu = self.unpack_m(z)
        pi = arb.pi()
        w30 = w3.dot(self.r0m)
        d0 = 4 * pi * mu - 2 * (2 * pi / self.Nt) * w30.sum()
        d1 = 4 * pi * mu - 2 * (2 * pi / self.Nt) * w3[:, 0].sum()
        return d0, d1


def interp_matrix(xs, xt, N):
    """Barycentric interpolation from the Lobatto points xs (arb) to xt (arb): (len(xt), N + 1) object."""
    w = [arb((-1) ** j) * (arb(1) / 2 if j in (0, N) else 1) for j in range(N + 1)]
    M = np.empty((len(xt), N + 1), dtype=object)
    for i, x0 in enumerate(xt):
        d = [x0 - xj for xj in xs]
        hit = [j for j in range(N + 1) if abs(float(d[j].mid())) < 1e-40]
        if hit:
            M[i, :] = [arb(1) if j == hit[0] else arb(0) for j in range(N + 1)]
            continue
        r = [w[j] / d[j] for j in range(N + 1)]
        sr = sum(r, arb(0))
        M[i, :] = [v / sr for v in r]
    return M


def resample_xi_m(zh, zl, D, Nt, N, Np, prec=192):
    """A double-double state (zh, zl) on Nt x N resampled exactly in xi to Nt x Np, in arb (issue #163). rt_model's
    resample_to works in f64 and leaves a rounding of about 1e-17, which the horizon rows amplify to about 1e-11."""
    flint.ctx.prec = prec
    src, tgt = RTmp(D, Nt, N, prec=prec), RTmp(D, Nt, Np, prec=prec)
    z = np.array([arb(float(h)) + arb(float(l)) for h, l in zip(zh, zl)], dtype=object)
    w1, w2, w3, w4, mu = src.unpack_m(z)
    M = interp_matrix(list(src.xm), list(tgt.xm), N)
    f = [g.dot(M.T) for g in (w1, w2, w3, w4)]
    return tgt.pack(f[0], f[1], f[2], f[3], mu)


def to_f64(z):
    return np.array([float(v.mid()) for v in z])


def newton_mixed(rm, z, tol=1e-30, it_max=40, J=None, log=print, refactor_every=0):
    """Iterative refinement with the f64 Jacobian at the f64 rounding of z (factored once, or every refactor_every
    steps), the residual in arb. z: f64 array or object array. Returns (z_obj, converged, history of max|F|)."""
    if z.dtype != object:
        z = obj(z)
    if J is None:
        J = rm.jacobian(to_f64(z))
    lu = sl.lu_factor(J)
    hist = []
    for it in range(it_max):
        F = rm.residual_m(z)
        Ff = to_f64(F)
        r = float(np.abs(Ff).max())
        hist.append(r)
        log(f"    mixed it {it}: max|F| {r:.3e}")
        if r < tol:
            return z, True, hist
        if it > 4 and r > 0.9 * hist[-2]:
            return z, False, hist
        if refactor_every and it and it % refactor_every == 0:
            lu = sl.lu_factor(rm.jacobian(to_f64(z)))
        dz = sl.lu_solve(lu, -Ff)
        z = z + obj(dz)
    return z, False, hist
