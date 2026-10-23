"""(Used by scan_hp.py --pin and --predictor lagrange_dd.)
Pin-to-trend for the weak p1 harmonics: a residual offset on the Rust solver.

With make_solver(..., kfreeze=kpin) the horizon block's p rows are  P_low(kpin)(state_p - target_p) + P_high(kpin) p1,
i.e. p1's harmonics above kpin are pinned to zero.  PinnedSolver subtracts c = h from those rows, h = the wanted
values of p1's harmonics in (kpin, kfreeze] (zero above kfreeze), so the rows become
P_low(kpin)(...) + P_high(kpin)(p1 - h): those harmonics are held at h instead of 0 and drop out of the Newton
system. The Jacobian is unchanged (constant offset); dz = -J^-1 (F - c) = dz_rust + J^-1 c with the stored LU.

h must be band-limited to double-double precision (any component outside the band is unreachable by the pinned
unknowns and is amplified by the remaining weak directions), so it is built from its Fourier coefficients in mpmath."""
import mpmath as mp
import numpy as np
from hpkernel.driver import dd_add, dd_from_mp   # noqa: F401 (dd_add re-exported for scan_hp)


def band_coeffs(f_hi, f_lo, M, ks, dps=40):
    """cos/sin coefficients (a_k, b_k) of an odd (antiperiodic) field on the half grid of N = 2M points, k in ks,
    from double-double samples: f(theta_j) = sum_k a_k cos(k theta_j) + b_k sin(k theta_j)"""
    with mp.workdps(dps):
        N = 2 * M
        f = [mp.mpf(float(h)) + mp.mpf(float(l)) for h, l in zip(f_hi, f_lo)]
        out = []
        for k in ks:
            a = b = mp.mpf(0)
            for j in range(M):
                th = 2 * mp.pi * j / N
                # full grid: f(theta + pi) = -f(theta); both halves contribute equally for odd k
                a += 2 * f[j] * mp.cos(k * th)
                b += 2 * f[j] * mp.sin(k * th)
            out.append((float(a * 2 / N), float(b * 2 / N)))
    return out


def band_field(coeffs, ks, M, dps=40):
    """double-double samples (hi, lo) on the half grid of sum_k a_k cos(k theta) + b_k sin(k theta)"""
    with mp.workdps(dps):
        N = 2 * M
        vals = []
        for j in range(M):
            th = 2 * mp.pi * j / N
            v = mp.mpf(0)
            for (a, b), k in zip(coeffs, ks):
                v += mp.mpf(a) * mp.cos(k * th) + mp.mpf(b) * mp.sin(k * th)
            vals.append(v)
        return dd_from_mp(vals)


class PinnedSolver:
    def __init__(self, s, M, h_hi, h_lo):
        self.s = s
        L = s.size
        K = (L - 1) // (4 * M) - 1          # horizon block index
        self.ch, self.cl = np.zeros(L), np.zeros(L)
        self.ch[K * 4 * M:K * 4 * M + M] = h_hi
        self.cl[K * 4 * M:K * 4 * M + M] = h_lo

    def __getattr__(self, name):
        return getattr(self.s, name)

    def _sub(self, Fh, Fl):
        return dd_add(np.asarray(Fh), np.asarray(Fl), -self.ch, -self.cl)

    def residual(self, zh, zl):
        return self._sub(*self.s.residual(zh, zl))

    def _corr(self, dh, dl):
        xh, xl = self.s.lu_solve(self.ch, self.cl)
        return dd_add(np.asarray(dh), np.asarray(dl), np.asarray(xh), np.asarray(xl))

    def newton(self, zh, zl):
        dh, dl, Fh, Fl, pmin, pmax = self.s.newton(zh, zl)
        dh, dl = self._corr(dh, dl)
        Fh, Fl = self._sub(Fh, Fl)
        return dh, dl, Fh, Fl, pmin, pmax

    def chord(self, zh, zl):
        dh, dl, Fh, Fl = self.s.chord(zh, zl)
        dh, dl = self._corr(dh, dl)
        Fh, Fl = self._sub(Fh, Fl)
        return dh, dl, Fh, Fl


def predict_dd(hist, D, n=4):
    """Lagrange extrapolation through the last up to n (D, zh, zl) points (default 4: cubic), in double-double: the
    increments z_i - z_last are formed from hi and lo separately (exact for nearby points) and only their weighted sum
    is f64, so the prediction carries no 1e-16 * |z| rounding (which the horizon block amplifies to ~1e-7 at D ~ 60)."""
    pts = hist[-n:]
    Ds = [p[0] for p in pts]
    zh0, zl0 = pts[-1][1], pts[-1][2]
    inc = np.zeros_like(zh0)
    for i, (Di, zhi, zli) in enumerate(pts[:-1]):
        w = 1.0
        for j, Dj in enumerate(Ds):
            if j != i:
                w *= (D - Dj) / (Di - Dj)
        inc = inc + w * ((zhi - zh0) + (zli - zl0))
    return dd_add(zh0, zl0, inc, np.zeros_like(inc))
