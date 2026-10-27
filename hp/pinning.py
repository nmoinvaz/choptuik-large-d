"""(Used by scan_hp.py --pin and --predictor lagrange_dd / hermite_pin, and by predictor_diag.py.)
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
from hpkernel.driver import QdhSolver, dd_add, dd_from_mp   # noqa: F401 (dd_add re-exported for scan_hp)
from multidouble import dd_scale


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

    def lm_steps(self, zh, zl, mus, fh=None, fl=None):
        """Levenberg-Marquardt steps for F - c (the Jacobian is unchanged): the wrapped solver's lm_steps with that
        residual (needs a kernel whose lm_steps takes fh, fl)"""
        if fh is None:
            fh, fl = self.residual(zh, zl)
        c = np.ascontiguousarray
        return self.s.lm_steps(c(zh), c(zl), list(mus), c(np.asarray(fh)), c(np.asarray(fl)))

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


def pinned_tangent(make, D, zh, zl, K, h=1e-12, tol_step=1e-32, hmax=0.05, qdh=True, parts=None, segs=None):
    """dz/dD at a converged point (zh, zl) of the kfreeze = K system: t = -J_K^-1 dF_K/dD, J_K's double-double LU at
    the point. p1's harmonics above K are out of the unknowns (their rows are P_high p1, independent of D), so t is
    zero there and the solve never meets their near-null directions; with all of them in (K = 13 at D = 67.7) t is
    1e6-1e22 instead of 3e-3 (notes in hp/data/large_d/t69/). dF_K/dD is the 4th-order difference on the f64 steps
    D + k h (k = -2, -1, 1, 2) actually taken (spacing 1.4e-14 near D = 68; at D = 67.7 it converges for h below
    1e-11, and F is linear in D only for steps below about 1e-7), each F(D + h_k) - F(D) formed in double-double.
    make(D, kfreeze): a solver as the scan builds it; qdh: the residual with the quad-double horizon (QdhSolver),
    parts: the state's third and fourth parts for it; segs: the step plan (default: planned from the state)."""
    s = make(D, K)
    if segs is None:
        segs = s.plan(zh, zl, tol_step, hmax)
    s.set_segs(segs)
    s.newton(zh, zl)                       # the Jacobian and LU at the point; the step is not taken

    def res(Dk):
        sk = make(Dk, K)
        sk.set_segs(segs)
        if qdh:
            sk = QdhSolver(sk)
            if parts is not None:
                sk._parts[sk._key(zh, zl)] = parts
        r = sk.residual(zh, zl)
        return np.asarray(r[0]), np.asarray(r[1])

    ks = (-2, -1, 1, 2)
    hk = np.array([(D + k * h) - D for k in ks])
    V = np.vander(np.r_[hk, 0.0], 5, increasing=True)
    w = np.linalg.solve(V.T, np.r_[0.0, 1.0, 0.0, 0.0, 0.0])      # d/dD at 0 of the quartic through the 5 points
    F0h, F0l = res(D)
    dh, dl = np.zeros(len(zh)), np.zeros(len(zh))
    for wk, k in zip(w[:4], ks):
        eh, el = dd_add(*res(D + k * h), -F0h, -F0l)
        eh, el = dd_scale(np.asarray(eh), np.asarray(el), wk)
        dh, dl = (np.asarray(v) for v in dd_add(dh, dl, eh, el))
    th, tl = s.lu_solve(-dh, -dl)
    return np.asarray(th) + np.asarray(tl)


def hermite_dd(D0, z0h, z0l, t0, D1, z1h, z1l, t1, D):
    """the cubic Hermite through (D0, z0, t0) and (D1, z1, t1) at D, in double-double: as z1 plus an increment,
    z1 + h00 (z0 - z1) + h (h10 t0 + h11 t1) (h00 + h01 = 1), with z0 - z1 formed from hi and lo separately, so no
    f64 rounding of z enters (pinning.predict_dd). Returns (zh, zl)."""
    h = D1 - D0
    u = (D - D0) / h
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h11 = u ** 3 - u ** 2
    inc = h00 * ((z0h - z1h) + (z0l - z1l)) + h * (h10 * t0 + h11 * t1)
    return dd_add(z1h, z1l, inc, np.zeros_like(inc))
