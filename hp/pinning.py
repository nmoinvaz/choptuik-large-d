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
from hpkernel.driver import dd_add, dd_from_mp, qd_axpy   # noqa: F401 (dd_add re-exported for scan_hp)


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


class PinnedQ:
    """PinnedSolver's offset for the quad-double methods (Solver.residual_q, newton_q, as hpkernel.driver.newton_qd
    calls them): the kernel's frozen rows read coefficient - held value, so the quad-double Newton converges to the
    pinned solution. ch, cl: the offset of the PinnedSolver (its ch, cl); everything else is the kernel's."""
    def __init__(self, s, ch, cl):
        self.s = s
        self.ch, self.cl = np.ascontiguousarray(ch), np.ascontiguousarray(cl)
        self.zero = np.zeros(len(self.ch))

    def __getattr__(self, name):
        return getattr(self.s, name)

    def residual_q(self, z0, z1, z2, z3, mutate=0):
        F = self.s.residual_q(z0, z1, z2, z3, mutate=mutate)
        return qd_axpy(-1.0, self.ch, self.cl, self.zero, self.zero, *(np.ascontiguousarray(np.asarray(p)) for p in F))

    def newton_q(self, z0, z1, z2, z3, mutate=0):
        d0, d1, d2, d3, fh, fl, pmin, pmax = self.s.newton_q(z0, z1, z2, z3, mutate=mutate)
        x = self.s.qlu_solve(self.ch, self.cl, self.zero, self.zero)       # dz = -J^-1 (F - c) = -J^-1 F + J^-1 c
        d = qd_axpy(1.0, *(np.ascontiguousarray(np.asarray(p)) for p in x), *(np.ascontiguousarray(np.asarray(p)) for p in (d0, d1, d2, d3)))
        fh, fl = dd_add(np.asarray(fh), np.asarray(fl), -self.ch, -self.cl)
        return (*d, fh, fl, pmin, pmax)


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


class HeldFit:
    """Held values from a fit of the full-system states (hp/held_fit.py, notes/HELD_HARMONICS.md), the alternative to
    anm_run.py --trend's power law: for each held coefficient v (a_k or b_k of p1), sqrt(n) v is a Chebyshev series in
    x = 1/n on the fit's domain, the centre scaling p1 ~ F(theta)/sqrt(n) with F analytic in 1/n (README, "How the
    solution scales with D"). path: the JSON of held_fit.py fit. Values and Taylor coefficients in D are formed in
    mpmath (dps 50) by power-series arithmetic and returned as floats or mpf."""
    def __init__(self, path):
        import json
        with open(path) as fh:
            d = json.load(fh)
        self.x0, self.x1 = (mp.mpf(v) for v in d["domain"])
        self.c = {key: [mp.mpf(v) for v in cs] for key, cs in d["coeffs"].items()}
        self.meta = d

    def _series(self, D0, N):
        """Taylor coefficients in u = D - D0, orders 0..N, of every fitted v: {key: [v_0, ..., v_N]} (mpf)"""
        with mp.workdps(50):
            n0 = mp.mpf(D0) - 2

            def mul(p, q):
                return [mp.fsum(p[i] * q[j - i] for i in range(j + 1)) for j in range(N + 1)]
            x = [mp.mpf(-1) ** j / n0 ** (j + 1) for j in range(N + 1)]                 # 1/(n0 + u)
            w = self.x1 - self.x0
            t = [2 * v / w for v in x]
            t[0] -= (self.x1 + self.x0) / w
            rs = [mp.binomial(mp.mpf(-0.5), j) / n0 ** j / mp.sqrt(n0) for j in range(N + 1)]     # (n0 + u)^(-1/2)
            deg = max(len(cs) for cs in self.c.values()) - 1
            T = [[mp.mpf(1)] + [mp.mpf(0)] * N, t]
            for _ in range(2, deg + 1):
                T.append([2 * a - b for a, b in zip(mul(t, T[-1]), T[-2])])
            out = {}
            for key, cs in self.c.items():
                y = [mp.fsum(cs[m] * T[m][j] for m in range(len(cs))) for j in range(N + 1)]
                out[key] = mul(y, rs)
            return out

    def values(self, D, ks):
        """[(a_k, b_k)] at D for k in ks, floats"""
        s = self._series(D, 0)
        return [(float(s[f"{k}a"][0]), float(s[f"{k}b"][0])) for k in ks]

    def increment(self, D0, D, ks):
        """[(a_k(D) - a_k(D0), b_k(D) - b_k(D0))] for k in ks, the differences formed in mpmath (mpf)"""
        s0, s1 = self._series(D0, 0), self._series(D, 0)
        return [(s1[f"{k}a"][0] - s0[f"{k}a"][0], s1[f"{k}b"][0] - s0[f"{k}b"][0]) for k in ks]

    def jets(self, D0, N, ks):
        """[[(a_k, b_k)'s Taylor coefficients of order j in D - D0, for k in ks] for j in 0..N], floats"""
        s = self._series(D0, N)
        return [[(float(s[f"{k}a"][j]), float(s[f"{k}b"][j])) for k in ks] for j in range(N + 1)]
