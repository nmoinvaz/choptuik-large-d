"""Numba backend for inner_np.Inner: the same formal series, the same right-hand side and the same DOP853 integrator
(scipy's algorithm and coefficients, step-size control, initial step and error norm transcribed), compiled. It is a
drop-in replacement for the class used by hier.py (inner_np.Inner with M0 = M, lam = 1, i.e. every homotopy weight 1).

Why: at M = 7 one residual of hier.py costs ~4-5 s on the 8-thread machine and minutes on a shared cloud core, almost
all of it Python overhead in solve_ivp and np.convolve (25 complex unknowns, ~10^4 right-hand-side calls). The
arithmetic is unchanged; only the order of summation inside the convolutions and the dot products differs, so the
residuals agree with inner_np's to rounding, amplified by the conditioning of the jump (check below).

Method: DOP853, E. Hairer, S. P. Norsett, G. Wanner, Solving Ordinary Differential Equations I, Sec. II.10, as
implemented in scipy.integrate (coefficients imported from scipy.integrate._ivp.dop853_coefficients).

    python3 inner_nb.py check [M ...]        residuals of hier.demixed_multi with both backends, at the M = 5 zero
                                             extended by the amplitude-expansion seed; exit 1 if they disagree beyond
                                             the tolerance (relative to the level's scale)
    python3 inner_nb.py check --mutate=sign  the same with a sign flipped in the compiled right-hand side: must fail
    python3 inner_nb.py time M               time one residual with the numba backend

Use from another script: `import inner_nb; inner_nb.install()` replaces inner_np.Inner by InnerNB (also in spawned
worker processes, if the importing module calls install() at import time).
"""
import os
import sys
import time
import numpy as np
import numba as nb
from scipy.integrate._ivp import dop853_coefficients as dc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import inner_np as I                                    # noqa: E402

NS = dc.N_STAGES
A_ = np.ascontiguousarray(dc.A[:NS, :NS], dtype=np.float64)
B_ = np.ascontiguousarray(dc.B, dtype=np.float64)
C_ = np.ascontiguousarray(dc.C[:NS], dtype=np.float64)
E3_ = np.ascontiguousarray(dc.E3, dtype=np.float64)
E5_ = np.ascontiguousarray(dc.E5, dtype=np.float64)
MUTATE = [0.0]
# INNER_NB_RTOL=1e-13 (say) overrides the integrator tolerance hier.py passes (1e-12): a finer floor for M = 7, where the
# Schur-corrected r_3 at xi = 0.2 inherits the integration noise of the large leaked harmonic-5 and -7 jumps.
RTOL_OVERRIDE = float(os.environ["INNER_NB_RTOL"]) if os.environ.get("INNER_NB_RTOL") else None


@nb.njit(cache=True)
def conv(a, b, space, L):
    """harmonic convolution truncated to |k| <= L, masked by space (np.convolve(a, b)[L:3L+1] * space)"""
    n = 2 * L + 1
    out = np.zeros(n, np.complex128)
    for k in range(n):
        if space[k] == 0.0:
            continue
        lo = max(0, k + L - (n - 1))
        hi = min(n - 1, k + L)
        s = 0j
        for i in range(lo, hi + 1):
            s += a[i] * b[k + L - i]
        out[k] = s
    return out


@nb.njit(cache=True)
def rhs(xi, Y, oi, ei, odd, even, imw, L, mut):
    n = 2 * L + 1
    no, ne = oi.shape[0], ei.shape[0]
    D = np.zeros(n, np.complex128)
    S = np.zeros(n, np.complex128)
    C = np.zeros(n, np.complex128)
    for a in range(no):
        D[oi[a]] = Y[a]
        S[oi[a]] = Y[no + a]
    for a in range(ne):
        C[ei[a]] = Y[2 * no + a]
    dD = (conv(C, S, odd, L) - imw * xi * D) / (xi * xi)
    E = conv(D, D, even, L) / 2
    cE = conv(C, E, even, L)
    r2 = conv(C, xi * dD, odd, L) + (1.0 - 2.0 * mut) * conv(cE, D, odd, L) - conv(C, D, odd, L)
    dC = cE / xi
    dS = (r2 - imw * xi * S) / (xi * xi)
    out = np.empty(2 * no + ne, np.complex128)
    for a in range(no):
        out[a] = dD[oi[a]]
        out[no + a] = dS[oi[a]]
    for a in range(ne):
        out[2 * no + a] = dC[ei[a]]
    return out


@nb.njit(cache=True)
def fpath(kind, par, t, y, oi, ei, odd, even, imw, L, mut):
    if kind == 0:                                       # ray: xi = t e^{i par}
        u = np.exp(1j * par)
        return rhs(t * u, y, oi, ei, odd, even, imw, L, mut) * u
    z = par * np.exp(1j * t)                            # arc of radius par: xi = par e^{i t}
    return rhs(z, y, oi, ei, odd, even, imw, L, mut) * (1j * z)


@nb.njit(cache=True)
def rmsnorm(v):
    s = 0.0
    for i in range(v.shape[0]):
        s += v[i].real * v[i].real + v[i].imag * v[i].imag
    return np.sqrt(s) / np.sqrt(v.shape[0])


@nb.njit(cache=True)
def dop853(kind, par, t0, tb, y0, rtol, atol, oi, ei, odd, even, imw, L, mut, A, B, C, E3, E5, maxfev=10**9):
    """scipy.integrate.solve_ivp(method='DOP853') transcribed: returns (y(tb), nfev, status) with status 0 = ok,
    1 = step size too small, 2 = more than maxfev right-hand sides"""
    SAFETY, MINF, MAXF = 0.9, 0.2, 10.0
    expo = -1.0 / 8.0
    n = y0.shape[0]
    ns = A.shape[0]
    direction = 1.0 if tb >= t0 else -1.0
    y = y0.copy()
    t = t0
    f = fpath(kind, par, t, y, oi, ei, odd, even, imw, L, mut)
    nfev = 1
    # select_initial_step (order 7)
    interval = abs(tb - t0)
    if interval == 0.0:
        return y, nfev, 0
    scale = atol + np.abs(y) * rtol
    d0 = rmsnorm(y / scale)
    d1 = rmsnorm(f / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    h0 = min(h0, interval)
    y1 = y + h0 * direction * f
    f1 = fpath(kind, par, t0 + h0 * direction, y1, oi, ei, odd, even, imw, L, mut)
    nfev += 1
    d2 = rmsnorm((f1 - f) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / 8.0)
    h_abs = min(100 * h0, h1, interval)
    K = np.empty((ns + 1, n), np.complex128)
    while True:
        if t == tb:
            break
        if nfev > maxfev:
            return y, nfev, 2
        min_step = 10 * abs(np.nextafter(t, direction * np.inf) - t)
        if h_abs < min_step:
            h_abs = min_step
        accepted = False
        rejected = False
        while not accepted:
            if h_abs < min_step:
                return y, nfev, 1
            h = h_abs * direction
            t_new = t + h
            if direction * (t_new - tb) > 0:
                t_new = tb
            h = t_new - t
            h_abs = abs(h)
            # rk_step
            K[0] = f
            for s in range(1, ns):
                dy = np.zeros(n, np.complex128)
                for q in range(s):
                    dy += K[q] * A[s, q]
                K[s] = fpath(kind, par, t + C[s] * h, y + dy * h, oi, ei, odd, even, imw, L, mut)
                nfev += 1
            acc = np.zeros(n, np.complex128)
            for q in range(ns):
                acc += K[q] * B[q]
            y_new = y + h * acc
            f_new = fpath(kind, par, t + h, y_new, oi, ei, odd, even, imw, L, mut)
            nfev += 1
            K[ns] = f_new
            sc = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            e5 = np.zeros(n, np.complex128)
            e3 = np.zeros(n, np.complex128)
            for q in range(ns + 1):
                e5 += K[q] * E5[q]
                e3 += K[q] * E3[q]
            e5 = e5 / sc
            e3 = e3 / sc
            n5 = 0.0
            n3 = 0.0
            for i in range(n):
                n5 += e5[i].real * e5[i].real + e5[i].imag * e5[i].imag
                n3 += e3[i].real * e3[i].real + e3[i].imag * e3[i].imag
            if n5 == 0.0 and n3 == 0.0:
                err = 0.0
            else:
                err = abs(h) * n5 / np.sqrt((n5 + 0.01 * n3) * n)
            if err < 1:
                if err == 0:
                    factor = MAXF
                else:
                    factor = min(MAXF, SAFETY * err ** expo)
                if rejected:
                    factor = min(1.0, factor)
                h_abs *= factor
                accepted = True
            else:
                h_abs *= max(MINF, SAFETY * err ** expo)
                rejected = True
        t = t_new
        y = y_new
        f = f_new
        if direction * (t - tb) >= 0:
            break
    return y, nfev, 0


@nb.njit(cache=True)
def series_nb(F, imw, odd, even, k, L, xi0, J):
    """inner_np.Inner.series_at compiled (gravity on, all weights 1): returns Dd, Sg, Cc (full arrays), jt, ratio"""
    n = 2 * L + 1
    q = xi0 * xi0
    Dl = np.zeros((J + 2, n), np.complex128)
    Sl = np.zeros((J + 1, n), np.complex128)
    Gl = np.zeros((J + 2, n), np.complex128)
    El = np.zeros((J + 2, n), np.complex128)
    CEl = np.zeros((J + 2, n), np.complex128)
    mags = np.zeros(J + 2)
    Dl[0] = 2 * F
    for i in range(n):
        if k[i] == 0 and even[i] != 0.0:
            Gl[0, i] = 1.0
    mags[0] = np.max(np.abs(Dl[0]))
    nm = 1
    for j in range(J):
        sj = (2 * j + 1) * Dl[j] + imw * Dl[j]
        for i in range(1, j + 1):
            sj = sj - conv(Gl[i], Sl[j - i], odd, L)
        Sl[j] = sj * odd
        m = j + 1
        em = np.zeros(n, np.complex128)
        for i in range(m):
            em = em + conv(Dl[i], Dl[m - 1 - i], even, L)
        em = q * em / 2
        El[m] = em
        cm = np.zeros(n, np.complex128)
        for i in range(m):
            cm = cm + conv(Gl[i], El[m - i], even, L)
        CEl[m] = cm
        Gl[m] = cm / (2 * m)
        num = q * ((2 * j + 2) * Sl[j] + imw * Sl[j])
        for i in range(1, j + 2):
            num = num - 2 * (j + 1 - i) * conv(Gl[i], Dl[j + 1 - i], odd, L)
        for mm in range(1, j + 2):
            num = num - conv(CEl[mm], Dl[j + 1 - mm], odd, L)
        Dl[j + 1] = num * odd / (2 * j + 2)
        mags[nm] = np.max(np.abs(Dl[j + 1]))
        nm += 1
        if j > 10 and mags[nm - 1] > 10 * np.min(mags[:nm]):
            break
    jt = int(np.argmin(mags[1:nm])) + 1
    Dd = np.zeros(n, np.complex128)
    Sg = np.zeros(n, np.complex128)
    Cc = np.zeros(n, np.complex128)
    for i in range(jt):
        Dd = Dd + Dl[i]
        Sg = Sg + Sl[i]
        Cc = Cc + Gl[i]
    return xi0 * Dd, q * Sg, Cc, jt, mags[jt] / mags[0]


class InnerNB(I.Inner):
    """inner_np.Inner with the compiled series and integrator (gravity on, M0 = M, lam = 1 only)"""

    def __init__(self, M, w, Fh, gravity=True, M0=None, lam=1.0):
        if not gravity or (M0 is not None and M0 != M) or lam != 1.0:
            raise ValueError("InnerNB: gravity on, M0 = M, lam = 1 only")
        super().__init__(M, w, Fh, gravity, M0, lam)
        self._oi = np.nonzero(self.odd)[0].astype(np.int64)
        self._ei = np.nonzero(self.even)[0].astype(np.int64)
        self._odd = self.odd.astype(np.float64)
        self._even = self.even.astype(np.float64)
        self._imw = np.ascontiguousarray(self.imw, dtype=np.complex128)

    def series_at(self, xi0, J):
        Dd, Sg, Cc, jt, ratio = series_nb(np.ascontiguousarray(self.F, np.complex128), self._imw, self._odd,
                                          self._even, self.k.astype(np.int64), self.L, complex(xi0), int(J))
        return self.pack(Dd, Sg, Cc), jt, ratio

    def integrate(self, Y, kind, a, b, par, rtol=1e-12, atol=1e-16):
        if RTOL_OVERRIDE is not None:
            rtol = RTOL_OVERRIDE
        y, nfev, st = dop853(0 if kind == "ray" else 1, float(par), float(a), float(b),
                             np.ascontiguousarray(Y, np.complex128), rtol, atol, self._oi, self._ei, self._odd,
                             self._even, self._imw, self.L, MUTATE[0], A_, B_, C_, E3_, E5_)
        if st != 0:
            raise RuntimeError("dop853: step size too small")
        return y


def install():
    I.Inner = InnerNB


def _point(M):
    """test points: the M = 5 zero (hier.py --schur, logs/hier_M5_full_schur.log) for M <= 5; for M = 7 the fixed-(A, w)
    point of logs/hier_M7_fixA_run2.log, where level 7 is tuned (|r_7| 0.03 at its approach radius; an untuned F_7
    makes the arc at xi = 0.2 cost ~10^6 right-hand sides in either backend)"""
    if M == 7:
        return np.array([2.72, 1.0, 0.149215713, 0.014707521, 0.00844556, 0.001508163, 0.000475916, 0.000121569])
    x5 = np.array([2.718579189, 0.999228404, 0.149203575, 0.014719052, 0.008405347, 0.001470345])
    return x5[:M + 1]


def check(Ms, mutate=False):
    """both backends through hier.demixed_multi at the same radii. Compared are the entries hier.py uses (r_1, r_3 at
    xi_top = 0.2; r_nu, nu >= 5, at its approach and polish radii; the others are rounding noise amplified by up to
    e^{1/(2 xi^2)} ~ 1e20 and unused). The yardstick is scipy's own noise: the change of the same entry when scipy's
    rtol goes from 1e-12 to 0.8e-12. Pass: |numba - scipy| <= 10 max(noise, 1e-13 |r|) for every used entry."""
    import hier as H
    orig = I.Inner

    class InnerRt(orig):
        def jumps(self, r1s=(0.2,), r0=None, rtol=1e-12):
            return orig.jumps(self, r1s, r0, 0.8e-12)

    ok = True
    for M in Ms:
        x = _point(M)
        h = H.Hier(M)
        low, pol = h.radii(x)
        radii = [0.2] + list(low.values()) + list(pol.values())
        used = {(0.2, 1), (0.2, 3)} | {(low[nu], nu) for nu in low if nu >= 5} | {(pol[nu], nu) for nu in pol if nu >= 5}
        I.Inner = orig
        t = time.time()
        Rs = H.demixed_multi(x, M, radii)
        ts = time.time() - t
        I.Inner = InnerRt
        Rs2 = H.demixed_multi(x, M, radii)
        I.Inner = InnerNB
        MUTATE[0] = 1.0 if mutate else 0.0
        t = time.time()
        Rn = H.demixed_multi(x, M, radii)
        tn = time.time() - t
        I.Inner = orig
        worst = 0.0
        for r in sorted(Rs):
            a, a2, b = Rs[r], Rs2[r], Rn[r]
            for li, nu in enumerate(range(1, M + 1, 2)):
                za, za2, zb = (complex(*v[2 * li:2 * li + 2]) for v in (a, a2, b))
                noise = max(abs(za - za2), 1e-13 * abs(za))
                q = abs(za - zb) / noise
                u = (r, nu) in used
                if u:
                    worst = max(worst, q)
                print(f"  M {M} xi {r:.4f} r_{nu}: scipy {za:+.6e}  numba {zb:+.6e}  |diff| {abs(za - zb):.1e}  "
                      f"scipy rtol noise {abs(za - za2):.1e}  ratio {q:.2g}{'' if u else '  (unused)'}")
        print(f"M {M}: scipy {ts:.1f} s, numba {tn:.2f} s, worst |numba - scipy| / noise over used entries {worst:.2g}",
              flush=True)
        ok = ok and worst <= 10
    MUTATE[0] = 0.0
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "check":
        mut = any(a == "--mutate=sign" for a in sys.argv)
        Ms = [int(a) for a in sys.argv[2:] if not a.startswith("--")] or [1, 3, 5]
        ok = check(Ms, mut)
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    elif sys.argv[1] == "time":
        import hier as H
        M = int(sys.argv[2])
        install()
        xs = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--x=")]
        x = np.array([float(t) for t in xs[0].split(",")]) if xs else _point(M)
        h = H.Hier(M)
        low, pol = h.radii(x)
        for rep in range(3):
            t = time.time()
            R = H.demixed_multi(x, M, [0.2] + list(low.values()) + list(pol.values()))
            print(f"M {M}: one residual {time.time() - t:.2f} s (rep {rep})", flush=True)
        for r, v in sorted(R.items()):
            print(f"  xi {r:.4f}: {np.array2string(v, precision=4)}")
