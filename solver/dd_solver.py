"""Newton / Levenberg-Marquardt in double-double: the unknowns z = z_hi + z_lo and the residual are
double-double (rustkernel residual_dd), the Jacobian and the linear algebra stay in f64 (exact Dual
Jacobian at z_hi, one SVD per Jacobian). This is iterative refinement: each step only needs the f64
Jacobian to be a good approximation, while the residual it drives to zero is accurate to ~1e-27, so
directions that f64 noise hides (the weak selection of Delta at large D) can be converged.

    from dd_solver import dd_solve
    zh, zl, nF, hist = dd_solve(R, zh, zl)          # R = rustkernel.RustResidual(S)
"""
import time
import numpy as np


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def dd_add(zh, zl, d):
    """(zh + zl) + d, renormalised, elementwise (d in f64)."""
    s, e = two_sum(zh, d)
    e = e + zl
    h = s + e
    return h, e - (h - s)


def dd_norm(Fh, Fl):
    return float(np.linalg.norm(Fh + Fl, axis=-1)) if np.ndim(Fh) == 1 else np.linalg.norm(Fh + Fl, axis=-1)


def dd_solve(R, zh, zl=None, maxjac=12, ncorr=4, mus=None, tol=1e-25, cut=0.0, verbose=False, stall=3, fix_delta=False, use_dd=True):
    """Levenberg-Marquardt with a damping grid and chord corrections, double-double residual.
    cut > 0 additionally drops singular directions with sigma/sigma_max < cut (junk directions the f64
    Jacobian cannot resolve). Returns zh, zl, |F|, history."""
    zh = np.array(zh, float)
    zl = np.zeros_like(zh) if zl is None else np.array(zl, float)
    if not use_dd:                       # plain f64 residual (when rounding is far below the |F| floor)
        R = _F64(R)
    mus = 10.0 ** np.arange(-4, -16.5, -0.5) if mus is None else np.asarray(mus)
    Fh, Fl = R.residual_dd(zh, zl)
    F = Fh + Fl
    nF = float(np.linalg.norm(F))
    hist = [(nF, zh[-1], zl[-1])]
    bad = 0
    t0 = time.time()
    for it in range(maxjac):
        if nF < tol:
            break
        _, J = R.jacobian(zh)
        if fix_delta:                       # Delta held fixed: solve for the other unknowns only
            U, s, Vt = np.linalg.svd(J[:, :-1], full_matrices=False)
            Vt = np.hstack([Vt, np.zeros((Vt.shape[0], 1))])
        else:
            U, s, Vt = np.linalg.svd(J, full_matrices=False)
        keep = s >= cut * s[0]
        filt = np.stack([np.where(keep, s / (s * s + (mu * s[0]) ** 2), 0.0) for mu in mus])

        def steps(Fb):
            return -((U.T @ Fb.T).T * filt) @ Vt

        H = np.repeat(zh[None, :], len(mus), 0)
        L = np.repeat(zl[None, :], len(mus), 0)
        H, L = dd_add(H, L, steps(np.repeat(F[None, :], len(mus), 0)))
        best = (nF, None, None, None, 0)
        for c in range(ncorr + 1):
            Fbh, Fbl = R.residual_dd(H, L)
            Fb = Fbh + Fbl
            nb = np.linalg.norm(Fb, axis=1)
            for q in range(len(mus)):
                if np.isfinite(nb[q]) and nb[q] < best[0]:
                    best = (nb[q], H[q].copy(), L[q].copy(), mus[q], c)
            if c == ncorr:
                break
            Fb[~np.isfinite(nb)] = 0.0
            H, L = dd_add(H, L, steps(Fb))
        if best[1] is None:
            break
        prev = nF
        zh, zl = best[1], best[2]
        Fh, Fl = R.residual_dd(zh, zl)
        F = Fh + Fl
        nF = float(np.linalg.norm(F))
        hist.append((nF, zh[-1], zl[-1]))
        if verbose:
            print(f"   dd jac {it}: |F| {nF:.3e}  Delta {zh[-1]:.15f}{zl[-1]:+.2e}  mu {best[3]:.1e} corr {best[4]}  ({time.time()-t0:.0f}s)", flush=True)
        bad = bad + 1 if nF > 0.5 * prev else 0
        if bad >= stall:
            break
    return zh, zl, nF, hist


class _F64:
    """Wraps a RustResidual so dd_solve runs on the f64 residual (the low word is added, then ignored)."""
    def __init__(self, R):
        self.R = R

    def residual_dd(self, H, L):
        F = self.R(np.asarray(H) + np.asarray(L))
        return F, np.zeros_like(F)

    def jacobian(self, z):
        return self.R.jacobian(z)


def valley_tangent(R, z, rcond=1e-12):
    """dz/dDelta along the solution valley: the least-squares solution of J_rest t = -J_Delta, so that
    z + t (Delta - z[-1]) is a first-order solution at the displaced Delta."""
    _, J = R.jacobian(np.asarray(z, float))
    t = -np.linalg.lstsq(J[:, :-1], J[:, -1], rcond=rcond)[0]
    return np.append(t, 1.0)


def profile(R, zh, zl, deltas, tangent=True, **kw):
    """phi(Delta) = min over the other unknowns of |F|, at each Delta in `deltas`. Each fixed-Delta solve
    starts from z moved along the valley tangent (tangent=True), which keeps the solves converging for
    displacements far larger than the free solver tolerates. The selected period is the minimiser of phi;
    phi^2 is quadratic near it."""
    out = []
    zh = np.array(zh, float); zl = np.array(zl, float)
    t = valley_tangent(R, zh) if tangent else None
    h, l = zh, zl
    for d in deltas:
        if tangent:
            h = zh + t * (d - zh[-1]); l = zl.copy()
        else:
            h = h.copy(); l = l.copy()
        h[-1] = d; l[-1] = 0.0
        h, l, nF, _ = dd_solve(R, h, l, fix_delta=True, **kw)
        out.append((d, nF))
    return np.array(out), h, l


def profile_minimum(prof):
    """Delta* and curvature from a least-squares parabola through phi^2(Delta)."""
    d, f = prof[:, 0], prof[:, 1]
    d0 = d.mean()
    c = np.polyfit(d - d0, f * f, 2)
    return d0 - c[1] / (2 * c[0]), c
