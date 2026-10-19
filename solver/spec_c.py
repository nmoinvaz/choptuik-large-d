"""Central-difference Jacobian for DSSSpec: the forward-difference Jacobian (eps = 1e-7) has errors ~1e-7, comparable
with the Delta-selection strength at D > 9, so Newton stops contracting along the Delta direction."""
import numpy as np
from newton_tsvd import harmonic_weights
import time

def jacobian_c(S, z, eps=1e-5):
    sizes, K, L = S.layout()
    offs = np.concatenate([[0], np.cumsum(sizes)])
    F0 = S.residual(z[None, :])[0]
    J = np.zeros((L, L))
    blk = sum(sizes[2:6]); node0 = offs[2]
    for b, cols, (dR, _) in S._blocks(z, central=True, eps=eps):
        r0 = b * blk; c0 = offs[cols[0]]
        J[r0:r0 + blk, c0:c0 + dR.shape[1]] = dR
        tgt = node0 + blk * (b if b < K else K - 1)
        J[r0:r0 + blk, tgt:tgt + blk] -= np.eye(blk)
    J[-1, offs[0] + len(S.kodd)] = 1.0
    hD = eps * max(1.0, z[-1])
    zp, zm = z.copy(), z.copy(); zp[-1] += hD; zm[-1] -= hD
    J[:, -1] = (S.residual(zp[None, :])[0] - S.residual(zm[None, :])[0]) / (2 * hD)
    return F0, J

def newton_c(S, z, mu_rel=1e-10, p=4, tol=1e-12, maxit=12, eps=1e-5, verbose=False):
    """Gauss-Newton with central-difference Jacobian and a tiny smoothness penalty (only to fix the near-null
    top harmonics). Returns the best iterate."""
    t0 = time.time(); w = harmonic_weights(S, p); best = (np.inf, z.copy()); smax = None
    for it in range(maxit):
        F, J = jacobian_c(S, z, eps)
        nF = np.linalg.norm(F)
        if nF < best[0]: best = (nF, z.copy())
        if verbose: print(f"   it {it}: |F| {nF:.3e} Delta {z[-1]:.12f}", flush=True)
        if nF < tol: break
        if smax is None: smax = np.linalg.norm(J, 2)
        mu = mu_rel * smax
        A = np.vstack([J, mu * np.diag(w)]); b = np.concatenate([-F, -mu * w * z])
        dz = np.linalg.lstsq(A, b, rcond=None)[0]
        for t in (1.0, 0.5, 0.25, 0.1):
            zt = z + t * dz
            Ft = np.linalg.norm(S.residual(zt[None, :])[0])
            if np.isfinite(Ft) and Ft < nF: break
        else:
            if verbose: print("   no decrease along the step; stopping", flush=True)
            break
        z = zt
    nF = np.linalg.norm(S.residual(z[None, :])[0])
    if nF < best[0]: best = (nF, z.copy())
    return best[1], best[0], time.time() - t0

import types
def freeze_steps(S, Delta_ref, margin=1.04):
    """Make the RK4 step counts independent of Delta (they used ceil(.../omega), which makes the residual jump
    by ~1e-9 whenever Delta crosses a threshold). Steps are chosen for omega_ref = margin * 2 pi / Delta_ref."""
    om_ref = margin * 2 * np.pi / Delta_ref
    def nseg(self, xa, xb, omega=None):
        h = min(self.seg_h, 0.15 / (0.5 * self.N * om_ref))
        return max(4, int(np.ceil(abs(xb - xa) / h)))
    def integrate_u(self, s, xa, xb, omega):
        ya, yb = np.log(1 - xa), np.log(1 - xb)
        hy = min(0.01, 0.45 / (0.5 * self.N * om_ref))
        steps = max(4, int(np.ceil(abs(yb - ya) / hy)))
        h = (yb - ya) / steps; y = ya
        f = lambda y, s: -np.exp(y) * self.rhs(1 - np.exp(y), s, omega)
        for _ in range(steps):
            k1 = f(y, s); k2 = f(y + h / 2, s + h / 2 * k1); k3 = f(y + h / 2, s + h / 2 * k2); k4 = f(y + h, s + h * k3)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); y += h
        return s
    S.nseg = types.MethodType(nseg, S); S.integrate_u = types.MethodType(integrate_u, S)
    return S

def lm_solve(S, z, mu_rel=1e-5, eps=3e-7, tol=1e-13, maxjac=25, verbose=False, jac=None):
    """Levenberg-Marquardt with central-difference Jacobian (uniform step penalty). Returns best z, |F|, seconds."""
    t0 = time.time()
    Fz = lambda zz: S.residual(zz[None, :])[0]
    F = Fz(z); nF = np.linalg.norm(F)
    for it in range(maxjac):
        if nF < tol: break
        _, J = (jac or jacobian_c)(S, z, eps)
        smax = np.linalg.norm(J, 2) if it == 0 else smax
        JTJ = J.T @ J; g = J.T @ F
        accepted = False
        for _ in range(12):
            mu = mu_rel * smax
            dz = np.linalg.solve(JTJ + mu * mu * np.eye(len(z)), -g)
            zt = z + dz; Ft = Fz(zt); nt = np.linalg.norm(Ft)
            if np.isfinite(nt) and nt < nF:
                z, F, nF = zt, Ft, nt; mu_rel = max(mu_rel / 4, 1e-12); accepted = True; break
            mu_rel *= 4
        if verbose: print(f"   jac {it}: |F| {nF:.3e} Delta {z[-1]:.12f} mu_rel {mu_rel:.1e} |dz| {np.linalg.norm(dz):.1e}", flush=True)
        if not accepted: break
    return z, nF, time.time() - t0


# ---------- complex-step Jacobian for the horizon (SSH-series) block ----------
def _cplx(fun):
    def g(f, *a, **k):
        if np.iscomplexobj(f):
            return fun(f.real, *a, **k) + 1j * fun(f.imag, *a, **k)
        return fun(f, *a, **k)
    return g

def horizon_block_cs(S, z, h=1e-30):
    """Exact (complex-step) derivative of the horizon block residual w.r.t. the SSH data and Delta.
    Returns (dR [blk x nh], dR/dDelta [blk])."""
    sizes, K, L = S.layout()
    parts, Dl = S.split(z[None, :]); parts = [p[0] for p in parts]; D = float(Dl[0])
    no, ne = sizes[-2], sizes[-1]; nh = no + ne
    base = np.concatenate([parts[-2], parts[-1]])
    X = np.repeat(base[None, :], nh + 1, axis=0).astype(complex)
    X[:nh] += 1j * h * np.eye(nh)
    om = np.full(nh + 1, 2 * np.pi / D, dtype=complex); om[-1] = 2 * np.pi / (D + 1j * h)
    physc = _cplx(lambda c, par: S.phys(c, par))
    mS = physc(X[:, :no], "odd"); aS = physc(X[:, no:], "even")
    old = (S._D_unit, S.trunc)
    S._D_unit = _cplx(old[0]); S.trunc = _cplx(old[1])
    try:
        s = S._ssh_to_last(mS, aS, om)
    finally:
        S._D_unit, S.trunc = old
    specc = _cplx(lambda f, par: S.spec(f, par))
    R = np.concatenate([specc(s[0], "odd"), specc(s[1], "odd"), specc(s[2], "even"), specc(s[3], "even")], axis=1)
    dR = (R.imag / h)
    return dR[:nh].T, dR[nh]

def jacobian_cs(S, z, eps=3e-7):
    """Central differences everywhere except the horizon row block, which uses the complex step."""
    F0, J = jacobian_c(S, z, eps)
    sizes, K, L = S.layout(); offs = np.concatenate([[0], np.cumsum(sizes)])
    blk = sum(sizes[2:6]); r0 = K * blk; c0 = offs[len(sizes) - 2]
    dR, dD = horizon_block_cs(S, z)
    J[r0:r0 + blk, c0:c0 + dR.shape[1]] = dR
    J[r0:r0 + blk, -1] = dD
    return F0, J


def lm_svd(S, z, tol=1e-13, maxjac=30, chord=4, verbose=False, jac=None, eps=3e-7, mus=None):
    """LM variant: one SVD per Jacobian, then scan a grid of damping values and keep the best trial point;
    extra chord steps reuse the same SVD while they keep reducing |F|. Returns best z, |F|, seconds."""
    jac = jac or jacobian_cs
    mus = mus if mus is not None else 10.0 ** np.arange(-2, -13.5, -0.5)
    t0 = time.time()
    Fz = lambda zz: S.residual(zz[None, :])[0]
    F = Fz(z); nF = np.linalg.norm(F)
    for it in range(maxjac):
        if nF < tol: break
        _, J = jac(S, z, eps)
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        improved = False
        for c in range(1 + chord):
            g = U.T @ F; best = (nF, None, None)
            for mu in mus:
                m = mu * s[0]
                dz = -(Vt.T @ (g * s / (s * s + m * m)))
                zt = z + dz; nt = np.linalg.norm(Fz(zt))
                if np.isfinite(nt) and nt < best[0]: best = (nt, zt, mu)
            if best[1] is None or best[0] > 0.98 * nF:
                if best[1] is not None and best[0] < nF:
                    z = best[1]; F = Fz(z); nF = best[0]; improved = True
                break
            z = best[1]; F = Fz(z); nF = best[0]; improved = True
            if verbose: print(f"   jac {it} chord {c}: |F| {nF:.3e} Delta {z[-1]:.12f} mu {best[2]:.1e}", flush=True)
            if nF < tol: break
        if not improved: break
    return z, nF, time.time() - t0


def lm_svd2(S, z, tol=1e-13, maxjac=30, ncorr=3, verbose=False, jac=None, eps=3e-7, mus=None):
    """lm_svd with curvature correction: every trial point z+dz(mu) is followed by up to ncorr chord corrections
    (same SVD, same mu) before being judged, so steps along curved valleys are not rejected for their
    second-order error. Returns best z, |F|, seconds."""
    jac = jac or jacobian_cs
    mus = mus if mus is not None else 10.0 ** np.arange(-3, -13.5, -0.5)
    t0 = time.time()
    Fz = lambda zz: S.residual(zz[None, :])[0]
    F = Fz(z); nF = np.linalg.norm(F)
    for it in range(maxjac):
        if nF < tol: break
        _, J = jac(S, z, eps)
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        step = lambda FF, m: -(Vt.T @ ((U.T @ FF) * s / (s * s + m * m)))
        best = (nF, None, None, 0)
        for mu in mus:
            m = mu * s[0]; zt = z + step(F, m); Ft = Fz(zt); nt = np.linalg.norm(Ft)
            for c in range(ncorr + 1):
                if np.isfinite(nt) and nt < best[0]: best = (nt, zt.copy(), mu, c)
                if c == ncorr or not np.isfinite(nt): break
                zt = zt + step(Ft, m); Ft = Fz(zt); nt = np.linalg.norm(Ft)
        if best[1] is None: break
        z = best[1]; F = Fz(z); nF = best[0]
        if verbose: print(f"   jac {it}: |F| {nF:.3e} Delta {z[-1]:.12f} mu {best[2]:.1e} corr {best[3]}", flush=True)
    return z, nF, time.time() - t0


def horizon_mode_cs(S, z, lam, h=1e-30):
    """Complex-step horizon block of the mode matrix M(lam) (perturbations ~ e^{lam tau})."""
    sizes, K, L = S.layout()
    parts, Dl = S.split(z[None, :]); parts = [p[0] for p in parts]; D = float(Dl[0])
    no, ne = sizes[-2], sizes[-1]; nh = no + ne
    base = np.concatenate([parts[-2], parts[-1]])
    X = np.repeat(base[None, :], nh + 1, axis=0).astype(complex)
    X[1:] += 1j * h * np.eye(nh)
    om = np.full(nh + 1, 2 * np.pi / D, dtype=complex)
    physc = _cplx(lambda c, par: S.phys(c, par))
    mS = physc(X[:, :no], "odd"); aS = physc(X[:, no:], "even")
    old = (S._D_unit, S.trunc)
    S._D_unit = _cplx(old[0]); S.trunc = _cplx(old[1]); S.lam = lam
    try:
        s = S._ssh_to_last(mS, aS, om)
    finally:
        S._D_unit, S.trunc = old; S.lam = 0.0
    specc = _cplx(lambda f, par: S.spec(f, par))
    R = np.concatenate([specc(s[0], "odd"), specc(s[1], "odd"), specc(s[2], "even"), specc(s[3], "even")], axis=1)
    return (R.imag[1:] / h).T

def mode_matrix_cs(S, z, lam, eps=1e-6):
    M = S.mode_matrix(z, lam, eps)
    sizes, K, L = S.layout(); offs = np.concatenate([[0], np.cumsum(sizes)])
    blk = sum(sizes[2:6]); r0 = K * blk; c0 = offs[len(sizes) - 2]
    dR = horizon_mode_cs(S, z, lam)
    M[r0:r0 + blk, c0:c0 + dR.shape[1]] = dR
    return M
