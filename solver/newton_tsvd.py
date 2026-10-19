import numpy as np, time

def tsvd_newton(S, z, tol=1e-11, maxit=25, rel=1e-8, verbose=False):
    """Newton with a truncated-SVD solve: ignores the handful of nearly-free top-harmonic directions."""
    t0 = time.time(); best = (np.inf, z)
    for it in range(maxit):
        F, J = S.jacobian(z)
        nF = np.linalg.norm(F)
        if nF < best[0]: best = (nF, z.copy())
        if verbose: print(f"   it {it}: |F| {nF:.3e} Delta {z[-1]:.12f}", flush=True)
        if nF < tol: break
        U, s, Vt = np.linalg.svd(J); keep = s > rel * s[0]
        dz = -(Vt[keep].T @ ((U[:, keep].T @ F) / s[keep]))
        # damp only if the full step makes things much worse
        for t in (1.0, 0.5, 0.25, 0.1):
            zt = z + t * dz
            Ft = np.linalg.norm(S.residual(zt[None, :])[0])
            if np.isfinite(Ft) and (Ft < 10 * nF or t == 0.1): break
        z = zt
    nF = np.linalg.norm(S.residual(z[None, :])[0])
    if nF > best[0]: nF, z = best
    return z, nF, time.time() - t0


def damp_top(S, z, start=0.6, floor=1e-3):
    """Smoothly damp harmonics above start*kc in every block (Delta untouched). Used on extrapolated guesses so
    the nearly-free top-harmonic directions, which Newton leaves as they are, start out small."""
    parts, Dl = S.split(z[None, :])
    out = []
    for c, par in zip(parts, S.parities()):
        ks = S.kodd if par == "odd" else S.keven
        k = np.concatenate([ks, ks]) if par == "odd" else np.concatenate([ks[:1], ks[1:], ks[1:]])
        x = np.clip((k - start * S.kc) / ((1 - start) * S.kc), 0, None)
        out.append(c[0] * np.exp(np.log(floor) * x ** 2))
    return np.concatenate(out + [Dl])


def harmonic_weights(S, p=4):
    """Diagonal weights (k/kc)^p on every spectral unknown; 0 for Delta."""
    w = []
    for par in S.parities():
        ks = S.kodd if par == "odd" else S.keven
        k = np.concatenate([ks, ks]) if par == "odd" else np.concatenate([ks[:1], ks[1:], ks[1:]])
        w.append((k / S.kc) ** p)
    return np.concatenate(w + [[0.0]])


def tikhonov_newton(S, z, mu_rel=1e-7, p=4, tol=1e-10, maxit=25, verbose=False, return_best=False):
    """Gauss-Newton with a small penalty on high-harmonic content: among nearly equivalent solutions (the
    weakly determined top harmonics), pick the smoothest. mu is relative to the largest singular value of J."""
    t0 = time.time(); w = harmonic_weights(S, p); best = (np.inf, z)
    for it in range(maxit):
        F, J = S.jacobian(z)
        nF = np.linalg.norm(F)
        if nF < best[0]: best = (nF, z.copy())
        if verbose: print(f"   it {it}: |F| {nF:.3e} Delta {z[-1]:.12f} |Wz| {np.linalg.norm(w * z):.2e}", flush=True)
        smax = np.linalg.norm(J, 2) if it == 0 else smax
        mu = mu_rel * smax
        A = np.vstack([J, mu * np.diag(w)]); b = np.concatenate([-F, -mu * w * z])
        dz = np.linalg.lstsq(A, b, rcond=None)[0]
        for t in (1.0, 0.5, 0.25, 0.1):
            zt = z + t * dz
            Ft = np.linalg.norm(S.residual(zt[None, :])[0])
            if np.isfinite(Ft) and (Ft < 10 * nF or t == 0.1): break
        if np.linalg.norm(t * dz) < 1e-13 * max(1, np.linalg.norm(z)) and nF < tol:
            z = zt; break
        z = zt
        if nF < tol and it > 2: break
    nF = np.linalg.norm(S.residual(z[None, :])[0])
    if return_best and best[0] < nF: z, nF = best[1], best[0]
    return z, nF, time.time() - t0
