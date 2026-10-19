"""Nonlinear elimination for the weakly selected DSS problem.

At large D the Jacobian J has three kinds of directions (SVD at a base point z_b, J = U diag(s) V^T):
  strong  s/s_max >= cut_strong   (includes Delta, s/s_max ~ 1e-8 at D = 28)
  weak    cut_junk <= s/s_max < cut_strong   the "shape" modes: centre-data harmonics k = 3, 5 that control the
          singular horizon modes. The residual floor of the LM solvers lives here (D = 28: 1.5e-8 in the k = 3
          pair, s/s_max = 4e-11, Gauss-Newton step only ~4e-4 but far outside the linear range of the strong part).
  junk    s/s_max < cut_junk       high centre harmonics with no effect on F; frozen.
Coordinates a = V^T (z - z_b). The strong equations U_s^T F = 0 are solved exactly (Newton, exact Jacobian) for
every value of the weak coordinates a_w, and the weak equations U_w^T F = 0 are solved for a_w with the Schur
complement G = U_w^T J V_w - U_w^T J V_s (U_s^T J V_s)^-1 U_s^T J V_w. The junk equations are left alone: their
residual is the discretisation floor.

    z, info = ne_solve(R, z)            # R = rustkernel.RustResidual(S)
"""
import time
import numpy as np


def _res(R, z, dd):
    if dd:
        h, l = R.residual_dd(z)
        return h + l
    return R(z)


def ne_solve(R, z, cut_strong=1e-10, cut_junk=3e-14, outer_max=25, inner_max=10, inner_tol=1e-15, tol_w=1e-15,
             fix_delta=None, dd=True, verbose=False, basis=None):
    """fix_delta: if a number, Delta is held at that value (its column removed before the SVD)."""
    z = np.array(z, float)
    if fix_delta is not None:
        z[-1] = fix_delta
    t0 = time.time()
    F, J = R.jacobian(z)
    cols = slice(0, len(z) - 1) if fix_delta is not None else slice(0, len(z))
    if basis is None:
        U, s, Vt = np.linalg.svd(J[:, cols])
    else:
        U, s, Vt = basis
    r = s / s[0]
    Si, Wi = r >= cut_strong, (r < cut_strong) & (r >= cut_junk)
    n = len(z) if fix_delta is None else len(z) - 1
    pad = np.zeros(U.shape[1] - len(s), bool)          # fixed Delta: one more equation than unknowns
    Us, Uw = U[:, np.r_[Si, pad]], U[:, np.r_[Wi, pad]]
    Usel = U[:, len(s):]                               # the selection equation (only with Delta fixed)
    Vs = np.zeros((len(z), Si.sum())); Vs[:n] = Vt[Si].T
    Vw = np.zeros((len(z), Wi.sum())); Vw[:n] = Vt[Wi].T
    info = dict(n_strong=int(Si.sum()), n_weak=int(Wi.sum()), weak_s=r[Wi].tolist(), hist=[])
    if verbose:
        print(f"  ne: strong {Si.sum()}, weak {Wi.sum()} (s/smax {', '.join(f'{x:.1e}' for x in r[Wi])}), junk {(~Si & ~Wi).sum()}", flush=True)

    def inner(z):
        """Newton on the strong equations with the weak and junk coordinates fixed."""
        for it in range(inner_max):
            F, J = R.jacobian(z)
            F = _res(R, z, dd)
            es = Us.T @ F
            nes = np.linalg.norm(es)
            if nes < inner_tol:
                break
            A = Us.T @ J @ Vs
            da = -np.linalg.solve(A, es)
            ok = False
            for a in (1.0, 0.5, 0.25, 0.1, 0.03):
                zt = z + a * (Vs @ da)
                Ft = _res(R, zt, dd)
                if np.all(np.isfinite(Ft)) and np.linalg.norm(Us.T @ Ft) < nes:
                    z, ok = zt, True
                    break
            if not ok:
                break
        F, J = R.jacobian(z)
        F = _res(R, z, dd)
        return z, F, J

    z, F, J = inner(z)
    for k in range(outer_max):
        es, ew, ej = Us.T @ F, Uw.T @ F, np.linalg.norm(F) ** 2
        esel = Usel.T @ F
        ej = np.sqrt(max(ej - es @ es - ew @ ew - esel @ esel, 0.0))
        info["hist"].append((float(np.linalg.norm(F)), float(np.linalg.norm(es)), float(np.linalg.norm(ew)), float(ej), float(z[-1])))
        if verbose:
            print(f"  outer {k}: |F| {np.linalg.norm(F):.3e}  strong {np.linalg.norm(es):.1e}  weak {np.linalg.norm(ew):.2e}"
                  f"  junk {ej:.1e}  Delta {z[-1]:.12f}  ({time.time()-t0:.0f}s)", flush=True)
        if np.linalg.norm(ew) < tol_w:
            break
        A = Us.T @ J @ Vs; B = Us.T @ J @ Vw; C = Uw.T @ J @ Vs; Dm = Uw.T @ J @ Vw
        AiB = np.linalg.solve(A, B)
        G = Dm - C @ AiB
        rhs = ew - C @ np.linalg.solve(A, es)
        dw = -np.linalg.solve(G, rhs)
        ds = -np.linalg.solve(A, es) - AiB @ dw
        best = None
        for a in (1.0, 0.5, 0.25, 0.1, 0.03, 0.01):
            zt, Ft, Jt = inner(z + a * (Vw @ dw + Vs @ ds))
            nw = np.linalg.norm(Uw.T @ Ft); ns = np.linalg.norm(Us.T @ Ft)
            if np.isfinite(nw) and nw < np.linalg.norm(ew) and ns < max(1e-12, 10 * np.linalg.norm(es)):
                best = (zt, Ft, Jt, a)
                break
        if best is None:
            if verbose:
                print("  outer: no decrease of the weak residual; stopping", flush=True)
            break
        z, F, J, a = best
    info["sel"] = float(np.linalg.norm(Usel.T @ F)) if Usel.shape[1] else None
    info["F"] = float(np.linalg.norm(F)); info["seconds"] = round(time.time() - t0)
    return z, info
