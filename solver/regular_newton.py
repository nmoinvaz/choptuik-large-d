"""Newton with the horizon regularity conditions made explicit and well scaled.

Near the self-similarity horizon the interior solution is a regular part plus singular modes ~ u^sigma
(Re sigma = s0 ~ 0.6 D). Regularity (the condition that selects Delta) says their amplitudes vanish. In the
residual F those amplitudes only appear through the mismatch d between the integrated interior state and the
regular horizon series at the matching point u_K, multiplied by u_K^s0 (~1e-7 at D = 31), so least squares
on F cannot enforce them and the solution drifts along a weak valley.

Here, at each outer iteration, the two matching blocks are replaced by an equivalent set of rows:

* r2 (horizon series - node K-1): kept,
* d = r1 - r2 (integrated state - horizon series), split with an orthonormal basis of the regular manifold's
  tangent at u_K (columns of dSeries/d(horizon data)):
    Q_reg^T d     the part the horizon data can absorb (O(1), kept),
    W Q_perp^T d  the part they cannot: the singular-mode content. W = Sigma^-1 U^T from the SVD of
                  Q_perp^T dd/d(other unknowns), so these rows measure, in units of the unknowns, how far the
                  solution is from regular. They are O(1)-conditioned at any D.

The transformation is invertible, so the solutions are those of F = 0; only the weighting of the merit
function and the singular-value truncation change. Steps are Newton (truncated SVD drops only the
numerically null "junk" directions), with backtracking on |G|.
"""
import time
import numpy as np


def _blocks(S):
    sizes, K, L = S.layout()
    offs = np.concatenate([[0], np.cumsum(sizes)])
    blk = sum(sizes[2:6])
    hor = np.arange(offs[len(sizes) - 2], offs[len(sizes)])
    return K, L, blk, hor


def make_transform(S, R, z, sigma_floor=1e-14):
    """Row transformation T (as a function on F and on J) at the current z."""
    K, L, blk, hor = _blocks(S)
    F0, J = R.jacobian(z)
    r1 = slice((K - 1) * blk, K * blk)
    r2 = slice(K * blk, (K + 1) * blk)
    Jh = J[r2][:, hor]                                     # d(series)/d(horizon data), 94 x 47
    Q, _ = np.linalg.qr(Jh, mode="complete")
    nreg = np.linalg.matrix_rank(Jh)
    Qreg, Qperp = Q[:, :nreg], Q[:, nreg:]
    rest = np.setdiff1d(np.arange(L), hor)
    Jd = J[r1] - J[r2]
    M = Qperp.T @ Jd[:, rest]
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    W = (U / np.maximum(s, sigma_floor * max(1.0, s[0]))).T   # Sigma^-1 U^T
    info = dict(nreg=int(nreg), s_perp=s)

    def apply(A):
        """Rows of T applied to F (1-D) or to J / a stack (2-D, rows = residual index)."""
        A = np.asarray(A)
        one = A.ndim == 1
        B = A[:, None] if one else A
        if B.shape[0] != L:
            B = B.T
            tr = True
        else:
            tr = False
        d = B[r1] - B[r2]
        out = B.copy()
        out[r1] = np.vstack([Qreg.T @ d, W @ (Qperp.T @ d)])
        if tr:
            out = out.T
        return out[:, 0] if one else out

    return apply, F0, J, info


def regular_newton(S, R, z, maxit=12, cut=1e-13, use_dd=True, verbose=False, fix_delta=False, tol=1e-22):
    """Newton on the regularity-weighted system. Returns z, |F|, |G|, history."""
    z = np.array(z, float)
    hist = []
    t0 = time.time()
    for it in range(maxit):
        T, F0, J, info = make_transform(S, R, z)
        F = _res(R, z, use_dd)
        G = T(F)
        JG = T(J)
        nG = float(np.linalg.norm(G))
        hist.append((float(np.linalg.norm(F)), nG, z[-1]))
        if verbose:
            print(f"   it {it}: |F| {np.linalg.norm(F):.3e} |G| {nG:.3e} (singular-mode rows {np.linalg.norm(G[_rows_perp(S)]):.2e})"
                  f" Delta {z[-1]:.12f}  ({time.time()-t0:.0f}s)", flush=True)
        if nG < tol:
            break
        A = JG[:, :-1] if fix_delta else JG
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        keep = s >= cut * s[0]
        dz = -(Vt[keep].T @ ((U[:, keep].T @ G) / s[keep]))
        if fix_delta:
            dz = np.append(dz, 0.0)
        best = None
        for a in (1.0, 0.5, 0.25, 0.1, 0.03, 0.01):
            zt = z + a * dz
            Gt = T(_res(R, zt, use_dd))
            nt = float(np.linalg.norm(Gt))
            if np.isfinite(nt) and nt < nG:
                best = (zt, nt, a)
                break
        if best is None:
            if verbose:
                print("   no decrease along the Newton direction; stopping", flush=True)
            break
        z = best[0]
    F = _res(R, z, use_dd)
    T, _, _, _ = make_transform(S, R, z)
    return z, float(np.linalg.norm(F)), float(np.linalg.norm(T(F))), hist


def _res(R, z, use_dd):
    if use_dd:
        h, l = R.residual_dd(z)
        return h + l
    return R(z)


def _rows_perp(S):
    K, L, blk, hor = _blocks(S)
    return slice((K - 1) * blk + len(hor), K * blk)
