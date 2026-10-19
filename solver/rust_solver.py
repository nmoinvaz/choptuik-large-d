"""Solver plumbing on top of the Rust residual kernel (the rustkernel package at the repo root).

make(D, Delta_ref)    -> (S, R): frozen Python DSSSpec (layout, diagnostics) + RustResidual R (threads=2, AVX-512)
jacobian_rust(S, R, z) -> (F0, J): central differences for all node/centre columns via one deduplicated batch
                         (each perturbed row only recomputes the block whose inputs changed); the horizon-series
                         columns and the whole Delta column by complex step (exact), also via one deduplicated batch.
"""
import os
import numpy as np
from rustkernel import RustResidual
from spec_long import make_long
from spec_c import freeze_steps

NODES_L = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6]


def make(D, Delta_ref, Ks=60, nodes=NODES_L, threads=2, kc=23, rk=None):
    """rk: multiply every RK4 step count (default: env NE_RK, else 1). The near-horizon u-segments dominate the
    error at x_K = 0.7: D = 29.3, Delta moves -7.9e-6 -> -3.7e-7 (vs x_K = 0.6) going from 1x to 2x steps.
    env NE_RK_SEG: comma list of extra factors per segment, centre first (x_K 0.7: 9 segments, the last three in
    u = 1 - x); missing entries are 1. At D = 34.18 the linearised Delta response to the RK x3 error is entirely the
    last segment (x = 0.6 -> 0.7), next to the horizon series' radius of convergence."""
    S = freeze_steps(make_long(kc, D - 2, nodes=nodes, Ks=Ks), Delta_ref)
    mult = int(rk if rk is not None else os.environ.get("NE_RK", "1"))
    seg = [int(x) for x in os.environ.get("NE_RK_SEG", "").split(",") if x]   # extra per-segment factors (centre -> x_K)
    if mult == 1 and not seg:
        R = RustResidual(S, threads=threads, backend="simd")
    else:
        import rustkernel as _rk
        orig = _rk.step_counts

        def counts(S_):
            s = orig(S_)
            f = seg + [1] * (len(s[0]) - len(seg))
            return [mult * fi * k for fi, k in zip(f, s[0])], s[1]
        _rk.step_counts = counts
        try:
            R = RustResidual(S, threads=threads, backend="simd")
        finally:
            _rk.step_counts = orig
    return S, R


def jacobian_rust(S, R, z, eps=3e-7, h=1e-30):
    L = len(z)
    sizes, K, _ = S.layout()
    offs = np.concatenate([[0], np.cumsum(sizes)])
    hor = np.arange(offs[len(sizes) - 2], offs[len(sizes)])          # horizon-series unknowns (mS, aS)
    fd_cols = np.setdiff1d(np.arange(L - 1), hor)                   # everything else except Delta
    # central differences, one batch: rows 0 = base, then +eps / -eps pairs
    nb = len(fd_cols)
    Zb = np.repeat(z[None, :], 1 + 2 * nb, axis=0)
    Zb[1 + np.arange(nb), fd_cols] += eps
    Zb[1 + nb + np.arange(nb), fd_cols] -= eps
    Fb = R(Zb, dedup=True)
    F0 = Fb[0]
    J = np.zeros((L, L))
    J[:, fd_cols] = ((Fb[1:1 + nb] - Fb[1 + nb:]) / (2 * eps)).T
    # complex step: horizon columns and Delta
    Zc = np.repeat(z[None, :].astype(complex), len(hor) + 1, axis=0)
    Zc[np.arange(len(hor)), hor] += 1j * h
    Zc[-1, -1] += 1j * h
    Fc = R.residual_complex(Zc, dedup=True)
    J[:, hor] = (Fc[:len(hor)].imag / h).T
    J[:, -1] = Fc[-1].imag / h
    return F0, J


def convert_06_to_07(D, z6):
    """Solution vector matched at x_K=0.6 -> matched at x_K=0.7: insert the node-0.7 state from the horizon series at u=0.3."""
    return convert_06_to(D, z6, 0.7)


def convert_06_to(D, z6, xk, Ks=60):
    """kc 23 solution matched at x_K = 0.6 -> matched at xk > 0.6: append a node at xk whose state is the horizon
    series (Ks terms) evaluated at u = 1 - xk."""
    S6 = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=Ks), z6[-1])
    parts, _ = S6.split(z6[None, :]); parts = [p[0] for p in parts]
    mS = S6.phys(parts[-2][None, :], "odd"); aS = S6.phys(parts[-1][None, :], "even"); om = np.array([2 * np.pi / z6[-1]])
    P, M, A, C = S6.ssh_series(mS, aS, om)
    st = S6.trunc(np.stack([S6.eval_series(Q, 1 - xk) for Q in (P, M, A, C)]))
    node7 = np.concatenate([S6.spec(st[0], "odd")[0], S6.spec(st[1], "odd")[0], S6.spec(st[2], "even")[0], S6.spec(st[3], "even")[0]])
    sizes6, K6, _ = S6.layout(); offs6 = np.concatenate([[0], np.cumsum(sizes6)])
    e = offs6[2 + 4 * K6]
    return np.concatenate([z6[:e], node7, z6[e:]])
