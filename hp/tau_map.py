#!/usr/bin/env python3
"""A mapped tau grid for the double-double solver (notes/LOW_D_SPEEDUP.md, step 5): the half-period grid is uniform in
s, and theta = g(s) follows the periodic arctan map (Boyd, J. Comput. Phys. 98 (1992), doi:10.1016/0021-9991(92)90136-M)

    g(s) = s0 + atan2(L sin(s - s0), cos(s - s0)),   g'(s) = L / (cos^2(s - s0) + L^2 sin^2(s - s0)),

which puts the points densest at theta = s0 and s0 + pi (both pulses, g' = L there) and keeps g(s + pi) = g(s) + pi
(mod 2 pi), so the odd and even fields keep their parities and the folded derivative still applies. The kernel sees
the tau grid in two inputs only (hp/src/model.rs: the odd-sector d/dtheta and the phase row; the even fields a, c have
no tau derivative): here d/dtheta = diag(1/g'(s_i)) d/ds, and the phase row is sum_j sin(g(s_j)) g'(s_j) p1_j, the
trapezoid rule in s for the uniform grid's integral of sin(theta) p1(theta), so a mapped solution has the uniform one's
tau origin and the same Delta. L = 1 is the identity (the control: the inputs equal the uniform ones).

    from tau_map import make_solver_mapped, map_state, interp_dd

The grid functions (g_and_gp, mapped_inputs, interp_dd, map_state) need only mpmath, numpy and multidouble's Python
package; make_solver_mapped imports the kernel (hpkernel) when called.
"""
import mpmath as mp
import numpy as np
from multidouble import dd_add, dd_from_mp, two_prod

# mapped_inputs(mutate=...): deliberate breakage for the controls (hp/tau_map_gain.py): "gp" drops the factor 1/g' from
# the derivative rows (d/ds in place of d/dtheta), "phase_gp" drops g' from the phase row
MUTATIONS = ("gp", "phase_gp")


def g_and_gp(s, s0, L):
    """g(s) and g'(s), written as the identity plus deviations with the factor L - 1, so that L = 1 gives s and 1
    exactly: g - s = atan2((L - 1) sin u cos u, cos^2 u + L sin^2 u) (the tangent of v - u for tan v = L tan u) and
    g' - 1 = (L - 1)(1 - (1 + L) sin^2 u) / (cos^2 u + L^2 sin^2 u)"""
    u = s - s0
    su, cu = mp.sin(u), mp.cos(u)
    dg = mp.atan2((L - 1) * su * cu, cu ** 2 + L * su ** 2)
    dgp = (L - 1) * (1 - (1 + L) * su ** 2) / (cu ** 2 + L ** 2 * su ** 2)
    return s + dg, 1 + dgp


def g_inverse(theta, s0, L):
    """s with g(s) = theta (mod 2 pi), as a float array"""
    v = np.asarray(theta, float) - s0
    return s0 + np.arctan2(np.sin(v), L * np.cos(v))


def mapped_inputs(M, s0, L, dps=40, mutate=None):
    """(dh, dl, ph, pl): the folded odd d/dtheta and the phase weights on the mapped half grid, double-double
    (mutate: MUTATIONS, the controls)"""
    if mutate not in (None,) + MUTATIONS:
        raise ValueError(f"mutate must be one of {MUTATIONS}, not {mutate!r}")
    with mp.workdps(dps):
        N = 2 * M
        L, s0 = mp.mpf(L), mp.mpf(s0)
        rows, ph = [], []
        for i in range(M):
            si = mp.pi * i / M
            gi, gpi = g_and_gp(si, s0, L)
            for j in range(M):
                v = mp.mpf(0)
                for jj, sgn in ((j, 1), (j + M, -1)):
                    if jj != i:
                        v += sgn * mp.mpf(0.5) * (-1) ** (i - jj) * mp.cot((i - jj) * mp.pi / N)
                rows.append(v if mutate == "gp" else v / gpi)
            ph.append(mp.sin(gi) * (1 if mutate == "phase_gp" else gpi))
        dh, dl = dd_from_mp(rows)
        phh, phl = dd_from_mp(ph)
    return dh.reshape(M, M), dl.reshape(M, M), phh, phl


def make_solver_mapped(D, M, s0, L, nodes=None, kt=36, kc=40, ks=120, segs=None, threads=0, match_node=None, mutate=None):
    """driver.make_solver on the mapped grid (no kfreeze: the projector is a uniform-grid object); mutate as
    mapped_inputs"""
    from hpkernel.driver import NODES_OLD
    from hpkernel._hpkernel import Solver
    nodes = list(NODES_OLD if nodes is None else nodes)
    with mp.workdps(40):
        nh, nl = dd_from_mp([mp.mpf(D) - 2])
    dh, dl, ph, pl = mapped_inputs(M, s0, L, mutate=mutate)
    return Solver(M, float(nh[0]), float(nl[0]), dh, dl, ph, pl, nodes, segs or [], kt, kc, ks, threads, None, None, match_node)


def map_state(zh, zl, M_from, M_to, K, frm=None, to=None):
    """A state on one grid (frm = None: uniform; else (s0, L)) as a start on another (to, same convention), every field
    block by trigonometric interpolation in the source's s (f64; the lo parts dropped, as driver.resample_z)"""
    z = np.asarray(zh, float) + np.asarray(zl, float)
    parity = ["odd", "even"] + ["odd", "odd", "even", "even"] * K + ["odd", "even"]
    s_to = np.pi * np.arange(M_to) / M_to
    theta = s_to if to is None else np.array([float(g_and_gp(mp.mpf(v), mp.mpf(to[0]), mp.mpf(to[1]))[0]) for v in s_to])
    s_src = theta if frm is None else g_inverse(theta, frm[0], frm[1])
    out = []
    for j, par in enumerate(parity):
        f = z[j * M_from:(j + 1) * M_from]
        full = np.r_[f, -f] if par == "odd" else np.r_[f, f]
        c = np.fft.rfft(full) / (2 * M_from)
        k = np.arange(len(c))
        w = np.where((k == 0) | (k == M_from), 1.0, 2.0)
        out.append((w[None, :] * c[None, :] * np.exp(1j * np.outer(s_src, k))).real.sum(1))
    zn = np.concatenate(out + [z[-1:]])
    return zn, np.zeros_like(zn)


def parities(K):
    """the parity of each field block in the state's layout (p1, c0 | node (p, m, a, c) x K | m_h, a_h)"""
    return ["odd", "even"] + ["odd", "odd", "even", "even"] * K + ["odd", "even"]


def _grid_theta(M, grid, dps):
    """the points theta_i = g(s_i), s_i = pi i / M, of a grid (None: uniform; else (s0, L)), as mpf"""
    s = [mp.pi * i / M for i in range(M)]
    if grid is None:
        return s
    return [g_and_gp(v, mp.mpf(grid[0]), mp.mpf(grid[1]))[0] for v in s]


def _g_inverse_mp(theta, grid):
    """s with g(s) = theta on the branch continuous with s = theta (grid None: the identity)"""
    if grid is None:
        return theta
    s0, L = mp.mpf(grid[0]), mp.mpf(grid[1])
    v = theta - s0
    return s0 + v + mp.atan2((1 - L) * mp.sin(v) * mp.cos(v), L * mp.cos(v) ** 2 + mp.sin(v) ** 2)


def interp_matrix(M_from, M_to, frm=None, to=None, dps=60, mutate=None):
    """the trigonometric interpolation of a field from one half grid to another, as double-double matrices
    {parity: (Th, Tl)} of shape (M_to, M_from): row i holds the source weights of the target point theta_i = g_to(s_i).
    A field F of parity +-1 (F(s + pi) = +-F(s); in the source's s, since g(s + pi) = g(s) + pi) on the N = 2 M_from
    grid is the sum of its samples times S(x) = sin(M x) cot(x / 2) / (2 M), the kernel of the rfft interpolant with
    the Nyquist cosine counted once (map_state's weights); folded on the half grid it is sin(M x) cot(x) / M (even)
    and sin(M x) / (M sin x) (odd), with sin(M (s - s_j)) = (-1)^j sin(M s) for even M. A target point on a source
    point (mod pi) gets the unit row exactly (the odd fold flips its sign past pi), so the identity map gives the
    source values bit for bit. A target within 1e-30 grid steps of a source point counts as on it; beyond that,
    sin(M s) at dps digits keeps 1e-(dps - 30) of the kernel's value, so dps 60 keeps every entry to double-double
    (a map's centre s0 is a float, and its point can lie 1e-17 from a source point). mutate="inverse": the target's
    theta taken as the source's s, the inverse map skipped (a control). mutate="parity": the odd kernel
    for the even fields too (the planted check's control)."""
    if M_from % 2:
        raise ValueError("M_from must be even")
    with mp.workdps(dps):
        M = M_from
        tol = mp.mpf(10) ** -30
        th = _grid_theta(M_to, to, dps)
        src = th if mutate == "inverse" else [_g_inverse_mp(t, frm) for t in th]
        out = {"odd": [], "even": []}
        for s in src:
            q = s * M / mp.pi                                 # s in source grid steps
            n = int(mp.nint(q))
            if abs(q - n) < tol:
                j, wrap = n % M, (n - n % M) // M             # s = s_j + wrap pi
                for par in ("odd", "even"):
                    row = [mp.mpf(0)] * M
                    row[j] = mp.mpf((-1) ** wrap if par == "odd" else 1)
                    out[par].extend(row)
                continue
            sm = mp.sin(M * s) / M
            for j in range(M):
                x = s - mp.pi * j / M
                c = (-1) ** j * sm
                out["odd"].append(c / mp.sin(x))
                out["even"].append(c / mp.sin(x) if mutate == "parity" else c * mp.cot(x))
        T = {}
        for par in ("odd", "even"):
            h, lo = dd_from_mp(out[par])
            T[par] = (h.reshape(M_to, M), lo.reshape(M_to, M))
    return T


def dd_matmul(Th, Tl, fh, fl):
    """(T f) in double-double for T = Th + Tl (rows, M_from columns) and the columns of f = fh + fl (M_from, n): each
    product exact (two_prod) with its cross terms, summed accurately (dd_add); about M_from 2^-106 of sum |T f|"""
    acc_h = np.zeros((Th.shape[0], fh.shape[1]))
    acc_l = np.zeros_like(acc_h)
    for j in range(Th.shape[1]):
        p, e = two_prod(Th[:, j:j + 1], fh[j:j + 1, :])
        e = e + (Th[:, j:j + 1] * fl[j:j + 1, :] + Tl[:, j:j + 1] * fh[j:j + 1, :])
        acc_h, acc_l = dd_add(acc_h, acc_l, p, e)
    return acc_h, acc_l


def interp_dd(zh, zl, M_from, M_to, K, frm=None, to=None, T=None, mutate=None):
    """a state (zh, zl) on one grid (frm None: uniform; else (s0, L)) at the points of another (to, same convention),
    every field block by trigonometric interpolation in the source's s, in double-double (map_state keeps only f64);
    Delta copied. T: interp_matrix(M_from, M_to, frm, to), when already made. mutate="lo": the lo parts dropped as
    in map_state (the planted check's control); "parity" and "inverse" as interp_matrix; "dps40" the matrix at 40
    digits, too few for a target 1e-17 from a source point (another control)"""
    zh, zl = np.asarray(zh, float), np.asarray(zl, float)
    if mutate == "lo":
        zl = np.zeros_like(zl)
    T = T or interp_matrix(M_from, M_to, frm, to, dps=40 if mutate == "dps40" else 60,
                           mutate=mutate if mutate in ("parity", "inverse") else None)
    par = parities(K)
    nb = len(par)
    oh, ol = np.zeros(nb * M_to + 1), np.zeros(nb * M_to + 1)
    for p in ("odd", "even"):
        idx = [j for j in range(nb) if par[j] == p]
        fh = np.stack([zh[j * M_from:(j + 1) * M_from] for j in idx], axis=1)
        fl = np.stack([zl[j * M_from:(j + 1) * M_from] for j in idx], axis=1)
        h, lo = dd_matmul(*T[p], fh, fl)
        for c, j in enumerate(idx):
            oh[j * M_to:(j + 1) * M_to], ol[j * M_to:(j + 1) * M_to] = h[:, c], lo[:, c]
    oh[-1], ol[-1] = zh[-1], zl[-1]
    return oh, ol
