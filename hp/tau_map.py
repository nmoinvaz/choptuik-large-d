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

    from tau_map import make_solver_mapped, map_state
"""
import mpmath as mp
import numpy as np
from hpkernel.driver import NODES_OLD, dd_from_mp, plow_odd  # noqa: F401
from hpkernel._hpkernel import Solver


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


def mapped_inputs(M, s0, L, dps=40):
    """(dh, dl, ph, pl): the folded odd d/dtheta and the phase weights on the mapped half grid, double-double"""
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
                rows.append(v / gpi)
            ph.append(mp.sin(gi) * gpi)
        dh, dl = dd_from_mp(rows)
        phh, phl = dd_from_mp(ph)
    return dh.reshape(M, M), dl.reshape(M, M), phh, phl


def make_solver_mapped(D, M, s0, L, nodes=None, kt=36, kc=40, ks=120, segs=None, threads=0, match_node=None):
    """driver.make_solver on the mapped grid (no kfreeze: the projector is a uniform-grid object)"""
    nodes = list(NODES_OLD if nodes is None else nodes)
    with mp.workdps(40):
        nh, nl = dd_from_mp([mp.mpf(D) - 2])
    dh, dl, ph, pl = mapped_inputs(M, s0, L)
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
