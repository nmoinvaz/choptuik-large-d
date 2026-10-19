"""Driver for the double-double solver: set-up (grid, d/dtheta, phase row), seeds from the f64 solutions, Newton.

Double-double vectors are (hi, lo) pairs of float64 arrays. Layout of z (M = half-grid points, K nodes):
    p1, c0 | node_0 (p, m, a, c) | ... | node_{K-1} | m_h, a_h | Delta              length 4M(K+1) + 1

    from hpkernel.driver import make_solver, seed_from_f64, newton_solve
"""
import json
import sys
import time

import mpmath as mp
import numpy as np

from ._hpkernel import Solver

SOLVER_DIR = "/home/claude/dss-residual-kernel/solver"
NODES_OLD = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7]


# ------------------------------------------------------------------ double-double helpers (numpy, elementwise)
def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def dd_add(ah, al, bh, bl):
    s, e = two_sum(ah, bh)
    t, f = two_sum(al, bl)
    e = e + t
    h = s + e
    e = e - (h - s)
    e = e + f
    h2 = h + e
    return h2, e - (h2 - h)


def dd_from_mp(v):
    v = list(v)
    hi = np.array([float(x) for x in v])
    lo = np.array([float(x - mp.mpf(h)) for x, h in zip(v, hi)])
    return hi, lo


def dd_norm(h, l):
    return float(np.max(np.abs(h + l))) if len(h) else 0.0


# ------------------------------------------------------------------ grid
def dmat_odd(M, dps=40):
    """Folded d/dtheta for odd fields on the half grid of N = 2M points, in double-double (hi, lo), row-major."""
    with mp.workdps(dps):
        N = 2 * M
        out = []
        for i in range(M):
            for j in range(M):
                v = mp.mpf(0)
                for jj, sgn in ((j, 1), (j + M, -1)):
                    if jj != i:
                        d = (i - jj) * mp.pi / N
                        v += sgn * mp.mpf(0.5) * (-1) ** (i - jj) * mp.cot(d)
                out.append(v)
        hi, lo = dd_from_mp(out)
    return hi.reshape(M, M), lo.reshape(M, M)


def phase_weights(M, dps=40):
    with mp.workdps(dps):
        return dd_from_mp([mp.sin(mp.pi * j / M) for j in range(M)])


def make_solver(D, M, nodes=None, kt=36, kc=40, ks=120, segs=None, threads=0):
    nodes = list(NODES_OLD if nodes is None else nodes)
    with mp.workdps(40):
        nh, nl = dd_from_mp([mp.mpf(D) - 2])
    dh, dl = dmat_odd(M)
    ph, pl = phase_weights(M)
    return Solver(M, float(nh[0]), float(nl[0]), dh, dl, ph, pl, nodes, segs or [], kt, kc, ks, threads)


# ------------------------------------------------------------------ seeds
def resample(F, M):
    """Full-grid samples F (any N0) -> half-grid samples of the N = 2M grid (trigonometric interpolation)."""
    N0 = len(F)
    c = np.fft.rfft(F) / N0
    th = 2 * np.pi * np.arange(M) / (2 * M)
    k = np.arange(len(c))
    w = np.where((k == 0) | ((N0 % 2 == 0) & (k == N0 // 2)), 1.0, 2.0)
    return (w * c[None, :] * np.exp(1j * k[None, :] * th[:, None])).real.sum(1)


def seed_from_f64(path, M, kc=23, nodes=None, kmax=None, kmax_all=None):
    """z (hi, lo = 0) from a converged f64 multiple-shooting solution (spec_long layout, x_K = 0.7 nodes).
    kmax: drop centre-data harmonics above kmax (they are noise in the f64 solutions)."""
    if SOLVER_DIR not in sys.path:
        sys.path.insert(0, SOLVER_DIR)
    from spec_long import make_long
    from spec_c import freeze_steps
    import re
    D = float(re.findall(r"D([0-9.]+)\.npy", path)[0])
    zo = np.load(path)
    nodes = NODES_OLD if nodes is None else nodes
    S = freeze_steps(make_long(kc, D - 2, nodes=nodes, Ks=60), zo[-1])
    parts, _ = S.split(zo[None, :])
    parts = [p[0] for p in parts]
    F = [S.phys(np.atleast_2d(c), p)[0] for c, p in zip(parts, S.parities())]

    def rs(f, cut=None):
        if cut is not None:
            c = np.fft.rfft(f)
            c[cut + 1:] = 0
            f = np.fft.irfft(c, n=len(f))
        return resample(f, M)
    z = [rs(F[0], kmax), rs(F[1], kmax)] + [rs(f, kmax_all) for f in F[2:]] + [np.array([zo[-1]])]
    zh = np.concatenate(z)
    return D, zh, np.zeros_like(zh)


# ------------------------------------------------------------------ Newton
def block_norms(s, Fh, Fl):
    M = s.m
    L = s.size
    F = Fh + Fl
    nb = (L - 1) // (4 * M)
    return [float(np.abs(F[b * 4 * M:(b + 1) * 4 * M]).max()) for b in range(nb)] + [abs(float(F[-1]))]


def rnorm(Fh, Fl):
    """2-norm of a double-double vector (to ~1e-16 relative, enough for step control)"""
    return float(np.sqrt(np.sum((Fh + Fl) ** 2)))


def newton_solve(s, zh, zl, tol=1e-28, it_max=12, chord=0, verbose=True, ls=True, max_halvings=10):
    """Newton with exact Jacobians and backtracking on |F|_2 (Armijo, factor 1/2); optional chord steps with the
    stored LU after full Newton steps. Returns zh, zl, history."""
    hist = []
    t0 = time.time()
    for it in range(it_max):
        dh, dl, Fh, Fl, pmin, pmax = s.newton(zh, zl)
        n0 = rnorm(Fh, Fl)
        nF = dd_norm(Fh, Fl)
        if nF < tol:
            hist.append(dict(it=it, F=nF, dz=dd_norm(dh, dl), Delta=float(zh[-1]), t=round(time.time() - t0, 1)))
            if verbose:
                print(f"  newton {it}: |F| {nF:.2e} converged (last |dz| {dd_norm(dh, dl):.1e}) Delta {zh[-1]:.15f}", flush=True)
            break
        lam = 1.0
        accepted = not ls
        for h in range(max_halvings + 1):
            th, tl = dd_add(zh, zl, dh * lam, dl * lam)
            if not ls:
                break
            Th, Tl = s.residual(th, tl)
            n1 = rnorm(Th, Tl)
            if np.isfinite(n1) and n1 <= (1 - 1e-4 * lam) * n0:
                accepted = True
                break
            lam *= 0.5
        if not accepted:
            hist.append(dict(it=it, F=nF, failed="line search", Delta=float(zh[-1]), t=round(time.time() - t0, 1)))
            if verbose:
                print(f"  newton {it}: |F| {nF:.2e}: no decrease along the Newton step, giving up", flush=True)
            break
        zh, zl = th, tl
        rec = dict(it=it, F=nF, dz=dd_norm(dh, dl), lam=lam, dDelta=float(dh[-1] + dl[-1]) * lam, Delta=float(zh[-1]),
                   piv=(pmin, pmax), t=round(time.time() - t0, 1))
        hist.append(rec)
        if verbose:
            print(f"  newton {it}: |F| {nF:.2e} |dz| {rec['dz']:.2e} lam {lam:.3g} dDelta {rec['dDelta']:+.2e} "
                  f"Delta {zh[-1]:.15f} piv {pmin:.1e}..{pmax:.1e}  t {rec['t']}s", flush=True)
        if lam == 1.0:
            for c in range(chord):
                ch, cl, Fh2, Fl2 = s.chord(zh, zl)
                zh, zl = dd_add(zh, zl, ch, cl)
                if verbose:
                    print(f"    chord {c}: |F| {dd_norm(Fh2, Fl2):.2e} |dz| {dd_norm(ch, cl):.2e}", flush=True)
        if rec["dz"] * lam < 1e-31:
            break
    return zh, zl, hist


def resample_z(zh, M_old, M_new, K):
    """Change the tau grid: every field block resampled (trigonometric interpolation, f64; lo parts dropped)."""
    nf = 4 * (K + 1)
    parity = ["odd", "even"] + ["odd", "odd", "even", "even"] * K + ["odd", "even"]
    out = []
    for j in range(nf):
        f = zh[j * M_old:(j + 1) * M_old]
        full = np.r_[f, -f] if parity[j] == "odd" else np.r_[f, f]
        c = np.fft.rfft(full) / (2 * M_old)
        if M_new < M_old:
            c = c[:M_new + 1]
        th = 2 * np.pi * np.arange(M_new) / (2 * M_new)
        k = np.arange(len(c))
        w = np.where((k == 0) | (k == M_old), 1.0, 2.0)
        out.append((w * c[None, :] * np.exp(1j * k[None, :] * th[:, None])).real.sum(1))
    zn = np.concatenate(out + [zh[-1:]])
    return zn, np.zeros_like(zn)


def save(path, zh, zl, meta):
    np.savez(path, zh=zh, zl=zl, meta=json.dumps(meta))


def load(path):
    d = np.load(path)
    return d["zh"], d["zl"], json.loads(str(d["meta"]))


def renode(s, zh, zl, nodes_old, nodes_new, M, h=0.004):
    """Solution vector for a new node set: each new node state from the old solution (Taylor integration from
    the nearest old node below, or the centre series below the first old node). Centre and horizon data and
    Delta are kept. s: a Solver for the old layout (provides the integrator and the centre series)."""
    K0 = len(nodes_old)
    st = lambda i: (zh[2 * M + 4 * M * i: 2 * M + 4 * M * (i + 1)], zl[2 * M + 4 * M * i: 2 * M + 4 * M * (i + 1)])
    ch = cl = None
    out_h, out_l = [zh[:2 * M]], [zl[:2 * M]]
    for x in nodes_new:
        if x in nodes_old:
            a, b = st(nodes_old.index(x))
        elif x < nodes_old[0]:
            if ch is None:
                ch, cl = s.series(zh, zl, 0)
                ch = (ch + cl).reshape(-1, 4, M)
            v = np.zeros((4, M))
            for k in range(ch.shape[0] - 1, -1, -1):
                v = v * x + ch[k]
            a, b = v.ravel(), np.zeros(4 * M)
        else:
            i = max(j for j in range(K0) if nodes_old[j] <= x)
            sh, sl = st(i)
            nst = max(2, int(np.ceil((x - nodes_old[i]) / h)))
            xs = list(np.linspace(nodes_old[i], x, nst + 1))
            xs[-1] = x
            a, b = s.integrate(zh, zl, sh, sl, xs)
        out_h.append(np.asarray(a)); out_l.append(np.asarray(b))
    K = len(nodes_old)
    tail = slice(2 * M + 4 * M * K, None)
    zn_h = np.concatenate(out_h + [zh[tail]])
    zn_l = np.concatenate(out_l + [zl[tail]])
    return zn_h, zn_l


def predict(hist, D):
    """Lagrange extrapolation of z (f64) from the last up to 4 (D, z) pairs."""
    pts = hist[-4:]
    Ds = [p[0] for p in pts]
    out = np.zeros_like(pts[-1][1])
    for i, (Di, zi) in enumerate(pts):
        w = 1.0
        for j, Dj in enumerate(Ds):
            if j != i:
                w *= (D - Dj) / (Di - Dj)
        out = out + w * zi
    return out


def tangent(s_of_D, s, D, zh, zl, h=1e-7):
    """dz/dD at a converged point: -J^-1 dF/dD with the LU stored in s (from the last Newton call at z) and dF/dD by
    central differences in D (the step points of s are reused)."""
    segs = s.get_segs()
    sp, sm = s_of_D(D + h), s_of_D(D - h)
    sp.set_segs(segs)
    sm.set_segs(segs)
    Fp, Fm = sp.residual(zh, zl), sm.residual(zh, zl)
    dF = ((Fp[0] - Fm[0]) + (Fp[1] - Fm[1])) / (2 * h)
    th, tl = s.lu_solve(-dF, np.zeros_like(dF))
    return th + tl


def hermite(D0, z0, t0, D1, z1, t1, D):
    """cubic Hermite through (D0, z0, z0') and (D1, z1, z1'), evaluated at D (extrapolation)"""
    h = D1 - D0
    u = (D - D0) / h
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * z0 + h10 * h * t0 + h01 * z1 + h11 * h * t1
