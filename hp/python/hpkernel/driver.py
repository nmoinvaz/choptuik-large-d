"""Driver for the double-double solver: set-up (grid, d/dtheta, phase row), seeds from the f64 solutions, Newton.

Double-double vectors are (hi, lo) pairs of float64 arrays. Layout of z (M = half-grid points, K nodes):
    p1, c0 | node_0 (p, m, a, c) | ... | node_{K-1} | m_h, a_h | Delta              length 4M(K+1) + 1

    from hpkernel.driver import make_solver, seed_from_f64, newton_solve
"""
import json
import math
import os
import sys
import time

import mpmath as mp
import numpy as np

from multidouble import dd_add, dd_axpy, dd_from_mp, dd_norm, dd_scale, two_sum  # noqa: F401
from multidouble import two_prod as _two_prod  # noqa: F401

from ._hpkernel import Solver, qd_add_dd

# The double-double helpers above (numpy, elementwise) are multidouble's Python package (github.com/nmoinvaz/multidouble,
# python/; hp/pyproject.toml pins it), the same float64 expressions as before; re-exported here under their old names
# for the scripts that import them from this module.

# The f64 solver's directory (for seed_from_f64): $DSS_SOLVER_DIR, else ../../../solver from this file when the package
# runs from the source tree, else the cloud checkout.
_TREE_SOLVER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "solver"))
SOLVER_DIR = os.environ.get("DSS_SOLVER_DIR") or (
    _TREE_SOLVER if os.path.isdir(_TREE_SOLVER) else "/home/claude/dss-residual-kernel/solver")
NODES_OLD = [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7]
# the kernel's dense n x n paths (jacobian, newton, lm_steps, mode_det, ...) stop at this M; above it only
# Solver.newton_mixed runs, in block storage (python.rs DENSE_M_MAX, notes/HIGH_M.md)
DENSE_M_MAX = 128


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


_PLOW = {}


def plow_odd(M, kf, dps=40):
    """Projector onto the odd harmonics k <= kf of an odd field on the half grid (double-double, row-major)."""
    key = (M, kf)
    if key not in _PLOW:
        with mp.workdps(dps):
            N = 2 * M
            th = [2 * mp.pi * j / N for j in range(M)]
            out = []
            for i in range(M):
                for j in range(M):
                    # full-grid interpolation of an antiperiodic field restricted to odd k <= kf, folded
                    v = mp.mpf(0)
                    for k in range(1, kf + 1, 2):
                        v += 4 * mp.cos(k * (th[i] - th[j])) / N
                    out.append(v)
            _PLOW[key] = tuple(a.reshape(M, M) for a in dd_from_mp(out))
    return _PLOW[key]


def make_solver(D, M, nodes=None, kt=36, kc=40, ks=120, segs=None, threads=0, kfreeze=None, match_node=None,
                horizon_alpha=0):
    """kfreeze: pin the p1 harmonics k > kfreeze to zero and drop the matching of p's harmonics k > kfreeze at the
    horizon block (see model.rs Ctx::freeze). horizon_alpha: Solver.set_horizon_alpha (0, the default, as before; 1
    the horizon series in alpha = a - 1; 2 its control)."""
    nodes = list(NODES_OLD if nodes is None else nodes)
    with mp.workdps(40):
        nh, nl = dd_from_mp([mp.mpf(D) - 2])
    dh, dl = dmat_odd(M)
    ph, pl = phase_weights(M)
    a = b = None
    if kfreeze:
        a, b = plow_odd(M, kfreeze)
    s = Solver(M, float(nh[0]), float(nl[0]), dh, dl, ph, pl, nodes, segs or [], kt, kc, ks, threads, a, b, match_node)
    if horizon_alpha:
        s.set_horizon_alpha(horizon_alpha)
    return s


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
def zadd(s, zh, zl, dh, dl):
    """The state z + dz: dd_add, or the solver's own update where it keeps more parts of the state than (zh, zl)
    (QdhSolver). Every update of a state in the Newton loops below goes through here."""
    f = getattr(s, "zadd", None)
    return f(zh, zl, dh, dl) if f is not None else dd_add(zh, zl, dh, dl)


class QdhSolver:
    """A Solver whose residual has the horizon block in quad-double (Solver.residual_qdh, notes/QUAD_DOUBLE.md
    section 11), for newton_solve and deflated_newton with jac="dd".

    The driver keeps working with double-double states (zh, zl). The two further parts of a state are kept here,
    keyed by the state's bits: zadd makes the state z + dz in quad-double (qd_add_dd), returns its double-double part
    and remembers the rest, and residual, newton and chord look the parts up again (a state not made by zadd, such as
    a predictor's or a loaded one, has them zero). The Jacobian and its LU are the double-double ones at (zh, zl): only
    the residual, and so the step -J^-1 F, sees the extra parts. keep: how many states' parts are remembered (the most
    recent). refine=False is the control (the horizon solves unrefined).
    gcr (None or a dict of gmres_refine options) with horizon (the horizon block's rows, a slice): the step then solves
    J_q dz = -F by GCR (gmres_refine) with the double-double LU as its preconditioner, J_q's products by jvp below,
    whose horizon rows are quad-double accurate; info of the last solve in last_gcr."""

    def __init__(self, s, keep=64, refine=True, horizon=None, gcr=None):
        if gcr is not None and horizon is None:
            raise ValueError("QdhSolver: gcr needs the horizon block's rows")
        self.s = s
        self.keep = keep
        self.refine = refine
        self.horizon = horizon
        self.gcr = gcr
        self.last_gcr = None
        self._parts = {}

    def __getattr__(self, name):
        return getattr(self.s, name)

    @staticmethod
    def _key(zh, zl):
        return np.ascontiguousarray(zh).tobytes() + np.ascontiguousarray(zl).tobytes()

    def parts(self, zh, zl):
        """the third and fourth parts of the state (zh, zl), zero if unknown"""
        p = self._parts.get(self._key(zh, zl))
        if p is None:
            z = np.zeros(len(zh))
            return z, z
        return p

    def zadd(self, zh, zl, dh, dl):
        z2, z3 = self.parts(zh, zl)
        c = np.ascontiguousarray
        a, b, q2, q3 = (np.asarray(v) for v in qd_add_dd(c(zh), c(zl), c(z2), c(z3), c(np.asarray(dh, dtype=float)),
                                                         c(np.asarray(dl, dtype=float))))
        self._parts[self._key(a, b)] = (q2, q3)
        while len(self._parts) > self.keep:
            self._parts.pop(next(iter(self._parts)))
        return a, b

    def _resq(self, q):
        """the residual of a four-part state q (QdSolver: every block in quad-double)"""
        return self.s.residual_qdh(*q, refine=self.refine)

    def residual(self, zh, zl):
        z2, z3 = self.parts(zh, zl)
        c = np.ascontiguousarray
        return self._resq((c(zh), c(zl), c(z2), c(z3)))

    def _step(self, Fh, Fl):
        c = np.ascontiguousarray
        return self.s.lu_solve(c(-np.asarray(Fh)), c(-np.asarray(Fl)))

    def lm_steps(self, zh, zl, mus, fh=None, fl=None):
        """Levenberg-Marquardt steps (Solver.lm_steps: the double-double Jacobian) for the quad-double-horizon F, or
        for the F given (a wrapper's)"""
        if fh is None:
            fh, fl = self.residual(zh, zl)
        c = np.ascontiguousarray
        return self.s.lm_steps(c(zh), c(zl), list(mus), c(np.asarray(fh)), c(np.asarray(fl)))

    def newton(self, zh, zl):
        """the double-double Jacobian and LU at (zh, zl) (Solver.newton), the step from the quad-double-horizon F:
        -J^-1 F, or with gcr the GCR solve of J_q dz = -F preconditioned by that LU"""
        _, _, _, _, pmin, pmax = self.s.newton(zh, zl)
        Fh, Fl = self.residual(zh, zl)
        if self.gcr is None:
            dh, dl = self._step(Fh, Fl)
        else:
            dh, dl, self.last_gcr = gmres_refine(self, zh, zl, -np.asarray(Fh), -np.asarray(Fl), **self.gcr)
        return dh, dl, Fh, Fl, pmin, pmax

    def jvp(self, zh, zl, uh, ul, cut=True):
        """J u at (zh, zl): the rows outside the horizon block from Solver.jvp (the double-double Jacobian's product,
        accurate there), the horizon rows by the central difference (F(z + h u) - F(z - h u)) / 2h of residual_qdh at
        the quad-double states z +- h u, h the power of two with h max|u| about 2^-66 max(1, max|z|): the differences'
        rounding, about 1e-64 G / (h |u|), and their truncation error, about (h |u|)^2 times F's third derivative, are
        both far below the double-double Jacobian's error there, about 1e-32 G. Returns (jh, jl)."""
        out = self.s.jvp(zh, zl, uh, ul, cut=cut)      # (jh, jl) or (jh, jl, fh, fl), as the kernel has it
        jh, jl = np.asarray(out[0]), np.asarray(out[1])
        um = float(np.abs(np.asarray(uh) + np.asarray(ul)).max())
        if um == 0:
            return jh, jl
        zs = max(1.0, float(np.abs(np.asarray(zh)).max()))
        h = 2.0 ** math.floor(math.log2(2.0 ** -66 * zs / um))
        z2, z3 = self.parts(zh, zl)
        c = np.ascontiguousarray
        uh, ul = np.asarray(uh, dtype=float), np.asarray(ul, dtype=float)
        F = []
        for sg in (h, -h):
            q = [c(np.asarray(v)) for v in qd_add_dd(c(zh), c(zl), c(z2), c(z3), c(sg * uh), c(sg * ul))]
            F.append([np.asarray(v) for v in self._resq(q)])
        dh, dl = dd_add(F[0][0], F[0][1], -F[1][0], -F[1][1])
        hb = self.horizon
        jh, jl = jh.copy(), jl.copy()
        jh[hb], jl[hb] = dh[hb] / (2 * h), dl[hb] / (2 * h)
        return jh, jl

    def chord(self, zh, zl):
        Fh, Fl = self.residual(zh, zl)
        dh, dl = self._step(Fh, Fl)
        return dh, dl, Fh, Fl

    def _no(self, *a, **k):
        raise NotImplementedError("QdhSolver: only the double-double Jacobian (jac='dd'); no mixed steps")

    newton_mixed = _no      # (lm_steps is defined above: the double-double Jacobian with this F)


class QdSolver(QdhSolver):
    """A Solver whose residual is quad-double in every block (Solver.residual_qd): as QdhSolver otherwise. With gcr
    (gmres_refine options) the step solves J_q dz = -F by GCR preconditioned by the double-double LU, J_q's products by
    the central difference of residual_qd in every row but the phase row (linear, exact in double-double). Near the
    double-double floor at large D the Jacobian's weakest singular value lies below the double-double residual's
    rounding (u sigma_max), so the double-double LU alone cannot resolve that direction; these products can (README,
    "Why the solver stalls")."""

    def __init__(self, s, keep=64, refine=True, gcr=None):
        super().__init__(s, keep=keep, refine=refine, horizon=slice(0, s.size - 1) if gcr is not None else None, gcr=gcr)

    def _resq(self, q):
        return self.s.residual_qd(*q, refine=self.refine)


def block_norms(s, Fh, Fl):
    M = s.m
    L = s.size
    F = Fh + Fl
    nb = (L - 1) // (4 * M)
    return [float(np.abs(F[b * 4 * M:(b + 1) * 4 * M]).max()) for b in range(nb)] + [abs(float(F[-1]))]


def rnorm(Fh, Fl):
    """2-norm of a double-double vector (to ~1e-16 relative, enough for step control)"""
    return float(np.sqrt(np.sum((Fh + Fl) ** 2)))


# ------------------------------------------------------------------ mixed-precision steps at large D
# The Jacobian variants of newton_step / newton_solve / deflated_newton (notes/LARGE_D_MIXED.md):
#   dd     the exact double-double Jacobian and LU (Solver.newton)
#   f64    the f64 Jacobian, its f64 LU, the solve in double-double (Solver.newton_mixed; notes/MIXED_PRECISION.md)
#   f64h   f64 with the horizon block's columns in double-double (newton_mixed(horizon_dd=HORIZON_DD))
#   f64g   f64, then the correction loop: GMRES on J M^-1 with M the f64 LU and exact products J v (Solver.jvp)
#   f64hg  f64h, then the loop
JACS = ("dd", "f64", "f64h", "f64g", "f64hg")
# Solver.jacobian_f64 / newton_mixed horizon_dd flags of the horizon fix (model.rs HDD_BLOCK: the horizon block's
# columns from the double-double lanes, rounded to f64; the cheapest variant that steers at D = 50-60, largeD_horizon.py)
HORIZON_DD = 4
# defaults of the correction loop (gmres_refine): 3-6 products per step away from the floor, up to ~30 near it at
# D >= 40, where 12 sent the deflated Newton to exact steps (dd_finish) too often (notes/LARGE_D_MIXED.md section 5)
GMRES = dict(max_products=30, step_tol=1e-10, confirm=1)


def gmres_refine(s, zh, zl, bh, bl, max_products=30, step_tol=1e-10, confirm=1, cut=True, verbose=False, ref=None):
    """Solve J x = b (b = bh + bl, double-double) for the exact Jacobian J at z: GMRES on J M^-1 (right
    preconditioning; Saad and Schultz, SIAM J. Sci. Stat. Comput. 7 (1986) 856, doi:10.1137/0907058) in its GCR form
    (Eisenstat, Elman and Schultz, SIAM J. Numer. Anal. 20 (1983) 345, doi:10.1137/0720023), M the LU stored
    in s (the mixed step's f64 LU), the products J u exact in double-double (Solver.jvp; cut: its series stopped at
    double-double resolution). Each iteration takes the preconditioned residual u = M^-1 r as the new direction, w = J u
    (one product), orthogonalizes w against the earlier w's (twice), applying the same coefficients to u, and updates
    x += c u, r -= c w with c = w.r. The u, w, x and r are double-double and updated with the same coefficients, so r
    stays the residual b - J x to double-double rounding without recomputing it (only the coefficients are f64).
    In exact arithmetic the iterates are GMRES's.
    Stopping is on the change of x, not on the residual (an error along a weak direction of J barely shows in the
    residual; notes/MIXED_PRECISION.md section 8): the loop ends after `confirm` iterations in a row with
    |c u| <= step_tol |x|, or after max_products products. The first iteration's u is M^-1 b, the plain mixed step.
    ref (optional, a double-double vector): the exact solution, for the checks: info["errors"] gets |x - ref| / |ref|
    after each iteration.
    Returns (xh, xl, info): info = dict(products, changes, converged, t_prod, t_solve, errors)."""
    n = s.size
    xh, xl = np.zeros(n), np.zeros(n)
    rh, rl = np.asarray(bh, dtype=float).copy(), np.asarray(bl, dtype=float).copy()
    U, W, W64 = [], [], []
    info = dict(products=0, changes=[], converged=False, t_prod=0.0, t_solve=0.0)
    if ref is not None:
        info["errors"] = []
        refv = np.asarray(ref[0]) + np.asarray(ref[1])
    run = 0
    while info["products"] < max_products:
        t = time.time()
        uh, ul = (np.asarray(a) for a in s.lu_solve(np.ascontiguousarray(rh), np.ascontiguousarray(rl)))
        info["t_solve"] += time.time() - t
        t = time.time()
        wh, wl = (np.asarray(a) for a in s.jvp(zh, zl, uh, ul, cut=cut)[:2])
        info["t_prod"] += time.time() - t
        info["products"] += 1
        for _ in range(2):
            for (Uh, Ul), (Wh, Wl), w64 in zip(U, W, W64):
                beta = float(w64 @ (wh + wl))
                wh, wl = dd_axpy(-beta, Wh, Wl, wh, wl)
                uh, ul = dd_axpy(-beta, Uh, Ul, uh, ul)
        nw = float(np.linalg.norm(wh + wl))
        if not nw > 0:
            break
        wh, wl = dd_scale(wh, wl, 1.0 / nw)
        uh, ul = dd_scale(uh, ul, 1.0 / nw)
        w64 = wh + wl
        c = float(w64 @ (rh + rl))
        xh, xl = dd_axpy(c, uh, ul, xh, xl)
        rh, rl = dd_axpy(-c, wh, wl, rh, rl)
        U.append((uh, ul))
        W.append((wh, wl))
        W64.append(w64)
        ch = abs(c) * float(np.linalg.norm(uh + ul)) / max(float(np.linalg.norm(xh + xl)), 1e-300)
        info["changes"].append(ch)
        if ref is not None:
            info["errors"].append(float(np.linalg.norm((xh - ref[0]) + (xl - ref[1])) / np.linalg.norm(refv)))
        if verbose:
            print(f"    gcr {len(U)}: change {ch:.1e}" + (f", error {info['errors'][-1]:.1e}" if ref is not None else ""),
                  flush=True)
        run = run + 1 if (len(U) > 1 and ch <= step_tol) else 0
        if run >= confirm:
            info["converged"] = True
            break
    return xh, xl, info


def newton_step(s, zh, zl, jac="dd", mixed=None, gmres=None, verbose=False):
    """One Newton step dz = -J^-1 F(z) with the Jacobian variant jac (JACS above). The LU stored in s is the step's
    factorization (the double-double one for dd, else the f64 LU of the f64 Jacobian, with the horizon block in
    double-double for f64h / f64hg), which chord, lu_solve, lu_solve_t and weak_subspace then use. mixed: further
    options of Solver.newton_mixed; gmres: options of gmres_refine (GMRES above, for f64g / f64hg).
    Returns dh, dl, Fh, Fl, pmin, pmax, info: info = dict(t_res, t_jac, t_lu, t_solve, t_prod, products) in seconds."""
    if jac not in JACS:
        raise ValueError(f"jac must be one of {JACS}, not {jac!r}")
    if jac == "dd":
        dh, dl, Fh, Fl, pmin, pmax = s.newton(zh, zl)
    else:
        kw = dict(mixed or {})
        if jac in ("f64h", "f64hg"):
            kw["horizon_dd"] = HORIZON_DD
        dh, dl, Fh, Fl, pmin, pmax = s.newton_mixed(zh, zl, **kw)
    st = getattr(s, "step_timing", None)       # (absent in builds before notes/LARGE_D_MIXED.md)
    tr, tj, tf, ts = st() if st else (float("nan"),) * 4
    info = dict(t_res=tr, t_jac=tj, t_lu=tf, t_solve=ts, t_prod=0.0, products=0)
    if jac in ("f64g", "f64hg"):
        g = dict(GMRES, **(gmres or {}))
        dh, dl, gi = gmres_refine(s, zh, zl, -np.asarray(Fh), -np.asarray(Fl), verbose=verbose, **g)
        info.update(t_prod=gi["t_prod"], t_solve=ts + gi["t_solve"], products=gi["products"],
                    gmres_converged=gi["converged"], changes=[float(f"{c:.2e}") for c in gi["changes"]])
    return dh, dl, Fh, Fl, pmin, pmax, info


def exact_last_step(s, zh, zl, tol):
    """At a point a mixed-precision Newton reached |F| < tol: one exact Newton step (Solver.newton), taken if |F| stays
    below tol. It is quadratic where the mixed steps converge linearly, so it removes the remainder they leave along
    the weakly determined directions (which carry Delta's sensitivity at large D and which an f64 LU does not resolve),
    and it leaves the exact LU stored for the sensitivities. Returns (zh, zl, record fields)."""
    dh, dl, Fh, Fl, pmin, pmax = s.newton(zh, zl)
    tr, tj, tf, ts = s.step_timing()
    th, tl = zadd(s, zh, zl, dh, dl)
    Th, Tl = s.residual(th, tl)
    n1 = dd_norm(Th, Tl)
    out = dict(exact_last=True, t_jac_last=tj, F_last=n1)
    if n1 < tol:
        zh, zl = th, tl
        out["dDelta_est"] = delta_correction(s, Th, Tl)
    else:
        out["dDelta_est"] = delta_correction(s, Fh, Fl)
        out["exact_last_rejected"] = True
    return zh, zl, out


def newton_solve(s, zh, zl, tol=1e-28, it_max=12, chord=0, verbose=True, ls=True, max_halvings=10, jac="dd", mixed=None,
                 gmres=None, dd_finish=True, dd_last=False):
    """Newton with exact Jacobians and backtracking on |F|_2 (Armijo, factor 1/2); optional chord steps with the
    stored LU after full Newton steps. Returns zh, zl, history.
    jac="f64": mixed precision, each step with the f64 Jacobian (s.newton_mixed(zh, zl, **mixed), mixed a dict of its
    options or None; the residual stays double-double, so the solution is the same; notes/MIXED_PRECISION.md). A step
    along which the line search finds no decrease is then retried once with the exact Jacobian (s.newton), and once
    |F| < tol the step computed at that point is still taken if |F| stays below tol: the linear convergence leaves an
    error of about that step's size in the near-null directions, which |F| does not see.
    jac="f64h", "f64g", "f64hg": the large-D variants of newton_step (notes/LARGE_D_MIXED.md), with the same two
    rules; gmres: options of gmres_refine. Their history records carry the step's time split (t_res, t_jac, t_lu,
    t_solve, t_prod) and the products. With dd_finish (their default) the remaining steps are exact once a mixed step
    is cut below lam = 1/4, needs the exact retry, or its correction loop ends without converging (as in
    deflated_newton). dd_last: see deflated_newton."""
    if jac not in JACS:
        raise ValueError(f"jac must be one of {JACS}, not {jac!r}")
    hist = []
    t0 = time.time()

    def search(zh, zl, dh, dl, n0):
        lam = 1.0
        accepted = not ls
        for h in range(max_halvings + 1):
            th, tl = zadd(s, zh, zl, dh * lam, dl * lam)
            if not ls:
                break
            Th, Tl = s.residual(th, tl)
            n1 = rnorm(Th, Tl)
            if np.isfinite(n1) and n1 <= (1 - 1e-4 * lam) * n0:
                accepted = True
                break
            lam *= 0.5
        return accepted, lam, th, tl

    cur = jac
    for it in range(it_max):
        # (jac "dd": s.newton; "f64": s.newton_mixed(**mixed), as before; info: the step's seconds by part)
        dh, dl, Fh, Fl, pmin, pmax, info = newton_step(s, zh, zl, cur, mixed, gmres)
        if cur != jac:
            info = dict(info, jac=cur)
        n0 = rnorm(Fh, Fl)
        nF = dd_norm(Fh, Fl)
        if nF < tol:
            rec = dict(it=it, F=nF, dz=dd_norm(dh, dl), Delta=float(zh[-1]), piv=(pmin, pmax), t=round(time.time() - t0, 1))
            rec.update(info)
            polish = ""
            if dd_last and cur not in ("dd", "f64") and s.m <= DENSE_M_MAX:
                zh, zl, extra = exact_last_step(s, zh, zl, tol)
                rec.update(extra)
                polish = f"; one exact step, |F| {extra['F_last']:.2e}"
            elif cur != "dd":
                # Linear convergence leaves an error of about this step's size in the weakly determined (near-null)
                # directions, which |F| does not see. The step is already computed: take it if |F| stays below tol.
                th, tl = zadd(s, zh, zl, dh, dl)
                Th, Tl = s.residual(th, tl)
                n1 = dd_norm(Th, Tl)
                if n1 < tol:
                    zh, zl = th, tl
                    rec.update(polished=True, F_polished=n1)
                    polish = f"; last step taken, |F| {n1:.2e}"
            hist.append(rec)
            if verbose:
                print(f"  newton {it}: |F| {nF:.2e} converged (last |dz| {dd_norm(dh, dl):.1e}) Delta {zh[-1]:.15f}{polish}",
                      flush=True)
            break
        accepted, lam, th, tl = search(zh, zl, dh, dl, n0)
        if not accepted and cur != "dd":
            if s.m <= DENSE_M_MAX:
                if verbose:
                    print(f"  newton {it}: |F| {nF:.2e}: no decrease along the f64-Jacobian step; retrying with the exact "
                          f"Jacobian", flush=True)
                dh, dl, Fh, Fl, pmin, pmax = s.newton(zh, zl)
                info["retry_dd"] = True
            else:
                # no dense exact Jacobian above DENSE_M_MAX: retry with the f64 tangents on a double-double base
                # trajectory (the more accurate f64 Jacobian, 'dd4'), in block storage
                if verbose:
                    print(f"  newton {it}: |F| {nF:.2e}: no decrease along the f64-Jacobian step; retrying with the dd4 "
                          f"Jacobian (M > {DENSE_M_MAX}: no exact dense Jacobian)", flush=True)
                dh, dl, Fh, Fl, pmin, pmax = s.newton_mixed(zh, zl, **dict(mixed or {}, tangents=4, dd_base=True))
            accepted, lam, th, tl = search(zh, zl, dh, dl, n0)
        if not accepted:
            hist.append(dict(it=it, F=nF, failed="line search", Delta=float(zh[-1]), t=round(time.time() - t0, 1)))
            if verbose:
                print(f"  newton {it}: |F| {nF:.2e}: no decrease along the Newton step, giving up", flush=True)
            break
        zh, zl = th, tl
        rec = dict(it=it, F=nF, dz=dd_norm(dh, dl), lam=lam, dDelta=float(dh[-1] + dl[-1]) * lam, Delta=float(zh[-1]),
                   piv=(pmin, pmax), t=round(time.time() - t0, 1))
        rec.update(info)
        hist.append(rec)
        if (dd_finish and cur in ("f64h", "f64g", "f64hg") and s.m <= DENSE_M_MAX
                and (lam < 0.25 or info.get("retry_dd") or (info.get("products") and not info.get("gmres_converged", True)))):
            cur = "dd"
            rec["switch_dd"] = True
        if verbose:
            print(f"  newton {it}: |F| {nF:.2e} |dz| {rec['dz']:.2e} lam {lam:.3g} dDelta {rec['dDelta']:+.2e} "
                  f"Delta {zh[-1]:.15f} piv {pmin:.1e}..{pmax:.1e}  t {rec['t']}s", flush=True)
        if lam == 1.0:
            for c in range(chord):
                ch, cl, Fh2, Fl2 = s.chord(zh, zl)
                zh, zl = zadd(s, zh, zl, ch, cl)
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


def save(path, zh, zl, meta, parts=None):
    """parts: the state's third and fourth parts (QdhSolver.parts), saved as z2, z3 (load ignores them; load_parts)"""
    if parts is None:
        np.savez(path, zh=zh, zl=zl, meta=json.dumps(meta))
    else:
        np.savez(path, zh=zh, zl=zl, meta=json.dumps(meta), z2=parts[0], z3=parts[1])


def load_parts(path):
    """a saved state's third and fourth parts (zero for a double-double state)"""
    d = np.load(path)
    if "z2" in d:
        return d["z2"], d["z3"]
    z = np.zeros(len(d["zh"]))
    return z, z


def load(path):
    d = np.load(path)
    return d["zh"], d["zl"], json.loads(str(d["meta"]))


def renode(s, zh, zl, nodes_old, nodes_new, M, h=0.004, beyond="horizon"):
    """Solution vector for a new node set: each new node state from the old solution (Taylor integration from
    the nearest old node below, the centre series below the first old node, or the horizon series beyond the last
    old node). Centre and horizon data and Delta are kept. s: a Solver for the old layout (provides the integrator
    and both series). beyond="integrate": a node beyond the last old node by Taylor integration outward from it
    instead (the old path; it can pass the horizon series' nearest singularity, D 3.075, M 256: NaN at 0.975)."""
    K0 = len(nodes_old)
    st = lambda i: (zh[2 * M + 4 * M * i: 2 * M + 4 * M * (i + 1)], zl[2 * M + 4 * M * i: 2 * M + 4 * M * (i + 1)])
    ch = cl = hh = None
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
        elif x > nodes_old[-1] and beyond == "horizon":
            # the horizon series at u = 1 - x, in its scaled variable u / sigma (the unscaled coefficients of a high
            # order overflow f64 where the series' radius is small)
            if hh is None:
                hh, hl = s.series(zh, zl, 1, True)
                hh = (hh + hl).reshape(-1, 4, M)
                sig = s.series_scale(1)
            t = (1.0 - x) / sig
            v = np.zeros((4, M))
            for k in range(hh.shape[0] - 1, -1, -1):
                v = v * t + hh[k]
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


def lm_solve(s, zh, zl, tol=1e-27, it_max=20, mus=(0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0), verbose=True,
             newton_after=1e-12):
    """Levenberg-Marquardt (Marquardt scaling) in double-double: each iteration tries every mu and keeps the
    step with the smallest |F|_2; once |F| < newton_after it hands over to newton_solve. Returns zh, zl, history."""
    hist = []
    t0 = time.time()
    for it in range(it_max):
        steps, Fh, Fl = s.lm_steps(zh, zl, list(mus))
        n0 = rnorm(Fh, Fl)
        best = None
        for mu, (dh, dl) in zip(mus, steps):
            th, tl = zadd(s, zh, zl, np.asarray(dh), np.asarray(dl))
            Th, Tl = s.residual(th, tl)
            n1 = rnorm(Th, Tl)
            if np.isfinite(n1) and (best is None or n1 < best[0]):
                best = (n1, mu, th, tl, dd_norm(np.asarray(dh), np.asarray(dl)))
        hist.append(dict(it=it, F=dd_norm(Fh, Fl), lm=True, t=round(time.time() - t0, 1)))
        if best is None or best[0] >= n0:
            if verbose:
                print(f"  lm {it}: |F|_2 {n0:.2e}: no mu decreases it", flush=True)
            break
        n1, mu, zh, zl, dz = best
        if verbose:
            print(f"  lm {it}: |F|_2 {n0:.2e} -> {n1:.2e} (mu {mu:g}, |dz| {dz:.1e}) Delta {zh[-1]:.15f} t {time.time() - t0:.0f}s",
                  flush=True)
        if n1 < newton_after:
            zh, zl, h2 = newton_solve(s, zh, zl, tol=tol, it_max=8, verbose=verbose)
            return zh, zl, hist + h2
    return zh, zl, hist


def weak_subspace(s, w, iters=8, seed=0):
    """Orthonormal basis (L, w) of the w weakest right singular directions of the Jacobian whose LU is stored in s
    (inverse subspace iteration on (J^T J)^-1, f64 directions)."""
    L = s.size
    V = np.linalg.qr(np.random.default_rng(seed).standard_normal((L, w)))[0]
    for _ in range(iters):
        V = np.linalg.qr(_normal_solve(s, V))[0]
    return V


def _normal_solve(s, V):
    """(J^T J)^-1 V column by column with the stored LU, hi parts only: y = J^-T v, then J^-1 y. All columns at once
    (Solver.lu_solve_many: 8 per lane vector, in parallel; bitwise the per-column solves) when the kernel has it."""
    L, w = V.shape
    if hasattr(s, "lu_solve_many"):
        Z = np.zeros((w, L))
        Y = np.asarray(s.lu_solve_many(np.ascontiguousarray(V.T), Z, transpose=True)[0])
        return np.asarray(s.lu_solve_many(np.ascontiguousarray(Y), Z)[0]).T.copy()
    zl = np.zeros(L)
    W = np.empty_like(V)
    for j in range(w):
        y = np.asarray(s.lu_solve_t(np.ascontiguousarray(V[:, j]), zl)[0])
        W[:, j] = np.asarray(s.lu_solve(np.ascontiguousarray(y), zl)[0])
    return W


def junk_subspace(s, w, delta_max=1e-9, iters=8):
    """Rayleigh-Ritz on the w weakest right singular directions; keep those whose Delta component is below delta_max
    (the near-null centre directions, which the equations fix only to tol / sigma and Delta does not feel).
    Returns (orthonormal basis, sigmas kept, sigmas dropped)."""
    L = s.size
    V = weak_subspace(s, w, iters=iters)
    W = _normal_solve(s, V)
    H = V.T @ W
    ev, U = np.linalg.eigh((H + H.T) / 2)
    X = V @ U
    X /= np.linalg.norm(X, axis=0)
    sig = 1 / np.sqrt(np.maximum(ev, 1e-300))
    keep = np.abs(X[-1, :]) < delta_max
    return np.linalg.qr(X[:, keep])[0] if keep.any() else np.zeros((L, 0)), sig[keep], sig[~keep]


def delta_correction(s, Fh, Fl):
    """Linearised Delta correction -y.F, y = J^-T e_Delta, with the LU stored by the last s.newton call."""
    L = s.size
    e = np.zeros(L)
    e[-1] = 1.0
    yh, yl = s.lu_solve_t(e, np.zeros(L))
    return float(-(np.asarray(yh) + np.asarray(yl)) @ (np.asarray(Fh) + np.asarray(Fl)))


def deflated_newton(s, zh, zl, tol=1e-27, it_max=20, w=24, verbose=True, max_halvings=12, junk_only=False,
                    delta_max=1e-9, delta_tol=1e-14, jac="dd", mixed=None, gmres=None, dd_finish=True, dd_last=False,
                    accept_stall=None):
    """Newton in phases, all with backtracking on |F|_2:
    1. 'all': the step J^-1(-F) with its component in the w weakest right singular directions removed, so the
       well-determined part converges without its second-order error leaking into them;
    2. 'sel' (junk_only): only the weak directions without Delta content (|Delta component| < delta_max) are removed,
       so the selection directions converge while the near-null centre ("junk") directions stay on the predictor's
       smooth trend (the equations fix them only to tol / sigma, and Delta does not feel them);
    3. 'full': plain Newton, if the previous phase stalls above tol.
    A phase ends when |F| stops dropping by a factor 0.3 per step or a step finds no decrease.
    jac="f64": the steps and the stored LU (from which the weak directions are computed) come from the mixed-precision
    step s.newton_mixed(**mixed) (block storage above M = 128) instead of the exact Jacobian.
    jac="f64h", "f64g", "f64hg" (the large-D variants of newton_step, notes/LARGE_D_MIXED.md; gmres: options of
    gmres_refine): likewise the weak subspace comes from the step's stored f64 LU; at convergence the step already
    computed is taken if |F| stays below tol (as newton_solve does: the convergence is linear); and with the correction
    loop (f64g, f64hg) the linearised Delta correction is the Delta component of the refined step (equal to
    -dDelta/dF . F for the exact J), since the stored LU is not the Jacobian the step solves with. "dd" and "f64" keep
    the loop of wip/high-m 2b3b253 unchanged.
    dd_finish (f64h, f64g, f64hg): in the plain-Newton phase, once a mixed step is cut below lam = 1/4 by the line
    search or finds no decrease, or its correction loop ends without converging, the remaining steps are exact
    (dd). Near the double-double floor at large D the residual lies in the weakest directions, which an f64 LU does
    not resolve (notes/LARGE_D_MIXED.md section 4); the exact steps finish there, and leave the exact LU stored for
    the sensitivities. Not above M = DENSE_M_MAX (no dense exact Jacobian there).
    dd_last (f64h, f64g, f64hg; default off): once |F| < tol, one exact step instead of taking the mixed step
    (exact_last_step): at D ~ 60 the mixed iterations stop just below tol with a residual along the weakest directions,
    which leaves Delta off by 1e-16..1e-15 where the exact Newton's is 1e-18 (notes/LARGE_D_MIXED.md section 4); the
    exact step costs one double-double Jacobian per solve and also leaves the exact LU for the sensitivities.
    accept_stall (F_max, dDelta_max), default None: when the phases end without |F| < tol, the point is still accepted
    if |F| <= F_max and the linearised Delta correction there (delta_correction with the LU at the final point) is at
    most dDelta_max: the stall is in directions Delta does not feel (the quad-double-horizon pilot, comms 2026-10-26).
    """
    if jac not in JACS:
        raise ValueError(f"jac must be one of {JACS}, not {jac!r}")
    hist = []
    t0 = time.time()
    # with junk_only there is no plain-Newton fallback: it would let the junk directions move to whatever tol / sigma
    # allows, and they then wreck the centre series; a stalled step is reported as failed (the scan halves dD)
    phases = ["all", "sel"] if junk_only else ["all", "full"]
    ph = 0
    V = {}
    n_prev = None
    cur = jac      # the variant of the next step (dd_finish switches it to dd)

    def dcorr(Fh, Fl, dz_delta):
        # the linearised Delta correction -dDelta/dF . F: by the stored LU's transpose solve (as before), or with the
        # correction loop the Delta component of the refined step (the same quantity for the exact J)
        return float(dz_delta) if cur in ("f64g", "f64hg") else delta_correction(s, Fh, Fl)

    for it in range(it_max):
        # (dd: s.newton; f64: s.newton_mixed(**mixed), as in 2b3b253; info: the step's seconds by part)
        dh, dl, Fh, Fl, pmin, pmax, info = newton_step(s, zh, zl, cur, mixed, gmres)
        info = dict(info, jac=cur)
        dz_delta = float(dh[-1] + dl[-1])
        nF = dd_norm(Fh, Fl)
        n0 = rnorm(Fh, Fl)
        mode = phases[ph]
        if nF < tol:
            rec = dict(it=it, F=nF, Delta=float(zh[-1]), piv=(pmin, pmax), mode=mode, accepted=True,
                       dDelta_est=dcorr(Fh, Fl, dz_delta), t=round(time.time() - t0, 1))
            polish = ""
            rec.update(info)
            if dd_last and cur not in ("dd", "f64") and s.m <= DENSE_M_MAX:
                zh, zl, extra = exact_last_step(s, zh, zl, tol)
                rec.update(extra)
                polish = f"; one exact step, |F| {extra['F_last']:.2e}"
            elif cur not in ("dd", "f64"):
                th, tl = zadd(s, zh, zl, dh, dl)
                Th, Tl = s.residual(th, tl)
                n1 = dd_norm(Th, Tl)
                if n1 < tol:
                    zh, zl = th, tl
                    rec.update(polished=True, F_polished=n1)
                    polish = f"; last step taken, |F| {n1:.2e}"
            hist.append(rec)
            if verbose:
                print(f"  dn {it}: |F| {nF:.2e} converged, Delta {zh[-1]:.15f}{polish}", flush=True)
            break
        if junk_only and mode == "sel" and n_prev is not None and n0 > 0.3 * n_prev:
            # stalled with the junk frozen: accept if Delta no longer moves (the floor is junk residual)
            dd = dcorr(Fh, Fl, dz_delta)
            hist.append(dict(it=it, F=nF, Delta=float(zh[-1]), piv=(pmin, pmax), mode=mode, dDelta_est=dd,
                             accepted=abs(dd) < delta_tol, t=round(time.time() - t0, 1)))
            if verbose:
                print(f"  dn {it}: stalled at |F| {nF:.1e} with junk frozen; Delta correction {dd:+.1e} -> "
                      f"{'accepted' if abs(dd) < delta_tol else 'failed'}", flush=True)
            break
        if n_prev is not None and n0 > 0.3 * n_prev and ph < len(phases) - 1:
            ph += 1
            mode = phases[ph]
            if verbose:
                print(f"  dn: phase '{phases[ph - 1]}' stalled at |F| {nF:.1e}; phase '{mode}'", flush=True)
        if mode in ("all", "sel"):
            if mode not in V:
                if mode == "all":
                    V[mode] = weak_subspace(s, w)
                else:
                    V[mode], sk, sd = junk_subspace(s, w, delta_max)
                    if verbose:
                        print(f"  dn: {V[mode].shape[1]} junk directions frozen (sigma {sk.min() if len(sk) else 0:.0e}.."
                              f"{sk.max() if len(sk) else 0:.0e}), {len(sd)} weak ones with Delta content solved", flush=True)
            d = dh + dl
            d = d - V[mode] @ (V[mode].T @ d)
            dh, dl = d, np.zeros_like(d)
        lam, accepted = 1.0, False
        for _ in range(max_halvings + 1):
            th, tl = zadd(s, zh, zl, dh * lam, dl * lam)
            Th, Tl = s.residual(th, tl)
            n1 = rnorm(Th, Tl)
            if np.isfinite(n1) and n1 <= (1 - 1e-4 * lam) * n0:
                accepted = True
                break
            lam *= 0.5
        if verbose:
            print(f"  dn {it} [{mode}]: |F| {nF:.2e} -> {dd_norm(Th, Tl) if accepted else float('nan'):.2e} lam {lam:.3g} "
                  f"Delta {th[-1]:.15f} t {time.time() - t0:.0f}s"
                  + (f" ({info['products']} products)" if info and info.get("products") else ""), flush=True)
        rec = dict(it=it, F=nF, mode=mode, lam=lam, Delta=float(th[-1]), piv=(pmin, pmax), t=round(time.time() - t0, 1))
        rec.update(info)
        hist.append(rec)
        if (dd_finish and cur in ("f64h", "f64g", "f64hg") and mode == "full" and s.m <= DENSE_M_MAX
                and (not accepted or lam < 0.25 or (info.get("products") and not info.get("gmres_converged", True)))):
            cur = "dd"
            rec["switch_dd"] = True
            if verbose:
                print(f"  dn {it}: the remaining steps exact (dd)", flush=True)
            if not accepted:
                continue
        if not accepted:
            if junk_only and mode == "sel":
                dd = dcorr(Fh, Fl, dz_delta)
                hist.append(dict(it=it, F=nF, Delta=float(zh[-1]), piv=(pmin, pmax), mode=mode, dDelta_est=dd,
                                 accepted=abs(dd) < delta_tol, t=round(time.time() - t0, 1)))
                if verbose:
                    print(f"  dn {it}: no decrease with junk frozen at |F| {nF:.1e}; Delta correction {dd:+.1e}", flush=True)
                break
            if ph == len(phases) - 1:
                break
            ph += 1
            n_prev = None
            continue
        zh, zl = th, tl
        n_prev = n0
    done = bool(hist) and (hist[-1]["F"] < tol or (hist[-1].get("accepted") is True and "dDelta_est" in hist[-1]))
    if accept_stall is not None and jac == "dd" and not done:
        _, _, Fh, Fl, pmin, pmax = s.newton(zh, zl)
        nF = dd_norm(Fh, Fl)
        dd = delta_correction(s, Fh, Fl)
        ok = nF < tol or (nF <= accept_stall[0] and abs(dd) <= accept_stall[1])
        hist.append(dict(it=len(hist), F=nF, Delta=float(zh[-1]), piv=(pmin, pmax), mode="stall", dDelta_est=dd,
                         accepted=ok, accept_stall=True, t=round(time.time() - t0, 1)))
        if verbose:
            print(f"  dn: stopped at |F| {nF:.2e}; Delta correction {dd:+.1e} -> "
                  f"{'accepted (accept_stall)' if ok else 'not accepted'}", flush=True)
    return zh, zl, hist