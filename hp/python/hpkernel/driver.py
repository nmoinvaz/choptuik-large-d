"""Driver for the double-double solver: set-up (grid, d/dtheta, phase row), seeds from the f64 solutions, Newton.

Double-double vectors are (hi, lo) pairs of float64 arrays. Layout of z (M = half-grid points, K nodes):
    p1, c0 | node_0 (p, m, a, c) | ... | node_{K-1} | m_h, a_h | Delta              length 4M(K+1) + 1

    from hpkernel.driver import make_solver, seed_from_f64, newton_solve
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

from multidouble import dd_add, dd_axpy, dd_dot, dd_from_mp, dd_norm, dd_scale, gcr, qd_axpy, two_sum  # noqa: F401
from multidouble import two_prod as _two_prod  # noqa: F401

from ._hpkernel import Solver

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
    Returns (xh, xl, info): info = dict(products, changes, converged, t_prod, t_solve, errors).
    The loop is multidouble's gcr with coef="f64" (this function's own loop, moved there; bitwise the same)."""
    return gcr(lambda uh, ul: s.jvp(zh, zl, uh, ul, cut=cut)[:2], s.lu_solve, bh, bl, max_products=max_products,
               step_tol=step_tol, confirm=confirm, ref=ref, verbose=verbose, coef="f64")


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
    th, tl = dd_add(zh, zl, dh, dl)
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
            th, tl = dd_add(zh, zl, dh * lam, dl * lam)
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
                th, tl = dd_add(zh, zl, dh, dl)
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
            th, tl = dd_add(zh, zl, np.asarray(dh), np.asarray(dl))
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
    """Linearised Delta correction -y.F, y = J^-T e_Delta, with the LU stored by the last s.newton call. The dot
    product is multidouble's dd_dot (double-double, to about (L / 8 + 12) 2^-106 of sum |y_i F_i|), the result
    rounded to f64.
    Until 2026-10-25 it was an f64 dot product of the rounded vectors, whose error grows with sum |y_i F_i| (|y| reaches
    1e15 at large D); for a residual that cancels in y.F (a planted one, far from convergence) it lost the result (comms
    2026-10-25, T57). At T48's 319 saved states the two differ by at most 1.9e-22 (2e-4 of the correction;
    delta_correction_check.py)."""
    L = s.size
    e = np.zeros(L)
    e[-1] = 1.0
    yh, yl = s.lu_solve_t(e, np.zeros(L))
    hi, _ = dd_dot(np.asarray(yh), np.asarray(yl), np.asarray(Fh), np.asarray(Fl))
    return float(-hi)


def deflated_newton(s, zh, zl, tol=1e-27, it_max=20, w=24, verbose=True, max_halvings=12, junk_only=False,
                    delta_max=1e-9, delta_tol=1e-14, jac="dd", mixed=None, gmres=None, dd_finish=True, dd_last=False):
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
    exact step costs one double-double Jacobian per solve and also leaves the exact LU for the sensitivities."""
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
                th, tl = dd_add(zh, zl, dh, dl)
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
            th, tl = dd_add(zh, zl, dh * lam, dl * lam)
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
    return zh, zl, hist

# ------------------------------------------------------------------ the quad-double solver (notes/QUAD_DOUBLE.md 12)
# A quad-double state is a 4-tuple of float64 arrays (z0, z1, z2, z3), its value z0 + z1 + z2 + z3 (multidouble's QD);
# qd_axpy (y + a x on such states) is multidouble's Python package's.
# Solver.residual_q, jacobian_q and newton_q compute in quad-double (hp/src/qsolve.rs; M = 32 and 48 only).
def qparts(zh, zl, z2=None, z3=None):
    """a four-part state from (hi, lo) and optional lower parts (zero if absent), each array contiguous float64"""
    c = lambda v: np.ascontiguousarray(np.asarray(v, dtype=float))
    z = np.zeros(len(zh))
    return (c(zh), c(zl), c(z if z2 is None else z2), c(z if z3 is None else z3))


def save_q(path, z, meta):
    """save a four-part state: save()'s format with the lower parts as z2, z3 (load() reads its first two)"""
    np.savez(path, zh=z[0], zl=z[1], z2=z[2], z3=z[3], meta=json.dumps(meta))


def load_q(path):
    """(z, meta) with z four parts; a state saved by save() has zero lower parts"""
    d = np.load(path)
    return qparts(d["zh"], d["zl"], d["z2"] if "z2" in d else None, d["z3"] if "z3" in d else None), json.loads(str(d["meta"]))


def qd_combine(ws, zs):
    """sum_i w_i z_i in quad-double (w_i f64, z_i four-part states): the Lagrange predictor of four-part states"""
    acc = qparts(np.zeros(len(zs[0][0])), np.zeros(len(zs[0][0])))
    for w, z in zip(ws, zs):
        acc = qd_axpy(float(w), *z, *acc)
    return acc


def predict_q(hist, D, n=4):
    """Lagrange extrapolation to D through the last n (D_i, z_i) pairs of four-part states, in quad-double"""
    pts = hist[-n:]
    Ds = [p[0] for p in pts]
    ws = []
    for i, Di in enumerate(Ds):
        w = 1.0
        for j, Dj in enumerate(Ds):
            if j != i:
                w *= (D - Dj) / (Di - Dj)
        ws.append(w)
    return qd_combine(ws, [p[1] for p in pts])


def qmax(F):
    """max |F| of a quad-double residual (four parts, or (hi, lo)), to f64 accuracy"""
    return float(np.abs(np.asarray(F[0]) + np.asarray(F[1])).max())


def weak_subspace_q(s, w, iters=8, seed=0):
    """Orthonormal basis (L, w) of the w weakest right singular directions of the quad-double Jacobian whose LU the
    last Solver.newton_q stored: inverse subspace iteration on (J^T J)^-1, the solves in quad-double (qlu_solve, the
    transpose first), the directions in f64 (as weak_subspace for the double-double LU)."""
    L = s.size
    V = np.linalg.qr(np.random.default_rng(seed).standard_normal((L, w)))[0]
    zero = np.zeros(L)
    for _ in range(iters):
        W = np.empty_like(V)
        for j in range(w):
            y = s.qlu_solve(np.ascontiguousarray(V[:, j]), zero, zero, zero, transpose=True)
            x = s.qlu_solve(*(np.ascontiguousarray(np.asarray(p)) for p in y))
            W[:, j] = np.asarray(x[0]) + np.asarray(x[1])
        V = np.linalg.qr(W)[0]
    return V


def junk_subspace_q(s, w, delta_max=1e-2, iters=8):
    """Rayleigh-Ritz on the w weakest right singular directions of the quad-double Jacobian (weak_subspace_q): the
    directions whose Delta component is below delta_max ("junk": p1's high harmonics, which the equations fix only to
    tol / sigma), orthonormal, with their singular values and those of the weak directions kept out (as junk_subspace
    for the double-double LU)."""
    L = s.size
    V = weak_subspace_q(s, w, iters=iters)
    zero = np.zeros(L)
    W = np.empty_like(V)
    for j in range(V.shape[1]):
        y = s.qlu_solve(np.ascontiguousarray(V[:, j]), zero, zero, zero, transpose=True)
        x = s.qlu_solve(*(np.ascontiguousarray(np.asarray(p)) for p in y))
        W[:, j] = np.asarray(x[0]) + np.asarray(x[1])
    H = V.T @ W
    ev, U = np.linalg.eigh((H + H.T) / 2)
    X = V @ U
    X /= np.linalg.norm(X, axis=0)
    sig = 1 / np.sqrt(np.maximum(ev, 1e-300))
    weak = ev > 0                                  # the rest of the w are not weak (Rayleigh-Ritz leftovers)
    keep = weak & (np.abs(X[-1, :]) < delta_max)
    return (np.linalg.qr(X[:, keep])[0] if keep.any() else np.zeros((L, 0))), sig[keep], sig[weak & ~keep]


def newton_qd(s, z, tol=1e-50, it_max=12, max_halvings=8, verbose=True, mutate=0, w=0, junk=False, delta_max=1e-2,
              tol_dz=1e-45, ls="dz", V=None):
    """Newton in quad-double (Solver.newton_q: the exact Jacobian in quad-double, its quad-double LU) from the
    four-part state z. A trial z + lam dz (lam = 1, 1/2, ..., 2^-max_halvings) is accepted when the Newton correction
    there, with its own Jacobian, is smaller than dz (max norms), or when its max |F| <= tol. At large D a full step
    raises max |F| by orders of magnitude (its quadratic term, in the strong directions, which the next step removes)
    while the Jacobian's weak direction turns between iterates, so neither a decrease of max |F| nor the natural
    monotonicity test with the old Jacobian (Deuflhard, Numer. Math. 22 (1974) 289, doi:10.1007/BF01406969)
    accepts it; the next correction
    with the new Jacobian does (notes/QUAD_DOUBLE.md, section 12). Stops when max |F| <= tol (converged), when no
    trial is accepted (stalled), or after it_max steps. A trial whose Jacobian is not finite or singular is rejected. mutate: Ctx::mutate for every call (the controls; 0 none).
    w > 0: deflated first, as deflated_newton's phase 'all': the step and the corrections it is judged by have their
    components along the w weakest right singular directions (weak_subspace_q, from the first Jacobian) removed, so
    the well-determined part converges without the nearly singular directions throwing the iterate (the layouts with
    the last node at 0.8-0.85, notes/QUAD_DOUBLE.md 12.3); when that phase stalls, or |F| stops falling by a factor
    0.3 per step, plain Newton ('full') finishes from where it got.
    junk (with w > 0): as deflated_newton's junk_only: only the weak directions without Delta content (|Delta
    component| < delta_max, junk_subspace_q) are removed, for the whole iteration, so the weak directions that carry
    Delta are solved while p1's near-null harmonics stay where the start put them; it ends converged when max |F| <=
    tol or the projected correction is below tol_dz (the residual left is the junk's).
    V: an orthonormal (L, k) basis of directions to freeze for the whole iteration (as junk, computed by the caller: p1's
    harmonics as band fields, say), instead of junk_subspace_q.
    ls: how a trial is accepted. "dz" (default): the next correction, with its own Jacobian, is smaller than dz (above).
    "F": backtracking on the residual's 2-norm as the double-double solver's newton_solve does, |F(z + lam dz)|_2 <=
    (1 - 1e-4 lam) |F(z)|_2, which creeps where the quadratic model fails (the pinned system near D 73, 12.3) but
    costs only a residual per trial; the Jacobian at the accepted point is then computed once.
    Returns (z, hist, converged); hist has one record per step (F, |dz|, lam, pivots, times, mode)."""
    st = s.newton_q(*z, mutate=mutate)
    hist = []
    L = len(z[0])
    zero = np.zeros(L)
    given = V
    mode, V, n_prev = ("sel" if given is not None else ("sel" if junk else "all") if w else "full"), given, None

    def proj(d):
        """the four-part step with its components along V removed"""
        if V is None:
            return d
        c = V @ (V.T @ (np.asarray(d[0]) + np.asarray(d[1]) + np.asarray(d[2]) + np.asarray(d[3])))
        return qd_axpy(-1.0, np.ascontiguousarray(c), zero, zero, zero, *d)

    for it in range(it_max + 1):
        nf = qmax(st[4:6])
        if nf <= tol:
            if verbose:
                print(f"  qn {it}: |F| {nf:.2e} converged, Delta {z[0][-1]:.17f} {z[1][-1]:+.4e}", flush=True)
            hist.append(dict(it=it, F=nf, dz=float(np.abs(st[0]).max()), mode=mode))
            return z, hist, True
        if it == it_max:
            break
        if mode == "all" and n_prev is not None and nf > 0.3 * n_prev:
            mode, V = "full", None
            if verbose:
                print(f"  qn: phase 'all' stalled at |F| {nf:.1e}; phase 'full'", flush=True)
        if mode == "all" and V is None:
            V = weak_subspace_q(s, w)
        if mode == "sel" and V is None:
            V, sk, sd = junk_subspace_q(s, w, delta_max)
            if verbose:
                print(f"  qn: {V.shape[1]} junk directions frozen (sigma {sk.min() if len(sk) else 0:.0e}..{sk.max() if len(sk) else 0:.0e}), "
                      f"{len(sd)} weak ones with Delta content solved (sigma {sd.min() if len(sd) else 0:.0e}..{sd.max() if len(sd) else 0:.0e})", flush=True)
        d = proj(st[:4])
        nd = float(np.abs(d[0]).max())
        if mode == "sel" and nd < tol_dz:
            if verbose:
                print(f"  qn {it}: projected correction {nd:.1e} below {tol_dz:.0e} with the junk frozen: converged at |F| {nf:.2e}, "
                      f"Delta {z[0][-1]:.17f} {z[1][-1]:+.4e}", flush=True)
            hist.append(dict(it=it, F=nf, dz=nd, mode=mode, junk_floor=True))
            return z, hist, True
        t0 = time.time()
        rec = dict(it=it, F=nf, dz=nd, dDelta=float(d[0][-1] + d[1][-1]), piv=[st[6], st[7]], mode=mode)
        lam = 1.0
        n2 = float(np.linalg.norm(np.asarray(st[4]) + np.asarray(st[5])))
        for _ in range(max_halvings + 1):
            zt = qd_axpy(lam, *d, *z)
            if ls == "F":
                Ft = s.residual_q(*zt)
                nt = float(np.linalg.norm(np.asarray(Ft[0]) + np.asarray(Ft[1])))
                if np.isfinite(nt) and nt <= (1 - 1e-4 * lam) * n2:
                    try:
                        stt = s.newton_q(*zt, mutate=mutate)
                        if np.isfinite(stt[0]).all() and np.isfinite(stt[4]).all():
                            break
                    except ValueError:
                        pass
                lam *= 0.5
                continue
            try:
                stt = s.newton_q(*zt, mutate=mutate)
                good = np.isfinite(stt[0]).all() and np.isfinite(stt[4]).all()
            except ValueError:          # a non-finite or singular Jacobian at the trial point
                good = False
            if good and (float(np.abs(proj(stt[:4])[0]).max()) < nd or qmax(stt[4:6]) <= tol):
                break
            lam *= 0.5
        else:
            rec.update(lam=0.0, seconds=round(time.time() - t0, 2))
            hist.append(rec)
            if mode == "all":
                mode, V, n_prev = "full", None, None
                if verbose:
                    print(f"  qn {it}: |F| {nf:.2e}, |dz| {nd:.1e}: no deflated step accepted; phase 'full'", flush=True)
                continue
            if verbose:
                print(f"  qn {it}: |F| {nf:.2e}, |dz| {nd:.1e}: no trial step accepted, stopping", flush=True)
            return z, hist, False
        z, st, n_prev = zt, stt, nf
        rec.update(lam=lam, F_new=qmax(st[4:6]), dz_new=float(np.abs(st[0]).max()), seconds=round(time.time() - t0, 2),
                   t=[round(v, 3) for v in s.step_timing()])
        hist.append(rec)
        if verbose:
            print(f"  qn {it} [{mode}]: |F| {nf:.2e} -> {rec['F_new']:.2e}, |dz| {nd:.1e} -> {rec['dz_new']:.1e}, "
                  f"lam {lam:g}, dDelta {rec['dDelta']:+.2e}, piv {st[6]:.1e}..{st[7]:.1e}, {rec['seconds']} s", flush=True)
    return z, hist, False


def tangent_q(s, s_of_D, D, z, h=1e-10):
    """dz/dD = -J^-1 dF/dD at a converged four-part state z: dF/dD by the central difference of Solver.residual_q at
    the f64 values D +- h with s's Taylor steps (s_of_D(D) a solver at D), divided by their exact difference, and J^-1
    by Solver.qlu_solve with the LU of a Solver.newton_q at z (made here). For a predictor: the difference's truncation
    (h^2 times the third D-derivative) and the f64 rounding of the division (1e-16 relative) are both far below the
    predictor's own error."""
    Dp, Dm = D + h, D - h
    fs = []
    for Dx in (Dp, Dm):
        sx = s_of_D(Dx)
        sx.set_segs(s.get_segs())
        fs.append(sx.residual_q(*z))
    g = tuple(np.ascontiguousarray(v / (Dp - Dm)) for v in qd_axpy(-1.0, *fs[1], *fs[0]))
    s.newton_q(*z)
    t = s.qlu_solve(*g)
    return tuple(np.ascontiguousarray(-np.asarray(v)) for v in t)
