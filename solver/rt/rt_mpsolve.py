"""Newton for the RT collocation (rt_model.py) past f64's condition-number limit: every residual in arb (rt_mp.py), and
each Jacobian solve in arb by flexible GMRES, right-preconditioned by the f64 LU of the f64 Jacobian applied by
double-double triangular solves (rows by multidouble's dd_dot). The products J(z) u are exact central differences of the
arb residual (it is quadratic, so (F(z + s u) - F(z - s u)) / (2 s) = J(z) u for every s), and the solve's accuracy is
set by the arb precision. It converges where kappa(J) 2^-53 > 1, where refinement with the f64 LU alone
(rt_mp.newton_mixed) does not. The exact inverse of the f64 factors, (LU)^-1 = (J + E)^-1 with |E| near 2^-53 |J|, is
wrong only along the few weak directions: J (LU)^-1 is the identity plus a perturbation of low rank and one small in
norm, and GMRES spends about one product on each weak direction.

    from rt_mpsolve import LUdd, newton_mp
    P = LUdd(J64)                             # J64: f64 Jacobian near z (rt_model.RT.jacobian)
    z, ok, info = newton_mp(rm, z, P)         # rm: rt_mp.RTmp; z: arb object array or f64

    python3 solver/rt/rt_mpsolve.py --selftest [--inner fgmres|gcr] [--precond lu_dd|lu_f64|inv_dd]
        [--mutate=f64apply|lowprec] [--out LOG.jsonl]

The preconditioner must be one fixed operator applied in extended precision, as the selftest's variants show
(I105-mac/planted_variants.log, planted.jsonl). At kappa = 1e20 the design takes 11 products per Newton step. The f64
triangular solves (lu_f64) are a different backward-perturbed inverse on every call, and since |(LU)^-1| is near
1/(2^-53 |J|) their rounding puts errors of the size of the solution into the well-conditioned directions. Flexible
GMRES then needs 45 to 59 products per step, and GCR (whose directions are updated by recurrences) fails with an error
of 47. The explicit f64 inverse (inv_dd, applied by dd_matvec) is fixed but hits the cap of 80 products on every step,
since LAPACK builds each of its columns by such a solve. GCR with lu_dd also passes (11 products); flexible GMRES is
kept because it stores each preconditioned vector once and forms x = Z y at the end, so no recurrence carries rounding
from one direction to the next.

Needs python-flint and multidouble's Python package with dd_dot (installed by hp/build.sh). If the interpreter with
python-flint lacks it or has an older one, MULTIDOUBLE_SITE names a site-packages directory that has it (on the Mac,
/Users/nathan/Source/venv-anm/lib/python3.13/site-packages); only the package multidouble is loaded from there.

Methods: flexible GMRES (Saad, SIAM J. Sci. Comput. 14 (1993) 461, doi:10.1137/0914028), the flexible form of GMRES
(Saad and Schultz, SIAM J. Sci. Stat. Comput. 7 (1986) 856, doi:10.1137/0907058), right-preconditioned by a
low-precision LU applied in higher precision as in GMRES-based iterative refinement (Carson and Higham, SIAM J. Sci.
Comput. 39 (2017) A2834, doi:10.1137/17M1122918); GCR (Eisenstat, Elman and Schultz, SIAM J. Numer. Anal. 20 (1983)
345, doi:10.1137/0720023) for the comparison.

Each inner solve runs to a relative residual of 1e-30, not to the loose tolerance of an inexact Newton method. An inner
residual r leaves an error of about |r| / sigma_min along the weak directions, and the quadratic term of that error
would then exceed the next residual.

Selftest (planted truth): A = U diag(s) V^T in arb, U and V products of eight Householder reflections each (exact in
arb), L = 1500, s log-uniform in [1e-4, 1] except twelve values log-spaced from 1e-9 to 1e-20 at random positions, so
kappa(A) = 1e20 exactly; x* random, b = A x* at 192 bits. newton_mp on F(x) = A x - b from x = 0, preconditioned with
the f64 LU of A rounded to f64 and the products by the same central difference as for the RT residual. Pass criteria,
written before the first run: (1) max|x - x*| / max|x*| < 1e-30 (the expected error is about kappa 2^-192, 2e-38);
(2) at most 8 Newton steps; (3) the f64 LU alone (refinement with the arb residual, rt_mp.newton_mixed's iteration)
leaves an error above 1e-6 after 20 steps, so the system is beyond f64. --mutate=f64apply computes the products with A
rounded to f64 and --mutate=lowprec runs everything at 64 bits; each must fail and exit 1.
"""
import os
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
import json
import math
import time
import argparse
import warnings
import numpy as np
import flint
import scipy.linalg as sl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
arb = flint.arb


def _multidouble():
    site = os.environ.get("MULTIDOUBLE_SITE")
    if site:
        import importlib.util
        init = os.path.join(site, "multidouble", "__init__.py")
        spec = importlib.util.spec_from_file_location("multidouble", init,
                                                      submodule_search_locations=[os.path.dirname(init)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["multidouble"] = mod
        spec.loader.exec_module(mod)
    else:
        import multidouble as mod
    if not hasattr(mod, "dd_dot"):
        raise ImportError(f"multidouble at {mod.__file__} has no dd_dot (an older package); set MULTIDOUBLE_SITE")
    return mod


md = _multidouble()


# ------------------------------------------------------------------------------------------------ arb vectors
def obj(a):
    return np.array([arb(float(v)) for v in np.asarray(a, dtype=float)], dtype=object)


def to_f64(z):
    return np.array([float(v.mid()) for v in z])


def to_dd(z):
    """(hi, lo) f64 arrays of an arb array: hi the f64 rounding of the midpoint, lo that of the rest."""
    hi = to_f64(z)
    lo = np.array([float((v.mid() - arb(h)).mid()) for v, h in zip(z, hi)])
    return hi, lo


def from_dd(hi, lo):
    return np.array([arb(float(h)) + arb(float(l)) for h, l in zip(hi, lo)], dtype=object)


def mids(v):
    """The midpoints: arb's radii only bound the rounding, and once a ball holds 0, sqrt of a norm is NaN."""
    return np.array([x.mid() for x in v], dtype=object)


def amax(v):
    return max(abs(float(x.mid())) for x in v)


def dot(a, b):
    return (a * b).sum().mid()


def nrm2(v):
    return arb(dot(v, v)).sqrt().mid()


def jv_central(residual, z, u):
    """J(z) u = (F(z + s u) - F(z - s u)) / (2 s), exact for a residual that is at most quadratic; s = 1 / max|u| keeps
    the perturbation at unit size, like rt_model.RT.jacobian's unit columns."""
    m = amax(u)
    if m == 0:
        return u * 0
    s = arb(1.0 / m)
    return mids((residual(z + s * u) - residual(z - s * u)) / (2 * s))


# ------------------------------------------------------------------------------------------------ preconditioners
def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _dd_sub(ah, al, bh, bl):
    s, e = _two_sum(ah, -bh)
    e += al - bl
    h = s + e
    return h, e - (h - s)


def _dd_div(ah, al, p):
    q1 = ah / p
    e = math.fma(-p, q1, ah)                 # ah - p q1 exactly
    q2 = (e + al) / p
    h = q1 + q2
    return h, q2 - (h - q1)


class LUdd:
    """r -> J_f^-1 r, J_f the product of the f64 factors of J64: LAPACK getrf (partial pivoting) of J64^T, so
    J64^T = P L U and J64 = U^T L^T P^T, with the triangular solves in double-double, each row a dd_dot of the f64 row
    with the dd solution so far. A fixed operator to about 2^-106 (n / 8) of the sums. J64^T of a C-order J64 is
    F-contiguous, so with overwrite the factors replace J64 in its own buffer (one copy of the matrix in memory), and the
    rows of U^T and L^T are rows of that C-order buffer: M[i, :i + 1] is row i of U^T, M[i, i + 1:] row i of L^T (unit
    diagonal)."""

    def __init__(self, J64, overwrite=False):
        t = time.time()
        A = J64 if (overwrite and J64.flags.c_contiguous) else np.array(J64, order="C")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f, piv = sl.lu_factor(A.T, overwrite_a=True, check_finite=False)
        if not np.shares_memory(f, A):                    # not factored in place: keep the factors' C-order transpose
            A = np.ascontiguousarray(f.T)
        del f
        self.M = A
        n = len(piv)
        perm = np.arange(n)
        for i, p in enumerate(piv):                       # P^T b = b[perm] (LAPACK's interchanges, in order)
            perm[i], perm[p] = perm[p], perm[i]
        self.perm = perm
        self.z = np.zeros(n)
        d = np.abs(np.diag(self.M))
        self.pivot_ratio = float(d.max() / d.min())     # an estimate of kappa, not an SVD
        self.secs = time.time() - t

    def solve_dd(self, bh, bl):
        """J_f x = b: U^T y = b, L^T w = y, x = P w."""
        n = len(bh)
        M, z, dd_dot = self.M, self.z, md._dd_dot
        yh, yl = np.zeros(n), np.zeros(n)
        for i in range(n):
            sh, sl_ = dd_dot(M[i, :i], z[:i], yh[:i], yl[:i])
            th, tl = _dd_sub(bh[i], bl[i], sh, sl_)
            yh[i], yl[i] = _dd_div(th, tl, M[i, i])
        wh, wl = np.zeros(n), np.zeros(n)
        for i in range(n - 1, -1, -1):
            sh, sl_ = dd_dot(M[i, i + 1:], z[i + 1:], wh[i + 1:], wl[i + 1:])
            wh[i], wl[i] = _dd_sub(yh[i], yl[i], sh, sl_)
        xh, xl = np.empty(n), np.empty(n)
        xh[self.perm], xl[self.perm] = wh, wl
        return xh, xl

    def __call__(self, r):
        return from_dd(*self.solve_dd(*to_dd(r)))


class LUf64:
    """The f64 LU of J64, its triangular solves in f64 on the f64 rounding of r (variant; see the module docstring)."""

    def __init__(self, J64, overwrite=False):
        t = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lu = sl.lu_factor(J64, overwrite_a=overwrite, check_finite=False)
        d = np.abs(np.diag(self.lu[0]))
        self.pivot_ratio = float(d.max() / d.min())
        self.secs = time.time() - t

    def __call__(self, r):
        return obj(sl.lu_solve(self.lu, to_f64(r), check_finite=False))


class InvDD:
    """r -> X r, X = inv(J64) in f64, applied to the dd rounding of r by dd_matvec (variant)."""

    def __init__(self, J64):
        t = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.X = np.ascontiguousarray(sl.inv(J64, check_finite=False))
        self.Z = np.zeros_like(self.X)
        self.secs = time.time() - t

    def __call__(self, r):
        rh, rl = to_dd(r)
        return from_dd(*md.dd_matvec(self.X, self.Z, rh, rl))


# ------------------------------------------------------------------------------------------------ inner solvers
def fgmres(apply, precond, b, tol=1e-30, max_products=80):
    """x with A x = b in arb: flexible GMRES (no restart), z_k = M^-1 v_k stored, w = A z_k orthogonalized against the
    v's (modified Gram-Schmidt), the least-squares problem by Givens rotations, x = Z y. Stops when the residual
    estimate |g_{k+1}| <= tol |b|_2 or after max_products products. Returns (x, info)."""
    info = dict(products=0, res=[], t_prod=0.0, t_prec=0.0)
    beta = nrm2(b)
    if not float(beta) > 0:
        info["converged"] = True
        return b * 0, info
    V, Z, H, cs, sn, g = [mids(b / beta)], [], [], [], [], [beta]
    nb = float(beta)
    for k in range(max_products):
        t = time.time()
        z = mids(precond(V[k]))
        info["t_prec"] += time.time() - t
        t = time.time()
        w = apply(z)
        info["t_prod"] += time.time() - t
        info["products"] += 1
        h = []
        for v in V:
            hij = dot(v, w)
            w = w - hij * v
            h.append(hij)
        hk1 = nrm2(w)
        h.append(hk1)
        Z.append(z)
        V.append(mids(w / hk1) if float(hk1) > 0 else w)
        for i in range(k):
            t1 = (cs[i] * h[i] + sn[i] * h[i + 1]).mid()
            h[i + 1] = (-sn[i] * h[i] + cs[i] * h[i + 1]).mid()
            h[i] = t1
        d = arb(h[k] * h[k] + h[k + 1] * h[k + 1]).sqrt().mid()
        c, s = (h[k] / d).mid(), (h[k + 1] / d).mid()
        cs.append(c)
        sn.append(s)
        h[k] = d
        h[k + 1] = arb(0)
        g.append((-s * g[k]).mid())
        g[k] = (c * g[k]).mid()
        H.append(h)
        rel = abs(float(g[k + 1])) / nb
        info["res"].append(rel)
        if rel <= tol or not float(hk1) > 0:
            break
    m = len(Z)
    y = [arb(0)] * m
    for i in range(m - 1, -1, -1):
        acc = g[i]
        for j in range(i + 1, m):
            acc = acc - H[j][i] * y[j]
        y[i] = (acc / H[i][i]).mid()
    x = b * 0
    for j in range(m):
        x = x + y[j] * Z[j]
    info["converged"] = bool(info["res"] and info["res"][-1] <= tol)
    return mids(x), info


def gcr(apply, precond, b, tol=1e-30, max_products=80):
    """x with A x = b in arb by GCR, right-preconditioned (variant): u = M^-1 r, w = A u, w orthogonalized against the
    earlier w's with the same coefficients applied to u, x += c u, r -= c w. Stops when the recursive residual
    |r|_2 <= tol |b|_2. Returns (x, info)."""
    info = dict(products=0, res=[], t_prod=0.0, t_prec=0.0)
    x, r = b * 0, b.copy()
    nb = float(nrm2(b))
    U, W = [], []
    if not nb > 0:
        info["converged"] = True
        return x, info
    while info["products"] < max_products:
        t = time.time()
        u = mids(precond(r))
        info["t_prec"] += time.time() - t
        t = time.time()
        w = apply(u)
        info["t_prod"] += time.time() - t
        info["products"] += 1
        for Ui, Wi in zip(U, W):
            beta = dot(Wi, w)
            w = w - beta * Wi
            u = u - beta * Ui
        nw = nrm2(w)
        if not float(nw) > 0:
            break
        w, u = mids(w / nw), mids(u / nw)
        c = dot(w, r)
        x, r = mids(x + c * u), mids(r - c * w)
        U.append(u)
        W.append(w)
        info["res"].append(float(nrm2(r)) / nb)
        if info["res"][-1] <= tol:
            break
    info["converged"] = bool(info["res"] and info["res"][-1] <= tol)
    return x, info


# ------------------------------------------------------------------------------------------------ Newton
def newton_mp(rm, z, prec, tol=1e-40, it_max=12, inner_tol=1e-30, max_products=80, residual=None, apply=None,
              inner=fgmres, make_prec=None, refresh_dz=1e-3, refresh_ratio=1.5, log=None, stall_any=True,
              full_below=None):
    """Newton with the residual in arb and each step solved by `inner` (fgmres) in arb, products by jv_central unless
    apply is given, preconditioned by prec (an LUdd of an f64 Jacobian near z). With make_prec (z -> a new
    preconditioner), prec is rebuilt at the new z after a step with max|dz| > refresh_dz, or after an inner solve that
    took more than refresh_ratio times the products of the first solve with the current prec. A preconditioner from a
    point far away is no longer the identity plus a low-rank term on J(z), and GMRES then needs several times the
    products (76 against 10 after a step of 8e-2 at D = 12, I105-mac/pilot); the count itself grows with D (about 10
    at D = 10, 25 at D = 20 on 32x81), so a fixed threshold rebuilds on every step there. rm: an rt_mp.RTmp (or None
    with residual given). Stops when max|F| < tol, or when max|F| stops falling (at the precision's floor: max|F| > 0.5
    of the previous after step 3; with stall_any=False only once max|F| < 1e6 tol), or after it_max steps. A step that
    does not lower max|F| is halved up to six times, unless full_below is given and the step's max|dz| is below it.
    For the quadratic residual a full step dz leaves F = B(dz, dz) exactly, so a small full step cannot raise max|F|
    by much. Near a nearly singular root Newton then reduces the error along the weak direction by about half per
    step, while halving the step makes almost no progress (32x101 at D = 34: max|F| near 3e-25, steps of 1/32 with
    max|dz| 7e-15, I145-mac).
    Returns (z, ok, info) with z an arb object array; info["dz"] is max|dz| of each step and info["refresh"] the steps
    after which prec was rebuilt. info["t_prod"] and info["t_prec"] are each step's seconds in products and in the
    preconditioner."""
    say = log or (lambda s: None)
    residual = residual or rm.residual_m
    apply = apply or (lambda zz, u: jv_central(residual, zz, u))
    if z.dtype != object:
        z = obj(z)
    F = residual(z)
    r = amax(F)
    info = dict(res=[r], products=[], inner_res=[], lam=[], secs=[], dz=[], refresh=[], t_refresh=0.0, t_prod=[],
                t_prec=[])
    ok = r < tol
    base = None                                            # products of the first solve with the current prec
    for it in range(it_max):
        if ok:
            break
        if make_prec is not None and info["dz"] and (info["dz"][-1] > refresh_dz
                                                      or info["products"][-1] > refresh_ratio * base):
            t = time.time()
            prec = None                                    # free the old factors first
            prec = make_prec(z)
            base = None
            info["refresh"].append(it)
            info["t_refresh"] += time.time() - t
            say(f"    newton_mp: preconditioner rebuilt at step {it} ({time.time() - t:.1f} s)")
        t0 = time.time()
        dz, gi = inner(lambda u: apply(z, u), prec, mids(-F), tol=inner_tol, max_products=max_products)
        lam = arb(1)
        small = full_below is not None and amax(dz) < full_below
        for _ in range(7):
            zn = mids(z + lam * dz)
            Fn = residual(zn)
            rn = amax(Fn)
            if rn < r or small:                            # a small full step is taken even if max|F| grows
                break
            lam = lam / 2
        z, F = zn, Fn
        if base is None:
            base = gi["products"]
        info["products"].append(gi["products"])
        info["inner_res"].append(gi["res"][-1] if gi["res"] else None)
        info["lam"].append(float(lam.mid()))
        info["secs"].append(round(time.time() - t0, 2))
        info["t_prod"].append(round(gi["t_prod"], 2))
        info["t_prec"].append(round(gi["t_prec"], 2))
        info["dz"].append(amax(dz) * float(lam.mid()))
        ir = info["inner_res"][-1]
        say(f"    newton_mp it {it}: max|F| {r:.3e} -> {rn:.3e}, {gi['products']} products to "
            f"{ir if ir is not None else float('nan'):.1e}, step {float(lam.mid()):g}, max|dz| {info['dz'][-1]:.1e}, "
            f"{info['secs'][-1]} s (products {gi['t_prod']:.1f} s, precond {gi['t_prec']:.1f} s)")
        prev, r = r, rn
        info["res"].append(r)
        if r < tol:
            ok = True
            break
        # the precision's floor; with stall_any=False only near tol, since far above it a step along a weak
        # direction can leave max|F| nearly unchanged and the next ones still converge (32x121 at D = 32, I145-mac)
        if it >= 3 and r > 0.5 * prev and (stall_any or r < 1e6 * tol):
            ok = r < 1e6 * tol
            break
    return z, ok, info


# ------------------------------------------------------------------------------------------------ planted selftest
def householders(L, k, rng):
    out = []
    for _ in range(k):
        w = rng.normal(size=L)
        wo = obj(w)
        out.append((wo, dot(wo, wo), w))
    return out


def apply_H(Hs, v, rev=False):
    """v -> H_1 ... H_k v (arb), each H = I - 2 w w^T / (w.w); rev: H_k ... H_1 v."""
    for wo, ww, _ in (Hs if rev else Hs[::-1]):
        v = v - (2 * dot(wo, v) / ww) * wo
    return v


def apply_H64(Hs, V, rev=False):
    for _, _, w in (Hs if rev else Hs[::-1]):
        V = V - np.outer(2 * w / (w @ w), w @ V)
    return V


def selftest(mutate=None, out=None, L=1500, kappa_exp=20, inner="fgmres", precond="lu_dd"):
    prec = 64 if mutate == "lowprec" else 192
    flint.ctx.prec = prec
    rng = np.random.default_rng(105)
    s = np.exp(rng.uniform(np.log(1e-4), 0.0, size=L))
    small = np.logspace(-9, -kappa_exp, 12)
    pos = rng.choice(L, size=12, replace=False)
    s[pos] = small
    s[np.argmax(s)] = 1.0                                          # sigma_max = 1, sigma_min = 10^-kappa_exp
    so = obj(s)
    HU = householders(L, 8, rng)
    HV = householders(L, 8, rng)
    A = lambda v: apply_H(HU, so * apply_H(HV, v, rev=True))      # U diag(s) V^T, V^T = H_k ... H_1
    A64 = apply_H64(HU, s[:, None] * apply_H64(HV, np.eye(L), rev=True))
    xs = obj(rng.normal(size=L))
    b = A(xs)
    t0 = time.time()
    P = dict(lu_dd=LUdd, lu_f64=LUf64, inv_dd=InvDD)[precond](A64)
    t_p = time.time() - t0
    residual = lambda x: A(x) - b
    apply = None
    if mutate == "f64apply":
        apply = lambda x, u: obj(A64 @ to_f64(u))
    t0 = time.time()
    x, ok, info = newton_mp(None, obj(np.zeros(L)), P, tol=1e-300, it_max=8, residual=residual, apply=apply,
                            inner=dict(fgmres=fgmres, gcr=gcr)[inner], log=print)
    t_solve = time.time() - t0
    err = amax(x - xs) / amax(xs)
    # the f64 LU alone: z <- z - M^-1 F(z) with F in arb (rt_mp.newton_mixed's iteration)
    lu = sl.lu_factor(A64)
    z = obj(np.zeros(L))
    for _ in range(20):
        z = z - obj(sl.lu_solve(lu, to_f64(residual(z))))
    err_f64 = amax(z - xs) / amax(xs)
    steps = len(info["res"]) - 1
    c1, c2, c3 = err < 1e-30, steps <= 8, err_f64 > 1e-6
    rec = dict(test="planted", mutate=mutate, inner=inner, precond=precond, prec=prec, L=L, kappa=10.0 ** kappa_exp,
               err=err, steps=steps, products=info["products"], res=info["res"], err_f64_lu_alone=err_f64,
               t_precond=round(t_p, 2), t_solve=round(t_solve, 2), crit=[bool(c1), bool(c2), bool(c3)])
    print(f"planted L {L} kappa 1e{kappa_exp} prec {prec} inner {inner} precond {precond} mutate {mutate}: error "
          f"{err:.2e} (< 1e-30: {c1}), {steps} Newton steps (<= 8: {c2}), products {info['products']}, f64 LU alone "
          f"{err_f64:.2e} (> 1e-6: {c3}); preconditioner {t_p:.1f} s, solve {t_solve:.1f} s")
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    passed = c1 and c2 and c3
    print("PASS" if passed else "FAIL")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", default=None, choices=["f64apply", "lowprec"])
    ap.add_argument("--inner", default="fgmres", choices=["fgmres", "gcr"])
    ap.add_argument("--precond", default="lu_dd", choices=["lu_dd", "lu_f64", "inv_dd"])
    ap.add_argument("--L", type=int, default=1500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.mutate, a.out, L=a.L, inner=a.inner, precond=a.precond) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
