"""Inner selection with M harmonics, well conditioned: S_1 = S_3 = ... = S_M = 0 by a nested solve.

Why nested: in the de-mixed residual (inner_np.residual_demixed) each condition r_nu is normalized by |K'_nu|, but the
finite-xi de-mixing leaves a leakage of S_mu (mu > nu) into r_nu amplified by about |K'_mu/K'_nu| ~ e^{pi (mu-nu) w/2}:
r_1 picks up ~1e4 r_3 and ~1e8-1e10 r_5. A joint solve then has condition ~1e10 and stalls on false near-zeros where a
small S_3 cancels S_1 inside r_1 (measured at M = 3: Jacobian singular values 6e9, 5e9, 23, 0.4). So:
  inner: at fixed (A, w) solve S_3 = ... = S_M = 0 for F_3..F_M to the noise floor (|r_nu>=3| ~ 1e-11); the leakage
         into r_1 is then negligible and r_1 measures S_1 alone. Seeds for missing harmonics: F_{nu+2} = q A^2 F_nu,
         q = 0.0085 + 0.0011i (agent C); if the cold inner solve fails, the new harmonics are switched on by homotopy
         (weight lam on every product touching them, lam: 0 -> 1, inner_np.Inner).
  outer: Newton on (A, w/w_inf) for r_1 = 0 with finite differences of inner solves (2 x 2, condition ~2 at M = 3).
At M = 3 this gives A = 2.726476, w/w_inf = 1.015603, F3/F1 = 0.052969 + 0.004067i at xi1 = 0.2 (r_1 ~ 1e-6).

    python -u full_M.py M A wf [B3 C3 [B5 C5 ...]] [--xi1=0.2]            full selection
    python -u full_M.py M A wf [B3 C3 ...] --fixA [--xi1=0.2]             A, w fixed: inner solve only; prints
                                                                          F_nu/F_1 and the harmonic-1 residual r_1
    add  --procs N  to compute the finite-difference Jacobian columns on N processes (multiprocessing, spawn;
    works on Windows; default 1).
F_nu = B_nu/(2i) + C_nu/2 is the e^{i nu theta} coefficient of F = A sin(theta) + ... (agent C's convention).
Examples (from the M = 3 solution):
    python -u full_M.py 5 2.726476 1.015603 0.144419 0.011089
    python -u full_M.py 5 2.72 1.0 0.14737 0.01139 --fixA
Needs numpy, scipy, mpmath and inner_np.py in the same folder. Single process; prints progress with timings.
"""
import sys
import time
import numpy as np
from scipy.optimize import least_squares
import inner_np as I

Q = 0.0085 + 0.0011j
T0 = time.time()
NEV = [0]
POOL = None          # multiprocessing pool for the Jacobian columns (--procs N); spawn start method, Windows-safe


def _worker(args):
    A, wf, v, M, r1, M0, lam = args
    return I.residual_demixed(np.concatenate([[A, wf], v]), M, M0, lam, r1)[2:]


def lsq(A, wf, M, r1, M0, lam, v0, nfev):
    """least squares on r_3..r_M at fixed (A, wf); Jacobian columns in parallel when POOL is set"""
    fun = lambda u: res_high(A, wf, u, M, r1, M0, lam)[2:]
    kw = dict(method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=nfev)
    if POOL is None:
        return least_squares(fun, v0, **kw)
    cache = {}

    def fun_c(u):
        r = fun(u)
        cache[u.tobytes()] = r
        return r

    def jac(u):
        f0 = cache.get(u.tobytes())
        if f0 is None:
            f0 = fun(u)
        h = 1e-7 * np.maximum(1.0, np.abs(u))
        jobs = [(A, wf, u + h[i] * np.eye(len(u))[i], M, r1, M0, lam) for i in range(len(u))]
        cols = POOL.map(_worker, jobs)
        NEV[0] += len(u)
        return np.array([(c - f0) / h[i] for i, c in enumerate(cols)]).T

    return least_squares(fun_c, v0, jac=jac, **kw)


def seed(x, Mg, M, q=Q):
    """extend x (harmonics up to Mg) to M with F_{nu+2} = q A^2 F_nu"""
    A = x[0]
    Fh = I.Fh_from(x, Mg)
    last = Fh[Mg]
    x = list(x)
    for nu in range(Mg + 2, M + 1, 2):
        last = last * q * A * A
        x += [-2 * last.imag, 2 * last.real]
    return np.array(x, float)


def res_high(A, wf, v, M, r1, M0=None, lam=1.0):
    NEV[0] += 1
    return I.residual_demixed(np.concatenate([[A, wf], v]), M, M0, lam, r1)


def inner(A, wf, v0, M, r1, Mg=None, verbose=False):
    """solve r_3..r_M = 0 for v = (B3, C3, ...) at fixed (A, wf); returns v, r_1 (complex), max|r_nu>=3|"""
    f = lambda v, M0=None, lam=1.0: res_high(A, wf, v, M, r1, M0, lam)[2:]
    start = np.abs(f(v0)).max()
    if start > 1e2 and Mg is not None and Mg < M:
        bad = start                       # seeded high harmonics far from their conditions: go straight to homotopy
    else:
        sol = lsq(A, wf, M, r1, None, 1.0, v0, 50 * len(v0))
        v, bad = sol.x, np.abs(sol.fun).max()
    if bad > 1e-8 and Mg is not None and Mg < M:
        if verbose:
            print(f"   inner solve: |r| {bad:.1e} after the cold start; homotopy in the new harmonics", flush=True)
        v = v0.copy()
        v[len(v) - (M - Mg):] = 0.0
        lam, dlam = 0.0, 0.25
        while lam < 1 - 1e-12:
            lt = min(1.0, lam + dlam)
            s2 = lsq(A, wf, M, r1, Mg, lt, v, 100 * len(v0))
            if np.abs(s2.fun).max() < 1e-8:
                v, lam = s2.x, lt
                dlam = min(0.5, 1.5 * dlam)
                if verbose:
                    print(f"      lam {lam:.3f} ok  [{time.time() - T0:.0f} s]", flush=True)
            else:
                dlam /= 2
                if dlam < 1 / 512:
                    raise RuntimeError("inner homotopy stalled")
    r = res_high(A, wf, v, M, r1)
    return v, r[0] + 1j * r[1], np.abs(r[2:]).max()


def report(A, wf, v, M, tag, r1c):
    Fh = I.Fh_from(np.concatenate([[A, wf], v]), M)
    w = wf * I.W0
    print(f"{tag} M {M}: A {A:.6f}  w/w_inf {wf:.6f}  (w {w:.5f}, Delta {2 * np.pi / w:.5f})  <F^2> {I.F2mean(Fh):.4f}  "
          f"r_1 {r1c:.3e}  [{time.time() - T0:.0f} s, {NEV[0]} residuals]", flush=True)
    for nu in range(3, M + 1, 2):
        rr = Fh[nu] / Fh[1]
        print(f"      F_{nu}/F_1 = {rr.real:+.6f}{rr.imag:+.6f}i   |.| {abs(rr):.6f}  arg {np.angle(rr):+.4f}", flush=True)


def main():
    global POOL
    argv = sys.argv[1:]
    procs = 1
    if "--procs" in argv:
        k = argv.index("--procs")
        procs = int(argv[k + 1])
        del argv[k:k + 2]
    args = [a for a in argv if not a.startswith("--")]
    fixA = "--fixA" in argv
    r1 = 0.2
    for a in argv:
        if a.startswith("--xi1="):
            r1 = float(a.split("=")[1])
    if procs > 1:
        import multiprocessing as mpc
        POOL = mpc.get_context("spawn").Pool(procs)
        print(f"Jacobian columns on {procs} processes", flush=True)
    M = int(args[0])
    xg = np.array([float(a) for a in args[1:]])
    Mg = len(xg) - 1
    x0 = seed(xg, Mg, M) if Mg < M else xg
    A, wf, v = x0[0], x0[1], x0[2:]
    v, R, rh = inner(A, wf, v, M, r1, Mg, verbose=True)
    print(f"inner solve: |r_nu>=3| {rh:.1e}, r_1 {R:.4e}  [{time.time() - T0:.0f} s]", flush=True)
    if fixA:
        report(A, wf, v, M, "FIXED-A", R)
        return
    for it in range(10):
        if abs(R) < 1e-5:
            break
        hA, hw = 1e-4, 1e-4
        _, RA, _ = inner(A + hA, wf, v, M, r1)
        _, Rw, _ = inner(A, wf + hw, v, M, r1)
        Jm = np.array([[(RA - R).real / hA, (Rw - R).real / hw], [(RA - R).imag / hA, (Rw - R).imag / hw]])
        sv = np.linalg.svd(Jm, compute_uv=False)
        step = -np.linalg.solve(Jm, [R.real, R.imag])
        A, wf = A + step[0], wf + step[1]
        v, R, rh = inner(A, wf, v, M, r1)
        print(f"outer {it}: A {A:.7f} w/w_inf {wf:.7f}  r_1 {R:.3e}  |r_nu>=3| {rh:.1e}  "
              f"(2x2 singular values {sv[0]:.2e} {sv[1]:.2e})  [{time.time() - T0:.0f} s]", flush=True)
    report(A, wf, v, M, "ZERO" if abs(R) < 1e-3 else "NOT CONVERGED", R)


if __name__ == "__main__":
    main()
