"""Nonlinear elimination with double-double unknowns and accurate derivatives in the near-null directions.

ne_solver.ne_solve freezes the "junk" directions (s/s_max < 3e-14: centre-data harmonics k ~ 7-21). Their residual
is ~1e-14..1e-12, but they are not irrelevant: two NE solutions at D = 23 that differ by ~3e-2 in those harmonics
differ by 7.5e-6 in Delta, with junk residuals 1e-14 and 1e-12. So every direction has to be solved, and to do that:

* z is double-double (zh + zl): f64 rounding of z alone leaves strong residuals ~1e-13..1e-11;
* the residual is double-double (rustkernel residual_dd, ~1e-27);
* the strong block (s/s_max >= cut_strong) uses the exact f64 Dual Jacobian (Newton, iterative refinement);
* J V_o for the other directions (weak + junk) is taken by double-double central differences, because the f64
  Jacobian's absolute error (~1e-16 s_max ~ 1e-10) is larger than their singular values;
* the outer (weak + junk) equations are solved through the Schur complement, re-solving the strong block at every
  trial point.

    zh, zl, info = ne_dd(R, z)          # R = rustkernel.RustResidual(S)
"""
import time
import numpy as np
from dd_solver import dd_add


def _F(R, zh, zl):
    Fh, Fl = R.residual_dd(zh, zl)
    return Fh + Fl


def ne_dd(R, zh, zl=None, cut_strong=1e-10, outer_max=20, inner_max=8, inner_tol=1e-24, tol_o=1e-24, h=1e-4,
          rcond=1e-12, verbose=False):
    zh = np.array(zh, float)
    zl = np.zeros_like(zh) if zl is None else np.array(zl, float)
    t0 = time.time()
    _, J = R.jacobian(zh)
    U, s, Vt = np.linalg.svd(J)
    r = s / s[0]
    Si = r >= cut_strong
    Us, Uo = U[:, Si], U[:, ~Si]
    Vs, Vo = Vt[Si].T, Vt[~Si].T
    info = dict(n_strong=int(Si.sum()), n_outer_dirs=int((~Si).sum()), hist=[])
    if verbose:
        print(f"  ne_dd: strong {Si.sum()}, weak+junk {(~Si).sum()} (s/smax {r[~Si].min():.0e} .. {r[~Si].max():.0e})", flush=True)

    def inner(zh, zl):
        F = _F(R, zh, zl)
        for _ in range(inner_max):
            es = Us.T @ F
            nes = np.linalg.norm(es)
            if nes < inner_tol:
                break
            _, J = R.jacobian(zh)
            da = -np.linalg.solve(Us.T @ J @ Vs, es)
            ok = False
            for a in (1.0, 0.5, 0.25, 0.1):
                th, tl = dd_add(zh, zl, a * (Vs @ da))
                Ft = _F(R, th, tl)
                if np.all(np.isfinite(Ft)) and np.linalg.norm(Us.T @ Ft) < nes:
                    zh, zl, F, ok = th, tl, Ft, True
                    break
            if not ok:
                break
        _, J = R.jacobian(zh)
        return zh, zl, F, J

    def jvo(zh, zl):
        P = dd_add(np.repeat(zh[None], Vo.shape[1], 0), np.repeat(zl[None], Vo.shape[1], 0), h * Vo.T)
        M = dd_add(np.repeat(zh[None], Vo.shape[1], 0), np.repeat(zl[None], Vo.shape[1], 0), -h * Vo.T)
        Fp, Fm = _F(R, *P), _F(R, *M)
        return ((Fp - Fm) / (2 * h)).T

    zh, zl, F, J = inner(zh, zl)
    for k in range(outer_max):
        es, eo = Us.T @ F, Uo.T @ F
        info["hist"].append((float(np.linalg.norm(F)), float(np.linalg.norm(es)), float(np.linalg.norm(eo)), float(zh[-1] + zl[-1])))
        if verbose:
            print(f"  outer {k}: |F| {np.linalg.norm(F):.2e}  strong {np.linalg.norm(es):.1e}  outer {np.linalg.norm(eo):.2e}"
                  f"  Delta {zh[-1]:.12f}{zl[-1]:+.1e}  ({time.time()-t0:.0f}s)", flush=True)
        if np.linalg.norm(eo) < tol_o:
            break
        JVo = jvo(zh, zl)
        A = Us.T @ J @ Vs
        B = Us.T @ JVo
        C = Uo.T @ J @ Vs
        Dm = Uo.T @ JVo
        AiB = np.linalg.solve(A, B)
        Aies = np.linalg.solve(A, es)
        G = Dm - C @ AiB
        rhs = eo - C @ Aies
        dw = -np.linalg.lstsq(G, rhs, rcond=rcond)[0]
        ds = -Aies - AiB @ dw
        best = None
        for a in (1.0, 0.5, 0.25, 0.1, 0.03):
            th, tl = dd_add(zh, zl, a * (Vo @ dw + Vs @ ds))
            th, tl, Ft, Jt = inner(th, tl)
            no = np.linalg.norm(Uo.T @ Ft)
            if np.isfinite(no) and no < np.linalg.norm(eo):
                best = (th, tl, Ft, Jt)
                break
        if best is None:
            if verbose:
                print("  outer: no decrease; stopping", flush=True)
            break
        zh, zl, F, J = best
    info["F"] = float(np.linalg.norm(F))
    info["seconds"] = round(time.time() - t0)
    return zh, zl, info
