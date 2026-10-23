#!/usr/bin/env python3
"""Does the mixed-precision Newton step (f64 Jacobian, double-double residual) still steer at large D, where the
Jacobian has directions 1e17 or more times weaker than its strongest? And if not, in how many directions does it fail?

For each saved large-D state (hp/data/states/up*_D<D>.npz; from D = 35 with the p1 harmonics above 13 frozen, as the
scan ran) it reports:
  - c_f64lu: the relative difference between the Newton step solved with an f64 LU of the double-double Jacobian's
    hi part and the double-double step (both from the same perturbed state). This isolates the f64 factorization;
  - c_mixed: the same for Solver.newton_mixed (f64 Jacobian built in f64, f64 LU); c_mixed_ddbase with the f64
    tangents on a double-double base trajectory (jacobian_f64(dd_base=True)). Newton with the f64 Jacobian
    contracts the error by about this factor per step once the quadratic part is spent; at c >= 1 it does not converge;
  - the error's rank: the singular values of the error matrix X_f64 - X_dd over R random right-hand sides (solves with
    the f64 Jacobian's LU against the double-double LU), relative to the largest |X_dd|. If only r of them are large,
    the f64 solve is wrong in r directions only, and a rank-r correction (a few Krylov steps with exact Jacobian-vector
    products, or deflation of r weak directions) repairs it;
  - sigma_min / sigma_max: the weakest singular values of the double-double Jacobian (inverse subspace iteration with
    its LU, driver.weak_subspace) against the largest (f64 SVD norm);
  - optionally (--solve) Newton from the perturbed state to tolerance with each Jacobian: iterations and the result.

    cd hp && python3 mixed_largeD.py [--D=8.25,20,30,40,50,60] [--eps=1e-12] [--R=48] [--threads=2] [--solve] [--json=out]
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import load, make_solver, newton_solve, weak_subspace  # noqa: E402


def states():
    out = {}
    for p in glob.glob(os.path.join(HERE, "data", "states", "up*_D*.npz")):
        D = float(re.search(r"_D([0-9.]+)\.npz$", p).group(1))
        out[D] = p
    return out


def nearest(avail, D):
    return min(avail, key=lambda d: abs(d - D))


def main():
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    Ds = [float(x) for x in opts.get("D", "8.25,16.76,20,25,30,35,40,45,50,55,60").split(",")]
    eps, R, threads = float(opts.get("eps", 1e-12)), int(opts.get("R", 48)), int(opts.get("threads", 2))
    avail = states()
    rng = np.random.default_rng(1)
    out = []
    for D0 in Ds:
        path = avail[nearest(avail, D0)]
        zh, zl, m = load(path)
        kf = 13 if m["D"] >= 35 else None
        s = make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads, kfreeze=kf)
        s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
        n = s.size
        Fh, Fl = s.residual(zh, zl)
        F0 = float(np.max(np.abs(np.asarray(Fh) + np.asarray(Fl))))
        ph = np.asarray(zh) * (1 + eps * rng.standard_normal(n))
        pl = np.asarray(zl).copy()
        t = time.time()
        Jh, Jl = s.jacobian(ph, pl)
        t_jdd = time.time() - t
        t = time.time()
        J64 = np.asarray(s.jacobian_f64(ph, pl))
        t_j64 = time.time() - t
        jdiff = float(np.max(np.abs(J64 - Jh)) / np.max(np.abs(Jh)))
        t = time.time()
        J64b = np.asarray(s.jacobian_f64(ph, pl, dd_base=True))
        t_j64b = time.time() - t
        jdiffb = float(np.max(np.abs(J64b - Jh)) / np.max(np.abs(Jh)))
        # double-double step (stores the dd LU in s)
        dh, dl, Fph, Fpl, pmin, pmax = s.newton(ph, pl)
        ddz = np.asarray(dh) + np.asarray(dl)
        b = -(np.asarray(Fph) + np.asarray(Fpl))
        lu_hi = sla.lu_factor(np.asarray(Jh), check_finite=False)
        lu_64 = sla.lu_factor(J64, check_finite=False)
        nrm = max(np.linalg.norm(ddz), 1e-300)
        c_f64lu = float(np.linalg.norm(sla.lu_solve(lu_hi, b, check_finite=False) - ddz) / nrm)
        c_mixed = float(np.linalg.norm(sla.lu_solve(lu_64, b, check_finite=False) - ddz) / nrm)
        lu_64b = sla.lu_factor(J64b, check_finite=False)
        c_mixed_ddbase = float(np.linalg.norm(sla.lu_solve(lu_64b, b, check_finite=False) - ddz) / nrm)
        # the error's rank over R random right-hand sides (dd LU still stored)
        B = rng.standard_normal((n, R))
        Xdd = np.empty_like(B)
        z0 = np.zeros(n)
        for j in range(R):
            xh, xl = s.lu_solve(np.ascontiguousarray(B[:, j]), z0)
            Xdd[:, j] = np.asarray(xh) + np.asarray(xl)
        X64 = sla.lu_solve(lu_64, B, check_finite=False)
        sv_err = np.linalg.svd(X64 - Xdd, compute_uv=False) / np.max(np.abs(Xdd))
        # weakest singular values of the dd Jacobian
        V = weak_subspace(s, 8, iters=8)
        W = np.empty_like(V)
        for j in range(V.shape[1]):
            y, _ = s.lu_solve_t(np.ascontiguousarray(V[:, j]), z0)
            x, _ = s.lu_solve(np.ascontiguousarray(np.asarray(y)), z0)
            W[:, j] = np.asarray(x)
        sig_weak = np.sort(1 / np.sqrt(np.abs(np.einsum("ij,ij->j", V, W))))
        sig_max = float(np.linalg.norm(np.asarray(Jh), 2))
        rec = dict(D=m["D"], M=m["M"], N=n, kfreeze=kf, F_state=F0, eps=eps, t_jac_dd=round(t_jdd, 2),
                   t_jac_f64=round(t_j64, 2), jac_f64_vs_hi=jdiff, c_f64lu=c_f64lu, c_mixed=c_mixed,
                   t_jac_f64_ddbase=round(t_j64b, 2), jac_f64_ddbase_vs_hi=jdiffb, c_mixed_ddbase=c_mixed_ddbase,
                   pivots=[pmin, pmax], err_sv=[float(f"{v:.2e}") for v in sv_err[:16]],
                   err_rank_1e_6=int((sv_err > 1e-6 * max(sv_err[0], 1e-300)).sum()),
                   sigma_weak_rel=[float(f"{v / sig_max:.2e}") for v in sig_weak], sigma_max=sig_max)
        if "solve" in opts:
            tol = max(10 * F0, 1e-28)
            for jac in ("dd", "f64"):
                t = time.time()
                try:
                    _, _, hist = newton_solve(s, ph, pl, tol=tol, it_max=12, verbose=False, jac=jac)
                    rec[f"solve_{jac}"] = dict(its=len(hist), F=float(hist[-1].get("F", np.nan)) if hist else None,
                                               t=round(time.time() - t, 1))
                except Exception as e:  # noqa: BLE001
                    rec[f"solve_{jac}"] = dict(error=str(e)[:200])
        out.append(rec)
        print(f"D = {m['D']:8.4f} M {m['M']} N {n}{' kf13' if kf else ''}: |F| {F0:.1e}; J_f64 vs J_hi {jdiff:.1e}; "
              f"step error f64 LU {c_f64lu:.1e}, mixed {c_mixed:.1e}, mixed on a dd base {c_mixed_ddbase:.1e} "
              f"(J {jdiffb:.1e}; Jacobian dd {t_jdd:.2f} s, f64 {t_j64:.2f} s, f64 on dd base {t_j64b:.2f} s); error sv {sv_err[0]:.1e} {sv_err[1]:.1e} "
              f"{sv_err[2]:.1e} {sv_err[4]:.1e} {sv_err[8]:.1e} {sv_err[15]:.1e} (rank@1e-6 {rec['err_rank_1e_6']}); "
              f"sigma_min/max {sig_weak[0] / sig_max:.1e} {sig_weak[1] / sig_max:.1e} {sig_weak[3] / sig_max:.1e} "
              f"{sig_weak[7] / sig_max:.1e}" + (f"; solve dd {rec['solve_dd']} f64 {rec['solve_f64']}" if "solve" in opts else ""),
              flush=True)
    if "json" in opts:
        json.dump(out, open(opts["json"], "w"), indent=1)


if __name__ == "__main__":
    main()
