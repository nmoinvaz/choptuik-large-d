"""Costs of the pieces of a multiprecision Newton step of the RT prototype on this machine, for the choice of
rt_mpsolve's design (notes/RT_FORMULATION.md): on an RT grid, the f64 residual, the arb residual, the f64 Jacobian,
its f64 LU, one double-double triangular solve pair (rt_mpsolve.LUdd) and one product J u (rt_mpsolve.jv_central); and
multidouble's dense double-double LU solve (dd_solve) at small sizes, as seconds / L^3.

    python3 solver/rt/rt_mp_cost.py OUT.jsonl --state RT.npz --grid 32x81 [--prec 192] [--sizes 500,1000,2000]
        [--mutate=perturb]

Controls: the dd_solve of a random well-conditioned system must return its planted solution to 1e-28 relative, and
the LUdd solve must have a normwise backward error below 1e-26 against the factors it applies,
|U^T L^T P^T x - b| / (|U^T| |L^T| |x| + |b|) in the infinity norm, the products in double-double. (Against J64 itself
the residual is that of E x, |E| near 2^-53 |J|, which is not the solve's error: the first version of this check used
J64 and failed with 2.5e-14, I105-mac/controls/mp_cost_32x81.log.) A failure exits 1. --mutate=perturb changes one
entry of the dd_solve matrix by 1e-10 after the right-hand side is formed, and the first pivot of the LU factors by
1e-10 relative after the solve; both checks must then fail (exit 1). Needs MULTIDOUBLE_SITE as rt_mpsolve does.
"""
import os
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
import json
import time
import argparse
import numpy as np
import flint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_model import RT
from rt_mp import RTmp
import rt_mpsolve as mps

arb = flint.arb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--state", required=True)
    ap.add_argument("--grid", default="32x81")
    ap.add_argument("--prec", type=int, default=192)
    ap.add_argument("--sizes", default="500,1000,2000")
    ap.add_argument("--mutate", default=None, choices=["perturb"])
    a = ap.parse_args()
    bad = False
    recs = []
    rng = np.random.default_rng(105)
    # ------------------------------------------------ dense dd LU (multidouble.dd_solve)
    for n in [int(v) for v in a.sizes.split(",")]:
        A = rng.normal(size=(n, n)) + np.sqrt(n) * np.eye(n)
        Al = np.zeros((n, n))
        xs = rng.normal(size=n)
        bh, bl = mps.md.dd_matvec(A, Al, xs, np.zeros(n))
        if a.mutate:
            A[0, 0] += 1e-10
        t = time.time()
        xh, xl = mps.md.dd_solve(np.ascontiguousarray(A), Al, bh, bl)
        dt = time.time() - t
        err = float(np.max(np.abs((xh - xs) + xl)) / np.max(np.abs(xs)))
        ok = err < 1e-28
        bad |= not ok
        recs.append(dict(what="dd_solve", n=n, secs=round(dt, 3), per_L3=dt / n ** 3, err=err, ok=ok))
        print(f"dd_solve n {n}: {dt:.3f} s, {dt / n ** 3:.2e} s/L^3, error {err:.1e} ({'ok' if ok else 'FAIL'})")
    # ------------------------------------------------ the RT pieces on one grid
    d = np.load(a.state)
    src = RT(float(d["D"]), int(d["Nt"]), int(d["N"]))
    Nt, N = (int(v) for v in a.grid.split("x"))
    rt = RT(float(d["D"]), Nt, N)
    z = d["zh"] if "zh" in d else d["z"]
    z = z if (src.Nt, src.N) == (Nt, N) else src.resample_to(z, rt)
    flint.ctx.prec = a.prec
    rm = RTmp(rt.D, Nt, N, prec=a.prec)
    zo = mps.obj(z)
    t = time.time(); rt.residual(z); t_f = time.time() - t
    t = time.time(); rm.residual_m(zo); t_m = time.time() - t
    t = time.time(); J = rt.jacobian(z); t_j = time.time() - t
    t = time.time(); P = mps.LUdd(J, overwrite=True); t_lu = time.time() - t
    del J
    u = mps.obj(rng.normal(size=rt.L))
    t = time.time(); b = mps.jv_central(rm.residual_m, zo, u); t_p = time.time() - t
    bh, bl = mps.to_dd(b)
    t = time.time(); xh, xl = P.solve_dd(bh, bl); t_s = time.time() - t
    if a.mutate:
        P.M[0, 0] *= 1 + 1e-10
    # the backward error against the factors: y = U^T (L^T (P^T x)) in double-double, row by row
    M, n, dd_dot = P.M, rt.L, mps.md._dd_dot
    zero = np.zeros(n)
    wh, wl = xh[P.perm], xl[P.perm]
    vh, vl = np.empty(n), np.empty(n)
    yh, yl = np.empty(n), np.empty(n)
    nU = nL = 0.0
    for i in range(n):
        sh, sl_ = dd_dot(M[i, i + 1:], zero[i + 1:], wh[i + 1:], wl[i + 1:])
        vh[i], vl[i] = mps._dd_sub(wh[i], wl[i], -sh, -sl_)       # w_i + sum
        nL = max(nL, 1.0 + float(np.abs(M[i, i + 1:]).sum()))
        nU = max(nU, float(np.abs(M[i, :i + 1]).sum()))
    for i in range(n):
        yh[i], yl[i] = dd_dot(M[i, :i + 1], zero[:i + 1], vh[:i + 1], vl[:i + 1])
    rh, rl = mps.md.dd_add(yh, yl, -bh, -bl)
    eta = float(np.max(np.abs(rh + rl))) / (nU * nL * float(np.max(np.abs(xh + xl))) + float(np.max(np.abs(bh))))
    ok = eta < 1e-26
    bad |= not ok
    recs.append(dict(what="rt", grid=[Nt, N], L=rt.L, prec=a.prec, t_residual_f64=round(t_f, 4),
                     t_residual_arb=round(t_m, 3), t_jacobian_f64=round(t_j, 2), t_lu_f64=round(t_lu, 2),
                     t_ludd_solve=round(t_s, 3), t_product=round(t_p, 3), ludd_backward_error=eta, ok=ok,
                     pivot_ratio=P.pivot_ratio))
    print(f"RT {Nt}x{N} L {rt.L}: f64 residual {t_f:.4f} s, arb residual ({a.prec} bits) {t_m:.3f} s, f64 Jacobian "
          f"{t_j:.2f} s, f64 LU {t_lu:.2f} s, dd triangular solves {t_s:.3f} s (backward error {eta:.1e}, "
          f"{'ok' if ok else 'FAIL'}), product J u {t_p:.3f} s, LU pivot ratio {P.pivot_ratio:.1e}")
    with open(a.out, "a") as f:
        for r in recs:
            r["mutate"] = a.mutate
            f.write(json.dumps(r) + "\n")
    print(f"exit {1 if bad else 0}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
