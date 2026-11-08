"""Costs of the pieces of a multiprecision Newton step of the RT prototype on this machine, for the choice of
rt_mpsolve's design (notes/RT_FORMULATION.md): on an RT grid, the f64 residual, the arb residual, the f64 Jacobian,
its f64 LU, one double-double triangular solve pair (rt_mpsolve.LUdd) and one product J u (rt_mpsolve.jv_central); and
multidouble's dense double-double LU solve (dd_solve) at small sizes, as seconds / L^3.

    python3 solver/rt/rt_mp_cost.py OUT.jsonl --state RT.npz --grid 32x81 [--prec 192] [--sizes 500,1000,2000]
        [--mutate=perturb]

Controls: the dd_solve of a random well-conditioned system must return its planted solution to 1e-28 relative, and
the LUdd solve of the f64 Jacobian must satisfy J x = b to 1e-28 relative in arb; a failure exits 1. --mutate=perturb
changes one entry of each matrix by 1e-10 after the right-hand side is formed; both checks must then fail (exit 1).
Needs MULTIDOUBLE_SITE as rt_mpsolve does.
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
    Jc = J.copy()
    t = time.time(); P = mps.LUdd(J, overwrite=True); t_lu = time.time() - t
    del J
    u = mps.obj(rng.normal(size=rt.L))
    t = time.time(); b = mps.jv_central(rm.residual_m, zo, u); t_p = time.time() - t
    if a.mutate:
        P.M[0, 0] += 1e-10
    t = time.time(); x = P(b); t_s = time.time() - t
    # J64 x = b in arb (J64 is exact as a matrix of f64 numbers): the LU solve's accuracy
    Jx = mps.from_dd(*mps.md.dd_matvec(Jc, np.zeros_like(Jc), *mps.to_dd(x)))
    rel = mps.amax(Jx - b) / mps.amax(b)
    ok = rel < 1e-28
    bad |= not ok
    recs.append(dict(what="rt", grid=[Nt, N], L=rt.L, prec=a.prec, t_residual_f64=round(t_f, 4),
                     t_residual_arb=round(t_m, 3), t_jacobian_f64=round(t_j, 2), t_lu_f64=round(t_lu, 2),
                     t_ludd_solve=round(t_s, 3), t_product=round(t_p, 3), ludd_rel_residual=rel, ok=ok,
                     pivot_ratio=P.pivot_ratio))
    print(f"RT {Nt}x{N} L {rt.L}: f64 residual {t_f:.4f} s, arb residual ({a.prec} bits) {t_m:.3f} s, f64 Jacobian "
          f"{t_j:.2f} s, f64 LU {t_lu:.2f} s, dd triangular solves {t_s:.3f} s (|J x - b| / |b| = {rel:.1e}, "
          f"{'ok' if ok else 'FAIL'}), product J u {t_p:.3f} s")
    with open(a.out, "a") as f:
        for r in recs:
            r["mutate"] = a.mutate
            f.write(json.dumps(r) + "\n")
    print(f"exit {1 if bad else 0}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
