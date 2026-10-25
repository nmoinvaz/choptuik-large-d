#!/usr/bin/env python3
"""Do the large-D values of Delta in hp/data/delta_dd.csv hold up against the exact residual? For every row of run
T48-up70 (D = 61.66-67.77, M = 32), at its saved state (data/large_d/up70_D<D>.npz):

    cd hp && python3 qdh_delta_check.py [--threads=N] [--out=data/large_d/qdh_delta_check.jsonl]

the linearised Delta correction -y.F (y = J^-T e_Delta with the double-double Jacobian's LU, the dot product in
double-double) from the residual with the horizon block in quad-double (Solver.residual_qdh, which removes the
horizon rows' rounding, notes/QUAD_DOUBLE.md section 11), next to the same correction from the double-double
residual and the row's stated error Delta_err. The solver is built as the scan's: the state's kt, kc, ks, kfreeze 13,
the Taylor plan at tol_step 1e-32 and hmax 0.05.

Version 2 (after T56/T57, comms 2026-10-25 22:22 and 22:5x UTC): the first version's control moved Delta alone by
1e-14, which is far outside the linear regime (a pure Delta move is linear only below about 1e-21 at D 61.66: its
second-order residual feeds the near-null directions), and its f64 dot product lost about three digits to
cancellation for such a move.
Version 3 (after T60, comms 2026-10-25 23:27 UTC): version 2's control B (the state moved along t = dz/dD from
driver.tangent) is dropped. At D 61.66 that tangent has |t| 9.8e18 and t_Delta -2.5e11: the central difference dF/dD
lies partly along the near-null directions (pivots down to 8.7e-15), which J^-1 amplifies, so the state + 1e-12 t is
1e7 away and its residual is NaN; no eps puts that move in the linear regime. The control is now A alone, run first,
and its planted residual also goes through the row criterion, which must reject it:
  A. F_q + delta J e_Delta (J e_Delta from Solver.jvp, delta = 1e-14, more than Delta_err) at the first row's state
     must give the correction minus delta, to 1e-6 relative (it tests the LU solve and the dot product, linear by
     construction), and |correction| > Delta_err there (the row criterion fails on it).
Pass (criteria written before the run): control A as stated, and every row |dDelta_q| <= Delta_err. Exit 1 on a FAIL.
"""
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import _two_prod as two_prod, dd_axpy, load, make_solver  # noqa: E402


def dot_dd(xh, xl, yh, yl):
    """sum (xh + xl)(yh + yl), the products' exact parts and the cross terms summed exactly rounded"""
    import math
    p, e = two_prod(np.asarray(xh, dtype=float), np.asarray(yh, dtype=float))
    terms = np.concatenate([p, e, np.asarray(xh) * np.asarray(yl) + np.asarray(xl) * np.asarray(yh)])
    return math.fsum(terms)


def y_delta(s):
    """y = J^-T e_Delta (double-double) with the LU of the last s.newton"""
    L = s.size
    e = np.zeros(L)
    e[-1] = 1.0
    yh, yl = s.lu_solve_t(e, np.zeros(L))
    return np.asarray(yh), np.asarray(yl)


def correction(y, F):
    return -dot_dd(y[0], y[1], np.asarray(F[0]), np.asarray(F[1]))


def solver(meta, zh, zl, threads):
    s = make_solver(meta["D"], meta["M"], nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=meta["ks"],
                    threads=threads, kfreeze=13)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    return s


def corrections(s, zh, zl):
    """(dDelta from the double-double F, dDelta from the quad-double-horizon F, |F_dd|, |F_q|), y from s.newton here"""
    _, _, Fh, Fl, _, _ = s.newton(zh, zl)
    y = y_delta(s)
    z = np.zeros(len(zh))
    qh, ql = s.residual_qdh(zh, zl, z, z)
    return (correction(y, (Fh, Fl)), correction(y, (qh, ql)), float(np.abs(np.asarray(Fh) + np.asarray(Fl)).max()),
            float(np.abs(np.asarray(qh) + np.asarray(ql)).max()))


def main():
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    threads = int(opts.get("threads", 2))
    out = opts.get("out", os.path.join(HERE, "data", "large_d", "qdh_delta_check.jsonl"))
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_dd.csv"))) if r["run"] == "T48-up70"]
    files = {round(float(re.search(r"D([0-9.]+)\.npz", f).group(1)), 4): f
             for f in glob.glob(os.path.join(HERE, "data", "large_d", "up70_D*.npz"))}
    print(f"{len(rows)} rows of T48-up70 in delta_dd.csv", flush=True)
    bad, recs = [], []
    t0 = time.time()
    # control A at the first row's state (first, so that a failing control does not lose the rows)
    zh, zl, meta = load(files[round(float(rows[0]["D"]), 4)])
    zh, zl = np.ascontiguousarray(zh), np.ascontiguousarray(zl)
    s = solver(meta, zh, zl, threads)
    _, c0, _, _ = corrections(s, zh, zl)          # leaves the LU at the state
    y = y_delta(s)
    z = np.zeros(len(zh))
    Fq = s.residual_qdh(zh, zl, z, z)
    e = np.zeros(len(zh))
    e[-1] = 1.0
    jh, jl = s.jvp(zh, zl, e, z)[:2]
    delta = 1e-14
    Ph, Pl = dd_axpy(delta, np.asarray(jh), np.asarray(jl), np.asarray(Fq[0]), np.asarray(Fq[1]))
    cA = correction(y, (Ph, Pl))
    errA = float(rows[0]["Delta_err"])
    okA = abs((cA - c0) / -delta - 1) <= 1e-6 and not abs(cA) <= errA
    print(f"  control A (F + 1e-14 J e_Delta at D {meta['D']}): (correction change) / (-1e-14) = "
          f"{(cA - c0) / -delta:.9f}, |correction| {abs(cA):.2e} against Delta_err {errA:.0e} "
          f"{'ok (the row criterion rejects it)' if okA else 'WRONG'}", flush=True)
    if not okA:
        bad.append("control A did not recover the planted residual, or the row criterion accepted it")
    for i, r in enumerate(rows):
        D = float(r["D"])
        f = files.get(round(D, 4))
        if f is None:
            bad.append(f"D {D}: no saved state")
            continue
        zh, zl, meta = load(f)
        zh, zl = np.ascontiguousarray(zh), np.ascontiguousarray(zl)
        if abs(meta["D"] - D) > 1e-9:
            bad.append(f"D {D}: the state is at D {meta['D']}")
            continue
        s = solver(meta, zh, zl, threads)
        d_dd, d_q, f_dd, f_q = corrections(s, zh, zl)
        err = float(r["Delta_err"])
        rec = dict(D=D, dDelta_q=d_q, dDelta_dd=d_dd, csv_dDelta_est=float(r["dDelta_est"]), Delta_err=err, F_dd=f_dd,
                   F_q=f_q, ks=meta["ks"], kc=meta["kc"])
        recs.append(rec)
        if not abs(d_q) <= err:
            bad.append(f"D {D}: |dDelta_q| {abs(d_q):.1e} > Delta_err {err:.0e}")
        if i % 20 == 0 or not abs(d_q) <= err:
            print(f"  D {D:.6f}: dDelta_q {d_q:+.2e}, dd {d_dd:+.2e} (csv {float(r['dDelta_est']):+.2e}), "
                  f"|F| q {f_q:.1e} dd {f_dd:.1e}  ({i + 1}/{len(rows)}, {time.time() - t0:.0f} s)", flush=True)
    q = np.array([abs(r["dDelta_q"]) for r in recs])
    dd = np.array([abs(r["dDelta_dd"]) for r in recs])
    summ = dict(rows=len(rows), checked=len(recs), max_dDelta_q=float(q.max()) if len(q) else None,
                median_dDelta_q=float(np.median(q)) if len(q) else None,
                max_dDelta_dd=float(dd.max()) if len(dd) else None, control_A=(cA - c0) / -delta,
                fails=bad, sec=round(time.time() - t0, 1), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"{len(recs)} rows: |dDelta_q| max {summ['max_dDelta_q']:.2e}, median {summ['median_dDelta_q']:.2e}; "
          f"|dDelta_dd| max {summ['max_dDelta_dd']:.2e}", flush=True)
    with open(out, "a") as fo:
        fo.write(json.dumps(dict(summary=summ, rows=recs)) + "\n")
    print("PASS" if not bad else "FAIL: " + "; ".join(bad[:10]), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
