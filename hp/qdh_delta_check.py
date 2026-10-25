#!/usr/bin/env python3
"""Do the large-D values of Delta in hp/data/delta_dd.csv hold up against the exact residual? For every row of run
T48-up70 (D = 61.66-67.77, M = 32), at its saved state (data/large_d/up70_D<D>.npz):

    cd hp && python3 qdh_delta_check.py [--threads=N] [--out=data/large_d/qdh_delta_check.jsonl]

the linearised Delta correction -y.F (driver.delta_correction, y = J^-T e_Delta with the double-double Jacobian's LU)
from the residual with the horizon block in quad-double (Solver.residual_qdh, which removes the horizon rows' rounding,
notes/QUAD_DOUBLE.md section 11), next to the same correction from the double-double residual (what the scan
recorded as dDelta_est) and the row's stated error Delta_err. The solver is built as the scan's: the state's kt, kc,
ks, kfreeze 13, the Taylor plan at tol_step 1e-32 and hmax 0.05.

Pass (criteria written before the run): at every row |dDelta_q| <= Delta_err. Control: the first state with Delta
moved by +1e-14 must give a correction of -1e-14 within 10% (the estimator sees an error in Delta). Exit 1 on a FAIL.
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
from hpkernel.driver import dd_add, delta_correction, load, make_solver  # noqa: E402


def solver(meta, zh, zl, threads):
    s = make_solver(meta["D"], meta["M"], nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=meta["ks"],
                    threads=threads, kfreeze=13)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    return s


def corrections(s, zh, zl):
    """(dDelta from the double-double F, dDelta from the quad-double-horizon F, |F_dd|, |F_q|)"""
    _, _, Fh, Fl, _, _ = s.newton(zh, zl)
    d_dd = delta_correction(s, Fh, Fl)
    z = np.zeros(len(zh))
    qh, ql = s.residual_qdh(zh, zl, z, z)
    d_q = delta_correction(s, qh, ql)
    return d_dd, d_q, float(np.abs(np.asarray(Fh) + np.asarray(Fl)).max()), float(np.abs(np.asarray(qh) + np.asarray(ql)).max())


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
            print(f"  D {D:.6f}: dDelta_q {d_q:+.2e}, dd {d_dd:+.2e} (csv {float(r['dDelta_est']):+.2e}), |F| q {f_q:.1e} "
                  f"dd {f_dd:.1e}  ({i + 1}/{len(rows)}, {time.time() - t0:.0f} s)", flush=True)
    # control: Delta moved by +1e-14 at the first state
    zh, zl, meta = load(files[round(float(rows[0]["D"]), 4)])
    zh, zl = np.ascontiguousarray(zh), np.ascontiguousarray(zl)
    sh = np.zeros(len(zh))
    sh[-1] = 1e-14
    ph, pl = (np.ascontiguousarray(v) for v in dd_add(zh, zl, sh, np.zeros(len(zh))))
    s = solver(meta, ph, pl, threads)
    _, c_q, _, _ = corrections(s, ph, pl)
    _, base_q, _, _ = corrections(s, zh, zl)
    ok_c = abs((c_q - base_q) + 1e-14) <= 1e-15
    print(f"  control: Delta + 1e-14 at D {meta['D']}: correction {c_q:+.3e} (unshifted {base_q:+.2e}), "
          f"{'recovers the shift' if ok_c else 'DOES NOT recover the shift'}", flush=True)
    if not ok_c:
        bad.append("the control did not recover the planted Delta shift")
    q = np.array([abs(r["dDelta_q"]) for r in recs])
    dd = np.array([abs(r["dDelta_dd"]) for r in recs])
    summ = dict(rows=len(rows), checked=len(recs), max_dDelta_q=float(q.max()) if len(q) else None,
                median_dDelta_q=float(np.median(q)) if len(q) else None, max_dDelta_dd=float(dd.max()) if len(dd) else None,
                control=c_q - base_q, fails=bad, sec=round(time.time() - t0, 1), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"{len(recs)} rows: |dDelta_q| max {summ['max_dDelta_q']:.2e}, median {summ['median_dDelta_q']:.2e}; "
          f"|dDelta_dd| max {summ['max_dDelta_dd']:.2e}", flush=True)
    with open(out, "a") as fo:
        fo.write(json.dumps(dict(summary=summ, rows=recs)) + "\n")
    print("PASS" if not bad else "FAIL: " + "; ".join(bad[:10]), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
