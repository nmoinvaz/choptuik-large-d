#!/usr/bin/env python3
"""One measurement for notes/HIGH_M.md section 1.4 and notes/LARGE_D_MIXED.md: does the mixed-precision Newton step
with the correction loop (driver.newton_solve(jac="f64g"): the f64 Jacobian in block storage, its f64 LU as the
preconditioner of GMRES in GCR form with exact double-double Jacobian-vector products, Solver.jvp) converge the
M = 192 lift at D = 3.7 from the state where plain f64-Jacobian Newton stalled
(hp/data/lowD_states/highm_p1_D3.7000_M192_stalled.npz, |F| 3.75e-10, its own 511-step plan)?

    cd hp && python3 highm_gmres_check.py [--state=...] [--tol=1e-28] [--it_max=6] [--threads=2] [--step_tol=1e-10]
                                          [--max_products=12] [--out=data/mixed_precision/largeD_mixed/M192_f64g]
Writes <out>.json (the Newton history: |F|, step, line-search factor, products, changes, seconds by part) after every
step, and <out>_D3.7000_M192.npz if it converges.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from hpkernel.driver import dd_norm, load, make_solver, newton_solve, save  # noqa: E402


def main():
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    path = opts.get("state", os.path.join(HERE, "data", "lowD_states", "highm_p1_D3.7000_M192_stalled.npz"))
    tol, it_max, threads = float(opts.get("tol", 1e-28)), int(opts.get("it_max", 6)), int(opts.get("threads", 2))
    g = dict(step_tol=float(opts.get("step_tol", 1e-10)), max_products=int(opts.get("max_products", 12)))
    out = opts.get("out", os.path.join(HERE, "data", "mixed_precision", "largeD_mixed", "M192_f64g"))
    zh, zl, m = load(path)
    s = make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads)
    s.set_segs(m["segs"])
    F0 = dd_norm(*s.residual(zh, zl))
    print(f"{os.path.basename(path)}: D {m['D']} M {m['M']} L {s.size}, {sum(len(q) - 1 for q in m['segs'])} Taylor steps, "
          f"|F| {F0:.2e}; newton_solve(jac='f64g', gmres={g}), tol {tol:g}", flush=True)
    hist = []
    t0 = time.time()
    # one Newton step per call, so that every step is recorded as soon as it is done
    for it in range(it_max):
        zh2, zl2, h = newton_solve(s, zh, zl, tol=tol, it_max=1, verbose=True, jac="f64g", gmres=g)
        h[0]["it"] = it
        hist += h
        json.dump(dict(state=os.path.basename(path), F0=F0, tol=tol, gmres=g, hist=hist, wall=round(time.time() - t0, 1)),
                  open(out + ".json", "w"), indent=1)
        r = h[-1]
        print(f"  step {it}: |F| {r['F']:.2e}, products {r.get('products')}, changes {r.get('changes')}, "
              f"Jacobian {r.get('t_jac', 0):.0f} s, LU {r.get('t_lu', 0):.0f} s, products {r.get('t_prod', 0):.1f} s"
              + (", retried with dd4" if r.get("retry_dd") else ""), flush=True)
        if r.get("failed"):
            print("  line search failed: stop", flush=True)
            break
        zh, zl = zh2, zl2
        if r["F"] < tol:
            save(out + f"_D{m['D']:.4f}_M{m['M']}.npz", np.asarray(zh), np.asarray(zl), dict({k: v for k, v in m.items() if k != "it"},
                                                                                        F=r.get("F_polished", r["F"])))
            print(f"converged in {it + 1} Newton calls, {time.time() - t0:.0f} s", flush=True)
            break


if __name__ == "__main__":
    main()
