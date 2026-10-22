#!/usr/bin/env python3
"""Where one double-double Newton step spends its time, and whether the Jacobian's structure or a lower-precision
factorization could cut the cubic cost (the "cheap first step" before optimizing the solver).

For each saved state (default: the D = 3.7 low-D states at M = 48, 64, 80) it times, on this machine:
  - the residual F(z) (s.residual),
  - the exact Jacobian by forward-mode duals (s.jacobian),
  - a full Newton step (s.newton = residual + Jacobian + double-double LU + solve); the LU part is the difference;
  - an f64 LU of the Jacobian's high part with LAPACK (scipy.linalg.lu_factor), for the mixed-precision estimate.
It also reports the Jacobian's structure (the fraction of exactly-zero entries, overall and in node blocks) and, for
mixed precision, how far one Newton step solved with the f64 LU differs from the double-double step (that relative
difference is the contraction factor a Newton iteration with an f64 factorization and double-double residuals would
have; well below 1 means it converges to full double-double accuracy, just in more steps).

    cd hp && python3 profile_newton.py [state.npz ...] [--threads=2] [--json=out.json]
"""
import json
import os
import sys
import time

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import load, make_solver  # noqa: E402

DEFAULT = [os.path.join(HERE, "data", "lowD_states", f"lowDv_D3.7000_M{M}.npz") for M in (48, 64, 80)]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    threads = int(opts.get("threads", 2))
    out = []
    for path in args or DEFAULT:
        zh, zl, m = load(path)
        s = make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads)
        s.set_segs(m.get("segs") or s.plan(zh, zl, 1e-30, 0.05))
        n = s.size
        t = time.time(); Fh, Fl = s.residual(zh, zl); t_res = time.time() - t
        t = time.time(); Jh, Jl = s.jacobian(zh, zl); t_jac = time.time() - t
        t = time.time(); dh, dl, _, _, pmin, pmax = s.newton(zh, zl); t_newton = time.time() - t
        t_lu = t_newton - t_res - t_jac
        t = time.time(); lu = sla.lu_factor(Jh, check_finite=False); t_f64 = time.time() - t
        x64 = sla.lu_solve(lu, -Fh, check_finite=False)
        dd = dh + dl
        contraction = float(np.linalg.norm(x64 - dd) / max(np.linalg.norm(dd), 1e-300))
        zero = float((Jh == 0).mean())
        nn = len(m["nodes"])
        bs = n // nn if nn else n
        blocks = np.zeros((nn, nn))
        for i in range(nn):
            for j in range(nn):
                blocks[i, j] = np.count_nonzero(Jh[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs]) / (bs * bs)
        nz_blocks = int((blocks > 0).sum())
        rec = dict(state=os.path.basename(path), D=m["D"], M=m["M"], N=n, threads=threads,
                   t_residual=round(t_res, 2), t_jacobian=round(t_jac, 2), t_newton=round(t_newton, 2),
                   t_dd_lu_and_solve=round(t_lu, 2), t_f64_lu=round(t_f64, 3),
                   f64_step_relative_difference=contraction, pivot_ratio=pmin / pmax,
                   zero_fraction=round(zero, 4), node_blocks=nn, block_size=bs, nonzero_blocks=nz_blocks,
                   block_density=[[round(v, 3) for v in row] for row in blocks])
        out.append(rec)
        print(f"M = {m['M']:3d}, N = {n}: residual {t_res:7.1f} s, Jacobian {t_jac:7.1f} s, dd LU + solve {t_lu:7.1f} s "
              f"(Newton step {t_newton:.1f} s); f64 LU {t_f64:.2f} s; f64-vs-dd step difference {contraction:.1e}; "
              f"pivots {pmin:.1e}..{pmax:.1e}; zero entries {zero:.1%}; nonzero {nn}x{nn} node blocks: {nz_blocks}",
              flush=True)
    if len(out) >= 2:
        Ms = np.log([r["M"] for r in out])
        for k in ("t_residual", "t_jacobian", "t_dd_lu_and_solve", "t_f64_lu"):
            ts = np.log([max(r[k], 1e-3) for r in out])
            print(f"  {k}: grows like M^{np.polyfit(Ms, ts, 1)[0]:.2f}")
    if "json" in opts:
        json.dump(out, open(opts["json"], "w"), indent=1)


if __name__ == "__main__":
    main()
