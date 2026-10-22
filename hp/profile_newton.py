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

--mixed adds the mixed-precision Newton (notes/MIXED_PRECISION.md): the f64 Jacobian s.jacobian_f64 for each of
--variants (mixed_check.py's names: f64xN all-f64 lanes with N columns per evaluation, ddN a double-double base with N
f64 tangents; default f64x1,f64x4,dd4), the f64 LU of the f64 Jacobian (Solver.lu_factor(J, f64=True)), and one mixed
step s.newton_mixed (residual + f64 Jacobian + LU + solve) for each of --step_variants (default f64x4,f64x4+ddlu: the f64
Jacobian with the f64 LU and with the double-double LU).
--solve adds whole solves to |F| <= 1e-28 (driver.newton_solve) from the perturbed state of mixed_check.py converge
(Delta + 1e-6, fields + 1e-8 pseudo-random, seed 1), with s.newton and with s.newton_mixed for each of
--step_variants: seconds, Newton calls, and the Delta difference between the solutions.

    cd hp && python3 profile_newton.py [state.npz ...] [--threads=2] [--json=out.json] [--mixed] [--solve] [--variants=..] [--step_variants=..]
"""
import json
import os
import sys
import time

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import load, make_solver, newton_solve  # noqa: E402
from mixed_check import jac_kw, perturb, variant  # noqa: E402

DEFAULT = [os.path.join(HERE, "data", "lowD_states", f"lowDv_D3.7000_M{M}.npz") for M in (48, 64, 80)]


def whole_solve(s, zh, zl, tol=1e-28, var=None):
    """driver.newton_solve from (zh, zl) with s.newton (var None) or s.newton_mixed (variant var)"""
    t = time.time()
    if var is None:
        a, b, hist = newton_solve(s, zh, zl, tol=tol, it_max=12, verbose=False)
    else:
        a, b, hist = newton_solve(s, zh, zl, tol=tol, it_max=12, verbose=False, jac="f64", mixed=variant(var))
    sec = time.time() - t
    Fh, Fl = s.residual(a, b)
    return dict(seconds=round(sec, 1), newton_calls=len(hist), F=float(np.abs(np.asarray(Fh) + np.asarray(Fl)).max()),
                Delta=float(a[-1]), Delta_lo=float(b[-1]))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:].split("=", 1) + [True])[:2] for a in sys.argv[1:] if a.startswith("--"))
    threads = int(opts.get("threads", 2))
    variants = str(opts.get("variants", "f64x1,f64x4,dd4")).split(",")
    step_variants = str(opts.get("step_variants", "f64x4,f64x4+ddlu")).split(",")
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
        if opts.get("mixed"):
            del Jh, Jl, lu
            J = None
            for v in variants:
                t = time.time(); J = s.jacobian_f64(zh, zl, 0, **jac_kw(v)); rec[f"t_jacobian_f64_{v}"] = round(time.time() - t, 2)
            t = time.time(); s.lu_factor(J, None, True); rec["t_f64_blocked_lu"] = round(time.time() - t, 2)
            del J
            for v in step_variants:
                t = time.time(); s.newton_mixed(zh, zl, **variant(v)); rec[f"t_newton_mixed_{v}"] = round(time.time() - t, 2)
            print("  mixed: jacobian_f64 " + ", ".join(f"{v} {rec[f't_jacobian_f64_{v}']:.2f} s "
                  f"({t_jac / rec[f't_jacobian_f64_{v}']:.1f}x)" for v in variants)
                  + f"; f64 LU {rec['t_f64_blocked_lu']:.2f} s; newton_mixed "
                  + ", ".join(f"{v} {rec[f't_newton_mixed_{v}']:.2f} s ({t_newton / rec[f't_newton_mixed_{v}']:.1f}x)"
                              for v in step_variants), flush=True)
        if opts.get("solve"):
            ph, pl = perturb(zh, zl, 1e-6, 1e-8, 1)
            rec["solve_newton"] = whole_solve(s, ph, pl)
            line = f"  whole solve to 1e-28: newton {rec['solve_newton']['seconds']:.1f} s ({rec['solve_newton']['newton_calls']} calls)"
            for v in step_variants:
                r = whole_solve(s, ph, pl, var=v)
                r["Delta_rel_diff"] = abs((r["Delta"] - rec["solve_newton"]["Delta"]) + (r["Delta_lo"] - rec["solve_newton"]["Delta_lo"])) / r["Delta"]
                rec[f"solve_mixed_{v}"] = r
                line += (f"; newton_mixed {v} {r['seconds']:.1f} s ({r['newton_calls']} calls, "
                         f"{rec['solve_newton']['seconds'] / r['seconds']:.1f}x, Delta differs by {r['Delta_rel_diff']:.0e})")
            print(line, flush=True)
    if len(out) >= 2:
        Ms = np.log([r["M"] for r in out])
        keys = ["t_residual", "t_jacobian", "t_dd_lu_and_solve", "t_f64_lu"]
        keys += [k for k in out[0] if k.startswith("t_jacobian_f64") or k.startswith("t_newton_mixed") or k == "t_f64_blocked_lu"]
        for k in keys:
            ts = np.log([max(r[k], 1e-3) for r in out])
            print(f"  {k}: grows like M^{np.polyfit(Ms, ts, 1)[0]:.2f}")
    if "json" in opts:
        json.dump(out, open(opts["json"], "w"), indent=1)


if __name__ == "__main__":
    main()
