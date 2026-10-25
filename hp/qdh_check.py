#!/usr/bin/env python3
"""Checks of the residual with the horizon block in quad-double (Solver.residual_qdh, driver.QdhSolver;
notes/QUAD_DOUBLE.md section 11) at a large-D state where the double-double residual has its floor.

    cd hp && python3 qdh_check.py [STATE.npz] [--threads=N] [--its=6] [--out=data/large_d/qdh_check.jsonl]

STATE defaults to data/large_d/up70_D67.7224.npz (T48's ks 130 state, |F| 5.9e-22 against tol 1.4e-21); the solver is
built from its meta (kt, kc, ks) with kfreeze 13 and the scan's Taylor plan (tol_step 1e-32, hmax 0.05). The pass
criteria were written before the first run.

A. Agreement at the double-double state (z2 = z3 = 0): every row outside the horizon block bitwise the double-double
   residual's; the horizon rows within 10 S_dd of it (S_dd from B: the double-double residual's own noise there).
B. Noise: F at z + t xi, t = 0, 1, 2, xi random with |xi_j| ~ 1e-30 |z_j| (Delta included), each state made in
   quad-double (qd_add_dd); S = max |F(2) - 2 F(1) + F(0)| (zero for a smooth F at this scale, so S is rounding).
   S_dd: the double-double residual (the state's first two parts); S_q: residual_qdh; S_c: residual_qdh with the
   horizon solves unrefined (the control). Pass: S_q <= 1e-3 S_dd, and S_c >= 1e3 S_q (the refinement is what
   removes it).
C. Newton (driver.newton_solve, jac dd, it_max --its, tol 1e-40) from the state: with QdhSolver the final |F| must be
   <= 1e-25 and the linearised Delta correction |dDelta_est| <= 1e-22, Delta within 1e-14 of the state's (the
   double-double runs' dDelta_est there are 1e-17..1e-15). Control: QdhSolver(refine=False) must stay above 1e-23.
D. residual_qdh twice at the same state: bitwise the same.
Exit 1 if a check fails or the control passes.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel import qd_add_dd  # noqa: E402
from hpkernel.driver import QdhSolver, dd_norm, delta_correction, load, make_solver, newton_solve  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    path = args[0] if args else os.path.join(HERE, "data", "large_d", "up70_D67.7224.npz")
    out = opts.get("out", os.path.join(HERE, "data", "large_d", "qdh_check.jsonl"))
    its = int(opts.get("its", 6))
    log = lambda m: print(m, flush=True)  # noqa: E731
    zh, zl, meta = load(path)
    zh, zl = np.ascontiguousarray(zh), np.ascontiguousarray(zl)
    D, M, nodes = meta["D"], meta["M"], meta["nodes"]
    s = make_solver(D, M, nodes=nodes, kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], threads=int(opts.get("threads", 2)),
                    kfreeze=13)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    L, K = s.size, len(nodes)
    hb = slice(K * 4 * M, (K + 1) * 4 * M)
    out_rows = np.ones(L, bool)
    out_rows[hb] = False
    zero = np.zeros(L)
    bad = []
    rec = dict(state=os.path.basename(path), D=D, M=M, K=K, kt=meta["kt"], kc=meta["kc"], ks=meta["ks"])
    log(f"D {D}: M {M}, K {K}, ks {meta['ks']}, kc {meta['kc']}; L {L}")

    def F(fn, z):
        h, l_ = fn(z)
        return np.asarray(h) + np.asarray(l_), (np.asarray(h), np.asarray(l_))

    dd_f = lambda z: s.residual(z[0], z[1])  # noqa: E731
    q_f = lambda z: s.residual_qdh(*z)  # noqa: E731
    c_f = lambda z: s.residual_qdh(*z, refine=False)  # noqa: E731

    # A and D
    t = time.time()
    fd, (fdh, fdl) = F(dd_f, (zh, zl))
    t_dd = time.time() - t
    t = time.time()
    fq, (fqh, fql) = F(q_f, (zh, zl, zero, zero))
    t_q = time.time() - t
    fq2, (fqh2, fql2) = F(q_f, (zh, zl, zero, zero))
    same_out = bool(np.array_equal(fdh[out_rows], fqh[out_rows]) and np.array_equal(fdl[out_rows], fql[out_rows]))
    dA = float(np.abs((fqh[hb] - fdh[hb]) + (fql[hb] - fdl[hb])).max())
    det = bool(np.array_equal(fqh, fqh2) and np.array_equal(fql, fql2))
    log(f"  A. rows outside the horizon block bitwise the double-double residual's: {same_out}; horizon rows differ by "
        f"{dA:.2e} (max |F| there: dd {np.abs(fd[hb]).max():.2e}, qdh {np.abs(fq[hb]).max():.2e}); residual {t_dd:.2f} s "
        f"(dd), {t_q:.2f} s (qdh)")
    log(f"  D. residual_qdh twice: bitwise the same: {det}")
    if not det:
        bad.append("check D failed (residual_qdh not reproducible)")

    # B
    rng = np.random.default_rng(7)
    xi = 1e-30 * np.abs(zh) * rng.standard_normal(L)
    states = []
    for k in range(3):
        states.append(tuple(np.asarray(v) for v in qd_add_dd(zh, zl, zero, zero, k * xi, zero)))
    S = {}
    for name, fn in (("dd", dd_f), ("qdh", q_f), ("ctl", c_f)):
        f = [F(fn, z)[0] for z in states]
        sec = f[2] - 2 * f[1] + f[0]
        S[name] = float(np.abs(sec).max())
        S[name + "_horizon"] = float(np.abs(sec[hb]).max())
        S[name + "_outside"] = float(np.abs(sec[out_rows]).max())
    log(f"  B. second differences at 1e-30: double-double {S['dd']:.2e} (horizon rows {S['dd_horizon']:.2e}, others "
        f"{S['dd_outside']:.2e}); qdh {S['qdh']:.2e} ({S['qdh_horizon']:.2e}, {S['qdh_outside']:.2e}); control, unrefined "
        f"{S['ctl']:.2e} ({S['ctl_horizon']:.2e})")
    rec["A"] = dict(outside_bitwise=same_out, horizon_diff=dA, bound=10 * S["dd_horizon"], t_dd=t_dd, t_qdh=t_q)
    rec["B"] = S
    rec["D"] = det
    if not same_out or not dA <= 10 * S["dd_horizon"]:
        bad.append("check A failed")
    if not S["qdh"] <= 1e-3 * S["dd"]:
        bad.append("check B failed (the quad-double horizon is not quieter)")
    if not S["ctl"] >= 1e3 * S["qdh"]:
        bad.append("control B passed (unrefined solves as quiet)")

    # C
    runs = {}
    for name, refine in (("qdh", True), ("ctl", False)):
        qs = QdhSolver(s, refine=refine)
        t = time.time()
        zh2, zl2, hist = newton_solve(qs, zh, zl, tol=1e-40, it_max=its, verbose=True)
        Fh, Fl = qs.residual(zh2, zl2)
        z2, z3 = qs.parts(zh2, zl2)
        nF = dd_norm(Fh, Fl)
        s.newton(zh2, zl2)          # the LU at the final state for the Delta correction
        dd_est = delta_correction(qs, Fh, Fl)
        dDelta = float((zh2[-1] - zh[-1]) + (zl2[-1] - zl[-1]) + z2[-1])
        runs[name] = dict(F=nF, F_hist=[r["F"] for r in hist], dDelta_est=dd_est, Delta=float(zh2[-1]),
                          Delta_lo=float(zl2[-1]), Delta_q2=float(z2[-1]), Delta_change=dDelta, sec=round(time.time() - t, 1))
        log(f"  C. {name}: |F| {nF:.2e} after {len(hist)} steps ({', '.join(f'{r:.1e}' for r in runs[name]['F_hist'])}); "
            f"dDelta_est {dd_est:+.2e}; Delta moved by {dDelta:+.2e} ({runs[name]['sec']} s)")
    rec["C"] = runs
    if not (runs["qdh"]["F"] <= 1e-25 and abs(runs["qdh"]["dDelta_est"]) <= 1e-22 and abs(runs["qdh"]["Delta_change"]) <= 1e-14):
        bad.append("check C failed")
    if not runs["ctl"]["F"] > 1e-23:
        bad.append("control C passed (the unrefined horizon converged as far)")

    rec["fails"] = bad
    with open(out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
    log("PASS" if not bad else "FAIL: " + "; ".join(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
