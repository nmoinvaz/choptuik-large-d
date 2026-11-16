#!/usr/bin/env python3
"""One step of the exact Taylor continuation in D on the low-D branch, checked against lowD_highM.py's converged
state at the target (#927, notes/LOW_D_ANM.md 6). The step follows anm_run.py's rule for the plan: the base's plan is
taken once, and the series, the prediction's residual and the corrector all use it; nothing is planned from the
predicted state (lowD_taylor_leg.py did that, and its plans froze near D 3.0425, notes/LOW_D.md 15-16).

    python3 anm_lowd_pilot.py RAWDIR OUTDIR --state BASE.npz --target D --ref REF.npz [--N 8] [--threads 4]
        [--newton_its 6] [--plan_tol 1e-32] [--tol_pred 1e-12] [--tol_delta 1e-12] [--stop_on_fail 1]
        [--mutate none|drop2|kappa|dshift]

BASE and REF are relative to RAWDIR (or absolute), both at the same M, nodes and series orders; REF is lowD_highM.py's
converged state at D. The step:
  1. the plan: the base state's own at its D with --plan_tol (lowD_highM.py's plan_prev, the step tolerance 1e-30 times
     its --plan_margin 1e-2), and on it the base's f64 Jacobian and LU (Solver.newton_mixed);
  2. the series z_1..z_N (anm_lowd.coefficients: Solver.residual_jet's right-hand sides, gmres_refine with the f64 LU
     and exact double-double products, stopped at a change of 1e-20);
  3. the prediction P = sum z_k h^k at h = D - D0, its residual on the fixed plan at D, and that of the base state
     itself there (the zeroth-order predictor);
  4. the corrector: Newton with the f64 Jacobian (hpkernel.driver.newton_solve, jac "f64", as lowD_dd.solve_point) on
     the fixed plan, to |F| below 1e-28 in at most --newton_its calls; converged below 1e-26.
The agreement check, fixed before the run: (A) |Delta(P) - Delta(REF)| <= --tol_pred, and (B) the corrector converged
and |Delta(corrected) - Delta(REF)| <= --tol_delta. Exit 0 when both pass, 1 otherwise (with --stop_on_fail 1 a failed
A ends the step before the corrector). The corrected state is saved as OUTDIR/anm_D<D>_M<M>.npz with the plan, the
base of a next step. Records: OUTDIR/pilot.jsonl.
Mutations, the controls of the check (each must fail it): drop2, z_2 left out of the sum (a dropped term); kappa, the
jets with kappa's dependence on n frozen (model::JET_MUT_KAPPA, a wrong D derivative of the horizon exponents); dshift,
the step taken to D + 1e-9 (series and corrector), so that (B) compares solutions 1e-9 apart in D.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from anm_lowd import coefficients, evaluate, fmax, solver_for  # noqa: E402
from hpkernel.driver import load, newton_solve, save  # noqa: E402
from raw_out import FLAG, HELP, require_raw  # noqa: E402

JET_MUT_KAPPA = 22
F_CONV = 1e-26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--state", required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--newton_its", type=int, default=6)
    ap.add_argument("--plan_tol", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--step_tol", type=float, default=1e-20)
    ap.add_argument("--tol_pred", type=float, default=1e-12)
    ap.add_argument("--tol_delta", type=float, default=1e-12)
    ap.add_argument("--stop_on_fail", type=int, default=1)
    ap.add_argument("--mutate", choices=("none", "drop2", "kappa", "dshift"), default="none")
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    rec_path = os.path.join(a.out, "pilot.jsonl")
    require_raw(rec_path)
    os.makedirs(a.out, exist_ok=True)
    jf = open(rec_path, "a")

    def put(**r):
        jf.write(json.dumps(r) + "\n"); jf.flush()

    t_all = time.time()
    zh, zl, meta = load(os.path.join(a.raw, a.state))
    rh, rl, mr = load(os.path.join(a.raw, a.ref))
    if (mr["M"], list(mr["nodes"]), mr["kc"], mr["ks"]) != (meta["M"], list(meta["nodes"]), meta["kc"], meta["ks"]):
        raise SystemExit("the reference differs from the base in M, nodes or series orders")
    D0, Dt = float(meta["D"]), a.target
    if abs(float(mr["D"]) - Dt) > 1e-12:
        raise SystemExit(f"the reference is at D {mr['D']}, not {Dt}")
    De = Dt + (1e-9 if a.mutate == "dshift" else 0.0)
    h = De - D0
    Dref = float(rh[-1]) + float(rl[-1])
    put(stage="start", state=a.state, ref=a.ref, D0=D0, target=Dt, D_step=De, h=h, M=meta["M"], N=a.N, mutate=a.mutate,
        threads=a.threads, tol_pred=a.tol_pred, tol_delta=a.tol_delta, Delta_ref=float(rh[-1]), Delta_ref_lo=float(rl[-1]))
    print(f"step D {D0} -> {De} (h {h:+.6g}), M {meta['M']}, N {a.N}, mutate {a.mutate}", flush=True)

    # 1. the fixed plan and the base's LU
    t0 = time.time()
    s = solver_for(meta, D0, a.threads)
    segs = s.plan(zh, zl, a.plan_tol, a.hmax)
    s.set_segs(segs)
    steps = sum(len(g) - 1 for g in segs)
    F0 = fmax(s, zh, zl)
    t1 = time.time()
    _, _, _, _, pmin, pmax = s.newton_mixed(zh, zl)
    put(stage="base", steps=steps, F=F0, pivots=[pmin, pmax], t_plan=round(t1 - t0, 1), t_lu=round(time.time() - t1, 1))
    print(f"  base: plan {steps} steps, |F| {F0:.2e}, LU pivots {pmin:.2e}..{pmax:.2e}, {time.time() - t0:.0f} s", flush=True)

    # 2. the series on that plan
    t0 = time.time()
    orders = []
    Zh, Zl = coefficients(s, zh, zl, a.N, orders, mutate=JET_MUT_KAPPA if a.mutate == "kappa" else 0, step_tol=a.step_tol)
    norms = [float(np.abs(zh).max())] + [o["norm"] for o in orders]
    ratios = [norms[k - 1] / norms[k] for k in range(2, len(norms)) if norms[k]]
    put(stage="series", orders=orders, ratios=ratios, delta_k=[float(z[-1]) for z in Zh], seconds=round(time.time() - t0, 1))
    print(f"  series: ratios |z_k-1|/|z_k| {' '.join(f'{r:.3g}' for r in ratios)}, all converged "
          f"{all(o['converged'] for o in orders)}, {time.time() - t0:.0f} s", flush=True)

    # 3. the prediction on the fixed plan, against the reference
    if a.mutate == "drop2":
        Zh[2], Zl[2] = np.zeros_like(Zh[2]), np.zeros_like(Zl[2])
    rows = []
    for n in range(a.N + 1):
        ph, pl = evaluate(Zh[:n + 1], Zl[:n + 1], h)
        rows.append((n, float(np.abs((ph - rh) + (pl - rl)).max()), float((ph[-1] - rh[-1]) + (pl[-1] - rl[-1]))))
    st = solver_for(meta, De, a.threads, segs)
    with np.errstate(all="ignore"):
        Fp, Fz = fmax(st, ph, pl), fmax(st, zh, zl)
    dpred = rows[-1][2]
    okA = bool(abs(dpred) <= a.tol_pred)
    put(stage="prediction", partial_sums=rows, F_pred=Fp if np.isfinite(Fp) else str(Fp), F_const=Fz if np.isfinite(Fz) else str(Fz),
        dDelta_pred=dpred, A=okA)
    print("  partial sums vs the reference (n, max |P - ref|, Delta - ref): "
          + "; ".join(f"{n} {e:.2e} {d:+.2e}" for n, e, d in rows), flush=True)
    print(f"  prediction on the fixed plan: |F| {Fp:.2e} (the base state's {Fz:.2e}); (A) |Delta(P) - Delta(ref)| "
          f"{abs(dpred):.2e} <= {a.tol_pred:.0e}: {'PASS' if okA else 'FAIL'}", flush=True)
    if not okA and a.stop_on_fail:
        put(stage="end", A=okA, B=None, passed=False, seconds=round(time.time() - t_all, 1))
        print("FAIL (A); the corrector is not run (--stop_on_fail 1)", flush=True)
        return 1

    # 4. the corrector on the fixed plan
    t0 = time.time()
    nh, nl, hist = newton_solve(st, ph, pl, tol=1e-28, it_max=a.newton_its, verbose=True, jac="f64")
    Fn = fmax(st, nh, nl)
    conv = bool(np.isfinite(Fn) and Fn < F_CONV)
    dcorr = float((nh[-1] - rh[-1]) + (nl[-1] - rl[-1]))
    okB = bool(conv and abs(dcorr) <= a.tol_delta)
    lams = [x.get("lam") for x in hist if "lam" in x]
    out = os.path.join(a.out, f"anm_D{Dt:.6f}_M{meta['M']}.npz")
    if conv and a.mutate == "none":
        save(out, nh, nl, dict(D=Dt, M=meta["M"], nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], segs=segs,
                               source=f"anm_lowd_pilot from {a.state} (D {D0}), N {a.N}"))
    put(stage="corrector", F=Fn, calls=len(hist), lams=lams, converged=conv, Delta=float(nh[-1]), Delta_lo=float(nl[-1]),
        dDelta=dcorr, B=okB, state=os.path.basename(out) if conv and a.mutate == "none" else None,
        seconds=round(time.time() - t0, 1))
    print(f"  corrector: |F| {Fn:.2e} after {len(hist)} calls (line-search factors {lams}), Delta {nh[-1]:.16f} "
          f"({nl[-1]:+.2e}); (B) |Delta - Delta(ref)| {abs(dcorr):.2e} <= {a.tol_delta:.0e}, converged {conv}: "
          f"{'PASS' if okB else 'FAIL'}, {time.time() - t0:.0f} s", flush=True)
    passed = okA and okB
    put(stage="end", A=okA, B=okB, passed=passed, Delta_ref=Dref, seconds=round(time.time() - t_all, 1))
    print(("PASS" if passed else "FAIL") + f" ({time.time() - t_all:.0f} s)", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
