#!/usr/bin/env python3
"""The low-D leg with the exact Taylor predictor (T107, notes/LOW_D.md 15): from a converged state, each target in
turn is predicted by the branch's Taylor series at the current base (hp/anm_lowd.py's coefficients, N orders from one
f64 Jacobian, GMRES-refined) and solved by Newton with the f64 Jacobian, as anm_lowd.py does for one target. A
converged point (|F| below 1e-26) becomes the next base. A failed one is retried at half the step from the same base
(the coefficients are reused), at most --halvings times; then the leg stops.

    python3 lowD_taylor_leg.py RAWDIR OUTDIR --state STATE.npz --targets 3.0425,3.04125
        [--N 4] [--threads 8] [--newton_its 6] [--halvings 2] [--step_tol 1e-20] [--mutate none|halve|const]

RAWDIR is a clone of the raw-data repository (STATE is relative to it); OUTDIR lies in a task folder of it
(raw_out.py). Writes OUTDIR/leg.jsonl (one record per base, attempt and point, flushed as it goes) and each converged
state as OUTDIR/states/taylor_D<D>_M<M>.npz with its plan. Residuals are max |F|, as anm_lowd.py reports them.

Per attempt it records the prediction's residual F_pred and, on the same plan, that of the base state itself at the
target (F_const, the zeroth-order predictor). The check: F_pred <= 1e-2 F_const, else the series does not predict and
the leg stops. Mutations (each must change the outcome, the controls of LOW_D.md 15): halve, the first attempt at
every target given one Newton call only, so it fails and the halving path runs; const, the prediction replaced by
the base state, so the check fails at the first target.
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
from raw_out import FLAG, HELP, require_raw  # noqa: E402,F401

F_CONV = 1e-26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--state", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--newton_its", type=int, default=6)
    ap.add_argument("--halvings", type=int, default=2)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--step_tol", type=float, default=1e-20)
    ap.add_argument("--mutate", choices=("none", "halve", "const"), default="none")
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    require_raw(os.path.join(a.out, "leg.jsonl"))
    os.makedirs(os.path.join(a.out, "states"), exist_ok=True)
    jf = open(os.path.join(a.out, "leg.jsonl"), "a")

    def put(**r):
        jf.write(json.dumps(r) + "\n"); jf.flush()

    targets = [float(x) for x in a.targets.split(",") if x]
    zh, zl, meta = load(os.path.join(a.raw, a.state))
    source = a.state
    put(stage="start", state=a.state, targets=targets, N=a.N, threads=a.threads, newton_its=a.newton_its, halvings=a.halvings,
        mutate=a.mutate)
    t_leg = time.time()
    ti = 0
    while ti < len(targets):
        # the base: its plan, residual, the f64 Jacobian's LU and the series
        D0 = float(meta["D"])
        t0 = time.time()
        s = solver_for(meta, D0, a.threads, meta.get("segs"))
        if not meta.get("segs"):
            s.set_segs(s.plan(zh, zl, a.tol_step, a.hmax))
        F0 = fmax(s, zh, zl)
        _, _, _, _, pmin, pmax = s.newton_mixed(zh, zl)
        orders = []
        Zh, Zl = coefficients(s, zh, zl, a.N, orders, step_tol=a.step_tol)
        norms = [o["norm"] for o in orders]
        radii = [norms[k - 1] / norms[k] for k in range(1, len(norms)) if norms[k]]
        put(stage="base", D=D0, source=source, F=F0, pivots=[pmin, pmax], orders=orders, radii=radii,
            seconds=round(time.time() - t0, 1))
        print(f"base D {D0} ({source}): |F| {F0:.1e}, pivots {pmin:.1e}..{pmax:.1e}, radii "
              f"{', '.join(f'{r:.3g}' for r in radii)}, {time.time() - t0:.0f} s", flush=True)
        if not all(o["converged"] for o in orders):
            put(stage="stop", reason="an order's GMRES did not converge"); print("stop: an order did not converge", flush=True)
            sys.exit(1)
        target = targets[ti]
        h = target - D0
        done = False
        for attempt in range(a.halvings + 1):
            Dt = D0 + h
            t1 = time.time()
            ph, pl = (np.array(zh), np.array(zl)) if a.mutate == "const" else evaluate(Zh, Zl, h)
            st = solver_for(meta, Dt, a.threads)
            st.set_segs(st.plan(ph, pl, a.tol_step, a.hmax))
            Fp, Fz = fmax(st, ph, pl), fmax(st, zh, zl)
            its = 1 if (a.mutate == "halve" and attempt == 0) else a.newton_its
            print(f"  D {Dt} (h {h:+.6g}, attempt {attempt}): |F| of the prediction {Fp:.2e}, of the base state {Fz:.2e} "
                  f"on its plan ({sum(len(g) - 1 for g in st.get_segs())} steps)", flush=True)
            rec = dict(stage="attempt", D=Dt, h=h, attempt=attempt, F_pred=Fp, F_const=Fz,
                       steps=sum(len(g) - 1 for g in st.get_segs()), newton_its=its)
            if not (np.isfinite(Fp) and Fp <= 1e-2 * Fz):
                rec.update(seconds=round(time.time() - t1, 1)); put(**rec)
                put(stage="stop", reason="the series does not predict (F_pred > 1e-2 F_const)")
                print("stop: the series does not predict", flush=True)
                sys.exit(1)
            nh, nl, hist = newton_solve(st, ph, pl, tol=1e-28, it_max=its, verbose=True, jac="f64")
            Fn = fmax(st, nh, nl)
            lams = [x.get("lam") for x in hist if "lam" in x]
            ok = bool(np.isfinite(Fn) and Fn < F_CONV)
            rec.update(F=Fn, calls=len(hist), lams=lams, Delta=float(nh[-1]), Delta_lo=float(nl[-1]), converged=ok,
                       seconds=round(time.time() - t1, 1))
            put(**rec)
            print(f"  D {Dt}: Newton |F| {Fn:.2e} after {len(hist)} calls, line-search factors {lams}, Delta {nh[-1]:.15f}, "
                  f"{'converged' if ok else 'not converged'}", flush=True)
            if ok:
                meta = dict(D=Dt, M=meta["M"], nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=meta["ks"], segs=st.get_segs(),
                            source=f"lowD_taylor_leg from D {D0}, N {a.N}")
                path = os.path.join(a.out, "states", f"taylor_D{Dt:.6f}_M{meta['M']}.npz")
                save(path, nh, nl, meta)
                put(stage="point", D=Dt, F=Fn, Delta=float(nh[-1]), Delta_lo=float(nl[-1]), state=os.path.relpath(path, a.raw))
                zh, zl, source = np.asarray(nh), np.asarray(nl), os.path.relpath(path, a.raw)
                done = True
                break
            h *= 0.5
        if not done:
            put(stage="stop", reason=f"no convergence at {target} after {a.halvings} halvings")
            print(f"stop: no convergence toward {target} after {a.halvings} halvings", flush=True)
            sys.exit(1)
        if float(meta["D"]) == target:
            ti += 1
    put(stage="done", seconds=round(time.time() - t_leg, 1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
