#!/usr/bin/env python3
"""Where the Taylor planner freezes on a low-D leg's predicted states (notes/LOW_D.md 16). From a converged base, the
branch's Taylor coefficients z_1..z_N (anm_lowd.py: one f64 Jacobian, GMRES-refined; stored once and reused), then for
each target D and each order n = 0..N the predicted state sum_{k<=n} z_k h^k (n = 0: the base state itself) is planned
on the target's solver (Solver.plan with HP_PLAN_TRACE) in a child process with a time cap. T115 found the plan of the
order-3 prediction at 3.0425 from 3.04375 frozen at x = 0.013725308400, its step h at 0 (T115-win2). The table
separates the causes: a freeze that moves or vanishes as h or n shrinks says the prediction leaves the branch; one at
the same x for every h and n, the base state's included, says the planner.

    python3 lowD_plan_probe.py RAWDIR OUTDIR --state STATE.npz --targets 3.0425,3.043,3.0434 [--N 3] [--threads 6]
        [--cap 900] [--tol_step 1e-32] [--hmax 0.05] [--trace 500]

RAWDIR is a clone of the raw-data repository (STATE is relative to it); OUTDIR lies in a task folder of it
(raw_out.py). Writes OUTDIR/probe.jsonl (the base, then one record per case: target, order, done or capped, seconds,
the last segment, x and h of its trace), OUTDIR/coeffs.npz (z_0..z_N) and each case's trace as
OUTDIR/plan_D<target>_n<order>.log. Control: --cap 0.001 must report every case capped (exit 1 when any case is
capped, 0 when all are done).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from anm_lowd import coefficients, evaluate, fmax, solver_for  # noqa: E402
from hpkernel.driver import load  # noqa: E402
from raw_out import FLAG, HELP, require_raw  # noqa: E402

TRACE = re.compile(r"plan_path (\S+)->(\S+): (?:step (\d+), x (\S+), h (\S+)|done, (\d+) steps), (\S+) s")


def child(a, Dt, n):
    """plan the order-n prediction at Dt (stdout: PLAN_DONE <seconds>)"""
    zh, zl, meta = load(os.path.join(a.raw, a.state))
    c = np.load(os.path.join(a.out, "coeffs.npz"))
    Zh, Zl = list(c["Zh"][:n + 1]), list(c["Zl"][:n + 1])
    ph, pl = evaluate(Zh, Zl, Dt - float(c["D0"]))
    st = solver_for(meta, Dt, a.threads)
    t0 = time.time()
    st.plan(ph, pl, a.tol_step, a.hmax)
    print(f"PLAN_DONE {time.time() - t0:.1f}", flush=True)


def last_trace(path):
    """the last plan_path line of a trace: (segment, x, h, done)"""
    seg = x = h = None
    done = False
    with open(path, errors="replace") as f:
        for line in f:
            m = TRACE.search(line)
            if m:
                seg = f"{m[1]}->{m[2]}"
                done = m[6] is not None
                x, h = (None, None) if done else (float(m[4]), float(m[5]))
    return seg, x, h, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--state", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cap", type=float, default=900.0)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--step_tol", type=float, default=1e-20)
    ap.add_argument("--trace", type=int, default=500)
    ap.add_argument("--child", default="")
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    if a.child:
        Dt, n = a.child.split(":")
        return child(a, float(Dt), int(n))
    require_raw(os.path.join(a.out, "probe.jsonl"))
    os.makedirs(a.out, exist_ok=True)
    jf = open(os.path.join(a.out, "probe.jsonl"), "a")

    def put(**r):
        jf.write(json.dumps(r) + "\n"); jf.flush()

    zh, zl, meta = load(os.path.join(a.raw, a.state))
    D0 = float(meta["D"])
    cpath = os.path.join(a.out, "coeffs.npz")
    have = None
    if os.path.exists(cpath):
        c = np.load(cpath)
        if str(c["state"]) == a.state and len(c["Zh"]) > a.N:
            have = c
    if have is None:
        # the base: its plan, residual, the f64 Jacobian's LU (newton_mixed) and the series, as lowD_taylor_leg.py
        t0 = time.time()
        s = solver_for(meta, D0, a.threads, meta.get("segs"))
        if not meta.get("segs"):
            s.set_segs(s.plan(zh, zl, a.tol_step, a.hmax))
        F0 = fmax(s, zh, zl)
        print(f"base D {D0}: plan and residual |F| {F0:.1e}, {time.time() - t0:.0f} s", flush=True)
        t1 = time.time()
        _, _, _, _, pmin, pmax = s.newton_mixed(zh, zl)
        # the f64 Jacobian and its LU print nothing while they run (about 33 min at M 320 on win2 at 8 threads)
        print(f"base D {D0}: Jacobian and LU, pivots {pmin:.1e}..{pmax:.1e}, {time.time() - t1:.0f} s", flush=True)
        orders = []
        Zh, Zl = coefficients(s, zh, zl, a.N, orders, step_tol=a.step_tol)
        np.savez(cpath, Zh=np.array(Zh), Zl=np.array(Zl), D0=D0, state=a.state)
        norms = [o["norm"] for o in orders]
        radii = [norms[k - 1] / norms[k] for k in range(1, len(norms)) if norms[k]]
        put(stage="base", D=D0, state=a.state, M=meta["M"], F=F0, pivots=[pmin, pmax], orders=orders, radii=radii,
            seconds=round(time.time() - t0, 1))
        print(f"base D {D0} ({a.state}): |F| {F0:.1e}, radii {', '.join(f'{r:.3g}' for r in radii)}, "
              f"{time.time() - t0:.0f} s", flush=True)
    else:
        print(f"base D {D0}: coefficients reused from {cpath}", flush=True)
    capped = 0
    env = dict(os.environ, HP_PLAN_TRACE=str(a.trace))
    for Dt in [float(x) for x in a.targets.split(",") if x]:
        for n in range(a.N + 1):
            log = os.path.join(a.out, f"plan_D{Dt:.6f}_n{n}.log")
            cmd = [sys.executable, os.path.abspath(__file__), a.raw, a.out, "--state", a.state, "--targets", repr(Dt),
                   "--child", f"{Dt!r}:{n}",
                   "--threads", str(a.threads), "--tol_step", repr(a.tol_step), "--hmax", repr(a.hmax), FLAG]
            t0 = time.time()
            with open(log, "w") as lf:
                try:
                    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=lf, env=env, timeout=a.cap, text=True)
                    done = "PLAN_DONE" in (r.stdout or "")
                    status = "done" if done else f"exit {r.returncode}"
                except subprocess.TimeoutExpired:
                    done, status = False, "capped"
            seg, x, h, _ = last_trace(log)
            capped += not done
            put(stage="case", D=Dt, h=Dt - D0, order=n, status=status, seconds=round(time.time() - t0, 1), segment=seg,
                x=x, h_step=h)
            print(f"  D {Dt} (h {Dt - D0:+.6f}) order {n}: {status}, {time.time() - t0:.0f} s"
                  + (f", last {seg} at x {x:.12f}, h {h:.3e}" if x is not None else ""), flush=True)
    put(stage="end", capped=capped)
    print(f"done: {capped} case(s) not planned", flush=True)
    return 1 if capped else 0


if __name__ == "__main__":
    sys.exit(main())
