#!/usr/bin/env python3
"""Solves and timings on the mapped tau grid (hp/tau_map.py) through the leg's point solver (notes/LOW_D_SPEEDUP.md,
"Step 5, the test").

solve: a converged uniform state REF sampled on the mapped grid (centre s0, parameter L) at M (tau_map.map_state) is
solved by lowD_dd.solve_point, unchanged, with its solver built by tau_map.make_solver_mapped, in the leg's two passes
(lowD_highM.py solve_up): pass 1 on the start's own plan (--tol_step, --hmax) to |F| <= --tol1, pass 2 on the pass-1
state's own plan to --tol, accepted below --accept. The f64-Jacobian Newton and the leg's stop rules (--stop_lam,
--floor_ratio). One record per solve in OUTDIR/runs.jsonl: the map, both plans' steps, every Newton call's seconds and
Jacobian seconds, |F|, Delta and its difference from REF's Delta, and the pass test of the procedure (converged below
--accept and |Delta - Delta(REF)| <= --tol_delta, by default |Delta(REF) - Delta(--ref_w)|). solve_point's own
records go to OUTDIR/points.jsonl, the states to OUTDIR/states (their meta carries tau_map = [s0, L, mutate]).
--mutate gp: the control of tau_map.mapped_inputs (d/dtheta without the 1/g' rows), which must miss.

time: one Solver.newton_mixed call (residual, Jacobian build, LU, solve) at each stored state on its own stored plan,
uniform or mapped (by its meta's tau_map), one after another at --threads; records in OUTDIR/timing.jsonl.

    python3 tau_map_solve.py solve OUTDIR --ref REF.npz [--ref_w REF_W.npz] --M 128 --L 0.35 --s0 1.9 [--mutate gp]
        [--threads 2] [--tag tm]
    python3 tau_map_solve.py time OUTDIR STATE.npz [STATE.npz ...] [--threads 2]

OUTDIR must be in a task folder of the raw-data repository (raw_out.py; --allow_unraw for tests).
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DSS_SOLVER_DIR", os.path.normpath(os.path.join(HERE, "..", "solver")))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import lowD_dd  # noqa: E402
from hpkernel.driver import load, make_solver, save  # noqa: E402
from raw_out import FLAG, HELP, require_raw  # noqa: E402
from tau_map import make_solver_mapped, map_state  # noqa: E402

CALLS = []          # every Newton call's record (newton_solve's history), with the pass it belongs to
PHASE = [None]


def _capture(fn):
    def wrapped(*args, **kw):
        zh, zl, hist = fn(*args, **kw)
        CALLS.extend(dict(r, phase=PHASE[0]) for r in hist)
        return zh, zl, hist
    return wrapped


def commit():
    r = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def fmax(s, zh, zl):
    fh, fl = s.residual(zh, zl)
    return float(np.abs(np.asarray(fh) + np.asarray(fl)).max())


def steps(segs):
    return sum(len(g) - 1 for g in segs)


def ns(a, out, tag, tol, accept):
    """solve_point's options, as lowD_highM.ns builds them"""
    return argparse.Namespace(out=out, tag=tag, resume=True, kt=a.kt, kcs=a.kc, ks=a.ks, tol_step=a.tol_step, hmax=a.hmax,
                              solver="newton", it_max=a.it_max, tol=tol, accept=accept, lm=0, jac="f64",
                              threads=a.threads, svd=0, dn_its=2, w=24, stop_lam=a.stop_lam, floor_ratio=a.floor_ratio,
                              chord_from=0.0)


def cmd_solve(a):
    out = os.path.join(a.outdir, "runs.jsonl")
    points = os.path.join(a.outdir, "points.jsonl")
    sdir = os.path.join(a.outdir, "states")
    require_raw(out, points, os.path.join(sdir, "x"), allow=a.allow_unraw)
    os.makedirs(sdir, exist_ok=True)
    log = lambda m: print(time.strftime("%H:%M:%S ") + m, flush=True)  # noqa: E731
    zh, zl, m = load(a.ref)
    D, Mr, nodes = m["D"], m["M"], list(m["nodes"])
    a.kt, a.kc, a.ks = m["kt"], m["kc"], m["ks"]
    Dref = (float(zh[-1]), float(zl[-1]))
    if a.ref_w:
        wh, wl, mw = load(a.ref_w)
        assert mw["D"] == D and list(mw["nodes"]) == nodes, "the two references must share D and nodes"
        err_est = abs((float(wh[-1]) - Dref[0]) + (float(wl[-1]) - Dref[1]))
    else:
        err_est = None
    tol_delta = a.tol_delta if a.tol_delta else err_est
    tag = a.tag or f"tm_L{a.L:g}_s{a.s0:.4f}_M{a.M}" + (f"_mut{a.mutate}" if a.mutate else "")
    # every solver of solve_point on this map (the leg's code path, unchanged): make_solver replaced, the Newton
    # history captured for the per-call seconds
    lowD_dd.make_solver = lambda D_, M_, **kw: make_solver_mapped(D_, M_, a.s0, a.L, mutate=a.mutate, **kw)
    lowD_dd.newton_solve = _capture(lowD_dd.newton_solve)
    mk = lambda: lowD_dd.make_solver(D, a.M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads)  # noqa: E731
    t0 = time.time()
    z0, z0l = map_state(zh, zl, Mr, a.M, len(nodes), None, (a.s0, a.L))
    s = mk()
    tp = time.time()
    seg1 = s.plan(z0, z0l, a.tol_step, a.hmax)
    s.set_segs(seg1)
    F0 = fmax(s, z0, z0l)
    log(f"D {D} M {a.M} map s0 {a.s0} L {a.L} mutate {a.mutate}: start |F| {F0:.2e} on its own plan, {steps(seg1)} steps "
        f"(plan {time.time() - tp:.0f}s)")
    rec = dict(D=D, M=a.M, s0=a.s0, L=a.L, mutate=a.mutate, ref=os.path.basename(a.ref), M_ref=Mr,
               Delta_ref=Dref, err_est_ref=err_est, tol_delta=tol_delta, threads=a.threads, tag=tag, commit=commit(),
               F_start=F0, steps_p1=steps(seg1))
    ok = np.isfinite(F0)
    if ok:
        PHASE[0] = 1
        ok, h1, l1 = lowD_dd.solve_point(D, a.M, z0, z0l, f"map of {os.path.basename(a.ref)}",
                                         ns(a, points, tag + "_p1", a.tol1, a.tol1), nodes, sdir, log, segs=seg1)
    rec["t_pass1"] = round(time.time() - t0, 1)
    if ok:
        PHASE[0] = 2
        s = mk()
        seg2 = s.plan(np.asarray(h1), np.asarray(l1), a.tol_step, a.hmax)
        rec["steps_p2"] = steps(seg2)
        ok, h2, l2 = lowD_dd.solve_point(D, a.M, np.asarray(h1), np.asarray(l1), "pass 1",
                                         ns(a, points, tag, a.tol, a.accept), nodes, sdir, log, segs=seg2)
        s.set_segs(seg2)
        F = fmax(s, h2, l2)
        diff = (float(h2[-1]) - Dref[0]) + (float(l2[-1]) - Dref[1])
        rec.update(Delta=(float(h2[-1]), float(l2[-1])), F=F, diff=diff)
        done = os.path.join(sdir, f"{tag}_D{D:.4f}_M{a.M}.npz")
        if os.path.exists(done):
            zh2, zl2, m2 = load(done)
            save(done, zh2, zl2, dict(m2, tau_map=[a.s0, a.L, a.mutate]))
            rec["state"] = os.path.relpath(done, a.outdir)
    p1 = os.path.join(sdir, f"{tag}_p1_D{D:.4f}_M{a.M}.npz")
    if os.path.exists(p1):
        zp, zpl, mp_ = load(p1)
        if "tau_map" not in mp_:
            save(p1, zp, zpl, dict(mp_, tau_map=[a.s0, a.L, a.mutate]))
    calls = [dict(phase=c["phase"], F=c["F"], lam=c.get("lam"), t=c.get("t"), t_jac=c.get("t_jac"), t_lu=c.get("t_lu"),
                  t_res=c.get("t_res")) for c in CALLS if "piv" in c or "failed" in c]
    rec.update(converged=bool(ok) and rec.get("F", np.inf) < a.accept, calls=calls,
               calls_p1=sum(c["phase"] == 1 for c in calls), calls_p2=sum(c["phase"] == 2 for c in calls),
               seconds=round(time.time() - t0, 1), peak_rss_mb=lowD_dd.peak_rss_mb(), load=lowD_dd.load_avg())
    if rec["converged"] and tol_delta is not None:
        rec["pass"] = abs(rec["diff"]) <= tol_delta
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"D {D} M {a.M} L {a.L} s0 {a.s0} mutate {a.mutate}: converged {rec['converged']}, |F| {rec.get('F', float('nan')):.1e}, "
        f"Delta {rec.get('Delta', (float('nan'),))[0]:.16f}, diff {rec.get('diff', float('nan')):+.2e} (tolerance "
        f"{tol_delta}), pass {rec.get('pass')}; calls {rec['calls_p1']} + {rec['calls_p2']}, steps {rec['steps_p1']} / "
        f"{rec.get('steps_p2')}, {rec['seconds']:.0f}s")
    return 0


def cmd_time(a):
    out = os.path.join(a.outdir, "timing.jsonl")
    require_raw(out, allow=a.allow_unraw)
    for path in a.states:
        zh, zl, m = load(path)
        tm = m.get("tau_map")
        kw = dict(nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=a.threads)
        s = make_solver_mapped(m["D"], m["M"], tm[0], tm[1], mutate=tm[2], **kw) if tm else make_solver(m["D"], m["M"], **kw)
        s.set_segs(m["segs"])
        l0 = lowD_dd.load_avg()
        t = time.time()
        F = fmax(s, zh, zl)
        t_res = time.time() - t
        t = time.time()
        s.newton_mixed(zh, zl)
        t_call = time.time() - t
        tr, tj, tf, ts = s.step_timing()
        rec = dict(state=os.path.basename(path), D=m["D"], M=m["M"], tau_map=tm, steps=steps(m["segs"]), threads=a.threads,
                   F=F, t_residual=round(t_res, 2), t_call=round(t_call, 1), t_jac=round(tj, 1), t_lu=round(tf, 2),
                   t_solve=round(ts, 2), s_per_step=round(tj / steps(m["segs"]), 4), load_before=l0,
                   load_after=lowD_dd.load_avg(), peak_rss_mb=lowD_dd.peak_rss_mb(), commit=commit())
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("solve")
    p.add_argument("outdir")
    p.add_argument("--ref", required=True)
    p.add_argument("--ref_w", default="")
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--L", type=float, required=True)
    p.add_argument("--s0", type=float, required=True)
    p.add_argument("--mutate", default=None, choices=[None, "gp"])
    p.add_argument("--tol_delta", type=float, default=0.0, help="the pass test's tolerance (default |Delta(REF) - Delta(REF_W)|)")
    p.add_argument("--tag", default="")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--tol_step", type=float, default=1e-30)
    p.add_argument("--hmax", type=float, default=0.05)
    p.add_argument("--tol1", type=float, default=1e-14)
    p.add_argument("--tol", type=float, default=1e-28)
    p.add_argument("--accept", type=float, default=1e-27)
    p.add_argument("--it_max", type=int, default=14)
    p.add_argument("--stop_lam", type=float, default=0.125)
    p.add_argument("--floor_ratio", type=float, default=0.5)
    p.add_argument(FLAG, action="store_true", help=HELP)
    q = sub.add_parser("time")
    q.add_argument("outdir")
    q.add_argument("states", nargs="+")
    q.add_argument("--threads", type=int, default=2)
    q.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    sys.exit(cmd_solve(a) if a.cmd == "solve" else cmd_time(a))


if __name__ == "__main__":
    main()
