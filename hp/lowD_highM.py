#!/usr/bin/env python3
"""Continuation in D toward D = 3 at two tau resolutions per point, moving up in M as needed (notes/HIGH_M.md).

    cd hp && python3 -u lowD_highM.py --start_w STATE_W.npz --start_n STATE_N.npz [--Ms 128,160,192,256,320,384]
        [--dD 0.05] [--D_end 3.0] [--min_step 0.005] [--diff_max 1e-8] [--hours 10] [--point_hours 3] [--auto_nodes]
        [--tag lowDhm] [--out data/lowD_highM/pilot.jsonl] [--points data/lowD_highM/pilot_points.jsonl]

The two start states are converged at the same D and nodes, at the working resolution M_w and the next one M_n
(consecutive in --Ms). At each new D (a grid of --dD below the start, D_end excluded):
  1. M_w: predictor = Lagrange (cubic) through the last <= 4 converged states at M_w, Newton on the predictor's Taylor
     plan (lowD_dd.solve_point: the f64-Jacobian Newton, Solver.newton_mixed, in block storage above M = 128);
  2. M_n: plan from the predictor through the last <= 4 states at M_n; start from the better (smaller |F|) of that
     predictor (with 2 or more states) and the M_w solution just found, resampled to M_n;
  3. the value is Delta(M_n), its error estimate |Delta(M_n) - Delta(M_w)| (an estimator: the change from the next
     resolution, not a bound);
  4. if that difference exceeds --diff_max, or a log-linear fit of the last 3 differences in D projects it above
     --diff_max two points ahead, the resolution moves up (M_w <- M_n, M_n <- the next in --Ms): the new M_n is solved
     at this D from the new M_w's solution resampled, in two passes (pass 1 on the resampled state's own plan to
     |F| <= 1e-14, state <tag>_p1_*; pass 2 on the plan of the pass-1 state, 2-4 times fewer steps), and the point is
     recorded again with the new pair;
  5. --auto_nodes: if the M_n solution's centre or horizon series tail (lowD_dd.tails) exceeds --tail_max (1e-30),
     the nodes change as in lowD_dd.py (first node halved; a node 0.1 further out while the last is below 0.85), both
     resolutions are re-noded (driver.renode) and re-solved at this D under a new tag, and the predictors restart.
The plans come from converged states or predictors through them: a state resampled to a larger M needs 3.5-4 times
more Taylor steps (its new top harmonics are aliasing products that the tau-derivative amplifies), and a coarser plan
blows up on it. A start whose residual on the given plan is not finite or above 1 gets its own plan instead.
A failed solve inserts the midpoint between the last converged D and the target (halving the step).
Stop rules (each recorded with its reason): the largest M in --Ms cannot keep the difference below --diff_max; a
point fails with the step below --min_step; the projected time of the next point (the last point's solves) exceeds
--point_hours; --hours of wall time in all, counted across restarts.
Restarts: every Newton step is checkpointed and every converged solve saved (lowD_dd.solve_point); the progress file
<out>.progress.json holds the continuation state, including a pending move up or re-noding; rerun the same command
with --resume to continue from the last converged point (a converged (D, M) is not solved again, a pair is not
recorded twice).
Records: one per solve in --points (lowD_dd.solve_point's record: Delta hi/lo, |F|, iterations, seconds per
iteration, steps, tails, sensitivities, harmonics, peak RSS), one per point and resolution pair in --out (D, M_w,
M_n, both Deltas, the difference, |F|, iterations, seconds, tails, the decision). States:
hp/data/lowD_states/<tag>_D<D>_M<M>.npz.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DSS_SOLVER_DIR", os.path.normpath(os.path.join(HERE, "..", "solver")))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import lowD_dd  # noqa: E402
from hpkernel.driver import load, make_solver, predict, renode, resample_z  # noqa: E402

STATES = os.path.join(HERE, "data", "lowD_states")


def find_record(path, D, M, nodes):
    """the last converged record of (D, M) on these nodes in a lowD_dd.solve_point jsonl (also after a resumed run,
    where solve_point skips a converged (D, M) without writing a record)"""
    out = None
    if os.path.exists(path):
        for ln in open(path):
            if ln.strip():
                r = json.loads(ln)
                if (r.get("converged") and r["F"] <= 1e-24 and abs(r["D"] - D) < 1e-9 and r["M"] == M
                        and list(r["nodes"]) == list(nodes)):
                    out = r
    return out


def projected_diff(pairs, D_next2):
    """log-linear extrapolation in D of |difference| from the last 3 points at the current resolution pair"""
    if len(pairs) < 3:
        return None
    Ds = np.array([p[0] for p in pairs[-3:]])
    ds = np.array([max(abs(p[1]), 1e-300) for p in pairs[-3:]])
    a, b = np.polyfit(Ds, np.log(ds), 1)
    return float(np.exp(a * D_next2 + b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_w", required=True)
    ap.add_argument("--start_n", required=True)
    ap.add_argument("--Ms", default="128,160,192,256,320,384")
    ap.add_argument("--dD", type=float, default=0.05)
    ap.add_argument("--D_end", type=float, default=3.0)
    ap.add_argument("--min_step", type=float, default=0.005)
    ap.add_argument("--diff_max", type=float, default=1e-8)
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--point_hours", type=float, default=3.0)
    ap.add_argument("--auto_nodes", action="store_true")
    ap.add_argument("--tail_max", type=float, default=1e-30)
    ap.add_argument("--tag", default="lowDhm")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "lowD_highM", "pilot.jsonl"))
    ap.add_argument("--points", default=os.path.join(HERE, "data", "lowD_highM", "pilot_points.jsonl"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-28)
    ap.add_argument("--accept", type=float, default=1e-27)
    ap.add_argument("--it_max", type=int, default=14)
    ap.add_argument("--kt", type=int, default=36)
    ap.add_argument("--tol_step", type=float, default=1e-30)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    log = lambda m: print(time.strftime("%H:%M:%S ") + m, flush=True)  # noqa: E731
    Ms = [int(v) for v in a.Ms.split(",")]
    prog_path = a.out + ".progress.json"

    zw, zwl, mw = load(a.start_w)
    zn, znl, mn = load(a.start_n)
    if mw["D"] != mn["D"] or list(mw["nodes"]) != list(mn["nodes"]) or Ms.index(mn["M"]) != Ms.index(mw["M"]) + 1:
        raise SystemExit("the start states must be at the same D and nodes, at consecutive M of --Ms")
    nodes = list(mw["nodes"])
    kc, ks = mw["kc"], mw["ks"]
    D0 = mw["D"]
    st = dict(t_used=0.0, Mw=mw["M"], Mn=mn["M"], nodes=nodes, tag=a.tag, hist_w=[[D0, a.start_w]],
              hist_n=[[D0, a.start_n]], pairs=[], targets=None, done=None, post=None, last_point_s=None)
    if a.resume and os.path.exists(prog_path):
        st = json.load(open(prog_path))
        log(f"resuming: M {st['Mw']}/{st['Mn']}, last D {st['hist_n'][-1][0]}, {st['t_used'] / 3600:.2f} h used")
    if st.get("done"):
        log(f"already stopped: {st['done']}")
        return
    t_start = time.time() - st["t_used"]
    if st["targets"] is None:
        n = int(round((D0 - a.D_end) / a.dD))
        st["targets"] = [round(D0 - a.dD * (i + 1), 6) for i in range(n) if D0 - a.dD * (i + 1) > a.D_end + 1e-9]

    def ns(M, tag=None, tol=None):
        return argparse.Namespace(out=a.points, tag=tag or st["tag"], resume=True, kt=a.kt, kcs=kc, ks=ks, tol_step=a.tol_step,
                                  hmax=a.hmax, solver="newton", it_max=a.it_max, tol=tol or a.tol, accept=tol or a.accept,
                                  lm=0, jac="f64", threads=a.threads, svd=0, dn_its=2, w=24)

    def save_prog():
        st["t_used"] = time.time() - t_start
        json.dump(st, open(prog_path + ".tmp", "w"), indent=1)
        os.replace(prog_path + ".tmp", prog_path)

    def stop(reason):
        st["done"] = reason
        save_prog()
        with open(a.out, "a") as f:
            f.write(json.dumps(dict(stop=reason, D_last=st["hist_n"][-1][0], Mw=st["Mw"], Mn=st["Mn"],
                                    hours=round(st["t_used"] / 3600, 2))) + "\n")
        log(f"STOP: {reason}")

    def zof(path):
        h, l, _ = load(path)
        return np.asarray(h) + np.asarray(l)

    def state_path(D, M):
        return os.path.join(STATES, f"{st['tag']}_D{D:.4f}_M{M}.npz")

    def plan_of(z, M, D):
        """Taylor plan (--tol_step, --hmax) of a state z at its own M, with the equations at D"""
        s0 = make_solver(D, M, nodes=st["nodes"], kt=a.kt, kc=kc, ks=ks, threads=a.threads)
        zz = np.asarray(z, float)
        return s0.plan(zz, np.zeros_like(zz), a.tol_step, a.hmax)

    def solve(D, M, starts, segs, tag=None, tol=None):
        """solve at (D, M) on the plan segs from the start with the smallest |F| among starts [(label, z)];
        returns (ok, record, path). The plan comes from a converged state or a predictor through converged states: a
        state resampled to a larger M needs 3.5-4 times more steps (its new top harmonics are aliasing products that
        the tau-derivative amplifies), which costs time, not accuracy."""
        s = make_solver(D, M, nodes=st["nodes"], kt=a.kt, kc=kc, ks=ks, threads=a.threads)
        s.set_segs(segs)
        best = None
        for lab, z in starts:
            zz = np.asarray(z, float)
            F = float(np.abs(np.add(*s.residual(zz, np.zeros_like(zz)))).max())
            log(f"  D {D} M {M}: start '{lab}' |F| {F:.2e} ({sum(len(g) - 1 for g in segs)} steps)")
            if np.isfinite(F) and (best is None or F < best[0]):
                best = (F, lab, zz)
        if best is None or best[0] > 1.0:
            # the plan does not hold for these starts (top harmonics blown up): plan from the first start itself
            zz = np.asarray(starts[0][1], float)
            segs = s.plan(zz, np.zeros_like(zz), a.tol_step, a.hmax)
            s.set_segs(segs)
            F = float(np.abs(np.add(*s.residual(zz, np.zeros_like(zz)))).max())
            log(f"  D {D} M {M}: plan from start '{starts[0][0]}' instead: {sum(len(g) - 1 for g in segs)} steps, |F| {F:.2e}")
            if not np.isfinite(F):
                return False, None, None
            best = (F, starts[0][0], zz)
        t = time.time()
        ok, zh, zl = lowD_dd.solve_point(D, M, best[2], np.zeros_like(best[2]), f"pilot: {best[1]}", ns(M, tag, tol), st["nodes"],
                                         STATES, log, segs=segs)
        if tol is not None:
            return ok, None, os.path.join(STATES, f"{tag or st['tag']}_D{D:.4f}_M{M}.npz")
        rec = find_record(a.points, D, M, st["nodes"]) if ok else None
        if ok and rec is None:
            return False, None, None
        if rec is not None:
            rec["wall"] = round(time.time() - t, 1)
        return ok, rec, os.path.join(STATES, f"{tag or st['tag']}_D{D:.4f}_M{M}.npz")

    def solve_up(D, path_src, M2):
        """the solve at a larger M2 from a converged state, in two passes: pass 1 from the state resampled to M2 on that
        start's own plan (a coarser plan blows up on it) to |F| <= 1e-14, pass 2 on the plan of the pass-1 solution
        (2-4 times leaner) to --tol. The record and state of pass 2 are the ones used."""
        z0 = lifted(path_src, M2)
        t = time.time()
        ok1, _, p1 = solve(D, M2, [("lift", z0)], plan_of(z0, M2, D), tag=st["tag"] + "_p1", tol=1e-14)
        if not ok1:
            return False, None, None
        t1 = time.time() - t
        z1 = zof(p1)
        ok2, r2, p2 = solve(D, M2, [("pass 1", z1)], plan_of(z1, M2, D))
        if ok2:
            r2["seconds_pass1"] = round(t1, 1)
            r2["seconds"] = round(r2["seconds"] + t1, 1)
        return ok2, r2, p2

    def lifted(path, M_new):
        h, l, m = load(path)
        return resample_z(np.asarray(h) + np.asarray(l), m["M"], M_new, len(m["nodes"]))[0]

    def write_pair(D, rw, rn, decision, extra=None):
        Dw = rw["Delta"] + rw["Delta_lo"]
        diff = (rn["Delta"] - rw["Delta"]) + (rn["Delta_lo"] - rw["Delta_lo"])
        rec = dict(D=D, Mw=rw["M"], Mn=rn["M"], Delta=rn["Delta"], Delta_lo=rn["Delta_lo"], err_est=abs(diff), diff=diff,
                   Delta_w=rw["Delta"], Delta_w_lo=rw["Delta_lo"], F_w=rw["F"], F_n=rn["F"], its_w=rw["its"], its_n=rn["its"],
                   seconds_w=rw["seconds"], seconds_n=rn["seconds"], steps_w=rw.get("steps"), steps_n=rn.get("steps"),
                   tail_c=rn.get("tail_c"), tail_h=rn.get("tail_h"), tail_c_w=rw.get("tail_c"), tail_h_w=rw.get("tail_h"),
                   peak_rss_mb_n=rn.get("peak_rss_mb"), nodes=st["nodes"], tag=st["tag"], decision=decision,
                   hours=round((time.time() - t_start) / 3600, 3))
        if extra:
            rec.update(extra)
        seen = os.path.exists(a.out) and any(
            r.get("Mw") == rec["Mw"] and r.get("Mn") == rec["Mn"] and abs(r.get("D", -9) - D) < 1e-9 and r.get("nodes") == rec["nodes"]
            for r in (json.loads(ln) for ln in open(a.out) if ln.strip()))
        if not seen:            # a resumed run does not record a pair twice
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        log(f"D {D}: Delta(M {rn['M']}) = {rn['Delta']:.16f} ({rn['Delta_lo']:+.2e}), M {rw['M']} -> {rn['M']}: {diff:+.2e}; "
            f"{decision}")
        return Dw, diff

    def post_point(post):
        """after a converged pair at post["D"]: the resolution decision (and the move up), then the series tails (and the
        re-noding). Resumable: the stage and its inputs are in the progress file, and the solves resume from their
        checkpoints or converged states. Returns a stop reason or None."""
        Dt = post["D"]
        if post["stage"] == "resolution":
            Mw, Mn = st["Mw"], st["Mn"]
            diff = [x for d, x, w_, n_ in st["pairs"] if abs(d - Dt) < 1e-9 and (w_, n_) == (Mw, Mn)][-1]
            same = [(d, x) for d, x, w_, n_ in st["pairs"] if (w_, n_) == (Mw, Mn)]
            D2 = round(Dt - 2 * a.dD, 6)
            proj = projected_diff(same, D2)
            why = None
            if abs(diff) > a.diff_max:
                why = f"|M {Mw} -> {Mn} difference| {abs(diff):.1e} > {a.diff_max:g}"
            elif proj is not None and proj > a.diff_max:
                why = f"difference projected to {proj:.1e} at D {D2} (> {a.diff_max:g})"
            if why and Ms.index(Mn) + 1 >= len(Ms):
                if abs(diff) > a.diff_max:
                    return (f"the largest M built ({Mn}) cannot keep the difference below {a.diff_max:g} at D {Dt} "
                            f"(M {Mw} -> {Mn}: {diff:+.2e})")
                log(f"{why}, but M {Mn} is the largest built; continuing at M {Mw} / {Mn}")
            elif why:
                M2 = Ms[Ms.index(Mn) + 1]
                log(f"{why}: moving up to M {Mn} / {M2}")
                ok2, r2, p2 = solve_up(Dt, st["hist_n"][-1][1], M2)
                if not ok2:
                    return f"the move up to M {M2} failed at D {Dt}"
                rn = find_record(a.points, Dt, Mn, st["nodes"])
                _, d2 = write_pair(Dt, rn, r2, f"moved up ({why})")
                st["Mw"], st["Mn"] = Mn, M2
                st["hist_w"] = st["hist_n"]
                st["hist_n"] = [[Dt, p2]]
                st["pairs"].append([Dt, d2, Mn, M2])
                post["t_point"] = rn["seconds"] + r2["seconds"]
            post["stage"] = "nodes"
            save_prog()
        if post["stage"] == "nodes":
            if "new_nodes" not in post:
                rlast = find_record(a.points, Dt, st["Mn"], st["nodes"])
                tc, th = rlast.get("tail_c", 0.0), rlast.get("tail_h", 0.0)
                new = list(st["nodes"])
                if a.auto_nodes and tc > a.tail_max:
                    new = [round(new[0] / 2, 6)] + new
                if a.auto_nodes and th > a.tail_max and new[-1] < 0.85:
                    new = new + [round(new[-1] + 0.1, 6)]
                if tc > a.tail_max or th > a.tail_max:
                    log(f"D {Dt}: tails {tc:.0e} (centre) {th:.0e} (horizon) above {a.tail_max:.0e}"
                        + (f": nodes {st['nodes']} -> {new}" if new != st["nodes"] else
                           (": the nodes cannot move further" if a.auto_nodes else " (no --auto_nodes)")))
                if new != st["nodes"]:
                    post.update(new_nodes=new, old_nodes=st["nodes"])
                    st["nodes"] = new
                    st["tag"] = st["tag"].split("_x")[0] + f"_x{new[0]:g}-{new[-1]:g}"
                    save_prog()
            if "new_nodes" in post:
                old, new = post["old_nodes"], post["new_nodes"]
                res = []
                for key in ("hist_w", "hist_n"):
                    D_, p_ = st[key][-1]
                    h, l, m = load(p_)
                    s0 = make_solver(D_, m["M"], nodes=old, kt=a.kt, kc=kc, ks=ks, threads=a.threads)
                    s0.set_segs(m.get("segs") or s0.plan(h, l, a.tol_step, a.hmax))
                    # Taylor steps of 0.001: renode's default 0.004 blows up toward a node at 0.9 at M = 160
                    zr, zrl = renode(s0, h, l, old, new, m["M"], h=0.001)
                    res.append((m["M"], np.asarray(zr) + np.asarray(zrl)))
                okw, rw2, pw2 = solve(Dt, res[0][0], [("re-noded", res[0][1])], plan_of(res[0][1], res[0][0], Dt))
                okn, rn2, pn2 = (solve(Dt, res[1][0], [("re-noded", res[1][1])], plan_of(res[1][1], res[1][0], Dt)) if okw
                                 else (False, None, None))
                if not (okw and okn):
                    return f"re-noded solve failed at D {Dt}"
                _, d2 = write_pair(Dt, rw2, rn2, f"re-noded {old} -> {new}")
                st["hist_w"], st["hist_n"] = [[Dt, pw2]], [[Dt, pn2]]
                st["pairs"].append([Dt, d2, st["Mw"], st["Mn"]])
                post["t_point"] += rw2["seconds"] + rn2["seconds"]
        # projected time of the next point: the solves of the current pair at this point (after a move up, the pair
        # at the new resolutions; after a re-noding, both solve sets)
        st["last_point_s"] = post["t_point"]
        log(f"point {Dt} took {post['t_point'] / 60:.1f} min of solves; {(time.time() - t_start) / 3600:.2f} h used")
        st["post"] = None
        save_prog()
        return None

    while True:
        if st.get("post"):
            why = post_point(st["post"])
            if why:
                return stop(why)
            continue
        if not st["targets"]:
            return stop(f"reached the end of the D grid ({a.D_end})")
        if time.time() - t_start > a.hours * 3600:
            return stop(f"{a.hours} h of wall time used")
        Dt = st["targets"][0]
        Dl = st["hist_n"][-1][0]
        if st.get("last_point_s") and st["last_point_s"] > a.point_hours * 3600:
            return stop(f"projected time of the next point {st['last_point_s'] / 3600:.1f} h > {a.point_hours} h")
        log(f"--- D {Dt} (from {Dl}), M {st['Mw']} / {st['Mn']}, nodes {st['nodes'][0]}..{st['nodes'][-1]}")
        Mw, Mn = st["Mw"], st["Mn"]
        zpw = predict([(d, zof(p)) for d, p in st["hist_w"][-4:]], Dt)
        ok_w, rw, path_w = solve(Dt, Mw, [("predictor", zpw)], plan_of(zpw, Mw, Dt))
        ok_n = False
        if ok_w:
            zpn = predict([(d, zof(p)) for d, p in st["hist_n"][-4:]], Dt)
            starts = [("lift of M_w", lifted(path_w, Mn))]
            if len(st["hist_n"]) >= 2:
                starts.insert(0, ("predictor", zpn))
            ok_n, rn, path_n = solve(Dt, Mn, starts, plan_of(zpn, Mn, Dt))
        if not (ok_w and ok_n):
            Dm = round(0.5 * (Dl + Dt), 6)
            if abs(Dm - Dl) < a.min_step:
                return stop(f"D {Dt} failed at M {Mw if not ok_w else Mn} with the step below {a.min_step}")
            log(f"D {Dt} failed at M {Mw if not ok_w else Mn}; inserting D {Dm}")
            st["targets"].insert(0, Dm)
            save_prog()
            continue
        _, diff = write_pair(Dt, rw, rn, "pair")
        st["hist_w"].append([Dt, path_w])
        st["hist_n"].append([Dt, path_n])
        st["pairs"].append([Dt, diff, Mw, Mn])
        st["targets"].pop(0)
        st["post"] = dict(D=Dt, stage="resolution", t_point=rw["seconds"] + rn["seconds"])
        save_prog()

if __name__ == "__main__":
    main()
