#!/usr/bin/env python3
"""Continuation in D toward D = 3 at two tau resolutions per point, moving up in M as needed (notes/HIGH_M.md).

    cd hp && python3 -u lowD_highM.py --start_w STATE_W.npz --start_n STATE_N.npz [--Ms 128,160,192,256,320,384]
        [--dD 0.05] [--D_end 3.0] [--min_step 0.005] [--diff_max 1e-8] [--hours 10] [--point_hours 3] [--auto_nodes]
        [--tag lowDhm] [--out data/lowD_highM/pilot.jsonl] [--points data/lowD_highM/pilot_points.jsonl]

The two start states are converged at the same D and nodes, at the working resolution M_w and the next one M_n
(consecutive in --Ms). At each new D (a grid of --dD below the start, D_end excluded):
  0. after the first move up, the previous working resolution M_c is kept as a helper: solved first from its own
     predictor (cheap), its solution resampled to M_w is a start for M_w (a predictor's first Newton step is large,
     |dz| ~ 3 at |F| ~ 3e-2, and at M = 128 it went astray in the weak directions; lifted starts converge in 3-5
     calls); its Delta is recorded with the pair (Delta_c, diff_c);
  1. M_w: predictor = Lagrange (cubic) through the last <= 4 converged states at M_w, or the lifted helper solution if
     its |F| is smaller; Newton (lowD_dd.solve_point: the f64-Jacobian Newton, Solver.newton_mixed, in block storage
     above M = 128) on the plan of the last converged state at M_w;
  2. M_n: plan from the predictor through the last <= 4 states at M_n; start from the better (smaller |F|) of that
     predictor (with 2 or more states) and the M_w solution just found, resampled to M_n;
  2a. if the best start of M_w or M_n has |F| above --two_pass_above (1e-3) on the reused plan, that plan does not fit
     it (on a lean plan the start's |F| is mostly Taylor error, and Newton went astray in the weak directions at
     D 3.25, M 160 from 3.5e-3): two passes as in a move up (4), from the start whose |F| is smallest on its own plan;
  3. the value is Delta(M_n), its error estimate |Delta(M_n) - Delta(M_w)| (an estimator: the change from the next
     resolution, not a bound);
  4. if that difference exceeds --diff_max, or a log-linear fit of the last 3 differences in D projects it above
     --diff_max two points ahead, the resolution moves up (M_w <- M_n, M_n <- the next in --Ms): the new M_n is solved
     at this D from the new M_w's solution resampled, in two passes (pass 1 on the resampled state's own plan to
     |F| <= 1e-14, state <tag>_p1_*; pass 2 on the plan of the pass-1 state, 2-4 times fewer steps), and the point is
     recorded again with the new pair;
  5. --auto_nodes: if the M_n solution's centre or horizon series tail (lowD_dd.tails) exceeds --tail_max (1e-30),
     the nodes change as in lowD_dd.py (a new first node at half the first; a node 0.1 further out while the last is
     below 0.85, then one halfway between the last and the horizon), all resolutions are re-noded (driver.renode)
     and re-solved at this D under a new tag, and the up to 3 earlier states at each resolution are re-noded (not
     re-solved) so that the predictors keep their cubic. The
     nodes also change when a series tail of the next point's M_n start (the M_w solution resampled to M_n) is above
     --lift_tail_max (1e-20): the centre tail of that start grows far faster than the solutions' as D falls (D 3.3:
     4e-8; D 3.25: 1e-2, and Newton went astray from that start).
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
from hpkernel.driver import load, make_solver, predict, renode, resample_z, save  # noqa: E402

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
    ap.add_argument("--plan_margin", type=float, default=1e-2, help="the step tolerance of the plan reused from the "
                    "last converged state, relative to --tol_step")
    ap.add_argument("--lift_tail_max", type=float, default=1e-20, help="with --auto_nodes, halve the first node when "
                    "the centre series tail of the next point's M_n start (the M_w solution resampled) is above this")
    ap.add_argument("--two_pass_above", type=float, default=1e-3, help="solve M_w or M_n in two passes when its best "
                    "start's |F| on the reused plan is above this (0: never)")
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
              hist_n=[[D0, a.start_n]], pairs=[], targets=None, done=None, post=None, last_point_s=None, Mc=None,
              hist_c=[])
    if a.resume and os.path.exists(prog_path):
        st = json.load(open(prog_path))
        log(f"resuming: M {st['Mw']}/{st['Mn']}, last D {st['hist_n'][-1][0]}, {st['t_used'] / 3600:.2f} h used")
    if st.get("done"):
        log(f"already stopped: {st['done']}")
        return
    st.setdefault("Mc", None)
    st.setdefault("hist_c", [])
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

    def plan_of(z, M, D, tol=None):
        """Taylor plan (--tol_step or tol, --hmax) of a state z (an f64 array, or a (hi, lo) pair: a converged state's
        plan needs its lo parts, rounded to f64 it is irregular enough at the sonic line to double the steps) at its
        own M, with the equations at D"""
        s0 = make_solver(D, M, nodes=st["nodes"], kt=a.kt, kc=kc, ks=ks, threads=a.threads)
        zh, zl = (np.asarray(z[0], float), np.asarray(z[1], float)) if isinstance(z, tuple) else (np.asarray(z, float), None)
        return s0.plan(zh, np.zeros_like(zh) if zl is None else zl, tol or a.tol_step, a.hmax)

    def zdd(path):
        h, l, _ = load(path)
        return np.asarray(h), np.asarray(l)

    def plan_prev(hist, M):
        """the plan for the next D at M: the plan of the last converged state at M, at its own D, with --plan_margin
        times the step tolerance. A state that is not a solution at the D of its equations (a predictor, a resampled
        state) is not regular at the sonic line, and its plan has 2.5-4 times more steps."""
        D_, p_ = hist[-1]
        return plan_of(zdd(p_), M, D_, a.tol_step * a.plan_margin)

    def plan_check(D, M, path, used):
        """steps of the converged solution's own plan against the steps used (a ratio above 1: the plan used was
        coarser than the solution's own at --tol_step)"""
        need = sum(len(g) - 1 for g in plan_of(zdd(path), M, D))
        have = sum(len(g) - 1 for g in used)
        if need > have:
            log(f"  D {D} M {M}: the solution's own plan has {need} steps, more than the {have} used")
        return need

    def solve(D, M, starts, segs, tag=None, tol=None, two_pass=False):
        """solve at (D, M) on the plan segs from the start with the smallest |F| among starts [(label, z)];
        returns (ok, record, path). The plan comes from a converged state or a predictor through converged states: a
        state resampled to a larger M needs 3.5-4 times more steps (its new top harmonics are aliasing products that
        the tau-derivative amplifies), which costs time, not accuracy. two_pass: if the best start's |F| on segs is
        above --two_pass_above, solve in two passes as in a move up (pass 1 from the start that is best on its own
        plan, to 1e-14; pass 2 on the plan of the pass-1 state)."""
        s = make_solver(D, M, nodes=st["nodes"], kt=a.kt, kc=kc, ks=ks, threads=a.threads)
        s.set_segs(segs)
        best = None
        for lab, z in starts:
            zz = (np.asarray(z[0], float), np.asarray(z[1], float)) if isinstance(z, tuple) else (np.asarray(z, float), np.zeros(len(z)))
            F = float(np.abs(np.add(*s.residual(*zz))).max())
            log(f"  D {D} M {M}: start '{lab}' |F| {F:.2e} ({sum(len(g) - 1 for g in segs)} steps)")
            if np.isfinite(F) and (best is None or F < best[0]):
                best = (F, lab, zz)
        if best is None or best[0] > 1.0:
            # the plan does not hold for these starts (top harmonics blown up): plan from the first start itself
            z = starts[0][1]
            zz = (np.asarray(z[0], float), np.asarray(z[1], float)) if isinstance(z, tuple) else (np.asarray(z, float), np.zeros(len(z)))
            segs = s.plan(*zz, a.tol_step, a.hmax)
            s.set_segs(segs)
            F = float(np.abs(np.add(*s.residual(*zz))).max())
            log(f"  D {D} M {M}: plan from start '{starts[0][0]}' instead: {sum(len(g) - 1 for g in segs)} steps, |F| {F:.2e}")
            if not np.isfinite(F):
                return False, None, None
            best = (F, starts[0][0], zz)
        if (two_pass and a.two_pass_above and best[0] > a.two_pass_above
                and not os.path.exists(os.path.join(STATES, f"{st['tag']}_D{D:.4f}_M{M}.npz"))):
            # the reused plan does not fit the starts: a lifted or predicted state is not regular at the sonic line, so
            # on a lean plan its |F| is mostly Taylor error, and Newton's correction of it lies in the weak directions
            # (at D 3.25, M 160 the lifted M 128 solution had |F| 3.5e-3 on the reused 257-step plan, and Newton went
            # astray: line-search factors 1/4, 1/128, 1/16). Each start on its own plan instead, then two passes.
            log(f"  D {D} M {M}: best start |F| {best[0]:.1e} > {a.two_pass_above:.0e} on the reused plan; two passes")
            own = []
            for lab, z in starts:
                zz = (np.asarray(z[0], float), np.asarray(z[1], float)) if isinstance(z, tuple) else (np.asarray(z, float), np.zeros(len(z)))
                sg = s.plan(*zz, a.tol_step, a.hmax)
                s.set_segs(sg)
                F = float(np.abs(np.add(*s.residual(*zz))).max())
                log(f"  D {D} M {M}: start '{lab}' |F| {F:.2e} on its own plan ({sum(len(g) - 1 for g in sg)} steps)")
                if np.isfinite(F):
                    own.append((F, lab, zz, sg))
            if not own:
                return False, None, None
            F1, lab1, zz1, sg1 = min(own, key=lambda o: o[0])
            t = time.time()
            ok1, _, p1 = solve(D, M, [(lab1, zz1)], sg1, tag=st["tag"] + "_p1", tol=1e-14)
            if not ok1:
                return False, None, None
            t1 = time.time() - t
            ok2, r2, p2 = solve(D, M, [("pass 1", zdd(p1))], plan_of(zdd(p1), M, D))
            if ok2:
                r2["seconds_pass1"] = round(t1, 1)
                r2["seconds"] = round(r2["seconds"] + t1, 1)
                r2["two_pass_from"] = lab1
            return ok2, r2, p2
        t = time.time()
        ok, zh, zl = lowD_dd.solve_point(D, M, best[2][0], best[2][1], f"pilot: {best[1]}", ns(M, tag, tol), st["nodes"],
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
        ok2, r2, p2 = solve(D, M2, [("pass 1", zdd(p1))], plan_of(zdd(p1), M2, D))
        if ok2:
            r2["seconds_pass1"] = round(t1, 1)
            r2["seconds"] = round(r2["seconds"] + t1, 1)
        return ok2, r2, p2

    def lifted(path, M_new):
        h, l, m = load(path)
        return resample_z(np.asarray(h) + np.asarray(l), m["M"], M_new, len(m["nodes"]))[0]

    def lift_tail(path, M_new):
        """centre and horizon series tails (lowD_dd.tails) of a state resampled to M_new, at its D and the current
        nodes"""
        m = load(path)[2]
        z = lifted(path, M_new)
        s0 = make_solver(m["D"], M_new, nodes=st["nodes"], kt=a.kt, kc=kc, ks=ks, threads=a.threads)
        return lowD_dd.tails(s0, z, 0 * z, st["nodes"], M_new)

    def write_pair(D, rw, rn, decision, extra=None):
        Dw = rw["Delta"] + rw["Delta_lo"]
        diff = (rn["Delta"] - rw["Delta"]) + (rn["Delta_lo"] - rw["Delta_lo"])
        rec = dict(D=D, Mw=rw["M"], Mn=rn["M"], Delta=rn["Delta"], Delta_lo=rn["Delta_lo"], err_est=abs(diff), diff=diff,
                   Delta_w=rw["Delta"], Delta_w_lo=rw["Delta_lo"], F_w=rw["F"], F_n=rn["F"], its_w=rw["its"], its_n=rn["its"],
                   seconds_w=rw["seconds"], seconds_n=rn["seconds"], steps_w=rw.get("steps"), steps_n=rn.get("steps"),
                   steps_own_w=rw.get("steps_own"), steps_own_n=rn.get("steps_own"), t_its_w=rw.get("t_its"),
                   t_its_n=rn.get("t_its"),
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
                st["Mc"], st["hist_c"] = Mw, st["hist_w"]
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
                # the start of the next point's M_n: the M_w solution resampled to M_n; its centre series converges far
                # worse than the solutions' (D 3.3: tail 4e-8 against 1e-60; D 3.25: 1e-2, |F| 3.5e-3 in the centre
                # block, and Newton went astray from it; with the first node halved, 3e-32 and |F| 1.9e-5)
                lt, lth = lift_tail(st["hist_w"][-1][1], st["Mn"])
                log(f"D {Dt}: series tails of the M {st['Mw']} solution resampled to M {st['Mn']}: {lt:.1e} (centre), "
                    f"{lth:.1e} (horizon)")
                if a.auto_nodes and (tc > a.tail_max or lt > a.lift_tail_max):
                    new = [round(new[0] / 2, 6)] + new
                    if lt > a.lift_tail_max:
                        log(f"D {Dt}: the centre tail is above {a.lift_tail_max:.0e}: first node {new[1]} -> {new[0]}")
                # the horizon series' radius in u = 1 - x shrinks as D falls (0.29 at D 3.7, 0.195 at 3.3, 0.155 at
                # 3.2, M = 160), so u0 = 1 - (last node) has to follow: a node 0.1 further out while the last is below
                # 0.85, then one halfway to the horizon
                if a.auto_nodes and (th > a.tail_max or lth > a.lift_tail_max):
                    new = new + [round(new[-1] + 0.1, 6) if new[-1] < 0.85 else round(1 - (1 - new[-1]) / 2, 6)]
                    if lth > a.lift_tail_max:
                        log(f"D {Dt}: the horizon tail is above {a.lift_tail_max:.0e}: last node {new[-2]} -> {new[-1]}")
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
                post.setdefault("old_hist", {k: st[k][:-1][-3:] for k in ("hist_c", "hist_w", "hist_n")})
                save_prog()
                res = []
                keys = ["hist_w", "hist_n"] + (["hist_c"] if st["hist_c"] and abs(st["hist_c"][-1][0] - Dt) < 1e-9 else [])
                for key in keys:
                    D_, p_ = st[key][-1]
                    h, l, m = load(p_)
                    s0 = make_solver(D_, m["M"], nodes=old, kt=a.kt, kc=kc, ks=ks, threads=a.threads)
                    s0.set_segs(m.get("segs") or s0.plan(h, l, a.tol_step, a.hmax))
                    # Taylor steps of 0.001: renode's default 0.004 blows up toward a node at 0.9 at M = 160
                    zr, zrl = renode(s0, h, l, old, new, m["M"], h=0.001)
                    res.append((m["M"], np.asarray(zr) + np.asarray(zrl), (np.asarray(zr), np.asarray(zrl))))
                okw, rw2, pw2 = solve(Dt, res[0][0], [("re-noded", res[0][2])], plan_of(res[0][2], res[0][0], Dt))
                okn, rn2, pn2 = (solve(Dt, res[1][0], [("re-noded", res[1][2])], plan_of(res[1][2], res[1][0], Dt)) if okw
                                 else (False, None, None))
                if not (okw and okn):
                    return f"re-noded solve failed at D {Dt}"
                _, d2 = write_pair(Dt, rw2, rn2, f"re-noded {old} -> {new}")
                st["hist_w"], st["hist_n"] = [[Dt, pw2]], [[Dt, pn2]]
                st["hist_c"] = []
                if len(res) > 2:
                    okc, rc2, pc2 = solve(Dt, res[2][0], [("re-noded", res[2][2])], plan_of(res[2][2], res[2][0], Dt))
                    if okc:
                        st["hist_c"] = [[Dt, pc2]]
                        post["t_point"] += rc2["seconds"]
                st["pairs"].append([Dt, d2, st["Mw"], st["Mn"]])
                post["t_point"] += rw2["seconds"] + rn2["seconds"]
                # the predictors keep their cubic: the up to 3 states before this point at each resolution, re-noded
                # (not re-solved: |F| ~ 1e-16 from the f64 evaluation of the centre series, far below a predictor's
                # error), saved as <new tag>_D*_M*_rn.npz
                for key in ("hist_c", "hist_w", "hist_n"):
                    if not st[key]:
                        continue
                    M_ = load(st[key][-1][1])[2]["M"]
                    pre = []
                    for D_, p_ in post["old_hist"].get(key, []):
                        h, l, m = load(p_)
                        if m["M"] != M_:
                            continue
                        out = os.path.join(STATES, f"{st['tag']}_D{D_:.4f}_M{M_}_rn.npz")
                        if not os.path.exists(out):
                            s0 = make_solver(D_, M_, nodes=old, kt=a.kt, kc=kc, ks=ks, threads=a.threads)
                            s0.set_segs(m.get("segs") or s0.plan(h, l, a.tol_step, a.hmax))
                            zr, zrl = renode(s0, h, l, old, new, M_, h=0.001)
                            save(out, np.asarray(zr), np.asarray(zrl), dict(m, nodes=new, renoded_from=p_))
                        pre.append([D_, out])
                    st[key] = pre + st[key]
                    log(f"  {key}: {len(pre)} earlier states re-noded for the predictor")
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
        Mw, Mn, Mc = st["Mw"], st["Mn"], st["Mc"]
        # the helper resolution (the previous working M, after a move up): solved from its own predictor, its solution
        # lifted is the working resolution's start; a predictor's first Newton step is large (|dz| ~ 3 at |F| ~ 3e-2)
        # and at M = 128 it went astray in the weak directions, while starts lifted from a converged lower M converge
        # in 3-5 calls
        rc = path_c = None
        if Mc and st["hist_c"]:
            zpc = predict([(d, zof(p)) for d, p in st["hist_c"][-4:]], Dt)
            ok_c, rc, path_c = solve(Dt, Mc, [("predictor", zpc)], plan_prev(st["hist_c"], Mc))
            if not ok_c:
                log(f"  D {Dt}: the helper M {Mc} did not converge; M {Mw} from its predictor")
                rc = path_c = None
        zpw = predict([(d, zof(p)) for d, p in st["hist_w"][-4:]], Dt)
        segs_w = plan_prev(st["hist_w"], Mw)
        starts_w = [("predictor", zpw)] + ([("lift of M_c", lifted(path_c, Mw))] if path_c else [])
        ok_w, rw, path_w = solve(Dt, Mw, starts_w, segs_w, two_pass=True)
        ok_n = False
        if ok_w:
            rw["steps_own"] = plan_check(Dt, Mw, path_w, load(path_w)[2]["segs"])
            zpn = predict([(d, zof(p)) for d, p in st["hist_n"][-4:]], Dt)
            starts = [("lift of M_w", lifted(path_w, Mn))]
            if len(st["hist_n"]) >= 2:
                starts.insert(0, ("predictor", zpn))
            segs_n = plan_prev(st["hist_n"], Mn)
            ok_n, rn, path_n = solve(Dt, Mn, starts, segs_n, two_pass=True)
            if ok_n:
                rn["steps_own"] = plan_check(Dt, Mn, path_n, load(path_n)[2]["segs"])
        if not (ok_w and ok_n):
            Dm = round(0.5 * (Dl + Dt), 6)
            if abs(Dm - Dl) < a.min_step:
                return stop(f"D {Dt} failed at M {Mw if not ok_w else Mn} with the step below {a.min_step}")
            log(f"D {Dt} failed at M {Mw if not ok_w else Mn}; inserting D {Dm}")
            st["targets"].insert(0, Dm)
            save_prog()
            continue
        extra = None
        if rc:
            extra = dict(Mc=Mc, Delta_c=rc["Delta"], Delta_c_lo=rc["Delta_lo"], F_c=rc["F"], its_c=rc["its"],
                         seconds_c=rc["seconds"], diff_c=(rw["Delta"] - rc["Delta"]) + (rw["Delta_lo"] - rc["Delta_lo"]))
        _, diff = write_pair(Dt, rw, rn, "pair", extra)
        st["hist_w"].append([Dt, path_w])
        st["hist_n"].append([Dt, path_n])
        if rc:
            st["hist_c"].append([Dt, path_c])
        st["pairs"].append([Dt, diff, Mw, Mn])
        st["targets"].pop(0)
        st["post"] = dict(D=Dt, stage="resolution", t_point=rw["seconds"] + rn["seconds"] + (rc["seconds"] if rc else 0))
        save_prog()

if __name__ == "__main__":
    main()
