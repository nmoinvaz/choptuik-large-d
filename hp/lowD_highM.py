#!/usr/bin/env python3
"""Continuation in D toward D = 3 at two tau resolutions per point, moving up in M as needed (notes/HIGH_M.md).

    cd hp && python3 -u lowD_highM.py --start_w STATE_W.npz --start_n STATE_N.npz [--Ms 128,160,192,256,320,384]
        [--dD 0.05] [--D_end 3.0] [--min_step 0.005] [--diff_max 1e-8] [--hours 10] [--point_hours 3] [--auto_nodes]
        [--tag lowDhm] [--out data/lowD_highM/pilot.jsonl] [--points data/lowD_highM/pilot_points.jsonl]

The two start states are converged at the same D and nodes, at the working resolution M_w and the next one M_n
(consecutive in --Ms). At each new D (a grid of --dD below the start, D_end excluded):
  1. M_w: predictor = Lagrange (cubic) through the last <= 4 converged states at M_w (lowD_dd.solve_point: the
     f64-Jacobian Newton, Solver.newton_mixed, in block storage above M = 128; every step checkpointed);
  2. M_n: the better (smaller |F|) of the predictor through the last <= 4 states at M_n (if there are 2 or more) and
     the M_w solution just found, resampled to M_n;
  3. the value is Delta(M_n), its error estimate |Delta(M_n) - Delta(M_w)| (an estimator: the change from the next
     resolution, not a bound);
  4. if that difference exceeds --diff_max, or a log-linear fit of the last 3 differences in D projects it above
     --diff_max two points ahead, the resolution moves up (M_w <- M_n, M_n <- the next in --Ms), the new M_n solved at
     this D from the new M_w's solution, and the point recorded again with the new pair;
  5. --auto_nodes: if the M_n solution's centre or horizon series tail (lowD_dd.tails) exceeds --tail_max (1e-30),
     the nodes change as in lowD_dd.py (first node halved; a node 0.1 further out while the last is below 0.85), both
     resolutions are re-noded (driver.renode) and re-solved at this D under a new tag, and the predictors restart.
A failed solve inserts the midpoint between the last converged D and the target (halving the step).
Stop rules (each recorded with its reason): the largest M in --Ms cannot keep the difference below --diff_max; a
point fails with the step below --min_step; the projected time of the next point exceeds --point_hours (the last
point's two solves, times (M'/M)^2.9 for a move up); --hours of wall time in all (counted across restarts through
the progress file <out>.progress.json; --resume continues from it).
Records: one per solve in --points (lowD_dd.solve_point's record: Delta hi/lo, |F|, iterations, seconds, steps, tails,
sensitivities, harmonics, peak RSS), one per point and resolution pair in --out (D, M_w, M_n, both Deltas, the
difference, |F|, iterations, seconds, tails, the decision). States: hp/data/lowD_states/<tag>_D<D>_M<M>.npz.
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
                if r.get("converged") and abs(r["D"] - D) < 1e-9 and r["M"] == M and list(r["nodes"]) == list(nodes):
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
    st = dict(t_used=0.0, Mw=mw["M"], Mn=mn["M"], nodes=nodes, tag=a.tag,
              hist_w=[[D0, a.start_w]], hist_n=[[D0, a.start_n]], pairs=[], targets=None, done=None)
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

    def ns(M):
        return argparse.Namespace(out=a.points, tag=f"{st['tag']}", resume=True, kt=a.kt, kcs=kc, ks=ks, tol_step=a.tol_step,
                                  hmax=a.hmax, solver="newton", it_max=a.it_max, tol=a.tol, accept=a.accept, lm=0, jac="f64",
                                  threads=a.threads, svd=0, dn_its=2, w=24)

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

    def solve(D, M, starts):
        """solve at (D, M) from the start with the smallest |F| among starts [(label, z)]; returns (ok, record, path)"""
        s = make_solver(D, M, nodes=st["nodes"], kt=a.kt, kc=kc, ks=ks, threads=a.threads)
        best = None
        for lab, z in starts:
            zz = np.asarray(z, float)
            s.set_segs(s.plan(zz, np.zeros_like(zz), a.tol_step, a.hmax))
            F = float(np.abs(np.add(*s.residual(zz, np.zeros_like(zz)))).max())
            log(f"  D {D} M {M}: start '{lab}' |F| {F:.2e}")
            if np.isfinite(F) and (best is None or F < best[0]):
                best = (F, lab, zz)
        if best is None:
            return False, None, None
        t = time.time()
        ok, zh, zl = lowD_dd.solve_point(D, M, best[2], np.zeros_like(best[2]), f"pilot: {best[1]}", ns(M), st["nodes"], STATES, log)
        rec = find_record(a.points, D, M, st["nodes"]) if ok else None
        if ok and rec is None:
            return False, None, None
        if rec is not None:
            rec["wall"] = round(time.time() - t, 1)
        return ok, rec, state_path(D, M)

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
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"D {D}: Delta(M {rn['M']}) = {rn['Delta']:.16f} ({rn['Delta_lo']:+.2e}), M {rw['M']} -> {rn['M']}: {diff:+.2e}; "
            f"{decision}")
        return Dw, diff

    last_point_s = None
    while st["targets"]:
        if time.time() - t_start > a.hours * 3600:
            return stop(f"{a.hours} h of wall time used")
        Dt = st["targets"][0]
        Dl = st["hist_n"][-1][0]
        if last_point_s is not None and last_point_s > a.point_hours * 3600:
            return stop(f"projected time of the next point {last_point_s / 3600:.1f} h > {a.point_hours} h")
        log(f"--- D {Dt} (from {Dl}), M {st['Mw']} / {st['Mn']}, nodes {st['nodes'][0]}..{st['nodes'][-1]}")
        Mw, Mn = st["Mw"], st["Mn"]
        pw = [(d, zof(p)) for d, p in st["hist_w"][-4:]]
        ok_w, rw, path_w = solve(Dt, Mw, [("predictor", predict(pw, Dt))])
        ok_n = False
        if ok_w:
            starts = [("lift of M_w", lifted(path_w, Mn))]
            if len(st["hist_n"]) >= 2:
                starts.insert(0, ("predictor", predict([(d, zof(p)) for d, p in st["hist_n"][-4:]], Dt)))
            ok_n, rn, path_n = solve(Dt, Mn, starts)
        if not (ok_w and ok_n):
            Dm = round(0.5 * (Dl + Dt), 6)
            if abs(Dm - Dl) < a.min_step:
                return stop(f"D {Dt} failed at M {Mw if not ok_w else Mn} with the step below {a.min_step}")
            log(f"D {Dt} failed at M {Mw if not ok_w else Mn}; inserting D {Dm}")
            st["targets"].insert(0, Dm)
            save_prog()
            continue
        # a converged pair
        t_point = rw["seconds"] + rn["seconds"]
        _, diff = write_pair(Dt, rw, rn, "pair")
        st["hist_w"].append([Dt, path_w])
        st["hist_n"].append([Dt, path_n])
        st["pairs"].append([Dt, diff, Mw, Mn])
        st["targets"].pop(0)
        save_prog()
        # resolution: move up when the difference exceeds diff_max, or is projected to two points ahead
        same = [(d, x) for d, x, w_, n_ in st["pairs"] if (w_, n_) == (Mw, Mn)]
        D2 = round(Dt - 2 * a.dD, 6)
        proj = projected_diff(same, D2)
        why = None
        if abs(diff) > a.diff_max:
            why = f"|M {Mw} -> {Mn} difference| {abs(diff):.1e} > {a.diff_max:g}"
        elif proj is not None and proj > a.diff_max:
            why = f"difference projected to {proj:.1e} at D {D2} (> {a.diff_max:g})"
        if why:
            if Ms.index(Mn) + 1 >= len(Ms):
                if abs(diff) > a.diff_max:
                    return stop(f"the largest M built ({Mn}) cannot keep the difference below {a.diff_max:g} at D {Dt} "
                                f"(M {Mw} -> {Mn}: {diff:+.2e})")
                log(f"{why}, but M {Mn} is the largest built; continuing at M {Mw} / {Mn}")
            else:
                M2 = Ms[Ms.index(Mn) + 1]
                log(f"{why}: moving up to M {Mn} / {M2}")
                ok2, r2, p2 = solve(Dt, M2, [("lift of M_n", lifted(path_n, M2))])
                if not ok2:
                    return stop(f"the move up to M {M2} failed at D {Dt}")
                write_pair(Dt, rn, r2, f"moved up ({why})")
                st["Mw"], st["Mn"] = Mn, M2
                st["hist_w"] = st["hist_n"]
                st["hist_n"] = [[Dt, p2]]
                t_point = rn["seconds"] + r2["seconds"]
                save_prog()
        # series tails and nodes (of the current M_n solution at this D)
        rlast = find_record(a.points, Dt, st["Mn"], st["nodes"])
        tc, th = rlast.get("tail_c", 0.0), rlast.get("tail_h", 0.0)
        if a.auto_nodes and (tc > a.tail_max or th > a.tail_max):
            new = list(st["nodes"])
            if tc > a.tail_max:
                new = [round(new[0] / 2, 6)] + new
            if th > a.tail_max and new[-1] < 0.85:
                new = new + [round(new[-1] + 0.1, 6)]
            if new != st["nodes"]:
                log(f"D {Dt}: tails {tc:.0e} (centre) {th:.0e} (horizon) above {a.tail_max:.0e}: nodes {st['nodes']} -> {new}")
                old = st["nodes"]
                res = []
                for key in ("hist_w", "hist_n"):
                    D_, p_ = st[key][-1]
                    h, l, m = load(p_)
                    s0 = make_solver(D_, m["M"], nodes=old, kt=a.kt, kc=kc, ks=ks, threads=a.threads)
                    s0.set_segs(s0.plan(h, l, a.tol_step, a.hmax))
                    zr, zrl = renode(s0, h, l, old, new, m["M"])
                    res.append((m["M"], np.asarray(zr) + np.asarray(zrl)))
                st["nodes"] = new
                st["tag"] = st["tag"].split("_x")[0] + f"_x{new[0]:g}-{new[-1]:g}"
                okw, rw2, pw2 = solve(Dt, res[0][0], [("re-noded", res[0][1])])
                okn, rn2, pn2 = solve(Dt, res[1][0], [("re-noded", res[1][1])]) if okw else (False, None, None)
                if not (okw and okn):
                    return stop(f"re-noded solve failed at D {Dt}")
                write_pair(Dt, rw2, rn2, f"re-noded {old} -> {new}")
                st["hist_w"], st["hist_n"] = [[Dt, pw2]], [[Dt, pn2]]
                t_point += rw2["seconds"] + rn2["seconds"]
                save_prog()
            else:
                log(f"D {Dt}: tails {tc:.0e} {th:.0e} above {a.tail_max:.0e}, but the nodes cannot move further")
        elif tc > a.tail_max or th > a.tail_max:
            log(f"D {Dt}: tails {tc:.0e} (centre) {th:.0e} (horizon) above {a.tail_max:.0e} (no --auto_nodes)")
        # projected time of the next point: the two solves of the current pair at this point (after a move up, the
        # pair at the new resolutions)
        last_point_s = t_point
        log(f"point {Dt} took {t_point / 60:.1f} min; next point projected {last_point_s / 60:.1f} min; "
            f"{(time.time() - t_start) / 3600:.2f} h used")
    stop(f"reached the end of the D grid ({a.D_end})")


if __name__ == "__main__":
    main()
