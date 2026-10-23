#!/usr/bin/env python3
"""How large a step in D does the large-D continuation tolerate, and do three changes enlarge it? (local task T43)

The question behind reaching D = 200. The scans' steps fell from about 0.07 at D = 45-51 to 0.002-0.005 at D = 61, and
whether that fall is a power law or an exponential decides between about 2.6e5 and 1.2e8 points to D = 200
(notes/BEYOND_LIGHT_CONE.md). But the scans' steps were also capped (--maxdD 0.03 from D = 59.8, and growth only after
a point that took at most 8 Newton iterations), so they need not show the largest step Newton tolerates. Here the step
grows by 1.5 after every converged point and halves after every failure, so it settles at the edge. Runs start from
four saved scan states (D = 45, 50, 55, 60) with these variants:

  base     the scans' setup: nodes 0.3-0.7, p1 harmonics above 13 frozen, double-double Lagrange predictor through 4
           points (cubic), deflated Newton then plain Newton to the bottom (scan_hp.py --corrector dnp, w 16, it_max 25)
  node06   matching at 0.6 instead of 0.7 (nodes 0.3-0.6, ks 160). max|dDelta/dF| is smaller by e^{n[Phi(0.6) -
           Phi(0.7)]} (5e-7 at D = 60) and the residual floor higher by about (0.7/0.6)^n; their product, the bound on
           the error in Delta, came out 70-220 times smaller at D = 50-60 (solver/beyond_lc/hp_node_test.py, where Newton
           was not run to its floor at 0.6)
  pred6, pred8    the Lagrange predictor through 6 / 8 points (degree 5 / 7) instead of 4
  pin11, pin9     p1's harmonic 13 (pin11), or 11 and 13 (pin9), held on a linear trend in D instead of solved
           (scan_hp.py --pin: the weakest directions; one full-system Newton step after each point measures them)
  kf11     harmonic 13 frozen at zero (kfreeze 11). At D = 40 and 45.3 it did not converge (|F| 2e-9 and 2e-14):
           harmonic 13 is not small there (2.1e-6 at 45.3; solver/notes.md). Start-state check; stepped only if its
           floor is within 100 times base's

Phase 1, start states: each saved state is re-noded / re-frozen to the variant's layout and solved (deflated Newton,
then plain Newton) to its floor. The floor, Delta and max|dDelta/dF| are recorded (start_states.jsonl); this also
finishes the 0.6 node test with converged runs. Phase 2: one hp/scan_hp.py run per (D0, variant) from that start, tol
5 x the start's floor, rising 1.45 times per unit of D (the floor's own rise), for up to --max_points converged points,
--max_hours or --max_span of D. Phase 3 (--combo): one run per D0 that combines every option that beat base at 3 or
more of the 4 D0 (D per 100 Newton iterations at least 1.2 times base's, Delta check passed).

Each run gets the scan's own 7 states before D0 as predictor history (hp/data/seeds/step_test/, copied from the cloud
scans up7-up13; scan_hp.py re-nodes and re-converges them in the run's layout) and starts with the scan's last step
there. A run cannot start from one state: at D = 60 the residual of the converged state moved to D + 0.002 is 4.5e12,
a linear extrapolation from states one unit apart gives 1e4 and a cubic one 8e-3, and the Jacobian-based tangent
(driver.tangent) is 1e18 in size, and with the 16 weakest directions removed its Delta component is still -2.9e5 per
unit of D (the scan's points give -0.0027); Newton does not move from any of these. So the predictor needs closely spaced converged points, and the first converged points of each run still
carry the history's spacing: the steady figures leave out the attempts before the run has n_pred points of its own.

Reported per run (summary.json, summary.md): the start's floor and tol; the converged points and failures; the D
covered; the largest converged step; the edge (median of the last 8 attempted steps); the Newton iterations of all
attempts (polish included); D per 100 iterations (machine-independent) and per hour; the largest |dDelta_est|; and
the largest |Delta - reference|. The reference is hp/data/delta_dd.csv (the scans' points, D >= 40): a local
degree-6 fit in D through the 16 nearest points for D <= 61.0 (check: |Delta - ref| <= 1e-12), and above it a
Chebyshev fit in 1/D on D = 37-61.03, extrapolated (check: <= 1e-10). --mutate=ref shifts the reference by 1e-9, so
every check must then fail; the report exits 1 when any check fails.

    cd hp && python3 step_test.py --outdir data/step_test/pilot --D 60.0020 --variants base,node06 --max_points 8
    cd hp && python3 step_test.py [--D 45.0507,50.0075,55.0351,60.0020]
                                  [--variants base,node06,pred6,pred8,pin11,pin9,kf11] [--combo]
                                  [--threads 3] [--parallel 2] [--max_points 20] [--max_hours 1] [--max_span 3]
    cd hp && python3 step_test.py --report [--outdir ...] [--mutate=ref]
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import (deflated_newton, delta_correction, dd_norm, load, make_solver, newton_solve,  # noqa: E402
                             plow_odd, renode, save)

STATES = {45.0507: "up7_D45.0507", 50.0075: "up8_D50.0075", 55.0351: "up10_D55.0351", 60.0020: "up13_D60.0020"}
BASE_NODES = [0.3, 0.4, 0.5, 0.6, 0.7]
KC = 220           # enough at D >= 36 (hp/kc_check.py); the runs lower it further with --kc_auto 30 (bitwise safe)
KS = {0.7: 100, 0.6: 160}
VARIANTS = {
    "base": dict(last=0.7, kf=13, args=[]),
    "node06": dict(last=0.6, kf=13, args=[]),
    "pred6": dict(last=0.7, kf=13, args=["--pred_n", "6"]),
    "pred8": dict(last=0.7, kf=13, args=["--pred_n", "8"]),
    "pin11": dict(last=0.7, kf=13, args=["--pin", "11"]),
    "pin9": dict(last=0.7, kf=13, args=["--pin", "9"]),
    "kf11": dict(last=0.7, kf=11, args=[]),
}
REF_LOCAL_TOL, REF_FAR_TOL, REF_LOCAL_MAX = 1e-12, 1e-10, 61.0
SEEDS = os.path.join(HERE, "data", "seeds", "step_test")


def history(D0):
    """the scan's 7 states before D0 (oldest first)"""
    out = []
    for p in glob.glob(os.path.join(SEEDS, "up*_D*.npz")):
        d = float(os.path.basename(p).rsplit("_D", 1)[1][:-4])
        if D0 - 1.0 < d < D0 - 1e-6:
            out.append((d, p))
    return [p for _, p in sorted(out)][-7:], (D0 - max(d for d, _ in out)) if out else 0.01


def layout(last):
    return [x for x in BASE_NODES if x < last - 1e-9] + [last]


def start_name(D0, last, kf):
    return f"start_D{D0:.4f}_x{last}_kf{kf}"


def sensitivity(s, M):
    L = s.size
    e = np.zeros(L)
    e[-1] = 1.0
    yh, _ = s.lu_solve_t(e, np.zeros(L))
    return float(np.abs(yh[:-1]).max())


def tails(s, zh, zl, nodes, M):
    out = []
    for which, x in ((0, nodes[0]), (1, 1 - nodes[-1])):
        h, l = s.series(zh, zl, which)
        c = np.abs((np.asarray(h) + np.asarray(l)).reshape(-1, 4, M)).max(axis=(1, 2))
        out.append(float((c * x ** np.arange(len(c)))[-3:].max()))
    return out


def jsonl(path):
    return [json.loads(x) for x in open(path) if x.strip()] if os.path.exists(path) else []


# ----------------------------------------------------------------------------------------------- phase 1: start states
def make_start(D0, last, kf, outdir, threads):
    """the saved state at D0 in the variant's layout, solved to its floor; returns its record (cached)"""
    name = start_name(D0, last, kf)
    recs = [r for r in jsonl(os.path.join(outdir, "start_states.jsonl")) if r["name"] == name]
    if recs and os.path.exists(os.path.join(outdir, name + ".npz")):
        return recs[-1]
    zh0, zl0, m = load(os.path.join(HERE, "data", "states", STATES[D0] + ".npz"))
    D, M, nodes0 = m["D"], m["M"], list(m["nodes"])
    nodes = layout(last)
    t0 = time.time()
    s0 = make_solver(D, M, nodes=nodes0, kt=m["kt"], kc=KC, ks=KS[0.7], threads=threads, kfreeze=13)
    s0.set_segs(s0.plan(zh0, zl0, 1e-32, 0.05))
    zh, zl = (zh0.copy(), zl0.copy()) if nodes == nodes0 else renode(s0, zh0, zl0, nodes0, nodes, M)
    if kf != 13:
        lo, lo2 = plow_odd(M, kf)
        zh[:M] = (lo + lo2) @ (zh[:M] + zl[:M])
        zl[:M] = 0.0
    s = make_solver(D, M, nodes=nodes, kt=m["kt"], kc=KC, ks=KS[last], threads=threads, kfreeze=kf)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    F0 = dd_norm(*s.residual(zh, zl))
    zh1, zl1, h1 = deflated_newton(s, zh, zl, tol=1e-40, it_max=25, w=16, verbose=False)
    zh2, zl2, h2 = newton_solve(s, zh1, zl1, tol=1e-40, it_max=6, verbose=False)
    Fs = [F0] + [r["F"] for r in h1 + h2]
    s.newton(zh2, zl2)                     # the stored LU at the final iterate, for the sensitivity
    Fh, Fl = s.residual(zh2, zl2)
    F_final = dd_norm(Fh, Fl)
    rec = dict(name=name, D=D, D0=D0, last=last, kfreeze=kf, nodes=nodes, M=M, kt=m["kt"], ks=KS[last], F_start=F0,
               F_hist=Fs, floor=min(Fs[1:] + [F_final]), F_final=F_final, Delta=float(zh2[-1]), Delta_lo=float(zl2[-1]),
               Delta_saved=float(zh0[-1]) + float(zl0[-1]), dDelta_est=delta_correction(s, Fh, Fl),
               sens_max=sensitivity(s, M), tails=tails(s, zh2, zl2, nodes, M), its=len(h1) + len(h2),
               seconds=round(time.time() - t0, 1))
    save(os.path.join(outdir, name + ".npz"), zh2, zl2, dict(D=D, M=M, nodes=nodes, kt=m["kt"], kc=KC, ks=KS[last],
                                                            kfreeze=kf))
    with open(os.path.join(outdir, "start_states.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"start {name}: |F| {F0:.1e} -> floor {rec['floor']:.1e} in {rec['its']} its; Delta {rec['Delta']:.16f} "
          f"(saved state {rec['Delta_saved'] - rec['Delta']:+.1e} from it); max dDelta/dF {rec['sens_max']:.2e}; "
          f"tails {rec['tails'][0]:.0e} {rec['tails'][1]:.0e} ({rec['seconds']} s)", flush=True)
    return rec


# ------------------------------------------------------------------------------------------------ phase 2: the runs
def run_cmd(D0, v, spec, st, outdir, a):
    tag = f"{v}_D{D0:.2f}"
    tol = 5 * st["floor"]
    hist, step0 = history(D0)
    if len(hist) < 7:
        raise SystemExit(f"D0 = {D0}: {len(hist)} history states in {SEEDS}, need 7")
    cmd = [sys.executable, "-u", os.path.join(HERE, "scan_hp.py"), os.path.join(outdir, tag + ".jsonl"),
           os.path.join(outdir, st["name"] + ".npz"), str(round(D0 + 50, 4)), "--history", ",".join(hist),
           "--dD", f"{step0:.6f}", "--dD0", f"{step0:.6f}", "--maxdD", "2.0", "--kt", "36", "--kc", str(KC),
           "--kc_auto", "30",
           "--ks", str(KS[spec["last"]]), "--tol", f"{tol:.3e}", "--tol_growth", "1.45", "--it_max", "25", "--lm", "0",
           "--w", "16", "--kfreeze", str(spec["kf"]), "--corrector", "dnp", "--predictor", "lagrange_dd",
           "--grow_its", "99", "--grow", "1.5", "--fail_factor", "0.5", "--min_dD", "1e-4",
           "--max_points", str(a.max_points), "--max_hours", str(a.max_hours), "--max_span", str(a.max_span),
           "--max_fail_row", "6", "--attempts", os.path.join(outdir, tag + "_attempts.jsonl"), "--tag", tag,
           "--threads", str(a.threads)] + spec["args"]
    return tag, cmd


def finished(outdir, tag):
    return any("end" in r for r in jsonl(os.path.join(outdir, tag + "_attempts.jsonl")))


def launch(cmd, log):
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        kw["preexec_fn"] = lambda: os.nice(10)
    return subprocess.Popen(cmd, cwd=HERE, stdout=open(log, "a"), stderr=subprocess.STDOUT, **kw)


def run_all(jobs, outdir, parallel):
    """jobs: (tag, cmd); runs those not finished, `parallel` at a time; a run that dies without an end record (a crash,
    a restart) is started again from its start state, its partial files kept under a .partN suffix"""
    todo = [j for j in jobs if not finished(outdir, j[0])]
    running = []
    while todo or running:
        while todo and len(running) < parallel:
            tag, cmd = todo.pop(0)
            for suf in ("_attempts.jsonl", ".jsonl"):
                p = os.path.join(outdir, tag + suf)
                if os.path.exists(p):
                    k = 1
                    while os.path.exists(f"{p}.part{k}"):
                        k += 1
                    os.replace(p, f"{p}.part{k}")
            print(f"{time.strftime('%H:%M:%S')} start {tag}", flush=True)
            running.append((tag, launch(cmd, os.path.join(outdir, tag + ".log"))))
        time.sleep(5)
        for tag, p in list(running):
            if p.poll() is not None:
                running.remove((tag, p))
                end = [r for r in jsonl(os.path.join(outdir, tag + "_attempts.jsonl")) if "end" in r]
                print(f"{time.strftime('%H:%M:%S')} {tag} exited {p.returncode}: "
                      f"{end[-1]['end'] if end else 'no end record (see the log)'}", flush=True)
        # remove intermediate states, keeping each run's last 8 (history for continuing it)
        for tag, _ in jobs:
            sts = sorted(glob.glob(os.path.join(outdir, f"{tag}_D[0-9]*.npz")), key=os.path.getmtime)
            if finished(outdir, tag):
                for f in sts[:-8]:
                    os.remove(f)


# -------------------------------------------------------------------------------------------------------- report
def reference(mutate):
    """Delta(D) from delta_dd.csv: local degree-6 fits up to REF_LOCAL_MAX, a Chebyshev fit in 1/D above"""
    pts = []
    with open(os.path.join(HERE, "data", "delta_dd.csv")) as fh:
        for r in csv.DictReader(fh):
            if not r["run"].startswith("lowD") and float(r["D"]) >= 37:     # above the first kfreeze-13 points
                pts.append((float(r["D"]), float(r["Delta_hi"]) + float(r["Delta_lo"])))
    pts.sort()
    Dv = np.array([p[0] for p in pts])
    Yv = np.array([p[1] for p in pts])
    cheb = np.polynomial.Chebyshev.fit(1.0 / Dv, Yv, 14)
    far_rms = float(np.sqrt(np.mean((cheb(1.0 / Dv) - Yv) ** 2)))
    shift = 1e-9 if mutate == "ref" else 0.0

    def ref(D):
        if 40 <= D <= REF_LOCAL_MAX:
            idx = np.argsort(np.abs(Dv - D))[:16]
            c = np.polyfit(Dv[idx] - D, Yv[idx], 6)
            return float(c[-1]) + shift, REF_LOCAL_TOL
        return float(cheb(1.0 / D)) + shift, REF_FAR_TOL
    return ref, far_rms, float(Dv.max())


def summarize(outdir, mutate=None):
    ref, far_rms, dmax = reference(mutate)
    starts = {r["name"]: r for r in jsonl(os.path.join(outdir, "start_states.jsonl"))}
    rows = []
    for att in sorted(glob.glob(os.path.join(outdir, "*_attempts.jsonl"))):
        tag = os.path.basename(att)[:-len("_attempts.jsonl")]
        v, D0s = tag.rsplit("_D", 1)
        A = jsonl(att)
        tries = [r for r in A if "end" not in r]
        end = [r for r in A if "end" in r]
        ok = [r for r in tries if r["ok"]]
        if not tries:
            continue
        D0 = tries[0]["D_from"]
        n_pred = 8 if "pred8" in v else 6 if "pred6" in v else 4
        # steady part: the attempts after the predictor has its n_pred points (the first converged points are warm-up)
        i0 = next((i for i in range(len(tries) + 1) if sum(q["ok"] for q in tries[:i]) >= n_pred), len(tries))
        steady = tries[i0:]
        its = sum(r["its"] for r in tries)
        span = abs(ok[-1]["D_to"] - D0) if ok else 0.0
        s_ok = [r for r in steady if r["ok"]]
        s_span = abs(s_ok[-1]["D_to"] - steady[0]["D_from"]) if s_ok else 0.0
        s_its = sum(r["its"] for r in steady)
        s_hours = (steady[-1]["t_wall"] - (tries[i0 - 1]["t_wall"] if i0 else 0.0)) / 3600 if steady else 0.0
        dev = []
        for r in ok:
            rv, tol = ref(r["D_to"])
            dev.append((abs(r["Delta"] + r["Delta_lo"] - rv), tol, r["D_to"]))
        recs = jsonl(os.path.join(outdir, tag + ".jsonl"))
        row = dict(tag=tag, variant=v, D0=D0, end=end[-1]["end"] if end else "running or crashed", n_ok=len(ok),
                   n_fail=len(tries) - len(ok), span=span, max_ok_dD=max((r["dD"] for r in ok), default=0.0),
                   edge=float(np.median([r["dD"] for r in tries[-8:]])), its=its,
                   D_per_100its=100 * span / its if its else 0.0, hours=tries[-1]["t_wall"] / 3600,
                   steady_n=len(steady), steady_D_per_100its=100 * s_span / s_its if s_its else 0.0,
                   steady_D_per_hour=s_span / s_hours if s_hours else 0.0,
                   fail_F=[f"{r['F_last']:.0e}" for r in tries if not r["ok"]],
                   max_dDelta_est=max((abs(r.get("dDelta_est") or 0.0) for r in recs), default=None),
                   max_dev=max((d[0] for d in dev), default=None),
                   check=all(d[0] <= d[1] for d in dev) if dev else None,
                   max_D=max((d[2] for d in dev), default=None))
        rows.append(row)
    return rows, starts, far_rms, dmax


def report(outdir, mutate):
    rows, starts, far_rms, dmax = summarize(outdir, mutate)
    json.dump(dict(rows=rows, starts=list(starts.values()), ref_far_rms=far_rms, ref_dmax=dmax, mutate=mutate),
              open(os.path.join(outdir, "summary.json" if not mutate else f"summary_mutate_{mutate}.json"), "w"), indent=1)
    out = ["# Step-size test (hp/step_test.py)", "",
           f"Reference: delta_dd.csv to D = {dmax}; Chebyshev fit rms {far_rms:.1e} (used above D = {REF_LOCAL_MAX})."
           + (f" MUTATED: reference shifted ({mutate})." if mutate else ""), "",
           "## Start states", "",
           "| start | D | floor | max dDelta/dF | floor x sens | Delta - saved state | its |",
           "|---|---|---|---|---|---|---|"]
    for r in starts.values():
        out.append(f"| {r['name']} | {r['D']:.4f} | {r['floor']:.1e} | {r['sens_max']:.1e} | "
                   f"{r['floor'] * r['sens_max']:.1e} | {r['Delta'] - r['Delta_saved']:+.1e} | {r['its']} |")
    out += ["", "## Runs", "",
            "| run | end | ok / failed | D covered | largest step | edge | Newton its | D per 100 its (steady) | "
            "D per hour (steady) | max abs dDelta_est | max abs(Delta - ref) | check |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["D0"], r["variant"])):
        out.append(f"| {r['tag']} | {r['end']} | {r['n_ok']} / {r['n_fail']} | {r['span']:.4f} | {r['max_ok_dD']:.4f} | "
                   f"{r['edge']:.4f} | {r['its']} | {r['D_per_100its']:.4f} ({r['steady_D_per_100its']:.4f}) | "
                   f"{r['steady_D_per_hour']:.3f} | {r['max_dDelta_est'] or 0:.1e} | "
                   f"{(r['max_dev'] if r['max_dev'] is not None else float('nan')):.1e} | "
                   f"{'pass' if r['check'] else 'FAIL' if r['check'] is not None else '-'} |")
    txt = "\n".join(out) + "\n"
    open(os.path.join(outdir, "summary.md" if not mutate else f"summary_mutate_{mutate}.md"), "w").write(txt)
    print(txt)
    failed = [r["tag"] for r in rows if r["check"] is False]
    if failed:
        print(f"Delta check FAILED for: {', '.join(failed)}")
        return 1
    return 0


def combo_spec(rows):
    """the options that beat base at >= 3 of the 4 D0 on steady D per 100 iterations (>= 1.2x) with the check passed"""
    base = {r["D0"]: r for r in rows if r["variant"] == "base"}
    wins = {}
    for r in rows:
        b = base.get(r["D0"])
        if r["variant"] in ("base", "kf11") or b is None or not r["check"]:
            continue
        if b["steady_D_per_100its"] and r["steady_D_per_100its"] >= 1.2 * b["steady_D_per_100its"]:
            wins.setdefault(r["variant"], 0)
            wins[r["variant"]] += 1
    good = [v for v, n in wins.items() if n >= 3]
    score = lambda v: sum(r["steady_D_per_100its"] for r in rows if r["variant"] == v)
    pick = {}
    if "node06" in good:
        pick["node06"] = True
    preds = [v for v in good if v.startswith("pred")]
    if preds:
        pick["pred"] = max(preds, key=score)
    pins = [v for v in good if v.startswith("pin")]
    if pins:
        pick["pin"] = max(pins, key=score)
    if len(pick) < 2:
        return None, good
    args = []
    if "pred" in pick:
        args += VARIANTS[pick["pred"]]["args"]
    if "pin" in pick:
        args += VARIANTS[pick["pin"]]["args"]
    name = "combo_" + "+".join(([("node06")] if "node06" in pick else []) + [pick[k] for k in ("pred", "pin") if k in pick])
    return (name, dict(last=0.6 if "node06" in pick else 0.7, kf=13, args=args)), good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "data", "step_test"))
    ap.add_argument("--D", default="45.0507,50.0075,55.0351,60.0020")
    ap.add_argument("--variants", default="base,node06,pred6,pred8,pin11,pin9,kf11")
    ap.add_argument("--combo", action="store_true")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--max_points", type=int, default=20)
    ap.add_argument("--max_hours", type=float, default=1.0)
    ap.add_argument("--max_span", type=float, default=3.0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--mutate", default=None, choices=[None, "ref"])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.report:
        return report(a.outdir, a.mutate)
    D0s = [float(x) for x in a.D.split(",")]
    vs = a.variants.split(",")
    # phase 1 (serial; each solve uses all the threads)
    starts = {}
    for D0 in D0s:
        for v in vs:
            spec = VARIANTS[v]
            key = (D0, spec["last"], spec["kf"])
            if key not in starts:
                starts[key] = make_start(D0, spec["last"], spec["kf"], a.outdir, a.threads * a.parallel)
    # phase 2
    jobs = []
    for D0 in D0s:
        for v in vs:
            spec = VARIANTS[v]
            st = starts[(D0, spec["last"], spec["kf"])]
            base_st = starts.get((D0, 0.7, 13))
            if v == "kf11" and base_st is not None and st["floor"] > 100 * base_st["floor"]:
                print(f"kf11 at D0 = {D0}: floor {st['floor']:.1e}, more than 100 x base's: not stepped", flush=True)
                continue
            jobs.append(run_cmd(D0, v, spec, st, a.outdir, a))
    run_all(jobs, a.outdir, a.parallel)
    rc = report(a.outdir, None)
    # phase 3
    if a.combo:
        rows = summarize(a.outdir)[0]
        spec, good = combo_spec(rows)
        print(f"options that beat base at 3+ of 4 D0: {good or 'none'}", flush=True)
        if spec:
            name, sp = spec
            cj = []
            for D0 in D0s:
                st = make_start(D0, sp["last"], sp["kf"], a.outdir, a.threads * a.parallel)
                cj.append(run_cmd(D0, name, sp, st, a.outdir, a))
            run_all(cj, a.outdir, a.parallel)
            rc = report(a.outdir, None)
        else:
            print("fewer than two winning options: no combined run", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
