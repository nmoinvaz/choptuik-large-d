#!/usr/bin/env python3
"""Real continuation steps at large D with each Newton variant, against all-double-double (notes/LARGE_D_MIXED.md
section 4). hp/scan_hp.py runs as the scan ran there (kfreeze 13, deflated Newton with w = 24, the scan's tolerance,
the Lagrange predictor through the last four converged points, it_max 25, no Levenberg-Marquardt), once per Jacobian
variant (scan_hp.py --jac), each in its own directory, from the same start and history to the same targets.

The scan's predictor needs a history spaced like its steps (0.01-0.1 in D); the saved states in hp/data/states are
one unit apart, and a cubic through them is off by |F| = 0.04 (D = 40) and 0.35 (D = 50) per unit of step, already
3.7e-6 and 3.5e-5 for a step of 1e-4 (outside the double-double Newton's basin for the scan's steps). So:
  - D = 60: the scan's own history, saved in hp/data/seeds: start up13_D59.8221, history up12_D59.7676 / 59.7858 /
    59.8041 (spacing 0.018), targets the scan's next two points (59.833359, 59.847422, run up13);
  - D = 40 and 50: first a double-double ladder from the saved state (history: the three saved states below it) to
    D0 + 2e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.02, 0.03 (D = 50: from 3e-4 on in ratios of 1.5, since the step to 1e-3
    failed there), one scan_hp.py call per point (--dD the gap, so a failure halves the step and ends the ladder;
    shared by all variants, directories 'ladder/x<offset>'); then each variant
    from D0 + 0.03 with the ladder's points at D0 + 0.01, 0.02 and the one at 3e-3 as history, to D0 + 0.06 and
    D0 + 0.09 (steps of 0.03, as the scan's 0.04-0.06 there).
Then:
  - per point: Delta against the dd run's (and the scan's recorded value where there is one), iterations, seconds, the
    time split (Jacobian / LU / solves / products), the products, |F|, dDelta_est; and the failures (halved steps);
  - at the dd run's last point: the sensitivities dDelta/dF (J^-T e_Delta) and the w = 24 weakest right singular
    directions from each variant's stored LU (the f64 LU for every mixed variant, as scan_hp.py uses them), against
    those of the double-double LU at the same state: the largest relative difference of dDelta/dF (per block maximum,
    as the scan records it, and over the whole vector), and the sine of the largest principal angle between the weak
    subspaces.

    cd hp && python3 largeD_continuation.py [--D=40,50,60] [--jacs=dd,f64,f64h,f64g,f64hg] [--threads=2]
                                            [--out=data/mixed_precision/largeD_cont] [--report]
A run whose directory has a 'done' marker is skipped (rerun to resume); --report only prints the comparison.
"""
import csv
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from hpkernel.driver import load, make_solver, newton_step, weak_subspace  # noqa: E402

# the scan's tolerance at each target (solver/notes.md: up7 1e-25, up8 1e-24, up13 3e-22: it accepted 1.6e-22)
TOL = {40: 1e-25, 50: 1e-24, 60: 3e-22}
LADDERS = {40: (2e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.02, 0.03),
           # at D = 50 the step from 3e-4 to 1e-3 failed (dd, |F| stalled at 4e-11): ratio 1.5 from there on
           50: (2e-5, 1e-4, 3e-4, 5e-4, 8e-4, 1.2e-3, 1.8e-3, 2.7e-3, 4e-3, 6e-3, 9e-3, 0.0135, 0.02, 0.03)}
STEPS = (0.06, 0.09)
STATES = os.path.join(HERE, "data", "states")
SEEDS = os.path.join(HERE, "data", "seeds")


def saved():
    return sorted((load(p)[2]["D"], p) for p in glob.glob(os.path.join(STATES, "up*_D*.npz")))


def csv_rows():
    return list(csv.DictReader(open(os.path.join(HERE, "data", "delta_dd.csv"))))


def scan_cmd(out, start, hist, targets, m, tol, jac, threads, dD=1.0):
    cmd = [sys.executable, "-u", os.path.join(HERE, "scan_hp.py"), out, start, "0", "--Dlist",
           ",".join(f"{x:.6f}" for x in targets), "--dD", f"{dD:g}", "--maxdD", f"{dD:g}", "--M", str(m["M"]), "--kc", str(m["kc"]), "--ks", str(m["ks"]),
           "--kt", str(m["kt"]), "--tol", f"{tol:g}", "--w", "24", "--it_max", "25", "--lm", "0", "--threads", str(threads),
           "--tag", jac, "--jac", jac, "--verbose", "--kfreeze", "13"]
    if hist:
        cmd[6:6] = ["--history", ",".join(hist)]
    return cmd


def run_cmd(d, cmd):
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "done")):
        print(f"{d}: done, skipped", flush=True)
        return
    for f in glob.glob(os.path.join(d, "*")):
        os.remove(f)
    t = time.time()
    with open(os.path.join(d, "log.txt"), "w") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        r = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    json.dump(dict(returncode=r.returncode, wall=round(time.time() - t, 1), cmd=cmd), open(os.path.join(d, "done"), "w"))
    print(f"{d}: exit {r.returncode}, {time.time() - t:.0f} s", flush=True)


def plan_for(Dt, outdir):
    """(base directory, start path, history paths, targets, the scan's recorded values {D: (hi, lo)}, tol, ladder
    step (dir, cmd) or None)"""
    rows = csv_rows()
    tol = TOL[min(TOL, key=lambda d: abs(d - Dt))]
    if Dt >= 59:
        start = os.path.join(SEEDS, "up13_D59.8221.npz")
        hist = [os.path.join(SEEDS, f) for f in ("up12_D59.7676.npz", "up12_D59.7858.npz", "up12_D59.8041.npz")]
        D0 = load(start)[2]["D"]
        nxt = sorted((float(r["D"]), float(r["Delta_hi"]), float(r["Delta_lo"])) for r in rows
                     if r["run"] == "up13" and float(r["D"]) > D0 + 1e-9)[:2]
        return os.path.join(outdir, f"D{D0:.4f}"), start, hist, [x[0] for x in nxt], {round(x[0], 6): x[1:] for x in nxt}, tol, None
    st = saved()
    i = min(range(len(st)), key=lambda k: abs(st[k][0] - Dt))
    D0, p0 = st[i]
    base = os.path.join(outdir, f"D{D0:.4f}")
    pts = {round(load(p)[2]["D"], 6): p for p in glob.glob(os.path.join(base, "ladder", "x*", "dd_D*.npz"))}
    have = [pts.get(round(D0 + x, 6)) for x in (3e-3, 0.01, 0.02, 0.03)]
    targets = [round(D0 + x, 6) for x in STEPS]
    return base, have[-1], have[:-1], targets, {}, tol, (st[i - 3:i + 1], D0, pts)


def run_all(Ds, jacs, outdir, threads):
    for Dt in Ds:
        base, start, hist, targets, _, tol, lad = plan_for(Dt, outdir)
        if lad is not None:
            coarse, D0, _ = lad
            chain = [p for _, p in coarse]          # the saved states, then the ladder's points as they come
            prev = D0
            for x in LADDERS[min(LADDERS, key=lambda d: abs(d - Dt))]:
                Dk = round(D0 + x, 6)
                d = os.path.join(base, "ladder", f"x{x:g}")
                m = load(chain[-1])[2]
                run_cmd(d, scan_cmd(os.path.join(d, "scan.jsonl"), chain[-1], chain[-4:-1], [Dk], m, tol, "dd", threads,
                                    dD=Dk - prev))
                got = [p for p in glob.glob(os.path.join(d, "dd_D*.npz")) if abs(load(p)[2]["D"] - Dk) < 1e-9]
                if not got:
                    break
                chain.append(got[0])
                prev = Dk
            base, start, hist, targets, _, tol, lad = plan_for(Dt, outdir)
            if start is None or any(h is None for h in hist):
                print(f"D {Dt}: the double-double ladder did not reach every point; no variant runs", flush=True)
                continue
        m = load(start)[2]
        for jac in jacs:
            run_cmd(os.path.join(base, jac), scan_cmd(os.path.join(base, jac, "scan.jsonl"), start, hist, targets, m, tol, jac, threads))


def lu_quantities(s, zh, zl, jac, w=24):
    """dDelta/dF (J^-T e_Delta, f64) and the w weakest right singular directions from the LU stored by one step of the
    variant jac at z"""
    newton_step(s, zh, zl, jac)
    n = s.size
    e = np.zeros(n)
    e[-1] = 1.0
    yh, yl = s.lu_solve_t(e, np.zeros(n))
    return np.asarray(yh) + np.asarray(yl), weak_subspace(s, w)


def report(Ds, jacs, outdir, threads):
    out = []
    for Dt in Ds:
        base, start, hist, targets, scanned, tol, _ = plan_for(Dt, outdir)
        recs = {}
        for jac in jacs:
            f = os.path.join(base, jac, "scan.jsonl")
            recs[jac] = [json.loads(x) for x in open(f)] if os.path.exists(f) else []
        print(f"\n== {os.path.basename(base)} (tol {tol:g}): from {os.path.basename(start or '?')} to "
              f"{', '.join(str(x) for x in targets)}")
        ref = {round(r["D"], 6): r for r in recs.get("dd", [])}
        for jac in jacs:
            log = os.path.join(base, jac, "log.txt")
            fails = sum(1 for line in open(log) if "no convergence" in line) if os.path.exists(log) else None
            for r in recs[jac]:
                key = round(r["D"], 6)
                q = ref.get(key)
                dDelta = float((r["Delta"] - q["Delta"]) + (r["Delta_lo"] - q["Delta_lo"])) if q is not None else None
                sc = scanned.get(key)
                dscan = float((r["Delta"] - sc[0]) + (r["Delta_lo"] - sc[1])) if sc else None
                sp = r.get("split", {})
                out.append(dict(D=r["D"], jac=jac, its=r["its"], seconds=r["seconds"], F=r["F"], dDelta_est=r.get("dDelta_est"),
                                dDelta_vs_dd=dDelta, dDelta_vs_scan=dscan, split=sp, fails=fails))
                print(f"  {jac:6s} D {r['D']:.6f}: its {r['its']:2d}, {r['seconds']:6.1f} s, |F| {r['F']:.1e}, "
                      f"dDelta_est {r.get('dDelta_est') or 0:+.1e}, Delta - dd {dDelta if dDelta is not None else float('nan'):+.1e}"
                      + (f", Delta - scan {dscan:+.1e}" if dscan is not None else "")
                      + (f"; Jacobian {sp.get('t_jac', 0):.1f} s, LU {sp.get('t_lu', 0):.2f} s, products {sp.get('t_prod', 0):.1f} s "
                         f"({sp.get('products', 0)})" if sp else "") + (f"; {fails} halved steps" if fails else ""), flush=True)
        # sensitivities and weak subspaces at the dd run's last point, from each variant's LU vs the dd LU
        last = sorted(glob.glob(os.path.join(base, "dd", "dd_D*.npz")))
        if not last:
            continue
        zh, zl, m = load(last[-1])
        s = make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads, kfreeze=13)
        s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
        y_dd, V_dd = lu_quantities(s, zh, zl, "dd")
        M = m["M"]
        nb = (s.size - 1) // (4 * M)
        blk = lambda y: np.array([np.abs(y[b * 4 * M:(b + 1) * 4 * M]).max() for b in range(nb)])  # noqa: E731
        for jac in jacs:
            if jac == "dd":
                continue
            y, V = lu_quantities(s, zh, zl, jac)
            rb = float(np.max(np.abs(blk(y) - blk(y_dd)) / blk(y_dd)))
            rv = float(np.max(np.abs(y - y_dd)) / np.max(np.abs(y_dd)))
            sang = float(np.sqrt(max(0.0, 1 - np.min(sla.svdvals(V_dd.T @ V)) ** 2)))
            out.append(dict(D=m["D"], jac=jac, sens_block_rel=rb, sens_vec_rel=rv, weak_sin_max=sang))
            print(f"  at D {m['D']:.6f}, {jac:6s} LU vs dd LU: dDelta/dF per-block max rel. diff {rb:.1e}, whole vector "
                  f"{rv:.1e}; weak subspace (w 24) largest principal angle sin {sang:.1e}", flush=True)
    json.dump(out, open(os.path.join(outdir, "report.json"), "w"), indent=1)


def main():
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    Ds = [float(x) for x in opts.get("D", "40,50,60").split(",")]
    jacs = opts.get("jacs", "dd,f64,f64h,f64g,f64hg").split(",")
    threads = int(opts.get("threads", 2))
    outdir = opts.get("out", os.path.join(HERE, "data", "mixed_precision", "largeD_cont"))
    if "report" not in opts:
        run_all(Ds, jacs, outdir, threads)
    report(Ds, jacs, outdir, threads)


if __name__ == "__main__":
    main()
