#!/usr/bin/env python3
"""Real continuation steps at large D with each Newton variant, against all-double-double (notes/LARGE_D_MIXED.md
section 4). For each target D, hp/scan_hp.py runs as the scan ran there: the start is the saved scan state nearest to
it, the history the three saved states before it, the steps go to the scan's next two recorded points (from
hp/data/delta_dd.csv, same run), with kfreeze 13, deflated Newton with w = 24, the scan's tolerance there, the Lagrange
predictor, it_max 25, no Levenberg-Marquardt. One run per Jacobian variant (scan_hp.py --jac), each in its own
directory; then:
  - per point: Delta against the dd run's (and against the scan's recorded value), iterations, seconds, the time split
    (Jacobian / LU / solves / products), the products, |F|, dDelta_est; and the failures (halved steps) in the logs;
  - at the last point of each run: the sensitivities dDelta/dF (J^-T e_Delta) and the w = 24 weakest right singular
    directions from the variant's stored LU (the f64 LU for every mixed variant, as scan_hp.py uses them), against
    those of the double-double LU at the same state: the largest relative difference of dDelta/dF (per block maximum,
    as the scan records it, and over the whole vector), and the sine of the largest principal angle between the two
    weak subspaces.

    cd hp && python3 largeD_continuation.py [--D=40,50,60] [--jacs=dd,f64,f64h,f64g,f64hg] [--threads=2]
                                            [--out=data/mixed_precision/largeD_cont] [--report]
A run whose directory has a 'done' marker is skipped (rerun to resume); --report only prints the comparison.
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from hpkernel.driver import load, make_solver, newton_step, weak_subspace  # noqa: E402

# the scan's tolerance at each target (solver/notes.md: up7 1e-25, up8 1e-24, up13 ~3e-22)
TOL = {40: 1e-25, 45: 1e-25, 50: 1e-24, 55: 1e-22, 60: 3e-22}


def saved():
    out = []
    for p in glob.glob(os.path.join(HERE, "data", "states", "up*_D*.npz")):
        out.append((float(re.search(r"_D([0-9.]+)\.npz$", p).group(1)), p))
    return sorted(out)


def plan_for(Dt):
    """(start path, three history paths, the scan's next two D values and their recorded Delta, run label, tol)"""
    st = saved()
    i = min(range(len(st)), key=lambda k: abs(st[k][0] - Dt))
    D0, p0 = st[i]
    hist = [p for _, p in st[i - 3:i]]
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "delta_dd.csv"))))
    run = next(r["run"] for r in rows if abs(float(r["D"]) - D0) < 1e-6)
    nxt = sorted((float(r["D"]), float(r["Delta_hi"]), float(r["Delta_lo"]), r.get("dDelta_est")) for r in rows
                 if r["run"] == run and float(r["D"]) > D0 + 1e-9)[:2]
    tol = TOL[min(TOL, key=lambda d: abs(d - Dt))]
    return p0, hist, nxt, run, tol


def run_one(Dt, jac, outdir, threads):
    p0, hist, nxt, run, tol = plan_for(Dt)
    _, _, m = load(p0)
    d = os.path.join(outdir, f"D{m['D']:.4f}", jac)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "done")):
        print(f"D {m['D']:.4f} {jac}: done, skipped", flush=True)
        return
    for f in glob.glob(os.path.join(d, "*")):
        os.remove(f)
    cmd = [sys.executable, "-u", os.path.join(HERE, "scan_hp.py"), os.path.join(d, "scan.jsonl"), p0, "0",
           "--history", ",".join(hist), "--Dlist", ",".join(f"{x[0]}" for x in nxt), "--M", str(m["M"]),
           "--kc", str(m["kc"]), "--ks", str(m["ks"]), "--kt", str(m["kt"]), "--kfreeze", "13", "--tol", f"{tol:g}",
           "--w", "24", "--it_max", "25", "--lm", "0", "--threads", str(threads), "--tag", jac, "--jac", jac, "--verbose"]
    t = time.time()
    with open(os.path.join(d, "log.txt"), "w") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        r = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    json.dump(dict(returncode=r.returncode, wall=round(time.time() - t, 1), cmd=cmd), open(os.path.join(d, "done"), "w"))
    print(f"D {m['D']:.4f} {jac}: exit {r.returncode}, {time.time() - t:.0f} s", flush=True)


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
        p0, hist, nxt, run, tol = plan_for(Dt)
        _, _, m0 = load(p0)
        base = os.path.join(outdir, f"D{m0['D']:.4f}")
        recs = {}
        for jac in jacs:
            f = os.path.join(base, jac, "scan.jsonl")
            recs[jac] = [json.loads(x) for x in open(f)] if os.path.exists(f) else []
        scanned = {round(x[0], 6): x for x in nxt}
        print(f"\n== D {m0['D']:.4f} (run {run}, tol {tol:g}): steps to {', '.join(str(x[0]) for x in nxt)}")
        ref = {round(r["D"], 6): r for r in recs.get("dd", [])}
        for jac in jacs:
            log = os.path.join(base, jac, "log.txt")
            fails = sum(1 for line in open(log) if "no convergence" in line) if os.path.exists(log) else None
            for r in recs[jac]:
                key = round(r["D"], 6)
                q = ref.get(key)
                dDelta = None
                if q is not None:
                    dDelta = float((r["Delta"] - q["Delta"]) + (r["Delta_lo"] - q["Delta_lo"]))
                sc = scanned.get(key)
                dscan = float((r["Delta"] - sc[1]) + (r["Delta_lo"] - sc[2])) if sc else None
                sp = r.get("split", {})
                row = dict(D=r["D"], jac=jac, its=r["its"], seconds=r["seconds"], F=r["F"], dDelta_est=r.get("dDelta_est"),
                           dDelta_vs_dd=dDelta, dDelta_vs_scan=dscan, split=sp, fails=fails)
                out.append(row)
                print(f"  {jac:6s} D {r['D']:.6f}: its {r['its']:2d}, {r['seconds']:6.1f} s, |F| {r['F']:.1e}, "
                      f"dDelta_est {r.get('dDelta_est', 0):+.1e}, Delta - dd {dDelta if dDelta is not None else float('nan'):+.1e}, "
                      f"Delta - scan {dscan if dscan is not None else float('nan'):+.1e}"
                      + (f"; jac {sp.get('t_jac', 0):.1f} s, LU {sp.get('t_lu', 0):.2f} s, products {sp.get('t_prod', 0):.1f} s "
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
        for Dt in Ds:
            for jac in jacs:
                run_one(Dt, jac, outdir, threads)
    report(Ds, jacs, outdir, threads)


if __name__ == "__main__":
    main()
