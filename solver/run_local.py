#!/usr/bin/env python3
"""Run a matrix of nonlinear-elimination D scans on this machine, several at a time.

Two constraints decide the setup (CONTEXT.md):
* the horizon series converges only for u = 1 - x_K below a radius that shrinks with D (0.70 at D = 23, 0.47 at 30,
  0.43 at 36): x_K = 0.4 fails above D ~ 25.5, x_K = 0.5 above ~29, x_K = 0.6 degrades above ~32;
* the period is selected through u_K^s0, so a larger x_K selects more weakly and needs finer resolution (kc) and the
  smoothness fix of the junk directions.
So the high-D jobs match at x_K = 0.7 with kc 31 / 39 and smoothing. Jobs:

    sj_07      x_K = 0.7, kc = 23, smoothed      D 23 -> 45     -> data/scan_ne7_sj.jsonl
    kc39_07    x_K = 0.7, kc = 39, smoothed      D 23 -> 45     -> data/scan_ne7_kc39_sj.jsonl (slowest)
    sj_06      x_K = 0.6, kc = 23, smoothed      D 23 -> 34     -> data/scan_ne6_sj.jsonl
    kc31_06    x_K = 0.6, kc = 31                D 23 -> 34     -> data/scan_ne6_kc31.jsonl
  (x4_23, x5_31, kc31_07, kc31_07_sj also available; the cloud runs x_K = 0.7, kc = 31, smoothed)

Each job re-solves stored seeds for D = 23-24, then continues in D, one JSON line per point (Delta, residual split,
gamma), solution vectors in data/.

Usage (repo root, after `pip install .`):
    python solver/run_local.py                      # the four jobs above, one thread each, four at a time
    python solver/run_local.py --only kc39_07 --threads 2
Commit solver/data/ and push whenever you like (partial results are useful); rerunning a job appends.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# name: (output, scan_ne.py arguments, last seed to re-solve before continuing, extra environment)
SJ = {"NE_SMOOTH": "1", "NE_STRONG_TOL": "1e-8"}
JOBS = {
    "sj_07": ("scan_ne7_sj.jsonl", ["23", "45", "0.2", "0.4", "0.7", "../seeds/long7_D", "23"], "24.0", SJ),
    "kc39_07": ("scan_ne7_kc39_sj.jsonl", ["23", "45", "0.2", "0.4", "0.7", "../seeds/long7_D", "39"], "24.0", SJ),
    "sj_06": ("scan_ne6_sj.jsonl", ["23", "34", "0.2", "0.4", "0.6", "../seeds/neB_D", "23"], "24.0", SJ),
    "kc31_06": ("scan_ne6_kc31.jsonl", ["23", "34", "0.2", "0.4", "0.6", "../seeds/neB_D", "31"], "24.0", {}),
    "kc31_07_sj": ("scan_ne7_kc31_sj.jsonl", ["23", "45", "0.2", "0.4", "0.7", "../seeds/long7_D", "31"], "24.0", SJ),
    "kc31_07": ("scan_ne7_kc31.jsonl", ["23", "40", "0.2", "0.4", "0.7", "../seeds/long7_D", "31"], "24.0", {}),
    "x4_23": ("scan_ne4.jsonl", ["23", "25.5", "0.2", "0.4", "0.4", "../seeds/neB_D", "23"], "24.0", dict(SJ, NE_KS="260")),
    "x5_31": ("scan_ne5_kc31.jsonl", ["23", "29", "0.2", "0.4", "0.5", "../seeds/neB_D", "31"], "24.0", dict(SJ, NE_KS="120")),
}
DEFAULT = ["sj_07", "kc39_07", "sj_06", "kc31_06"]


def last_record(log):
    try:
        with open(log) as f:
            lines = [l for l in f if l.startswith("{") or "giving up" in l or "Traceback" in l]
    except OSError:
        return None
    return lines[-1].strip() if lines else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=1, help="threads per scan (default 1)")
    ap.add_argument("--parallel", type=int, default=4, help="scans running at once (default 4)")
    ap.add_argument("--only", choices=sorted(JOBS), action="append", help="run only this scan (repeatable)")
    args = ap.parse_args()
    names = args.only or DEFAULT
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NE_THREADS=str(args.threads))
    procs, queue = {}, list(names)

    def launch(name):
        out, a, seed_max, extra = JOBS[name]
        log = os.path.join(DATA, out.replace(".jsonl", ".log"))
        e = dict(env, NE_SEED_MAX=seed_max, **extra)
        procs[name] = (subprocess.Popen([sys.executable, "-u", os.path.join("..", "scan_ne.py"), out, *a],
                                        cwd=DATA, env=e, stdout=open(log, "a"), stderr=subprocess.STDOUT), log)
        print(f"started {name}: solver/data/{out} (log {os.path.relpath(log)})", flush=True)

    t0 = time.time()
    while queue or any(p.poll() is None for p, _ in procs.values()):
        while queue and sum(p.poll() is None for p, _ in procs.values()) < args.parallel:
            launch(queue.pop(0))
        time.sleep(30)
        for name, (p, log) in procs.items():
            rec = last_record(log)
            if rec and rec.startswith("{"):
                r = json.loads(rec)
                g = r.get("gamma")
                rec = (f"D={r['D']:.3f} Delta={r['Delta']:.10f} weak={r['weak']:.0e} "
                       f"gamma={'-' if g is None else f'{g:.8f}'} ok={r['ok']}")
            state = "running" if p.poll() is None else f"exit {p.returncode}"
            print(f"[{(time.time() - t0) / 60:5.1f} min] {name} ({state}): {rec or 'starting'}", flush=True)
    print("done. Commit and push solver/data/ (scan_ne*.jsonl, *.log, ne*_D*.npy).")


if __name__ == "__main__":
    main()
