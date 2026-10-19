#!/usr/bin/env python3
"""Run a matrix of nonlinear-elimination D scans on this machine, several at a time.

The period is selected by horizon regularity, felt through u_K^s0 (u_K = 1 - x_K, s0 ~ 0.53 (D-2)), so matching
further from the horizon (smaller x_K) selects Delta far more strongly and makes the answer less sensitive to
discretisation error; it needs more horizon-series terms (radius ~0.7 in u). At D = 23 x_K = 0.6, 0.5 and 0.4 agree
to 3e-8, while x_K = 0.7 is off by 1e-5 (CONTEXT.md). Jobs:

    x4_23     x_K = 0.4, Ks = 260, kc = 23, smoothed (strongest selection; slowest) -> data/scan_ne4.jsonl
    x5_31     x_K = 0.5, Ks = 120, kc = 31, smoothed                           -> data/scan_ne5_kc31.jsonl
    sj_06     x_K = 0.6, kc = 23, junk fixed by smoothness                     -> data/scan_ne6_sj.jsonl
    kc31_06   x_K = 0.6, kc = 31                                               -> data/scan_ne6_kc31.jsonl
  (also available: kc31_07, sj_07 at x_K = 0.7, the weakly selecting setup)

The cloud runs x_K = 0.5 (Ks = 120, kc = 23) and x_K = 0.6 kc = 31 with smoothing. Each job re-solves stored seeds
for D = 23-24, then continues in D up to 40 (about 80 points), one JSON line per point (Delta, residual split,
gamma), solution vectors in data/. Expect 1-4 hours per job on one core.

Usage (repo root, after `pip install .`):
    python solver/run_local.py                      # the four jobs above, one thread each, four at a time
    python solver/run_local.py --only x4_23 --threads 2
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
JOBS = {
    "x4_23": ("scan_ne4.jsonl", ["23", "40", "0.2", "0.4", "0.4", "../seeds/neB_D", "23"], "24.0", {"NE_KS": "260", "NE_SMOOTH": "1"}),
    "x5_31": ("scan_ne5_kc31.jsonl", ["23", "40", "0.2", "0.4", "0.5", "../seeds/neB_D", "31"], "24.0", {"NE_KS": "120", "NE_SMOOTH": "1"}),
    "sj_06": ("scan_ne6_sj.jsonl", ["23", "40", "0.2", "0.4", "0.6", "../seeds/neB_D", "23"], "24.0", {"NE_SMOOTH": "1"}),
    "kc31_06": ("scan_ne6_kc31.jsonl", ["23", "40", "0.2", "0.4", "0.6", "../seeds/neB_D", "31"], "24.0", {}),
    "kc31_07": ("scan_ne7_kc31.jsonl", ["23", "40", "0.2", "0.4", "0.7", "../seeds/long7_D", "31"], "24.0", {}),
    "sj_07": ("scan_ne7_sj.jsonl", ["23", "40", "0.2", "0.4", "0.7", "../seeds/long7_D", "23"], "24.0", {"NE_SMOOTH": "1"}),
}
DEFAULT = ["x4_23", "x5_31", "sj_06", "kc31_06"]


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
