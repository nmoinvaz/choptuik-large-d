#!/usr/bin/env python3
"""Run the nonlinear-elimination D scans on this machine, in parallel.

Two scans that complement the ones running in the cloud (x_K = 0.6, kc = 23), so that every D above 23 gets
three independent discretisations and their spread is the error bar on Delta:

    xK07   matching at x_K = 0.7, kc = 23     (seeds/long7_D*.npy)  -> data/scan_ne7.jsonl
    kc31   matching at x_K = 0.6, kc = 31     (seeds/longB_D*.npy)  -> data/scan_ne6_kc31.jsonl

Each re-solves its stored seeds for D = 23-24 with ne_solver, then continues in D up to 40 from those, writing one JSON line
per point (Delta, residual split, gamma) and the solution vectors (data/ne7_D*.npy, data/neB_kc31_D*.npy).

Usage (repo root, after `pip install .` so the kernel matches the Python side):
    python solver/run_local.py                 # both scans, 2 threads each (4 cores)
    python solver/run_local.py --threads 1     # 1 thread each
    python solver/run_local.py --only xK07     # one scan
When it finishes (or whenever), commit solver/data/ and push.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# name: (output, scan_ne.py arguments, last seed to re-solve before continuing from the new solutions)
JOBS = {
    "xK07": ("scan_ne7.jsonl", ["23", "40", "0.2", "0.4", "0.7", "../seeds/long7_D", "23"], "24.0"),
    "kc31": ("scan_ne6_kc31.jsonl", ["23", "40", "0.2", "0.4", "0.6", "../seeds/longB_D", "31"], "24.0"),
}


def last_record(log):
    try:
        with open(log) as f:
            lines = [l for l in f if l.startswith("{") or "giving up" in l or "Traceback" in l]
    except OSError:
        return None
    return lines[-1].strip() if lines else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=2, help="threads per scan (default 2)")
    ap.add_argument("--only", choices=sorted(JOBS), action="append", help="run only this scan (repeatable)")
    args = ap.parse_args()
    names = args.only or list(JOBS)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NE_THREADS=str(args.threads))
    procs = {}
    for name in names:
        out, a, seed_max = JOBS[name]
        log = os.path.join(DATA, out.replace(".jsonl", ".log"))
        procs[name] = (subprocess.Popen([sys.executable, "-u", os.path.join("..", "scan_ne.py"), out, *a],
                                        cwd=DATA, env=dict(env, NE_SEED_MAX=seed_max), stdout=open(log, "w"), stderr=subprocess.STDOUT), log)
        print(f"started {name}: solver/data/{out} (log {os.path.relpath(log)})", flush=True)
    t0 = time.time()
    while any(p.poll() is None for p, _ in procs.values()):
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
