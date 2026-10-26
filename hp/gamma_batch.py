#!/usr/bin/env python3
"""Batch driver for gamma_dd.py: one process per state, N processes in parallel (each single-threaded).

    python3 gamma_batch.py OUT.jsonl [--procs=N] [--m48=D1,D2,...] [--kf=none|13|auto] [STATE.npz ...]

Without state arguments it takes every state in hp/data/states/ plus hp/data/seeds/*.npz and hp/data/seeds/local/*.npz.
--m48 lists the D values (nearest state) for which the M = 48 self-convergence test is also run (slower: one Newton
solve at M = 48 first). Results are appended to OUT.jsonl as gamma_dd.py writes them; a state already present in OUT
is skipped, so the batch can be restarted. Every state is first re-converged in the rebuilt layout (gamma_dd.py
--polish=4: one Newton step when it is already at the double-double floor; it brings the up6 states at D = 27-35 from
|F| ~ 1e-12 down to 1e-25). --kf is passed through (default: the kfreeze-13 cut from D = 35 on).

Needs the hpkernel wheel built from this tree (hp/build.sh), which provides Solver.mode_det.
Measured on the shared 2-core cloud machine (nice 15, 1 thread per process): 40-240 s per state (D = 10-15 at M = 48:
~100 s; D = 16-61 at M = 32: 30-130 s, more near D = 61 where the det is noisy), plus 2-5 min for each --m48 state.
"""
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from raw_out import require_raw

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    out = os.path.abspath(args[0])
    require_raw(out)                    # --allow_unraw writes anyway
    procs = int(opts.get("procs", 4))
    states = args[1:] or (sorted(glob.glob(os.path.join(HERE, "data", "states", "*.npz")))
                          + sorted(glob.glob(os.path.join(HERE, "data", "seeds", "*.npz")))
                          + sorted(glob.glob(os.path.join(HERE, "data", "seeds", "local", "*.npz"))))
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                done.add(json.loads(l)["state"])
            except Exception:
                pass
    m48 = [float(v) for v in opts.get("m48", "").split(",") if v]
    Ds = {}
    for f in states:
        try:
            Ds[f] = float(os.path.basename(f).split("_D")[1][:-4])
        except Exception:
            Ds[f] = None
    pick48 = {min(Ds, key=lambda f: abs((Ds[f] or 1e9) - d)) for d in m48}
    todo = [f for f in states if os.path.basename(f) not in done]
    print(f"{len(todo)} states to do ({len(states) - len(todo)} already in {out}), {procs} processes", flush=True)

    def run(f):
        cmd = ([sys.executable, os.path.join(HERE, "gamma_dd.py"), out, f, "--threads=1", "--polish=4"]
               + (["--M48"] if f in pick48 else []) + ([f"--kf={opts['kf']}"] if "kf" in opts else []))
        r = subprocess.run(cmd, capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or [""])[-1]
        print(f"{os.path.basename(f)}: {'ok' if r.returncode == 0 else 'FAILED'} {last[:160]}", flush=True)
        if r.returncode:
            print(r.stderr[-2000:], flush=True)

    with ThreadPoolExecutor(procs) as ex:
        list(ex.map(run, todo))


if __name__ == "__main__":
    main()
