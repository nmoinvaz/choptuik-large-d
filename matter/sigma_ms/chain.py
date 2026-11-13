#!/usr/bin/env python3
"""A chain of run_eta.py runs in eps = 1/eta at one M: each coupling seeded by the secant through the two previous
states (the first two from a given state and from it alone), one process at a time.

    PYTHONPATH=<site> python3 chain.py --M 32 --start <out>/sm_inf_M32.npz --etas 20,14,10,8.5,7,6,5 --out <out> \
        [--threads 4] [--jac dd|mixed] [--extra "..."]

The tag of each run is eta<eta>_M<M>. With --resample_from M0, each coupling is instead seeded by that coupling's
state at M0 (tag eta<eta>_M<M0>), resampled to M. Stops at the first run that does not converge.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--start", default="")
    ap.add_argument("--etas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resample_from", type=int, default=0)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    prev = [a.start] if a.start else []
    etas_prev = [None] if a.start else []
    for eta in a.etas.split(","):
        tag = f"eta{eta}_M{a.M}"
        cmd = [sys.executable, "-u", os.path.join(HERE, "run_eta.py"), "--eta", eta, "--M", str(a.M), "--out", a.out,
               "--tag", tag, "--threads", str(a.threads)] + a.extra.split()
        if a.resample_from:
            cmd += ["--from", os.path.join(a.out, f"sm_eta{eta}_M{a.resample_from}.npz")]
        elif len(prev) >= 2 and etas_prev[-2] not in (None, etas_prev[-1]):
            # the secant through the last two couplings (not through two states of one coupling)
            cmd += ["--from", prev[-1], "--from2", prev[-2]]
        else:
            cmd += ["--from", prev[-1]]
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"chain: {tag} exit {r.returncode}, stopping", flush=True)
            sys.exit(r.returncode)
        prev.append(os.path.join(a.out, f"sm_{tag}.npz"))
        etas_prev.append(eta)


if __name__ == "__main__":
    main()
