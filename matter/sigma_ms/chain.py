#!/usr/bin/env python3
"""A chain of run_eta.py runs in eps = 1/eta at one M: each coupling seeded by the secant through the two previous
states (the first two from a given state and from it alone), one process at a time. With --Ds, the chain is in D at
the fixed coupling --eta instead (#652), the secant through the two previous D, tags eta<eta>_D<D>_M<M>.

    PYTHONPATH=<site> python3 chain.py --M 32 --start <out>/sm_inf_M32.npz --etas 20,14,10,8.5,7,6,5 --out <out> \
        [--threads 4] [--jac dd|mixed] [--extra "..."]
    PYTHONPATH=<site> python3 chain.py --M 32 --start <out>/sm_eta1_M32.npz --eta 1 --Ds 4.5,5,5.5 --out <out>
        [--start2 <the state before --start>]
    PYTHONPATH=<site> python3 chain.py --M 32 --start <D 7.0625 state> --start2 <D 7 state> --eta 1 --D_from 7.0625 \
        --D_to 10.475 [--dD 0.0625 --dD_min 0.0078125 --dD_max 0.125] --out <out>      (adaptive steps in D)
        --D_to below --D_from runs the chain downward in D. --wrap "CMD ..." puts a command before every run, for
        example a load gate.

The tag of each run is eta<eta>_M<M>. With --resample_from M0, each coupling is instead seeded by that coupling's
state at M0 (tag eta<eta>_M<M0>), resampled to M. Stops at the first run that does not converge.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def adaptive(a, prev, Ds_prev):
    """the chain in D from --D_from to --D_to: a failed run halves the step (stop below --dD_min), three runs in a row
    double it (up to --dD_max) when D is a multiple of the doubled step, so the integers are reached"""
    D, dD, ok_run = a.D_from, a.dD, 0
    sgn = 1 if a.D_to >= a.D_from else -1                   # the chain runs downward when --D_to is below --D_from
    while sgn * (a.D_to - D) > 1e-12:
        Dn = D + sgn * dD
        if sgn * (Dn - a.D_to) > 0:
            Dn = a.D_to
        Ds = "%.10g" % Dn
        tag = f"eta{a.eta}_D{Ds}_M{a.M}"
        cmd = a.wrap.split() + [sys.executable, "-u", os.path.join(HERE, "run_eta.py"), "--eta", a.eta, "--D", Ds,
                                "--M", str(a.M), "--out", a.out, "--tag", tag, "--threads", str(a.threads)] \
            + a.extra.split()
        if len(prev) >= 2 and Ds_prev[-2] not in (None, Ds_prev[-1]):
            cmd += ["--from", prev[-1], "--from2", prev[-2]]
        else:
            cmd += ["--from", prev[-1]]
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            dD, ok_run = dD / 2, 0
            print(f"chain: {tag} exit {r.returncode}, step halved to {dD:g}", flush=True)
            if dD < a.dD_min * (1 - 1e-9):
                print("chain: below the smallest step, stopping", flush=True)
                sys.exit(r.returncode)
            continue
        prev.append(os.path.join(a.out, f"sm_{tag}.npz"))
        Ds_prev.append(Ds)
        D, ok_run = Dn, ok_run + 1
        if ok_run >= 3 and 2 * dD <= a.dD_max * (1 + 1e-9) and abs(D / (2 * dD) - round(D / (2 * dD))) < 1e-9:
            dD, ok_run = 2 * dD, 0
            print(f"chain: step doubled to {dD:g}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--start", default="")
    ap.add_argument("--start2", default="", help="a chain in D: the state before --start, so the first step is a secant")
    ap.add_argument("--etas", default="")
    ap.add_argument("--eta", default="", help="the fixed coupling of a chain in D (--Ds)")
    ap.add_argument("--Ds", default="", help="a chain in D at --eta: comma-separated D values")
    ap.add_argument("--D_from", type=float, default=0.0, help="adaptive chain in D: the D of --start")
    ap.add_argument("--D_to", type=float, default=0.0, help="adaptive chain in D: the last D")
    ap.add_argument("--dD", type=float, default=0.0625)
    ap.add_argument("--dD_min", type=float, default=0.0078125)
    ap.add_argument("--dD_max", type=float, default=0.125)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resample_from", type=int, default=0)
    ap.add_argument("--extra", default="")
    ap.add_argument("--wrap", default="", help="adaptive chain in D: a command put before every run (a load gate)")
    a = ap.parse_args()
    prev = [a.start] if a.start else []
    etas_prev = [None] if a.start else []
    if a.Ds or a.D_to:
        Ds_prev = [None] if a.start else []
        if a.start2:                                        # the states' metas carry their D (run_eta.py)
            prev, Ds_prev = [a.start2, a.start], ["start2", "start"]
        if a.D_to:
            return adaptive(a, prev, Ds_prev)
        for D in a.Ds.split(","):
            tag = f"eta{a.eta}_D{D}_M{a.M}"
            cmd = [sys.executable, "-u", os.path.join(HERE, "run_eta.py"), "--eta", a.eta, "--D", D, "--M", str(a.M),
                   "--out", a.out, "--tag", tag, "--threads", str(a.threads)] + a.extra.split()
            if len(prev) >= 2 and Ds_prev[-2] not in (None, Ds_prev[-1]):
                cmd += ["--from", prev[-1], "--from2", prev[-2]]
            else:
                cmd += ["--from", prev[-1]]
            print(" ".join(cmd), flush=True)
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"chain: {tag} exit {r.returncode}, stopping", flush=True)
                sys.exit(r.returncode)
            prev.append(os.path.join(a.out, f"sm_{tag}.npz"))
            Ds_prev.append(D)
        return
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
