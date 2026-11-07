#!/usr/bin/env python3
"""A planted test of the junk-frozen full-system step (full_release.py --octo 1 --junk 4) with no re-solve
(notes/QUAD_DOUBLE.md 13.29). T140's chains hold the harmonics on the fits of degree 6 and 10, chain_fit (T138) on the
fit of degree 8, and each state is a converged pinned state at its D. The full system holds nothing, so it is the same
for all three, and the step from a degree-k state must differ from chain_fit's by the difference of their pinned
Deltas: r = dDelta(k) - dDelta(8) = -(Delta(k) - Delta(8)) = -b. b is the planted bias, exact from the states' four
parts. Pass, set before the run: |r / (-b) - 1| <= 0.02 at every pair. chain_fit's steps come from
data/release_steps.csv (release_steps.py), the degree-k steps from the logs of T147-win2. --mutate=pair (the control,
which must fail) scales the first pair's r by 1.03.

    cd hp && python3 release_pairs.py RAWDIR [--tasks=T147-win2] [--mutate=pair]

Writes data/release_pairs.csv and prints the table. Exits 1 when a pair fails or no pair is found.
"""
import csv
import glob
import os
import sys

import mpmath as mp

import export_delta_all as E
from release_steps import read

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "release_pairs.csv")
TOL = 0.02


def delta(state):
    """the state's Delta from its four parts"""
    with mp.workdps(40):
        return sum((mp.mpf(p) for p in E.load_state(state)[1] if p != ""), mp.mpf(0))


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    opts = dict((a[2:] + "=").split("=")[:2] for a in argv if a.startswith("--"))
    if len(args) != 1:
        sys.exit(__doc__)
    E.RAW = os.path.abspath(args[0])
    fit = {round(float(r["D"]), 6): r for r in csv.DictReader(open(os.path.join(HERE, "data", "release_steps.csv")))}
    rows, ok = [], True
    for t in opts.get("tasks", "T147-win2").split(","):
        for p in sorted(glob.glob(os.path.join(E.RAW, t, "release_*_D*.log"))):
            r = read(p)
            rel = os.path.relpath(p, E.RAW)
            if r is None:
                print(f"  incomplete: {rel}")
                continue
            f = fit.get(round(r["D"], 6))
            if f is None:
                print(f"  no chain_fit step at D {r['D']}: {rel}")
                continue
            b = float(delta(r["state"]) - delta(f["state"]))
            if b == 0:
                print(f"  the same Delta as chain_fit's state, no bias: {rel}")
                continue
            resp = r["dDelta"] - float(f["dDelta"])
            rows.append(dict(D=r["D"], state=r["state"], bias=b, step=r["dDelta"], step_fit=float(f["dDelta"]),
                             response=resp, log=rel, log_fit=f["log"]))
    if not rows:
        print("FAIL: no pair")
        return 1
    if opts.get("mutate") == "pair":
        rows[0]["response"] *= 1.03
        print(f"mutated: the response at D {rows[0]['D']} ({rows[0]['log']}) scaled by 1.03")
    rows.sort(key=lambda x: (x["D"], x["state"]))
    print(f"{'D':>7} {'state':44s} {'bias':>11} {'response':>11} {'ratio':>10}")
    for x in rows:
        x["ratio"] = x["response"] / -x["bias"]
        x["pass"] = abs(x["ratio"] - 1) <= TOL
        ok &= x["pass"]
        print(f"{x['D']:7.2f} {x['state']:44s} {x['bias']:+11.4e} {x['response']:+11.4e} {x['ratio']:10.6f}"
              f"  {'pass' if x['pass'] else 'FAIL'}")
    if not opts.get("mutate"):
        with open(OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} pairs -> {OUT}")
    print(f"{'pass' if ok else 'FAIL'}: {len(rows)} pairs")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
