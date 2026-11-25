#!/usr/bin/env python3
"""The highest D reached at large D against time, past the last point the coordination log gives: the second part of
the paper's figure of the frontier (paper/ai/main.tex), from the history of the raw-data repository.

    python3 paper/ai/frontier.py RAWDIR --ref b6c344480 [-o paper/ai/frontier.csv]
    python3 paper/ai/frontier.py --selftest [--mutate=pred]

RAWDIR is a clone of the raw-data repository (AGENTS.md), and --ref the commit of it to read (default origin/main).
A chain saves a state anm_pin<k>_D<D>.npz only after its corrector converged (hp/anm_run.py), so the highest D reached
at a time is the largest D among the states pushed by then. A state's time is the author time of the commit that added
it, which is the runner's push: late by up to the interval between two pushes, about an hour.

Writes one row per push that raised the highest D: day (days from 2026-10-19 00:00 UTC), D, and the folder of the
state. The first row is START, the figure's last point from the log, so that the two parts join.

--selftest runs the parser on a planted history with a known answer. With --mutate=pred it counts the predictions
(pred_pin<k>_D<D>.npz, saved before the corrector runs), which must fail. Exit 1 on a failed check.
"""
import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = datetime(2026, 10, 19, tzinfo=timezone.utc)
START = (18.78, 206.77)                       # the log's last point: win2's entry of 2026-11-06 18:47 UTC (SOURCES.md)
STATE = re.compile(r"(?:^|/)anm_pin\d+_D(\d+(?:\.\d+)?)\.npz$")
PRED = re.compile(r"(?:^|/)(?:anm|pred)_pin\d+_D(\d+(?:\.\d+)?)\.npz$")


def frontier(lines, start=START, pattern=STATE):
    """Rows (day, D, folder) of the pushes that raised the highest D, from `git log --reverse --diff-filter=A
    --name-only --format=@%aI` output."""
    rows, best, day = [(start[0], start[1], "the log")], start[1], None
    top = None                                # the push's highest new state
    for line in list(lines) + ["@end"]:
        line = line.rstrip("\n")
        if line.startswith("@"):
            if top is not None and top[0] > best and day > start[0]:
                best = top[0]
                rows.append((round(day, 3), top[0], top[1]))
            top = None
            if line != "@end":
                day = (datetime.fromisoformat(line[1:]) - T0).total_seconds() / 86400.0
            continue
        m = pattern.search(line)
        if m and (top is None or float(m.group(1)) > top[0]):
            top = (float(m.group(1)), line.split("/")[0])
    return rows


def selftest(mutate):
    planted = ["@2026-11-06T12:00:00+00:00", "A/chain/anm_pin3_D205.0000.npz",        # before START: not counted
               "@2026-11-07T00:00:00+00:00", "A/chain/anm_pin3_D210.0000.npz", "A/chain/anm_pin3_D212.5000.npz",
               "A/chain/pred_pin3_D213.0000.npz", "A/chain/jet_pin3_D212.5000.npz",
               "@2026-11-08T00:00:00+00:00", "B/rerun/anm_pin3_D211.0000.npz",        # below the highest: no row
               "@2026-11-09T12:00:00+00:00", "C/anm_pin3_D220.2500.npz", "C/pred_pin3_D221.0000.npz"]
    want = [(START[0], START[1], "the log"), (19.0, 212.5, "A"), (21.5, 220.25, "C")]
    got = frontier(planted, pattern=PRED if mutate == "pred" else STATE)
    ok = got == want
    print(f"selftest: planted history, {len(got)} rows, {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  got  {got}\n  want {want}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rawdir", nargs="?")
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("-o", default=os.path.join(HERE, "frontier.csv"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", default="")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.mutate) else 1)
    if not a.rawdir or not os.path.isdir(os.path.join(a.rawdir, ".git")):
        sys.exit("give RAWDIR, a clone of the raw-data repository (see the docstring)")
    log = subprocess.run(["git", "-C", a.rawdir, "log", "--reverse", "--diff-filter=A", "--name-only", "--format=@%aI",
                          a.ref], capture_output=True, text=True, check=True).stdout
    rows = frontier(log.splitlines())
    with open(a.o, "w") as f:
        f.write("day,D,folder\n")
        for day, D, folder in rows:
            f.write(f"{day},{D},{folder}\n")
    print(f"{len(rows)} rows to {os.path.relpath(a.o)}; highest D {rows[-1][1]} at day {rows[-1][0]} ({rows[-1][2]})")


if __name__ == "__main__":
    main()
