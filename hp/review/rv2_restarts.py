#!/usr/bin/env python3
"""Review RV2: each run re-converges its start state (and its history states) before continuing, and prints that
Delta (hi part, 16 decimals) in its log. Where the start D is a point of an earlier run, the difference is a free
reproducibility check; where the run changed the layout (M, nodes, kfreeze), it measures the layout's effect.
Resolution: the printed hi part, about 2e-16.

    python3 hp/review/rv2_restarts.py [CLOUD_SCAN_DIR]
"""
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HP = os.path.dirname(HERE)
scan = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/dss/hp_scan"
rows = {round(float(r[0]), 6): (float(r[1]), float(r[2]), r[6]) for r in list(csv.reader(open(os.path.join(HP, "data", "delta_dd.csv"))))[1:]}
pat_s = re.compile(r"\(re-\)converged start D=([0-9.]+) M=(\d+) nodes=(\[[^\]]*\]): Delta ([0-9.]+)")
pat_h = re.compile(r"history point D=([0-9.]+): ok (\w+), Delta ([0-9.]+)")
logs = sorted(glob.glob(os.path.join(scan, "scan_up*.log")), key=lambda s: (len(s), s))
diffs = []
for f in logs:
    for line in open(f):
        for kind, m in (("start", pat_s.match(line)), ("history", pat_h.match(line))):
            if not m:
                continue
            D = round(float(m.group(1)), 6)
            val = float(m.group(4) if kind == "start" else m.group(3))
            if D in rows:
                hi, lo, run = rows[D]
                d = val - (hi + lo)
                diffs.append(abs(d))
                extra = f"M {m.group(2)} nodes {len(eval(m.group(3)))}" if kind == "start" else ""
                print(f"{os.path.basename(f):16s} {kind:7s} D {D:10.6f} {extra:14s} log {val:.16f}  csv ({run}) "
                      f"{hi + lo:.16f}  diff {d:+.1e}")
print(f"{len(diffs)} re-converged points found in the CSV; max |diff| {max(diffs):.1e} (hi printed to 16 decimals)")
