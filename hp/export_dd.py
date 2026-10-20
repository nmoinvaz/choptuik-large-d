#!/usr/bin/env python3
"""Collect the double-double points from the continuation runs into hp/data/delta_dd.csv.

    python3 export_dd.py [scan_dir]        (default /home/claude/dss/hp_scan, files scan_up*.jsonl)

Runs from the local machine, pushed to hp/data/local/scan_up*.jsonl, are read after the cloud runs (run name prefixed
"local-"); where both have a point at the same D the local one is kept.

Columns: D, Delta as hi + lo (double-double), the final residual |F|, the largest |dDelta/dF| (how far a residual
error moves Delta), the linearised Delta correction -dDelta/dF . F where the run recorded it, and the run name.
A D that appears in several runs (to 1e-6) keeps the last run's value, as the fits do.
"""
import glob
import json
import os
import sys

scan = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/dss/hp_scan"
rows = {}
here = os.path.dirname(os.path.abspath(__file__))
files = [(f, "") for f in sorted(glob.glob(os.path.join(scan, "scan_up*.jsonl")))]
files += [(f, "local-") for f in sorted(glob.glob(os.path.join(here, "data", "local", "scan_up*.jsonl")))]
for f, prefix in files:
    run = prefix + os.path.basename(f)[5:-6]
    for line in open(f):
        r = json.loads(line)
        rows[round(r["D"], 6)] = (r, run)
out = os.path.join(here, "data", "delta_dd.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    fh.write("D,Delta_hi,Delta_lo,F,max_dDelta_dF,dDelta_est,run\n")
    for d in sorted(rows):
        r, run = rows[d]
        est = r.get("dDelta_est")
        fh.write(f"{r['D']!r},{r['Delta']!r},{r['Delta_lo']!r},{r['F']:.3e},{max(r['sens']):.3e},"
                 f"{'' if est is None else format(est, '.2e')},{run}\n")
print(f"{len(rows)} points, D = {min(rows):.4f} .. {max(rows):.4f} -> {out}")
