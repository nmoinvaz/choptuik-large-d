#!/usr/bin/env python3
"""The quad-double values of Delta at large D into hp/data/delta_qd.csv: the full system (kfreeze 13) converged in
quad-double from the double-double states (notes/QUAD_DOUBLE.md 12.1 and 12.4), one row per D with Delta in four
parts (Delta = q0 + q1 + q2 + q3, each an f64; the first two are the double-double value) and, where a second start
with its own Taylor plan converged, the difference between the two solutions as the error estimate (a spread
between two plans and starts, not a bound).

    cd hp && python3 export_qd.py ../../choptuik-large-d-raw      (the raw-data clone; folders T84-mac and T81)
"""
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
raw = sys.argv[1]
sols = {}        # D -> list of (Delta parts, source)
for f in sorted(glob.glob(os.path.join(raw, "T84-mac", "fullleg_pin*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r["conv"]:
            sols.setdefault(round(r["D"], 4), []).append((r["Delta_q"], f"T84-mac/{os.path.basename(f)}:{r['state']}"))
for f in sorted(glob.glob(os.path.join(raw, "T81", "conv_pin*_D7*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r.get("ok") and r.get("F", 1) < 1e-50:
            sols.setdefault(round(r["D"], 4), []).append((r["Delta_q"], f"T81/{os.path.basename(f)}"))
rows = []
for D in sorted(sols):
    # distinct solutions only (a re-convergence of a saved solution repeats its record)
    distinct = []
    for parts, src in sols[D]:
        if not any(p == parts for p, _ in distinct):
            distinct.append((parts, src))
    sols[D] = distinct
    parts, src = sols[D][0]
    err = ""
    if len(sols[D]) > 1:
        q = sols[D][1][0]
        err = f"{abs(sum(parts) - sum(q)) + abs(parts[2] - q[2]) + abs(parts[3] - q[3]):.1e}"
    rows.append(dict(D=D, Delta_q0=repr(parts[0]), Delta_q1=repr(parts[1]), Delta_q2=repr(parts[2]), Delta_q3=repr(parts[3]),
                     n_solutions=len(sols[D]), spread=err, source=src))
out = os.path.join(HERE, "data", "delta_qd.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {out}")
for r in rows:
    print(f"  D {r['D']:8.4f} Delta {r['Delta_q0']} {float(r['Delta_q1']):+.6e} {float(r['Delta_q2']):+.3e}  solutions {r['n_solutions']} spread {r['spread'] or '-'}")
