#!/usr/bin/env python3
"""#1098: every value of Delta(D) used by the singularity analysis (hp/sing_types.py) in one JSON file, read exactly.

    cd hp && python3 sing_data.py LADDER_FOLDER... --out=FILE [--csv=data/delta_all.csv]

Sources and sigma (each an estimate, not a bound):
  ladder     full form-S records (ladder_fit.full_points, sigma err_sum), D 250-8000
  qd         delta_all.csv kind qd (D 67.7-73, quad-double full system; err the kernel gap 2.4e-23, which omits M)
  dd         delta_all.csv kind dd, err_kind check (D 8.25-67.8; err the size of the resolution checks near D)
  chain_fit  delta_all.csv kind chain_fit, err_kind fits d10 / fits d6 and their interp rows (D 74-245.6)
Left out: dd rows with err_kind dM (D < 8.25, M-limited), pinned rows (single-pin dd, gaps to 1e-14), chain_fit weak
pivot and extrapolated rows (err 1e-10 to 1.5), the series rows (fits, not solves) and the non-ladder direct rows.
Values are written as 40-digit strings of the exact sums of their parts.
"""
import csv
import json
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
mp.mp.dps = 60
args = dict(a.lstrip("-").split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
folders = [a for a in sys.argv[1:] if not a.startswith("--")]
sys.argv = [sys.argv[0], "sing"]
import ladder_fit as LF  # noqa: E402

mp.mp.dps = 60
pts = []
for D, v in LF.full_points(folders).items():
    pts.append(dict(D=v[4], kind="ladder", Delta=mp.nstr(v[0], 40), sigma=v[1], src=v[2]))
KEEP = {("qd", "kernel"), ("dd", "check"), ("chain_fit", "fits d10"), ("chain_fit", "fits d6"),
        ("chain_fit", "fits d10 interp"), ("chain_fit", "fits d6 interp")}
for r in csv.DictReader(open(args.get("csv", os.path.join(HERE, "data", "delta_all.csv")))):
    if (r["kind"], r["err_kind"]) not in KEEP:
        continue
    v = mp.fsum(mp.mpf(r["Delta_q%d" % i]) for i in range(4) if r["Delta_q%d" % i])
    pts.append(dict(D=r["D"], kind=r["kind"], Delta=mp.nstr(v, 40), sigma=float(r["err"]), src=r["source"]))
pts.sort(key=lambda p: float(p["D"]))
json.dump(dict(folders=folders, points=pts), open(args["out"], "w"), indent=0)
for k in ("ladder", "qd", "dd", "chain_fit"):
    s = [p for p in pts if p["kind"] == k]
    print("%-9s %4d points, D %s to %s, sigma %.1e to %.1e" % (k, len(s), s[0]["D"], s[-1]["D"],
                                                              min(p["sigma"] for p in s), max(p["sigma"] for p in s)))
