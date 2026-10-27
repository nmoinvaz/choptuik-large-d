#!/usr/bin/env python3
"""T79's double-double points for 4 <= D <= 8.25 (folder T79-win of the raw-data repository) in the low-D exporter's
formats: data/lowD_dd_t79.jsonl (scan_hp's records of Run 1 at M 48, and the M 64 re-solves of Run 2 as records with
Delta from the saved state and |F| from the log) and data/lowD_gamma_t79.jsonl (Run 3's records). export_lowD.py then
reads them with the other low-D runs.

    cd hp && python3 t79_rows.py ../../choptuik-large-d-raw/T79-win
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1]
rows = []
for line in open(os.path.join(src, "down.jsonl")):
    if line.strip():
        r = json.loads(line)
        r["source_folder"] = "T79-win"
        rows.append(r)
for f in sorted(glob.glob(os.path.join(src, "m64_D*.npz"))):
    log = f[:-4] + ".log"
    m = re.search(r"\|F\| ([0-9.e+-]+) after (\d+) steps; Delta ([0-9.]+) here", open(log).read())
    if not m:
        continue
    d = np.load(f)
    meta = json.loads(str(d["meta"]))
    if float(m.group(1)) > 1e-25:
        print(f"{os.path.basename(f)}: |F| {m.group(1)}, not converged, left out")
        continue
    rows.append(dict(D=meta["D"], M=meta["M"], Delta=float(d["zh"][-1]), Delta_lo=float(d["zl"][-1]), F=float(m.group(1)),
                     its=int(m.group(2)), nodes=meta["nodes"], source_folder="T79-win", relayout=True))
with open(os.path.join(HERE, "data", "lowD_dd_t79.jsonl"), "w") as fh:
    for r in sorted(rows, key=lambda r: (r["D"], r["M"])):
        fh.write(json.dumps(r) + "\n")
gam = [json.loads(l) for l in open(os.path.join(src, "gamma.jsonl")) if l.strip()]
with open(os.path.join(HERE, "data", "lowD_gamma_t79.jsonl"), "w") as fh:
    for r in sorted(gam, key=lambda r: r["D"]):
        fh.write(json.dumps(dict(r, source_folder="T79-win")) + "\n")
print(f"{len(rows)} Delta rows ({sum(1 for r in rows if r.get('relayout'))} at M 64), {len(gam)} gamma rows")
