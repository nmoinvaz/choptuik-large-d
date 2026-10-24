#!/usr/bin/env python3
"""Collect every double-double point below D = 8.25 (the low-D runs, cloud and local) into hp/data/delta_dd_lowD.csv, one
row per (D, M). The best value at each D (the largest clean M) also goes into delta_dd.csv with the continuation runs'
points (hp/export_dd.py, run name lowD-M<M>, or lowD-M<M>-unresolved).

Input: data/lowD_dd.jsonl, data/lowD_dd_local*.jsonl, data/lowD_highM/*points*.jsonl if present (records of
hp/lowD_dd.py and the larger-M runs), and data/lowD_gamma*.jsonl for gamma. One row per (D, M); where several runs
have the same (D, M), the one with the smaller |F| is kept. Columns:
  D, M, Delta_hi, Delta_lo   Delta as hi + lo (double-double), at tau resolution M
  F                          the final residual |F|
  tail_c, tail_h             the centre and horizon series tails at that solution
  change_from_lower_M        Delta(M) - Delta(next lower M at the same D), empty if none
  best                       1 for the largest M at that D (the value to use)
  resolved                   1 if the best value's change from the next lower M is at most 1e-8, or if it has M = 80
                             alone at D >= 3.7 (where M 80 -> 96 is 1.2e-9); otherwise 0 (converged, not yet resolved)
  gamma                      gamma at that (D, M) if computed (exact growth-mode matrix, hp/lowD_gamma.py)
  status                     "clean", or "polluted: ..." for converged solutions with content along the near-null
                             directions (POLLUTED below); those are listed but never used as a value
  source                     the file the record came from

    cd hp && python3 export_lowD.py
"""
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESOLVED = 1e-8
# converged solutions that picked up content along the near-null directions (centre-series tail 1e-9 to 1e-10 where
# clean neighbours have 1e-50; an M 96 solve from them does not converge); the local machine's report in comms.txt,
# 2026-10-23 05:51 PDT. Kept as rows with status "polluted", never used as a value.
POLLUTED = {(3.4, 128): "centre tail 1.2e-9", (3.3875, 128): "centre tail 2.8e-10; stepped from the polluted 3.40"}


def records(pattern):
    for f in sorted(glob.glob(os.path.join(HERE, "data", pattern))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                r["_src"] = os.path.relpath(f, HERE)
                yield r


def build():
    """the rows of delta_dd_lowD.csv (dicts), sorted by D and M"""
    best = {}
    for r in list(records("lowD_dd*.jsonl")) + list(records("lowD_highM/*points*.jsonl")):
        if r.get("converged") is False or r.get("Delta") is None or r.get("D", 99) >= 8.25:
            continue
        if r.get("F") is not None and r["F"] > 1e-25:
            continue
        key = (round(r["D"], 6), int(r["M"]))
        d = r["Delta"]
        hi, lo = (d[0], d[1]) if isinstance(d, (list, tuple)) else (d, r.get("Delta_lo", 0.0))
        sens = r.get("sens_max", max(r["sens"]) if r.get("sens") else None)
        row = dict(D=key[0], M=key[1], Delta_hi=hi, Delta_lo=lo, F=r.get("F"), tail_c=r.get("tail_c"),
                   tail_h=r.get("tail_h"), sens_max=sens, dDelta_est=r.get("dDelta_est"), source=r["_src"])
        row["status"] = "polluted: " + POLLUTED[key] if key in POLLUTED else "clean"
        if key not in best or (row["F"] or 1) < (best[key]["F"] or 1):
            best[key] = row
    gam = {(round(r["D"], 6), int(r["M"])): r["gamma"] for r in records("lowD_gamma*.jsonl") if r.get("gamma") is not None}
    rows = []
    for D in sorted({k[0] for k in best}):
        Ms = sorted(m for d, m in best if d == D and best[(d, m)]["status"] == "clean")
        for key in sorted(k for k in best if k[0] == D and best[k]["status"] != "clean"):
            rows.append(dict(best[key], change_from_lower_M=None, best=0, resolved=0, gamma=gam.get(key)))
        for i, M in enumerate(Ms):
            row = dict(best[(D, M)])
            ch = None
            if i > 0:
                # in double-double: the hi parts first (often equal at the largest M), then the lo parts
                lower = best[(D, Ms[i - 1])]
                ch = (row["Delta_hi"] - lower["Delta_hi"]) + (row["Delta_lo"] - lower["Delta_lo"])
            row["change_from_lower_M"] = ch
            row["best"] = int(M == Ms[-1])
            row["resolved"] = int(M == Ms[-1] and ((ch is not None and abs(ch) <= RESOLVED)
                                                   or (ch is None and M == 80 and D >= 3.7)))
            row["gamma"] = gam.get((D, M))
            rows.append(row)
    rows.sort(key=lambda r: (r["D"], r["M"]))
    return rows


def main():
    rows = build()
    out = os.path.join(HERE, "data", "delta_dd_lowD.csv")
    cols = ["D", "M", "Delta_hi", "Delta_lo", "F", "tail_c", "tail_h", "change_from_lower_M", "best", "resolved", "gamma",
            "status", "source"]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r[c] is None else (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in cols])
    nb = sum(r["best"] for r in rows)
    print(f"wrote {out}: {len(rows)} rows, {nb} values of D from {rows[0]['D']} to {rows[-1]['D']}, "
          f"{sum(r['resolved'] for r in rows)} resolved")


if __name__ == "__main__":
    main()
