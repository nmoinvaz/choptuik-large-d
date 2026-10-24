#!/usr/bin/env python3
"""Collect the double-double points from the continuation runs into hp/data/delta_dd.csv.

    python3 export_dd.py [scan_dir]        (default /home/claude/dss/hp_scan, files scan_up*.jsonl)

Runs from the local machine, pushed to hp/data/local/scan_up*.jsonl, are read after the cloud runs (run name prefixed
"local-"); where both have a point at the same D the local one is kept. Below D = 8.25 the CSV also holds the low-D
runs' best value per D (export_lowD.py; run name lowD-M<M>, or lowD-M<M>-unresolved), which the fits (D >= 10) do not
use.

Columns: D, Delta as hi + lo (double-double: Delta_lo is the low word, digits 17-32, not an error), the final
residual |F|, the largest |dDelta/dF| (how far a residual error moves Delta), the linearised Delta correction
-dDelta/dF . F where the run recorded it, and the run name; then
  Delta_err, err_kind   an estimate of Delta's error, and what it is: "dM" (low-D rows: |Delta(M) - Delta(next lower M)|
                        at this D, an estimator that overestimates where the change falls fast with M) or "check" (the
                        continuation rows: the size of the resolution and harmonic-freezing checks near this D, README
                        "Accuracy", 1e-16 below D = 35 and 2e-15 from D = 35 on, not computed at this D)
  published_ref, published_diff, published_match
                        a published value at this D (data/published_delta.csv: reference key, Delta - published), and
                        "yes" when the difference is within the uncertainty the publication states, "no" otherwise;
                        empty where no published value is known at this D
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
# below D = 8.25: the best value per D of the low-D runs (export_lowD.py; one row per D at its largest clean M, run name
# lowD-M<M>, with "-unresolved" when the two largest M differ by more than 1e-8); the per-M detail is in
# data/delta_dd_lowD.csv. The fits use D >= 10 and do not see these rows.
sys.path.insert(0, here)
import export_lowD  # noqa: E402
for r in export_lowD.build():
    if r["best"] and round(r["D"], 6) not in rows:
        rec = dict(D=r["D"], Delta=r["Delta_hi"], Delta_lo=r["Delta_lo"], F=r["F"], sens=[r["sens_max"] or float("nan")],
                   dDelta_est=r["dDelta_est"], err=r["change_from_lower_M"])
        rows[round(r["D"], 6)] = (rec, f"lowD-M{r['M']}" + ("" if r["resolved"] else "-unresolved"))
# published values at these D (reference, value, the stated uncertainty)
pub = {}
for line in open(os.path.join(here, "data", "published_delta.csv")):
    if line.startswith("#") or line.startswith("D,"):
        continue
    d, v, u, ref = line.strip().split(",")
    pub.setdefault(round(float(d), 6), []).append((ref, float(v), float(u)))
CHECK_BELOW_35, CHECK_FROM_35 = 1e-16, 2e-15   # README "Accuracy": tau resolution (D = 7-17, 40) and freezing (40-60)
out = os.path.join(here, "data", "delta_dd.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    fh.write("D,Delta_hi,Delta_lo,F,max_dDelta_dF,dDelta_est,run,Delta_err,err_kind,published_ref,published_diff,"
             "published_match\n")
    for d in sorted(rows):
        r, run = rows[d]
        est = r.get("dDelta_est")
        if run.startswith("lowD-"):
            err, kind = ("", "") if r.get("err") is None else (format(abs(r["err"]), ".1e"), "dM")
        else:
            err, kind = format(CHECK_BELOW_35 if r["D"] < 35 else CHECK_FROM_35, ".0e"), "check"
        # the most precise published value at this D
        ref = diff = match = ""
        if d in pub:
            ref, v, u = min(pub[d], key=lambda t: t[2])
            dv = (r["Delta"] - v) + r["Delta_lo"]
            diff, match = format(dv, ".2e"), ("yes" if abs(dv) <= u else "no")
        fh.write(f"{r['D']!r},{r['Delta']!r},{r['Delta_lo']!r},{r['F']:.3e},{max(r['sens']):.3e},"
                 f"{'' if est is None else format(est, '.2e')},{run},{err},{kind},{ref},{diff},{match}\n")
print(f"{len(rows)} points, D = {min(rows):.4f} .. {max(rows):.4f} -> {out}")
