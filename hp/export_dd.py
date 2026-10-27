#!/usr/bin/env python3
"""Collect the double-double points from the continuation runs into hp/data/delta_dd.csv.

    python3 export_dd.py [scan_dir] [--allow_drop]

scan_dir: the cloud runs' scan_up*.jsonl. Those records stayed on the cloud disk (/home/claude/dss/hp_scan) and are
missing since 2026-10-26. Without scan_dir their rows come from data/large_d/cloud_scan_rows.csv, the D, Delta,
|F|, largest |dDelta/dF| and dDelta_est that delta_dd.csv held for them, which reproduce those rows exactly. The
script exits non-zero without writing when the new CSV would lose a D that the current one has (--allow_drop
writes anyway).

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
  dDelta_exact          T48-up70 rows only: the linearised Delta correction as dDelta_est, but from the residual with
                        the horizon block in quad-double, which removes the horizon rows' rounding that |dDelta/dF|
                        (up to 1e15 there) amplifies; the dot product in double-double (qdh_delta_check.py version 3 on
                        branch wip/qd-horizon, run on the Mac as T60b: data/large_d/qdh_delta_check3_mac.jsonl, every
                        |dDelta_exact| <= Delta_err, its control passing); empty elsewhere
A D that appears in several runs (to 1e-6) keeps the last run's value, as the fits do.

Above the scans' last point (61.026449) come T48's points: the cloud pilot (data/large_d/pilot_cloud/pilotB.jsonl,
run T48-pilotB, ks 100) and the Mac's continuation (data/large_d/up70.jsonl, run T48-up70, ks 100 to 67.43 and ks 130
after). Left out of up70.jsonl, by rules written down here: the diagnostic runs near 67.75 (ks 200 with the junk hold;
the floor test at four times the tolerance, tol > 2.5e-21; the horizon_alpha test, the file's last five records, which
carry no field that marks them); points after a horizon-tail jump (tail_h more than 100 times the median of the 20
neighbouring records of the same ks: 67.4167, 67.4267, 67.7625, 67.7708, 67.7762, 67.7899); and |dDelta_est| > 1e-15.
The fits quoted in the README use D <= 61.03 and do not see these rows.
"""
import csv
import glob
import json
import os
import sys

args = [v for v in sys.argv[1:] if v != "--allow_drop"]
scan = args[0] if args else None
rows = {}
here = os.path.dirname(os.path.abspath(__file__))
files = [(f, "") for f in sorted(glob.glob(os.path.join(scan, "scan_up*.jsonl")))] if scan else []
if scan and not files:
    sys.exit(f"export_dd: no scan_up*.jsonl in {scan}")
if not files:
    for r in csv.DictReader(ln for ln in open(os.path.join(here, "data", "large_d", "cloud_scan_rows.csv"))
                            if not ln.startswith("#")):
        rows[round(float(r["D"]), 6)] = (dict(D=float(r["D"]), Delta=float(r["Delta_hi"]), Delta_lo=float(r["Delta_lo"]),
                                           F=float(r["F"]), sens=[float(r["max_dDelta_dF"])],
                                           dDelta_est=float(r["dDelta_est"]) if r["dDelta_est"] else None), r["run"])
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
# T48 above the scans' last point (see the docstring for the rules)
LAST_SCAN = max(rows)
T48_ALPHA_TEST = 5            # up70.jsonl's last records: the horizon_alpha test (2026-10-25 16:13-16:22 UTC, comms)


def t48_records():
    out = []
    for f, run in (("pilot_cloud/pilotB.jsonl", "T48-pilotB"), ("up70.jsonl", "T48-up70")):
        recs = [json.loads(line) for line in open(os.path.join(here, "data", "large_d", f)) if line.strip()]
        if run == "T48-up70":
            recs = recs[:-T48_ALPHA_TEST]
        keep = [r for r in recs if r.get("ks") in (100, 130) and r["tol"] <= 2.5e-21
                and abs(r.get("dDelta_est") or 0.0) <= 1e-15]
        for i, r in enumerate(keep):
            same = [q["tail_h"] for q in keep[max(0, i - 10):i + 11] if q is not r and q["ks"] == r["ks"]]
            if same and r["tail_h"] > 100 * sorted(same)[len(same) // 2]:
                continue                                  # a horizon-tail jump
            if round(r["D"], 6) > LAST_SCAN + 1e-9:
                out.append((r, run))
    return out


T48_ROWS = t48_records()
for r, run in T48_ROWS:
    rows[round(r["D"], 6)] = (r, run)
# published values at these D (reference, value, the stated uncertainty)
pub = {}
for line in open(os.path.join(here, "data", "published_delta.csv")):
    if line.startswith("#") or line.startswith("D,"):
        continue
    d, v, u, ref = line.strip().split(",")
    pub.setdefault(round(float(d), 6), []).append((ref, float(v), float(u)))
# dDelta_exact for T48-up70 (T60b, comms 2026-10-25 23:42 UTC)
EXACT = {round(r["D"], 6): r["dDelta_q"] for r in json.loads(open(os.path.join(
    here, "data", "large_d", "qdh_delta_check3_mac.jsonl")).read().splitlines()[-1])["rows"]}
CHECK_BELOW_35, CHECK_FROM_35 = 1e-16, 2e-15   # README "Accuracy": tau resolution (D = 7-17, 40) and freezing (40-60)
out = os.path.join(here, "data", "delta_dd.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
# never lose a row silently: every D of the current CSV must have an input
old = {round(float(ln.split(",")[0]), 6) for ln in open(out).readlines()[1:]} if os.path.exists(out) else set()
lost = sorted(old - set(rows))
if lost and "--allow_drop" not in sys.argv:
    sys.exit(f"export_dd: {len(lost)} rows of {out} have no input (D = {lost[0]} .. {lost[-1]}); not written "
             "(--allow_drop writes anyway)")
with open(out, "w") as fh:
    fh.write("D,Delta_hi,Delta_lo,F,max_dDelta_dF,dDelta_est,run,Delta_err,err_kind,published_ref,published_diff,"
             "published_match,dDelta_exact\n")
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
        exact = format(EXACT[d], ".2e") if run == "T48-up70" else ""
        fh.write(f"{r['D']!r},{r['Delta']!r},{r['Delta_lo']!r},{r['F']:.3e},{max(r['sens']):.3e},"
                 f"{'' if est is None else format(est, '.2e')},{run},{err},{kind},{ref},{diff},{match},{exact}\n")
print(f"{len(rows)} points, D = {min(rows):.4f} .. {max(rows):.4f} -> {out}; of them {len(T48_ROWS)} from T48 above "
      f"{LAST_SCAN}")
