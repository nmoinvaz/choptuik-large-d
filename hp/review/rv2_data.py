#!/usr/bin/env python3
"""Review RV2: audit of hp/data/delta_dd.csv against the run files it is built from.

    python3 hp/review/rv2_data.py [CLOUD_SCAN_DIR] [--break=KIND]     (default /home/claude/dss/hp_scan)

Independent of export_dd.py (written from its docstring, not its code). Checks:
  A  the CSV is exactly the merge "cloud runs in run order, then local runs, the last record per D (to 1e-6) wins";
  B  every D is the float the solver ran at: D == round(D, 6) (dd_fits_mp keys points by round(D, 6), and the
     solver's n = D - 2 came from that same float), and no two kept points are closer than 1e-6;
  C  hi + lo is a normalised double-double: |lo| <= ulp(hi) / 2, and hi + lo == hi in f64;
  D  per point: centre / horizon tail (series truncation) times the block's max |dDelta/dF|, |F|, |dDelta_est|;
     the points where that bound or the accepted residual is large are listed;
  E  points present in both a cloud and a local run, or in two runs of the same machine: their Delta difference
     (a free reproducibility check).
--break=KIND corrupts the input once to show the check fires: 'merge' (drop the local override), 'round' (a D with
7 decimals), 'lo' (an unnormalised lo).
"""
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HP = os.path.dirname(HERE)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
brk = next((a[8:] for a in sys.argv[1:] if a.startswith("--break=")), None)
scan = args[0] if args else "/home/claude/dss/hp_scan"


def runs():
    cloud = sorted(glob.glob(os.path.join(scan, "scan_up*.jsonl")))      # export_dd.py sorts by name
    local = sorted(glob.glob(os.path.join(HP, "data", "local", "scan_up*.jsonl")))
    out = []
    for f in cloud:
        out.append(("cloud", os.path.basename(f)[5:-6], [json.loads(x) for x in open(f)]))
    if brk != "merge":
        for f in local:
            out.append(("local", "local-" + os.path.basename(f)[5:-6], [json.loads(x) for x in open(f)]))
    return out


R = runs()
merged, allrec = {}, {}
for mach, name, recs in R:
    for r in recs:
        k = round(r["D"], 6)
        merged[k] = (r, name)
        allrec.setdefault(k, []).append((mach, name, r))

csv = [line.strip().split(",") for line in list(open(os.path.join(HP, "data", "delta_dd.csv")))[1:]]
if brk == "round":
    csv[100][0] = repr(float(csv[100][0]) + 3e-7)
if brk == "lo":
    csv[200][2] = repr(float(csv[200][1]) * 1e-15)
fails = 0

# ---------------------------------------------------------------- A: the merge
mism = []
if len(csv) != len(merged):
    mism.append(f"count {len(csv)} vs {len(merged)}")
for c in csv:
    k = round(float(c[0]), 6)
    if k not in merged:
        mism.append(f"{c[0]} not in runs")
        continue
    r, name = merged[k]
    if (float(c[0]), float(c[1]), float(c[2]), c[6]) != (r["D"], r["Delta"], r["Delta_lo"], name):
        mism.append(f"{c[0]}: csv {c[1]} {c[2]} {c[6]} vs runs {r['Delta']!r} {r['Delta_lo']!r} {name}")
print(f"A merge: {len(csv)} CSV rows, {len(merged)} distinct D in the runs; mismatches {len(mism)}")
for m in mism[:5]:
    print("   ", m)
fails += bool(mism)

# ---------------------------------------------------------------- B: D values
Dv = [float(c[0]) for c in csv]
bad = [d for d in Dv if round(d, 6) != d]
gaps = sorted(Dv)
close = [(a, b) for a, b in zip(gaps, gaps[1:]) if b - a < 1e-6]
print(f"B D values: {len(bad)} with more than 6 decimals {bad[:5]}; {len(close)} pairs closer than 1e-6; "
      f"smallest spacing {min(b - a for a, b in zip(gaps, gaps[1:])):.2e}")
fails += bool(bad) + bool(close)

# ---------------------------------------------------------------- C: hi + lo normalisation
badlo = [c[0] for c in csv if not (abs(float(c[2])) <= math.ulp(float(c[1])) / 2 and float(c[1]) + float(c[2]) == float(c[1]))]
print(f"C hi/lo: {len(badlo)} unnormalised {badlo[:5]}")
fails += bool(badlo)

# ---------------------------------------------------------------- D: per-point error indicators
print("D per-point indicators (kept points):")
rows = []
for c in csv:
    r, name = merged[round(float(c[0]), 6)]
    s = r["sens"]
    tb = max(r.get("tail_c", 0) * s[0], r.get("tail_h", 0) * s[-1])     # series truncation x block sensitivity
    est = r.get("dDelta_est")
    rows.append((r["D"], name, r["F"], tb, est, max(s)))
for lo, hi in ((8, 27), (27, 35.25), (35.25, 45.34), (45.34, 60.1), (60.1, 62)):
    sel = [x for x in rows if lo <= x[0] < hi]
    ests = [abs(x[4]) for x in sel if x[4] is not None]
    print(f"   D {lo:5.2f}-{hi:5.2f}: {len(sel):3d} pts  |F| max {max(x[2] for x in sel):.1e}  "
          f"tail x sens max {max(x[3] for x in sel):.1e}  |dDelta_est| max {max(ests) if ests else float('nan'):.1e} "
          f"(recorded for {len(ests)})  |F| x max sens: max {max(x[2] * x[5] for x in sel):.1e}")
print("   points with tail x sens > 1e-20 or |F| > 1e-20:")
for x in rows:
    if x[3] > 1e-20 or x[2] > 1e-20:
        print(f"     D {x[0]:10.6f} {x[1]:10s} |F| {x[2]:.1e} tail x sens {x[3]:.1e} est "
              f"{'' if x[4] is None else format(x[4], '.1e')}")

# ---------------------------------------------------------------- E: repeated points
print("E points solved in more than one run (last one kept):")
for k in sorted(allrec):
    v = allrec[k]
    if len(v) > 1:
        base = v[-1][2]
        diffs = [((r["Delta"] - base["Delta"]) + (r["Delta_lo"] - base["Delta_lo"])) for _, _, r in v[:-1]]
        print(f"   D {k:10.6f}: " + ", ".join(f"{n} {'(renoded)' if r.get('renoded') else ''}" for _, n, r in v)
              + "  Delta(earlier) - Delta(kept): " + ", ".join(f"{d:+.1e}" for d in diffs))
print("FAIL" if fails else "PASS (A-C)")
sys.exit(1 if fails else 0)
