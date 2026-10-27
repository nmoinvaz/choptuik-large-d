#!/usr/bin/env python3
"""Review RV2: one line per continuation run (settings, D range, residuals, tails, dDelta_est).

    python3 hp/review/rv2_runs.py CLOUD_SCAN_DIR

CLOUD_SCAN_DIR held the cloud runs' scan_up*.jsonl (/home/claude/dss/hp_scan); it is missing since 2026-10-26.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HP = os.path.dirname(HERE)
scan = sys.argv[1] if len(sys.argv) > 1 else None
if not scan or not glob.glob(os.path.join(scan, "scan_up*.jsonl")):
    sys.exit("rv2_runs: no scan_up*.jsonl in CLOUD_SCAN_DIR; the cloud runs' records have been missing since the cloud "
             "session ended (2026-10-26), so this audit cannot be rerun without them (notes/NUMERICS_REVIEW.md)")
files = sorted(glob.glob(os.path.join(scan, "scan_up*.jsonl")), key=lambda f: (len(f), f))
files += sorted(glob.glob(os.path.join(HP, "data", "local", "scan_up*.jsonl")))
for f in files:
    rs = [json.loads(line) for line in open(f)]
    if not rs:
        continue
    Ds = [r["D"] for r in rs]
    est = [abs(r["dDelta_est"]) for r in rs if r.get("dDelta_est") is not None]
    tc = [r.get("tail_c", 0) for r in rs]
    th = [r.get("tail_h", 0) for r in rs]
    print(f"{('local/' if 'local' in f else '') + os.path.basename(f):24s} n {len(rs):3d}  D {min(Ds):8.4f}..{max(Ds):8.4f}"
          f"  M {sorted({r.get('M') for r in rs})}  K {sorted({len(r.get('nodes', [])) for r in rs})}"
          f"  kc {sorted({r.get('kc') for r in rs})}  ks {sorted({r.get('ks') for r in rs})}  kt {sorted({r.get('kt') for r in rs})}"
          f"  renoded {sum(1 for r in rs if r.get('renoded'))}  |F| {min(r['F'] for r in rs):.1e}..{max(r['F'] for r in rs):.1e}"
          f"  est {max(est) if est else float('nan'):.1e}  tail_c max {max(tc):.0e}  tail_h max {max(th):.0e}"
          f"  polish {sum(1 for r in rs if r.get('polish'))}")
