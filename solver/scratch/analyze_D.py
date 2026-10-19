"""Collect the dimension scan, compare with the literature, and fit large-D behaviour."""
import json, glob, sys
import numpy as np

rows = {}
for fn in sorted(glob.glob("/home/claude/dss/scan_up_kc*.jsonl")):
    for line in open(fn):
        r = json.loads(line)
        key = (round(r["D"], 4), r["kc"])
        rows[key] = r
D = np.array(sorted({k[0] for k in rows}))
print(f"{'D':>6} {'kc':>3} {'Delta':>12} {'gamma':>12} {'lam0':>10} {'lam_gauge':>10} {'resid':>8} {'tail':>8}")
for d in D:
    for kc in sorted({k[1] for k in rows if k[0] == d}):
        r = rows[(d, kc)]
        g = f"{r['gamma']:.8f}" if r["gamma"] else "     -"
        l = f"{r['lam']:.6f}" if r["lam"] else "   -"
        print(f"{d:6.2f} {kc:3d} {r['Delta']:12.8f} {g:>12} {l:>10} {r['lam_gauge']:10.6f} {r['residual']:8.1e} {r['tail']:8.1e}")
