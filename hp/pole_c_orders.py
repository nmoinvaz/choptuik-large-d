#!/usr/bin/env python3
"""notes/QUAD_DOUBLE.md 13.6's pole constant C against the horizon series' order count (#546).

    python3 hp/pole_c_orders.py RAW

RAW is a clone of the raw-data repository. The script reads the per-state tables of `hp/pole_bc.py summary` in
`I538-mac/` (the states' own 100 orders, and 130 and 160 at D 40.02 and 45.05) and `I546-win/` (130 and 160 at
D 50.01, 55.04 and 60.00; 200 and 250 at all five), keeps the states D 40-60, and writes
`hp/data/pole_c_orders.csv`: D, orders, C(R136) (13.6's estimator) and C(a), with each source folder's commit in the
header. check_numbers.py reads the CSV for the ranges quoted in 13.6's addendum.
"""
import os
import subprocess
import sys

SOURCES = [  # (folder, summary log, orders, which D: None for all states of the log)
    ("I538-mac", "pole_bc_summary.log", 100),
    ("I538-mac", "pole_bc_ks130_D40_45_summary.log", 130),
    ("I538-mac", "pole_bc_ks160_D40_45_summary.log", 160),
    ("I546-win", "pole_bc_ks130_summary.log", 130),
    ("I546-win", "pole_bc_ks160_summary.log", 160),
    ("I546-win", "pole_bc_ks200_summary.log", 200),
    ("I546-win", "pole_bc_ks250_summary.log", 250),
]
D_MAX = 61.0


def rows(path):
    """the per-state table of a summary log: [(D, C_R136, C_a)]"""
    out, cols = [], None
    for line in open(path):
        w = line.split()
        if w[:2] == ["D", "C_R136"]:
            cols = w
            continue
        if cols is None:
            continue
        try:
            D = float(w[0])
        except (ValueError, IndexError):
            if out:
                break
            continue
        out.append((D, float(w[cols.index("C_R136")]), float(w[cols.index("C_a")])))
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    raw = sys.argv[1]
    commits = {f: subprocess.run(["git", "-C", raw, "log", "-1", "--format=%h", "--", f], capture_output=True,
                                 text=True, check=True).stdout.strip() for f in {s[0] for s in SOURCES}}
    seen, recs = set(), []
    for folder, log, orders in SOURCES:
        for D, cr, ca in rows(os.path.join(raw, folder, log)):
            if D > D_MAX or (D, orders) in seen:
                continue
            seen.add((D, orders))
            recs.append((D, orders, cr, ca, folder))
    recs.sort()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pole_c_orders.csv")
    with open(out, "w") as fh:
        fh.write("# hp/pole_c_orders.py from the raw-data repository: "
                 + ", ".join(f"{f} at {c}" for f, c in sorted(commits.items())) + "\n")
        fh.write("D,orders,C_R136,C_a,source\n")
        for D, o, cr, ca, f in recs:
            fh.write(f"{D:.4f},{o},{cr:.3f},{ca:.3f},{f}\n")
    print(f"wrote {out}: {len(recs)} rows")
    for D in sorted({r[0] for r in recs}):
        print(f"  D {D:.4f}: " + ", ".join(f"{o} {cr:.3f}" for d, o, cr, _, _ in recs if d == D))
    return 0


if __name__ == "__main__":
    sys.exit(main())
