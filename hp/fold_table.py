#!/usr/bin/env python3
"""The square-root point of #603's branch near D 259.9 (notes/LARGE_D_WALL.md "Where the series is singular"): Delta's
D* and exponent mu at each base, from hp/singularity_type.py's runs in the raw-data repository, into
hp/data/fold_259.csv (D0, plan, the ratio fit's and the Dlog Pade's D*, mu and class, the source file).

    cd hp && python3 fold_table.py RAWDIR [--out=data/fold_259.csv]

RAWDIR is a clone of the raw-data repository. The runs: #794 (win2, I794-win2), #792 (dh610, I792-dh610) and #799
(dh370, I799-dh370) on #603's plan ("kept"), and #795 (dh370, I795-dh370), which repeated two bases on #603's plan and
on a plan made at the state ("new"). It exits 1 unless it finds 8 bases on the kept plan and 2 on the new one, each
with both estimators.
"""
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ("I794-win2/type794.txt", "I792-dh610/type792.txt", "I799-dh370/type799.txt", "I795-dh370/compare795.txt")
LINE = re.compile(r"D0 ([0-9.]+) Delta (ratio|dlog): D\* ([0-9.]+) \(s [0-9.]+\), mu ([+-][0-9.]+) ([A-Za-z]+)")
ROW = re.compile(r"\| ([0-9.]+) \| (kept|new) \| ([0-9.]+), ([+-][0-9.]+) ([A-Za-z]+) \| ([0-9.]+), ([+-][0-9.]+) ([A-Za-z]+) \|")
COLS = ["D0", "plan", "ratio_Dstar", "ratio_mu", "ratio_class", "dlog_Dstar", "dlog_mu", "dlog_class", "source"]
EXPECT = {"kept": 8, "new": 2}


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if not pos:
        sys.exit(__doc__)
    raw = os.path.abspath(pos[0])
    rows = {}
    for f in FILES:
        for ln in open(os.path.join(raw, f)):
            m = ROW.search(ln)
            if m:
                k = (m.group(1), m.group(2))
                rows.setdefault(k, dict(D0=m.group(1), plan=m.group(2), ratio_Dstar=m.group(3), ratio_mu=m.group(4),
                                        ratio_class=m.group(5), dlog_Dstar=m.group(6), dlog_mu=m.group(7),
                                        dlog_class=m.group(8), source=f))
                continue
            m = LINE.search(ln)
            if m:
                r = rows.setdefault((m.group(1), "kept"), dict(D0=m.group(1), plan="kept", source=f))
                r.update({f"{m.group(2)}_Dstar": m.group(3), f"{m.group(2)}_mu": m.group(4),
                          f"{m.group(2)}_class": m.group(5)})
    out = sorted(rows.values(), key=lambda r: (r["plan"], float(r["D0"])))
    bad = [r["D0"] for r in out if any(c not in r for c in COLS)]
    n = {p: sum(r["plan"] == p for r in out) for p in EXPECT}
    path = opts.get("out", os.path.join(HERE, "data", "fold_259.csv"))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)
    head = subprocess.run(["git", "-C", raw, "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"{len(out)} rows (raw at {head}) -> {path}")
    for p in EXPECT:
        rs = [r for r in out if r["plan"] == p and r["D0"] not in bad]
        if not rs:
            continue
        for est in ("ratio", "dlog"):
            ds, ms = [float(r[f"{est}_Dstar"]) for r in rs], [float(r[f"{est}_mu"]) for r in rs]
            print(f"  {p}: {len(rs)} bases, D0 {rs[0]['D0']} to {rs[-1]['D0']}; {est}: D* {min(ds):.4f} to {max(ds):.4f},"
                  f" mu {min(ms):+.3f} to {max(ms):+.3f}")
    ok = n == EXPECT and not bad
    print("verify: " + ("8 bases on #603's plan and 2 on the new plan, each with both estimators" if ok else
                        f"FAILS: {n} bases, incomplete {bad}"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
