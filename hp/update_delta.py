#!/usr/bin/env python3
"""Rebuild every Delta table from the raw-data repository and check the master table, in one command. The tables are
updated after every new result.

    cd hp && python3 update_delta.py RAWDIR

The steps, in order:
1. export_qd.py RAWDIR writes data/delta_qd.csv.
2. export_lowD.py --raw RAWDIR writes data/delta_dd_lowD.csv.
3. export_dd.py writes data/delta_dd.csv, which takes the low-D rows.
4. export_delta_all.py RAWDIR writes data/delta_all.csv.
5. export_delta_all.py --verify checks the master table.
6. Each --mutate control runs and must fail (one per kind, and held and frontier for the two chains). The master table is then restored and verified again.
7. compare_published.py writes data/delta_published_compare.csv, every published value against ours.

It prints each table's row count before and after, and the master table's new and changed rows. It exits 1 if a step
fails, a control passes or a row is gone (other than chain rows that chain_fit replaces). A low-D task with new points must first be added to export_lowD.py's RAW_TASKS, and the
counts quoted in notes/QUAD_DOUBLE.md 13.25 are updated by hand.
"""
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TABLES = ["delta_qd.csv", "delta_dd_lowD.csv", "delta_dd.csv", "delta_all.csv"]


def rows(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return {}
    with open(p, newline="") as f:
        return {r[0]: r for r in list(csv.reader(f))[1:] if r}


def run(args, expect=0):
    r = subprocess.run([sys.executable] + args, cwd=HERE, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    ok = (r.returncode == 0) if expect == 0 else (r.returncode != 0)
    print(f"  {'ok  ' if ok else 'FAIL'} {' '.join(args)}: exit {r.returncode}; {last[:110]}", flush=True)
    if not ok:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
    return ok


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    raw = os.path.abspath(sys.argv[1])
    before = {t: rows(t) for t in TABLES}
    ok = True
    print("exports:", flush=True)
    ok &= run(["export_qd.py", raw])
    ok &= run(["export_lowD.py", "--raw", raw])
    ok &= run(["export_dd.py"])
    ok &= run(["export_delta_all.py", raw])
    if not ok:
        return 1
    master = os.path.join(DATA, "delta_all.csv")
    with open(master, "rb") as f:
        built = f.read()
    print("checks of the master table:", flush=True)
    ok &= run(["export_delta_all.py", raw, "--verify"])
    for m in ("dd", "qd", "pinned", "chain_fit", "chain", "held", "frontier"):
        ok &= run(["export_delta_all.py", raw, "--verify", f"--mutate={m}"], expect=1)
    with open(master, "wb") as f:
        f.write(built)
    ok &= run(["export_delta_all.py", raw, "--verify"])
    ok &= run(["compare_published.py"])
    print("tables:", flush=True)
    for t in TABLES:
        b, a = before[t], rows(t)
        new = sorted(set(a) - set(b), key=float)
        changed = [k for k in a if k in b and a[k] != b[k]]
        gone = [k for k in b if k not in a]
        # chain rows at or below the chain_fit frontier are replaced by it (export_delta_all.py)
        front = max((float(k) for k in a if t == "delta_all.csv" and a[k][5] == "chain_fit"), default=float("-inf"))
        repl = [k for k in gone if b[k][5] == "chain" and float(k) <= front + 1e-9]
        print(f"  {t}: {len(b)} -> {len(a)} rows, {len(new)} new, {len(changed)} changed, {len(gone)} gone"
              + (f" ({len(repl)} chain rows replaced by chain_fit)" if repl else "")
              + (f"; new D {new[0]}..{new[-1]}" if new else ""), flush=True)
        ok &= len(gone) == len(repl)
    print("done" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
