#!/usr/bin/env python3
"""Rebuild every Delta table from the raw-data repository and check the master table, in one command. The tables are
updated after every new result.

    cd hp && python3 update_delta.py RAWDIR

The steps, in order:
1. export_qd.py RAWDIR writes data/delta_qd.csv.
2. export_lowD.py --raw RAWDIR writes data/delta_dd_lowD.csv.
3. export_dd.py --raw RAWDIR writes data/delta_dd.csv, which takes the low-D rows.
4. export_delta_all.py RAWDIR writes data/delta_all.csv and data/delta_fits.csv.
5. export_delta_all.py --verify checks the master table.
6. Each --mutate control runs and must fail by its check (one per kind, and held, frontier and t85 for the chains'
   checks, fits, fits_table and half for the error bar over the held fit's degree, pinned_qd, gap, neighbours and
   cross for the pinned rows' rules, qd_neighbours, dm_neighbours and one_sided for the rules of the qd rows with one
   start and the low-D rows at one M). A kind with no rows has nothing to mutate, and its control is reported as
   skipped.
   The master table is then restored and verified again.
7. export_delta_all.py --interp_check checks the interpolation of the comparison fits at D 173, 174 and 186 to 191, and
   its control --mutate=interp must fail.
7b. export_delta_all.py --lam_screen screens the chain_fit backbone's gauge rate (#637), and its control
   --mutate=lam must fail.
8. compare_published.py writes data/delta_published_compare.csv, every published value against ours.
9. compare_gamma_published.py writes data/gamma_published_compare.csv, the published gamma and Delta below D = 14
   against ours (#620), whose check in check_numbers.py fails once our values move by more than 1e-10.

It prints each table's row count before and after, and the master table's new and changed rows. It exits 1 if a step
fails, a control passes or a row is gone (other than rows that chain_fit replaces). A low-D task with new points must first be added to export_lowD.py's RAW_TASKS, and the
counts quoted in notes/QUAD_DOUBLE.md 13.25 are updated by hand.
"""
import csv
import os
import subprocess
import sys

from export_delta_all import COLS, fit_span, replaced

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TABLES = ["delta_qd.csv", "delta_dd_lowD.csv", "delta_dd.csv", "delta_all.csv", "delta_fits.csv"]


def rows(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return {}
    with open(p, newline="") as f:
        return {r[0]: r for r in list(csv.reader(f))[1:] if r}


def run(args, expect=0):
    r = subprocess.run([sys.executable] + args, cwd=HERE, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    # a control fails by its check (exit 1 after the verify line), not by a crash; exit 2 is a kind with no rows
    skip = expect == 1 and r.returncode == 2
    ok = (r.returncode == 0) if expect == 0 else (skip or (r.returncode == 1 and last.startswith("verify:")))
    tag = "skip" if skip else "ok  " if ok else "FAIL"
    print(f"  {tag} {' '.join(args)}: exit {r.returncode}; {last[:110]}", flush=True)
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
    ok &= run(["export_dd.py", "--raw", raw])
    ok &= run(["export_delta_all.py", raw])
    if not ok:
        return 1
    master = os.path.join(DATA, "delta_all.csv")
    with open(master, "rb") as f:
        built = f.read()
    print("checks of the master table:", flush=True)
    ok &= run(["export_delta_all.py", raw, "--verify"])
    for m in ("dd", "qd", "pinned", "pinned_qd", "gap", "neighbours", "cross", "chain_fit", "chain", "held", "frontier",
              "projector", "t85", "fits", "fits_table", "half", "qd_neighbours", "dm_neighbours", "one_sided"):
        ok &= run(["export_delta_all.py", raw, "--verify", f"--mutate={m}"], expect=1)
    with open(master, "wb") as f:
        f.write(built)
    ok &= run(["export_delta_all.py", raw, "--verify"])
    ok &= run(["export_delta_all.py", raw, "--interp_check"])
    ok &= run(["export_delta_all.py", raw, "--interp_check", "--mutate=interp"], expect=1)
    ok &= run(["export_delta_all.py", raw, "--lam_screen"])
    ok &= run(["export_delta_all.py", raw, "--lam_screen", "--mutate=lam"], expect=1)
    ok &= run(["compare_published.py"])
    ok &= run(["compare_gamma_published.py"])
    print("tables:", flush=True)
    for t in TABLES:
        b, a = before[t], rows(t)
        new = sorted(set(a) - set(b), key=float)
        changed = [k for k in a if k in b and a[k] != b[k]]
        gone = [k for k in b if k not in a]
        # rows that chain_fit replaces (export_delta_all.py), the old ones staying in git history
        span = fit_span([dict(zip(COLS, a[k])) for k in a]) if t == "delta_all.csv" else None
        repl = [k for k in gone if span and replaced(dict(zip(COLS, b[k])), span)]
        print(f"  {t}: {len(b)} -> {len(a)} rows, {len(new)} new, {len(changed)} changed, {len(gone)} gone"
              + (f" ({len(repl)} rows replaced by chain_fit)" if repl else "")
              + (f"; new D {new[0]}..{new[-1]}" if new else ""), flush=True)
        ok &= len(gone) == len(repl)
    print("done" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
