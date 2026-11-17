#!/usr/bin/env python3
"""How far each solver reaches in D, and what stops it: one row per solver of hp/data/delta_all.csv on a log D axis.

    cd hp && python3 plot_solver_limits.py RAWDIR [-o solver_limits.svg]

RAWDIR is a clone of the raw-data repository (AGENTS.md). The rows are the table's kinds, with the ANM chain split by
its corrector's precision: T138-win2's chain (quad-double jets) and chain_o (octo-double jets) were corrected in
quad-double (their logs' "quad-double Newton" lines), every later chain folder in octo-double. The marked limits:
- the sonic closing, where min over tau of c - x reaches 0 (notes/LOW_D.md 17): the line through the narrowest-segment
  grid gaps of the three lowest-D M 384 states of RAWDIR/I933-win2/gap_M384.jsonl, as sonic_valley.py tangency;
- the quad-double corrector's stall, the last quad-double chain row against the first octo-double one;
- the fold of the discrete system, D*: the median of hp/data/fold_259.csv's kept estimates, as check_numbers.py.
Prints the rows and the limits as a table.
"""
import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys

import data_graphs as dg
from data_graphs import INK, INK_SOFT, SERIES, Svg, axes, better_hint, callout, caption, footnote, host_specs, segment

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/nmoinvaz/choptuik-large-d"
# chain folders corrected in quad-double; every other chain_fit folder was corrected in octo-double
QD_CORRECTED = ("T138-win2/chain", "T138-win2/chain_o")


def chain_folder(src):
    return os.path.dirname(src)


def rows_by_solver():
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "delta_all.csv"))))
    out = {k: [] for k in ("dd", "pinned", "qd", "chain_q", "chain_o", "series")}
    for r in rows:
        k = r["kind"]
        if k == "chain_fit":
            k = "chain_q" if chain_folder(r["source"]) in QD_CORRECTED else "chain_o"
        if k in out:
            out[k].append(float(r["D"]))
    return out


def sonic_closing(raw):
    """the line through the three lowest-D states' narrowest grid gaps reaches 0 (sonic_valley.py tangency --last 3)"""
    gaps = {}
    for line in open(os.path.join(raw, "I933-win2", "gap_M384.jsonl")):
        r = json.loads(line)
        if r.get("finite") and r.get("gap") is not None:
            D = float(r["D"])
            gaps[D] = min(gaps.get(D, math.inf), float(r["gap"]))
    pts = sorted(gaps.items())[:3]
    Dm = statistics.fmean(d for d, _ in pts)
    ym = statistics.fmean(g for _, g in pts)
    a = sum((d - Dm) * (g - ym) for d, g in pts) / sum((d - Dm) ** 2 for d, _ in pts)
    return Dm - ym / a, pts


def fold_dstar():
    fold = list(csv.DictReader(open(os.path.join(HERE, "data", "fold_259.csv"))))
    return statistics.median(float(r[f"{e}_Dstar"]) for r in fold if r["plan"] == "kept" for e in ("ratio", "dlog"))


def commit():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("-o", default=os.path.join(HERE, "solver_limits.svg"))
    a = ap.parse_args()
    by = rows_by_solver()
    Dc, gap_pts = sonic_closing(a.raw)
    Ds = fold_dstar()
    q_end, o_start = max(by["chain_q"]), min(by["chain_o"])
    spec = [  # label, kind, color slot (the extrapolation takes the neutral last slot)
        ("double-double (hp/, M ≤ 384)", "dd", 0),
        ("double-double, pinned runs", "pinned", 1),
        ("quad-double scan (scan_qd.py)", "qd", 2),
        ("ANM chain, quad-double corrector", "chain_q", 3),
        ("ANM chain, octo-double corrector", "chain_o", 6),
        ("1/n series fit: an extrapolation, not a solve", "series", 9),
    ]
    n = len(spec)
    print(f"{'solver':48s} {'rows':>5s} {'D from':>11s} {'D to':>11s}")
    for label, k, _ in spec:
        print(f"{label:48s} {len(by[k]):5d} {min(by[k]):11.5f} {max(by[k]):11.5f}")
    print(f"\nsonic closing (I933-win2, M 384, line through D {', '.join(f'{d:.5f}' for d, _ in gap_pts)}): D {Dc:.6f}")
    print(f"quad-double corrector: last row D {q_end:.4f}; octo-double corrector: first row D {o_start:.4f}")
    print(f"fold of the discrete system (fold_259.csv, median of the kept estimates): D* {Ds:.4f}")

    W = 980
    svg = Svg(W)
    top = caption(svg, 16, 22, "How far each solver reaches in D, and what stops it",
                  "rows of hp/data/delta_all.csv by solver and precision; rings mark the measured limits")
    svg.text(16, 52, "Inside the octo-double chain, the double-double horizon rows are rounding noise past order ~100 "
                     "near D 229 (#903).", size=10)
    top += 10
    x, w, h = 300, W - 300 - 96, 34 * n
    orig = dg.nice_ticks
    dg.nice_ticks = lambda lo, hi, k=5: list(range(n)) if (lo, hi) == (-0.5, n - 0.5) else orig(lo, hi, k)
    try:
        xmap, ymap = axes(svg, x, top, w, h, (0, 400), (-0.5, n - 0.5), "spacetime dimension D", "solver",
                          xfmt=lambda v: f"{v:g}", yfmt=lambda v: spec[n - 1 - int(v)][0])
    finally:
        dg.nice_ticks = orig
    row_y = {k: ymap(n - 1 - i) for i, (_, k, _) in enumerate(spec)}
    bh = 16
    for label, k, slot in spec:
        lo, hi = min(by[k]), max(by[k])
        ink = "#ffffff" if slot in (0, 6, 9) else INK
        segment(svg, xmap(lo), xmap(hi) + 2, row_y[k] - bh / 2, bh, SERIES[slot],
                f"{label}: {len(by[k])} rows, D {lo:g} to {hi:g}",
                label=[f"{lo:g} – {hi:g}", f"{lo:.4g} – {hi:.4g}", f"{lo:.3g}–{hi:.3g}"], ink=ink)
    # labels go into the empty upper right (rows 0-2 end by D 73.3)
    callout(svg, xmap(Dc), row_y["dd"], f"sonic closing, D {Dc:.4f}", dx=130, dy=row_y["pinned"] - row_y["dd"])
    jx, jy = xmap((q_end + o_start) / 2), (row_y["chain_q"] + row_y["chain_o"]) / 2
    callout(svg, jx, jy, f"quad-double floor > 1e-28 at D {q_end:g}", dx=40, dy=row_y["qd"] - jy)
    callout(svg, xmap(Ds), row_y["chain_o"], f"fold, D* {Ds:.2f}", dx=40,
            dy=(row_y["dd"] + row_y["pinned"]) / 2 - row_y["chain_o"])
    better_hint(svg, x, top, w, h, "axis", label="farther")
    fh = footnote(svg, [f"hp/data/delta_all.csv, fold_259.csv at {commit()}", "sonic closing: raw I933-win2",
                        "chain correctors: T138-win2 logs"], "drawn on " + host_specs(), [], top + h + 46, REPO)
    open(a.o, "w").write(svg.finish(fh))
    print(f"\nwrote {a.o}")


if __name__ == "__main__":
    sys.exit(main())
