#!/usr/bin/env python3
"""solve_times.svg: how long each D took to solve. Low D (3 < D <= 4): hours of every solve at that D (all tau
resolutions M, both passes of each lift, failed and check solves) and of the converged solves at the largest M that
gave the value; large D: wall-clock seconds of each accepted continuation point's Newton solve (failed steps are not
recorded). Standard library only; the primitives are data_graphs.py, vendored unchanged from
github.com/nmoinvaz/speedy-gonzales (skills/graph-data). Prints the plotted numbers.

Data: data/solve_times.csv, written by --export from the solve records: data/lowD_dd*.jsonl and
data/lowD_highM/*points*.jsonl (low D; a solve copied into several files counts once), the cloud runs' scan_up*.jsonl
(SCAN_DIR, default /home/claude/dss/hp_scan, as for export_dd.py), data/local/scan_up*.jsonl (the Windows machine),
data/large_d/pilot_cloud/pilotB.jsonl and data/large_d/up70*.jsonl (the Mac).

    cd hp && python3 plot_solve_times.py [--export [SCAN_DIR]]
"""
import collections
import csv
import glob
import json
import math
import os
import sys

from data_graphs import INK, SERIES, Svg, axes, better_hint, callout, caption, esc, footnote, legend, marker

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "solve_times.csv")
REPO = "https://github.com/nmoinvaz/choptuik-large-d"
WIDTH, X, W = 1080, 80, 952
MACHINE = {"cloud": "cloud (2 shared cores)", "win": "Windows i7-1185G7", "mac": "Mac, Apple M5 (4 threads)"}
# where the continuation runs' settings changed (README, "The double-double solver"): run, label, callout offsets
CHANGES = (("up2", "M 48 → 32 τ modes, 10 → 8 nodes", 26, 40),
           ("up3", "centre series 80 → 260+ terms, 5 nodes", 26, -40),
           ("up7", "harmonics above 13 frozen from here", -26, -44))


def nice_lo(v):
    """the largest 1, 2 or 5 x 10^k at or below v"""
    e = 10 ** math.floor(math.log10(v))
    return max(m * e for m in (1, 2, 5) if m * e <= v)


def nice_hi(v):
    """the smallest 1, 2 or 5 x 10^k at or above v"""
    e = 10 ** math.floor(math.log10(v))
    return min(m * e for m in (1, 2, 5, 10) if m * e >= v)


def jl(f):
    return [json.loads(x) for x in open(f) if x.strip()]


def export(scan):
    rows = []
    seen, per = set(), collections.defaultdict(list)
    files = sorted(glob.glob(os.path.join(HERE, "data", "lowD_dd*.jsonl"))) + \
        sorted(glob.glob(os.path.join(HERE, "data", "lowD_highM", "*points*.jsonl")))
    for f in files:
        src = os.path.relpath(f, HERE).replace(os.sep, "/")
        for r in jl(f):
            if "D" not in r or "seconds" not in r or r["D"] > 4.0001:
                continue
            d = r["Delta"][0] if isinstance(r.get("Delta"), list) else r.get("Delta")
            key = (round(r["D"], 6), r.get("M"), round(r["seconds"], 1), d)
            if key in seen:
                continue
            seen.add(key)
            per[round(r["D"], 6)].append((r.get("M"), r["seconds"], src, r.get("converged", True)))
    best = {float(r["D"]): (int(r["M"]), r["source"])
            for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_dd_lowD.csv"))) if r["best"] == "1"}
    for D, rs in sorted(per.items()):
        rows.append(dict(kind="lowD_all", D=D, seconds=round(sum(r[1] for r in rs), 1), M="", solves=len(rs),
                         its="", machine="", run=""))
        bm = best.get(D)
        if bm:
            # the converged solves at the value's M in the value's file: both passes of a lift, not failed attempts
            fin = sum(s for M, s, src, ok in rs if M == bm[0] and src == bm[1] and ok)
            rows.append(dict(kind="lowD_value", D=D, seconds=round(fin, 1), M=bm[0], solves="", its="", machine="",
                             run=bm[1]))
    runs = [("cloud", f) for f in sorted(glob.glob(os.path.join(scan, "scan_up*.jsonl")))]
    runs += [("win", f) for f in sorted(glob.glob(os.path.join(HERE, "data", "local", "scan_up*.jsonl")))]
    runs += [("cloud", os.path.join(HERE, "data", "large_d", "pilot_cloud", "pilotB.jsonl"))]
    runs += [("mac", f) for f in sorted(glob.glob(os.path.join(HERE, "data", "large_d", "up70*.jsonl")))
             if not f.endswith("_attempts.jsonl")]
    if not runs[0][1].startswith(scan):
        sys.exit(f"no scan_up*.jsonl in {scan}: the cloud runs' records are needed for --export")
    for machine, f in runs:
        run = os.path.basename(f).replace("scan_", "").replace(".jsonl", "")
        for r in jl(f):
            if "seconds" in r:
                rows.append(dict(kind="largeD_point", D=r["D"], seconds=r["seconds"], M=r.get("M", ""), solves="",
                                 its=r.get("its", ""), machine=machine, run=run))
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["kind", "D", "seconds", "M", "solves", "its", "machine", "run"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {CSV}: {len(rows)} rows")


def plot():
    rows = list(csv.DictReader(open(CSV)))
    low = collections.defaultdict(dict)
    for r in rows:
        if r["kind"] == "lowD_all":
            low[float(r["D"])].update(total=float(r["seconds"]), n=int(r["solves"]))
        elif r["kind"] == "lowD_value":
            low[float(r["D"])].update(final=float(r["seconds"]), M=int(r["M"]))
    large = [(r["machine"], r["run"], float(r["D"]), float(r["seconds"]), r["its"]) for r in rows
             if r["kind"] == "largeD_point"]

    svg = Svg(WIDTH)
    svg.text(16, 28, "How long each D took to solve", size=15, fill=INK, weight="bold")
    ca, cf = SERIES[0], SERIES[1]
    legend(svg, WIDTH - 16, 58, [("every solve at this D", ca), ("the solve that gave the value", cf)])
    y = caption(svg, X, 58, "3 < D ≤ 4: hours of solving per point",
                "Every solve at that D (all τ resolutions M, both passes of each lift, failed and check solves) and the "
                "converged solves at the largest M that gave the value (both passes of a lift). Triangles: no value yet.")
    h = 300
    hours = [v["total"] / 3600 for v in low.values()] + [v["final"] / 3600 for v in low.values() if v.get("final")]
    xmap, ymap = axes(svg, X, y, W, h, (3.0, 4.0), (nice_lo(min(hours)), nice_hi(max(hours)), "log"),
                      "dimension D", "hours (log scale)", xfmt=lambda v: f"{v:.1f}", yfmt=lambda v: f"{v:g} h")
    better_hint(svg, X, y, W, h, "right", label="faster", reverse=True)
    print("low D: hours per point (every solve; the converged solves that gave the value, at its M)")
    for D in sorted(low):
        v = low[D]
        has = "M" in v
        marker(svg, "circle" if has else "triangle", xmap(D), ymap(v["total"] / 3600), ca,
               f"D = {D:g}: {v['total'] / 3600:.2f} h in {v['n']} solves" + ("" if has else ", no value yet"))
        if v.get("final"):
            marker(svg, "circle", xmap(D), ymap(v["final"] / 3600), cf,
                   f"D = {D:g}: the M = {v['M']} solve that gave the value, {v['final'] / 3600:.2f} h")
        print(f"{D:8g} {v['total'] / 3600:7.2f} h ({v['n']:2d} solves)   "
              + (f"{v['final'] / 3600:6.2f} h at M = {v['M']}" if v.get("final") else "no value"))
    print(f"   total {sum(v['total'] for v in low.values()) / 3600:.1f} h")
    top = max(low, key=lambda D: low[D]["total"])
    callout(svg, xmap(top), ymap(low[top]["total"] / 3600), f"D = {top:g}: {low[top]['total'] / 3600:.0f} h, "
            f"the resolution study to M = {low[top]['M']}", dx=-30, dy=-6)
    lowest = min(D for D in low if "M" in low[D])
    callout(svg, xmap(lowest), ymap(low[lowest]["total"] / 3600), f"D = {lowest:g}: lowest point, M = {low[lowest]['M']}",
            dx=30, dy=-22)

    y2 = y + h + 92
    color = {"cloud": SERIES[2], "win": SERIES[3], "mac": SERIES[4]}
    legend(svg, WIDTH - 16, y2, [(MACHINE[m], color[m]) for m in ("cloud", "win", "mac") if any(p[0] == m for p in large)])
    dmin, dmax = min(p[2] for p in large), max(p[2] for p in large)
    yb = caption(svg, X, y2, f"{dmin:g} ≤ D ≤ {dmax:.1f}: seconds per continuation point",
                 "The Newton solve of each accepted point; failed steps are not counted. Hover a point for its run, "
                 "seconds and iterations.")
    h2 = 300
    secs = [p[3] for p in large]
    xm2, ym2 = axes(svg, X, yb, W, h2, (math.floor(dmin), math.ceil(dmax)),
                    (nice_lo(min(secs)), nice_hi(max(secs)), "log"),
                    "dimension D", "seconds (log scale)", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g} s")
    better_hint(svg, X, yb, W, h2, "right", label="faster", reverse=True)
    # Hundreds of points: small dots, each with its hover title, instead of 11px ringed markers
    for m, run, D, s, its in large:
        tip = esc(f"D = {D:.4f}: {s:.1f} s, {its} iterations ({MACHINE[m]}, run {run})")
        svg.add(f'<circle cx="{xm2(D):.1f}" cy="{ym2(s):.1f}" r="2.4" fill="{color[m]}">'
                f'<title>{tip}</title></circle>')
    for run, label, dx, dy in CHANGES:
        p = min((p for p in large if p[1] == run), key=lambda p: p[2])
        callout(svg, xm2(p[2]), ym2(p[3]), f"D = {p[2]:.2f}: {label}", dx=dx, dy=dy)
    med = collections.defaultdict(list)
    for p in large:
        med[int(p[2] // 5) * 5].append(p[3])
    print("\nlarge D: median seconds per accepted point, by 5 units of D")
    for k in sorted(med):
        v = sorted(med[k])
        print(f"  D {k:2d}-{k + 5:2d}: {v[len(v) // 2]:6.1f} s over {len(v):3d} points (max {v[-1]:.0f})")
    print(f"  all: {len(large)} points, {sum(secs) / 3600:.1f} h of solves")
    height = footnote(svg, ["hp/data/solve_times.csv (plot_solve_times.py --export)"], "wall-clock seconds of each solve",
                      ["machines and thread counts differ (cloud 2 shared cores; Windows 4-6 threads; Mac 4 threads), "
                       "and the solver settings changed between runs"], yb + h2 + 46, repo_url=REPO)
    with open(os.path.join(HERE, "solve_times.svg"), "w") as fh:
        fh.write(svg.finish(height))


def main():
    if "--export" in sys.argv:
        i = sys.argv.index("--export")
        export(sys.argv[i + 1] if len(sys.argv) > i + 1 else "/home/claude/dss/hp_scan")
    plot()


if __name__ == "__main__":
    main()
