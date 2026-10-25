#!/usr/bin/env python3
"""point_spacing.svg: how far apart in D our solved points are. Low D (3 < D <= 4): the distance from each value of
data/delta_dd_lowD.csv to the next one below it, on the grid the runs chose. Large D: each continuation run's own
step, the distance from each point to the run's previous point, with the runs kept apart: the runs of
data/delta_dd.csv (up to D = 61.03; its points grouped by run), the cloud pilot (data/large_d/pilot_cloud/pilotB.jsonl)
and T48 (data/large_d/up70.jsonl, split where a restart goes back in D). Points of different runs that overlap in D are
not compared, since their distance is not a step. T48 points after a horizon-tail jump are open rings. Standard
library only; the primitives are data_graphs.py, vendored unchanged from github.com/nmoinvaz/speedy-gonzales
(skills/graph-data). Prints the plotted numbers.

    cd hp && python3 plot_point_spacing.py
"""
import collections
import csv
import json
import math
import os

from data_graphs import INK, SERIES, Svg, axes, callout, caption, esc, footnote, legend

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/nmoinvaz/choptuik-large-d"
WIDTH, X, W = 1080, 80, 952
MACHINE = {"cloud": "cloud", "win": "Windows i7-1185G7", "mac": "Mac, Apple M5"}
# T48 points after a horizon-tail jump (left out when T48 goes into delta_dd.csv), and its ks 200 points from 67.767
JUMPED = {67.4167, 67.4267, 67.7625, 67.7708, 67.7762, 67.7899}


def nice_lo(v):
    e = 10 ** math.floor(math.log10(v))
    return max(m * e for m in (1, 2, 5) if m * e <= v)


def nice_hi(v):
    e = 10 ** math.floor(math.log10(v))
    return min(m * e for m in (1, 2, 5, 10) if m * e >= v)


def lowD_points():
    """D values of the low-D table (3 < D <= 4), sorted"""
    return sorted({round(float(r["D"]), 6) for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_dd.csv")))
                   if float(r["D"]) <= 4.0001})


def steps():
    """(D, step, machine, run, jumped) for each point of each large-D run that has a previous point in its run"""
    out = []
    groups = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_dd.csv"))):
        if float(r["D"]) > 4.0001 and not r["run"].startswith("T48-"):   # T48's rows: read from its own files below
            groups[r["run"]].append(round(float(r["D"]), 6))
    for run, Ds in groups.items():
        m = "win" if run.startswith("local-") else "cloud"
        Ds = sorted(set(Ds))
        out += [(D, D - prev, m, run, False) for prev, D in zip(Ds, Ds[1:])]
    for f, m, run in (("pilot_cloud/pilotB.jsonl", "cloud", "pilotB"), ("up70.jsonl", "mac", "up70")):
        prev, seg = None, 0
        for line in open(os.path.join(HERE, "data", "large_d", f)):
            if not line.strip():
                continue
            r = json.loads(line)
            D = round(r["D"], 6)
            jumped = round(D, 4) in JUMPED or (r.get("ks") == 200 and D >= 67.767)
            if prev is None or D <= prev:
                seg += 1                       # a (re)start: the run went back in D
            else:
                out.append((D, D - prev, m, f"{run} part {seg}, ks {r.get('ks')}", jumped))
            prev = D
    return sorted(out)


def main():
    lowD = lowD_points()
    low = [(D, D - prev, "low", "lowD", False) for prev, D in zip(lowD, lowD[1:])]
    large = steps()
    first = min(g[0] - g[1] for g in large)       # the first large-D point (no double-double points from 4 up to it)

    svg = Svg(WIDTH)
    svg.text(16, 28, "How far apart in D our solved points are", size=15, fill=INK, weight="bold")

    cl = SERIES[0]
    y = caption(svg, X, 58, "3 < D ≤ 4: distance to the next point below",
                "The low-D values: 0.05 apart through most of 3.15-3.7, with extra points at 3.4-3.5 and 3.7-3.8, then "
                "0.0125 and 0.00625 toward D = 3.")
    h = 240
    gl = [g[1] for g in low]
    xmap, ymap = axes(svg, X, y, W, h, (3.0, 4.0), (nice_lo(min(gl)), nice_hi(max(gl)), "log"),
                      "dimension D", "ΔD to the next point (log scale)", xfmt=lambda v: f"{v:.1f}",
                      yfmt=lambda v: f"{v:g}")
    print("low D: D, distance to the next point below")
    for D, g, m, run, _ in low:
        svg.add(f'<circle cx="{xmap(D):.1f}" cy="{ymap(g):.1f}" r="4" fill="{cl}">'
                f'<title>{esc(f"D = {D:g}: {g:.5g} from D = {D - g:g} (run {run})")}</title></circle>')
        print(f"  {D:8g}  {g:.5g}")
    lo_pt = min(low)
    callout(svg, xmap(lo_pt[0]), ymap(lo_pt[1]), f"steps of {lo_pt[1]:g} down to D = {lo_pt[0] - lo_pt[1]:g}, "
            "the lowest point", dx=34, dy=-26)

    y2 = y + h + 92
    color = {"cloud": SERIES[2], "win": SERIES[3], "mac": SERIES[4]}
    legend(svg, WIDTH - 16, y2, [(MACHINE[m], color[m]) for m in ("cloud", "win", "mac")])
    dmin, dmax = first, max(g[0] for g in large)
    hole_txt = f"none between 4 and {first:g}"
    yb = caption(svg, X, y2, f"{dmin:g} ≤ D ≤ {dmax:.2f}: each continuation run's step",
                 f"The distance from each point to its run's previous point ({hole_txt}); runs that overlap are kept "
                 f"apart. Open rings: T48 points after a horizon-tail jump. Hover a point for its run.")
    h2 = 320
    gs = [g[1] for g in large]
    xm2, ym2 = axes(svg, X, yb, W, h2, (math.floor(dmin), math.ceil(dmax)), (nice_lo(min(gs)), nice_hi(max(gs)), "log"),
                    "dimension D", "step in D (log scale)", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    # Over a thousand points: small dots, each with its hover title, instead of 11px ringed markers
    for D, g, m, run, jumped in large:
        tip = esc(f"D = {D:.4f}: a step of {g:.4g} from D = {D - g:.4f} ({MACHINE[m]}, run {run})"
                  + (", after a horizon-tail jump" if jumped else ""))
        fill = f'fill="none" stroke="{color[m]}" stroke-width="1.3"' if jumped else f'fill="{color[m]}"'
        svg.add(f'<circle cx="{xm2(D):.1f}" cy="{ym2(g):.1f}" r="{3.2 if jumped else 2.4}" {fill}>'
                f'<title>{tip}</title></circle>')
    big = max(large, key=lambda g: g[1])
    callout(svg, xm2(big[0]), ym2(big[1]), f"D = {big[0]:.2f}: the largest step, {big[1]:.3f}", dx=30, dy=4)
    for lo, hi, label, dx, dy in ((35.0, 35.5, "a cloud run's steps shrank to {g:.4f} before it stopped", -30, 22),
                                  (60.9, 61.1, "a stall like 67.78's, cured by holding the near-free harmonics", -40, 14)):
        g = min((g for g in large if lo <= g[0] <= hi), key=lambda g: g[1])
        callout(svg, xm2(g[0]), ym2(g[1]), f"D = {g[0]:.2f}: " + label.format(g=g[1]), dx=dx, dy=dy)
    stop = max(large, key=lambda g: g[0])
    callout(svg, xm2(stop[0]), ym2(stop[1]), f"T48 ends at D = {stop[0]:.2f}: near 67.78 its steps shrank, then failed",
            dx=-160, dy=60)
    med = collections.defaultdict(list)
    for D, g, *_ in large:
        med[int(D // 5) * 5].append(g)
    print("\nlarge D: median step by 5 units of D")
    for k in sorted(med):
        v = sorted(med[k])
        print(f"  D {k:2d}-{k + 5:2d}: median {v[len(v) // 2]:.4f}, min {v[0]:.5f}, max {v[-1]:.4f} over {len(v)} points")
    print(f"  {len(large)} steps; {sum(1 for g in large if g[4])} after a tail jump; " + hole_txt)
    height = footnote(svg, ["hp/data/delta_dd.csv, delta_dd_lowD.csv", "hp/data/large_d/up70.jsonl, pilot_cloud/pilotB.jsonl"],
                      "low D: distance to the next point below; large D: each run's step",
                      ["the large-D runs used different settings and machines; their steps are not comparable as "
                       "a benchmark"], yb + h2 + 46, repo_url=REPO)
    with open(os.path.join(HERE, "point_spacing.svg"), "w") as fh:
        fh.write(svg.finish(height))


if __name__ == "__main__":
    main()
