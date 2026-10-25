#!/usr/bin/env python3
"""point_spacing.svg: the distance in D from each solved point to the next one below it. Low D (3 < D <= 4): the
values of data/delta_dd_lowD.csv, on the grid the runs chose (halved toward D = 3). Large D: every double-double
point, data/delta_dd.csv (the runs up to D = 61.03) with the cloud pilot (data/large_d/pilot_cloud/pilotB.jsonl) and
T48 (data/large_d/up70.jsonl); there the spacing is the continuation's adaptive step, and where two runs overlap the
distance between their points can be much smaller than either run's step. Points after a horizon-tail jump (T48) are
open rings. Standard library only; the primitives are data_graphs.py, vendored unchanged from
github.com/nmoinvaz/speedy-gonzales (skills/graph-data). Prints the plotted numbers.

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


def points():
    """{D: (machine, run, jumped)} over every solved point, first source wins"""
    pts = {}
    for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_dd.csv"))):
        run = r["run"]
        m = "low" if run.startswith("lowD") else "win" if run.startswith("local-") else "cloud"
        pts[round(float(r["D"]), 6)] = (m, run, False)
    for f, m, run in (("pilot_cloud/pilotB.jsonl", "cloud", "pilotB"), ("up70.jsonl", "mac", "up70")):
        for line in open(os.path.join(HERE, "data", "large_d", f)):
            if not line.strip():
                continue
            r = json.loads(line)
            D = round(r["D"], 6)
            jumped = round(D, 4) in JUMPED or (r.get("ks") == 200 and D >= 67.767)
            if D not in pts:
                pts[D] = (m, run, jumped)
            elif jumped:
                pts[D] = pts[D][:2] + (True,)
    return dict(sorted(pts.items()))


def main():
    pts = points()
    Ds = list(pts)
    gaps = [(D, D - prev, *pts[D]) for prev, D in zip(Ds, Ds[1:])]
    low = [g for g in gaps if g[0] <= 4.0001]
    large = [g for g in gaps if g[0] > 4.0001 and g[1] < 4]      # the 4 -> 8.25 hole is stated in the caption
    hole = [g for g in gaps if g[1] >= 4]

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
    dmin, dmax = (hole[0][0] if hole else min(g[0] for g in large)), max(g[0] for g in large)
    hole_txt = "; ".join(f"none between {D - g:g} and {D:g}" for D, g, *_ in hole)
    yb = caption(svg, X, y2, f"{dmin:g} ≤ D ≤ {dmax:.2f}: distance to the next point below",
                 f"Each continuation run's adaptive step ({hole_txt}). Very small distances are where two runs overlap. "
                 f"Open rings: T48 points after a horizon-tail jump. Hover a point for its run.")
    h2 = 320
    gs = [g[1] for g in large]
    xm2, ym2 = axes(svg, X, yb, W, h2, (math.floor(dmin), math.ceil(dmax)), (nice_lo(min(gs)), nice_hi(max(gs)), "log"),
                    "dimension D", "ΔD to the next point (log scale)", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    # Over a thousand points: small dots, each with its hover title, instead of 11px ringed markers
    for D, g, m, run, jumped in large:
        tip = esc(f"D = {D:.4f}: {g:.4g} from D = {D - g:.4f} ({MACHINE[m]}, run {run})"
                  + (", after a horizon-tail jump" if jumped else ""))
        fill = f'fill="none" stroke="{color[m]}" stroke-width="1.3"' if jumped else f'fill="{color[m]}"'
        svg.add(f'<circle cx="{xm2(D):.1f}" cy="{ym2(g):.1f}" r="{3.2 if jumped else 2.4}" {fill}>'
                f'<title>{tip}</title></circle>')
    big = max(large, key=lambda g: g[1])
    callout(svg, xm2(big[0]), ym2(big[1]), f"D = {big[0]:.2f}: the largest step, {big[1]:.3f}", dx=30, dy=4)
    stop = max(large, key=lambda g: g[0])
    callout(svg, xm2(stop[0]), ym2(stop[1]), f"T48 ends at D = {stop[0]:.2f}: near 67.78 its steps shrank, then failed",
            dx=-160, dy=60)
    med = collections.defaultdict(list)
    for D, g, *_ in large:
        med[int(D // 5) * 5].append(g)
    print("\nlarge D: median distance by 5 units of D")
    for k in sorted(med):
        v = sorted(med[k])
        print(f"  D {k:2d}-{k + 5:2d}: median {v[len(v) // 2]:.4f}, min {v[0]:.5f}, max {v[-1]:.4f} over {len(v)} points")
    print(f"  {len(large)} distances; {sum(1 for g in large if g[4])} after a tail jump; " + hole_txt)
    height = footnote(svg, ["hp/data/delta_dd.csv, delta_dd_lowD.csv", "hp/data/large_d/up70.jsonl, pilot_cloud/pilotB.jsonl"],
                      "distance in D from each point to the next solved point below it",
                      ["the large-D runs used different settings and machines; their steps are not comparable as "
                       "a benchmark"], yb + h2 + 46, repo_url=REPO)
    with open(os.path.join(HERE, "point_spacing.svg"), "w") as fh:
        fh.write(svg.finish(height))


if __name__ == "__main__":
    main()
