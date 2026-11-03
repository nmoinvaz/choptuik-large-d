#!/usr/bin/env python3
"""large_d_dropoffs.svg: the drop-offs of the continuation's step in D, and the precision wall that sets them.

Top: each large-D run's step in D (the records of plot_point_spacing.py: data/delta_dd.csv, the cloud pilot and T48),
with the stalls named by their cause. Bottom: u·κ, double-double's unit roundoff u = 2^-104 times the Jacobian's
condition number at converged states (hp/stall_margin.py's runs in folder T76 of the raw-data repository), for the
full system with harmonics to 13 solved and for the pinned systems (p1's harmonics above 9, 7 or 5 held), the full
system's power-law fit in n = D - 2 (README, "Why the solver stalls": n^38.9, reaching 1 at D = 59.9, the stall at
60.96), and the crossings of 1 the same slope predicts for the pinned systems from their last measured points, and
for quad-double (u = 2^-208) for the full system. A wall of this kind moves when the arithmetic or the harmonic set
changes; the walls at 73.27 and 78.1 (notes/QUAD_DOUBLE.md 12.4, 12.7) are the horizon block's, not precision.
Prints the numbers and writes data/dropoffs.json (what the README quotes). Standard library and numpy; the primitives
are data_graphs.py.

    git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw   (beside the repository, once)
    cd hp && python3 plot_dropoffs.py ../../choptuik-large-d-raw/T76
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

from data_graphs import GRID, INK, INK_SOFT, SERIES, Svg, axes, better_hint, callout, caption, footnote, host_specs, legend, marker
from plot_newton_stall import polyline
from plot_point_spacing import steps

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/nmoinvaz/choptuik-large-d"
WIDTH, X, W = 1080, 90, 900
U_DD, U_QD = 2.0 ** -104, 2.0 ** -208
PINS = ((None, "full system (harmonics to 13 solved)"), (9, "pin 9 (11-13 held)"), (7, "pin 7 (9-13 held)"), (5, "pin 5 (7-13 held)"))
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pow10(v):
    return "10" + str(int(round(v))).translate(SUP)


def jl(path):
    return [json.loads(x) for x in open(path) if x.strip()]


def main():
    if len(sys.argv) < 2 or not os.path.isfile(os.path.join(sys.argv[1], "kappa_states.jsonl")):
        sys.exit(__doc__)
    raw = sys.argv[1]
    rev = subprocess.run(["git", "-C", raw, "log", "-1", "--format=%h", "--", ".", ":(exclude)README.md"], capture_output=True, text=True).stdout.strip()
    recs = [r for f in ("kappa_states.jsonl", "smoke_kappa.jsonl", "kappa_pinned.jsonl", "t75_watch.jsonl") for r in jl(os.path.join(raw, f))]
    pts = {}
    for r in recs:
        pts[(r["pin"], round(r["D"], 4))] = dict(D=r["D"], pin=r["pin"], uk=r["dirs"][0]["u_kappa"])
    by = {p: sorted((v for v in pts.values() if v["pin"] == p), key=lambda v: v["D"]) for p, _ in PINS}
    fit = [v for v in by[None] if v["D"] <= 67.8]
    b, a = np.polyfit(np.log([v["D"] - 2 for v in fit]), np.log([v["uk"] for v in fit]), 1)
    cross_dd = math.exp(-a / b) + 2
    cross_qd = math.exp((math.log(U_DD / U_QD) - a) / b) + 2
    pred = {}
    for p in (9, 7, 5):
        last = by[p][-1]
        pred[p] = dict(D_from=last["D"], uk=last["uk"], D_cross=2 + (last["D"] - 2) * (1 / last["uk"]) ** (1 / b))
    walls = {"35.2": "every harmonic kept, double-double", "60.96": "harmonics to 13, double-double", "67.78": "harmonics held on their trend past the wall"}
    # the states with every harmonic solved around the 35.2 stall (T101-mac/stall_up6.jsonl: u kappa, the basin measure
    # kappa = c F0 / sigma^2 along the weakest direction, |F|), when the raw-data repository is beside T76
    up6 = os.path.join(os.path.dirname(os.path.abspath(raw)), "T101-mac", "stall_up6.jsonl")
    allh = [dict(D=r["D"], uk=r["dirs"][0]["u_kappa"], kappa=r["dirs"][0]["kappa"], F=r["F"]) for r in (jl(up6) if os.path.isfile(up6) else [])
            if r.get("kfreeze") is None]
    out = dict(slope=b, cross_dd=cross_dd, cross_qd=cross_qd, pinned=pred, u_kappa_67_74=[v["uk"] for v in by[None] if abs(v["D"] - 67.7422) < 1e-3][0],
               u_kappa_60_97=[v["uk"] for v in by[None] if abs(v["D"] - 60.9656) < 1e-3][0], all_harmonics=allh)
    for v in allh:
        print(f"every harmonic solved, D {v['D']:.2f}: u*kappa {v['uk']:.1e}, basin kappa {v['kappa']:.1e}, |F| {v['F']:.1e}")
    print(f"full system: u*kappa ~ n^{b:.1f}, = 1 at D {cross_dd:.1f} (double-double), at D {cross_qd:.0f} for quad-double")
    for p, v in pred.items():
        print(f"pin {p}: u*kappa {v['uk']:.2e} at D {v['D_from']:.2f}; with the same slope, 1 at D {v['D_cross']:.1f}")
    json.dump(out, open(os.path.join(HERE, "data", "dropoffs.json"), "w"), indent=1)

    svg = Svg(WIDTH)
    svg.text(16, 28, "The drop-offs of the continuation's step, and the precision wall behind them", size=15, fill=INK, weight="bold")
    # top: the steps
    large = [t for t in steps() if 8 <= t[0] <= 80]
    y = caption(svg, X, 58, "Each run's step in D, 8 ≤ D ≤ 68 (as in the point-spacing chart)",
                "The step collapses at a wall: the junk floor with every harmonic kept (35.2), precision, u·κ near 1, with harmonics to 13 (60.96). The other dips are not walls.")
    H1 = 250
    xm, ym = axes(svg, X, y, W, H1, (8, 80), (-4, 0.5), "dimension D", "step in D", xfmt=lambda v: f"{v:g}", yfmt=pow10)
    for D, g, m, run, jumped in large:
        if g > 0:
            svg.add(f'<circle cx="{xm(D):.1f}" cy="{ym(math.log10(g)):.1f}" r="2.6" fill="{SERIES[0]}" opacity="0.75"><title>D = {D:.4f}: a step of {g:.4g} (run {run})</title></circle>')
    callout(svg, xm(35.2), ym(math.log10(0.00235)), "35.2: the junk floor, every harmonic kept (u·κ 0.01)", dx=60, dy=28)
    callout(svg, xm(60.96), ym(math.log10(0.00086)), f"60.96: harmonics to 13; u·κ = 1 predicted at {cross_dd:.1f}", dx=-60, dy=-44)
    callout(svg, xm(67.75), ym(math.log10(0.0001)), "67.78: past the wall, harmonics held on their trend", dx=-40, dy=-30)
    svg.text(xm(45.34), ym(math.log10(0.0055)) - 24, "45.3: a run stopped, no wall", size=10, anchor="middle")
    svg.text(xm(20), ym(math.log10(0.01)) - 12, "16-27: the f64 solver's own limits", size=10, anchor="middle")
    better_hint(svg, X, y, W, H1, where="right", label="longer steps")
    # bottom: u kappa
    y2 = caption(svg, X, y + H1 + 60, "u·κ at converged states: double-double's wall is u·κ = 1",
                 f"Dashed: the fit n^{b:.1f} and the same slope through each pinned system's last point. Quad-double reaches 1 at D {cross_qd:.0f}.")
    H2 = 300
    xm2, ym2 = axes(svg, X, y2, W, H2, (30, 130), (-9, 4), "dimension D", "u·κ (log scale)", xfmt=lambda v: f"{v:g}", yfmt=pow10)
    svg.line(xm2(30), ym2(0), xm2(130), ym2(0), INK, 1.2)
    svg.text(xm2(130) - 4, ym2(0) - 5, "u·κ = 1: double-double's wall", size=10, anchor="end")
    polyline(svg, [(xm2(d), ym2((a + b * math.log(d - 2)) / math.log(10))) for d in range(40, 131)
                   if -9 <= (a + b * math.log(d - 2)) / math.log(10) <= 4], SERIES[0], 1.4, "6 4")
    for p, v in pred.items():
        col = SERIES[[q for q, _ in PINS].index(p)]
        a_p = math.log(v["uk"]) - b * math.log(v["D_from"] - 2)
        polyline(svg, [(xm2(d), ym2((a_p + b * math.log(d - 2)) / math.log(10))) for d in range(int(v["D_from"]), 131)
                       if -9 <= (a_p + b * math.log(d - 2)) / math.log(10) <= 4], col, 1.4, "6 4")
        marker(svg, "triangle", xm2(v["D_cross"]), ym2(0), col, f"pin {p}: u·κ = 1 predicted at D {v['D_cross']:.1f}")
    for v in allh:
        marker(svg, "square", xm2(v["D"]), ym2(math.log10(v["uk"])), SERIES[-1], f"every harmonic solved, D = {v['D']:.4f}: u·κ {v['uk']:.1e}, basin κ {v['kappa']:.1e}, |F| {v['F']:.1e}")
    for i, (p, lab) in enumerate(PINS):
        for v in by[p]:
            marker(svg, "circle", xm2(v["D"]), ym2(math.log10(v["uk"])), SERIES[i], f"{lab}, D = {v['D']:.4f}: u·κ {v['uk']:.2e}")
    callout(svg, xm2(cross_dd), ym2(0), f"1 at D {cross_dd:.1f}, stall at 60.96", dx=14, dy=30)
    callout(svg, xm2(pred[5]["D_cross"]), ym2(0), f"pin 5 would run out near D {pred[5]['D_cross']:.0f}", dx=-30, dy=80)
    legend(svg, X + W - 8, y2 + H2 - 8, [(lab, SERIES[i]) for i, (_, lab) in enumerate(PINS)] + [("every harmonic solved", SERIES[-1])])
    better_hint(svg, X, y2, W, H2, where="right", label="precision", reverse=True)
    height = footnote(svg, ["steps: data/delta_dd.csv, data/large_d (plot_point_spacing.py)", "u·κ: hp/stall_margin.py runs in T76 " + (rev or "?")],
                      host_specs(), ["the pinned systems' crossings assume the full system's slope; their runs stopped earlier at the horizon block"],
                      y2 + H2 + 50, REPO)
    with open(os.path.join(HERE, "large_d_dropoffs.svg"), "w") as fh:
        fh.write(svg.finish(height))
    print("wrote large_d_dropoffs.svg and data/dropoffs.json")


if __name__ == "__main__":
    main()
