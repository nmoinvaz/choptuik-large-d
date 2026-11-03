#!/usr/bin/env python3
"""Delta against s = sqrt(D - 3): the approach to D = 3, where gravity's reach r^-(D-3) vanishes, drawn in the
variable the low-D formula uses (README, "A formula for 3 < D <= 4"; hp/lowD_formula.py). A square-root branch point
at D = 3 is a curve with a finite intercept at s = 0 (the fits put Delta(3) between 1.72 and 1.80, degree-dependent);
Ecker, Ecker, Grumiller & Jechtl's conjecture, Delta -> 0, bends toward the origin. The two readings separate below
s = 0.2, where T100's planned points land (the shaded band; comms entry 2026-11-03 02:15 UTC).

Data: data/delta_dd_lowD.csv (hp/export_lowD.py), the largest clean M at each D, filled where resolved (the two
largest M agree to 1e-8, plot_lowD_values.py's rule) and a diamond where not yet.

    cd hp && python3 export_lowD.py && python3 plot_lowD_sqrt.py [-o lowD_sqrt.svg]
"""
import argparse
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_graphs import GRID, INK, INK_SOFT, SERIES, Svg, axes, better_hint, callout, caption, footnote, host_specs, legend, marker  # noqa: E402,E501
from plot_lowD_values import C_DELTA, formula  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
INTERCEPT = (1.72, 1.80)              # Delta(3) over the fit degrees (README, "A formula for 3 < D <= 4")
T100_TARGETS = (3.005, 3.044375)      # the quarter-step leg's targets (comms 2026-11-03 02:15 UTC cloud to win2)


def points():
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_dd_lowD.csv"))) if r["status"] == "clean"]
    by_d = {}
    for r in rows:
        by_d.setdefault(float(r["D"]), {})[int(r["M"])] = r
    out = []
    for D, ms in sorted(by_d.items()):
        if D > 4.0:                    # the file holds every record below D = 8.25; the figure ends at D = 4
            continue
        top = max(ms)
        r = ms[top]
        out.append((D, float(r["Delta_hi"]) + float(r["Delta_lo"]), top, r["resolved"] == "1"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "lowD_sqrt.svg"))
    a = ap.parse_args()
    pts = points()
    svg = Svg(900)
    X, W, H = 70, 700, 380
    y = caption(svg, X, 40, "The approach to D = 3 in s = √(D − 3): fold, or Δ → 0?",
                "a square-root branch point keeps a finite Δ(3) with an infinite slope in D; the conjectured Δ → 0 bends to the origin")
    xm, ym = axes(svg, X, y, W, H, (0, 1.02), (0, 3.6), "s = √(D − 3)   (D = 3 at the left, D = 4 at the right)",
                  "echo period Δ", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    # where T100's planned points land
    s0, s1 = (math.sqrt(d - 3) for d in T100_TARGETS)
    svg.add(f'<rect x="{xm(s0):.1f}" y="{y:.1f}" width="{xm(s1) - xm(s0):.1f}" height="{H:.1f}" fill="{GRID}" opacity="0.7"/>')
    svg.text((xm(s0) + xm(s1)) / 2, y + 16, "T100's planned points", size=10, anchor="middle")
    # the formula
    ss = [i / 400 for i in range(0, 401)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{xm(s):.1f},{ym(formula(C_DELTA, 3 + s * s)):.1f}" for i, s in enumerate(ss))
    svg.add(f'<path d="{path}" fill="none" stroke="{SERIES[-1]}" stroke-width="1.8" stroke-dasharray="6 4"/>')
    # the intercept's spread over the fit degrees
    svg.line(xm(0.006), ym(INTERCEPT[0]), xm(0.006), ym(INTERCEPT[1]), SERIES[-1], width=4)
    svg.line(xm(0.006), ym(0), xm(0.006), ym(0.02), SERIES[-1], width=4)
    # the points
    lowest = None
    for D, v, M, ok in pts:
        s = math.sqrt(D - 3)
        marker(svg, "circle" if ok else "diamond", xm(s), ym(v), SERIES[0] if ok else SERIES[1],
               f"D = {D} (s = {s:.4f}): Δ = {v:.12f} at M = {M}, {'resolved' if ok else 'not yet resolved'}; "
               f"formula {formula(C_DELTA, D):.6f}")
        if ok and (lowest is None or D < lowest[0]):
            lowest = (D, v)
    D0, v0 = lowest
    callout(svg, xm(math.sqrt(D0 - 3)), ym(v0), f"lowest resolved: D = {D0}, Δ = {v0:.4f}", dx=60, dy=70)
    callout(svg, xm(0.006), ym(sum(INTERCEPT) / 2), f"fits: Δ(3) from {INTERCEPT[0]} to {INTERCEPT[1]}", dx=40, dy=-165)
    callout(svg, xm(0.006), ym(0.01), "conjectured (Ecker et al.): Δ → 0", dx=90, dy=-40)
    better_hint(svg, X, y, W, H, where="axis", label="toward D = 3", reverse=True)
    legend(svg, X + W, y - 10, [("resolved", SERIES[0]), ("not yet resolved", SERIES[1]), ("√(D − 3) formula", SERIES[-1])])
    srcs = ["hp/data/delta_dd_lowD.csv (hp/export_lowD.py; T82-win and lowD_highM records)",
            "formula: hp/lowD_formula.py on Ecker, Ecker, Grumiller & Jechtl's values"]
    height = footnote(svg, srcs, host_specs(), [], y + H + 50, REPO)
    open(a.o, "w").write(svg.finish(height))
    print("wrote", a.o)
    print("| D | s | Δ | M | resolved | formula − Δ |")
    print("|---|---|---|---|---|---|")
    for D, v, M, ok in pts:
        print(f"| {D} | {math.sqrt(D - 3):.4f} | {v:.12f} | {M} | {'yes' if ok else 'no'} | {formula(C_DELTA, D) - v:+.1e} |")


if __name__ == "__main__":
    main()
