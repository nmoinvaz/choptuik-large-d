#!/usr/bin/env python3
"""Graph how well each large-D model predicts double-double points it never saw.

Two panels: the models fitted on 10 <= D <= 37.5 and on 10 <= D <= 45 (cleaned points, dd_fits_mp.py), and
|model - computed Delta| on every point above the fit range, on a log axis. The models: power series in 1/D (degree
16) and in 1/(D-2) (degree 18), which are analytic in 1/D at D = infinity, and power series in (D-2)^(-1/2) and
D^(-1/2) (degree 20), which are not. All four fit their range to the level of the points (shaded).

    python3 plot_heldout.py [-o heldout.svg]
"""
import argparse
import math
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "solver"))
from dd_fits_mp import clean, fit, load_points  # noqa: E402
from plot_delta import BAND, GRID, INK, INK_SOFT, SERIES, Svg  # noqa: E402

MODELS = [("1/D", 16, "power series in 1/D, degree 16", SERIES[0]),
          ("1/n", 18, "power series in 1/(D−2), degree 18", SERIES[2]),
          ("n^-1/2", 20, "power series in (D−2)^−1/2, degree 20", SERIES[1]),
          ("D^-1/2", 20, "power series in D^−1/2, degree 20", SERIES[3])]
CUTS = (37.5, 45.0)
LY0, LY1 = -16, -8
FLOOR = 4e-15          # the fits' residual on their own range: the level of the points


def sup(e):
    return "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(ch)] if ch.isdigit() else "⁻" for ch in str(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", default=os.path.join(HERE, "heldout.svg"))
    args = ap.parse_args()
    pts, _ = load_points()
    Ds = clean(pts, sorted(d for d in pts if d >= 10), verbose=False)
    Dmax = Ds[-1]
    X0, X1 = 35.0, math.ceil(Dmax)

    W, px, pw, ph = 760, 80, 520, 230
    svg = Svg(W)
    svg.text(px, 28, f"Predicting double-double points the fits never saw (computed up to D = {Dmax:.1f})",
             size=14, fill=INK, weight="bold")
    svg.text(px, 48, "|model − computed Δ| on every point above the fitted range; shaded: the level of the points",
             size=11)
    y = 90
    for cut in CUTS:
        fitD = [d for d in Ds if d <= cut]
        hold = [d for d in Ds if d > cut]
        py = y + 24
        sx = lambda D: px + (D - X0) / (X1 - X0) * pw  # noqa: E731
        sy = lambda v: py + (LY1 - math.log10(max(v, 10.0 ** LY0))) / (LY1 - LY0) * ph  # noqa: E731
        svg.text(px, y + 8, f"Fitted on 10 ≤ D ≤ {cut:g} ({len(fitD)} points), compared with {len(hold)} points above",
                 size=12, fill=INK)
        svg.add(f'<rect x="{px:.1f}" y="{sy(FLOOR):.1f}" width="{pw:.1f}" height="{py + ph - sy(FLOOR):.1f}" '
                f'fill="{BAND}"/>')
        for e in range(LY0, LY1 + 1, 2):
            yy = sy(10.0 ** e)
            svg.line(px, yy, px + pw, yy, GRID)
            svg.text(px - 8, yy + 4, f"10{sup(e)}", size=10, anchor="end")
        for D in range(int(X0), int(X1) + 1, 5):
            xx = sx(D)
            svg.line(xx, py, xx, py + ph, GRID)
            svg.text(xx, py + ph + 16, f"{D}", size=10, anchor="middle")
        svg.line(sx(cut), py, sx(cut), py + ph, INK_SOFT, dash="3 3")
        svg.text(sx(cut) + 5, py + 12, "fit range ends", size=10)
        svg.line(px, py + ph, px + pw, py + ph, INK_SOFT)
        svg.text(px + pw / 2, py + ph + 34, "D", size=11, anchor="middle")
        ends = []
        for model, deg, label, color in MODELS:
            ev, res, _ = fit(pts, fitD, model, deg)
            err = [abs(ev(d) - pts[d]) for d in hold]
            path = [(sx(d), sy(float(e))) for d, e in zip(hold, err)]
            worst = float(max(err))
            svg.path(path, color, width=2, opacity=0.9,
                     title=f"{label}: fit residual {float(res):.1e}, worst on the held-out points {worst:.1e}")
            ends.append([path[-1][1], label.replace("power series in ", "").split(",")[0], color, worst])
        ends.sort()
        for i in range(1, len(ends)):                      # keep the end labels 13 px apart
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
        for yy, short, color, worst in ends:
            svg.add(f'<circle cx="{px + pw + 6:.1f}" cy="{yy - 4:.1f}" r="3" fill="{color}"/>')
            svg.text(px + pw + 13, yy, f"{short}  ({worst:.0e})", size=10, fill=INK)
        y = py + ph + 64
    # legend
    for i, (model, deg, label, color) in enumerate(MODELS):
        lx, ly = px + (i % 2) * 330, y + (i // 2) * 18
        svg.line(lx, ly - 4, lx + 22, ly - 4, color, width=2)
        svg.text(lx + 28, ly, label + (" (analytic in 1/D)" if model in ("1/D", "1/n") else ""), size=11)
    y += 40
    svg.text(px, y, "Regenerate: python3 hp/plot_heldout.py", size=10)
    open(args.output, "w").write(svg.finish(int(y + 16)))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
