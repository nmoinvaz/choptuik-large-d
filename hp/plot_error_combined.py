#!/usr/bin/env python3
"""The large-D chain's errors in Delta and the condition of its solve, as two panels over one D axis.

    cd hp && python3 plot_error_combined.py [-o .]

error_combined.svg, from D 60:
  top     the plot of error_sizes.svg (plot_error_methods.sizes_panel): our err column, each published value's
          distance from ours, the uncertainty each source states, the series' spread, the held-trend bias that
          chain_fit removed and chain_fit's own bias;
  bottom  the largest over the smallest pivot of the chain's Jacobian at each base (data/chain_pivots.csv, which
          plot_conditioning.py RAWDIR writes from the chain's logs).
The two have no common unit, an error in Delta and a condition number, so they share the D axis and not the vertical
one. Standard library only; the primitives are data_graphs.py. Prints the plotted numbers.
"""
import argparse
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_graphs as dg  # noqa: E402
from data_graphs import INK, SERIES, Svg, axes, better_arrow, callout, footnote, marker  # noqa: E402
from plot_error_methods import REPO, DecadeAxis, sizes_panel  # noqa: E402

XAXIS = (60, 240)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=HERE)
    a = ap.parse_args()
    piv = list(csv.DictReader(open(os.path.join(HERE, "data", "chain_pivots.csv"))))
    cond = [(float(r["D"]), float(r["pivot_max"]) / float(r["pivot_min"])) for r in piv if float(r["D"]) >= XAXIS[0]]

    W, x, w = 1040, 92, 700
    y1, h1, h2 = 112, 400, 250
    y2 = y1 + h1 + 96
    svg = Svg(W)
    svg.text(16, 24, "The large-D chain: the errors in Δ and the condition of its solve", size=14, fill=INK, weight="bold")
    svg.text(16, 42, "Top: measured differences in Δ and the uncertainty each source states. Bottom: the largest over "
             "the smallest pivot of the chain's Jacobian at each base.", size=11)
    svg.text(16, 58, "The two panels share the D axis. An error in Δ and a condition number have no common unit, so "
             "each has its own vertical axis.", size=11)
    svg.text(x, y1 - 12, "Errors in Δ", size=12, fill=INK, weight="bold")
    sizes_panel(svg, x, y1, w, h1, xaxis=XAXIS)

    svg.text(x, y2 - 12, "Condition of the solve", size=12, fill=INK, weight="bold")
    dg.Axis = DecadeAxis
    xm, ym = axes(svg, x, y2, w, h2, XAXIS, (1e14, 1e58, "log"), "spacetime dimension D",
                  "largest / smallest pivot (lower is better)", xfmt=lambda v: f"{v:g}",
                  yfmt=lambda v: f"1e{round(math.log10(v))}")
    svg.add('<polyline points="' + " ".join(f"{xm(D):.1f},{ym(v):.1f}" for D, v in cond)
            + f'" fill="none" stroke="{SERIES[1]}" stroke-width="2"/>')
    for D, v in [q for q in cond if q[0] % 10 == 0] + [cond[-1]]:
        marker(svg, "circle", xm(D), ym(v), SERIES[1], f"D {D:g}: largest / smallest pivot {v:.1e}")

    def at(D):
        return min(cond, key=lambda q: abs(q[0] - D))
    knee, wall, last = at(170.0), at(215.03125), cond[-1]
    callout(svg, xm(knee[0]), ym(knee[1]), "the smallest pivot starts to fall", dx=-30, dy=-34)
    callout(svg, xm(wall[0]), ym(wall[1]), "D 215.03, where the chain had stopped", dx=-40, dy=-26)
    callout(svg, xm(last[0]), ym(last[1]), f"D {last[0]:.2f}, the chain's last base", dx=-36, dy=-20)
    better_arrow(svg, x + w + 22, y2 + 6, x + w + 22, y2 + 126, label="better")
    lx, ly = x + w + 46, y2 + 150
    svg.line(lx - 2, ly - 4, lx + 12, ly - 4, SERIES[1], 2)
    marker(svg, "circle", lx + 5, ly - 4, SERIES[1], "largest / smallest pivot")
    svg.text(lx + 18, ly, "largest / smallest pivot", size=11, fill=INK)

    print("\n| condition of the solve | D | largest / smallest pivot |\n|---|---|---|")
    for q in (at(73.0), at(100.0), at(160.0), knee, at(200.0), wall, last):
        print(f"| chain base | {q[0]:g} | {q[1]:.1e} |")
    hgt = footnote(svg, ["errors: hp/data/delta_all.csv, delta_published_compare.csv, published_vs_series.txt, "
                         "chain_fit_shift.txt, release_steps.csv",
                         f"pivots: hp/data/chain_pivots.csv (raw {piv[0]['raw_commit']})"],
                   "chain run on win2, Intel Core Ultra 7 265K, 18 threads", [], y2 + h2 + 44, REPO)
    out = os.path.join(a.o, "error_combined.svg")
    open(out, "w").write(svg.finish(hgt))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
