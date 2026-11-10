#!/usr/bin/env python3
"""How fast the chain's error bar on Delta and the difficulty of its solve grow with D, as two lines on one graph.

    cd hp && python3 plot_conditioning.py [RAWDIR] [-o .]

conditioning_vs_error.svg draws, against D and as factors of their values at D 100:
  the error bar on Delta   the err column of data/delta_all.csv for the chain (kind chain_fit, err_kind fits, the
                           spread over the held fit's degree), the largest in each bin of 8 in D;
  the Jacobian's condition the largest over the smallest pivot of the LU at each base of the chain, as its logs print
                           them ("step k: D ..., pivots a..b").
With RAWDIR, a checkout of the raw-data repository, the pivots are read again from the chain's logs (LOGS) and
data/chain_pivots.csv is rewritten with the checkout's commit. Without it the stored table is drawn. The slopes are
least-squares fits of log10 against D over the stated ranges. Standard library only; the primitives are
data_graphs.py. Prints the plotted numbers.
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_graphs as dg  # noqa: E402
from data_graphs import INK, INK_SOFT, SERIES, Svg, axes, better_hint, callout, footnote, marker  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
PIVOTS = os.path.join(HERE, "data", "chain_pivots.csv")
# the chain's logs in the order of its backbone; a D printed by two of them keeps the first
LOGS = ["T138-win2/chain.log", "T138-win2/chain_o.log", "I19-win2/crawl.log", "I461-win2/crawlB2.log",
        "I461-win2/crawlB3.log", "I461-win2/crawlB4.log", "I461-win2/crawlB5.log", "I461-win2/crawlC.log"]
STEP = re.compile(r"^step \d+: D ([0-9.]+), pivots ([0-9.e+-]+)\.\.([0-9.e+-]+),")
D0 = 100.0   # both lines are 1 here
BIN = 8.0


def export(raw):
    """data/chain_pivots.csv from the logs of raw"""
    commit = subprocess.run(["git", "-C", raw, "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    seen = {}
    for log in LOGS:
        for line in open(os.path.join(raw, log), errors="replace"):
            m = STEP.match(line)
            if m and float(m.group(1)) not in seen:
                seen[float(m.group(1))] = (m.group(1), m.group(2), m.group(3), log)
    with open(PIVOTS, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["D", "pivot_min", "pivot_max", "log", "raw_commit"])
        for D in sorted(seen):
            w.writerow([*seen[D], commit])
    print(f"wrote {PIVOTS}: {len(seen)} bases, D {min(seen):g} to {max(seen):g}, raw {commit}")


def slope(pts, lo, hi):
    """least-squares slope of log10(value) against D over lo <= D <= hi, in decades per unit of D"""
    p = [(D, math.log10(v)) for D, v in pts if lo <= D <= hi]
    mx, my = sum(x for x, _ in p) / len(p), sum(y for _, y in p) / len(p)
    return sum((x - mx) * (y - my) for x, y in p) / sum((x - mx) ** 2 for x, _ in p)


def at(pts, D):
    """the value of the point nearest D"""
    return min(pts, key=lambda q: abs(q[0] - D))[1]


class DecadeAxis(dg.Axis):
    """a log axis over many decades ticks every fourth decade"""

    def ticks(self):
        if self.log and math.log10(self.hi / self.lo) > 8:
            return [10.0 ** e for e in range(math.ceil(math.log10(self.lo)), math.floor(math.log10(self.hi)) + 1)
                    if e % 4 == 0]
        return super().ticks()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?")
    ap.add_argument("-o", default=HERE)
    a = ap.parse_args()
    if a.raw:
        export(a.raw)
    piv = list(csv.DictReader(open(PIVOTS)))
    cond = [(float(r["D"]), float(r["pivot_max"]) / float(r["pivot_min"])) for r in piv]
    pmin = [(float(r["D"]), float(r["pivot_min"])) for r in piv]
    pmax = [(float(r["D"]), float(r["pivot_max"])) for r in piv]
    rows = csv.DictReader(open(os.path.join(HERE, "data", "delta_all.csv")))
    fits = [(float(r["D"]), float(r["err"])) for r in rows if r["kind"] == "chain_fit" and r["err_kind"] == "fits"]
    # the error bar: the largest spread in each bin, since rows that only one other fit reached show a part of it
    bins = {}
    for D, e in fits:
        k = math.floor(D / BIN)
        if e > bins.get(k, (0, 0))[1]:
            bins[k] = (D, e)
    err = [bins[k] for k in sorted(bins)]
    top, wall = cond[-1][0], 215.03125
    s_err = slope(err, D0, 1e9)
    s_lo, s_hi = slope(cond, D0, 160), slope(cond, wall, 1e9)
    print("| line | D | value | factor of its value at D 100 |\n|---|---|---|---|")
    for name, pts in (("error bar on Delta", err), ("Jacobian condition", cond), ("smallest pivot", pmin),
                      ("largest pivot", pmax)):
        for q in dict.fromkeys(min(pts, key=lambda q: abs(q[0] - D)) for D in (D0, 160.0, wall, top)):
            print(f"| {name} | {q[0]:g} | {q[1]:.1e} | {q[1] / at(pts, D0):.1e} |")
    print("\n| line | range of D | decades per unit of D | one decade per |\n|---|---|---|---|")
    for name, pts, lo, hi in (("error bar on Delta", err, D0, 1e9), ("Jacobian condition", cond, D0, 160),
                              ("Jacobian condition", cond, wall, 1e9), ("smallest pivot", pmin, D0, 160),
                              ("smallest pivot", pmin, wall, 1e9), ("largest pivot", pmax, D0, 1e9)):
        s = slope(pts, lo, hi)
        print(f"| {name} | {lo:g} to {min(hi, pts[-1][0]):g} | {s:+.4f} | {abs(1 / s):.1f} in D |")

    W, x, y, w, h = 1040, 84, 96, 880, 470
    svg = Svg(W)
    svg.text(16, 24, "The chain's error bar on Δ and the difficulty of its solve, against D", size=14, fill=INK,
             weight="bold")
    svg.text(16, 42, f"Each line is the factor by which the quantity has grown since D {D0:g}, so a steeper line "
             "grows faster.", size=11)
    svg.text(16, 58, "The error bar is the spread of Δ over the held fit's degree. The condition is the largest over "
             "the smallest pivot of the Jacobian at each base of the chain.", size=11)
    dg.Axis = DecadeAxis
    # the axes follow the data: D to the next multiple of 5 past the last base, the growth to the next fourth decade
    x_hi = 5 * math.ceil((top + 2) / 5)
    y_hi = 10.0 ** (4 * math.ceil((math.log10(cond[-1][1] / at(cond, D0)) + 1) / 4))
    xm, ym = axes(svg, x, y, w, h, (D0, x_hi), (0.1, y_hi, "log"), "spacetime dimension D",
                  f"factor of growth since D {D0:g} (lower is better)", xfmt=lambda v: f"{v:g}",
                  yfmt=lambda v: f"1e{round(math.log10(v))}")
    e0, c0 = at(err, D0), at(cond, D0)
    for pts, v0, col, name in ((cond, c0, SERIES[1], "condition"), (err, e0, SERIES[0], "error bar")):
        p = [(D, v) for D, v in pts if D >= D0]
        svg.add('<polyline points="' + " ".join(f"{xm(D):.1f},{ym(v / v0):.1f}" for D, v in p)
                + f'" fill="none" stroke="{col}" stroke-width="2"/>')
        for D, v in (p if name == "error bar" else [q for q in p if q[0] % 5 == 0] + [p[-1]]):
            marker(svg, "circle", xm(D), ym(v / v0), col, f"D {D:g}: {name} {v:.1e}, {v / v0:.1e} of its value at D {D0:g}")
    # the rates, written beside the lines they belong to
    svg.text(xm(150), ym(at(err, 150) / e0) - 14, f"error bar on Δ: one decade per {1 / s_err:.0f} in D", size=11,
             fill=INK, weight="bold")
    svg.text(xm(128), ym(at(cond, 128) / c0) - 22, f"one decade per {1 / s_lo:.1f} in D", size=11, fill=INK,
             weight="bold", anchor="end")
    svg.text(xm(top) - 16, ym(cond[-1][1] / c0) + 4, f"one decade per {1 / s_hi:.1f} in D", size=11, fill=INK,
             weight="bold", anchor="end")
    knee = 170.0
    callout(svg, xm(knee), ym(at(cond, knee) / c0), "the smallest pivot starts to fall", dx=30, dy=44)
    callout(svg, xm(wall), ym(at(cond, wall) / c0), "D 215.03, where the chain had stopped", dx=-64, dy=8)
    better_hint(svg, x, y, w, h, "right", label="smaller", reverse=True)
    lx, ly = x + 22, y + 24
    for col, s in ((SERIES[1], "Jacobian's condition, largest / smallest pivot"), (SERIES[0], "error bar on Δ")):
        svg.line(lx - 2, ly - 4, lx + 12, ly - 4, col, 2)
        marker(svg, "circle", lx + 5, ly - 4, col, s)
        svg.text(lx + 20, ly, s, size=11, fill=INK)
        ly += 20
    hgt = footnote(svg, ["error bar: hp/data/delta_all.csv (chain_fit, err_kind fits)",
                         f"pivots: hp/data/chain_pivots.csv from the chain's logs (raw {piv[0]['raw_commit']})"],
                   "chain run on win2, Intel Core Ultra 7 265K, 18 threads", [], y + h + 44, REPO)
    out = os.path.join(a.o, "conditioning_vs_error.svg")
    open(out, "w").write(svg.finish(hgt))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
