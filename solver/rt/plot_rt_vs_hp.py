#!/usr/bin/env python3
"""|Delta_RT - Delta_hp| against D for every RT grid of #105 and #145: the Reiterer-Trubowitz prototype's Delta against
this repository's double-double values, one series per Chebyshev grid (notes/RT_FORMULATION.md sections 9-10).

Reads every *.jsonl record with a "diff" (Delta_RT - Delta_hp) in the raw folders I105-mac/ and I145-mac/ of the
raw-data repository; mutated runs are left out. Two different values at one (grid, D) are two roots from different
starts, and both are drawn.

    python3 solver/rt/plot_rt_vs_hp.py ../choptuik-large-d-raw [-o solver/rt/rt_vs_hp.svg]
"""
import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hp"))
import data_graphs as dg  # noqa: E402
from data_graphs import (INK, INK_SOFT, SERIES, Svg, axes, better_hint, callout, footnote,  # noqa: E402
                         host_specs, marker)

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
GRIDS = [((32, 101), "32 × 101"), ((32, 121), "32 × 121"), ((32, 141), "32 × 141"), ((32, 81), "32 × 81")]


class DecadeAxis(dg.Axis):
    """a log axis over many decades ticks every second decade"""

    def ticks(self):
        if self.log and math.log10(self.hi / self.lo) > 8:
            return [10.0 ** e for e in range(math.ceil(math.log10(self.lo)), math.floor(math.log10(self.hi)) + 1)
                    if e % 2 == 0]
        return super().ticks()


def load(raw):
    pts = {}
    for f in glob.glob(os.path.join(raw, "I105-mac", "**", "*.jsonl"), recursive=True) + \
            glob.glob(os.path.join(raw, "I145-mac", "**", "*.jsonl"), recursive=True):
        for line in open(f):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("diff") is None or not r.get("grid") or r.get("mutate"):
                continue
            g, D, d = tuple(r["grid"]), round(r["D"], 6), r["diff"]
            pts.setdefault(g, set()).add((D, float(f"{d:.3e}")))
    return {g: sorted(v) for g, v in pts.items()}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("-o", default=os.path.join(HERE, "rt_vs_hp.svg"))
    a = ap.parse_args(argv)
    pts = load(a.raw)
    print("| grid | D | Delta_RT - Delta_hp |")
    print("|---|---|---|")
    for g, name in GRIDS:
        for D, d in pts.get(g, []):
            print(f"| {name} | {D:g} | {d:+.2e} |")

    W, x, y, w, h = 980, 96, 76, 680, 420
    svg = Svg(W)
    svg.text(16, 24, "The RT prototype's Δ against ours, by Chebyshev grid", size=14, fill=INK, weight="bold")
    svg.text(16, 42, "|Δ_RT − Δ_hp| at each D; two markers at one D are two converged roots from different starts. "
             "Above D ≈ 40 a finer grid is worse.", size=11)
    # 18 decades: tick every second one
    dg.Axis = DecadeAxis
    xm, ym = axes(svg, x, y, w, h, (8, 62), (1e-23, 1e-5, "log"), "spacetime dimension D",
                  "|Δ_RT − Δ_hp| (lower is better)", xfmt=lambda v: f"{v:g}",
                  yfmt=lambda v: f"1e{round(math.log10(v))}")
    # the acceptance line of #105 and #145
    svg.add(f'<line x1="{x}" y1="{ym(1e-12):.1f}" x2="{x + w}" y2="{ym(1e-12):.1f}" stroke="{INK_SOFT}" '
            f'stroke-width="1.5" stroke-dasharray="6 4"/>')
    svg.text(xm(31), ym(1e-12) - 6, "acceptance, 1e-12", size=11)
    for (g, name), color in zip(GRIDS, [SERIES[0], SERIES[1], SERIES[2], SERIES[-1]]):
        p = pts.get(g, [])
        first = {}
        for D, d in p:
            first.setdefault(D, d)
        line = " ".join(f"{xm(D):.1f},{ym(max(abs(d), 1e-23)):.1f}" for D, d in sorted(first.items()))
        svg.add(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/>')
        for D, d in p:
            marker(svg, "circle", xm(D), ym(max(abs(d), 1e-23)), color, f"{name}, D {D:g}: {d:+.2e}")
    callout(svg, xm(54.996085), ym(2.97e-7), "32 × 141 at D 55: 3e-7", dx=-40, dy=-6)
    callout(svg, xm(60.001952), ym(2.33e-12), "32 × 101 misses at D 60", dx=-90, dy=-50)
    callout(svg, xm(15.0), ym(2.28e-22), "32 × 121 to 2e-22", dx=26, dy=-14)
    better_hint(svg, x, y, w, h, "right", label="smaller", reverse=True)
    lx, ly = x + w + 46, y + 140
    for (g, name), color in zip(GRIDS, [SERIES[0], SERIES[1], SERIES[2], SERIES[-1]]):
        marker(svg, "circle", lx + 5, ly - 4, color, name)
        svg.text(lx + 18, ly, name, size=11, fill=INK)
        ly += 20
    hgt = footnote(svg, ["solver/rt/rt_run.py --solver mp (main b4d401f)", "raw I105-mac 6c9d697, I145-mac 62b275a",
                         "hp double-double rows of hp/data/delta_all.csv"], host_specs(), [], y + h + 44, REPO)
    open(a.o, "w").write(svg.finish(hgt))
    print(f"wrote {a.o}")


if __name__ == "__main__":
    main(sys.argv[1:])
