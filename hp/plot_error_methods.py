#!/usr/bin/env python3
"""How each group computes Delta and estimates its error, in two SVGs.

  error_methods.svg  one row per group: how Delta is computed, the error estimate on each value, the fit or
                     extrapolation, and the quoted limit. Each estimate is coloured by its kind, and a dashed box is
                     our reading of a method the source does not state.
  error_sizes.svg    those errors against D: our err column (the largest per D bin), each published value's distance
                     from ours, the uncertainty each source states, the series' spread over fit choices, the
                     held-trend bias that chain_fit removed, and chain_fit's own bias, the release step's |dDelta|.

The methods are as notes/LIT_REVIEW.md, notes/DINF_ANALYTIC.md and README.md record them. The data are
data/delta_all.csv, data/delta_published_compare.csv, data/published_vs_series.txt, data/chain_fit_shift.txt and
data/release_steps.csv.

    cd hp && python3 plot_error_methods.py [-o .]
"""
import argparse
import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_graphs as dg  # noqa: E402
from data_graphs import GRID, INK, INK_SOFT, SERIES, SHAPES, SURFACE, Svg, axes, better_arrow, callout, esc, footnote, marker  # noqa: E402,E501

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
KINDS = [("two", "spread between two computations", SERIES[0]),
         ("fits", "spread over fits or estimators", SERIES[1]),
         ("proof", "rigorous enclosure", SERIES[2]),
         ("round", "quoted precision or rounding only", SERIES[3]),
         ("none", "none stated", SERIES[-1])]
COLOR = {k: c for k, _, c in KINDS}

# (group lines, compute, error, fit, limit); each cell is (lines, kind or None, surmised)
ROWS = [
    (["Martín-García, Gundlach", "PRD 2003 · D = 4"],
     (["Periodic boundary-value solve,", "analytic at centre and light cone"], None, True),
     (["Δ = 3.445452402(3), the (3)", "from convergence in resolution"], "two", True),
     (["None, one dimension"], None, False),
     (["—"], None, False)),
    (["Reiterer, Trubowitz", "CMP 2019 · D = 4"],
     (["Computer-assisted existence proof", "of a real analytic solution"], None, False),
     (["Rigorous enclosure of Δ/2", "to 80 digits"], "proof", False),
     (["None, one dimension"], None, False),
     (["—"], None, False)),
    (["Evolution codes, 2005–11", "Sorkin–Oren, Bland et al.,", "Taves–Kunstatter · D ≤ 14"],
     (["Time evolution and bisection,", "Δ read off the echoes"], None, True),
     (["Not recorded in our notes"], "none", False),
     (["γ ≈ 0.466(1 − e^(−0.408 D)), γ only", "(Bland et al., 3.5 ≤ D ≤ 14)"], None, False),
     (["Δ∞ = 5/2, conjectured", "in Bland's thesis (2007)"], "none", False)),
    (["Ecker, Ecker, Grumiller", "PRL 2026 · 1/D expansion"],
     (["Expansion in ε = 1/(D − 1), the", "NLO condition Eq. (20) fixes Δ"], None, False),
     (["Analytic, no error stated; we", "dispute that Eq. (20) selects Δ"], "none", False),
     (["Existing data suggest Δ → 0"], None, False),
     (["Δ → 0, conjectured"], "none", False)),
    (["Ecker, Ecker, Grumiller,", "Jechtl · SciPost 2026", "3.05 ≤ D ≤ 5.5"],
     (["Series about the self-similarity", "horizon, order raised above D 4.8"], None, False),
     (["Precision 1e-6, quoted at the", "maximum of Δ and at D = 4"], "round", False),
     (["Expansions in 1/D and in D − 3"], None, False),
     (["Δ → 0 at D → 3⁺ and at", "D → ∞, conjectured"], "none", False)),
    (["Ecker, Ecker, Grumiller", "arXiv:2611.07007", "D 4–200 in steps of 1/2"],
     (["Spectral Newton in GMP, on", "Reiterer–Trubowitz infrastructure"], None, False),
     (["No uncertainty stated, six", "decimals, so rounding 5e-7"], "round", False),
     (["Δ ≈ 1.6439 + 10.37/D − O(1/D²);", "the D = ∞ entry's fit range and", "method are not stated"], None, False),
     (["Δ∞ = 1.643911,", "no uncertainty stated"], "none", False)),
    (["This project", "full system, D 3.04–73"],
     (["Newton on the full system in", "double- and quad-double"], None, False),
     (["Two resolutions M, or two starts,", "a spread, not a bound", "(1e-22 to 4e-10; 1e-28 in qd)"], "two", False),
     (["1/D and 1/n series (degree 16–20),", "rational in 1/(D − 2), cut-plane", "ζ-series: 48 fit choices"], None, False),
     (["Δ∞ = 1.643893746 ± 1.5e-8,", "fit spread, corrected by", "a planted-truth test"], "fits", False)),
    (["This project", "ANM chain, D 74–215"],
     (["Taylor steps in D, harmonics 5–13", "held on a fit of 138 full states"], None, False),
     (["Spread over the held fit's degree,", "1e-15 to 1e-11; release step:", "bias 1e-14 to 3e-12"], "fits", False),
     (["Checked against the 1/D series,", "inside its spread at D 74–190", "(8e-12 at 80, 2e-9 at 190)"], None, False),
     (["—"], None, False)),
    (["This project", "D = ∞ inner problem"],
     (["Zero of the harmonic-1 Stokes", "multiplier, 22 amplitude terms"], None, False),
     (["Estimator range over nine orders,", "tested on planted functions"], "fits", False),
     (["None, D = ∞ directly"], None, False),
     (["Δ∞ = 1.64387 ± 0.00006,", "under Conjecture 1"], "fits", False)),
]


def box(svg, x, y, w, h, cell):
    lines, kind, surmised = cell
    dash = ' stroke-dasharray="4 3"' if surmised else ""
    svg.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="{SURFACE}" '
            f'stroke="{INK_SOFT}" stroke-width="1"{dash}/>')
    if kind:
        svg.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="6" height="{h:.1f}" rx="2" fill="{COLOR[kind]}"/>')
    ty = y + h / 2 - 7 * (len(lines) - 1) + 4
    for i, s in enumerate(lines):
        svg.text(x + 14, ty + 14 * i, s, size=11, fill=INK if i == 0 else INK_SOFT)


def arrow(svg, x1, x2, y):
    svg.line(x1, y, x2 - 6, y, INK_SOFT)
    svg.add(f'<polygon points="{x2:.1f},{y:.1f} {x2 - 7:.1f},{y - 4:.1f} {x2 - 7:.1f},{y + 4:.1f}" fill="{INK_SOFT}"/>')


def fig_methods(path):
    cols = [("group", 200), ("computed by", 250), ("error on each value", 250), ("fit or extrapolation", 250),
            ("quoted limit Δ∞", 190)]
    gap, x0 = 28, 16
    W = x0 * 2 + sum(w for _, w in cols) + gap * (len(cols) - 1)
    svg = Svg(W)
    svg.text(x0, 24, "How each group computes the echo period Δ(D) and estimates its error", size=14, fill=INK,
             weight="bold")
    svg.text(x0, 42, "Rows run left to right: the computation, the error quoted on each value, the fit across D, and "
             "the limit quoted. Colour is the kind of error estimate.", size=11)
    lx = x0
    for i, (k, label, c) in enumerate(KINDS):
        svg.add(f'<rect x="{lx:.1f}" y="54" width="10" height="10" rx="2" fill="{c}"/>')
        svg.text(lx + 15, 63, label, size=11, fill=INK)
        lx += 15 + dg.text_width(label, 11) + 22
    svg.add(f'<rect x="{lx:.1f}" y="54" width="22" height="10" rx="2" fill="none" stroke="{INK_SOFT}" '
            f'stroke-dasharray="4 3"/>')
    svg.text(lx + 28, 63, "our reading, not stated by the source", size=11, fill=INK)
    xs, x = [], x0
    for _, w in cols:
        xs.append(x)
        x += w + gap
    y = 96
    for (label, w), cx in zip(cols, xs):
        svg.text(cx, y, label.upper(), size=10, fill=INK_SOFT, weight="bold")
    y += 12
    rh, rg = 52, 10
    for r, row in enumerate(ROWS):
        if r == 6:
            svg.line(x0, y + 1, W - x0, y + 1, GRID, 2)
            y += 10
        group, cells = row[0], row[1:]
        cy = y + rh / 2
        gy = cy - 7 * (len(group) - 1) + 4
        for i, s in enumerate(group):
            svg.text(xs[0], gy + 14 * i, s, size=12 if i == 0 else 10, fill=INK if i == 0 else INK_SOFT,
                     weight="bold" if i == 0 else "normal")
        for j, cell in enumerate(cells):
            box(svg, xs[j + 1], y, cols[j + 1][1], rh, cell)
            if j:
                arrow(svg, xs[j] + cols[j][1] + 3, xs[j + 1] - 3, cy)
        y += rh + rg
    h = footnote(svg, ["notes/LIT_REVIEW.md", "notes/DINF_ANALYTIC.md", "README.md", "hp/data/delta_all.csv (raw 987f15c)"],
                 "", [], y, REPO)
    open(path, "w").write(svg.finish(h))


class DecadeAxis(dg.Axis):
    """a log axis over many decades ticks every fourth decade"""

    def ticks(self):
        if self.log and math.log10(self.hi / self.lo) > 8:
            return [10.0 ** e for e in range(math.ceil(math.log10(self.lo)), math.floor(math.log10(self.hi)) + 1)
                    if e % 4 == 2]
        return super().ticks()


def table_col(path, col):
    """the first column and column col of a markdown table, as floats"""
    out = []
    for line in open(path):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if re.match(r"^[0-9.]+$", cells[0]):
            out.append((float(cells[0]), float(cells[col])))
    return out


def sizes_panel(svg, x, y, w, h, xaxis=(3, 220, "log")):
    """the plot of error_sizes.svg in the box (x, y, w, h) of svg, with its legend to the right: every series over the
    D range of xaxis, points below it left out. Prints the plotted numbers and returns the x map."""
    lo_D = xaxis[0]
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "delta_all.csv"))))
    pub = list(csv.DictReader(open(os.path.join(HERE, "data", "delta_published_compare.csv"))))
    spread = table_col(os.path.join(HERE, "data", "published_vs_series.txt"), 3)
    shift = table_col(os.path.join(HERE, "data", "chain_fit_shift.txt"), 3)
    steps = sorted((float(r["D"]), abs(float(r["dDelta"])))
                   for r in csv.DictReader(open(os.path.join(HERE, "data", "release_steps.csv"))))
    # our err column, the largest of each kind in each of 60 log-spaced D bins
    shape = {"dd": SHAPES[0], "qd": SHAPES[1], "pinned": SHAPES[2], "chain_fit": SHAPES[3]}
    best = {}
    for r in rows:
        if r["err"] and float(r["err"]) > 0:
            b = (r["kind"], int(60 * math.log(float(r["D"]) / 3) / math.log(220 / 3)))
            if b not in best or float(r["err"]) > float(best[b]["err"]):
                best[b] = r
    ours = sorted((float(r["D"]), float(r["err"]), r["kind"], r["err_kind"]) for r in best.values())
    diffs = {ref: [(float(p["D"]), abs(float(p["diff"]))) for p in pub if p["published_ref"] == ref and p["diff"]]
             for ref in ("EEG2026", "EEGJ2026", "MGG2003")}
    stated = {"EEG2026": (5.75, 200, 5e-7), "EEGJ2026": (3.15, 4.0, 1e-6), "MGG2003": (3.9, 4.1, 3e-9)}
    col = {"EEG2026": SERIES[1], "EEGJ2026": SERIES[2], "MGG2003": SERIES[3]}
    name = {"EEG2026": "Ecker et al. 2026, D ≤ 200", "EEGJ2026": "Ecker et al. 2026, D ≤ 5.5",
            "MGG2003": "Martín-García, Gundlach"}

    print("| series | D | value |\n|---|---|---|")
    for D, e, k, ek in ours:
        print(f"| ours err ({k}, {ek}) | {D:.6g} | {e:.1e} |")
    for ref, pts in diffs.items():
        print(f"| {ref} |published - ours|, {len(pts)} values | {pts[0][0]:g}-{pts[-1][0]:g} | "
              f"max {max(v for _, v in pts):.2e} |")
    for D, v in spread:
        print(f"| series spread | {D:g} | {v:.1e} |")
    for D, v in shift:
        print(f"| held-trend bias removed | {D:g} | {v:.4e} |")
    for D, v in steps:
        print(f"| chain_fit bias, release step | {D:g} | {v:.3e} |")

    dg.Axis = DecadeAxis
    xm, ym = axes(svg, x, y, w, h, xaxis, (1e-30, 1e-2, "log"), "spacetime dimension D",
                  "error or difference in Δ (lower is better)", xfmt=lambda v: f"{v:g}",
                  yfmt=lambda v: f"1e{round(math.log10(v))}")
    for ref, (a, b, u) in stated.items():
        if b <= lo_D:
            continue
        a = max(a, lo_D)
        svg.add(f'<line x1="{xm(a):.1f}" y1="{ym(u):.1f}" x2="{xm(b):.1f}" y2="{ym(u):.1f}" stroke="{col[ref]}" '
                f'stroke-width="2" stroke-dasharray="6 4"/>')
    spread, shift, steps = ([q for q in t if q[0] >= lo_D] for t in (spread, shift, steps))
    ours = [q for q in ours if q[0] >= lo_D]
    diffs = {ref: [q for q in pts if q[0] >= lo_D] for ref, pts in diffs.items()}
    pts = " ".join(f"{xm(D):.1f},{ym(v):.1f}" for D, v in spread)
    svg.add(f'<polyline points="{pts}" fill="none" stroke="{SERIES[-1]}" stroke-width="2"/>')
    for D, e, k, ek in ours:
        marker(svg, shape[k], xm(D), ym(e), SERIES[0], f"D {D:.6g}: err {e:.1e} ({k}, {ek})")
    for ref in ("EEG2026", "EEGJ2026", "MGG2003"):
        for D, v in diffs[ref]:
            marker(svg, "circle", xm(D), ym(max(v, 1e-30)), col[ref], f"D {D:g}: |{ref} - ours| {v:.2e}")
    # the bias chain_fit removed, drawn over the published differences it sits among
    pts = " ".join(f"{xm(D):.1f},{ym(v):.1f}" for D, v in shift)
    svg.add(f'<polyline points="{pts}" fill="none" stroke="{SERIES[4]}" stroke-width="2"/>')
    for D, v in shift:
        marker(svg, "circle", xm(D), ym(v), SERIES[4], f"D {D:g}: chain_fit - chain {v:+.4e}")
    # the bias left in chain_fit, the release step's |dDelta| (release_steps.py)
    pts = " ".join(f"{xm(D):.1f},{ym(v):.1f}" for D, v in steps)
    svg.add(f'<polyline points="{pts}" fill="none" stroke="{SERIES[5]}" stroke-width="2"/>')
    for D, v in steps:
        marker(svg, "triangle-down", xm(D), ym(v), SERIES[5], f"D {D:g}: release step |dDelta| {v:.3e}")
    if lo_D <= 3.2:
        callout(svg, xm(22.5), ym(2.01e-3), "a bump at D 21–27", dx=40, dy=10)
        callout(svg, xm(3.2), ym(2.47e-6), "outside its stated 1e-6", dx=30, dy=-22)
    # the spread between two step paths is in the table only where no fit spread replaced it
    runs = [r for r in ours if r[3] == "runs" and r[2] == "chain_fit"]
    if runs:
        callout(svg, xm(runs[0][0]), ym(runs[0][1]), "two step paths: float64 held values", dx=-30, dy=-40)
    if lo_D <= 79.77:
        callout(svg, xm(79.77), ym(5.0074e-8), "bias removed by the held fit", dx=-30 if lo_D <= 3.2 else 36, dy=36)
    # a long arrow in the right margin, above the legend: down is better on this axis
    better_arrow(svg, x + w + 22, y + 6, x + w + 22, y + 126, label="better")
    # legend at the right of the plot
    lx, ly = x + w + 46, y + 140
    items = [(SERIES[0], "ours, err column", "marker"), (SERIES[1], name["EEG2026"], "marker"),
             (SERIES[2], name["EEGJ2026"], "marker"), (SERIES[3], name["MGG2003"], "marker"),
             (SERIES[4], "held-trend bias removed", "line"), (SERIES[5], "chain_fit's bias, release step", "line"),
             (SERIES[-1], "1/D, 1/n series spread", "line")]
    for c, s, kind in items:
        if kind == "marker":
            marker(svg, "circle", lx + 5, ly - 4, c, s)
        else:
            svg.line(lx - 2, ly - 4, lx + 12, ly - 4, c, 2)
        svg.text(lx + 18, ly, s, size=11, fill=INK)
        ly += 20
    ly += 8
    svg.text(lx, ly, "ours, by kind", size=11, fill=INK_SOFT)
    for k, s in shape.items():
        ly += 20
        marker(svg, s, lx + 5, ly - 4, SERIES[0], k)
        svg.text(lx + 18, ly, k, size=11, fill=INK)
    ly += 28
    svg.add(f'<line x1="{lx - 2}" y1="{ly - 4}" x2="{lx + 12}" y2="{ly - 4}" stroke="{INK_SOFT}" stroke-width="2" '
            f'stroke-dasharray="6 4"/>')
    svg.text(lx + 18, ly, "stated uncertainty", size=11, fill=INK)
    return xm


def fig_sizes(path):
    W, x, y, w, h = 1040, 92, 92, 700, 470
    svg = Svg(W)
    svg.text(16, 24, "The size of each error, against D", size=14, fill=INK, weight="bold")
    svg.text(16, 42, "Markers are measured differences, dashed segments the uncertainty a source states. Ours: the "
             "largest err in each D bin, shape by kind.", size=11)
    sizes_panel(svg, x, y, w, h)
    hgt = footnote(svg, ["hp/data/delta_all.csv and delta_published_compare.csv (raw 987f15c)",
                         "published_vs_series.txt", "chain_fit_shift.txt", "release_steps.csv"], "", [], y + h + 44, REPO)
    open(path, "w").write(svg.finish(hgt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=HERE)
    a = ap.parse_args()
    fig_methods(os.path.join(a.o, "error_methods.svg"))
    fig_sizes(os.path.join(a.o, "error_sizes.svg"))
