#!/usr/bin/env python3
"""Graph the echo period Delta(D) as an SVG.

Three panels: Delta against D with the large-D fits, Delta against 1/sqrt(D-2) (D = infinity is the left
edge, so the limit is the intercept), and the local power-law exponent -dlnDelta/dln(D-2). Points above
PINNED_MAX are drawn hollow and shaded: the solver does not pin Delta there yet, and the fits ignore them.

Usage (from solver/data, where the scan files live):
    python3 ../plot_delta.py [-o ../delta.svg]

The SVG is written by hand in the style of zlib-ng/codecbench's graph scripts: fixed categorical colors,
a light surface with a prefers-color-scheme dark override, 2px lines, ringed markers with hover tooltips.
"""
import argparse
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_best import load  # noqa: E402

PINNED_MAX = 25.5   # above this the profile test shows Delta is not pinned by the current solver

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e7e6e2"
BAND = "#efeeea"
# Categorical slots in fixed order, never cycled; the last slot is the neutral one.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#a6339a", "#7a7668"]
NEUTRAL = SERIES[-1]

# Dark-mode counterparts, the same hues re-stepped for the dark surface.
DARK = {
    SURFACE: "#201f1d",
    INK: "#e8e6e3",
    INK_SOFT: "#a8a49d",
    GRID: "#343230",
    BAND: "#2a2927",
    "#eb6834": "#dd5c26",
    "#7a7668": "#948e82",
}

REPO_URL = "https://github.com/nmoinvaz/dss-residual-kernel"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    """Collects body elements; the height is decided at finish time."""

    def __init__(self, width):
        self.w = width
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=12, fill=INK_SOFT, anchor="start", weight="normal"):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
                 f'text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke, width=1, dash=None, opacity=None):
        extra = (f' stroke-dasharray="{dash}"' if dash else "") + \
                (f' stroke-opacity="{opacity}"' if opacity is not None else "")
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{stroke}" stroke-width="{width}"{extra}/>')

    def path(self, pts, stroke, width=2, dash=None, opacity=0.85, title=None):
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        body = (f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" '
                f'stroke-opacity="{opacity}" stroke-linejoin="round"{extra}/>')
        self.add(f"<g>{body}<title>{esc(title)}</title></g>" if title else body)

    def finish(self, height):
        # Attribute selectors outrank presentation attributes, so dark mode
        # swaps every themed color in place without touching the body markup.
        rules = "".join(f'[fill="{l}"]{{fill:{d}}}[stroke="{l}"]{{stroke:{d}}}'
                        for l, d in DARK.items())
        header = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
            f'height="{height}" viewBox="0 0 {self.w} {height}" '
            f'font-family="system-ui, sans-serif">',
            f'<style>@media (prefers-color-scheme: dark){{{rules}}}</style>',
            f'<rect width="{self.w}" height="{height}" fill="{SURFACE}"/>']
        return "\n".join(header + self.parts + ["</svg>"]) + "\n"


def marker(svg, x, y, color, title, hollow=False, shape="circle", r=4.5):
    """Marker with a 2px surface ring; hollow markers carry the not-pinned state, never color alone."""
    if hollow:
        style = f'fill="{SURFACE}" stroke="{color}" stroke-width="1.6"'
    else:
        style = f'fill="{color}" stroke="{SURFACE}" stroke-width="2"'
    if shape == "diamond":
        q = r + 1.5
        pts = f"{x:.1f},{y - q:.1f} {x + q:.1f},{y:.1f} {x:.1f},{y + q:.1f} {x - q:.1f},{y:.1f}"
        body = f'<polygon points="{pts}" {style}/>'
    else:
        body = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {style}/>'
    svg.add(f"<g>{body}<title>{esc(title)}</title></g>")


def nice_ticks(lo, hi, n=6):
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10 * mag)
    t = math.ceil(lo / step - 1e-9) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


class Panel:
    """A plot area with linear axes, grid, tick labels, a title and a legend key line."""

    def __init__(self, svg, px, py, pw, ph, xr, yr, title, xlabel, ylabel, xfmt="{:g}", yfmt="{:g}"):
        self.svg, self.px, self.py, self.pw, self.ph = svg, px, py, pw, ph
        self.xr, self.yr = xr, yr
        svg.text(px, py - 40, title, size=12, fill=INK)
        for v in nice_ticks(*yr):
            y = self.sy(v)
            svg.line(px, y, px + pw, y, GRID)
            svg.text(px - 8, y + 4, yfmt.format(v), size=10, anchor="end")
        for v in nice_ticks(*xr):
            x = self.sx(v)
            svg.line(x, py, x, py + ph, GRID)
            svg.text(x, py + ph + 16, xfmt.format(v), size=10, anchor="middle")
        svg.line(px, py + ph, px + pw, py + ph, INK_SOFT)
        svg.text(px + pw / 2, py + ph + 36, xlabel, size=11, anchor="middle")
        svg.add(f'<text x="{px - 44:.1f}" y="{py + ph / 2:.1f}" font-size="11" fill="{INK_SOFT}" '
                f'text-anchor="middle" transform="rotate(-90 {px - 44:.1f} {py + ph / 2:.1f})">{esc(ylabel)}</text>')
        self.kx = px

    def sx(self, v):
        return self.px + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.pw

    def sy(self, v):
        return self.py + self.ph - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.ph

    def inside(self, x, y):
        return self.xr[0] <= x <= self.xr[1] and self.yr[0] <= y <= self.yr[1]

    def curve(self, xs, ys, color, dash=None, width=2, title=None, opacity=0.85):
        pts = [(self.sx(x), self.sy(y)) for x, y in zip(xs, ys) if self.inside(x, y)]
        if len(pts) > 1:
            self.svg.path(pts, color, width=width, dash=dash, opacity=opacity, title=title)

    def band(self, x0, x1, label=None):
        a, b = sorted((self.sx(max(x0, self.xr[0])), self.sx(min(x1, self.xr[1]))))
        self.svg.add(f'<rect x="{a:.1f}" y="{self.py:.1f}" width="{b - a:.1f}" height="{self.ph:.1f}" fill="{BAND}"/>')
        if label:
            self.svg.text((a + b) / 2, self.py + 14, label, size=10, anchor="middle")

    def key(self, kind, color, label, dash=None):
        """One legend entry on the key line above the panel."""
        y = self.py - 20
        if kind == "line":
            self.svg.line(self.kx, y - 4, self.kx + 22, y - 4, color, width=2, dash=dash)
            x = self.kx + 28
        else:
            marker(self.svg, self.kx + 5, y - 4, color, label, hollow=(kind == "hollow"))
            x = self.kx + 14
        self.svg.text(x, y, label, size=10)
        self.kx = x + 6.0 * len(label) + 18


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", default="../delta.svg")
    args = ap.parse_args()

    rows = load()
    Dall = np.array(sorted(d for d in rows if d >= 3.69))
    yall = np.array([rows[d]["Delta"] for d in Dall])
    rel = Dall <= PINNED_MAX
    D, y = Dall[rel], yall[rel]
    n = D - 2

    # Fits on the pinned points only
    m8 = D >= 8
    A8 = np.stack([np.ones(m8.sum())] + [n[m8] ** -p for p in (1, 1.5, 2)], 1)
    cL, *_ = np.linalg.lstsq(A8, y[m8], rcond=None)
    def fL(nn):
        return cL[0] + cL[1] / nn + cL[2] * nn ** -1.5 + cL[3] / nn ** 2
    m10 = D >= 10
    def f2(p, nn):
        return p[0] + p[1] * nn ** -p[2] * (1 + p[3] / nn)
    p2 = least_squares(lambda p: f2(p, n[m10]) - y[m10], [1.5, 2, 0.5, 0]).x

    fitL = (f"Δ∞ + a/(D−2) + b/(D−2)^1.5 + c/(D−2)², Δ∞ = {cL[0]:.4f}, "
            f"a = {cL[1]:.3f}, b = {cL[2]:.3f}, c = {cL[3]:.3f} (D ≥ 8)")
    fit2 = f"L + A(D−2)^−α (1 + b/(D−2)), L = {p2[0]:.4f}, A = {p2[1]:.3f}, α = {p2[2]:.3f}, b = {p2[3]:.3f} (D ≥ 10)"

    width = 1080
    svg = Svg(width)
    svg.text(16, 28, "Choptuik echo period Δ(D) of massless scalar critical collapse", size=15, fill=INK, weight="bold")
    svg.text(16, 48, f"{len(Dall)} computed solutions, D = {Dall.min():.1f} to {Dall.max():.1f}; "
                     f"the {rel.sum()} pinned points (D ≤ {PINNED_MAX:g}) carry the fits, hollow points are not pinned yet",
             size=12)

    # Panel A: Delta against D, full width
    A = Panel(svg, 78, 120, 984, 300, (3, 60), (1.2, 3.6),
              "Δ levels off at a finite value instead of falling to zero",
              "spacetime dimension D", "echo period Δ")
    A.band(PINNED_MAX, Dall.max() + 0.3, "not pinned")
    DD = np.linspace(8, 60, 400)
    svg.line(A.sx(3), A.sy(cL[0]), A.sx(60), A.sy(cL[0]), SERIES[1], width=1, opacity=0.5)
    A.curve(DD, fL(DD - 2), SERIES[1], dash="6 4", title=fitL)
    D2 = np.linspace(10, 60, 400)
    A.curve(D2, f2(p2, D2 - 2), SERIES[2], dash="6 4", title=fit2)
    D3 = np.linspace(3, 60, 400)
    A.curve(D3, 4.24 * (D3 - 2) ** -0.25, NEUTRAL, dash="2 3", width=1.5, title="old crossover law 4.24 (D−2)^−1/4")
    for d, v, ok in zip(Dall, yall, rel):
        marker(svg, A.sx(d), A.sy(v), SERIES[0], f"D = {d:.4f}, Δ = {v:.10f}" + ("" if ok else " (not pinned)"), hollow=not ok)
    svg.text(A.sx(34), A.sy(cL[0]) + 16, f"Δ∞ ≈ {cL[0]:.3f}  (fit limit)", size=11, fill=INK)
    A.key("dot", SERIES[0], "computed, pinned")
    A.key("hollow", SERIES[0], "computed, not pinned")
    A.key("line", SERIES[1], "Δ∞ + a/(D−2) + b/(D−2)^1.5 + c/(D−2)²", dash="6 4")
    A.key("line", SERIES[2], "L + A(D−2)^−α(1 + b/(D−2))", dash="6 4")
    A.key("line", NEUTRAL, "old law 4.24 (D−2)^−1/4", dash="2 3")

    # Panel B: Delta against 1/sqrt(D-2), D = infinity at the left edge
    B = Panel(svg, 78, 540, 452, 280, (0, 0.8), (1.4, 3.6),
              "Against 1/√(D−2) the curve runs into a finite intercept",
              "1/√(D−2)   (D = ∞ at the left edge)", "echo period Δ", xfmt="{:.2f}")
    B.band((Dall.max() - 2) ** -0.5 - 0.004, (PINNED_MAX - 2) ** -0.5)
    X = np.linspace(1e-3, 0.41, 300)
    B.curve(np.r_[0, X], np.r_[cL[0], fL(X ** -2)], SERIES[1], dash="6 4", title=fitL)
    for d, v, ok in zip(Dall, yall, rel):
        marker(svg, B.sx((d - 2) ** -0.5), B.sy(v), SERIES[0], f"D = {d:.4f}, Δ = {v:.10f}" + ("" if ok else " (not pinned)"),
               hollow=not ok, r=3.8)
    marker(svg, B.sx(0), B.sy(cL[0]), SERIES[1], f"D = ∞: Δ∞ = {cL[0]:.4f} (fit)", shape="diamond")
    svg.text(B.sx(0) + 12, B.sy(cL[0]) + 18, f"D = ∞:  Δ∞ ≈ {cL[0]:.3f}", size=11, fill=INK)
    for d in (4, 6, 10, 18):
        xv = (d - 2) ** -0.5
        yv = np.interp(xv, (n ** -0.5)[::-1], y[::-1])
        svg.text(B.sx(xv) + 6, B.sy(yv) + 16, f"D={d}", size=10)
    B.key("dot", SERIES[0], "computed, pinned")
    B.key("hollow", SERIES[0], "not pinned")
    B.key("line", SERIES[1], "fit (D ≥ 8) extended to D = ∞", dash="6 4")

    # Panel C: local exponent
    nall = Dall - 2
    q = -np.gradient(np.log(yall), np.log(nall))
    C = Panel(svg, 610, 540, 452, 280, (3, 32), (0.1, 0.3),
              "Local exponent −d ln Δ / d ln(D−2) peaks near D ≈ 7, then falls",
              "spacetime dimension D", "local exponent", yfmt="{:.2f}")
    C.band(PINNED_MAX, 32, "not pinned")
    C.curve([3, 32], [0.25, 0.25], NEUTRAL, dash="2 3", width=1.5, title="1/4, the old crossover law")
    svg.text(C.sx(31.5), C.sy(0.25) - 6, "¼ (old law)", size=10, anchor="end")
    C.curve(Dall[rel], q[rel], SERIES[0], width=2, opacity=0.65)
    for d, v, ok in zip(Dall, q, rel):
        if C.inside(d, v):
            marker(svg, C.sx(d), C.sy(v), SERIES[0], f"D = {d:.4f}, exponent {v:.4f}" + ("" if ok else " (not pinned)"),
                   hollow=not ok, r=3.5)
    C.key("dot", SERIES[0], "pinned")
    C.key("hollow", SERIES[0], "not pinned")
    C.key("line", NEUTRAL, "¼", dash="2 3")

    # Footnote and repository link
    height = 540 + 280 + 80
    svg.text(16, height - 12, "Δ from spectral multiple shooting (solver/README.md); "
                              "hover a point for its value, a curve for its coefficients", size=10)
    svg.add(f'<a href="{REPO_URL}"><text x="{width - 16}" y="{height - 12}" font-size="10" fill="{INK_SOFT}" '
            f'text-anchor="end" text-decoration="underline">{esc(REPO_URL.removeprefix("https://"))}</text></a>')
    with open(args.output, "w") as f:
        f.write(svg.finish(height))
    print(f"wrote {args.output}: {len(Dall)} points, {rel.sum()} pinned")
    print(f"fit 1: {fitL}")
    print(f"fit 2: {fit2}")


if __name__ == "__main__":
    main()
