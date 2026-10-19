#!/usr/bin/env python3
"""Graph the echo period Delta(D) and the critical exponent gamma(D) as an SVG.

Three panels: Delta against D with the power series in 1/(D-2) and its limit, Delta against 1/(D-2) (D = infinity
is the left edge, so the limit is the intercept), and gamma against 1/(D-2) with gamma_inf = 1/2. The series are
fitted on D <= FIT_MAX, where the independent setups agree to ~1e-8; the points above are drawn hollow and are held
out of every fit, so the graph shows how well the formula predicts them.

Usage (from solver/data, where the scan files live):
    python3 ../plot_delta.py [-o ../delta.svg]

The SVG is written by hand in the style of zlib-ng/codecbench's graph scripts: fixed categorical colors,
a light surface with a prefers-color-scheme dark override, 2px lines, ringed markers with hover tooltips.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_ne import load  # noqa: E402

FIT_MAX = 25.5      # setups agree to ~1e-8 below this; points above are held out of the fits

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


def series_fit(x, y, w, m, c0=None):
    """Weighted polynomial in x of degree m; c0 fixed if given. Returns coefficients c0..cm."""
    A = np.vander(x, m + 1, increasing=True)
    if c0 is None:
        c, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        return c
    c, *_ = np.linalg.lstsq(A[:, 1:] * w[:, None], (y - c0) * w, rcond=None)
    return np.r_[c0, c]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", default="../delta.svg")
    args = ap.parse_args()

    # Points: below FIT_MAX the setups agree to ~1e-8 (x_K 0.4 / 0.5 / 0.6, kc 23 / 31); above it, the strongly
    # selecting smoothed scans, held out of every fit.
    base = load()
    Dfit = np.array(sorted(d for d in base if 3.69 <= d <= FIT_MAX))
    yfit = np.array([base[d]["Delta"] for d in Dfit])
    hold = {}
    for setup in (dict(xK=0.6, kc=31, smooth=True), dict(xK=0.5, smooth=True), dict(xK=0.4, smooth=True)):
        for k, r in load(**setup).items():
            if k > FIT_MAX and r["src"] == "ne":
                hold.setdefault(k, r)
    Dh = np.array(sorted(hold)); yh = np.array([hold[k]["Delta"] for k in Dh])

    # Delta: power series in 1/(D-2), degree 9, fitted on 8 <= D <= FIT_MAX (errors 1e-8 below 17.2, 3e-8 above)
    m8 = Dfit >= 8
    w = np.where(Dfit[m8] < 17.2, 1e8, 1 / 3e-8)
    cD = series_fit(1 / (Dfit[m8] - 2), yfit[m8], w, 9)
    fD = lambda d: np.polyval(cD[::-1], 1 / (np.asarray(d) - 2))
    hold_err = np.abs(fD(Dh) - yh).max() if len(Dh) else np.nan

    # gamma: Rust mode-matrix values below 17.2, NE above; series with gamma_inf = 1/2
    graw = {k: r["gamma"] for k, r in base.items() if r.get("gamma") and 3.69 <= k < 17.2}
    try:
        for l in open("gamma_rust.jsonl"):
            r = json.loads(l)
            if r.get("gamma") and 3.69 <= r["D"] < 17.2:
                graw[round(r["D"], 4)] = r["gamma"]
    except OSError:
        pass
    for k, r in base.items():
        if 17.2 <= k <= FIT_MAX and r.get("gamma"):
            graw[k] = r["gamma"]
    Dg = np.array(sorted(graw)); g = np.array([graw[k] for k in Dg])
    gh = {k: r["gamma"] for k, r in hold.items() if r.get("gamma")}
    Dgh = np.array(sorted(gh)); ggh = np.array([gh[k] for k in Dgh])
    mg = Dg >= 8
    cG = series_fit(1 / (Dg[mg] - 2), g[mg], np.ones(mg.sum()), 9, c0=0.5)
    fG = lambda d: np.polyval(cG[::-1], 1 / (np.asarray(d) - 2))

    fitD = "Δ = " + " ".join(f"{c:+.4g}/(D−2)^{k}" if k else f"{c:.5f}" for k, c in enumerate(cD[:5])) + " … (degree 9, fit 8 ≤ D ≤ %g)" % FIT_MAX
    fitG = "γ = 1/2 " + " ".join(f"{c:+.4g}/(D−2)^{k}" for k, c in enumerate(cG[:5]) if k) + " … (degree 9, fit 8 ≤ D ≤ %g)" % FIT_MAX
    zeta2 = np.pi ** 2 / 6

    width = 1080
    svg = Svg(width)
    svg.text(16, 28, "Choptuik echo period Δ(D) and critical exponent γ(D) of massless scalar collapse", size=15, fill=INK, weight="bold")
    svg.text(16, 48, f"Power series in 1/(D−2) fitted on D ≤ {FIT_MAX:g} ({len(Dfit)} points); "
                     f"the {len(Dh)} hollow points above D = {FIT_MAX:g} are held out and match it to {hold_err:.0e}",
             size=12)

    # Panel A: Delta against D, full width
    A = Panel(svg, 78, 120, 984, 300, (3, 60), (1.4, 3.6),
              f"Δ levels off at Δ∞ ≈ {cD[0]:.4f} instead of falling to zero",
              "spacetime dimension D", "echo period Δ")
    if len(Dh):
        A.band(FIT_MAX, Dh.max() + 0.3, "held out")
    svg.line(A.sx(3), A.sy(cD[0]), A.sx(60), A.sy(cD[0]), SERIES[1], width=1, opacity=0.6)
    svg.line(A.sx(3), A.sy(zeta2), A.sx(60), A.sy(zeta2), NEUTRAL, width=1, dash="2 3")
    DD = np.linspace(8, 60, 400)
    A.curve(DD, fD(DD), SERIES[1], dash="6 4", title=fitD)
    for d, v in zip(Dfit, yfit):
        marker(svg, A.sx(d), A.sy(v), SERIES[0], f"D = {d:.4f}, Δ = {v:.10f}")
    for d, v in zip(Dh, yh):
        marker(svg, A.sx(d), A.sy(v), SERIES[0], f"D = {d:.4f}, Δ = {v:.10f} (held out; series {fD(d) - v:+.1e})", hollow=True)
    svg.text(A.sx(44), A.sy(cD[0]) + 16, f"Δ∞ ≈ {cD[0]:.4f}  (series limit)", size=11, fill=INK)
    svg.text(A.sx(44), A.sy(zeta2) - 8, f"π²/6 = {zeta2:.4f}", size=10)
    A.key("dot", SERIES[0], "computed, in the fit")
    A.key("hollow", SERIES[0], "computed, held out")
    A.key("line", SERIES[1], "power series in 1/(D−2), extended to D = 60", dash="6 4")
    A.key("line", NEUTRAL, "π²/6 for reference", dash="2 3")

    # Panel B: Delta against 1/(D-2), D = infinity at the left edge
    B = Panel(svg, 78, 540, 452, 280, (0, 0.6), (1.4, 3.6),
              "Against 1/(D−2), Δ runs into a finite intercept",
              "1/(D−2)   (D = ∞ at the left edge)", "echo period Δ", xfmt="{:.1f}")
    X = np.linspace(0, 1 / 6, 300)
    B.curve(X, np.polyval(cD[::-1], X), SERIES[1], dash="6 4", title=fitD)
    for d, v in zip(Dfit, yfit):
        marker(svg, B.sx(1 / (d - 2)), B.sy(v), SERIES[0], f"D = {d:.4f}, Δ = {v:.10f}", r=3.8)
    for d, v in zip(Dh, yh):
        marker(svg, B.sx(1 / (d - 2)), B.sy(v), SERIES[0], f"D = {d:.4f}, Δ = {v:.10f} (held out)", hollow=True, r=3.8)
    marker(svg, B.sx(0), B.sy(cD[0]), SERIES[1], f"D = ∞: Δ∞ = {cD[0]:.5f} (series)", shape="diamond")
    svg.text(B.sx(0) + 12, B.sy(cD[0]) + 18, f"D = ∞:  Δ∞ ≈ {cD[0]:.4f}", size=11, fill=INK)
    for d in (4, 6, 10, 20):
        svg.text(B.sx(1 / (d - 2)) + 6, B.sy(np.interp(1 / (d - 2), (1 / (Dfit - 2))[::-1], yfit[::-1])) + 16, f"D={d}", size=10)
    B.key("dot", SERIES[0], "in the fit")
    B.key("hollow", SERIES[0], "held out")
    B.key("line", SERIES[1], "series, D ≥ 8", dash="6 4")

    # Panel C: gamma against 1/(D-2)
    C = Panel(svg, 610, 540, 452, 280, (0, 0.6), (0.36, 0.52),
              "γ runs into 1/2 as D → ∞",
              "1/(D−2)   (D = ∞ at the left edge)", "critical exponent γ", xfmt="{:.1f}", yfmt="{:.2f}")
    C.curve([0, 0.6], [0.5, 0.5], NEUTRAL, dash="2 3", width=1.5, title="1/2")
    C.curve(X, np.polyval(cG[::-1], X), SERIES[1], dash="6 4", title=fitG)
    for d, v in zip(Dg, g):
        if C.inside(1 / (d - 2), v):
            marker(svg, C.sx(1 / (d - 2)), C.sy(v), SERIES[0], f"D = {d:.4f}, γ = {v:.10f}", r=3.8)
    for d, v in zip(Dgh, ggh):
        marker(svg, C.sx(1 / (d - 2)), C.sy(v), SERIES[0], f"D = {d:.4f}, γ = {v:.10f} (held out; series {fG(d) - v:+.1e})", hollow=True, r=3.8)
    marker(svg, C.sx(0), C.sy(0.5), SERIES[1], "D = ∞: γ = 1/2", shape="diamond")
    svg.text(C.sx(0) + 12, C.sy(0.5) + 18, "D = ∞:  γ = 1/2", size=11, fill=INK)
    for d in (4, 6, 10):
        if 1 / (d - 2) <= 0.6 and Dg.min() <= d:
            svg.text(C.sx(1 / (d - 2)) + 6, C.sy(np.interp(1 / (d - 2), (1 / (Dg - 2))[::-1], g[::-1])) + 16, f"D={d}", size=10)
    C.key("dot", SERIES[0], "in the fit")
    C.key("hollow", SERIES[0], "held out")
    C.key("line", SERIES[1], "series with γ∞ = 1/2", dash="6 4")

    # Footnote and repository link
    height = 540 + 280 + 80
    svg.text(16, height - 12, "Spectral multiple shooting with nonlinear elimination (solver/CONTEXT.md); "
                              "hover a point for its value, a curve for its coefficients", size=10)
    svg.add(f'<a href="{REPO_URL}"><text x="{width - 16}" y="{height - 12}" font-size="10" fill="{INK_SOFT}" '
            f'text-anchor="end" text-decoration="underline">{esc(REPO_URL.removeprefix("https://"))}</text></a>')
    with open(args.output, "w") as f:
        f.write(svg.finish(height))
    print(f"wrote {args.output}: {len(Dfit)} fitted + {len(Dh)} held-out points (max series error {hold_err:.1e})")
    print(fitD)
    print(fitG)


if __name__ == "__main__":
    main()
