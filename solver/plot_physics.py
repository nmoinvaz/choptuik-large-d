#!/usr/bin/env python3
"""What the numbers mean physically, across dimensions: an SVG with three panels against x = 1/(D-2), so that D = infinity
is the left edge.

1. How much the solution shrinks per echo, e^Delta.
2. How much fine-tuning one more echo costs, in decimal digits: Delta / (gamma ln 10). A near-critical collapse with
   initial-data parameter p makes about gamma ln(1/|p - p*|) / Delta echoes, so each extra echo needs |p - p*| smaller by
   e^(Delta/gamma).
3. How strong the field and its gravity are in the echoing region: the first harmonic of the centre slope dp/dx and the
   departure of c from flat space at the centre, 1 - <c(0)>, measured on the double-double solutions (README, "How the
   solution scales with D"), with their large-D forms 2.72/sqrt(n) and 3.65/n.

Delta and gamma come from the README's rational formulas (fitted on 4 <= D <= 30.2; within 6e-8 of the double-double
points up to D = 61). Landmarks: D = 4, the f64 limit near D = 26, the double-double range 8-61, and D = infinity.

    python3 plot_physics.py [-o physics.svg]
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_delta import BAND, GRID, INK, INK_SOFT, SERIES, Svg, esc, marker  # noqa: E402

# README coefficient table (solver/formula_all.py)
A = [1.6438975180, 4.0378045077e1, 2.5233285989e2, 2.0673502321e2, -3.5709850575e2, 3.1347320996e2,
     -4.1891374574e2, 3.2695252782e2, -1.6050467699e2]
B = [1, 1.8251453626e1, 6.8182050167e1]
E = [0.5, 1.8083629635e1, 1.5241588758e2, -5.2341588541e1, 9.9259594505, -9.8386450353e1, 7.4529506074e1,
     -4.8272358903e1]
Fq = [1, 3.7683686025e1, 3.4245505235e2]
DINF = 1.643893746


def poly(c, x):
    return sum(ci * x ** i for i, ci in enumerate(c))


def delta(x):
    return poly(A, x) / poly(B, x)


def gamma(x):
    return poly(E, x) / poly(Fq, x)


# Measured on the double-double solutions (README table): D, first harmonic of sqrt(n) dp/dx(0), n (1 - <c(0)>)
MEAS = [(20, 3.480, 3.021), (30, 3.212, 3.201), (40, 3.086, 3.304), (50, 3.011, 3.372), (60, 2.962, 3.421)]
XTICKS = [(4, "4"), (5, "5"), (6, "6"), (8, "8"), (10, "10"), (20, "20"), (None, "∞")]


class XPanel:
    """Plot area over x = 1/(D-2) in [0, 0.5] with D-labelled ticks."""

    def __init__(self, svg, px, py, pw, ph, yr, title, ylabel, yticks, yfmt="{:g}", xlabels=True):
        self.svg, self.px, self.py, self.pw, self.ph, self.yr = svg, px, py, pw, ph, yr
        svg.text(px, py - 30, title, size=12, fill=INK)
        for v in yticks:
            y = self.sy(v)
            svg.line(px, y, px + pw, y, GRID)
            svg.text(px - 8, y + 4, yfmt.format(v), size=10, anchor="end")
        for D, lab in XTICKS:
            x = self.sx(0.0 if D is None else 1 / (D - 2))
            svg.line(x, py, x, py + ph, GRID)
            if xlabels:
                svg.text(x, py + ph + 16, lab, size=10, anchor="middle")
        svg.line(px, py + ph, px + pw, py + ph, INK_SOFT)
        if xlabels:
            svg.text(px + pw / 2, py + ph + 34, "D (axis linear in 1/(D − 2); D = ∞ at the left edge)", size=11,
                     anchor="middle")
        svg.add(f'<text x="{px - 46:.1f}" y="{py + ph / 2:.1f}" font-size="11" fill="{INK_SOFT}" text-anchor="middle" '
                f'transform="rotate(-90 {px - 46:.1f} {py + ph / 2:.1f})">{esc(ylabel)}</text>')
        self.kx = px

    def sx(self, x):
        return self.px + x / 0.5 * self.pw

    def sy(self, v):
        return self.py + self.ph - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.ph

    def curve(self, xs, ys, color, dash=None, title=None, width=2):
        pts = [(self.sx(x), self.sy(y)) for x, y in zip(xs, ys) if self.yr[0] <= y <= self.yr[1]]
        self.svg.path(pts, color, width=width, dash=dash, title=title)

    def key(self, kind, color, label, dash=None):
        y = self.py - 12
        if kind == "line":
            self.svg.line(self.kx, y - 4, self.kx + 22, y - 4, color, width=2, dash=dash)
            x = self.kx + 28
        else:
            marker(self.svg, self.kx + 5, y - 4, color, label)
            x = self.kx + 14
        self.svg.text(x, y, label, size=10)
        self.kx = x + 5.6 * len(label) + 18

    def band(self, x0, x1, label=None):
        a, b = self.sx(x0), self.sx(x1)
        self.svg.add(f'<rect x="{a:.1f}" y="{self.py:.1f}" width="{b - a:.1f}" height="{self.ph:.1f}" fill="{BAND}"/>')
        if label:
            self.svg.text((a + b) / 2, self.py + 13, label, size=10, anchor="middle")

    def vline(self, x, label=None, anchor="start", dy=13):
        X = self.sx(x)
        self.svg.line(X, self.py, X, self.py + self.ph, INK_SOFT, dash="3 3", opacity=0.7)
        if label:
            self.svg.text(X + (5 if anchor == "start" else -5), self.py + dy, label, size=10, anchor=anchor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "physics.svg"))
    args = ap.parse_args()
    W, px, pw, ph = 900, 78, 790, 170
    svg = Svg(W)
    svg.text(px, 26, "What the echo looks like physically, from D = 4 to D = ∞", size=15, fill=INK, weight="bold")
    svg.text(px, 46, "From the rational formulas for Δ(D) and γ(D) and the double-double solutions; hover a mark for its value",
             size=11)
    xs = np.linspace(1e-6, 0.5, 400)
    xdd0, xdd1 = 1 / (61.03 - 2), 1 / (8 - 2)
    c0, c1, c2 = SERIES[0], SERIES[1], SERIES[2]

    def landmarks(p, labels):
        p.band(xdd0, xdd1, "double-double points, D = 8–61" if labels else None)
        p.vline(1 / 24, "f64 limit, D ≈ 26" if labels else None, anchor="start", dy=27)
        p.vline(0.5, "D = 4: our world" if labels else None, anchor="end", dy=13)

    # Panel 1: shrink per echo
    y0 = 100
    p1 = XPanel(svg, px, y0, pw, ph, (4, 36), "Each echo is smaller than the last by a factor e^Δ", "e^Δ",
                [5, 10, 15, 20, 25, 30, 35], xlabels=False)
    landmarks(p1, True)
    p1.curve(xs, np.exp(delta(xs)), c0, title="e^Delta from the rational formula")
    for D in (4, 6, 10, 20):
        x = 1 / (D - 2)
        marker(svg, p1.sx(x), p1.sy(math.exp(delta(x))), c0, f"D = {D}: Delta = {delta(x):.4f}, e^Delta = {math.exp(delta(x)):.2f}")
    marker(svg, p1.sx(0), p1.sy(math.exp(DINF)), c0, f"D = infinity: Delta = {DINF}, e^Delta = {math.exp(DINF):.3f}")
    svg.text(p1.sx(0.5) - 10, p1.sy(math.exp(delta(0.5))) + 18, f"{math.exp(delta(0.5)):.0f}× at D = 4", size=11, fill=INK,
             anchor="end")
    svg.text(p1.sx(0) + 8, p1.sy(math.exp(DINF)) - 22, f"{math.exp(DINF):.1f}× at D = ∞", size=11, fill=INK)

    # Panel 2: digits of fine-tuning per extra echo
    y0 += ph + 70
    dig = lambda x: delta(x) / (gamma(x) * math.log(10))
    p2 = XPanel(svg, px, y0, pw, ph, (1, 4.5), "Digits of fine-tuning that one more echo costs, Δ / (γ ln 10)",
                "decimal digits", [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5], xlabels=False)
    landmarks(p2, False)
    p2.curve(xs, dig(xs), c1, title="Delta / (gamma ln 10) from the rational formulas")
    for D in (4, 6, 10, 20):
        x = 1 / (D - 2)
        marker(svg, p2.sx(x), p2.sy(dig(x)), c1, f"D = {D}: gamma = {gamma(x):.4f}, {dig(x):.2f} digits per echo")
    dinf = DINF / (0.5 * math.log(10))
    marker(svg, p2.sx(0), p2.sy(dinf), c1, f"D = infinity: gamma = 1/2, {dinf:.3f} digits per echo")
    svg.text(p2.sx(0.5) - 10, p2.sy(dig(0.5)) + 18, f"{dig(0.5):.1f} at D = 4", size=11, fill=INK, anchor="end")
    svg.text(p2.sx(0) + 8, p2.sy(dinf) - 22, f"{dinf:.2f} at D = ∞", size=11, fill=INK)

    # Panel 3: field and gravity strength
    y0 += ph + 70
    p3 = XPanel(svg, px, y0, pw, ph, (0, 0.9), "How strong the field and its gravity are where the echo happens",
                "dimensionless", [0, 0.2, 0.4, 0.6, 0.8], yfmt="{:.1f}")
    landmarks(p3, False)
    p3.key("marker", c2, "field: first harmonic of the centre slope ∂p/∂x")
    p3.key("marker", c0, "gravity: 1 − ⟨c⟩ at the centre (0 in flat space)")
    p3.key("line", INK_SOFT, "large-D forms 2.72/√n and 3.65/n", dash="5 4")
    xl = np.linspace(1e-6, 0.06, 120)
    p3.curve(xl, 2.72 * np.sqrt(xl), c2, dash="5 4", title="2.72/sqrt(n)", width=1.5)
    p3.curve(xl, 3.65 * xl, c0, dash="5 4", title="3.65/n", width=1.5)
    for D, F1, g in MEAS:
        n = D - 2
        marker(svg, p3.sx(1 / n), p3.sy(F1 / math.sqrt(n)), c2,
               f"D = {D}: centre slope amplitude {F1 / math.sqrt(n):.3f} (sqrt(n) x it = {F1})")
        marker(svg, p3.sx(1 / n), p3.sy(g / n), c0, f"D = {D}: 1 - <c(0)> = {g / n:.4f} (n x it = {g})")
    svg.text(p3.sx(1 / 18) + 10, p3.sy(3.48 / math.sqrt(18)) + 4, "field", size=11, fill=INK)
    svg.text(p3.sx(1 / 18) + 10, p3.sy(3.021 / 18) + 4, "gravity", size=11, fill=INK)
    svg.text(p3.sx(0.5) - 8, p3.sy(0.45) , "D = 4: strong field, not measured here", size=10, anchor="end")
    svg.text(p3.sx(0.19), p3.sy(0.08), "at D = ∞ (left edge) both → 0: a free wave on flat space (Proposition 2)", size=10)
    H = y0 + ph + 52
    open(args.o, "w").write(svg.finish(H))
    print("wrote", args.o)


if __name__ == "__main__":
    main()
