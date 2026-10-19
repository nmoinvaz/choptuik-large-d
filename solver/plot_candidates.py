#!/usr/bin/env python3
"""Graph how much forcing each candidate constant as Delta_inf changes the fitted Delta(D) over the computed range.

For each candidate the degree-10 series in 1/D is refitted on 10 <= D <= 30.2 with c0 fixed to the candidate; the
graph shows |forced fit - free fit| against D on a log scale, over the shaded precision of the computed points.
A curve inside the shaded area is indistinguishable from the data; a curve above it misfits them.

Usage (from solver/data):
    python3 ../plot_candidates.py [-o ../candidates.svg]
"""
import argparse
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_fit import step_converged, weighted_series  # noqa: E402
from plot_delta import INK, INK_SOFT, GRID, BAND, SERIES, SURFACE, Svg, esc  # noqa: E402

FIT_MAX = 30.2
YLO, YHI = -13.0, -2.0           # log10 range


def candidates():
    j01, j11, j21 = (float(mp.besseljzero(n, 1)) for n in (0, 1, 2))
    G = float(mp.catalan)
    inside = [("5/7 √G j₀,₁", 5 / 7 * G ** .5 * j01), ("7/5 (ln 2)⁻² π^(−1/2)", 7 / 5 * math.log(2) ** -2 * math.pi ** -.5),
              ("7/2 √(ln 2/π)", 3.5 * (math.log(2) / math.pi) ** .5), ("9/8 √(j₂,₁/j₀,₁)", 9 / 8 * (j21 / j01) ** .5),
              ("10/9 √(j₀,₁/ln 3)", 10 / 9 * (j01 / math.log(3)) ** .5)]
    outside = [("π²/6", math.pi ** 2 / 6, SERIES[1]), ("2π/j₁,₁", 2 * math.pi / j11, SERIES[2]),
               ("√e", math.e ** .5, SERIES[3]), ("golden ratio φ", (1 + 5 ** .5) / 2, SERIES[4])]
    return inside, outside


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", default="../candidates.svg")
    args = ap.parse_args()

    D, y, s, _ = step_converged()
    k = (D >= 10) & (D <= FIT_MAX)
    free, _ = weighted_series(1 / D[k], y[k], 1 / s[k], 10)
    grid = np.linspace(10, 41, 600)
    f0 = np.polyval(free[::-1], 1 / grid)
    inside, outside = candidates()

    def curve(v):
        c, chi = weighted_series(1 / D[k], y[k], 1 / s[k], 10, v)
        return np.log10(np.maximum(np.abs(np.polyval(c[::-1], 1 / grid) - f0), 10 ** YLO)), chi

    W, px, py, pw, ph = 1080, 84, 132, 960, 470
    svg = Svg(W)
    svg.text(16, 28, "Candidate constants for Δ∞: how far each one moves Δ(D) over the computed range", size=15,
             fill=INK, weight="bold")
    svg.text(16, 48, "Degree-10 series in 1/D refitted on 10 ≤ D ≤ 30.2 with Δ∞ fixed to the candidate; curve = |forced "
                     "fit − free fit|. Shaded: precision of the computed Δ.", size=12)
    xs = lambda d: px + (d - 10) / (41 - 10) * pw
    ys = lambda v: py + ph - (v - YLO) / (YHI - YLO) * ph

    # precision of the computed points (the error model of final_fit), as a filled area from the bottom
    sig = np.array([s[np.argmin(np.abs(D - d))] for d in grid])
    area = [(xs(grid[0]), ys(YLO))] + [(xs(d), ys(math.log10(v))) for d, v in zip(grid, sig)] + [(xs(grid[-1]), ys(YLO))]
    svg.add('<path d="' + " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{yy:.1f}" for i, (x, yy) in enumerate(area))
            + f' Z" fill="{BAND}"/>')
    svg.text(xs(11), ys(-12.7), "shaded: within the precision of the computed points", size=11)
    # held-out range
    svg.line(xs(FIT_MAX), py, xs(FIT_MAX), py + ph, INK_SOFT, dash="3 4", opacity=0.6)
    svg.text(xs(FIT_MAX) + 6, py + 14, "fit ends, held-out points beyond", size=10)

    for e in range(int(YLO), int(YHI) + 1):
        svg.line(px, ys(e), px + pw, ys(e), GRID)
        svg.text(px - 8, ys(e) + 4, f"1e{e}", size=10, anchor="end")
    for d in range(10, 42, 5):
        svg.line(xs(d), py, xs(d), py + ph, GRID)
        svg.text(xs(d), py + ph + 16, f"{d}", size=10, anchor="middle")
    svg.line(px, py + ph, px + pw, py + ph, INK_SOFT)
    svg.text(px + pw / 2, py + ph + 38, "spacetime dimension D", size=11, anchor="middle")
    svg.add(f'<text x="{px - 56:.1f}" y="{py + ph / 2:.1f}" font-size="11" fill="{INK_SOFT}" text-anchor="middle" '
            f'transform="rotate(-90 {px - 56:.1f} {py + ph / 2:.1f})">|Δ with forced Δ∞ − Δ free fit|</text>')

    def draw(vals, color, title, dash=None, width=2):
        pts = " ".join(f"{'M' if i == 0 else 'L'}{xs(d):.1f},{ys(v):.1f}" for i, (d, v) in enumerate(zip(grid, vals)))
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        svg.add(f'<g><path d="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="0.9" '
                f'stroke-linejoin="round"{extra}/><title>{esc(title)}</title></g>')

    # legend (key line above the plot)
    kx, ky = px, py - 22
    dashes = [None, "8 3", "3 3", "10 3 2 3", "1 3"]
    for i, (name, v) in enumerate(inside):
        vals, chi = curve(v)
        draw(vals, SERIES[0], f"{name} = {v:.7f}: inside the window, chi2/dof {chi:.2f}", dash=dashes[i], width=1.6)
    svg.line(kx, ky - 4, kx + 22, ky - 4, SERIES[0], width=2)
    label = "5 contrived forms inside the window (dash styles; hover for each)"
    svg.text(kx + 28, ky, label, size=10)
    kx += 28 + 5.6 * len(label) + 18
    for name, v, color in outside:
        vals, chi = curve(v)
        draw(vals, color, f"{name} = {v:.7f}: chi2/dof {chi:.0f}")
        svg.line(kx, ky - 4, kx + 22, ky - 4, color, width=2)
        lab = f"{name} = {v:.5f}"
        svg.text(kx + 28, ky, lab, size=10)
        kx += 28 + 5.6 * len(lab) + 18
    svg.text(px, py - 48, "Inside the window every candidate stays within the data; the excluded ones climb out of it",
             size=12, fill=INK)

    svg.text(16, py + ph + 70, "Delta_inf window from all acceptable fits: 1.64388-1.64403. solver/candidates.py lists "
                               "the search; this graph: solver/plot_candidates.py.", size=10)
    open(args.output, "w").write(svg.finish(py + ph + 86))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
