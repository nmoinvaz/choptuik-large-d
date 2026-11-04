#!/usr/bin/env python3
"""The local exponent L(D) = (D - 3) Delta'/Delta below D = 3.3 (lowD_logslope.py's table): our double-double points,
the sqrt(D - 3) formula's L (which falls to 0 as 0.71 sqrt(D - 3)), and Ecker et al.'s power-law exponents, which
would hold L constant (0.15, read by them as a lower bound, and their scaling exponent 0.3). Prints the table.
    cd hp && python3 plot_lowD_logslope.py [-o lowD_logslope.svg]
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_graphs import INK_SOFT, SERIES, Svg, axes, better_hint, callout, caption, footnote, host_specs, legend, marker  # noqa: E402
from lowD_logslope import formula_L, table  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"


def main():
    out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else os.path.join(HERE, "lowD_logslope.svg")
    lim = json.load(open(os.path.join(HERE, "data", "published_lowD_limits.json")))
    a110, asc = lim["EEGJ2026_eq110"]["alpha"], lim["EEGJ2026_scaling"]["alpha"]
    rows = table(3.3)
    W, X, Y, PW, PH = 780, 80, 70, 560, 300
    svg = Svg(W)
    y0 = caption(svg, 16, 22, "The local exponent toward D = 3: a power law holds it constant, ours turns down",
                 "L = (D − 3) Δ′/Δ from neighbouring double-double points (dots), the √(D − 3) formula (curve), and Ecker et al.'s exponents")
    xm, ym = axes(svg, X, y0 + 10, PW, PH, (3.0, 3.3), (0.0, 0.34), "dimension D", "local exponent (D − 3) Δ′/Δ",
                  xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    dlo = min(r[0] for r in rows)
    for lo, hi, dash in ((3.0 + 1e-6, dlo, ' stroke-dasharray="4 3"'), (dlo, 3.3, "")):
        pts = [f"{xm(D):.1f},{ym(formula_L(D)):.1f}" for D in (lo + (hi - lo) * k / 200 for k in range(201))]
        svg.add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{SERIES[1]}" stroke-width="2"{dash}/>')
    svg.text(xm(3.004), ym(0.04), "dashed: no data", size=10, fill=INK_SOFT)
    for a, col, lab in ((a110, SERIES[2], f"Ecker et al.'s power law: L = {a110} (a lower bound to them)"),
                        (asc, SERIES[3], f"their scaling exponent: L = {asc}")):
        svg.add(f'<line x1="{xm(3.0):.1f}" y1="{ym(a):.1f}" x2="{xm(3.3):.1f}" y2="{ym(a):.1f}" stroke="{col}" stroke-width="2" stroke-dasharray="6 4"/>')
    for Dm, L, Lf in rows:
        marker(svg, "circle", xm(Dm), ym(L), SERIES[0], f"D {Dm:.6f}: L {L:.5f} (formula {Lf:.5f})")
    peak = max(rows, key=lambda r: r[1])
    low = min(rows, key=lambda r: r[0])
    callout(svg, xm(peak[0]), ym(peak[1]), f"peak {peak[1]:.3f} at D {peak[0]:.3f}", dx=34, dy=-26)
    callout(svg, xm(low[0]), ym(low[1]), f"{low[1]:.3f} at D {low[0]:.4f}, below 0.15", dx=40, dy=34)
    legend(svg, X + PW, y0 + 2, [("our points", SERIES[0]), ("√(D − 3) formula", SERIES[1]), (f"power law {a110}", SERIES[2]),
                                  (f"scaling {asc}", SERIES[3])])
    better_hint(svg, X, y0 + 10, PW, PH, where="axis", label="toward D = 3", reverse=True)
    srcs = ["points: hp/data/delta_dd_lowD.csv (largest M per D; hp/lowD_logslope.py)", "formula: hp/lowD_formula.py",
            "Ecker et al.: hp/data/published_lowD_limits.json"]
    height = footnote(svg, srcs, host_specs(), [], y0 + 10 + PH + 50, REPO)
    open(out, "w").write(svg.finish(height))
    print("wrote", out)
    print("| D (midpoint) | L measured | L of the formula |")
    print("|---|---|---|")
    for Dm, L, Lf in rows:
        print(f"| {Dm:.6f} | {L:.5f} | {Lf:.5f} |")


if __name__ == "__main__":
    main()
