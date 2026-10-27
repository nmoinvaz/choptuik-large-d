#!/usr/bin/env python3
"""light_cone.svg: the past light cone of the singularity and the fast mode's size inside it (notes/BEYOND_LIGHT_CONE.md).

Left, a spacetime diagram in units of the singularity's time t* (exact geometry, no data): the singularity at r = 0,
t = t*, its past light cone r = t* - t, which is the self-similar coordinate x = r / (t* - t) = 1, and rays of constant
x: the solver's nodes x = 0.3 .. 0.7 (it matches to the horizon series at the last, 0.7) and x = 1/sqrt 2. Right, the
modulus of the fast mode exp(-n Phi(x)), Phi = ln x - (1/2) ln(1 - x^2), along the real axis for D = 20, 40 and 60
(n = D - 2): exact formulas. It is 1 at x = 1/sqrt 2 for every n, so the effect that selects Delta is of order one
where the solver imposes its condition; it is exponentially small only near the light cone. Standard library only;
the primitives are data_graphs.py, vendored unchanged from github.com/nmoinvaz/speedy-gonzales (skills/graph-data).
Prints the numbers.

    cd hp && python3 plot_light_cone.py
"""
import math
import os

from data_graphs import INK, SERIES, Svg, axes, callout, caption, footnote, legend, marker
from plot_newton_stall import polyline

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/nmoinvaz/choptuik-large-d"
NODES = (0.3, 0.4, 0.5, 0.6, 0.7)
X_ONE = 1 / math.sqrt(2)
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def phi(x):
    return math.log(x) - 0.5 * math.log(1 - x * x)


def main():
    svg = Svg(1080)
    svg.text(16, 28, "The past light cone, and the fast mode's size inside it", size=15, fill=INK, weight="bold")

    # left: spacetime diagram
    X0, W0, h = 90, 400, 400
    y = caption(svg, X0, 58, "Spacetime near the singularity",
                "In units of t*; each ray from the singularity is one value of x.")
    xm, ym = axes(svg, X0, y, W0, h, (0, 1.6), (0, 1.0), "radius r (units of t*)", "time t (units of t*)",
                  xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    inside = [(xm(0), ym(1.0)), (xm(1.0), ym(0.0)), (xm(0), ym(0.0))]
    svg.add(f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in inside)}" fill="{SERIES[0]}" '
            f'fill-opacity="0.10"/>')
    polyline(svg, [(xm(0), ym(1.0)), (xm(1.0), ym(0.0))], INK, 2.2)
    for x in NODES:
        polyline(svg, [(xm(0), ym(1.0)), (xm(x), ym(0.0))], SERIES[0], 1.2, "4 3")
    polyline(svg, [(xm(0), ym(1.0)), (xm(X_ONE), ym(0.0))], SERIES[1], 1.8)
    marker(svg, "diamond", xm(0), ym(1.0), INK, "the singularity: r = 0, t = t*")
    svg.text(xm(0.06), ym(0.97), "singularity (r = 0, t = t*)", size=11, fill=INK)
    svg.text(xm(1.04), ym(0.06), "x = 1: the past light cone", size=11, fill=INK)
    svg.text(xm(1.2), ym(0.55), "x > 1", size=12, fill=INK)
    svg.text(xm(1.2), ym(0.55) + 16, "outside: no signal", size=10)
    svg.text(xm(1.2), ym(0.55) + 30, "reaches the singularity", size=10)
    svg.text(xm(0.03), ym(0.2), "x < 1: inside, the solver's region", size=11, fill=INK)
    callout(svg, xm(0.7 * 0.35), ym(0.65), "x = 0.3 .. 0.7: the solver's nodes (matching at 0.7)", dx=150, dy=-10)
    svg.text(xm(X_ONE) + 4, ym(0.03), "x = 1/√2", size=10, fill=SERIES[1])

    # right: the fast mode's modulus along x
    X1, W1 = 600, 420
    legend(svg, 1064, 58, [(f"D = {d}", SERIES[i + 2]) for i, d in enumerate((20, 40, 60))])
    y1 = caption(svg, X1, 58, "The fast mode's size, |e^(−nΦ(x))|",
                 "Φ = ln x − ½ ln(1 − x²), n = D − 2; log scale.")
    xm1, yl1 = axes(svg, X1, y1, W1, h, (0.2, 1.0), (-20, 20), "self-similar coordinate x = r / (t* − t)",
                    "|e^(−nΦ)| (log scale)", xfmt=lambda v: f"{v:g}",
                    yfmt=lambda v: "10" + str(int(round(v))).translate(SUP))
    ym1 = yl1                             # log10 values
    svg.line(xm1(X_ONE), y1, xm1(X_ONE), y1 + h, SERIES[1], width=1.6)
    svg.line(xm1(0.7), y1, xm1(0.7), y1 + h, SERIES[0], width=1.2)
    svg.line(X1, ym1(0), X1 + W1, ym1(0), INK, width=1)
    print("x, log10 |exp(-n Phi(x))| for D = 20, 40, 60")
    for i, d in enumerate((20, 40, 60)):
        n = d - 2
        xs = [0.2 + 0.79 * k / 300 for k in range(301)]
        pts = [(x, -n * phi(x) / math.log(10)) for x in xs]
        polyline(svg, [(xm1(x), ym1(v)) for x, v in pts if -20 <= v <= 20], SERIES[i + 2], 2)
    for x in (0.3, 0.5, 0.7, X_ONE, 0.9, 0.99):
        print(f"  x {x:.4f}: " + ", ".join(f"{-(d - 2) * phi(x) / math.log(10):+8.3f}" for d in (20, 40, 60)))
    callout(svg, xm1(X_ONE), ym1(0), "size 1 at x = 1/√2 for every D", dx=30, dy=-60)
    callout(svg, xm1(0.95), ym1(-18 * phi(0.95) / math.log(10)), "exponentially small near the light cone", dx=-300,
            dy=60)
    svg.text(xm1(0.7) - 4, y1 + 14, "solver matches at 0.7", size=10, anchor="end")

    height = footnote(svg, ["exact formulas: x = r / (t* − t), Φ(x) = ln x − ½ ln(1 − x²)",
                            "notes/BEYOND_LIGHT_CONE.md §2.6"], "", ["no data: both panels are exact formulas"],
                      y + h + 50, repo_url=REPO)
    with open(os.path.join(HERE, "light_cone.svg"), "w") as fh:
        fh.write(svg.finish(height))


if __name__ == "__main__":
    main()
