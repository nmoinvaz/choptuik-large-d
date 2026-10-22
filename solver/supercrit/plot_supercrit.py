#!/usr/bin/env python3
"""The supercritical branch near beta^2 x^2 = 1 (notes/SUPERCRITICAL.md, section 3): the layer's front, the lapse
collapsing behind it, and the exact self-similar (CSS) equations at n = 100-6400 approaching the layer.

Data: the layer ODE integrated by layer.run (theta0 = 40, as in logs/layer.log), and the boundary-value results in
logs/css_bvp_b1_f0.8.log and logs/css_bvp_b1.5_f1.0.log ((n-1)(b-1)/(kappa sqrt n) at theta = -6, behind the front).

    cd solver/supercrit && python3 plot_supercrit.py [-o ../supercrit.svg]
"""
import argparse
import contextlib
import io
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from plot_delta import INK, INK_SOFT, SERIES, Panel, Svg, marker  # noqa: E402

LAYER = {"__name__": "layer_src", "__file__": os.path.join(HERE, "layer.py")}
with contextlib.redirect_stdout(io.StringIO()):   # layer.py runs its own check at module level and calls sys.exit
    try:
        exec(compile(open(LAYER["__file__"]).read(), LAYER["__file__"], "exec"), LAYER)
    except SystemExit:
        pass


def bvp(name):
    rows = []
    for line in open(os.path.join(HERE, "logs", name)):
        m = re.search(r"n =\s+(\d+):.*\(n-1\)\(b-1\)/\(kappa sqrt n\) = ([0-9.]+), 1 - 1/b = ([0-9.]+)", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "..", "supercrit.svg"))
    args = ap.parse_args()
    sol = LAYER["run"](40.0)
    th = np.linspace(-8, 10, 400)
    Y, S, C = sol.sol(th)
    yinf = float(sol.sol(-25.0)[0])
    i = np.argmin(np.abs(C * Y - math.sqrt(2) / 2))
    thf = th[i]
    cY, cC, cB = SERIES[0], SERIES[1], SERIES[2]

    W = 900
    svg = Svg(W)
    svg.text(78, 26, "The supercritical branch: a moving front, not an apparent horizon", size=15, fill=INK,
             weight="bold")
    svg.text(78, 46, "Near β²x² = 1 the 1/n series breaks down and a front of Rozali–Way's D = ∞ equations takes over "
             "(θ: the layer coordinate; front moving to the left)", size=11)

    p = Panel(svg, 78, 120, 230, 330, (-8, 10), (0, 2.2), "Field Y₁ across the front", "θ", "Y₁")
    p.curve(th, Y, cY, width=2.2)
    tl = th[th > 0.4]
    p.curve(tl, 1 / np.sqrt(tl), INK_SOFT, dash="5 4", width=1.4)
    svg.line(p.px, p.sy(yinf), p.px + p.pw, p.sy(yinf), INK_SOFT, width=1.2, dash="6 4")
    svg.text(p.sx(-7.6), p.sy(yinf) - 7, f"behind: {yinf:.4f}", size=10, fill=INK)
    svg.text(p.sx(9.6), p.sy(1 / math.sqrt(9.6)) - 10, "outer series 1/√θ", size=10, fill=INK, anchor="end")
    svg.line(p.sx(thf), p.py, p.sx(thf), p.py + p.ph, SERIES[3], width=1.2, dash="3 3")
    svg.text(p.sx(thf) + 4, p.py + 14, "front", size=10, fill=INK)

    q = Panel(svg, 378, 120, 230, 330, (-8, 10), (-5.5, 1.0), "Lapse C₁ collapses behind it", "θ",
              "log₁₀ C₁", yfmt="{:g}")
    q.curve(th, np.log10(C), cC, width=2.2)
    q.curve(tl, np.log10(np.sqrt(2 * tl)), INK_SOFT, dash="5 4", width=1.4)
    svg.line(q.sx(thf), q.py, q.sx(thf), q.py + q.ph, SERIES[3], width=1.2, dash="3 3")
    svg.text(q.sx(9.6), q.sy(math.log10(math.sqrt(19.2))) + 16, "outer series √(2θ)", size=10, fill=INK, anchor="end")
    svg.text(q.sx(-7.6), q.sy(0.15), "behind: C₁ ~ e^(1.63θ)", size=10, fill=INK)
    svg.text(q.sx(-7.6), q.sy(0.15) + 14, "no horizon", size=10, fill=INK)

    r = Panel(svg, 678, 120, 190, 330, (1.8, 4.0), (1.55, 1.9), "Exact CSS equations at finite n", "log₁₀ n",
              "(n − 1)(b − 1)/(κ√n)", yfmt="{:.2f}")
    target = yinf ** 2 / 2
    svg.line(r.px, r.sy(target), r.px + r.pw, r.sy(target), INK_SOFT, width=1.2, dash="6 4")
    svg.text(r.px + r.pw - 2, r.sy(target) + 14, f"layer: {target:.3f}", size=10, fill=INK, anchor="end")
    for name, col, lab in (("css_bvp_b1_f0.8.log", cB, "κ = 0.8"), ("css_bvp_b1.5_f1.0.log", SERIES[4], "κ = 1.5")):
        rows = bvp(name)
        r.curve([math.log10(n) for n, _, _ in rows], [v for _, v, _ in rows], col, width=1.6)
        for n, v, h in rows:
            marker(svg, r.sx(math.log10(n)), r.sy(v), col, f"{lab}, n = {n}: (n−1)(b−1)/(κ√n) = {v}; 1 − 1/b = {h}",
                   r=4.5)
        n, v, _ = rows[0]
        svg.text(r.sx(math.log10(n)) + 8, r.sy(v) + (-8 if col == SERIES[4] else 16), lab, size=10, fill=INK)
    open(args.o, "w").write(svg.finish(120 + 330 + 56))
    print("wrote", args.o, "Y1(-inf)", yinf, "front theta", thf, "target", target)


if __name__ == "__main__":
    main()
