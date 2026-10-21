#!/usr/bin/env python3
"""Why gamma_inf = 1/2: an SVG with two panels.

1. The growth rate lambda = 1/gamma of the unstable mode, computed in double-double at 67 points (D = 8.25-61;
   hp/data/gamma_dd.jsonl and hp/data/local/gamma_dd.jsonl), against 1/n = 1/(D-2), with a fit in powers of 1/n with
   lambda_inf = 2 fixed (D >= 16, 9 powers; drawn over its fit range) carried to 1/n = 0, and its leading term 2 + a1/n. At D = infinity the
   lambda = 2 mode is exact: the r^2-shift symmetry of the inner heat equation (GAMMA.md).
2. Free fits, lambda_inf left free: lambda_inf - 2 against the number of powers of 1/n, for three fit ranges
   (hp/data/local/gamma_compare.txt); it falls about 3.5-fold per added power, toward 0.

    python3 plot_gamma.py [-o gamma.svg]
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_delta import GRID, INK, INK_SOFT, SERIES, Svg, marker  # noqa: E402
from plot_selection import Axes  # noqa: E402

ROOT = os.path.join(HERE, "..")


def load():
    pts = {}
    for fn in ("hp/data/gamma_dd.jsonl", "hp/data/local/gamma_dd.jsonl"):
        for line in open(os.path.join(ROOT, fn)):
            r = json.loads(line)
            if r.get("lam"):
                pts[round(r["D"], 4)] = r["lam"]
    D = np.array(sorted(pts))
    return D, np.array([pts[d] for d in D])


def free_fits():
    out = {}
    for line in open(os.path.join(ROOT, "hp/data/local/gamma_compare.txt")):
        m = re.match(r"D >= (\d+) \(\d+ points\), (\d) powers of 1/n: lam_inf - 2 = ([+-][\d.e+-]+) \+- ([\d.e+-]+)", line)
        if m:
            out.setdefault(int(m.group(1)), []).append((int(m.group(2)), float(m.group(3)), float(m.group(4))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "gamma.svg"))
    args = ap.parse_args()
    D, lam = load()
    n = D - 2
    sel = D >= 16
    A = np.vander(1 / n[sel], 10, increasing=True)[:, 1:]
    c, *_ = np.linalg.lstsq(A, lam[sel] - 2, rcond=None)
    fit = lambda x: 2 + sum(ci * x ** (i + 1) for i, ci in enumerate(c))
    c0 = SERIES[0]

    W = 900
    svg = Svg(W)
    svg.text(78, 26, "Why the critical exponent tends to 1/2", size=15, fill=INK, weight="bold")
    svg.text(78, 46, "Growth rate λ = 1/γ of the unstable mode; hover a mark for its value", size=11)

    p = Axes(svg, 78, 120, 470, 300, (0, 0.165), (1.96, 2.24),
             [(x, f"{x:g}") for x in (0, 0.04, 0.08, 0.12, 0.16)], [(v, f"{v:.2f}") for v in (2.0, 2.05, 2.1, 2.15, 2.2)],
             f"λ against 1/n: {len(D)} double-double points, D = {D.min():g}–{D.max():.0f}", "1/n = 1/(D − 2)", "λ = 1/γ")
    xs = np.linspace(0, 0.162, 200)
    svg.path([(p.sx(x), p.sy(2 + c[0] * x)) for x in xs if 2 + c[0] * x < 2.24], INK_SOFT, width=1.3, dash="5 4",
             title=f"leading term 2 + {c[0]:.4f}/n")
    svg.path([(p.sx(x), p.sy(fit(x))) for x in xs if x <= 1 / 14], c0, width=2,
             title="fit with lambda_inf = 2 fixed (D >= 16, 9 powers of 1/n)")
    for d, l in zip(D, lam):
        marker(svg, p.sx(1 / (d - 2)), p.sy(l), c0, f"D = {d:g}: lambda = {l:.12f}, gamma = {1 / l:.12f}", r=3.2)
    marker(svg, p.sx(0), p.sy(2), SERIES[1], "D = infinity: lambda = 2 exactly (r^2-shift symmetry)", shape="diamond", r=4.5)
    svg.text(p.sx(0.004), p.sy(2) + 18, "D = ∞: λ = 2 exactly, from the r²-shift", size=10, fill=INK)
    svg.text(p.sx(0.004), p.sy(2) + 31, "symmetry of the inner heat equation", size=10, fill=INK)
    svg.text(p.sx(0.05) - 8, p.sy(2 + c[0] * 0.05), f"leading term 2 + {c[0]:.4f}/n", size=10, fill=INK, anchor="end")
    p.key(lambda x, y: marker(svg, x, y, c0, "data", r=3.2), "computed λ")
    p.key(lambda x, y: svg.line(x - 6, y, x + 10, y, c0, width=2), "fit, λ∞ = 2 fixed")
    p.key(lambda x, y: marker(svg, x, y, SERIES[1], "symmetry", shape="diamond", r=3.5), "symmetry value")

    ff = free_fits()
    q = Axes(svg, 640, 120, 230, 300, (3.5, 8.5), (-6.5, -2.8), [(k, str(k)) for k in range(4, 9)],
             [(10.0 ** e, f"10{'⁻' + '⁰¹²³⁴⁵⁶⁷'[-e]}") for e in (-6, -5, -4, -3)],
             "λ∞ left free", "powers of 1/n in the fit", "fitted λ∞ − 2", ylog=True)
    for i, (dmin, rows) in enumerate(sorted(ff.items())[1:]):
        col = SERIES[[2, 3, 4][i]]
        pts = [(q.sx(k), q.sy(v)) for k, v, e in rows if v > 0]
        svg.path(pts, col, width=2)
        for (k, v, e), (x, y) in zip(rows, pts):
            marker(svg, x, y, col, f"D >= {dmin}, {k} powers: lambda_inf - 2 = {v:.2e} +- {e:.1e}")
        k, v, _ = rows[-1]
        svg.text(q.sx(k) + 8, q.sy(v) + 4, f"D ≥ {dmin}", size=10, fill=INK)
    svg.text(q.px, q.py + q.ph + 52, "each added power divides it by about 3.5", size=10)
    H = 120 + 300 + 70
    open(args.o, "w").write(svg.finish(H))
    print("wrote", args.o, "coefficients with lambda_inf = 2:", np.round(c, 4))


if __name__ == "__main__":
    main()
