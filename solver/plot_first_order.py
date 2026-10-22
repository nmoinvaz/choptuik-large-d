#!/usr/bin/env python3
"""The first 1/D corrections from the O(1/n) inner problem, keeping M harmonics, against the values fitted to the
computed solutions (notes/FIRST_ORDER.md), with the echo frequency itself for comparison.

Data: solver/inner_pert/data/first_order_hier_M{1,3,5}*.json (omega_1 and lambda_1; Delta_1 = -2 pi omega_1/omega*^2;
M = 5 the mean of its three runs), the fitted Delta_1 and omega_1 (inner_pert/logs/first_order_fits.log, centre of the
ranges), lambda_1 = 3.0324 from the double-double gamma (hp/data/local/gamma_compare.txt), and the zero omega*/omega_inf
for M = 1-7 (plot_selection.TRUNC).

    python3 plot_first_order.py [-o first_order.svg]
"""
import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_delta import INK, INK_SOFT, SERIES, Svg, marker  # noqa: E402
from plot_selection import TRUNC, Axes  # noqa: E402

D = os.path.join(HERE, "inner_pert", "data")


def run(name):
    return json.load(open(os.path.join(D, name + ".json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "first_order.svg"))
    args = ap.parse_args()
    r1, r3 = run("first_order_hier_M1_xi0.2"), run("first_order_hier_M3_xi0.2")
    r5 = [run(n) for n in ("first_order_hier_M5_xi0.2", "first_order_hier_M5_xi0.19", "first_order_hier_M5_xi0.2_rtol1e-13")]
    d1 = lambda r: -2 * math.pi * r["omega1"] / r["omega"] ** 2   # noqa: E731
    vals = {"ω₁": {1: r1["omega1"], 3: r3["omega1"], 5: sum(r["omega1"] for r in r5) / 3},
            "Δ₁": {1: d1(r1), 3: d1(r3), 5: sum(d1(r) for r in r5) / 3},
            "λ₁": {1: r1["lam1"], 3: r3["lam1"], 5: sum(r["lam1"] for r in r5) / 3}}
    fit = open(os.path.join(HERE, "inner_pert", "logs", "first_order_fits.log")).read()
    m = re.search(r"Delta_1 in \[([0-9.]+), ([0-9.]+)\], omega_1 in \[(-[0-9.]+), (-[0-9.]+)\]", fit)
    gl = next(l for l in open(os.path.join(HERE, "..", "hp", "data", "local", "gamma_compare.txt"))
              if l.startswith("D >= 30") and "fixed and 9 powers" in l)
    ref = {"ω₁": (float(m.group(3)) + float(m.group(4))) / 2, "Δ₁": (float(m.group(1)) + float(m.group(2))) / 2,
           "λ₁": float(re.search(r"a1 = ([0-9.]+)", gl).group(1))}
    cols = {"ω*/ω∞": SERIES[0], "ω₁": SERIES[3], "Δ₁": SERIES[1], "λ₁": SERIES[2]}

    W = 900
    svg = Svg(W)
    svg.text(78, 26, "The D = ∞ theory carried one order further: the 1/D corrections, keeping M harmonics", size=15,
             fill=INK, weight="bold")
    svg.text(78, 46, "Relative miss against the values measured on the computed solutions; each pair of harmonics "
             "cuts it 10–20 times. Hover for values.", size=11)
    p = Axes(svg, 78, 120, 790, 320, (0.5, 7.5), (-5, 0.4), [(k, f"M = {k}") for k in (1, 3, 5, 7)],
             [(10.0 ** e, {0: "100%", -1: "10%", -2: "1%", -3: "0.1%", -4: "0.01%", -5: "0.001%"}[e])
              for e in range(-5, 1)],
             "Computed from the D = ∞ equations, against the measured value", "harmonics kept", "relative miss (log scale)",
             ylog=True)
    series = {"ω*/ω∞": {k: v for k, v in TRUNC.items()}}
    series.update({k: {M: v / ref[k] for M, v in d.items()} for k, d in vals.items()})
    for name, d in series.items():
        pts = [(p.sx(M), p.sy(abs(v - 1))) for M, v in sorted(d.items())]
        svg.path(pts, cols[name], width=2)
        for (M, v), (x, y) in zip(sorted(d.items()), pts):
            raw = v if name == "ω*/ω∞" else v * ref[name]
            marker(svg, x, y, cols[name], f"{name}, M = {M}: {raw:.6g} ({(v - 1) * 100:+.3g}% from the measured value)")
    for name, txt in (("ω*/ω∞", "ω*/ω∞, the echo frequency"), ("ω₁", f"ω₁, its 1/D term ({ref['ω₁']:.3f})"),
                      ("Δ₁", f"Δ₁, 1/D term of Δ ({ref['Δ₁']:.3f})"), ("λ₁", f"λ₁, 1/D term of 1/γ ({ref['λ₁']:.4f})")):
        p.key(lambda x, y, c=cols[name], t=txt: marker(svg, x, y, c, t), txt)
    open(args.o, "w").write(svg.finish(120 + 320 + 56))
    print("wrote", args.o, {k: {M: round(v, 5) for M, v in d.items()} for k, d in vals.items()}, ref)


if __name__ == "__main__":
    main()
