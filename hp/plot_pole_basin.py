#!/usr/bin/env python3
"""Two views of T5's radius law (notes/QUAD_DOUBLE.md 13.6; hp/pole_test.py, raw folder route1-mac). Left: the
horizon series' radius R for horizon data a fraction delta off the solution, against log10(1/delta), at five of the
nine stored states, with the fitted lines R = R0 + C ln(1/delta)/(n - 1) and the matching distance u0 = 0.30 of the
standard layout: below that line the series summed at the last node diverges, so Newton cannot step. Right: the
basin threshold delta_max(u0) = exp(-(u0 - R0)(n - 1)/C) each line implies, for the three layouts (last node at
0.7, 0.8, 0.85), against D, with the extrapolation to D = 200, the walls each layout met (hp/data/large_d_walls.json),
the predictor's accuracy per unit step and the double-double resolution.

    cd hp && python3 plot_pole_basin.py RAWDIR [-o pole_basin.svg]      (RAWDIR: the raw-data repository's clone)
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_graphs import GRID, INK, INK_SOFT, SERIES, Svg, axes, better_hint, callout, caption, footnote, host_specs, legend, marker  # noqa: E402,E501

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
SHOW = (40.0, 50.0, 60.0, 67.7, 73.3)      # the five states of section 13.6's table (matched to the nearest stored D)
U0 = {"0.3": 0.3, "0.2": 0.2, "0.15": 0.15}
LAYOUT = {"0.3": "last node 0.7", "0.2": "last node 0.8", "0.15": "last node 0.85"}
PRED = (1e-9, 1e-6)                 # the predictor's error per unit step (QUAD_DOUBLE.md 12.6)
DD_EPS = 1e-32                      # double-double resolution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("-o", default=os.path.join(HERE, "pole_basin.svg"))
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(a.raw, "route1-mac", "pole_test.jsonl")) if l.strip()]
    pts = [r for r in recs if "delta" in r and r["delta"] > 0 and r.get("R")]
    lines = {r["D"]: r for r in recs if r.get("fit")}
    dmax = {r["D"]: r["delta_max"] for r in recs if "delta_max" in r and "D" in r}
    ext = next(r for r in recs if r.get("extrapolation"))
    walls = json.load(open(os.path.join(HERE, "data", "large_d_walls.json")))["walls"]
    near = lambda D: min(lines, key=lambda k: abs(k - D))
    SHOWN = [near(D) for D in SHOW]
    svg = Svg(1180)
    y = caption(svg, 60, 40, "Why the horizon series stops: its radius for data off the solution, and the basin it leaves Newton",
                "left: R = R₀ + C ln(1/δ)/(n − 1) at five D, against the matching distance u₀ = 0.30; right: the iterate accuracy Newton needs, against D")
    # ---- left: R against log10(1/delta)
    X, W, H = 60, 440, 360
    xm, ym = axes(svg, X, y, W, H, (0, 15.5), (0, 0.8), "log₁₀(1/δ), the horizon data's accuracy", "radius R of the horizon series (in u = 1 − x)",
                  xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    svg.line(xm(0), ym(0.3), xm(15.5), ym(0.3), INK, width=1.4)
    svg.text(xm(0.3), ym(0.3) - 6, "u₀ = 0.30: the series must converge here", size=10, fill=INK)
    svg.text(xm(0.3), ym(0.3) + 14, "below it: no Newton step", size=10)
    for i, D in enumerate(SHOWN):
        col = SERIES[i]
        ln = lines[D]
        x0, x1 = 0.5, 12.6
        f = lambda lg: ln["a0"] + ln["b"] * lg * math.log(10)
        svg.add(f'<line x1="{xm(x0):.1f}" y1="{ym(f(x0)):.1f}" x2="{xm(x1):.1f}" y2="{ym(f(x1)):.1f}" stroke="{col}" stroke-width="1.6" stroke-dasharray="6 4"/>')
        for r in pts:
            if r["D"] == D:
                marker(svg, "circle", xm(math.log10(1 / r["delta"])), ym(r["R"]), col,
                       f"D = {D:.2f}, δ = {r['delta']:.0e}: R = {r['R']:.3f} (line {f(math.log10(1 / r['delta'])):.3f})")
        svg.text(xm(x1) + 6, ym(f(x1)) + 4, f"D {D:.0f}", size=10, fill=INK)
    # the on-solution radii, at the double-double rounding
    for i, D in enumerate(SHOWN):
        r0 = next(r for r in recs if "delta" in r and r["D"] == D and r["delta"] == 0)
        marker(svg, "diamond", xm(14.8), ym(r0["R"]), SERIES[i], f"D = {D:.2f} on the solution: R = {r0['R']:.3f}")
    svg.text(xm(14.8), ym(0.785), "on the", size=9, anchor="middle")
    svg.text(xm(14.8), ym(0.76), "solution", size=9, anchor="middle")
    better_hint(svg, X, y, W, H, where="right", label="converges")
    legend(svg, X + W, y - 10, [(f"D {D:.0f}", SERIES[i]) for i, D in enumerate(SHOWN)])
    # ---- right: delta_max against D
    X2, W2 = 620, 440
    xm2, yl = axes(svg, X2, y, W2, H, (35, 205), (-33, -3), "dimension D",
                   "δ_max, how close the iterate must be for the series to converge (log₁₀)", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    ym2 = lambda v: yl(math.log10(v))
    # the predictor's band and the double-double floor
    svg.add(f'<rect x="{xm2(35):.1f}" y="{ym2(PRED[1]):.1f}" width="{xm2(205) - xm2(35):.1f}" height="{ym2(PRED[0]) - ym2(PRED[1]):.1f}" fill="{GRID}" opacity="0.8"/>')
    svg.text(xm2(203), ym2(PRED[1]) - 4, "predictor error per unit step, 10⁻⁹ to 10⁻⁶", size=10, anchor="end")
    svg.line(xm2(35), ym2(DD_EPS), xm2(205), ym2(DD_EPS), INK_SOFT, width=1.2)
    svg.text(xm2(203), ym2(DD_EPS) - 5, "double-double resolution", size=10, anchor="end")
    Ds = sorted(dmax)
    for i, k in enumerate(("0.3", "0.2", "0.15")):
        col = SERIES[i]
        path = " ".join(f"{'M' if j == 0 else 'L'}{xm2(D):.1f},{ym2(dmax[D][k]):.1f}" for j, D in enumerate(Ds))
        svg.add(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="1.6"/>')
        svg.add(f'<line x1="{xm2(Ds[-1]):.1f}" y1="{ym2(dmax[Ds[-1]][k]):.1f}" x2="{xm2(200):.1f}" y2="{ym2(ext["delta_max"][k]):.1f}" stroke="{col}" stroke-width="1.6" stroke-dasharray="6 4"/>')
        for D in Ds:
            marker(svg, "circle", xm2(D), ym2(dmax[D][k]), col, f"{LAYOUT[k]} (u₀ = {k}), D = {D:.2f}: δ_max = {dmax[D][k]:.1e}")
        marker(svg, "diamond", xm2(200), ym2(ext["delta_max"][k]), col, f"{LAYOUT[k]}, D = 200 (extrapolated): δ_max = {ext['delta_max'][k]:.1e}")
        svg.text(xm2(200) - 6, ym2(ext["delta_max"][k]) - 8, f"{ext['delta_max'][k]:.0e}", size=9, anchor="end", fill=INK)
    for name, key, dx, dy in (("0.7 pinned", "0.3", 40, 96), ("0.8 pinned", "0.2", 60, 56), ("0.85 pinned", "0.15", 80, 20)):
        Dw = walls[name]
        # the line's value at the wall, interpolated in log between the measured D
        lo = max(D for D in Ds if D <= Dw) if any(D <= Dw for D in Ds) else Ds[0]
        hi = min(D for D in Ds if D >= Dw) if any(D >= Dw for D in Ds) else None
        v = dmax[lo][key] if hi is None or hi == lo else math.exp(
            math.log(dmax[lo][key]) + (math.log(dmax[hi][key]) - math.log(dmax[lo][key])) * (Dw - lo) / (hi - lo))
        if hi is None:
            v = math.exp(math.log(dmax[Ds[-1]][key]) + (math.log(ext["delta_max"][key]) - math.log(dmax[Ds[-1]][key])) * (Dw - Ds[-1]) / (200 - Ds[-1]))
        callout(svg, xm2(Dw), ym2(v), f"{LAYOUT[key]}: wall at D {Dw:.1f}", dx=dx, dy=dy)
    better_hint(svg, X2, y, W2, H, where="right", label="easier")
    legend(svg, X2 + W2, y - 10, [(LAYOUT[k], SERIES[i]) for i, k in enumerate(("0.3", "0.2", "0.15"))])
    srcs = ["hp/pole_test.py on the nine stored series states (raw route1-mac/pole_test.jsonl)",
            "walls: hp/data/large_d_walls.json (hp/wall_model.py)"]
    height = footnote(svg, srcs, host_specs(), ["D = 200 is an extrapolation of the D 73 line with C fixed and R₀ scaled as 1/(n − 1)"], y + H + 50, REPO)
    open(a.o, "w").write(svg.finish(height))
    print("wrote", a.o)
    print("| D | R₀ | C | δ_max u₀ 0.30 | 0.20 | 0.15 |")
    print("|---|---|---|---|---|---|")
    for D in Ds:
        ln = lines[D]
        print(f"| {D:.2f} | {ln['a0']:.3f} | {ln['C']:.2f} | {dmax[D]["0.3"]:.1e} | {dmax[D]["0.2"]:.1e} | {dmax[D]['0.15']:.1e} |")
    print(f"| 200 (extrapolated) | {ext['a0']:.4f} | {ext['C']:.2f} | {ext["delta_max"]["0.3"]:.1e} | {ext["delta_max"]["0.2"]:.1e} | {ext['delta_max']['0.15']:.1e} |")


if __name__ == "__main__":
    main()
