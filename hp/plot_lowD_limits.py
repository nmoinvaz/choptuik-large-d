#!/usr/bin/env python3
"""The limit D -> 3 of the echoing period: the two readings on one graph (README, "A formula for 3 < D <= 4").
Left: Delta against s = sqrt(D - 3) over 3 < D <= 4; right: the same within D <= 3.0625, where the readings separate.
Ours: the polynomial in sqrt(D - 3) fitted to Ecker, Ecker, Grumiller & Jechtl's values (hp/lowD_formula.py; a
finite Delta(3), the fits' intercepts 1.72-1.80). Theirs (hp/data/published_lowD_limits.json): the power law
Delta = 3.88 (D - 3)^0.15 of their Eq. (110), the scaling-regime exponent 0.3 (drawn through their D = 3.05 value),
and Bland's D = 3.02 point as they cite it. Our double-double points from hp/data/delta_dd_lowD.csv, and with RAWDIR
given, T100's converged quarter-step point from T100-win2/down25e.jsonl.

    python3 plot_lowD_limits.py [RAWDIR] [-o lowD_limits.svg]
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_graphs import GRID, INK_SOFT, SERIES, Svg, axes, better_hint, callout, caption, footnote, host_specs, legend, marker  # noqa: E402,E501
from plot_lowD_sqrt import INTERCEPT, T100_TARGETS, points  # noqa: E402
from plot_lowD_values import C_DELTA, formula  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"


def readings():
    lim = json.load(open(os.path.join(HERE, "data", "published_lowD_limits.json")))
    e110 = lim["EEGJ2026_eq110"]
    sc, p305 = lim["EEGJ2026_scaling"], lim["EEGJ2026_D3.05"]
    c_sc = p305["Delta"] / (p305["D"] - 3) ** sc["alpha"]
    return {
        "ours": ("polynomial in √(D − 3), finite Δ(3)", lambda D: formula(C_DELTA, D)),
        "eq110": (f"Ecker et al. Eq. (110): {e110['a']} (D − 3)^{e110['alpha']}", lambda D: e110["a"] * (D - 3) ** e110["alpha"]),
        "scaling": (f"their scaling exponent {sc['alpha']}, through D = {p305['D']}", lambda D: c_sc * (D - 3) ** sc["alpha"]),
    }, lim


def t100_point(raw):
    """the converged quarter-step point of the T100 leg (D 3.044375), from its records"""
    if not raw:
        return None
    path = os.path.join(raw, "T100-win2", "down25e.jsonl")
    if not os.path.exists(path):
        return None
    best = None
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        # a converged pair record: Delta at the narrow resolution Mn with its low part, the residual F_n at the floor
        if "Delta" in r and "Mn" in r and r.get("F_n", 1.0) < 1e-20:
            if best is None or (r["Mn"], -abs(r["D"] - 3.044375)) > (best["Mn"], -abs(best["D"] - 3.044375)):
                best = r
    return (best["D"], best["Delta"] + best.get("Delta_lo", 0.0), best["Mn"]) if best else None


def curve(svg, xm, ym, f, s_lo, s_hi, color, dash=None, n=400):
    pts = []
    for i in range(n + 1):
        s = s_lo + (s_hi - s_lo) * i / n
        v = f(3 + s * s) if s > 0 else f(3 + 1e-30)
        if 0 <= v <= 3.6:
            pts.append(f"{xm(s):.1f},{ym(v):.1f}")
    d = " ".join(("M" if i == 0 else "L") + p for i, p in enumerate(pts))
    dasharray = f' stroke-dasharray="{dash}"' if dash else ""
    svg.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"{dasharray}/>')


def panel(svg, X, y, W, H, s_max, y_max, pts, rd, lim, t100, zoom):
    xm, ym = axes(svg, X, y, W, H, (0, s_max), (0, y_max), "s = √(D − 3)" + ("   (D = 3 to 3.0625)" if zoom else "   (D = 3 at the left, D = 4 at the right)"),
                  "echo period Δ", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    s0, s1 = (math.sqrt(d - 3) for d in T100_TARGETS)
    svg.add(f'<rect x="{xm(s0):.1f}" y="{y:.1f}" width="{xm(min(s1, s_max)) - xm(s0):.1f}" height="{H:.1f}" fill="{GRID}" opacity="0.7"/>')
    if zoom:
        svg.text((xm(s0) + xm(s1)) / 2, y + 16, "T100's planned points", size=10, anchor="middle")
    curve(svg, xm, ym, rd["ours"][1], 0.0, s_max, SERIES[-1], dash="6 4")
    curve(svg, xm, ym, rd["eq110"][1], 0.0, s_max, SERIES[2])
    curve(svg, xm, ym, rd["scaling"][1], 0.0, s_max, SERIES[3], dash="2 3")
    svg.line(xm(0.004), ym(INTERCEPT[0]), xm(0.004), ym(INTERCEPT[1]), SERIES[-1], width=4)
    for D, v, M, ok in pts:
        s = math.sqrt(D - 3)
        if s <= s_max:
            marker(svg, "circle" if ok else "diamond", xm(s), ym(v), SERIES[0] if ok else SERIES[1],
                   f"ours: D = {D} (s = {s:.4f}): Δ = {v:.12f} at M = {M}, {'resolved' if ok else 'not yet resolved'}")
    if t100 and math.sqrt(t100[0] - 3) <= s_max:
        marker(svg, "square", xm(math.sqrt(t100[0] - 3)), ym(t100[1]), SERIES[0], f"T100: D = {t100[0]}, Δ = {t100[1]:.12f} at M = {t100[2]}")
    p305, bl = lim["EEGJ2026_D3.05"], lim["Bland2007_D3.02"]
    sp = math.sqrt(p305["D"] - 3)
    if sp <= s_max:
        marker(svg, "triangle", xm(sp), ym(p305["Delta"]), SERIES[2], f"Ecker et al., text: D = {p305['D']}, Δ ≈ {p305['Delta']}")
    sb = math.sqrt(bl["D"] - 3)
    if sb <= s_max:
        svg.line(xm(sb), ym(bl["Delta"] - bl["err"]), xm(sb), ym(bl["Delta"] + bl["err"]), SERIES[4], width=2)
        marker(svg, "diamond", xm(sb), ym(bl["Delta"]), SERIES[4], f"Bland 2007 as cited by Ecker et al. (unverified): D = {bl['D']}, Δ = {bl['Delta']} ± {bl['err']}")
    return xm, ym, sb, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?", default="")
    ap.add_argument("-o", default=os.path.join(HERE, "lowD_limits.svg"))
    a = ap.parse_args()
    pts = points()
    rd, lim = readings()
    t100 = t100_point(a.raw)
    svg = Svg(1040)
    y = caption(svg, 60, 40, "The limit D → 3: the echoing period's two readings",
                "ours keeps Δ(3) finite (a square-root branch point); Ecker et al. read a power law Δ → 0; they separate below D ≈ 3.04")
    y += 24                                    # room for two legend rows above the panels
    X1, W1, X2, W2, H = 60, 450, 590, 330, 360
    panel(svg, X1, y, W1, H, 1.02, 3.6, pts, rd, lim, t100, zoom=False)
    xm, ym, sb, sp = panel(svg, X2, y, W2, H, 0.25, 3.6, pts, rd, lim, t100, zoom=True)
    better_hint(svg, X2, y, W2, H, where="axis", label="toward D = 3", reverse=True)
    # the zoom's callouts: where the readings sit at T100's far target, and the cited 3.02 point
    Dt = T100_TARGETS[0]
    ours, e110 = rd["ours"][1](Dt), rd["eq110"][1](Dt)
    callout(svg, xm(math.sqrt(Dt - 3)), ym(ours), f"at D = {Dt}: ours {ours:.2f}", dx=40, dy=-60)
    callout(svg, xm(math.sqrt(Dt - 3)), ym(e110), f"power law {e110:.2f}", dx=60, dy=40)
    bl = lim["Bland2007_D3.02"]
    callout(svg, xm(sb), ym(bl["Delta"]), f"Bland 2007 as cited: D = {bl['D']}, Δ = {bl['Delta']}", dx=-30, dy=-60)
    legend(svg, X2 + W2, y - 30, [("ours, resolved (circles) and T100 (square)", SERIES[0]), ("ours: " + rd["ours"][0], SERIES[-1]),
                                  ("Ecker et al. Eq. (110): 3.88 (D − 3)^0.15", SERIES[2])])
    legend(svg, X2 + W2, y - 10, [("their scaling exponent 0.3, through D = 3.05", SERIES[3]), ("Ecker et al., D = 3.05 (triangle)", SERIES[2]),
                                  ("Bland 2007 as cited (unverified)", SERIES[4])])
    srcs = ["hp/data/delta_dd_lowD.csv (hp/export_lowD.py)" + ("; T100-win2/down25e.jsonl" if t100 else ""),
            "formula: hp/lowD_formula.py on Ecker et al.'s values; their readings: hp/data/published_lowD_limits.json"]
    height = footnote(svg, srcs, host_specs(), [], y + H + 50, REPO)
    open(a.o, "w").write(svg.finish(height))
    print("wrote", a.o)
    print("| D | s | ours (√ polynomial) | Eq. (110) power law | scaling α 0.3 |")
    print("|---|---|---|---|---|")
    for D in (3.044375, 3.02, 3.01, 3.005, 3.001):
        print(f"| {D} | {math.sqrt(D - 3):.4f} | {rd['ours'][1](D):.4f} | {rd['eq110'][1](D):.4f} | {rd['scaling'][1](D):.4f} |")
    if t100:
        print(f"T100 point: D = {t100[0]}, Δ = {t100[1]:.12f} (M = {t100[2]})")


if __name__ == "__main__":
    main()
