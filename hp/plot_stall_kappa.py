#!/usr/bin/env python3
"""stall_kappa.svg: where double-double runs out (README, "Why the solver stalls"), from the stall detector's runs on the
folder T76 of the raw-data repository (hp/stall_margin.py).

Top: u·κ, double-double's unit roundoff u = 2^-104 times the Jacobian's condition number σ_max/σ_min, at converged
states: the full system (kfreeze 13) at D 40-67.74 (saved states) and at 70.98, with its power-law fit in n = D - 2,
and the pinned systems the runs solve (T73, T75: p1's harmonics above 9, 7 or 5 held), with the stalls at 60.96 and
67.75 marked. Bottom: the residual each full-system state reached against u·σ_max·|z|, the rounding floor of
evaluating F at a double-double state, scaled by their median ratio. Standard library and numpy; the primitives are
data_graphs.py, vendored unchanged from github.com/nmoinvaz/speedy-gonzales (skills/graph-data). Prints the numbers.

    git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw   (beside the repository, once)
    cd hp && python3 plot_stall_kappa.py ../../choptuik-large-d-raw/T76
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

from data_graphs import GRID, INK, SERIES, Axis, Svg, axes, better_hint, callout, caption, footnote, legend, marker
from plot_newton_stall import polyline

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/nmoinvaz/choptuik-large-d"
WIDTH, X, W = 1080, 90, 900
U = 2.0 ** -104
PINS = ((None, "full system (harmonics to 13 solved)"), (9, "pin 9 (11-13 held)"), (7, "pin 7 (9-13 held)"),
        (5, "pin 5 (7-13 held)"))


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pow10(v):
    """tick label for log10 = v (the log axes are drawn as linear axes in log10, one tick per decade)"""
    return "10" + str(int(round(v))).translate(SUP)


def decades(svg, x, w, yl, lo, hi):
    """grid lines and labels for the decades axes() left out (it picks a 5-decade step over a wide range)"""
    drawn = set(Axis(lo, hi, 0, 1).ticks())
    for v in range(lo, hi + 1):
        if v not in drawn:
            svg.line(x, yl(v), x + w, yl(v), GRID)
            svg.text(x - 6, yl(v) + 4, pow10(v), size=11, anchor="end", tabular=True)


def jl(path):
    return [json.loads(x) for x in open(path) if x.strip()]


def main():
    if len(sys.argv) < 2 or not os.path.isfile(os.path.join(sys.argv[1], "kappa_states.jsonl")):
        sys.exit(__doc__)
    raw = sys.argv[1]
    rev = subprocess.run(["git", "-C", raw, "log", "-1", "--format=%h", "--", ".", ":(exclude)README.md"], capture_output=True, text=True).stdout.strip()
    recs = [r for f in ("kappa_states.jsonl", "smoke_kappa.jsonl", "kappa_pinned.jsonl", "t75_watch.jsonl")
            for r in jl(os.path.join(raw, f))]
    pts = {}
    for r in recs:
        w = r["dirs"][0]
        pts[(r["pin"], round(r["D"], 4))] = dict(D=r["D"], pin=r["pin"], uk=w["u_kappa"], smax=w["sigma_max"],
                                                 sig=w["sigma"], F=r["F"], state=r["state"], qdh=r.get("qdh", 0))
    by = {p: sorted((v for v in pts.values() if v["pin"] == p), key=lambda v: v["D"]) for p, _ in PINS}
    full = by[None]
    fit = [v for v in full if v["D"] <= 67.8]
    b, a = np.polyfit(np.log([v["D"] - 2 for v in fit]), np.log([v["uk"] for v in fit]), 1)
    cross = math.exp(-a / b) + 2

    svg = Svg(WIDTH)
    svg.text(16, 28, "Where double-double runs out: u·κ against D", size=15, fill=INK, weight="bold")

    # top: u kappa
    legend(svg, WIDTH - 16, 58, [(lab, SERIES[i]) for i, (_, lab) in enumerate(PINS)])
    y = caption(svg, X, 58, "u·κ at converged states",
                f"u = 2⁻¹⁰⁴ (double-double) times σ_max/σ_min. Above 1 a double-double solve cannot resolve the weakest "
                f"direction. Dashed: the full system's fit, n^{b:.1f} (n = D − 2), reaching 1 at D = {cross:.1f}.")
    h = 330
    xm, yl = axes(svg, X, y, W, h, (38, 75), (-9, 4), "dimension D", "u·κ (log scale)", xfmt=lambda v: f"{v:g}",
                  yfmt=pow10)
    ym = lambda v: yl(math.log10(v))  # noqa: E731
    decades(svg, X, W, yl, -9, 4)
    better_hint(svg, X, y, W, h, "right", label="better", reverse=True)
    svg.line(X, ym(1.0), X + W, ym(1.0), INK, width=1.2)
    svg.text(X + 8, ym(1.0) - 6, "u·κ = 1", size=11, fill=INK)
    dd = np.linspace(38, 75, 80)
    polyline(svg, [(xm(d), ym(math.exp(a + b * math.log(d - 2)))) for d in dd
                   if 1e-9 <= math.exp(a + b * math.log(d - 2)) <= 1e4], SERIES[0], 1.4, "6 4")
    print("u*kappa by system (D, sigma_min, sigma_max, u*kappa)")
    for i, (p, lab) in enumerate(PINS):
        for v in by[p]:
            marker(svg, "circle" if p is None else ("diamond", "square", "triangle")[i - 1], xm(v["D"]), ym(v["uk"]),
                   SERIES[i], f"{lab}, D = {v['D']:.4f}: u·κ {v['uk']:.2e}, σ_min {v['sig']:.2e}, σ_max {v['smax']:.2e}")
            print(f"  {lab:38s} D {v['D']:8.4f}  {v['sig']:9.2e}  {v['smax']:9.2e}  {v['uk']:9.2e}")
    s60 = min(full, key=lambda v: abs(v["D"] - 60.97))
    s67 = min(full, key=lambda v: abs(v["D"] - 67.74))
    callout(svg, xm(s60["D"]), ym(s60["uk"]), f"D = {s60['D']:.2f}: the stall that holding harmonics on their trend cured",
            dx=-60, dy=-46)
    callout(svg, xm(s67["D"]), ym(s67["uk"]), f"D = {s67['D']:.2f}: T48 stopped (u·κ {s67['uk']:.0f})", dx=-40, dy=-40)
    p5 = by[5][0]
    callout(svg, xm(p5["D"]), ym(p5["uk"]), "pinning lowers u·κ by 8-11 orders", dx=-150, dy=30)

    # bottom: the residual floor
    y2 = y + h + 92
    rows = []
    for v in full:
        if v["qdh"]:
            continue                      # the floor comparison uses the saved states' own (double-double) residual
        zh = np.load(v["state"] if os.path.isabs(v["state"]) else os.path.join(HERE, v["state"]))["zh"]
        rows.append((v["D"], v["F"], U * v["smax"] * float(np.linalg.norm(zh))))
    ratio = float(np.median([F / e for _, F, e in rows]))
    legend(svg, WIDTH - 16, y2, [("residual the state reached", SERIES[0]), (f"{ratio:.3f} × u·σ_max·|z|", SERIES[-1])])
    yb = caption(svg, X, y2, "The residual floor is rounding: it follows u·σ_max·|z|",
                 f"Each saved full-system state's |F| against the rounding floor of evaluating F at a double-double "
                 f"state, scaled by their median ratio {ratio:.3f}; σ_max grows 1.36 times per unit of D at D = 60.")
    h2 = 270
    xm2, yl2 = axes(svg, X, yb, W, h2, (38, 75), (-27, -19), "dimension D", "|F| (log scale)",
                    xfmt=lambda v: f"{v:g}", yfmt=pow10)
    ym2 = lambda v: yl2(math.log10(v))  # noqa: E731
    decades(svg, X, W, yl2, -27, -19)
    better_hint(svg, X, yb, W, h2, "right", label="better", reverse=True)
    polyline(svg, [(xm2(d), ym2(ratio * e)) for d, _, e in rows], SERIES[-1], 1.6, "5 4")
    print(f"\nresidual floor (D, |F|, u*sigma_max*|z|, ratio); median ratio {ratio:.3f}")
    for d, F, e in rows:                  # the floor first, so the measured point stays on top where they meet
        marker(svg, "triangle", xm2(d), ym2(ratio * e), SERIES[-1], f"D = {d:.4f}: {ratio:.3f} × u·σ_max·|z| = {ratio * e:.1e}")
        marker(svg, "circle", xm2(d), ym2(F), SERIES[0], f"D = {d:.4f}: |F| {F:.1e}, u·σ_max·|z| {e:.1e}, ratio {F / e:.3f}")
        print(f"  {d:8.4f}  {F:8.1e}  {e:8.1e}  {F / e:6.3f}")
    hi = max(rows, key=lambda r: r[1] / r[2])
    callout(svg, xm2(hi[0]), ym2(hi[1]), f"D = {hi[0]:.1f}: {hi[1] / hi[2]:.2f}, above the floor", dx=-70, dy=-30)

    height = footnote(svg, ["hp/stall_margin.py runs on raw/T76 " + (rev or "?"),
                            "states: hp/data/states, seeds, large_d/up70_* (D 40-67.74); raw/T73, raw/T75 (D 70.6-73)"],
                      "runs: cloud, Windows i7-1185G7, Mac Apple M5",
                      ["full-system points at D ≤ 67.74 with the double-double horizon, at 70.98 and the pinned ones "
                       "with the quad-double horizon"], yb + h2 + 46, repo_url=REPO)
    with open(os.path.join(HERE, "stall_kappa.svg"), "w") as fh:
        fh.write(svg.finish(height))


if __name__ == "__main__":
    main()
