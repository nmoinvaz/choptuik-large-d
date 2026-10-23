#!/usr/bin/env python3
"""All our values below D = 4: Delta(D) and gamma(D) against the sqrt(D - 3) formula (README, hp/lowD_formula.py, a
fit to Ecker, Ecker, Grumiller & Jechtl's published values, good to about 1e-5 in Delta and 1e-6 in gamma), and how
much Delta moves when the tau resolution M is raised, which is what stops the computation as D falls toward 3.

Data: data/delta_dd_lowD.csv (hp/export_lowD.py: every converged double-double record below D = 8.25, one row per
(D, M), with gamma where computed; the rows marked polluted are left out). At each D the value shown is the largest
clean M's. A point counts as resolved when the two largest M agree to 1e-8, or when it has M = 80 alone at D >= 3.7,
where M 80 -> 96 is 1.2e-9. The other points are drawn hollow: converged solutions at that M, not yet resolved.

    cd hp && python3 export_lowD.py && python3 plot_lowD_values.py [-o lowD_values.svg]
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "solver"))
from plot_delta import INK, INK_SOFT, SERIES, Panel, Svg, marker  # noqa: E402

# the sqrt(D - 3) fits of hp/lowD_formula.py (README, "A formula for 3 < D <= 4")
C_DELTA = [1.752403, 2.486429, 8.257623, -29.245836, 44.944706, -45.106964, 30.660868, -12.653060, 2.349289]
C_GAMMA = [0.081437, 0.285158, 0.634589, -1.439299, 1.261792, -0.547493, 0.097777]
RESOLVED = 1e-8


def formula(c, D):
    s = math.sqrt(D - 3)
    return sum(v * s ** k for k, v in enumerate(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "lowD_values.svg"))
    args = ap.parse_args()
    import csv
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_dd_lowD.csv"))) if r["status"] == "clean"]
    by_d = {}
    for r in rows:
        by_d.setdefault(float(r["D"]), {})[int(r["M"])] = (float(r["Delta_hi"]) + float(r["Delta_lo"]), r,
                                                           float(r["Delta_hi"]), float(r["Delta_lo"]))
    gam = {}
    for r in rows:
        if r["gamma"] and int(r["M"]) >= gam.get(float(r["D"]), (0, 0))[0]:
            gam[float(r["D"])] = (int(r["M"]), float(r["gamma"]))
    pts = []          # (D, Delta, M, change from the next lower M or None, resolved)
    steps = {}        # (M_lo, M_hi) -> [(D, |change|)]
    for D, ms in sorted(by_d.items()):
        Ms = sorted(ms)
        for a, b in zip(Ms, Ms[1:]):
            # in double-double: the hi parts' difference is exact for nearby values, so changes below 1e-16 survive
            steps.setdefault((a, b), []).append((D, abs((ms[b][2] - ms[a][2]) + (ms[b][3] - ms[a][3]))))
        top = Ms[-1]
        r = ms[top][1]
        err = abs(float(r["change_from_lower_M"])) if r["change_from_lower_M"] else None
        pts.append((D, ms[top][0], top, err, r["resolved"] == "1"))

    cR, cU, cF = SERIES[0], SERIES[3], INK_SOFT
    W = 1010
    svg = Svg(W)
    svg.text(78, 26, "Below D = 4: every double-double value so far, and the resolution it took", size=15, fill=INK,
             weight="bold")
    svg.text(78, 46, "Filled: resolved (two τ resolutions agree to 10⁻⁸); hollow: converged but not yet resolved; "
             "line: the √(D − 3) formula. Hover for values.", size=11)
    xs = np.linspace(3.04, 4.0, 200)

    p = Panel(svg, 78, 120, 250, 340, (3.0, 4.02), (2.3, 3.55), "Echo period Δ", "D", "Δ", yfmt="{:.1f}")
    p.curve(xs, [formula(C_DELTA, d) for d in xs], cF, dash="5 4", width=1.6)
    for D, v, M, err, ok in pts:
        e = f", change from the next lower M {err:.1e}" if err is not None else ", one resolution only"
        marker(svg, p.sx(D), p.sy(v), cR if ok else cU, f"D = {D}: Δ = {v:.13f} (M = {M}{e}); formula "
               f"{formula(C_DELTA, D):.6f}", hollow=not ok, r=4.5)
    p.key("dot", cR, "resolved")
    p.key("hollow", cU, "not yet")
    p.key("line", cF, "formula", dash="5 4")

    g = Panel(svg, 388, 120, 200, 340, (3.0, 4.02), (0.13, 0.40), "Mass-scaling exponent γ", "D", "γ",
              yfmt="{:.2f}")
    g.curve(xs, [formula(C_GAMMA, d) for d in xs], cF, dash="5 4", width=1.6)
    for D, (M, v) in sorted(gam.items()):
        marker(svg, g.sx(D), g.sy(v), cR, f"D = {D}: γ = {v:.10f} (M = {M}); formula {formula(C_GAMMA, D):.6f}", r=4.5)
    g.key("dot", cR, "ours")
    g.key("line", cF, "formula", dash="5 4")

    lows = [math.log10(max(c, 1e-24)) for (a, b), lst in steps.items() if a >= 64 for _, c in lst]
    q = Panel(svg, 648, 120, 330, 340, (3.1, 4.05), (math.floor(min(lows + [-9])) - 0.5, -6),
              "Change in Δ when M is raised", "D",
              "log₁₀ |change in Δ|", yfmt="{:g}")
    svg.line(q.px, q.sy(-8), q.px + q.pw, q.sy(-8), INK_SOFT, width=1.2, dash="6 4")
    svg.text(q.px + q.pw - 2, q.sy(-8) - 5, "10⁻⁸: resolved", size=10, fill=INK, anchor="end")
    for i, ((a, b), lst) in enumerate(sorted(steps.items())):
        if a < 64:
            continue
        col = SERIES[(i + 1) % len(SERIES)]
        lst = sorted(lst)
        q.curve([d for d, _ in lst], [math.log10(max(c, 1e-24)) for _, c in lst], col, width=1.4, opacity=0.6)
        for D, c in lst:
            marker(svg, q.sx(D), q.sy(math.log10(max(c, 1e-24))), col, f"D = {D}: M {a} → {b} moves Δ by {c:.2e}", r=4)
        D1, c1 = lst[-1]
        svg.text(q.sx(D1) + 7, q.sy(math.log10(max(c1, 1e-24))) + 4, f"{a}→{b}", size=10, fill=INK)
    open(args.o, "w").write(svg.finish(120 + 340 + 56))
    print("wrote", args.o)
    for D, v, M, err, ok in pts:
        print(f"  D = {D}: Delta {v:.13f} (M {M}, err {err if err is None else f'{err:.1e}'}, "
              f"{'resolved' if ok else 'not resolved'}); formula - ours {formula(C_DELTA, D) - v:+.1e}")
    print("  gamma:", {D: (M, round(v, 10)) for D, (M, v) in sorted(gam.items())})


if __name__ == "__main__":
    main()
