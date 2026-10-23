#!/usr/bin/env python3
"""The selection rule's estimates as distance from the finite-D answer, on a log scale, and what order 14 showed.

The arg crossings (from above, k = 4..14) and the ratio estimates (from below, k = 9..14) are recomputed from
inner_amp/scan29J1000/ (sigma_0..14 at w/w_inf = 0.99, 1.00, 1.01, the 960-bit files) with plot_selection.crossings; the
k = 5..8 ratio estimates and the factored-series values are those of plot_selection.py (k = 13, 14 from
notes/PLANTED_SELECTION.md sections 5.1 and 6.3). The band is the planted-truth calibration at K = 13 (section 5.4):
w*/w_inf = 0.99992-0.99998.

The projection to k = 14 is a simple geometric model, not a calibrated estimator: the from-above estimate is taken as
w* + c r^k, with c and r fixed by k = 12 and 13 for each assumed zero w*. It was drawn before sigma_14 was known, to show
how far apart the hypotheses "zero exactly at 1" and "zero in the K = 13 band" are at the next order; the measured
k = 14 estimate is drawn beside it (section 6.7).

    python3 plot_selection_zoom.py [-o selection_zoom.svg]
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_delta import INK, INK_SOFT, SERIES, Svg, marker  # noqa: E402
from plot_selection import ENTIRE, RATIO_WIDE, SCAN, SCAN_FILES, Axes, crossings, square  # noqa: E402

BAND = (0.99992, 0.99998)           # the K = 13 calibration, notes/PLANTED_SELECTION.md section 5.4
HYPOTHESES = [1.0, 0.99998, 0.99995, 0.99992]


def project(above, wstar):
    """Next-order from-above estimate if est_k = w* + c r^k through k = 12, 13."""
    d12, d13 = above[12] - wstar, above[13] - wstar
    r = d13 / d12
    return wstar + d13 * r, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "selection_zoom.svg"))
    args = ap.parse_args()
    above, below14 = crossings(SCAN, SCAN_FILES)
    below = {**RATIO_WIDE, **below14}
    gser = dict(ENTIRE)
    cA, cB, cE, cH = SERIES[0], SERIES[1], SERIES[3], SERIES[1]

    ks = sorted(above)
    xs, ys = ks, [math.log10(above[k] - 1) for k in ks]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    trend = lambda k: 10 ** (my + slope * (k - mx))   # noqa: E731
    proj = {w: project(above, w) for w in HYPOTHESES}

    W = 900
    svg = Svg(W)
    svg.text(78, 26, "The selection rule: distance from the finite-D answer, and what order 14 showed", size=15,
             fill=INK, weight="bold")
    svg.text(78, 46, "|estimate − 1| on a log scale, where steady convergence is a straight line; hover a mark for its "
             "value", size=11)
    p = Axes(svg, 78, 120, 790, 300, (3.5, 15.2), (-6, -1.3), [(k, str(k)) for k in range(4, 15)],
             [(10.0 ** e, f"10⁻{'⁰¹²³⁴⁵⁶'[-e]}") for e in range(-6, -1)],
             "From above (blue) the estimates fall about ×{:.2f} per order; the shaded band is where the K = 13 "
             "calibration put the zero".format(10 ** slope),
             "order k of the amplitude expansion", "|ω*/ω∞ − 1|", ylog=True)
    lo, hi = 1 - BAND[1], 1 - BAND[0]
    svg.add(f'<rect x="{p.px:.1f}" y="{p.sy(hi):.1f}" width="{p.pw:.1f}" height="{p.sy(lo) - p.sy(hi):.1f}" '
            f'fill="{cH}" opacity="0.13"><title>calibrated zero at K = 13: ω*/ω∞ = {BAND[0]}–{BAND[1]}</title></rect>')
    svg.text(p.sx(3.65), p.sy(4e-5) + 4, "zero as calibrated at K = 13: 2–8×10⁻⁵ below 1", size=10, fill=INK)
    svg.path([(p.sx(4), p.sy(trend(4))), (p.sx(14.8), p.sy(trend(14.8)))], INK_SOFT, width=1.2, dash="6 4")
    svg.text(p.sx(4.2), p.sy(3e-4), f"dashed: trend of k = 4–{max(ks)}, ×{10 ** slope:.2f} per order", size=10, fill=INK)
    svg.path([(p.sx(k + 0.12), p.sy(1 - v)) for k, v in sorted(below.items())], cB, width=1.5)
    for k, v in sorted(below.items()):
        marker(svg, p.sx(k + 0.12), p.sy(1 - v), cB, f"k = {k}: from below, ω/ω∞ = {v:.6f}", hollow=True)
    for k, v in sorted(gser.items()):
        square(svg, p.sx(k - 0.12), p.sy(abs(v - 1)), cE, f"k = {k}: factored series, ω/ω∞ = {v}", r=3.5)
    svg.path([(p.sx(k), p.sy(above[k] - 1)) for k in ks], cA, width=2)
    for k in ks:
        marker(svg, p.sx(k), p.sy(above[k] - 1), cA, f"k = {k}: from above, ω/ω∞ = {above[k]:.7f}")
    xp = p.sx(14.35)                    # the predictions from k = 12, 13, beside the measured k = 14
    for w, (v, r) in proj.items():
        col = INK if w == 1.0 else cH
        marker(svg, xp, p.sy(v - 1), col, f"k = 14 predicted from k = 12, 13 if the zero is at {w}: {v:.7f} "
               f"(r = {r:.2f} per order)", hollow=True)
    v1 = proj[1.0][0] - 1
    svg.text(xp + 10, p.sy(v1) + 4, "zero at 1", size=10, fill=INK)
    vb = [proj[w][0] - 1 for w in HYPOTHESES[1:]]
    svg.text(xp + 10, p.sy(min(vb)) + 4, "zero in the band", size=10, fill=INK)
    svg.text(p.sx(14) - 14, p.sy(above[14] - 1) + 20, f"measured at k = 14: {(above[14] - 1) * 1e5:.1f}×10⁻⁵", size=10,
             fill=INK, anchor="end")
    p.key(lambda x, y: marker(svg, x, y, cA, "from above"), "from above (simple zero)")
    p.key(lambda x, y: marker(svg, x, y, cB, "from below", hollow=True), "from below (exponent free)")
    p.key(lambda x, y: square(svg, x, y, cE, "factored series", r=4), "factored series")
    p.key(lambda x, y: marker(svg, x, y, INK_SOFT, "k = 14 projections", hollow=True), "k = 14, predicted from 12, 13")
    open(args.o, "w").write(svg.finish(120 + 300 + 56))
    print("wrote", args.o)
    print("above:", {k: round(v, 7) for k, v in above.items()})
    print("below:", {k: round(v, 6) for k, v in below.items()})
    print(f"trend: x{10 ** slope:.3f} per order; at k = 14: {trend(14):.2e} above 1")
    for w, (v, r) in proj.items():
        print(f"k = 14 if the zero is at {w}: {v - 1:+.2e} from 1 (r = {r:.3f})")
    print(f"k = 14 measured: {above[14] - 1:+.2e} from 1")


if __name__ == "__main__":
    main()
