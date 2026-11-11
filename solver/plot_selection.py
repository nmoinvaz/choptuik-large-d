#!/usr/bin/env python3
"""The D = infinity selection rule evaluated from the inner problem alone, as an SVG with three panels.

1. Where the harmonic-1 condition S_1(A, w) = 0 has a real zero, as w/w_inf, order by order in the amplitude expansion:
   the arg(tau_k) = pi crossings (simple-zero estimator, from above), the ratio estimates (exponent left free, from
   below), the central Pade approximants of (ln S_1)' (range over the consistent approximants) and the factored series
   G = exp(-kbar A^2) S_1. The line at 1 is w_inf from the finite-D solutions, which none of these use.
2. The truncated route: |w/w_inf - 1| for M = 1, 3, 5, 7 harmonics (hier.py, nested.py, zero_M1.py; M = 7 from
   m7_outer.py, proof/B_tail_M7.md) and the M = 7 range predicted from M <= 5 in proof/B_tail.md, on a log scale; the
   sign alternates.
3. The effective exponent c of tau_k ~ -c/(k x0^k) at w_inf, from two consecutive orders: it tends to 1, a simple zero,
   which is what makes the arg crossings the asymptotically correct estimator.

The arg crossings (k = 4..15), the ratio estimates (k = 9..15) and the effective exponents (k = 6..15, at w_inf) are
recomputed from inner_amp/scan31J1000/ (sigma_1..15 at w/w_inf = 0.99, 1.00, 1.01, the 960-bit files; quadratic
interpolation); the k = 5..8 ratio estimates come from the wider scans (scan15phys/, scan17phys/), the Pade ranges from
the zero.txt files of the K 15 ... K 25 scans (k <= 12) and notes/PLANTED_SELECTION.md sections 5.1, 6.3 and 7.3
(k = 13, 14, 15), the factored-series values from solver/proof/entire_cert.py (k <= 12) and the same sections.

    python3 plot_selection.py [-o selection.svg] [--frame T]   # animated (SMIL, 42 s loop); --frame: one still
    python3 plot_selection.py --static -o FILE                 # the same layout without the animation
    python3 plot_selection.py --print    # the paper's layout, selection_print.svg (static)
"""
import argparse
import glob
import json
import math
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "inner_amp"))
from plot_delta import BAND, GRID, INK, INK_SOFT, SERIES, SURFACE, Svg, esc, marker  # noqa: E402
from arg_crossing import taus  # noqa: E402

mp.mp.dps = 50

# from below: ratio tau_{k-1}/tau_k real (exponent free); k = 5..8 from the wide scans, 9..12 recomputed below
RATIO_WIDE = {5: 0.9821, 6: 0.9803, 7: 0.9913, 8: 0.9946}
# k = 14, 15 (notes/PLANTED_SELECTION.md sections 6.3, 7.3): central Pade [3/10]-[9/4], [3/11]-[10/4] and the factored
# series (production kbar)
PADE14 = (0.99995, 1.00013)
G14 = 1.000027
PADE15 = (0.999984, 1.000063)
G15 = 0.999999
# Pade approximants of (ln S_1)' with k coefficients: range of the Im A_0 = 0 crossings over the consistent approximants
# (k = 13, 14, 15: the central ones, notes/PLANTED_SELECTION.md sections 5.1, 6.3 and 7.3)
PADE = {7: (0.981, 0.991), 8: (0.987, 0.998), 9: (0.992, 0.998), 10: (0.9995, 1.0007), 11: (0.9992, 1.0008),
        12: (0.99987, 0.99991), 13: (0.99980, 0.99990), 14: PADE14, 15: PADE15}
# factored series G = exp(-kbar A^2) S_1 truncated at order K (solver/proof/entire_cert.py; k = 13-15: the same sections)
ENTIRE = {9: 0.99767, 10: 1.00035, 11: 0.99987, 12: 0.99988, 13: 0.999995, 14: G14, 15: G15}
# truncated route: zero of the M-harmonic selection (zero_M1.py, nested.py, hier.py --schur)
TRUNC = {1: 0.776994, 3: 1.015603, 5: 0.999228, 7: 1.000022}  # M = 7: proof/B_tail_M7.md
M7_PRED = (0.99997, 1.00017)          # proof/B_tail.md, reconcile.py
SCAN = os.path.join(HERE, "inner_amp", "scan31J1000")
SCAN_FILES = "*_p960.json"


def exponents(scandir, pattern="*.json", wf=1.0):
    """effective exponent c of tau_k = -c/(k x0^k) at w/w_inf = wf, with x0 = (k-1) tau_{k-1}/(k tau_k) from orders
    k-1, k (this reproduces the README table's column and the earlier hard-coded k = 6..12 values)"""
    for fn in sorted(glob.glob(os.path.join(scandir, pattern))):
        r = json.load(open(fn))
        if abs(r["wf"] - wf) < 1e-9:
            t = taus([mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in r["sigma_str"]])
            return {k: complex(-k * t[k] * ((k - 1) * t[k - 1] / (k * t[k])) ** k) for k in range(6, len(t))}
    raise FileNotFoundError(wf)


def crossings(scandir, pattern="*.json"):
    rows = []
    for fn in sorted(glob.glob(os.path.join(scandir, pattern))):
        r = json.load(open(fn))
        sig = [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in r["sigma_str"]]
        rows.append((r["wf"], taus(sig)))
    rows.sort()
    ws = [w for w, _ in rows]

    def root(vals):                     # quadratic through the three points
        import numpy as np
        c = np.polyfit(ws, vals, 2)
        rts = [x.real for x in np.roots(c) if abs(x.imag) < 1e-12 and ws[0] - 0.03 < x.real < ws[-1] + 0.03]
        return min(rts, key=lambda x: abs(x - 1)) if rts else None
    above = {k: root([float(mp.arg(t[k]) / mp.pi) % 2 - 1 for _, t in rows]) for k in range(4, len(rows[0][1]))}
    below = {k: root([float(mp.im(mp.sqrt((k - 1) * t[k - 1] / (k * t[k])))) for _, t in rows])
             for k in range(9, len(rows[0][1]))}
    return above, below


class Axes:
    """Plot area with explicit ticks (linear or log10 in y)."""

    def __init__(self, svg, px, py, pw, ph, xr, yr, xticks, yticks, title, xlabel, ylabel, ylog=False, tdy=40):
        self.svg, self.px, self.py, self.pw, self.ph, self.xr, self.yr, self.ylog = svg, px, py, pw, ph, xr, yr, ylog
        svg.text(px, py - tdy, title, size=12, fill=INK)
        for v, lab in yticks:
            y = self.sy(v)
            svg.line(px, y, px + pw, y, GRID)
            svg.text(px - 8, y + 4, lab, size=10, anchor="end")
        for v, lab in xticks:
            x = self.sx(v)
            svg.line(x, py, x, py + ph, GRID)
            svg.text(x, py + ph + 16, lab, size=10, anchor="middle")
        svg.line(px, py + ph, px + pw, py + ph, INK_SOFT)
        svg.text(px + pw / 2, py + ph + 34, xlabel, size=11, anchor="middle")
        svg.add(f'<text x="{px - 48:.1f}" y="{py + ph / 2:.1f}" font-size="11" fill="{INK_SOFT}" text-anchor="middle" '
                f'transform="rotate(-90 {px - 48:.1f} {py + ph / 2:.1f})">{esc(ylabel)}</text>')
        self.kx, self.ky = px, py - 20

    def sx(self, v):
        return self.px + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.pw

    def sy(self, v):
        if self.ylog:
            v = math.log10(v)
        return self.py + self.ph - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.ph

    def key(self, draw, label):
        y = self.ky
        draw(self.kx + 6, y - 4)
        self.svg.text(self.kx + 16, y, label, size=10)
        self.kx += 16 + 5.9 * len(label) + 20


def square(svg, x, y, color, title, r=4.5):
    svg.add(f'<g><rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}" height="{2 * r:.1f}" fill="{color}" '
            f'stroke="{SURFACE}" stroke-width="2"/><title>{esc(title)}</title></g>')


T = 42.0          # the animation's cycle, seconds
STEP_TEXT = [(0.0, 17.5, "1 of 3: order by order, the estimate from above comes down and the one from below comes up"),
             (17.5, 26.0, "2 of 3: keeping M harmonics, the error alternates in sign and shrinks"),
             (26.0, 34.0, "3 of 3: the effective exponent settles on 1, so the zero is simple"),
             (34.0, T, "all of it at once")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o")
    ap.add_argument("--print", action="store_true", help="the paper layout, 480 px wide (default -o selection_print.svg)")
    ap.add_argument("--static", action="store_true", help="the README layout without the animation")
    ap.add_argument("--frame", type=float, help="the animation's still at this time, in seconds")
    ap.add_argument("--mutate", help="drop_order: leave the last order out of the animation (the hold-frame control)")
    args = ap.parse_args()
    P = args.print
    anim = not P and not args.static
    args.o = args.o or os.path.join(HERE, "selection_print.svg" if P else "selection.svg")
    above, below9 = crossings(SCAN, SCAN_FILES)
    below = {**RATIO_WIDE, **below9}
    expo = {k: c.real for k, c in exponents(SCAN, SCAN_FILES).items()}
    kmax = max(above)
    cA, cB, cP, cE = SERIES[0], SERIES[1], SERIES[2], SERIES[3]
    cT = SERIES[4]

    # Layout: panel boxes (px, py, pw, ph). --print is the math paper's: 480 px at 360 pt, no title (the caption
    # carries it), panel 1's key in two rows, shorter titles on the lower panels.
    if P:
        W, H, tdy = 480, 580, 14
        box1, box2, box3 = (60, 74, 406, 210), (60, 364, 150, 170), (274, 364, 194, 170)
    else:
        W, H, tdy = 900, 470 + 190 + 52, 40
        box1, box2, box3 = (78, 120, 790, 260), (78, 470, 340, 190), (528, 470, 340, 190)
    svg = Svg(W)

    def sched(t_on, t_off=None):
        """opacity points: hidden, faded in at t_on, held, faded out at t_off (the cycle's end by default)"""
        t_off = T - 0.7 if t_off is None else t_off
        return [(0.0, 0.0), (t_on, 0.0), (t_on + 0.3, 1.0), (t_off, 1.0), (min(t_off + 0.3, T), 0.0), (T, 0.0)]

    def at(pts, t):
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t0 <= t <= t1:
                return v1 if t1 == t0 else v0 + (v1 - v0) * (t - t0) / (t1 - t0)
        return pts[-1][1]

    def show(t_on, fn, t_off=None):
        """draw fn's elements; animated, as one group visible from t_on (SMIL, or at --frame's time)"""
        if not anim:
            fn()
            return
        n = len(svg.parts)
        fn()
        body = "".join(svg.parts[n:])
        del svg.parts[n:]
        pts = sched(t_on, t_off)
        if args.frame is not None:
            op = at(pts, args.frame)
            if op > 0:
                svg.add(f'<g opacity="{op:.2f}">{body}</g>')
            return
        kt = ";".join(f"{t / T:.4f}" for t, _ in pts)
        vs = ";".join(f"{v:g}" for _, v in pts)
        svg.add(f'<g opacity="0">{body}<animate attributeName="opacity" values="{vs}" keyTimes="{kt}" dur="{T}s" '
                f'repeatCount="indefinite"/></g>')

    if not P:
        svg.text(78, 26, "The selection rule, evaluated from the D = ∞ inner problem alone", size=15, fill=INK, weight="bold")
        svg.text(78, 46, "Where the harmonic-1 Stokes multiplier has a real zero, against ω∞ from the finite-D solutions "
                 "(the line at 1); hover a mark for its value", size=11)
    if anim:
        for t0, t1, text in STEP_TEXT:
            show(t0, lambda text=text: svg.text(78, 64, text, size=11, fill=SERIES[1]), t1 - 0.3)

    # the animation's times: panel 1 order by order, then its Pade ranges and factored series, then panels 2 and 3
    t_k = {k: 1.0 + 1.1 * (k - 4) for k in range(4, kmax + 1)}
    t_pade = {k: 14.6 + 0.25 * (k - 7) for k in range(7, kmax + 1)}
    drop = {kmax} if args.mutate == "drop_order" else set()

    # panel 1
    p = Axes(svg, *box1, (3.5, kmax + 0.6), (0.976, 1.022),
             [(k, str(k)) for k in range(4, kmax + 1)], [(v / 100, f"{v / 100:.2f}") for v in range(98, 103)],
             "Order by order: two estimates close in from opposite sides", "order k of the amplitude expansion in A²",
             "ω*/ω∞", tdy=58 if P else 40)
    ks = sorted(set(above) & set(below))
    if anim:                                    # the bracket grows one order at a time
        for k0, k1 in zip(ks, ks[1:]):
            if k1 in drop:
                continue
            quad = [(p.sx(k0), p.sy(above[k0])), (p.sx(k1), p.sy(above[k1])), (p.sx(k1), p.sy(below[k1])),
                    (p.sx(k0), p.sy(below[k0]))]
            show(t_k[k1], lambda quad=quad: svg.add('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in quad)
                                                    + f'" fill="{BAND}"/>'))
    else:
        poly = [(p.sx(k), p.sy(above[k])) for k in ks] + [(p.sx(k), p.sy(below[k])) for k in reversed(ks)]
        svg.add('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in poly) + f'" fill="{BAND}"/>')
    y1 = p.sy(1.0)
    svg.line(p.px, y1, p.px + p.pw, y1, INK_SOFT, width=1.2, dash="6 4")
    svg.text(p.sx(3.6), y1 + 14, "ω∞ from finite-D solutions" if P else "ω∞ from the finite-D solutions (D ≤ 61)", size=10)
    for k, (lo, hi) in PADE.items():
        x = p.sx(k + 0.12)
        show(t_pade[k], lambda x=x, lo=lo, hi=hi: svg.line(x, p.sy(lo), x, p.sy(hi), cP, width=2.5))
    for k, (lo, hi) in PADE.items():
        x = p.sx(k + 0.12)
        show(t_pade[k], lambda x=x, k=k, lo=lo, hi=hi: marker(svg, x, p.sy((lo + hi) / 2), cP, f"Padé, {k} coefficients: {lo}–{hi}",
                                                             shape="diamond", r=3.2))
    up = [(k, v) for k, v in sorted(above.items()) if v < 1.022]
    lowr = sorted(below.items())
    if anim:                                    # each line segment with the order it reaches
        for (k0, v0), (k1, v1) in zip(up, up[1:]):
            if k1 not in drop:
                show(t_k[k1], lambda a=(k0, v0), b=(k1, v1): svg.path([(p.sx(a[0]), p.sy(a[1])), (p.sx(b[0]), p.sy(b[1]))], cA, width=2))
        for (k0, v0), (k1, v1) in zip(lowr, lowr[1:]):
            if k1 not in drop:
                show(t_k[k1], lambda a=(k0, v0), b=(k1, v1): svg.path([(p.sx(a[0]), p.sy(a[1])), (p.sx(b[0]), p.sy(b[1]))], cB, width=2))
    else:
        svg.path([(p.sx(k), p.sy(v)) for k, v in up], cA, width=2)
        svg.path([(p.sx(k), p.sy(v)) for k, v in lowr], cB, width=2)
    for k, v in sorted(above.items()):
        if v < 1.022 and k not in drop:
            show(t_k[k], lambda k=k, v=v: marker(svg, p.sx(k), p.sy(v), cA, f"k = {k}: arg τ_k = π at ω/ω∞ = {v:.6f}"))
    for k, v in sorted(below.items()):
        if k not in drop:
            show(t_k[k], lambda k=k, v=v: marker(svg, p.sx(k), p.sy(v), cB, f"k = {k}: ratio estimate real at ω/ω∞ = {v:.5f}",
                                                 hollow=True))
    for k, v in ENTIRE.items():
        show(t_pade[k] + 0.1, lambda k=k, v=v: square(svg, p.sx(k - 0.12), p.sy(v), cE, f"factored series to order {k}: ω/ω∞ = {v}"))
    show(t_k[4], lambda: svg.text(p.sx(4) + 10, p.sy(above[4]) + 4, f"{above[4]:.4f}", size=10, fill=INK))
    if anim:                                    # the bracket's ends at the latest order shown
        for k0, k1 in zip(ks, ks[1:] + [None]):
            if k0 in drop:
                continue
            t1 = t_k[k1] if k1 is not None and k1 not in drop else None
            show(t_k[k0], lambda k0=k0: (svg.text(p.sx(kmax + 0.4), p.sy(1.0) + 30, f"bracket at k = {k0}:", size=10, fill=INK, anchor="end"),
                                         svg.text(p.sx(kmax + 0.4), p.sy(1.0) + 44, f"{below[k0]:.5f} – {above[k0]:.5f}", size=10, fill=INK,
                                                  anchor="end")), t1)
    else:
        svg.text(p.sx(kmax + 0.4), p.sy(1.0) + 30, f"bracket at k = {kmax}:", size=10, fill=INK, anchor="end")
        svg.text(p.sx(kmax + 0.4), p.sy(1.0) + 44, f"{below[kmax]:.5f} – {above[kmax]:.5f}", size=10, fill=INK,
                 anchor="end")
    show(t_k[4], lambda: svg.text(p.sx(6.2), p.sy(1.016), "from above: simple-zero estimate (arg τ_k = π)", size=10, fill=INK))
    show(t_k[5], lambda: svg.text(p.sx(6.2), p.sy(0.979), "from below: exponent left free (ratio of τ_k−1, τ_k)", size=10, fill=INK))
    if P:                                       # two key rows
        p.ky = p.py - 38
    p.key(lambda x, y: marker(svg, x, y, cA, "from above"), "from above (simple zero)")
    p.key(lambda x, y: marker(svg, x, y, cB, "from below", hollow=True), "from below (exponent free)")
    if P:
        p.kx, p.ky = p.px, p.py - 20
    p.key(lambda x, y: marker(svg, x, y, cP, "Padé", shape="diamond", r=3.2), "Padé approximants (range)")
    p.key(lambda x, y: square(svg, x, y, cE, "factored series", r=4), "factored series e^(−κA²)S₁")

    # panel 2: truncation sequence
    q = Axes(svg, *box2, (0, 8), (-5, 0), [(m, f"M = {m}") for m in (1, 3, 5, 7)],
             [(10.0 ** e, f"10{'⁻' + '⁰¹²³⁴⁵'[-e] if e else '⁰'}") for e in range(-5, 1)],
             "Keeping M harmonics" if P else "Keeping M harmonics: the error alternates and shrinks", "harmonics kept",
             "|ω*/ω∞ − 1|", ylog=True, tdy=tdy)
    t_m = {1: 18.5, 3: 20.0, 5: 21.5, 7: 24.5}
    t_pred = 23.0
    pts = [(q.sx(m), q.sy(abs(v - 1))) for m, v in TRUNC.items()]
    if anim:
        for (m0, a), (m1, b) in zip(list(zip(TRUNC, pts)), list(zip(TRUNC, pts))[1:]):
            show(t_m[m1], lambda a=a, b=b: svg.path([a, b], cT, width=2))
    else:
        svg.path(pts, cT, width=2)
    for (m, v), (x, y) in zip(TRUNC.items(), pts):
        show(t_m[m], lambda m=m, v=v, x=x, y=y: marker(svg, x, y, cT, f"M = {m}: ω/ω∞ = {v} ({(v - 1) * 100:+.3f}%)"))
        dx, dy, anchor = (-9, 4, "end") if P and m == 7 else (9, -6, "start")     # print: M = 7 left, clear of panel 3
        show(t_m[m], lambda m=m, v=v, x=x, y=y, dx=dx, dy=dy, anchor=anchor:
             svg.text(x + dx, y + dy, f"{(v - 1) * 100:+.3g}%".replace("-", "−"), size=10, fill=INK, anchor=anchor))
    lo, hi = M7_PRED
    x7 = q.sx(7)
    ylo = 1e-5 if lo < 1 < hi else max(min(abs(lo - 1), abs(hi - 1)), 1e-5)  # a range that contains 1 reaches the floor
    show(t_pred, lambda: svg.line(x7, q.sy(ylo), x7, q.sy(max(abs(lo - 1), abs(hi - 1))), cT, width=2, dash="3 3"))
    show(t_pred, lambda: marker(svg, x7, q.sy(abs(hi - 1)), cT, f"M = 7 predicted from M ≤ 5: {lo}–{hi}", hollow=True))
    if P:                                       # above the marker, off the line from M = 5
        svg.text(x7, q.sy(abs(hi - 1)) - 9, "predicted", size=10, fill=INK, anchor="middle")
    else:
        show(t_pred, lambda: svg.text(x7 - 9, q.sy(abs(hi - 1)) + 4, "predicted", size=10, fill=INK, anchor="end"))

    # panel 3: effective exponent
    r3 = Axes(svg, *box3, (5.5, kmax + 0.5), (0.95, 1.2), [(k, str(k)) for k in range(6, kmax + 1)],
              [(v / 100, f"{v / 100:.2f}") for v in range(95, 121, 5)],
              "Effective exponent tends to 1" if P else "Is the zero simple? The effective exponent tends to 1",
              "last order used, k", "effective exponent, Re c", tdy=tdy)
    svg.line(r3.px, r3.sy(1.0), r3.px + r3.pw, r3.sy(1.0), INK_SOFT, width=1.2, dash="6 4")
    svg.text(r3.px + r3.pw - 4, r3.sy(1.0) + 14, "simple zero: c = 1", size=10, anchor="end")
    t_e = {k: 26.5 + 0.7 * (k - 6) for k in expo}
    pts = [(r3.sx(k), r3.sy(v)) for k, v in expo.items()]
    if anim:
        for (k0, a), (k1, b) in zip(list(zip(expo, pts)), list(zip(expo, pts))[1:]):
            show(t_e[k1], lambda a=a, b=b: svg.path([a, b], cA, width=2))
    else:
        svg.path(pts, cA, width=2)
    for (k, v), (x, y) in zip(expo.items(), pts):
        show(t_e[k], lambda k=k, v=v, x=x, y=y: marker(svg, x, y, cA, f"orders {k - 1}–{k}: Re c = {v:.3f}"))

    open(args.o, "w").write(svg.finish(H))
    print("wrote", args.o, {k: round(v, 6) for k, v in above.items()}, {k: round(v, 5) for k, v in below.items()})
    print("effective exponent c at w_inf:",
          {k: f"{c.real:.3f}{c.imag:+.3f}i" for k, c in exponents(SCAN, SCAN_FILES).items()})


if __name__ == "__main__":
    main()
