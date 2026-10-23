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

The arg crossings (k = 4..14), the ratio estimates (k = 9..14) and the effective exponents (k = 6..14, at w_inf) are
recomputed from inner_amp/scan29J1000/ (sigma_1..14 at w/w_inf = 0.99, 1.00, 1.01, the 960-bit files; quadratic
interpolation); the k = 5..8 ratio estimates come from the wider scans (scan15phys/, scan17phys/), the Pade ranges from
the zero.txt files of the K 15 ... K 25 scans (k <= 12) and notes/PLANTED_SELECTION.md sections 5.1 and 6.3
(k = 13, 14), the factored-series values from solver/proof/entire_cert.py (k <= 12) and the same sections (k = 13, 14).

    python3 plot_selection.py [-o selection.svg]
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
# k = 14 (notes/PLANTED_SELECTION.md section 6.3): central Pade [3/10]-[9/4] and the factored series (production kbar)
PADE14 = (0.99995, 1.00013)
G14 = 1.000027
# Pade approximants of (ln S_1)' with k coefficients: range of the Im A_0 = 0 crossings over the consistent approximants
# (k = 13, 14: the central ones, notes/PLANTED_SELECTION.md sections 5.1 and 6.3)
PADE = {7: (0.981, 0.991), 8: (0.987, 0.998), 9: (0.992, 0.998), 10: (0.9995, 1.0007), 11: (0.9992, 1.0008),
        12: (0.99987, 0.99991), 13: (0.99980, 0.99990), 14: PADE14}
# factored series G = exp(-kbar A^2) S_1 truncated at order K (solver/proof/entire_cert.py; k = 13, 14: the same sections)
ENTIRE = {9: 0.99767, 10: 1.00035, 11: 0.99987, 12: 0.99988, 13: 0.999995, 14: G14}
# truncated route: zero of the M-harmonic selection (zero_M1.py, nested.py, hier.py --schur)
TRUNC = {1: 0.776994, 3: 1.015603, 5: 0.999228, 7: 1.000022}  # M = 7: proof/B_tail_M7.md
M7_PRED = (0.99997, 1.00017)          # proof/B_tail.md, reconcile.py
SCAN = os.path.join(HERE, "inner_amp", "scan29J1000")
SCAN_FILES = "*_p960.json"              # the directory also holds a 1280-bit pilot at w_inf (bitwise the same sigma_k)


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

    def __init__(self, svg, px, py, pw, ph, xr, yr, xticks, yticks, title, xlabel, ylabel, ylog=False):
        self.svg, self.px, self.py, self.pw, self.ph, self.xr, self.yr, self.ylog = svg, px, py, pw, ph, xr, yr, ylog
        svg.text(px, py - 40, title, size=12, fill=INK)
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
        self.kx = px

    def sx(self, v):
        return self.px + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.pw

    def sy(self, v):
        if self.ylog:
            v = math.log10(v)
        return self.py + self.ph - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.ph

    def key(self, draw, label):
        y = self.py - 20
        draw(self.kx + 6, y - 4)
        self.svg.text(self.kx + 16, y, label, size=10)
        self.kx += 16 + 5.9 * len(label) + 20


def square(svg, x, y, color, title, r=4.5):
    svg.add(f'<g><rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}" height="{2 * r:.1f}" fill="{color}" '
            f'stroke="{SURFACE}" stroke-width="2"/><title>{esc(title)}</title></g>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "selection.svg"))
    args = ap.parse_args()
    above, below9 = crossings(SCAN, SCAN_FILES)
    below = {**RATIO_WIDE, **below9}
    expo = {k: c.real for k, c in exponents(SCAN, SCAN_FILES).items()}
    kmax = max(above)
    cA, cB, cP, cE = SERIES[0], SERIES[1], SERIES[2], SERIES[3]
    cT = SERIES[4]

    W = 900
    svg = Svg(W)
    svg.text(78, 26, "The selection rule, evaluated from the D = ∞ inner problem alone", size=15, fill=INK, weight="bold")
    svg.text(78, 46, "Where the harmonic-1 Stokes multiplier has a real zero, against ω∞ from the finite-D solutions "
             "(the line at 1); hover a mark for its value", size=11)

    # panel 1
    p = Axes(svg, 78, 120, 790, 260, (3.5, kmax + 0.6), (0.976, 1.022),
             [(k, str(k)) for k in range(4, kmax + 1)], [(v / 100, f"{v / 100:.2f}") for v in range(98, 103)],
             "Order by order: two estimates close in from opposite sides", "order k of the amplitude expansion in A²",
             "ω*/ω∞")
    ks = sorted(set(above) & set(below))
    poly = [(p.sx(k), p.sy(above[k])) for k in ks] + [(p.sx(k), p.sy(below[k])) for k in reversed(ks)]
    svg.add('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in poly) + f'" fill="{BAND}"/>')
    y1 = p.sy(1.0)
    svg.line(p.px, y1, p.px + p.pw, y1, INK_SOFT, width=1.2, dash="6 4")
    svg.text(p.sx(3.6), y1 + 14, "ω∞ from the finite-D solutions (D ≤ 61)", size=10)
    for k, (lo, hi) in PADE.items():
        x = p.sx(k + 0.12)
        svg.line(x, p.sy(lo), x, p.sy(hi), cP, width=2.5)
    for k, (lo, hi) in PADE.items():
        x = p.sx(k + 0.12)
        marker(svg, x, p.sy((lo + hi) / 2), cP, f"Padé, {k} coefficients: {lo}–{hi}", shape="diamond", r=3.2)
    svg.path([(p.sx(k), p.sy(v)) for k, v in sorted(above.items()) if v < 1.022], cA, width=2)
    svg.path([(p.sx(k), p.sy(v)) for k, v in sorted(below.items())], cB, width=2)
    for k, v in sorted(above.items()):
        if v < 1.022:
            marker(svg, p.sx(k), p.sy(v), cA, f"k = {k}: arg τ_k = π at ω/ω∞ = {v:.6f}")
    for k, v in sorted(below.items()):
        marker(svg, p.sx(k), p.sy(v), cB, f"k = {k}: ratio estimate real at ω/ω∞ = {v:.5f}", hollow=True)
    for k, v in ENTIRE.items():
        square(svg, p.sx(k - 0.12), p.sy(v), cE, f"factored series to order {k}: ω/ω∞ = {v}")
    svg.text(p.sx(4) + 10, p.sy(above[4]) + 4, f"{above[4]:.4f}", size=10, fill=INK)
    svg.text(p.sx(kmax + 0.4), p.sy(1.0) + 30, f"bracket at k = {kmax}:", size=10, fill=INK, anchor="end")
    svg.text(p.sx(kmax + 0.4), p.sy(1.0) + 44, f"{below[kmax]:.5f} – {above[kmax]:.5f}", size=10, fill=INK,
             anchor="end")
    svg.text(p.sx(6.2), p.sy(1.016), "from above: simple-zero estimate (arg τ_k = π)", size=10, fill=INK)
    svg.text(p.sx(6.2), p.sy(0.979), "from below: exponent left free (ratio of τ_k−1, τ_k)", size=10, fill=INK)
    p.key(lambda x, y: marker(svg, x, y, cA, "from above"), "from above (simple zero)")
    p.key(lambda x, y: marker(svg, x, y, cB, "from below", hollow=True), "from below (exponent free)")
    p.key(lambda x, y: marker(svg, x, y, cP, "Padé", shape="diamond", r=3.2), "Padé approximants (range)")
    p.key(lambda x, y: square(svg, x, y, cE, "factored series", r=4), "factored series e^(−κA²)S₁")

    # panel 2: truncation sequence
    yt = 470
    q = Axes(svg, 78, yt, 340, 190, (0, 8), (-5, 0), [(m, f"M = {m}") for m in (1, 3, 5, 7)],
             [(10.0 ** e, f"10{'⁻' + '⁰¹²³⁴⁵'[-e] if e else '⁰'}") for e in range(-5, 1)],
             "Keeping M harmonics: the error alternates and shrinks", "harmonics kept", "|ω*/ω∞ − 1|", ylog=True)
    pts = [(q.sx(m), q.sy(abs(v - 1))) for m, v in TRUNC.items()]
    svg.path(pts, cT, width=2)
    for (m, v), (x, y) in zip(TRUNC.items(), pts):
        marker(svg, x, y, cT, f"M = {m}: ω/ω∞ = {v} ({(v - 1) * 100:+.3f}%)")
        svg.text(x + 9, y - 6, f"{(v - 1) * 100:+.3g}%".replace("-", "−"), size=10, fill=INK)
    lo, hi = M7_PRED
    x7 = q.sx(7)
    ylo = 1e-5 if lo < 1 < hi else max(min(abs(lo - 1), abs(hi - 1)), 1e-5)  # a range that contains 1 reaches the floor
    svg.line(x7, q.sy(ylo), x7, q.sy(max(abs(lo - 1), abs(hi - 1))), cT, width=2, dash="3 3")
    marker(svg, x7, q.sy(abs(hi - 1)), cT, f"M = 7 predicted from M ≤ 5: {lo}–{hi}", hollow=True)
    svg.text(x7 - 9, q.sy(abs(hi - 1)) + 4, "predicted", size=10, fill=INK, anchor="end")

    # panel 3: effective exponent
    r3 = Axes(svg, 528, yt, 340, 190, (5.5, kmax + 0.5), (0.95, 1.2), [(k, str(k)) for k in range(6, kmax + 1)],
              [(v / 100, f"{v / 100:.2f}") for v in range(95, 121, 5)],
              "Is the zero simple? The effective exponent tends to 1", "last order used, k", "effective exponent, Re c")
    svg.line(r3.px, r3.sy(1.0), r3.px + r3.pw, r3.sy(1.0), INK_SOFT, width=1.2, dash="6 4")
    svg.text(r3.px + r3.pw - 4, r3.sy(1.0) + 14, "simple zero: c = 1", size=10, anchor="end")
    pts = [(r3.sx(k), r3.sy(v)) for k, v in expo.items()]
    svg.path(pts, cA, width=2)
    for (k, v), (x, y) in zip(expo.items(), pts):
        marker(svg, x, y, cA, f"orders {k - 1}–{k}: Re c = {v:.3f}")

    H = yt + 190 + 52
    open(args.o, "w").write(svg.finish(H))
    print("wrote", args.o, {k: round(v, 6) for k, v in above.items()}, {k: round(v, 5) for k, v in below.items()})
    print("effective exponent c at w_inf:",
          {k: f"{c.real:.3f}{c.imag:+.3f}i" for k, c in exponents(SCAN, SCAN_FILES).items()})


if __name__ == "__main__":
    main()
