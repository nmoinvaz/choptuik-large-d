#!/usr/bin/env python3
"""What the large-D chain does with each harmonic of the centre field p1, and how the held ones are chosen, as one SVG.

    cd hp && python3 plot_harmonics.py RAWDIR [STATE] [-o .]

harmonics_held.svg has three panels:
  1. the size of each odd harmonic k of p1 (sqrt(a_k^2 + b_k^2)) in a chain state (default the clean chain's state at
     D 232.63125): harmonics 1 and 3 are solved, 5 to 13 are held at the fit's values, 15 and up are set to zero;
  2. harmonic 5's cosine coefficient a_5 against D: the full-system states the fit is made from (held_fit.py's
     full_states) and the fits of degree 6, 8 and 10 (data/held_fit_d6.json, held_fit.json, held_fit_d10.json)
     extended to D 240;
  3. how far the fits of degree 6 and 10 are from degree 8 in a_5, against D: inside the states' range they agree to
     the fit's residual, past it they part, and that parting is the error bar on Delta.
Standard library and numpy; the primitives are data_graphs.py. Prints the plotted numbers.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_graphs as dg  # noqa: E402
from data_graphs import INK, INK_SOFT, SERIES, Svg, axes, better_hint, callout, footnote, marker  # noqa: E402
from held_fit import full_states  # noqa: E402
from octo_corrector import load_o  # noqa: E402
from pinning import HeldFit, band_coeffs  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
STATE = "I461-win2/crawlG/anm_pin3_D232.6312.npz"
class DecadeAxis(dg.Axis):
    """a log axis ticked every STEP-th decade, so labels over many decades do not meet"""
    STEP = 1

    def ticks(self):
        if self.log:
            return [10.0 ** e for e in range(math.ceil(math.log10(self.lo)), math.floor(math.log10(self.hi)) + 1)
                    if e % DecadeAxis.STEP == 0]
        return super().ticks()


FITS = [(6, "data/held_fit_d6.json"), (8, "data/held_fit.json"), (10, "data/held_fit_d10.json")]


def fits_dmin():
    """the smallest D of the states the fits are made from"""
    import json
    return min(float(json.load(open(os.path.join(HERE, p)))["Dmin"]) for _, p in FITS) - 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("state", nargs="?", default=STATE)
    ap.add_argument("-o", default=HERE)
    a = ap.parse_args()

    z, meta = load_o(os.path.join(a.raw, a.state))
    M, D_state = int(meta["M"]), float(meta["D"])
    ks = list(range(1, M, 2))          # the half grid of M points resolves the odd harmonics below M
    co = band_coeffs(np.asarray(z[0])[:M], np.asarray(z[1])[:M], M, ks)
    amp = [math.hypot(x, y) for x, y in co]
    held = list(meta.get("pin_held", [5, 7, 9, 11, 13]))
    role = {k: ("held" if k in held else ("solved" if k <= int(meta["kfreeze"]) else "zero")) for k in ks}
    pts = [p for p in full_states(a.raw) if float(p["D"]) >= fits_dmin()]
    fits = [(deg, HeldFit(os.path.join(HERE, p))) for deg, p in FITS]
    Ds = np.linspace(38.0, 240.0, 203)
    curves = {deg: [f.values(float(D), [5])[0][0] for D in Ds] for deg, f in fits}

    print(f"| harmonic k | size at D {D_state:g} | role |\n|---|---|---|")
    for k, v in zip(ks, amp):
        print(f"| {k} | {v:.3e} | {role[k]} |")
    print(f"\nfull-system states: {len(pts)}, D {float(pts[0]['D']):g} to {float(pts[-1]['D']):g}")
    print("| D | a5 degree 6 | a5 degree 8 | a5 degree 10 |\n|---|---|---|---|")
    for D in (73, 100, 150, 200, 232.63125):
        i = int(round(D - 38))
        print(f"| {D:g} | {curves[6][i]:.12e} | {curves[8][i]:.12e} | {curves[10][i]:.12e} |")

    W = 1040
    svg = Svg(W)
    svg.text(16, 26, "How the chain handles the harmonics of the centre field p1", size=15, fill=INK, weight="bold")
    svg.text(16, 46, "Over one echo period the field p1 at the centre is a sum of odd harmonics, each with a cosine and a sine "
             "coefficient. Above D 73 the chain solves the first two,", size=11)
    svg.text(16, 61, "takes the next five from a fit of the solutions computed below D 73, and sets the rest to zero. "
             "Everything else in the solution, Δ included, is solved at every step.", size=11)

    # panel 1: the spectrum
    x, y, w, h = 86, 112, 600, 250
    svg.text(16, y - 18, f"1. Size of each harmonic of p1 at D {D_state:g}, on the clean chain", size=12, fill=INK, weight="bold")
    dg.Axis, DecadeAxis.STEP = DecadeAxis, 6
    xm, ym = axes(svg, x, y, w, h, (0, M), (1e-36, 1, "log"), "harmonic number k", "size of the harmonic",
                  xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"1e{round(math.log10(v))}")
    col = {"solved": SERIES[0], "held": SERIES[1], "zero": SERIES[-1]}
    bw = 0.62 * (xm(2) - xm(0))
    for k, v in zip(ks, amp):
        top = ym(max(v, 1e-36))
        svg.add(f'<rect x="{xm(k) - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{ym(1e-36) - top:.1f}" '
                f'fill="{col[role[k]]}"><title>k {k}: {v:.2e} ({role[k]})</title></rect>')
    lx, ly = x + w + 30, y + 30
    for r, text in (("solved", "solved at every step (k = 1, 3)"), ("held", "held at the fit's value (k = 5 to 13)"),
                    ("zero", "set to zero (k = 15 to 31); what shows is rounding")):
        svg.add(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="2" fill="{col[r]}"/>')
        svg.text(lx + 18, ly, text, size=11, fill=INK)
        ly += 22
    svg.text(lx, ly + 16, "10 of the 769 numbers in a solution are held:", size=11, fill=INK_SOFT)
    svg.text(lx, ly + 32, "the cosine and sine of k = 5, 7, 9, 11 and 13.", size=11, fill=INK_SOFT)
    callout(svg, xm(5), ym(amp[2]), "held: 8e-4 down to 6e-8", dx=40, dy=-26)

    # panel 2: the fit of a5 and its extension
    y2 = y + h + 110
    svg.text(16, y2 - 18, "2. Harmonic 5's cosine coefficient: the computed solutions and the fits that extend them", size=12,
             fill=INK, weight="bold")
    data_D = [float(p["D"]) for p in pts]
    data_v = [p["c"]["5a"] for p in pts]
    lo = min(min(data_v), min(min(c) for c in curves.values()))
    hi = max(max(data_v), max(max(c) for c in curves.values()))
    pad = 0.06 * (hi - lo)
    dg.Axis = DecadeAxis
    xm2, ym2 = axes(svg, x, y2, w, h, (30, 240), (lo - pad, hi + pad), "spacetime dimension D", "a5",
                    xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:.4f}")
    svg.add(f'<rect x="{xm2(38):.1f}" y="{y2}" width="{xm2(73) - xm2(38):.1f}" height="{h}" fill="{SERIES[2]}" fill-opacity="0.08"/>')
    svg.text((xm2(38) + xm2(73)) / 2, y2 + 14, "full solutions", size=10, fill=INK_SOFT, anchor="middle")
    for (deg, _), c in zip(fits, (SERIES[3], SERIES[1], SERIES[4])):
        pts_s = " ".join(f"{xm2(D):.1f},{ym2(v):.1f}" for D, v in zip(Ds, curves[deg]))
        svg.add(f'<polyline points="{pts_s}" fill="none" stroke="{c}" stroke-width="{2.4 if deg == 8 else 1.6}"/>')
    for D, v in zip(data_D, data_v):
        marker(svg, "circle", xm2(D), ym2(v), SERIES[2], f"D {D:.4g}: a5 {v:.6e}")
    marker(svg, "diamond", xm2(D_state), ym2(co[2][0]), SERIES[0], f"chain at D {D_state:g}: a5 {co[2][0]:.6e}")
    callout(svg, xm2(D_state), ym2(co[2][0]), "the chain uses the degree-8 value", dx=-40, dy=-40)
    lx, ly = x + w + 30, y2 + 30
    for text, c, kind in (("computed full solutions, D 38 to 73", SERIES[2], "dot"), ("fit of degree 6", SERIES[3], "line"),
                          ("fit of degree 8 (the chain's)", SERIES[1], "line"), ("fit of degree 10", SERIES[4], "line")):
        if kind == "dot":
            marker(svg, "circle", lx + 5, ly - 4, c, text)
        else:
            svg.line(lx - 2, ly - 4, lx + 12, ly - 4, c, 2)
        svg.text(lx + 18, ly, text, size=11, fill=INK)
        ly += 20
    svg.text(lx, ly + 12, "On this scale the three fits are one line.", size=11, fill=INK_SOFT)

    # panel 3: how far the fits part
    y3 = y2 + h + 110
    svg.text(16, y3 - 18, "3. How far the fits of degree 6 and 10 are from degree 8, in a5", size=12, fill=INK, weight="bold")
    d6 = [abs(u - v) for u, v in zip(curves[6], curves[8])]
    d10 = [abs(u - v) for u, v in zip(curves[10], curves[8])]
    pos = [v for v in d6 + d10 if v > 0]
    DecadeAxis.STEP = 2
    xm3, ym3 = axes(svg, x, y3, w, h, (30, 240), (min(pos) / 3, max(pos) * 3, "log"), "spacetime dimension D",
                    "difference in a5 (lower is better)", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"1e{round(math.log10(v))}")
    svg.add(f'<rect x="{xm3(38):.1f}" y="{y3}" width="{xm3(73) - xm3(38):.1f}" height="{h}" fill="{SERIES[2]}" fill-opacity="0.08"/>')
    for vals, c, name in ((d6, SERIES[3], "degree 6 minus degree 8"), (d10, SERIES[4], "degree 10 minus degree 8")):
        seg = " ".join(f"{xm3(D):.1f},{ym3(max(v, min(pos) / 3)):.1f}" for D, v in zip(Ds, vals))
        svg.add(f'<polyline points="{seg}" fill="none" stroke="{c}" stroke-width="2"><title>{name}</title></polyline>')
    i = int(round(D_state - 38))
    callout(svg, xm3(Ds[i]), ym3(d6[i]), f"{d6[i]:.0e} at D {D_state:.0f}", dx=-40, dy=-24)
    better_hint(svg, x, y3, w, h, "right", label="smaller", reverse=True)
    lx, ly = x + w + 46, y3 + 30
    for text, c in (("degree 6 minus degree 8", SERIES[3]), ("degree 10 minus degree 8", SERIES[4])):
        svg.line(lx - 2, ly - 4, lx + 12, ly - 4, c, 2)
        svg.text(lx + 18, ly, text, size=11, fill=INK)
        ly += 20
    for t in ("Inside the computed range the fits agree", "to the fit's own residual. Past D 73 each", "extends the data its own way, and Δ moves",
              "with the held values: that is the error bar."):
        svg.text(lx, ly + 12, t, size=11, fill=INK_SOFT)
        ly += 16

    hgt = footnote(svg, [f"state: raw {a.state}", f"{len(pts)} full-system states (held_fit.py)",
                         "fits: data/held_fit_d6.json, held_fit.json, held_fit_d10.json"], "", [], y3 + h + 44, REPO)
    out = os.path.join(a.o, "harmonics_held.svg")
    open(out, "w").write(svg.finish(hgt))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
