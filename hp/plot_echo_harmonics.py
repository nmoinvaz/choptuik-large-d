#!/usr/bin/env python3
"""One echo period of the centre field p1 and the harmonics it is made of, as one SVG.

    cd hp && python3 plot_echo_harmonics.py RAWDIR [STATE] [-o .]

echo_harmonics.svg: on top p1 over one echo period (theta = 2 pi tau / Delta from 0 to 2 pi) at the state's D, from its
odd harmonics (pinning.band_coeffs on the half grid); below it harmonics 1, 3, 5 and 13 drawn alone, each on its own
scale, with its size. p1 changes sign after half a period, so only odd harmonics occur, and harmonic k runs k times
through the period. Standard library and numpy; the primitives are data_graphs.py. Prints the plotted numbers.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_graphs as dg  # noqa: E402
from data_graphs import INK, INK_SOFT, SERIES, Svg, axes, footnote  # noqa: E402
from octo_corrector import load_o  # noqa: E402
from pinning import band_coeffs  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
STATE = "I461-win2/crawlG/anm_pin3_D232.6312.npz"
SHOWN = [1, 3, 5, 13]
PI_TICKS = {0.0: "0", math.pi / 2: "π/2", math.pi: "π", 3 * math.pi / 2: "3π/2", 2 * math.pi: "2π"}


class PiAxis(dg.Axis):
    """the period's axis ticked at multiples of pi/2; any other axis as before"""

    def ticks(self):
        if not self.log and self.lo == 0 and abs(self.hi - 2 * math.pi) < 1e-12:
            return list(PI_TICKS)
        return super().ticks()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("state", nargs="?", default=STATE)
    ap.add_argument("-o", default=HERE)
    a = ap.parse_args()
    z, meta = load_o(os.path.join(a.raw, a.state))
    M, D = int(meta["M"]), float(meta["D"])
    Delta = float(np.asarray(z[0])[-1])
    ks = list(range(1, M, 2))
    co = dict(zip(ks, band_coeffs(np.asarray(z[0])[:M], np.asarray(z[1])[:M], M, ks)))
    th = np.linspace(0, 2 * math.pi, 721)

    def wave(k):
        ak, bk = co[k]
        return ak * np.cos(k * th) + bk * np.sin(k * th)
    total = sum(wave(k) for k in ks)
    print(f"D {D:g}, Delta {Delta:.12f}, e^Delta {math.exp(Delta):.3f}")
    print("| harmonic k | size |\n|---|---|")
    for k in ks:
        print(f"| {k} | {math.hypot(*co[k]):.3e} |")

    W, x, w = 1040, 96, 880
    dg.Axis = PiAxis
    svg = Svg(W)
    svg.text(16, 26, "One echo period of the centre field, and the harmonics it is made of", size=15, fill=INK, weight="bold")
    svg.text(16, 46, f"At D {D:g} the critical solution repeats itself after one period, each time {math.exp(Delta):.1f} times "
             f"smaller (e^Δ, Δ = {Delta:.5f}). The solver computes one period.", size=11)
    svg.text(16, 61, "Over it the field p1 at the centre is a sum of waves: harmonic k runs k times through the period. p1 "
             "changes sign after half a period, so only odd k occur.", size=11)

    def panel(y, h, vals, label, color, yl, xl="θ"):
        lim = float(np.max(np.abs(vals))) * 1.15
        svg.text(x, y - 8, label, size=12, fill=INK)
        xm, ym = axes(svg, x, y, w, h, (0, 2 * math.pi), (-lim, lim), xl, yl,
                      xfmt=lambda v: PI_TICKS.get(v, f"{v:.2f}"), yfmt=lambda v: f"{v:.0e}" if v else "0")
        pts = " ".join(f"{xm(t):.1f},{ym(v):.1f}" for t, v in zip(th, vals))
        svg.add(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        return y + h + 70

    y = panel(110, 190, total, "p1 at the centre: the sum of all its harmonics", SERIES[0], "p1",
              "one echo period: θ = 2π τ / Δ, from 0 to 2π")
    svg.text(16, y - 30, "The harmonics alone, each on its own vertical scale:", size=12, fill=INK, weight="bold")
    y += 6
    role = {1: "solved", 3: "solved", 5: "held at the fit", 13: "held at the fit"}
    for i, k in enumerate(SHOWN):
        size = math.hypot(*co[k])
        y = panel(y, 96, wave(k), f"harmonic {k}: size {size:.1e}, {role[k]}", SERIES[1 + i % 4], f"k = {k}")
    svg.text(x, y - 34, "Harmonics 7, 9 and 11 lie between 5 and 13 in size; 15 and up are set to zero.", size=11, fill=INK_SOFT)
    hgt = footnote(svg, [f"state: raw {a.state}", "harmonics: pinning.band_coeffs"], "", [], y - 20, REPO)
    out = os.path.join(a.o, "echo_harmonics.svg")
    open(out, "w").write(svg.finish(hgt))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
