#!/usr/bin/env python3
"""The echo across dimensions: small multiples of the critical solution's centre profile F(tau) = sqrt(D-2) dp/dx at
r = 0, for D = 4, 10, 20, 60, over the same range of the logarithmic time tau, normalized to each profile's peak.
A dashed line shows the first harmonic alone, so the departure from a pure sine is visible; ticks mark the echoes
(one period Delta each, after which the solution repeats a factor e^Delta smaller).

Data: solver/data/echo_spectrum.npz (written by echo_data.py from the saved solutions).

    python3 plot_echo.py [-o echo.svg]
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_delta import GRID, INK, INK_SOFT, NEUTRAL, SERIES, Svg, esc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SHOW = (4.0, 10.0, 20.0, 60.0)
TMAX = 8.8


def profile(Fk, theta):
    k = np.arange(len(Fk))
    return np.real(np.exp(1j * np.outer(theta, k)) @ Fk)


def aligned_phase(Fk):
    """theta0 of the upward zero crossing of F nearest the start, so every panel starts at the same point of its echo"""
    th = np.linspace(0, 2 * np.pi, 4001)
    f = profile(Fk, th)
    i = np.where((f[:-1] < 0) & (f[1:] >= 0))[0][0]
    return th[i] - f[i] * (th[i + 1] - th[i]) / (f[i + 1] - f[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "echo.svg"))
    args = ap.parse_args()
    d = np.load(os.path.join(HERE, "data", "echo_spectrum.npz"))
    Ds, Fs, Deltas = d["D"], d["F"], d["Delta"]
    W, px, pw, ph, gap = 900, 70, 800, 92, 50
    svg = Svg(W)
    svg.text(px, 26, "The echo across dimensions", size=15, fill=INK, weight="bold")
    svg.text(px, 46, "The field's slope at the centre over the same stretch of logarithmic time τ, scaled to its peak. "
                     "Each echo repeats the last", size=11)
    svg.text(px, 62, "a factor e^Δ smaller; as D grows the echoes come faster and the profile turns into a pure sine.",
             size=11)
    # key
    ky = 86
    svg.line(px, ky - 4, px + 24, ky - 4, SERIES[0], width=2)
    svg.text(px + 30, ky, "centre profile F(τ) = √(D−2) ∂p/∂x at r = 0", size=10)
    kx = px + 30 + 5.6 * 46 + 24
    svg.line(kx, ky - 4, kx + 24, ky - 4, NEUTRAL, width=1.5, dash="5 4")
    svg.text(kx + 30, ky, "its first harmonic alone (a pure sine)", size=10)
    y0 = 110
    for j, Dt in enumerate(SHOW):
        i = int(np.argmin(np.abs(Ds - Dt)))
        D, Fk, Dl = Ds[i], Fs[i], Deltas[i]
        w = 2 * np.pi / Dl
        th0 = aligned_phase(Fk)
        tau = np.linspace(0, TMAX, 900)
        f = profile(Fk, th0 + w * tau)
        f1 = profile(np.r_[0, Fk[1], np.zeros(len(Fk) - 2)], th0 + w * tau)
        peak = np.abs(profile(Fk, np.linspace(0, 2 * np.pi, 4001))).max()
        py = y0 + j * (ph + gap)
        sx = lambda t: px + t / TMAX * pw
        sy = lambda v: py + ph / 2 - v / 1.25 * (ph / 2)
        # frame: zero line, echo ticks
        svg.line(px, sy(0), px + pw, sy(0), GRID)
        for v in (-1, 1):
            svg.line(px, sy(v), px + pw, sy(v), GRID)
            svg.text(px - 6, sy(v) + 4, f"{v:+d}", size=9, anchor="end")
        svg.text(px - 6, sy(0) + 4, "0", size=9, anchor="end")
        k = 0
        while k * Dl <= TMAX + 1e-9:
            x = sx(k * Dl)
            svg.line(x, py, x, py + ph, INK_SOFT, dash="2 3", opacity=0.6)
            k += 1
        svg.path([(sx(t), sy(v / peak)) for t, v in zip(tau, f1)], NEUTRAL, width=1.5, dash="5 4", opacity=0.9,
                 title=f"D = {D:g}: first harmonic alone")
        r3 = abs(Fk[3] / Fk[1])
        svg.path([(sx(t), sy(v / peak)) for t, v in zip(tau, f)], SERIES[0], width=2,
                 title=f"D = {D:g}: Delta = {Dl:.6f}, peak F = {peak:.3f}, |F3/F1| = {r3:.3f}")
        # labels: dimension on the left above the panel, echo facts on the right
        lab = f"D = {D:g}" if abs(D - round(D)) < 1e-9 else f"D = {D:.2f}"
        svg.text(px, py - 8, lab, size=12, fill=INK, weight="bold")
        svg.text(px + 8 + 7.4 * len(lab), py - 8, f"Δ = {Dl:.3f}: each echo e^Δ = {math.exp(Dl):.1f}× smaller than the last;  "
                                  f"third harmonic {100 * r3:.0f}% of the first;  peak F = {peak:.2f}",
                 size=10, fill=INK_SOFT)
    yb = y0 + (len(SHOW) - 1) * (ph + gap) + ph
    for t in range(0, int(TMAX) + 1):
        svg.line(px + t / TMAX * pw, yb, px + t / TMAX * pw, yb + 4, INK_SOFT)
        svg.text(px + t / TMAX * pw, yb + 16, f"{t}", size=10, anchor="middle")
    svg.line(px, yb, px + pw, yb, INK_SOFT)
    svg.text(px + pw / 2, yb + 34, "τ, logarithmic time to the singularity (dotted lines: the start of each echo)",
             size=11, anchor="middle")
    open(args.o, "w").write(svg.finish(yb + 50))
    print("wrote", args.o)


if __name__ == "__main__":
    main()
