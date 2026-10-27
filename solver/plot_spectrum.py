#!/usr/bin/env python3
"""The echo's harmonic spectrum: |F_nu| against the harmonic number nu (log scale) for D = 4, 10, 20, 40, 60, with the
D = infinity prediction of the inner problem's amplitude expansion (README, Large D), and the decay rate per harmonic,
(1/2) ln|F_nu / F_{nu+2}|, against 1/n with its extrapolation to D = infinity.

Data: solver/data/echo_spectrum.npz (echo_data.py) and hp/data/fratios.npy (hp/largeD_Fratios.py).
From D = 35 on the double-double runs pin harmonics above 13 to zero, so those lines stop at nu = 13.

    python3 plot_spectrum.py [-o spectrum.svg]
    python3 plot_spectrum.py --print     # the paper's layout, spectrum_print.svg
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_delta as PD  # noqa: E402
from plot_delta import BAND, GRID, INK, INK_SOFT, SERIES, SURFACE, Svg, esc, marker  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# ordinal blue ramp (reference palette steps 250-650; validated with --ordinal in light and dark)
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
RAMP_DARK = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]
SHOW = (4.0, 10.0, 20.0, 40.0, 60.0)
A_INF, F31, F51, QA2 = 2.72, 0.055, 0.0030, 0.0634   # amplitude expansion at omega_inf: |F3/F1|, |F5/F1|, |q| A^2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o")
    ap.add_argument("--print", action="store_true", help="the paper layout, 620 px wide (default -o spectrum_print.svg)")
    args = ap.parse_args()
    P = args.print
    args.o = args.o or os.path.join(HERE, "spectrum_print.svg" if P else "spectrum.svg")
    PD.DARK.update(dict(zip(RAMP, RAMP_DARK)))
    d = np.load(os.path.join(HERE, "data", "echo_spectrum.npz"))
    Ds, Fs = d["D"], d["F"]
    # Layout: panel boxes, panel B's header x and the extra header height (print: a second key row). --print is the
    # paper's: 620 px at 468 pt, no title (the caption carries it).
    if P:
        W, (px, py, pw, ph), (qx, qy, qw, qh), bx, hy = 620, (46, 64, 316, 290), (440, 64, 140, 290), 390, 18
    else:
        W, (px, py, pw, ph), (qx, qy, qw, qh), bx, hy = 900, (80, 120, 440, 330), (630, 120, 215, 330), 590, 0
    svg = Svg(W)
    if not P:
        svg.text(40, 26, "The echo's harmonic spectrum", size=15, fill=INK, weight="bold")
        svg.text(40, 46, "Each odd harmonic of the centre profile is smaller than the last by a factor that grows with D "
                         "and levels off: at D = ∞ the", size=11)
        svg.text(40, 62, "inner problem predicts a factor of about 1/16 per two harmonics, a finite decay rate of "
                         "1.38–1.45 per harmonic.", size=11)

    # ---- panel A: |F_nu| vs nu, log y ----
    nmax = 21
    ylo, yhi = -8, 2     # log10
    sx = lambda nu: px + (nu - 1) / (nmax - 1) * pw
    sy = lambda v: py + ph - (math.log10(v) - ylo) / (yhi - ylo) * ph
    svg.text(px, py - 30 - hy, "|F_ν|, amplitude of harmonic ν (log scale)", size=12, fill=INK)
    kx, ky = px, py - 12 - hy
    for j, Dt in enumerate(SHOW):
        svg.line(kx, ky - 4, kx + 16, ky - 4, RAMP[j], width=2)
        svg.text(kx + 20, ky, f"D = {Dt:g}", size=10)
        kx += 20 + 5.6 * len(f"D = {Dt:g}") + 12
    if P:
        kx, ky = px, py - 12
    svg.line(kx, ky - 4, kx + 16, ky - 4, INK, width=1.5, dash="5 4")
    svg.text(kx + 20, ky, "D = ∞, predicted", size=10)
    for e in range(ylo, yhi + 1):
        y = py + ph - (e - ylo) / (yhi - ylo) * ph
        svg.line(px, y, px + pw, y, GRID)
        sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        lab = "1" if e == 0 else ("10" if e == 1 else "10" + str(e).translate(sup))
        svg.text(px - 8, y + 4, lab, size=10, anchor="end")
    for nu in range(1, nmax + 1, 2):
        svg.line(sx(nu), py, sx(nu), py + ph, GRID)
        svg.text(sx(nu), py + ph + 16, str(nu), size=10, anchor="middle")
    svg.line(px, py + ph, px + pw, py + ph, INK_SOFT)
    svg.text(px + pw / 2, py + ph + 36, "harmonic ν", size=11, anchor="middle")
    # prediction: F1 = 2.72, F3, F5 from the amplitude expansion, then the ratio |q| A^2 per two harmonics
    pred = {1: A_INF, 3: A_INF * F31, 5: A_INF * F51}
    for nu in range(7, nmax + 1, 2):
        pred[nu] = pred[nu - 2] * QA2
    pts = [(sx(nu), sy(v)) for nu, v in sorted(pred.items()) if v >= 10 ** ylo]
    svg.path(pts, INK, width=1.5, dash="5 4", opacity=0.9, title="D = infinity prediction of the amplitude expansion")
    for nu in (3, 5):
        marker(svg, sx(nu), sy(pred[nu]), INK_SOFT, f"D = infinity prediction: |F{nu}/F1| = {pred[nu] / A_INF:.4f}",
               hollow=True, r=3.5)
    last = max(nu for nu, v in pred.items() if v >= 10 ** ylo)
    svg.text(sx(last) - 4, sy(pred[last]) + 16, "D = ∞ (predicted)", size=10, fill=INK, anchor="end")
    for j, Dt in enumerate(SHOW):
        i = int(np.argmin(np.abs(Ds - Dt)))
        D, F = Ds[i], Fs[i]
        top = 13 if D > 35 else nmax
        nus = [nu for nu in range(1, top + 1, 2) if abs(F[nu]) >= 10 ** ylo]
        col = RAMP[j]
        lab = f"D = {D:g}" if abs(D - round(D)) < 1e-9 else f"D = {D:.2f}"
        svg.path([(sx(nu), sy(abs(F[nu]))) for nu in nus], col, width=2, title=lab)
        for nu in nus:
            marker(svg, sx(nu), sy(abs(F[nu])), col, f"{lab}: |F{nu}| = {abs(F[nu]):.3e}", r=3.2)
        nu_end = nus[-1]
        svg.text(sx(nu_end) + 8, sy(abs(F[nu_end])) + 4, f"D = {round(D)}", size=10, fill=INK)

    # ---- panel B: decay rate per harmonic vs 1/n ----
    xr = (0.0, 0.1)
    yr = (0.5, 1.6)
    tx = lambda x: qx + (x - xr[0]) / (xr[1] - xr[0]) * qw
    ty = lambda v: qy + qh - (v - yr[0]) / (yr[1] - yr[0]) * qh
    svg.text(bx, qy - 30 - hy, "Decay rate, ½ ln |F_ν / F_(ν+2)|" if P else
             "Decay rate per harmonic, ½ ln |F_ν / F_(ν+2)|", size=12, fill=INK)
    for v in np.arange(0.6, 1.61, 0.2):
        svg.line(qx, ty(v), qx + qw, ty(v), GRID)
        svg.text(qx - 8, ty(v) + 4, f"{v:.1f}", size=10, anchor="end")
    for Dl, lab in ((None, "∞"), (60, "60"), (30, "30"), (20, "20"), (15, "15"), (12, "12")):
        x = 0.0 if Dl is None else 1 / (Dl - 2)
        svg.line(tx(x), qy, tx(x), qy + qh, GRID)
        svg.text(tx(x), qy + qh + 16, lab, size=10, anchor="middle")
    svg.line(qx, qy + qh, qx + qw, qy + qh, INK_SOFT)
    svg.text(qx + qw / 2, qy + qh + 36, "D (axis linear in 1/(D − 2))", size=11, anchor="middle")
    # prediction band at D = infinity
    b0, b1 = 0.5 * math.log(1 / F31), 0.5 * math.log(1 / QA2)
    svg.add(f'<rect x="{tx(0) - 1:.1f}" y="{ty(b0):.1f}" width="10" height="{ty(b1) - ty(b0):.1f}" fill="{INK}" '
            f'fill-opacity="0.25"><title>D = infinity prediction: {b1:.2f}-{b0:.2f}</title></rect>')
    svg.text(tx(0) + 14, ty(b0) - 6, f"predicted {b1:.2f}–{b0:.2f}", size=10, fill=INK)
    R = np.load(os.path.join(HERE, "..", "hp", "data", "fratios.npy"))
    Dd = R[:, 0].real
    x = 1 / R[:, 1].real
    m = Dd >= 20
    for k, (col_idx, nu) in enumerate(((0, 1), (1, 5), (2, 9))):
        col = SERIES[col_idx]
        rate = 0.5 * np.log(1 / np.abs(R[:, 3 + (nu - 1) // 2]))
        ratio = np.abs(R[:, 3 + (nu - 1) // 2])
        c = np.polyfit(x[m], ratio[m], 3)
        xs = np.linspace(0, x[m].max(), 60)
        svg.path([(tx(a), ty(0.5 * np.log(1 / np.polyval(c, a)))) for a in xs], col, width=1.5, dash="4 3",
                 opacity=0.8, title=f"nu = {nu}: ratio cubic in 1/n through D >= 20, limit rate "
                                    f"{0.5 * np.log(1 / np.polyval(c, 0)):.3f}")
        svg.path([(tx(a), ty(b)) for a, b in zip(x, rate)], col, width=2, title=f"nu = {nu}")
        for a, b, Dv in zip(x, rate, Dd):
            marker(svg, tx(a), ty(b), col, f"D = {Dv:.2f}, nu = {nu}: rate {b:.4f}", r=3.2)
        svg.text(tx(x[0]) + 6, ty(rate[0]) + {1: 4, 5: -2, 9: 10}[nu], f"ν = {nu}", size=10, fill=INK)
    svg.text(bx, qy - 12 - hy, "dashed: extrapolated from D ≥ 20" if P else
             "dashed: extrapolation in 1/n from D ≥ 20", size=10)
    open(args.o, "w").write(svg.finish(qy + qh + 52))
    print("wrote", args.o)


if __name__ == "__main__":
    main()
