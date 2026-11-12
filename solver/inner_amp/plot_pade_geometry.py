#!/usr/bin/env python3
"""The Borel-Pade geometry of S_1 against the planted chord and arc (issue #541), one SVG with two panels.

  left   the arc fraction R_k of the [k/k] poles, k = 6..11, at wf 1.00: the real series (scan45J1250) and the plants
         chord2 (two branch points) and arcjump (branch points along the arc), with each one's noise spread at the
         measured per-order errors (pade_geometry.py --noise=measured, post-freeze)
  right  the [10/10] poles in chord coordinates (s along P0 -> kappa_*, d the offset in t units), with the arc's profile

Reads the records of pade_geometry.py from the raw-data repository's folder I541-mac (clone it beside this checkout:
git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw):

    cd solver/inner_amp && python3 plot_pade_geometry.py ../../../choptuik-large-d-raw/I541-mac [-o .]
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "hp"))
sys.path.insert(0, HERE)
from data_graphs import GRID, INK_SOFT, SERIES, Svg, axes, better_hint, callout, caption, footnote, host_specs, legend, marker  # noqa: E402,E501

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
KS = range(6, 12)
WF = "1.0"
NAMES = (("real S₁", SERIES[0], "circle"), ("chord2, two branch points", SERIES[1], "square"),
         ("arcjump, branch points on the arc", SERIES[2], "diamond"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("-o", default=HERE)
    a = ap.parse_args()
    real = json.load(open(os.path.join(a.raw, "real_noise_measured.json")))["real"]["J1250"][WF]["orders"]
    plants = json.load(open(os.path.join(a.raw, "plants_noise_measured.json")))["plants"]
    rows = [real, plants["chord2"][WF]["orders"], plants["arcjump"][WF]["orders"]]

    print("k   " + "  ".join(f"{n[0][:12]:>20s}" for n in NAMES))
    for k in KS:
        print(f"{k:2d}  " + "  ".join(f"{r[str(k)]['R']:+8.3f} ± {r[str(k)]['noise_dR']:8.1e}" for r in rows))
    # poles past kappa_* (s > 0.98, |d| < 0.2, any class) of the real series at every node and J, and of the plants
    allreal = json.load(open(os.path.join(a.raw, "real_noise_measured.json")))["real"]
    print("real poles past kappa_*: J  wf  k  t  |t|  s  d  pole-zero gap")
    for lab, nodes in allreal.items():
        for wf, rec in nodes.items():
            for k in KS:
                for p in rec["orders"][str(k)]["poles"]:
                    if p["s"] > 0.98 and abs(p["d"]) < 0.2:
                        t = complex(*p["t"])
                        print(f"  {lab} {wf} {k:2d}  {t.real:.4f}{t.imag:+.4f}i  {abs(t):.3f}  {p['s']:.3f}  "
                              f"{p['d']:+.3f}  {p['dz']:.1e}")
    for kind, nodes in plants.items():
        smax = max(p["s"] for rec in nodes.values() for k in KS for p in rec["orders"][str(k)]["poles"]
                   if abs(p["d"]) < 0.5 and 0.3 < abs(complex(*p["t"])) < 3)
        print(f"  plant {kind}: largest s of a pole with |d| < 0.5, 0.3 < |t| < 3, k = 6..11: {smax:.3f}")
    for lab, nodes in allreal.items():
        for wf, rec in nodes.items():
            kap = complex(*rec["kappa_ratio"])
            print(f"  ratio law k = 12..21, {lab} {wf}: kappa = {kap.real:.4f}{kap.imag:+.4f}i, |kappa| = {abs(kap):.3f}")

    svg = Svg(1000)
    # ---- left: R_k
    x, y, w, h = 80, 90, 340, 300
    caption(svg, 16, 22, "Arc fraction R_k of the [k/k] Borel-Padé poles, wf 1.00",
            "1 = on the arc through the [6/6] poles, 0 = on the chord P0 → κ⋆; bars: noise spread at the measured errors")
    xm, ym = axes(svg, x, y, w, h, (5.5, 11.5), (-0.4, 1.4), "Padé order k ([k/k], σ₀…σ₂ₖ)", "arc fraction R_k",
                  yfmt=lambda v: f"{v:.1f}")
    svg.line(x, ym(0), x + w, ym(0), INK_SOFT)
    for j, (r, (name, col, shape)) in enumerate(zip(rows, NAMES)):
        off = (j - 1) * 7
        pts = [(xm(k) + off, ym(max(-0.4, min(1.4, r[str(k)]["R"])))) for k in KS]
        svg.add('<polyline points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in pts) +
                f'" fill="none" stroke="{col}" stroke-width="1.5"/>')
        for k, (px, py) in zip(KS, pts):
            e = r[str(k)]["noise_dR"]
            lo, hi = ym(max(-0.4, r[str(k)]["R"] - e)), ym(min(1.4, r[str(k)]["R"] + e))
            svg.line(px, lo, px, hi, col, 1.5)
            marker(svg, shape, px, py, col, f"{name}, k = {k}: R = {r[str(k)]['R']:+.3f} ± {e:.1e}")
    callout(svg, xm(10) - 7, ym(real["10"]["R"]), "real k = 10: on the chord", dx=-30, dy=40)
    better_hint(svg, x, y, w, h, where="right", label="arc")
    legend(svg, 980, 62, [(n, c) for n, c, _ in NAMES])

    # ---- right: [10/10] poles
    x2, w2 = 560, 400
    xs, ys = axes(svg, x2, y, w2, h, (-0.15, 1.15), (-0.2, 0.15), "chord coordinate s (0 = P0, 1 = κ⋆)",
                  "offset from the chord d (t units)", xfmt=lambda v: f"{v:.1f}", yfmt=lambda v: f"{v:.2f}")
    svg.line(x2, ys(0), x2 + w2, ys(0), INK_SOFT)
    import pade_geometry as pg                          # noqa: E402  (the frame is the same for every input)
    fr = pg.Frame(1.0)
    step = max(1, len(fr.prof_s) // 200)
    svg.add('<polyline points="' + " ".join(f"{xs(s):.1f},{ys(d):.1f}" for s, d in
                                            zip(fr.prof_s[::step], fr.prof_d[::step])) +
            f'" fill="none" stroke="{SERIES[-1]}" stroke-width="2" stroke-dasharray="5 3"/>')
    svg.text(xs(0.45), ys(0.118) - 6, "arc through the [6/6] poles", size=10, anchor="middle")
    svg.text(xs(0.2), ys(0) + 14, "chord", size=10)
    beyond = None
    for r, (name, col, shape) in zip(rows, NAMES):
        for p in r["10"]["poles"]:
            on = p["class"] == "geometry"
            far = p["class"] == "doublet" and 0.98 < p["s"] < 1.15 and abs(p["d"]) < 0.2
            if not (on or far):
                continue
            px, py = xs(p["s"]), ys(p["d"])
            marker(svg, shape, px, py, col, f"{name}, [10/10]: t = {p['t'][0]:.4f}{p['t'][1]:+.4f}i, s = {p['s']:.3f},"
                   f" d = {p['d']:+.4f}, pole-zero gap {p['dz']:.1e}")
            if far and r is real:
                beyond = (px, py, abs(complex(*p["t"])))
    if beyond:
        callout(svg, beyond[0], beyond[1], f"|t| = {beyond[2]:.2f}, beyond κ⋆", dx=-40, dy=-50)
    caption(svg, x2 - 60, 22, "[10/10] poles in chord coordinates, wf 1.00",
            "s along P0 → κ⋆, d the offset; the dashed curve is the arc")
    better_hint(svg, x2, y, w2, h, where="right", label="arc")

    rev = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    H = footnote(svg, [f"pade_geometry.py {rev.strip()}", "σ₀…σ₂₂ scan45J1250 (K 45, J 1250, 1280 bits)",
                       "raw I541-mac"], host_specs(),
                 ["the real k = 11 geometry is not determined by the data (noise spread 0.47, J 1250 against J 1500: 0.31)"],
                 y + h + 50, REPO)
    out = os.path.join(a.o, "pade_geometry.svg")
    open(out, "w").write(svg.finish(H))
    print("wrote", out)


if __name__ == "__main__":
    main()
