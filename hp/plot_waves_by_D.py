#!/usr/bin/env python3
"""Each spacetime dimension D has its own critical solution, with its own wave over one echo period: one row of them.

    cd hp && python3 plot_waves_by_D.py RAWDIR [-o .]

waves_by_D.svg: seven solutions at D 38 to 232.6 (full-system states from held_fit.py below D 73, the chain_fit chain's
states of T138-win2 above, the clean chain's state at the front), each drawn as p1 at the centre over one echo period
on one common vertical scale, with its D, its echo period Delta and its a5, the amount of cos 5 theta in that wave.
Below the row, a5 against D: each solution gives one point, the fit runs through the points below D 73, and the
chain holds harmonic 5 on that fit above it. Standard library and numpy; the primitives are data_graphs.py. Prints the
plotted numbers.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_graphs import INK, INK_SOFT, SERIES, Svg, axes, footnote, marker  # noqa: E402
from held_fit import full_states, load  # noqa: E402
from hpkernel.driver import load_q  # noqa: E402
from pinning import HeldFit  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
CHAIN = ["T138-win2/chain/anm_pin3_D100.0000.npz", "T138-win2/chain/anm_pin3_D150.0000.npz",
         "T138-win2/chain_o/anm_pin3_D200.0000.npz", "I461-win2/crawlG/anm_pin3_D232.6312.npz"]
FULL_D = [38.02, 52.0, 73.0]
M = 32


def coeffs(f, ks):
    th = np.pi * np.arange(M) / M
    return {k: ((2 / M) * np.sum(f * np.cos(k * th)), (2 / M) * np.sum(f * np.sin(k * th))) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("-o", default=HERE)
    a = ap.parse_args()
    ks = list(range(1, M, 2))
    sols = []
    pts = full_states(a.raw)
    for D0 in FULL_D:
        p = min(pts, key=lambda r: abs(float(r["D"]) - D0))
        D, z, m, delta = load(p["path"])
        sols.append((float(D), float(delta), coeffs(np.asarray(z[:M], float), ks), "full solution"))
    for rel in CHAIN:
        z, m = load_q(os.path.join(a.raw, rel))
        f = np.asarray(z[0], float)[:M] + np.asarray(z[1], float)[:M]
        sols.append((float(m["D"]), float(z[0][-1]) + float(z[1][-1]), coeffs(f, ks), "chain, harmonics 5-13 held"))
    fit = HeldFit(os.path.join(HERE, "data", "held_fit.json"))
    print("| D | Delta | a5 | kind |\n|---|---|---|---|")
    for D, dl, c, kind in sols:
        print(f"| {D:g} | {dl:.10f} | {c[5][0]:.6e} | {kind} |")

    th = np.linspace(0, 2 * np.pi, 241)
    waves = [sum(x * np.cos(k * th) + y * np.sin(k * th) for k, (x, y) in c.items()) for _, _, c, _ in sols]
    amp = max(float(np.max(np.abs(w))) for w in waves) * 1.1

    W = 1040
    svg = Svg(W)
    svg.text(16, 26, "One solution per dimension D, each with its own wave", size=15, fill=INK, weight="bold")
    svg.text(16, 46, "D is not a time or a place: it is the number of spacetime dimensions, a setting of the problem. For each "
             "D there is a separate critical solution.", size=11)
    svg.text(16, 61, "Each repeats in its own echo period Δ, and over one period its centre field p1 is a wave. Higher D gives "
             "a smaller wave, a shorter period and a smaller harmonic 5.", size=11)
    n = len(sols)
    x0, gap, tw, th_, ty = 24, 12, (W - 48 - 6 * 12) / 7, 110, 96
    for i, ((D, dl, c, kind), w) in enumerate(zip(sols, waves)):
        bx = x0 + i * (tw + gap)
        col = SERIES[2] if kind.startswith("full") else SERIES[0]
        svg.add(f'<rect x="{bx:.1f}" y="{ty}" width="{tw:.1f}" height="{th_}" rx="6" fill="none" stroke="{col}" stroke-opacity="0.5"/>')
        mid = ty + th_ / 2
        svg.line(bx + 4, mid, bx + tw - 4, mid, INK_SOFT, 0.5)
        line = " ".join(f"{bx + 4 + (tw - 8) * t / (2 * np.pi):.1f},{mid - (th_ / 2 - 6) * v / amp:.1f}" for t, v in zip(th, w))
        svg.add(f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="1.8"/>')
        svg.text(bx + tw / 2, ty + th_ + 18, f"D {D:.1f}", size=12, fill=INK, anchor="middle", weight="bold")
        svg.text(bx + tw / 2, ty + th_ + 34, f"Δ = {dl:.4f}", size=11, fill=INK, anchor="middle")
        svg.text(bx + tw / 2, ty + th_ + 49, f"a5 = {c[5][0]:.2e}", size=11, fill=INK_SOFT, anchor="middle")
    svg.text(24, ty - 8, "each box: p1 over one echo period of that solution, all on one vertical scale", size=11, fill=INK_SOFT)

    # a5 against D: one point per solution
    x, y, w, h = 96, ty + th_ + 110, 760, 220
    Dg = np.linspace(38.02, 240, 300)
    curve = [fit.values(float(D), [5])[0][0] for D in Dg]
    xm, ym = axes(svg, x, y, w, h, (30, 240), (0, max(c[5][0] for _, _, c, _ in sols) * 1.15), "spacetime dimension D",
                  "a5, from each solution's wave", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:.4f}")
    svg.text(16, y - 18, "Harmonic 5 against D: each solution above gives one point", size=12, fill=INK, weight="bold")
    svg.add(f'<polyline points="{" ".join(f"{xm(D):.1f},{ym(v):.1f}" for D, v in zip(Dg, curve))}" fill="none" '
            f'stroke="{SERIES[1]}" stroke-width="2"/>')
    for r in pts:
        if float(r["D"]) >= 38.0:
            marker(svg, "circle", xm(float(r["D"])), ym(r["c"]["5a"]), SERIES[2], f"D {float(r['D']):.2f}: a5 {r['c']['5a']:.5e}")
    for D, dl, c, kind in sols:
        marker(svg, "diamond", xm(D), ym(c[5][0]), SERIES[2] if kind.startswith("full") else SERIES[0], f"D {D:g}: a5 {c[5][0]:.5e}")
    lx, ly = x + w + 24, y + 20
    for text, col, kind in (("full solutions (every harmonic solved)", SERIES[2], "dot"), ("chain solutions (harmonics 5-13 held)", SERIES[0], "dot"),
                            ("the fit, extended past D 73", SERIES[1], "line")):
        if kind == "dot":
            marker(svg, "circle", lx + 5, ly - 4, col, text)
        else:
            svg.line(lx - 2, ly - 4, lx + 12, ly - 4, col, 2)
        for j, part in enumerate(text.split(" (")):
            svg.text(lx + 18, ly + 14 * j, part if j == 0 else "(" + part, size=11, fill=INK if j == 0 else INK_SOFT)
        ly += 40
    hgt = footnote(svg, ["full solutions: held_fit.py's states", "chain: raw T138-win2, I461-win2/crawlG", "fit: data/held_fit.json"],
                   "", [], y + h + 44, REPO)
    out = os.path.join(a.o, "waves_by_D.svg")
    open(out, "w").write(svg.finish(hgt))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
