#!/usr/bin/env python3
"""The S^3 sigma-model's echoing period Delta(eta) at D = 4 (matter/SIGMA_DSS.md; #623, #630, #631, #633): the whole
DSS branch from the CSS-DSS transition near eta_c = 0.170 to eta = infinity at k_c = 15, and, beside it, the minimum
near eta = 7 at k_c = 15, 23 and 39 as Delta(eta) - Delta(infinity) of each k_c.

    python3 solver/matter/plot_sigma_delta.py ../choptuik-large-d-raw [-o matter/sigma_delta_eta.svg]

The k_c = 15 points are the continuation in solver/matter/data/sigdss_d4eta15.npz and the targets of the later
continuations (I630-win, I631-win, I633-win); the k_c = 23 and 39 points are the refines in I623-win, I630-win,
I631-win and I633-win of the raw-data repository. eta = infinity is drawn at the right end of each axis.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "hp"))
from data_graphs import INK, INK_SOFT, SERIES, Svg, axes, callout, caption, footnote, legend, marker  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
FOLDERS = ("I623-win", "I630-win", "I631-win", "I633-win")
INF = 100.0     # where eta = infinity is drawn on the log axes
COLORS = {15: SERIES[-1], 23: SERIES[0], 39: SERIES[1]}


def load(raw):
    """{k_c: {eta: Delta}} with eta = inf for eps = 0"""
    pts = {15: {}, 23: {}, 39: {}}
    d = np.load(os.path.join(ROOT, "solver/matter/data/sigdss_d4eta15.npz"))
    for eps, z in zip(d["v"], d["Z"]):
        pts[15][math.inf if eps == 0 else round(1 / eps, 6)] = float(z[-1])
    for folder in FOLDERS:
        for f in sorted(glob.glob(os.path.join(raw, folder, "data", "sigdss_*.jsonl"))):
            for line in open(f):
                if not line.strip().startswith("{"):
                    continue
                r = json.loads(line)
                if r.get("D") != 4.0 or "Delta" not in r or ("F" in r and r["F"] > 1e-10):
                    continue
                if "refine" not in os.path.basename(f) and not r.get("target"):
                    continue
                eta = math.inf if r["eps"] == 0 else round(1 / r["eps"], 6)
                pts.setdefault(r["kc"], {})[eta] = float(r["Delta"])
    return pts


def xpos(eta):
    return INF if eta == math.inf else eta


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("-o", default=os.path.join(ROOT, "matter", "sigma_delta_eta.svg"))
    a = ap.parse_args(argv)
    pts = load(a.raw)
    print("| k_c | eta | Delta | Delta - Delta(inf) |")
    print("|---|---|---|---|")
    for kc in (15, 23, 39):
        dinf = pts[kc].get(math.inf)
        for eta in sorted(pts[kc]):
            v = pts[kc][eta]
            print(f"| {kc} | {eta:g} | {v:.10f} | {v - dinf:+.3e} |")

    W = 980
    svg = Svg(W)
    svg.text(16, 24, "The σ-model's echoing period Δ(η) at D = 4", size=14, fill=INK, weight="bold")
    svg.text(16, 42, "S³ wave maps coupled to gravity, η the coupling. η = ∞ is drawn at the right end of each axis.",
             size=11)
    fmt_eta = lambda v: "∞" if v >= INF else f"{v:g}"

    # left: the whole DSS branch at k_c = 15
    x, y, w, h = 76, 110, 380, 360
    caption(svg, x, 80, "The DSS branch, k_c = 15", "Δ falls from the CSS–DSS transition near η_c = 0.170")
    xm, ym = axes(svg, x, y, w, h, (0.15, INF, "log"), (0.4, 1.4), "coupling η (log)", "echoing period Δ",
                  xfmt=fmt_eta, yfmt=lambda v: f"{v:.1f}")
    p15 = sorted((xpos(e), v) for e, v in pts[15].items())
    svg.add('<polyline points="' + " ".join(f"{xm(e):.1f},{ym(v):.1f}" for e, v in p15)
            + f'" fill="none" stroke="{COLORS[15]}" stroke-width="2"/>')
    for e, v in p15:
        marker(svg, "circle", xm(e), ym(v), COLORS[15], f"k_c 15, η {fmt_eta(e)}: Δ {v:.7f}")
    e0 = min(pts[15])
    callout(svg, xm(e0), ym(pts[15][e0]), f"η = {e0:.2f}: Δ {pts[15][e0]:.3f}", dx=34, dy=-10)
    callout(svg, xm(INF), ym(pts[15][math.inf]), "η = ∞, the ℓ = 1 multiplet", dx=-30, dy=-40)

    # right: the minimum, Delta(eta) - Delta(inf) of each k_c, in units of 1e-5
    x2, w2 = 576, 380
    caption(svg, x2, 80, "The minimum near η = 7", "Δ(η) − Δ(∞) per k_c; it dips below Δ(∞) and comes back up")
    xm2, ym2 = axes(svg, x2, y, w2, h, (4, INF, "log"), (-8, 1), "coupling η (log)",
                    "Δ(η) − Δ(∞)  (×10⁻⁵)", xfmt=fmt_eta, yfmt=lambda v: f"{v:g}")
    svg.add(f'<line x1="{x2}" y1="{ym2(0):.1f}" x2="{x2 + w2}" y2="{ym2(0):.1f}" stroke="{INK_SOFT}" '
            f'stroke-width="1.5" stroke-dasharray="6 4"/>')
    for kc, shape in ((15, "circle"), (23, "square"), (39, "diamond")):
        dinf = pts[kc][math.inf]
        pk = sorted((xpos(e), (v - dinf) * 1e5) for e, v in pts[kc].items() if e >= 4)
        svg.add('<polyline points="' + " ".join(f"{xm2(e):.1f},{ym2(v):.1f}" for e, v in pk)
                + f'" fill="none" stroke="{COLORS[kc]}" stroke-width="2"/>')
        for e, v in pk:
            marker(svg, shape, xm2(e), ym2(v), COLORS[kc], f"k_c {kc}, η {fmt_eta(e)}: {v:+.3f}e-5")
    # the parabola's vertex through eta = 6, 7, 8.5 at k_c = 39 (#633)
    e3 = [6.0, 7.0, 8.5]
    if all(e in pts[39] for e in e3):
        ys = [pts[39][e] - pts[39][math.inf] for e in e3]
        c = np.polyfit(e3, ys, 2)
        ev = -c[1] / (2 * c[0])
        callout(svg, xm2(ev), ym2(np.polyval(c, ev) * 1e5), f"minimum near η ≈ {ev:.1f} (k_c 39)", dx=40, dy=-110)
    svg.text(xm2(40), ym2(0) - 8, "Δ(∞)", size=11, fill=INK_SOFT, anchor="end")
    legend(svg, x2 + 210, y + 20, [("k_c 15", COLORS[15]), ("k_c 23", COLORS[23]), ("k_c 39", COLORS[39])])

    hgt = footnote(svg, ["sigdss_run.py cont and refine", "raw I623/I630/I631/I633-win"],
                   "win (i7-1185G7, Windows)", [], y + h + 44, REPO)
    open(a.o, "w").write(svg.finish(hgt))
    print(f"wrote {a.o}")


if __name__ == "__main__":
    main(sys.argv[1:])
