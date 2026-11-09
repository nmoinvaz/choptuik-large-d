#!/usr/bin/env python3
"""The extra zeros of det M(lambda) above D = 54 against resolution and cut (notes/MODE_COUNT.md section 5).

Left: D = 55.04, the zeros of the M = 48 count (#238) confirmed by small circles, the M = 40 circles drawn at those
positions (#231), which wind 0, and M = 40's own zeros; the M = 48 pair at D = 54.04 is joined to its D = 55.04 place.
Right: D = 61.03, M = 48, the row of zeros at the cut kf 21 and kf 23 (#230). lambda_phys is the control in both.
Upper half plane only: det M(conj lambda) = conj det M(lambda).

Reads from the raw-data repository: I238-macm3/{D54,D55}/data/modecount_circles.jsonl, I231-macm3/D55_M40/data/
modecount_circles.jsonl and moments.json, and I230-mac/data/circles_kf2[13]_*.jsonl.

    python3 hp/plot_modecount_extras.py ../choptuik-large-d-raw [-o hp/modecount_extras.svg]
"""
import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_graphs import (INK, INK_SOFT, SERIES, SURFACE, Svg, axes, better_arrow, callout, caption,  # noqa: E402
                         footnote, marker)

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
LAM48 = "λ_phys"
C48, C40, CPHYS = SERIES[0], SERIES[1], SERIES[-1]


def jl(path):
    return [json.loads(x) for x in open(path) if x.strip().startswith("{")]


def upper(zs, tol=1e-6):
    """distinct zeros with Im >= 0, from circle records' 'zeros' lists"""
    out = []
    for z in zs:
        if z[1] < -tol:
            continue
        if all(math.hypot(z[0] - a, z[1] - b) > 1e-3 for a, b in out):
            out.append((z[0], z[1]))
    return out


def circle_zeros(recs, M=None):
    """zeros of circles that wound > 0 (lambda_phys circles excluded), and lambda_phys's"""
    zs, phys = [], []
    for r in recs:
        if M is not None and r.get("M") != M:
            continue
        if r.get("winding", 0) < 0.5 or not r.get("zeros"):
            continue
        c = r["center"]
        (phys if abs(c[1]) < 1e-9 and 1.9 < c[0] < 2.2 else zs).extend(r["zeros"])
    return upper(zs), upper(phys)


def load(raw):
    d = {}
    d55 = jl(os.path.join(raw, "I238-macm3", "D55", "data", "modecount_circles.jsonl"))
    d54 = jl(os.path.join(raw, "I238-macm3", "D54", "data", "modecount_circles.jsonl"))
    m40 = jl(os.path.join(raw, "I231-macm3", "D55_M40", "data", "modecount_circles.jsonl"))
    d["z55"], d["p55"] = circle_zeros(d55, 48)
    d["z54"], _ = circle_zeros(d54, 48)
    d["z40"], d["p40"] = circle_zeros(m40, 40)
    # the M 40 circles that wound 0 (placed at the M 48 zeros)
    d["miss40"] = [(r["center"][0], r["center"][1], r["R"]) for r in m40 if r.get("winding", 1) == 0]
    # M 40's moment positions (no circle unless one is in z40)
    mom = []
    for r in jl(os.path.join(raw, "I231-macm3", "D55_M40", "moments.json")):
        if r.get("ok"):
            mom += [z for z in r.get("roots", []) if z[1] > 0]
    d["mom40"] = [z for z in upper(mom) if all(math.hypot(z[0] - a, z[1] - b) > 0.05 for a, b in d["z40"])]
    kf = {}
    for k in (21, 23):
        recs = []
        for f in sorted(glob.glob(os.path.join(raw, "I230-mac", "data", f"circles_kf{k}_*.jsonl"))):
            recs += jl(f)
        kf[k] = circle_zeros(recs)
    d["kf21"], d["p21"] = kf[21]
    d["kf23"], d["p23"] = kf[23]
    return d


def nearest(z, pts):
    return min(pts, key=lambda p: math.hypot(p[0] - z[0], p[1] - z[1]))


def hollow(svg, x, y, color, title, shape="circle"):
    r = 5.5
    if shape == "diamond":
        pts = f"{x:.1f},{y - r - 1:.1f} {x + r + 1:.1f},{y:.1f} {x:.1f},{y + r + 1:.1f} {x - r - 1:.1f},{y:.1f}"
        body = f'<polygon points="{pts}" fill="{SURFACE}" stroke="{color}" stroke-width="2"/>'
    else:
        body = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r - 1}" fill="{SURFACE}" stroke="{color}" stroke-width="2"/>'
    svg.add(f"<g>{body}<title>{title}</title></g>")


def arrow(svg, x1, y1, x2, y2, color):
    ang = math.atan2(y2 - y1, x2 - x1)
    ln = math.hypot(x2 - x1, y2 - y1)
    xa, ya = x2 - 9 * math.cos(ang), y2 - 9 * math.sin(ang)
    if ln > 18:
        svg.line(x1 + 8 * math.cos(ang), y1 + 8 * math.sin(ang), xa, ya, color, 1.5)
    hx, hy = xa - 6 * math.cos(ang), ya - 6 * math.sin(ang)
    px, py = -math.sin(ang) * 4, math.cos(ang) * 4
    svg.add(f'<polygon points="{xa:.1f},{ya:.1f} {hx + px:.1f},{hy + py:.1f} {hx - px:.1f},{hy - py:.1f}" '
            f'fill="{color}"/>')


def fz(z):
    return f"{z[0]:+.4f} {'+' if z[1] >= 0 else '−'} {abs(z[1]):.4f}i"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("-o", default=os.path.join(HERE, "modecount_extras.svg"))
    a = ap.parse_args(argv)
    d = load(a.raw)

    print("| D | M | cut | what | λ (upper half) |")
    print("|---|---|---|---|---|")
    for z in d["z54"]:
        print(f"| 54.04 | 48 | 23 | zero, circle +1 | {fz(z)} |")
    for z in d["z55"]:
        print(f"| 55.04 | 48 | 23 | zero, circle +1 | {fz(z)} |")
    for c in d["miss40"]:
        print(f"| 55.04 | 40 | 23 | circle R {c[2]:g} winds 0 | centre {fz(c)} |")
    for z in d["z40"]:
        print(f"| 55.04 | 40 | 23 | zero, circle +1 | {fz(z)} |")
    for z in d["mom40"]:
        print(f"| 55.04 | 40 | 23 | zero, moment only | {fz(z)} |")
    for k in (21, 23):
        for z in d[f"kf{k}"]:
            print(f"| 61.03 | 48 | {k} | zero, circle +1 | {fz(z)} |")
    for lab, p in (("55.04 M 48", d["p55"]), ("55.04 M 40", d["p40"]), ("61.03 kf 21", d["p21"]),
                   ("61.03 kf 23", d["p23"])):
        for z in p:
            print(f"| {lab} | | | λ_phys control, circle +1 | {fz(z)} |")

    W = 1000
    svg = Svg(W)
    svg.text(16, 24, "The extra zeros above D ≈ 54 move with the resolution and the cut, λ_phys does not",
             size=14, fill=INK, weight="bold")
    svg.text(16, 42, "Zeros of det M(λ) beyond the expected ones, each confirmed by a small circle (upper half plane). "
             "A genuine growth rate would keep its λ.", size=11)
    leg = [("M 48 zero", C48, "circle"), ("M 48 zero at D 54.04", C48, "triangle"), ("M 40 zero", C40, "diamond"),
           ("λ_phys (control)", CPHYS, "cross")]
    lx = 16
    for lab, col, shp in leg:
        marker(svg, shp, lx + 6, 62, col, lab)
        svg.text(lx + 16, 66, lab, size=11, fill=INK)
        lx += 28 + 6.3 * len(lab)
    hollow(svg, lx + 6, 62, C40, "moment only", "diamond")
    svg.text(lx + 16, 66, "M 40, moment only", size=11, fill=INK)
    lx += 140
    svg.add(f'<circle cx="{lx + 6}" cy="62" r="7" fill="none" stroke="{C40}" stroke-width="1.5" '
            f'stroke-dasharray="3 2"/>')
    svg.text(lx + 18, 66, "M 40 circle that winds 0", size=11, fill=INK)

    # left panel: D = 55.04
    x, y, w, h = 80, 128, 420, 430
    caption(svg, x - 60, y - 24, "D = 55.04, cut kf 23", "M 48's zeros are absent at M 40, which has others")
    xm, ym = axes(svg, x, y, w, h, (-0.4, 2.2), (0, 5.2), "Re λ (growth rate)", "Im λ",
                  xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    svg.add(f'<line x1="{xm(0):.1f}" y1="{y}" x2="{xm(0):.1f}" y2="{y + h}" stroke="{INK_SOFT}" '
            f'stroke-width="1.5" stroke-dasharray="6 4"/>')
    svg.text(xm(0) + 5, y + 14, "Re λ = 0", size=11)
    better_arrow(svg, xm(-0.06), y + 34, xm(-0.36), y + 34, label="decays")
    for cx, cy, R in d["miss40"]:
        svg.add(f'<ellipse cx="{xm(cx):.1f}" cy="{ym(cy):.1f}" rx="{R * (xm(1) - xm(0)):.1f}" '
                f'ry="{R * (ym(0) - ym(1)):.1f}" fill="none" stroke="{C40}" stroke-width="1.5" '
                f'stroke-dasharray="3 2"/>')
    pair55 = min(d["z55"], key=lambda z: z[1])
    for z in d["z54"]:
        marker(svg, "triangle", xm(z[0]), ym(z[1]), C48, f"M 48, D 54.04: {fz(z)}")
    if d["z54"]:
        z0 = nearest(pair55, d["z54"])
        arrow(svg, xm(z0[0]), ym(z0[1]), xm(pair55[0]), ym(pair55[1]), C48)
        svg.text(xm(z0[0]) - 8, ym(z0[1]) - 10, "D 54.04", size=10, anchor="end")
    for z in d["z55"]:
        marker(svg, "circle", xm(z[0]), ym(z[1]), C48, f"M 48: {fz(z)}")
    for z in d["mom40"]:
        hollow(svg, xm(z[0]), ym(z[1]), C40, f"M 40, moment only: {fz(z)}", "diamond")
    for z in d["z40"]:
        marker(svg, "diamond", xm(z[0]), ym(z[1]), C40, f"M 40: {fz(z)}")
    for z in d["p55"] + d["p40"]:
        marker(svg, "cross", xm(z[0]), ym(z[1]), CPHYS, f"λ_phys: {fz(z)}")
    callout(svg, xm(pair55[0]), ym(pair55[1]), "M 48 pair crosses Re λ = 0 between D 54 and 55", dx=40, dy=-62)

    # right panel: D = 61.03, M = 48
    x2, w2 = 590, 370
    caption(svg, x2 - 60, y - 24, "D = 61.03, M 48", "the row at cut kf 21 (open) and kf 23 (filled)")
    xm2, ym2 = axes(svg, x2, y, w2, h, (1.1, 3.3), (0, 1.6), "Re λ (growth rate)", "Im λ",
                    xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    for z in d["kf21"]:
        hollow(svg, xm2(z[0]), ym2(z[1]), C48, f"kf 21: {fz(z)}")
        z1 = nearest(z, d["kf23"])
        svg.line(xm2(z[0]), ym2(z[1]), xm2(z1[0]), ym2(z1[1]), C48, 1.5)
    for z in d["kf23"]:
        marker(svg, "circle", xm2(z[0]), ym2(z[1]), C48, f"kf 23: {fz(z)}")
    for z in d["p21"] + d["p23"]:
        marker(svg, "cross", xm2(z[0]), ym2(z[1]), CPHYS, f"λ_phys: {fz(z)}")
    if d["p21"] and d["p23"]:
        dp = math.hypot(d["p21"][0][0] - d["p23"][0][0], d["p21"][0][1] - d["p23"][0][1])
        callout(svg, xm2(d["p23"][0][0]), ym2(d["p23"][0][1]), f"λ_phys moves {dp:.1e}", dx=40, dy=-250)
    new = [z for z in d["kf23"] if math.hypot(*(lambda q: (q[0] - z[0], q[1] - z[1]))(nearest(z, d["kf21"]))) > 0.3]
    if new:
        callout(svg, xm2(new[0][0]), ym2(new[0][1]), "new pair at kf 23", dx=-40, dy=-40)

    hgt = footnote(svg, ["hp/modecount_circle.py, hp/modecount_moment.py (main 7035fa0)",
                         "raw I238-macm3 d44579f, I231-macm3 a0f232a, I230-mac e7dc533"],
                   "circles on macm3 (Apple M3) and mac (Apple M5), double-double det M", [], y + h + 44, REPO)
    open(a.o, "w").write(svg.finish(hgt))
    print(f"wrote {a.o}")


if __name__ == "__main__":
    main(sys.argv[1:])
