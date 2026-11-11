#!/usr/bin/env python3
"""How the held harmonics' fit is made, and how the chain then steps in D, animated in one SVG.

    cd hp && python3 plot_fit_steps.py RAWDIR [-o .] [--frame T]

fit_steps.svg plays five steps on a loop (SMIL, as plot_algorithm.py's figures):
  1. the full-system solutions below D 73: p1 over one echo period at six of them;
  2. harmonic 5 read from each wave (drawn magnified), its cosine coefficient a5 becoming a point against D;
  3. the least-squares fit of degree 8 drawn through all 138 points (held_fit.py, data/held_fit.json);
  4. the fit extended to D 240, beside the fits of degree 6 and 10, and its value at the chain's front setting
     harmonic 5 of the wave there;
  5. the calculus of the chain's step: Delta of the clean chain's states against D with the tangent at a base, and the
     size of each Taylor term |z_k| a^k of that step (anm_run.jsonl), summed to the prediction that Newton corrects.
--frame T writes the still at time T. Standard library and numpy; the primitives are data_graphs.py and the clock of
plot_algorithm.py. Prints the plotted numbers.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_algorithm as pa  # noqa: E402
from chain_monitor import parse_log, rows_of  # noqa: E402
from data_graphs import INK, INK_SOFT, SERIES, Svg, axes, footnote, marker  # noqa: E402
from held_fit import full_states, load  # noqa: E402
from octo_corrector import load_o  # noqa: E402
from pinning import HeldFit  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
FRONT = "I461-win2/crawlG/anm_pin3_D232.6312.npz"
CHAIN_LOGS = ["I461-win2/crawlG.log", "I461-win2/crawlH.log", "I461-win2/crawlI.log"]
JSONL = "I461-win2/crawlG/anm_run.jsonl"
BASE_D = 228.13125
T = 44.0
STEPS = [(0.0, 7.0), (7.0, 17.0), (17.0, 23.0), (23.0, 31.0), (31.0, 44.0)]
TITLES = [
    ("1. Take the full solutions", "Below D 73 every harmonic can still be solved. Here is p1 at the centre over one echo period, at six of the 138."),
    ("2. Read harmonic 5 from each", "Each wave contains a small harmonic-5 part (drawn 40 times larger). Its cosine coefficient a5 becomes one point against D."),
    ("3. Fit a polynomial through the points", "A least-squares polynomial of degree 8 in 1/(D−2), times 1/√(D−2), passes within 10⁻¹² of all 138 points."),
    ("4. Extend it to larger D", "Past D 73 the polynomial gives the held value. At D 232.6 it sets harmonic 5 of the wave. Fits of degree 6 and 10 differ by 3×10⁻⁹ there."),
    ("5. Step in D with calculus", "At a solved D the chain computes the derivatives of the whole solution in D, 24 of them. Their Taylor sum predicts the next D; Newton then corrects it."),
]
EXAMPLE_D = [38.02, 45.0, 52.0, 59.0, 66.0, 73.0]


def coeffs(f, M, ks):
    """cos and sin coefficients of an antiperiodic field from its samples on the half grid theta_j = pi j / M"""
    th = np.pi * np.arange(M) / M
    return {k: ((2 / M) * np.sum(f * np.cos(k * th)), (2 / M) * np.sum(f * np.sin(k * th))) for k in ks}


def wave(co, th, ks=None):
    return sum(a * np.cos(k * th) + b * np.sin(k * th) for k, (a, b) in co.items() if ks is None or k in ks)


def polylen(pts):
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pts, pts[1:]))


def draw_in(clock, pts_xy, color, width, t0, t1, dash=None):
    """a polyline drawn from its start between t0 and t1 (stroke-dashoffset), shown until the cycle's end"""
    L = polylen(pts_xy) + 1
    d = pa.path_d(pts_xy)
    kt = ";".join(f"{v:.4f}" for v in (0, t0 / clock.T, t1 / clock.T, 1))
    anim = (f'<animate attributeName="stroke-dashoffset" values="{L:.0f};{L:.0f};0;0" keyTimes="{kt}" dur="{clock.T}s" '
            f'repeatCount="indefinite"/>')
    if clock.frame is not None:
        f = 0 if clock.frame >= t1 else (L if clock.frame <= t0 else L * (1 - (clock.frame - t0) / (t1 - t0)))
        return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-dasharray="{L:.0f}" stroke-dashoffset="{f:.0f}"/>'
    extra = f' stroke-dasharray="{L:.0f}"'
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{extra} stroke-dashoffset="{L:.0f}">{anim}</path>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("-o", default=HERE)
    ap.add_argument("--frame", type=float, default=None)
    a = ap.parse_args()
    pa.FRAME = a.frame
    clock = pa.Clock(T)

    # data: the full solutions and their waves
    fitd = json.load(open(os.path.join(HERE, "data", "held_fit.json")))
    pts = [p for p in full_states(a.raw) if float(p["D"]) >= float(fitd["Dmin"]) - 1e-6]
    M = 32
    ks = list(range(1, M, 2))
    ex = []
    for D0 in EXAMPLE_D:
        p = min(pts, key=lambda r: abs(float(r["D"]) - D0))
        _, z, m, _ = load(p["path"])
        ex.append((float(p["D"]), coeffs(np.asarray(z[:M], float), M, ks)))
    zf, mf = load_o(os.path.join(a.raw, FRONT))
    co_front = coeffs(np.asarray(zf[0], float)[:M] + np.asarray(zf[1], float)[:M], M, ks)
    D_front = float(mf["D"])
    fits = {deg: HeldFit(os.path.join(HERE, f)) for deg, f in ((6, "data/held_fit_d6.json"), (8, "data/held_fit.json"),
                                                               (10, "data/held_fit_d10.json"))}
    Dgrid = np.linspace(float(fitd["Dmin"]), 240.0, 220)
    a5 = {deg: [f.values(float(D), [5])[0][0] for D in Dgrid] for deg, f in fits.items()}
    # data: the clean chain's Delta and one step's Taylor terms
    att = []
    for lg in CHAIN_LOGS:
        p = os.path.join(a.raw, lg)
        if os.path.exists(p):
            att += parse_log(open(p, errors="replace").read(), "win2", os.path.basename(lg))
    chain = [(float(r["D_to"]), float(r["Delta"]) + float(r["Delta_lo"] or 0)) for r in rows_of(att) if r["accepted"]]
    rec = next(json.loads(l) for l in open(os.path.join(a.raw, JSONL)) if l.strip() and abs(json.loads(l)["D0"] - BASE_D) < 1e-9)
    acc = next(x for x in rec["attempts"] if x["accepted"])
    step = float(acc["a"])
    terms = [rec["norms"][k] * step ** k for k in range(1, len(rec["norms"]))]
    print(f"full states {len(pts)}; examples at D {[round(d, 2) for d, _ in ex]}; front D {D_front:g}, a5 {co_front[5][0]:.6e}")
    print(f"chain points {len(chain)}; base D {BASE_D}, step a {step}; |z_k| a^k: " + " ".join(f"{t:.1e}" for t in terms))

    W, H = 1040, 640
    svg = Svg(W)
    svg.text(16, 26, "How the held harmonics are fitted, and how the chain steps in D", size=15, fill=INK, weight="bold")
    svg.text(16, 44, "The animation loops through five steps. Without animation every step shows at once.", size=11, fill=INK_SOFT)
    LX, LY, LW, LH = 74, 96, 400, 250
    RX, RY, RW, RH = 590, 96, 400, 250
    th = np.linspace(0, 2 * np.pi, 361)
    ymax = max(float(np.max(np.abs(wave(c, th)))) for _, c in ex) * 1.15
    MAG = 40.0
    pi_fmt = {0.0: "0", np.pi / 2: "π/2", np.pi: "π", 3 * np.pi / 2: "3π/2", 2 * np.pi: "2π"}

    import data_graphs as dg
    base_axis = dg.Axis

    class Ax(base_axis):
        def ticks(self):
            if not self.log and self.lo == 0 and abs(self.hi - 2 * np.pi) < 1e-12:
                return list(pi_fmt)
            if self.log:
                return [10.0 ** e for e in range(math.ceil(math.log10(self.lo)), math.floor(math.log10(self.hi)) + 1) if e % 6 == 0]
            return super().ticks()
    dg.Axis = Ax

    # steps 1 to 4: the wave panel and the a5 panel
    lo5 = min(min(a5[8]), min(c[5][0] for _, c in ex)) * 0.85
    hi5 = max(r["c"]["5a"] for r in pts) * 1.08
    holder = {}

    def left_axes():
        holder["L"] = axes(svg, LX, LY, LW, LH, (0, 2 * np.pi), (-ymax, ymax), "one echo period (θ)", "p1 at the centre",
                           xfmt=lambda v: pi_fmt.get(v, f"{v:.2f}"), yfmt=lambda v: f"{v:.1f}")

    def right_axes():
        holder["R"] = axes(svg, RX, RY, RW, RH, (30, 240), (lo5, hi5), "spacetime dimension D", "a5, harmonic 5's cosine",
                           xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:.4f}")
    clock.group(svg, pa.capture(svg, left_axes) + pa.capture(svg, right_axes), 0.0, STEPS[3][1])
    xL, yL = holder["L"]
    xR, yR = holder["R"]
    cols = SERIES[:6]
    t_wave = np.linspace(STEPS[0][0] + 0.3, STEPS[0][1] - 1.2, len(ex))
    t_read = np.linspace(STEPS[1][0] + 0.2, STEPS[1][1] - 2.2, len(ex))
    for i, (D, c) in enumerate(ex):
        line = [(xL(t), yL(v)) for t, v in zip(th, wave(c, th))]
        body = [f'<path d="{pa.path_d(line)}" fill="none" stroke="{cols[i]}" stroke-width="1.8"/>',
                f'<text x="{LX + LW + 8}" y="{LY + 14 + 14 * i}" font-size="11" fill="{cols[i]}">D {D:.1f}</text>']
        clock.group(svg, body, float(t_wave[i]), STEPS[1][1], dim=0.25, t_dim=float(t_read[i]))
        h5 = [(xL(t), yL(MAG * v)) for t, v in zip(th, wave({5: c[5]}, th))]
        clock.group(svg, [f'<path d="{pa.path_d(h5)}" fill="none" stroke="{cols[i]}" stroke-width="2.4" stroke-dasharray="5 3"/>',
                          f'<text x="{LX + 6}" y="{LY + LH - 8}" font-size="11" fill="{cols[i]}">harmonic 5 at D {D:.1f}, ×{MAG:g}</text>'],
                    float(t_read[i]), float(t_read[i] + 1.4))
        dot = pa.capture(svg, lambda: marker(svg, "circle", xR(D), yR(c[5][0]), cols[i], f"D {D:.1f}: a5 {c[5][0]:.6e}"))
        clock.group(svg, dot, float(t_read[i] + 0.9), STEPS[3][1])
    alldots = pa.capture(svg, lambda: [marker(svg, "circle", xR(float(r["D"])), yR(r["c"]["5a"]), SERIES[2],
                                              f"D {float(r['D']):.2f}: a5 {r['c']['5a']:.6e}") for r in pts])
    clock.group(svg, alldots, STEPS[1][1] - 1.6, STEPS[3][1], dim=0.55, t_dim=STEPS[2][0] + 3)
    inside = [(xR(D), yR(v)) for D, v in zip(Dgrid, a5[8]) if D <= 73.0]
    outside = [(xR(D), yR(v)) for D, v in zip(Dgrid, a5[8]) if D >= 73.0]
    clock.group(svg, draw_in(clock, inside, SERIES[1], 2.6, STEPS[2][0] + 0.4, STEPS[2][0] + 3.0), STEPS[2][0], STEPS[3][1])
    clock.group(svg, draw_in(clock, outside, SERIES[1], 2.6, STEPS[3][0] + 0.3, STEPS[3][0] + 3.0), STEPS[3][0], STEPS[3][1])
    for deg, col in ((6, SERIES[3]), (10, SERIES[4])):
        line = [(xR(D), yR(v)) for D, v in zip(Dgrid, a5[deg]) if D >= 73.0]
        clock.group(svg, [f'<path d="{pa.path_d(line)}" fill="none" stroke="{col}" stroke-width="1.2" stroke-dasharray="3 3"/>'],
                    STEPS[3][0] + 3.2, STEPS[3][1])
    clock.group(svg, [f'<text x="{RX + RW - 4}" y="{RY + 16}" text-anchor="end" font-size="11" fill="{SERIES[1]}">fit of degree 8</text>',
                      f'<text x="{RX + RW - 4}" y="{RY + 30}" text-anchor="end" font-size="11" fill="{INK_SOFT}">dashed: degrees 6 and 10, on top of it</text>'],
                STEPS[2][0] + 1.0, STEPS[3][1])
    fx, fy = xR(D_front), yR(co_front[5][0])
    front = pa.capture(svg, lambda: marker(svg, "diamond", fx, fy, SERIES[0], f"D {D_front:g}: a5 {co_front[5][0]:.6e}"))
    front.append(f'<text x="{fx - 8}" y="{fy - 14}" text-anchor="end" font-size="11" fill="{INK}">held value at D {D_front:.1f}</text>')
    clock.group(svg, front, STEPS[3][0] + 3.4, STEPS[3][1])
    wl = [(xL(t), yL(v)) for t, v in zip(th, wave(co_front, th))]
    h5f = [(xL(t), yL(MAG * v)) for t, v in zip(th, wave({5: co_front[5]}, th))]
    clock.group(svg, [f'<path d="{pa.path_d(wl)}" fill="none" stroke="{SERIES[0]}" stroke-width="2"/>',
                      f'<path d="{pa.path_d(h5f)}" fill="none" stroke="{SERIES[0]}" stroke-width="2.4" stroke-dasharray="5 3"/>',
                      f'<text x="{LX + LW - 4}" y="{LY + 14}" text-anchor="end" font-size="11" fill="{SERIES[0]}">D {D_front:.1f}: wave, and harmonic 5 ×{MAG:g}</text>'],
                STEPS[3][0] + 4.4, STEPS[3][1])

    # step 5: the calculus of one chain step
    cD = [d for d, _ in chain]
    cV = [v for _, v in chain]
    i0 = min(range(len(cD)), key=lambda i: abs(cD[i] - BASE_D))
    slope = (cV[i0 + 1] - cV[i0 - 1]) / (cD[i0 + 1] - cD[i0 - 1])

    def left5():
        holder["L5"] = axes(svg, LX, LY, LW, LH, (214, 236), (min(cV) - 0.0003, max(cV) + 0.0003), "spacetime dimension D",
                            "Δ, the echo period", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:.4f}")

    def right5():
        holder["R5"] = axes(svg, RX, RY, RW, RH, (0, 25), (1e-32, 1e-2, "log"), "order k of the Taylor term",
                            "size of the term |z_k| aᵏ", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"1e{round(math.log10(v))}")
    clock.group(svg, pa.capture(svg, left5) + pa.capture(svg, right5), STEPS[4][0], T - 0.7)
    x5, y5 = holder["L5"]
    xb, yb = holder["R5"]
    cdots = pa.capture(svg, lambda: [marker(svg, "circle", x5(d), y5(v), SERIES[0], f"D {d:g}: Δ {v:.12f}") for d, v in chain])
    clock.group(svg, cdots, STEPS[4][0] + 0.2, T - 0.7)
    bx, by = x5(cD[i0]), y5(cV[i0])
    base = pa.capture(svg, lambda: marker(svg, "diamond", bx, by, SERIES[1], f"base D {cD[i0]:g}"))
    base.append(f'<text x="{bx - 10}" y="{by + 4}" text-anchor="end" font-size="11" fill="{INK}">solved at D {cD[i0]:.2f}</text>')
    clock.group(svg, base, STEPS[4][0] + 1.0, T - 0.7)
    tang = [(x5(cD[i0] - 3), y5(cV[i0] - 3 * slope)), (x5(cD[i0] + step + 3), y5(cV[i0] + (step + 3) * slope))]
    clock.group(svg, draw_in(clock, tang, SERIES[1], 1.4, STEPS[4][0] + 1.6, STEPS[4][0] + 3.0, dash="4 3"), STEPS[4][0] + 1.6, T - 0.7)
    clock.group(svg, [f'<text x="{LX + LW - 6}" y="{LY + 18}" text-anchor="end" font-size="11" fill="{SERIES[1]}">dashed: the first derivative, the tangent</text>'],
                STEPS[4][0] + 2.6, T - 0.7)
    t_bar = np.linspace(STEPS[4][0] + 3.2, STEPS[4][1] - 4.0, len(terms))
    bw = 0.62 * (xb(1) - xb(0))
    for k, (tk, v) in enumerate(zip(t_bar, terms), start=1):
        top = yb(max(v, 1e-32))
        clock.group(svg, [f'<rect x="{xb(k) - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{yb(1e-32) - top:.1f}" '
                          f'fill="{SERIES[1] if k == 1 else SERIES[3]}"><title>order {k}: {v:.2e}</title></rect>'], float(tk), T - 0.7)
    nx, ny = x5(cD[i0 + 1]), y5(cV[i0 + 1])
    pred = pa.capture(svg, lambda: marker(svg, "circle", nx, ny, SERIES[4], f"prediction and solution at D {cD[i0 + 1]:g}"))
    pred.append(f'<text x="{LX + LW - 6}" y="{LY + 34}" text-anchor="end" font-size="11" fill="{SERIES[4]}">24 terms summed: the prediction at D {cD[i0 + 1]:.2f}</text>')
    pred.append(f'<text x="{LX + LW - 6}" y="{LY + 50}" text-anchor="end" font-size="11" fill="{INK_SOFT}">Newton corrects it by about 10⁻³¹ in 3 steps</text>')
    clock.group(svg, pred, STEPS[4][1] - 3.6, T - 0.7)
    clock.group(svg, [f'<text x="{RX + RW - 4}" y="{RY + 16}" text-anchor="end" font-size="11" fill="{INK}">each order 10 to 300 times smaller</text>',
                      f'<text x="{xb(1) + 10:.1f}" y="{yb(terms[0]) + 12:.1f}" font-size="11" fill="{SERIES[1]}">order 1: the tangent</text>'],
                STEPS[4][0] + 4.0, T - 0.7)

    # the step titles under the panels
    for (t0, t1), (title, text) in zip(STEPS, TITLES):
        clock.group(svg, [f'<text x="16" y="{LY + LH + 66}" font-size="14" font-weight="bold" fill="{INK}">{title}</text>',
                          f'<text x="16" y="{LY + LH + 88}" font-size="12" fill="{INK}">{text}</text>'], t0, t1 - 0.3)
    # progress dots for the five steps
    for i, (t0, t1) in enumerate(STEPS):
        cx = 16 + 22 * i
        svg.add(f'<circle cx="{cx + 6}" cy="{LY + LH + 112}" r="5" fill="none" stroke="{INK_SOFT}"/>')
        clock.group(svg, [f'<circle cx="{cx + 6}" cy="{LY + LH + 112}" r="5" fill="{SERIES[1]}"/>'], t0, t1 - 0.3)
    hgt = footnote(svg, ["138 full solutions (held_fit.py)", "fits: data/held_fit*.json",
                         f"chain: raw I461-win2 crawlG, base D {BASE_D:g}"], "", [], LY + LH + 140, REPO)
    out = os.path.join(a.o, "fit_steps.svg")
    open(out, "w").write(svg.finish(hgt))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
