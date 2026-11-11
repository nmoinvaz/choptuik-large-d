#!/usr/bin/env python3
"""How the held harmonics' fit is made, and how the chain then steps in D, animated in one SVG.

    cd hp && python3 plot_fit_steps.py RAWDIR [-o .] [--frame T]

fit_steps.svg plays five steps on a loop (SMIL, as plot_algorithm.py's figures):
  1. the full-system solutions below D 73: p1 over one echo period at six of them;
  2. harmonic 5 measured in each wave: the wave times cos 5 theta, averaged over the period, is a5 / 2, and a5 becomes a
     point against D;
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
T = 54.0
STEPS = [(0.0, 7.0), (7.0, 26.0), (26.0, 32.0), (32.0, 41.0), (41.0, 54.0)]
TITLES = [
    ("1. Take the full solutions", "Below D 73 every harmonic can still be solved. Here is p1 at the centre over one echo period, at six of the 138."),
    ("2. Measure harmonic 5 in each", "Multiply the wave by cos 5θ and average over the period. Twice the average is a5, the amount of cos 5θ in the wave: one point against D."),
    ("3. Fit a polynomial through the points", "A least-squares polynomial of degree 8 in 1/(D−2), times 1/√(D−2), passes within 10⁻¹² of all 138 points."),
    ("4. Extend it to larger D", "Past D 73 the polynomial gives the held value. At D 232.6 it sets the wave's harmonic-5 part. Fits of degree 6 and 10 differ by 3×10⁻⁹ there."),
    ("5. Step in D with calculus", "Δ looks straight, but its slope changes. At a solved D the chain computes 24 derivatives of the whole solution; their Taylor sum predicts the next D."),
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

    W = 1040
    svg = Svg(W)
    svg.text(16, 26, "How the held harmonics are fitted, and how the chain steps in D", size=15, fill=INK, weight="bold")
    svg.text(16, 44, "The animation loops through five steps. Without animation every step shows at once.", size=11, fill=INK_SOFT)
    LX, LW = 74, 400
    AY, AH = 96, 130          # strip A: the wave p1
    BY, BH = 268, 64          # strip B: cos 5θ, or the harmonic-5 part
    CY, CH = 390, 74          # strip C: p1 · cos 5θ and its average
    LY, LH = AY, CY + CH - AY  # step 5's left panel, the whole column
    RX, RY, RW, RH = 590, 96, 400, CY + CH - 96
    th = np.linspace(0, 2 * np.pi, 361)
    thm = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    ymax = max(float(np.max(np.abs(wave(c, th)))) for _, c in ex) * 1.15
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

    def grp(body, pts_):
        """a group with an explicit opacity schedule [(t, opacity)]"""
        body = "".join(body) if isinstance(body, list) else body
        pts_ = sorted(pts_)
        if clock.frame is None:
            svg.add(f"<g>{body}{clock.sched(pts_)}</g>")
        elif clock.at(pts_, clock.frame) > 0:
            svg.add(f'<g opacity="{clock.at(pts_, clock.frame):.2f}">{body}</g>')

    def strip_axes(key, y, h, lim, ylab, yfmt):
        def f():
            holder[key] = axes(svg, LX, y, LW, h, (0, 2 * np.pi), (-lim, lim), "θ, one echo period", ylab,
                               xfmt=lambda v: pi_fmt.get(v, f"{v:.2f}"), yfmt=yfmt)
        return pa.capture(svg, f)

    # steps 1 to 4: the strips on the left, a5 against D on the right
    S1, S2, S3, S4, S5 = STEPS
    lo5 = min(min(a5[8]), min(c[5][0] for _, c in ex)) * 0.85
    hi5 = max(r["c"]["5a"] for r in pts) * 1.08
    holder = {}
    clock.group(svg, strip_axes("A", AY, AH, ymax, "p1, the wave", lambda v: f"{v:.1f}"), 0.0, S4[1])

    def right_axes():
        holder["R"] = axes(svg, RX, RY, RW, RH, (30, 240), (lo5, hi5), "spacetime dimension D", "a5: the amount of cos 5θ in p1",
                           xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:.4f}")
    clock.group(svg, pa.capture(svg, right_axes), 0.0, S4[1])
    xA, yA = holder["A"]
    xR, yR = holder["R"]
    cols = SERIES[:6]
    clock.group(svg, strip_axes("B", BY, BH, 1.15, "cos 5θ", lambda v: f"{v:g}"), S2[0] + 0.2, S2[1])
    xB, yB = holder["B"]
    cos5 = [(xB(t), yB(math.cos(5 * t))) for t in th]
    clock.group(svg, [f'<path d="{pa.path_d(cos5)}" fill="none" stroke="{INK_SOFT}" stroke-width="1.6"/>',
                      f'<text x="{LX + LW + 8}" y="{BY + BH / 2 - 2}" font-size="11" fill="{INK_SOFT}">the same</text>',
                      f'<text x="{LX + LW + 8}" y="{BY + BH / 2 + 12}" font-size="11" fill="{INK_SOFT}">at every D</text>'],
                S2[0] + 0.4, S2[1])
    plim = max(float(np.max(np.abs(wave(c, th) * np.cos(5 * th)))) for _, c in ex) * 1.15
    clock.group(svg, strip_axes("C", CY, CH, plim, "p1 · cos 5θ", lambda v: f"{v:.2f}"), S2[0] + 0.4, S2[1])
    xC, yC = holder["C"]
    slots = [(S2[0] + 0.8, S2[0] + 6.8)] + [(S2[0] + 6.8 + 2.2 * j, S2[0] + 9.0 + 2.2 * j) for j in range(len(ex) - 1)]
    t_wave = np.linspace(S1[0] + 0.3, S1[1] - 1.2, len(ex))
    for i, ((D, c), (s0, s1)) in enumerate(zip(ex, slots)):
        line = [(xA(t), yA(v)) for t, v in zip(th, wave(c, th))]
        grp([f'<path d="{pa.path_d(line)}" fill="none" stroke="{cols[i]}" stroke-width="1.8"/>',
             f'<text x="{LX + LW + 8}" y="{AY + 14 + 14 * i}" font-size="11" fill="{cols[i]}">D {D:.1f}</text>'],
            [(0, 0), (t_wave[i], 0), (t_wave[i] + 0.3, 1), (S2[0], 1), (S2[0] + 0.3, 0.2), (s0, 0.2), (s0 + 0.3, 1),
             (s1 - 0.3, 1), (s1, 0.2), (S2[1] - 0.3, 0.2), (S2[1], 0), (T, 0)])
        prod = wave(c, th) * np.cos(5 * th)
        mean = float(np.mean(wave(c, thm) * np.cos(5 * thm)))
        pl = [(xC(t), yC(v)) for t, v in zip(th, prod)]
        area = pa.path_d([(xC(0), yC(0))] + pl + [(xC(2 * np.pi), yC(0))]) + " Z"
        # each D's product stays, faded, once the next one is drawn; only its shading and its numbers are cleared
        grp([f'<path d="{area}" fill="{cols[i]}" fill-opacity="0.18" stroke="none"/>'],
            [(0, 0), (s0 + 0.6, 0), (s0 + 0.9, 1), (s1 - 0.3, 1), (s1, 0), (T, 0)])
        grp([f'<path d="{pa.path_d(pl)}" fill="none" stroke="{cols[i]}" stroke-width="1.6"/>'],
            [(0, 0), (s0 + 0.6, 0), (s0 + 0.9, 1), (s1 - 0.3, 1), (s1, 0.35), (S2[1] - 0.3, 0.35), (S2[1], 0), (T, 0)])
        y_mean = yC(mean)
        t_mean = s0 + (2.6 if i == 0 else 0.9)
        grp([f'<line x1="{LX}" y1="{y_mean:.1f}" x2="{LX + LW}" y2="{y_mean:.1f}" stroke="{cols[i]}" stroke-width="1.6" stroke-dasharray="6 3"/>'],
            [(0, 0), (t_mean, 0), (t_mean + 0.3, 1), (s1 - 0.3, 1), (s1, 0.35), (S2[1] - 0.3, 0.35), (S2[1], 0), (T, 0)])
        grp([f'<text x="{LX + LW}" y="{CY - 6}" text-anchor="end" font-size="11" fill="{INK}">D {D:.1f}: dashed average {mean:.5f}; a5 = 2 × average = {2 * mean:.5f}</text>'],
            [(0, 0), (t_mean, 0), (t_mean + 0.3, 1), (s1 - 0.3, 1), (s1, 0), (T, 0)])
        dot = pa.capture(svg, lambda: marker(svg, "circle", xR(D), yR(c[5][0]), cols[i], f"D {D:.1f}: a5 {c[5][0]:.6e}"))
        grp(dot, [(0, 0), (s1 - 0.6, 0), (s1 - 0.3, 1), (S4[1], 1), (S4[1] + 0.3, 0), (T, 0)])
        print(f"D {D:.2f}: a5 from the state {c[5][0]:.6e}, 2 x average of p1 cos 5θ {2 * mean:.6e}")
    alldots = pa.capture(svg, lambda: [marker(svg, "circle", xR(float(r["D"])), yR(r["c"]["5a"]), SERIES[2],
                                              f"D {float(r['D']):.2f}: a5 {r['c']['5a']:.6e}") for r in pts])
    clock.group(svg, alldots, S2[1] - 1.6, S4[1], dim=0.55, t_dim=S3[0] + 3)
    inside = [(xR(D), yR(v)) for D, v in zip(Dgrid, a5[8]) if D <= 73.0]
    outside = [(xR(D), yR(v)) for D, v in zip(Dgrid, a5[8]) if D >= 73.0]
    clock.group(svg, draw_in(clock, inside, SERIES[1], 2.6, S3[0] + 0.4, S3[0] + 3.0), S3[0], S4[1])
    clock.group(svg, draw_in(clock, outside, SERIES[1], 2.6, S4[0] + 0.3, S4[0] + 3.0), S4[0], S4[1])
    for deg, col in ((6, SERIES[3]), (10, SERIES[4])):
        line = [(xR(D), yR(v)) for D, v in zip(Dgrid, a5[deg]) if D >= 73.0]
        clock.group(svg, [f'<path d="{pa.path_d(line)}" fill="none" stroke="{col}" stroke-width="1.2" stroke-dasharray="3 3"/>'],
                    S4[0] + 3.2, S4[1])
    clock.group(svg, [f'<text x="{RX + RW - 4}" y="{RY + 16}" text-anchor="end" font-size="11" fill="{SERIES[1]}">fit of degree 8</text>',
                      f'<text x="{RX + RW - 4}" y="{RY + 30}" text-anchor="end" font-size="11" fill="{INK_SOFT}">dashed: degrees 6 and 10, on top of it</text>'],
                S3[0] + 1.0, S4[1])
    fx, fy = xR(D_front), yR(co_front[5][0])
    front = pa.capture(svg, lambda: marker(svg, "diamond", fx, fy, SERIES[0], f"D {D_front:g}: a5 {co_front[5][0]:.6e}"))
    front.append(f'<text x="{fx - 8}" y="{fy - 14}" text-anchor="end" font-size="11" fill="{INK}">held value at D {D_front:.1f}</text>')
    clock.group(svg, front, S4[0] + 3.4, S4[1])
    wl = [(xA(t), yA(v)) for t, v in zip(th, wave(co_front, th))]
    clock.group(svg, [f'<path d="{pa.path_d(wl)}" fill="none" stroke="{SERIES[0]}" stroke-width="2"/>',
                      f'<text x="{LX + LW - 4}" y="{AY + 14}" text-anchor="end" font-size="11" fill="{SERIES[0]}">the wave at D {D_front:.1f}</text>'],
                S4[0] + 4.4, S4[1])
    h5 = wave({5: co_front[5]}, th)
    hlim = float(np.max(np.abs(h5))) * 1.25
    clock.group(svg, strip_axes("B4", BY, BH, hlim, "its harmonic 5", lambda v: f"{v:.0e}" if v else "0"), S4[0] + 4.4, S4[1])
    xB4, yB4 = holder["B4"]
    h5l = [(xB4(t), yB4(v)) for t, v in zip(th, h5)]
    clock.group(svg, [f'<path d="{pa.path_d(h5l)}" fill="none" stroke="{SERIES[0]}" stroke-width="2"/>',
                      f'<text x="{LX + LW - 4}" y="{BY + 12}" text-anchor="end" font-size="11" fill="{INK}">a5 cos 5θ + b5 sin 5θ, set by the fit</text>'],
                S4[0] + 4.8, S4[1])

    # step 5: the calculus of one chain step
    cD = np.array([d for d, _ in chain])
    cV = np.array([v for _, v in chain])
    i0 = int(np.argmin(np.abs(cD - BASE_D)))
    # Delta along the chain as one polynomial (degree 6 through the chain's points): the slope dDelta/dD at each point
    poly = np.polynomial.Polynomial.fit(cD, cV, 6)
    dpoly = poly.deriv()
    slopes = dpoly(cD)
    slope = float(slopes[i0])
    print(f"base D {cD[i0]:g}: slope {slope:.6e}; slopes from {slopes[0]:.4e} at D {cD[0]:g} to {slopes[-1]:.4e} at D {cD[-1]:g}")
    TH_, BH_ = 150, 150
    BY_ = LY + LH - BH_

    def left5():
        holder["L5"] = axes(svg, LX, LY, LW, TH_, (214, 236), (cV.min() - 0.0003, cV.max() + 0.0003), "spacetime dimension D",
                            "Δ, the echo period", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:.4f}")

    def left5b():
        pad = (slopes.max() - slopes.min()) * 0.15
        holder["L5b"] = axes(svg, LX, BY_, LW, BH_, (214, 236), (slopes.min() - pad, slopes.max() + pad), "spacetime dimension D",
                             "slope dΔ/dD (×10⁻⁴)", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v * 1e4:.1f}")

    def right5():
        holder["R5"] = axes(svg, RX, RY, RW, RH, (0, 25), (1e-32, 1e-2, "log"), "order k of the Taylor term",
                            "size of the term |z_k| aᵏ", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"1e{round(math.log10(v))}")
    clock.group(svg, pa.capture(svg, left5) + pa.capture(svg, right5), S5[0], T - 0.7)
    x5, y5 = holder["L5"]
    xb, yb = holder["R5"]
    cdots = pa.capture(svg, lambda: [marker(svg, "circle", x5(d), y5(v), SERIES[0], f"D {d:g}: Δ {v:.12f}") for d, v in chain])
    clock.group(svg, cdots, S5[0] + 0.2, T - 0.7)
    bx, by = x5(cD[i0]), y5(cV[i0])
    base = pa.capture(svg, lambda: marker(svg, "diamond", bx, by, SERIES[1], f"base D {cD[i0]:g}"))
    base.append(f'<text x="{bx - 10}" y="{by + 4}" text-anchor="end" font-size="11" fill="{INK}">solved at D {cD[i0]:.2f}</text>')
    clock.group(svg, base, S5[0] + 1.0, T - 0.7)
    # the slope at every point: a short tangent through each, drawn along the chain
    t_tan = np.linspace(S5[0] + 1.4, S5[0] + 3.4, len(cD))
    for d, v, sl, tk in zip(cD, cV, slopes, t_tan):
        seg = f'<line x1="{x5(d - 0.7):.1f}" y1="{y5(v - 0.7 * sl):.1f}" x2="{x5(d + 0.7):.1f}" y2="{y5(v + 0.7 * sl):.1f}" stroke="{SERIES[1]}" stroke-width="2"/>'
        clock.group(svg, [seg], float(tk), T - 0.7)
    clock.group(svg, [f'<text x="{LX + LW - 6}" y="{LY + 18}" text-anchor="end" font-size="11" fill="{SERIES[1]}">at each point its tangent: the first derivative there</text>',
                      f'<text x="{LX + LW - 6}" y="{LY + 34}" text-anchor="end" font-size="11" fill="{INK_SOFT}">at this scale Δ looks straight</text>'],
                S5[0] + 2.0, T - 0.7)
    # the bend the tangent misses
    clock.group(svg, pa.capture(svg, left5b), S5[0] + 3.6, T - 0.7)
    xr, yr = holder["L5b"]
    rdots = pa.capture(svg, lambda: [marker(svg, "circle", xr(d), yr(v), SERIES[1], f"D {d:g}: slope {v:.5e}")
                                     for d, v in zip(cD, slopes)])
    rline = [(xr(d), yr(float(dpoly(d)))) for d in np.linspace(cD[0], cD[-1], 80)]
    rdots.append(f'<path d="{pa.path_d(rline)}" fill="none" stroke="{SERIES[1]}" stroke-width="1.2"/>')
    rdots.append(f'<text x="{LX + LW - 6}" y="{BY_ + 14}" text-anchor="end" font-size="11" fill="{INK}">the slope is not constant: it changes along the line,</text>')
    rdots.append(f'<text x="{LX + LW - 6}" y="{BY_ + 28}" text-anchor="end" font-size="11" fill="{INK}">and its change is the second derivative</text>')
    clock.group(svg, rdots, S5[0] + 4.0, T - 0.7)
    t_bar = np.linspace(S5[0] + 4.6, S5[1] - 3.6, len(terms))
    bw = 0.62 * (xb(1) - xb(0))
    for k, (tk, v) in enumerate(zip(t_bar, terms), start=1):
        top = yb(max(v, 1e-32))
        clock.group(svg, [f'<rect x="{xb(k) - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{yb(1e-32) - top:.1f}" '
                          f'fill="{SERIES[1] if k == 1 else SERIES[3]}"><title>order {k}: {v:.2e}</title></rect>'], float(tk), T - 0.7)
    nx, ny = x5(cD[i0 + 1]), y5(cV[i0 + 1])
    pred = pa.capture(svg, lambda: marker(svg, "circle", nx, ny, SERIES[4], f"prediction and solution at D {cD[i0 + 1]:g}"))
    pred.append(f'<text x="{LX + LW - 6}" y="{LY + 50}" text-anchor="end" font-size="11" fill="{SERIES[4]}">24 terms summed: the prediction at D {cD[i0 + 1]:.2f}</text>')
    pred.append(f'<text x="{LX + LW - 6}" y="{LY + 66}" text-anchor="end" font-size="11" fill="{INK_SOFT}">Newton corrects it by about 10⁻³¹ in 3 steps</text>')
    clock.group(svg, pred, S5[1] - 3.4, T - 0.7)
    clock.group(svg, [f'<text x="{RX + RW - 4}" y="{RY + 16}" text-anchor="end" font-size="11" fill="{INK}">terms of the whole solution, all 769 numbers</text>',
                      f'<text x="{RX + RW - 4}" y="{RY + 30}" text-anchor="end" font-size="11" fill="{INK_SOFT}">each order 10 to 300 times smaller</text>',
                      f'<text x="{xb(1) + 10:.1f}" y="{yb(terms[0]) + 12:.1f}" font-size="11" fill="{SERIES[1]}">order 1: the tangent</text>',
                      f'<text x="{xb(2) + 10:.1f}" y="{yb(terms[1]) + 12:.1f}" font-size="11" fill="{SERIES[3]}">order 2 and up: the bend</text>'],
                S5[0] + 5.0, T - 0.7)

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
