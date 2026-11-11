#!/usr/bin/env python3
"""The critical solutions of all D form one smooth family of waves, animated in one SVG.

    cd hp && python3 plot_wave_family.py RAWDIR [-o .] [--frame T]

wave_family.svg plays four steps on a loop (SMIL, as plot_algorithm.py's figures):
  1. one wave per D: p1 at the centre over one echo period, swept from D 38 to 232.6 through the stored states (the
     full-system states of held_fit.py to D 73, the chain_fit chain of T138-win2 to D 215, the clean chain
     I461-win2/crawlG to D 232.6), beside the size of its harmonic 1 against D;
  2. the same waves times sqrt(D-2): they nearly coincide, so across D the solution is close to one shape shrunk by
     1/sqrt(D-2);
  3. within one wave each odd harmonic is a nearly fixed factor smaller than the one before: the wave at D 38 built up
     harmonic by harmonic, and |harmonic k| against k at seven D on a log scale;
  4. each wave is the slope at the centre of a surface: p over the radius x and the echo phase theta at D 38 and
     232.6, drawn from the solver's five nodes x = 0.3 to 0.7, where p is close to p1 x.
Harmonics 15 and up are zero in all these runs (kfreeze 13 for the full states, 3 for the chains), and the chains hold
harmonics 5 to 13 on the fit of data/held_fit.json. --frame T writes the still at time T. Standard library and numpy;
the primitives are data_graphs.py and the clock of plot_algorithm.py. Prints the plotted numbers.
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_graphs as dg  # noqa: E402
import plot_algorithm as pa  # noqa: E402
from data_graphs import INK, INK_SOFT, SERIES, Svg, axes, footnote, marker  # noqa: E402
from held_fit import full_states, load  # noqa: E402
from hpkernel.driver import load_q  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
CHAINS = ["T138-win2/chain", "T138-win2/chain_o", "I461-win2/crawlG"]
M = 32
KS = list(range(1, 15, 2))       # harmonics 1 to 13; 15 and up are zero in every run here
REP_D = [38.02, 52.04, 73.0, 100.0, 150.0, 200.0, 232.63]
T = 48.0
STEPS = [(0.0, 13.0), (13.0, 23.0), (23.0, 36.0), (36.0, 48.0)]
TITLES = [
    ("1. One wave for each D", "Each D has its own critical solution. Over one echo period its centre field p1 is a wave: here it is swept "
     "from D 38 to 232.6 through the stored solutions."),
    ("2. Multiply each wave by √(D−2)", "Scaled by √(D−2) the waves nearly coincide. Across D the solution is close to one shape, "
     "shrunk by 1/√(D−2)."),
    ("3. Inside one wave, each harmonic is a fixed factor smaller", "Built harmonic by harmonic, the wave at D 38 converges by a "
     "steady factor per term. On a log scale the harmonics fall on near-straight lines, steeper at higher D."),
    ("4. The wave is the slope of a surface at the centre", "A solution is a field p over the radius x and the echo phase θ. "
     "Near the centre p ≈ p1 x, so p1 is the slope of this surface at x = 0."),
]
PI_FMT = {0.0: "0", np.pi / 2: "π/2", np.pi: "π", 3 * np.pi / 2: "3π/2", 2 * np.pi: "2π"}


class Ax(dg.Axis):
    """ticks at multiples of pi/2 on a phase axis, and every decade on a log axis"""

    def ticks(self):
        if not self.log and self.lo == 0 and abs(self.hi - 2 * np.pi) < 1e-12:
            return list(PI_FMT)
        if self.log:
            return [10.0 ** e for e in range(math.ceil(math.log10(self.lo)), math.floor(math.log10(self.hi)) + 1)]
        return super().ticks()


dg.Axis = Ax


def coeffs(f):
    """cos and sin coefficients of an antiperiodic field from its samples on the half grid theta_j = pi j / M"""
    th = np.pi * np.arange(M) / M
    return {k: ((2 / M) * np.sum(f * np.cos(k * th)), (2 / M) * np.sum(f * np.sin(k * th))) for k in KS}


def wave(co, th, kmax=None):
    return sum(a * np.cos(k * th) + b * np.sin(k * th) for k, (a, b) in co.items() if kmax is None or k <= kmax)


def size(co, k):
    return math.hypot(*co[k])


def catalogue(raw):
    """every stored solution from D 38 on as (D, kind, path): the full states to D 73, then the chains"""
    out = [(float(p["D"]), "full", p["path"]) for p in full_states(raw) if float(p["D"]) >= 38.0]
    top = max(d for d, _, _ in out)
    for sub in CHAINS:
        for f in sorted(os.listdir(os.path.join(raw, sub))):
            if f.startswith("anm_pin3_D") and f.endswith(".npz"):
                D = float(f[len("anm_pin3_D"):-4])
                if D > top + 0.5 and all(abs(D - d) > 0.05 for d, _, _ in out):
                    out.append((D, "chain", os.path.join(raw, sub, f)))
    return sorted(out)


def state(D, kind, path):
    """D, Delta, the p1 harmonics and p's harmonics at each node"""
    if kind == "full":
        D, z, m, delta = load(path)
        f, delta = np.asarray(z, float), float(delta)
    else:
        z, m = load_q(path)
        f = np.asarray(z[0], float) + np.asarray(z[1], float)
        delta, D = float(z[0][-1]) + float(z[1][-1]), float(m["D"])
    nodes = [(float(x), coeffs(f[2 * M + 4 * M * i: 3 * M + 4 * M * i])) for i, x in enumerate(m["nodes"])]
    return {"D": float(D), "delta": delta, "co": coeffs(f[:M]), "nodes": nodes, "kind": kind}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("-o", default=HERE)
    ap.add_argument("--frame", type=float, default=None)
    a = ap.parse_args()
    pa.FRAME = a.frame
    clock = pa.Clock(T)
    S1, S2, S3, S4 = STEPS

    cat = catalogue(a.raw)
    allst = [state(*c) for c in cat]
    Ds = np.array([s["D"] for s in allst])
    print(f"{len(allst)} solutions, D {Ds.min():g} to {Ds.max():g} ({sum(s['kind'] == 'full' for s in allst)} full)")

    def nearest(D0):
        return allst[int(np.argmin(np.abs(Ds - D0)))]

    reps = [nearest(D) for D in REP_D]
    c1_0 = size(allst[0]["co"], 1)
    sc1_0 = math.sqrt(allst[0]["D"] - 2) * c1_0
    print("| D | Delta | sqrt(D-2) |c1| | |c1|/|c3| | |c3|/|c5| | |c5|/|c7| | kind |\n|---|---|---|---|---|---|---|")
    for s in reps:
        co = s["co"]
        print(f"| {s['D']:.2f} | {s['delta']:.10f} | {math.sqrt(s['D'] - 2) * size(co, 1):.5f} | {size(co, 1) / size(co, 3):.2f} | "
              f"{size(co, 3) / size(co, 5):.2f} | {size(co, 5) / size(co, 7):.2f} | {s['kind']} |")

    W = 1040
    svg = Svg(W)
    svg.text(16, 26, "The solutions of all D form one smooth family of waves", size=15, fill=INK, weight="bold")
    svg.text(16, 44, "The animation loops through four steps. Without animation every step shows at once.", size=11, fill=INK_SOFT)
    LX, LY, LW, LH = 80, 96, 400, 300
    RX, RY, RW, RH = 610, 96, 380, 300
    th = np.linspace(0, 2 * np.pi, 161)
    holder = {}
    cols = SERIES[:len(reps)]

    def grp(body, sched):
        """a group with an explicit opacity schedule [(t, opacity)]"""
        body = "".join(body) if isinstance(body, list) else body
        sched = sorted(sched)
        if clock.frame is None:
            svg.add(f"<g>{body}{clock.sched(sched)}</g>")
        elif clock.at(sched, clock.frame) > 0:
            svg.add(f'<g opacity="{clock.at(sched, clock.frame):.2f}">{body}</g>')

    def frames_full(frames):
        frames = sorted(frames, key=lambda f: f[0])
        if frames[0][0] > 0:
            frames = [(0.0, frames[0][1])] + frames
        if frames[-1][0] < T:
            frames = frames + [(T, frames[-1][1])]
        return frames

    def lerp_at(frames, t):
        for (t0, v0), (t1, v1) in zip(frames, frames[1:]):
            if t0 <= t <= t1:
                w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return [tuple(p + (q - p) * w for p, q in zip(u, v)) for u, v in zip(v0, v1)]
        return frames[-1][1]

    def morph(frames, color, width, dash=None):
        """a path whose points move through frames [(t, points)], linearly in time and held outside them"""
        frames = frames_full(frames)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        if clock.frame is not None:
            return f'<path d="{pa.path_d(lerp_at(frames, clock.frame))}" fill="none" stroke="{color}" stroke-width="{width}"{da}/>'
        vals = ";".join(pa.path_d(p) for _, p in frames)
        kt = ";".join(f"{t / T:.4f}" for t, _ in frames)
        return (f'<path d="{pa.path_d(frames[0][1])}" fill="none" stroke="{color}" stroke-width="{width}"{da}>'
                f'<animate attributeName="d" values="{vals}" keyTimes="{kt}" dur="{T}s" repeatCount="indefinite"/></path>')

    def moving_dot(frames, color, r=5.5):
        frames = frames_full([(t, [(x, y)]) for t, x, y in frames])
        if clock.frame is not None:
            (x, y), = lerp_at(frames, clock.frame)
            return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" stroke="{dg.SURFACE}" stroke-width="2"/>'
        kt = ";".join(f"{t / T:.4f}" for t, _ in frames)
        cx = ";".join(f"{p[0][0]:.1f}" for _, p in frames)
        cy = ";".join(f"{p[0][1]:.1f}" for _, p in frames)
        return (f'<circle cx="{frames[0][1][0][0]:.1f}" cy="{frames[0][1][0][1]:.1f}" r="{r}" fill="{color}" stroke="{dg.SURFACE}" '
                f'stroke-width="2"><animate attributeName="cx" values="{cx}" keyTimes="{kt}" dur="{T}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cy}" keyTimes="{kt}" dur="{T}s" repeatCount="indefinite"/></circle>')

    def on(t0, t1, op=1.0):
        return [(0, 0), (t0, 0), (t0 + 0.3, op), (t1 - 0.3, op), (t1, 0), (T, 0)]

    # the left panel's two vertical scales: p1 as it is, and sqrt(D-2) p1
    ymax = max(float(np.max(np.abs(wave(s["co"], th)))) for s in allst) * 1.12
    ysc = max(float(np.max(np.abs(math.sqrt(s["D"] - 2) * wave(s["co"], th)))) for s in allst) * 1.12

    def left(key, lim, label):
        def f():
            holder[key] = axes(svg, LX, LY, LW, LH, (0, 2 * np.pi), (-lim, lim), "θ, the phase over one echo period", label,
                               xfmt=lambda v: PI_FMT.get(v, f"{v:.2f}"), yfmt=lambda v: f"{v:.1f}")
        return pa.capture(svg, f)
    t_swap = S2[0] + 1.0
    grp(left("raw", ymax, "p1, the centre field"), [(0, 0), (0.01, 1), (t_swap, 1), (t_swap + 0.3, 0), (S3[0], 0), (S3[0] + 0.3, 1),
                                                    (S3[1] - 0.3, 1), (S3[1], 0), (T, 0)])
    grp(left("sc", ysc, "√(D−2) · p1"), [(0, 0), (t_swap, 0), (t_swap + 0.3, 1), (S2[1] - 0.3, 1), (S2[1], 0), (T, 0)])
    xA, yA = holder["raw"]
    xS, yS = holder["sc"]

    def raw_pts(s, kmax=None):
        return [(xA(t), yA(v)) for t, v in zip(th, wave(s["co"], th, kmax))]

    def sc_pts(s):
        return [(xS(t), yS(v)) for t, v in zip(th, math.sqrt(s["D"] - 2) * wave(s["co"], th))]

    # step 1: the sweep through D, evenly in log(D - 2)
    t0, t1 = S1[0] + 0.6, S1[1] - 2.4
    u0, u1 = math.log(Ds.min() - 2), math.log(Ds.max() - 2)
    sweep = sorted({id(nearest(math.exp(u) + 2)): nearest(math.exp(u) + 2) for u in np.linspace(u0, u1, 34)}.values(),
                   key=lambda s: s["D"])

    def t_of(D):
        return t0 + (t1 - t0) * (math.log(D - 2) - u0) / (u1 - u0)
    grp([morph([(t_of(s["D"]), raw_pts(s)) for s in sweep], INK, 2.4)], on(S1[0] + 0.3, S1[1]))
    for s, nxt in zip(sweep, sweep[1:] + [None]):
        ta, tb = t_of(s["D"]), (t_of(nxt["D"]) if nxt else S1[1] - 0.3)
        grp([f'<text x="{LX + LW / 2}" y="{LY + 18}" text-anchor="middle" font-size="13" font-weight="bold" fill="{INK}">D {s["D"]:.1f}</text>',
             f'<text x="{LX + LW / 2}" y="{LY + 34}" text-anchor="middle" font-size="11" fill="{INK}">Δ = {s["delta"]:.4f}</text>'],
            [(0, 0), (ta, 0), (ta + 0.02, 1), (tb, 1), (tb + 0.02, 0), (T, 0)] if s is not sweep[0] else
            [(0, 1), (tb, 1), (tb + 0.02, 0), (T, 0)])
    # the seven kept waves, faded once the sweep has passed them, then scaled in step 2
    t_m0, t_m1 = S2[0] + 1.2, S2[0] + 4.2
    for i, s in enumerate(reps):
        tg = t_of(s["D"])
        grp([morph([(t_m0, raw_pts(s)), (t_m1, sc_pts(s))], cols[i], 1.7)],
            [(0, 0), (tg, 0), (tg + 0.3, 0.55), (S1[1] - 0.3, 0.55), (S1[1], 1), (S2[1] - 0.3, 1), (S2[1], 0), (T, 0)])
        grp([f'<text x="{LX + LW + 8}" y="{LY + 14 + 15 * i}" font-size="11" fill="{cols[i]}">D {s["D"]:.1f}</text>'],
            [(0, 0), (tg, 0), (tg + 0.3, 1), (S2[1] - 0.3, 1), (S2[1], 0), (T, 0)])
    spread = [math.sqrt(s["D"] - 2) * size(s["co"], 1) for s in allst]
    grp([f'<text x="{LX + LW / 2}" y="{LY + 18}" text-anchor="middle" font-size="11" fill="{INK}">times √(D−2), the seven waves nearly coincide:</text>',
         f'<text x="{LX + LW / 2}" y="{LY + 32}" text-anchor="middle" font-size="11" fill="{INK}">their harmonic 1 falls by {100 * (1 - spread[-1] / spread[0]):.0f}% '
         f'from D 38 to 232.6,</text>',
         f'<text x="{LX + LW / 2}" y="{LY + 46}" text-anchor="middle" font-size="11" fill="{INK}">against {100 * (1 - size(allst[-1]["co"], 1) / c1_0):.0f}% '
         f'unscaled</text>'], on(t_m1, S2[1]))

    # the right panel for steps 1 and 2: the size of harmonic 1 against D, as it is and times sqrt(D-2)
    def right12():
        holder["R"] = axes(svg, RX, RY, RW, RH, (30, 240), (0, 1.12), "spacetime dimension D", "size of harmonic 1, relative to D 38",
                           xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:.1f}")
    grp(pa.capture(svg, right12), on(S1[0] + 0.01, S2[1]))
    xR, yR = holder["R"]
    rel = [size(s["co"], 1) / c1_0 for s in allst]
    rel_sc = [math.sqrt(s["D"] - 2) * size(s["co"], 1) / sc1_0 for s in allst]
    raw_line = [(xR(d), yR(v)) for d, v in zip(Ds, rel)]
    grp([f'<path d="{pa.path_d(raw_line)}" fill="none" stroke="{SERIES[0]}" stroke-width="2"/>'],
        [(0, 0), (0.01, 0.25), (t1, 0.25), (t1 + 0.3, 1), (S2[1] - 0.3, 1), (S2[1], 0), (T, 0)])
    grp([moving_dot([(t_of(s["D"]), xR(s["D"]), yR(size(s["co"], 1) / c1_0)) for s in sweep], SERIES[0])], on(S1[0] + 0.3, S1[1]))
    grp([f'<text x="{xR(236):.1f}" y="{yR(rel[-1]) + 20:.1f}" text-anchor="end" font-size="11" fill="{SERIES[0]}">'
         f'|harmonic 1|: down to {rel[-1]:.2f}</text>'], on(t1, S2[1]))
    grp([f'<line x1="{RX}" y1="{yR(1):.1f}" x2="{RX + RW}" y2="{yR(1):.1f}" stroke="{INK_SOFT}" stroke-width="1" stroke-dasharray="4 3"/>'],
        on(S2[0] + 0.3, S2[1]))
    sc_line = [(xR(d), yR(v)) for d, v in zip(Ds, rel_sc)]
    grp([draw_in(clock, sc_line, SERIES[1], 2.4, S2[0] + 1.2, S2[0] + 4.2)], on(S2[0] + 1.0, S2[1]))
    grp([f'<text x="{xR(236):.1f}" y="{yR(rel_sc[-1]) + 18:.1f}" text-anchor="end" font-size="11" fill="{SERIES[1]}">'
         f'√(D−2) × |harmonic 1|: {rel_sc[-1]:.2f}</text>'], on(S2[0] + 4.2, S2[1]))
    grp([f'<text x="{RX + RW - 6}" y="{RY + RH - 40}" text-anchor="end" font-size="11" fill="{INK}">the wave shrinks by 1/√(D−2),</text>',
         f'<text x="{RX + RW - 6}" y="{RY + RH - 26}" text-anchor="end" font-size="11" fill="{INK}">and its shape changes only slowly</text>'],
        on(S2[0] + 4.6, S2[1]))

    # step 3, left: the wave at D 38 built up harmonic by harmonic
    s38 = reps[0]
    full38 = raw_pts(s38)
    grp([f'<path d="{pa.path_d(full38)}" fill="none" stroke="{INK_SOFT}" stroke-width="5" stroke-opacity="0.35"/>'], on(S3[0] + 0.3, S3[1]))
    stages = [1, 3, 5, 7, 13]
    t_st = [S3[0] + 0.6 + 2.0 * j for j in range(len(stages))]
    grp([morph([(t + 0.9, raw_pts(s38, k)) for t, k in zip(t_st, stages)], SERIES[0], 2.2)], on(S3[0] + 0.5, S3[1]))
    thm = np.linspace(0, 2 * np.pi, 4001)
    full = wave(s38["co"], thm)
    print(f"D {s38['D']:.2f}, the wave built up harmonic by harmonic (largest miss over one period):")
    for j, (t, k) in enumerate(zip(t_st, stages)):
        miss = float(np.max(np.abs(full - wave(s38["co"], thm, k))))
        name = "harmonic 1 alone" if k == 1 else (f"+ harmonic {k}" if k < 13 else "+ harmonics 9 to 13")
        row = f"{name}: misses by {miss:.1e}" if k < 13 else f"{name}: the whole wave"
        print(f"  up to harmonic {k}: miss {miss:.3e}")
        grp([f'<text x="{LX + LW / 2}" y="{LY + 40 + 15 * j}" text-anchor="middle" font-size="11" fill="{INK if j else SERIES[0]}">{row}</text>'],
            on(t + 0.6, S3[1]))
    grp([f'<text x="{LX + LW / 2}" y="{LY + 18}" text-anchor="middle" font-size="13" font-weight="bold" fill="{INK}">D {s38["D"]:.1f}</text>',
         f'<text x="{LX + 8}" y="{LY + LH - 26}" font-size="11" fill="{INK_SOFT}">grey: the whole wave</text>',
         f'<text x="{LX + 8}" y="{LY + LH - 12}" font-size="11" fill="{SERIES[0]}">blue: the sum so far</text>'],
        on(S3[0] + 0.3, S3[1]))

    # step 3, right: |harmonic k| against k at the seven D, on a log scale
    lo = min(size(s["co"], 13) for s in reps) / 3

    def right3():
        holder["R3"] = axes(svg, RX, RY, RW, RH, (0, 14), (lo, 1.0, "log"), "harmonic k (odd)", "size of harmonic k in p1",
                            xfmt=lambda v: f"{v:g}", yfmt=lambda v: pa.pow10(math.log10(v)))
    grp(pa.capture(svg, right3), on(S3[0] + 0.01, S3[1]))
    xK, yK = holder["R3"]
    t_ln = np.linspace(S3[0] + 1.0, S3[0] + 6.0, len(reps))
    for i, (s, tl) in enumerate(zip(reps, t_ln)):
        pts_ = [(xK(k), yK(size(s["co"], k))) for k in KS]
        body = [f'<path d="{pa.path_d(pts_)}" fill="none" stroke="{cols[i]}" stroke-width="1.6"/>']
        for k, (px, py) in zip(KS, pts_):
            held = s["kind"] == "chain" and k >= 5
            if held:
                body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.2" fill="{dg.SURFACE}" stroke="{cols[i]}" stroke-width="1.6">'
                            f'<title>D {s["D"]:.2f}, harmonic {k}: {size(s["co"], k):.3e} (held)</title></circle>')
            else:
                body += pa.capture(svg, lambda: marker(svg, "circle", px, py, cols[i], f"D {s['D']:.2f}, harmonic {k}: {size(s['co'], k):.3e}"))
        body.append(f'<text x="{RX + RW - 70}" y="{RY + 16 + 14 * i}" font-size="11" fill="{cols[i]}">D {s["D"]:.1f}</text>')
        grp(body, on(float(tl), S3[1]))
    r_lo, r_hi = size(reps[0]["co"], 1) / size(reps[0]["co"], 3), size(reps[-1]["co"], 1) / size(reps[-1]["co"], 3)
    grp([f'<text x="{xK(0.6):.1f}" y="{RY + RH - 54}" font-size="11" fill="{INK}">a near-straight line: each odd harmonic is a</text>',
         f'<text x="{xK(0.6):.1f}" y="{RY + RH - 40}" font-size="11" fill="{INK}">nearly fixed factor smaller, from 1 to 3 about</text>',
         f'<text x="{xK(0.6):.1f}" y="{RY + RH - 26}" font-size="11" fill="{INK}">{r_lo:.0f}× at D 38 and {r_hi:.0f}× at D 232.6</text>',
         f'<text x="{xK(0.6):.1f}" y="{RY + RH - 10}" font-size="11" fill="{INK_SOFT}">hollow: held on the fit (the chains, D above 73)</text>'],
        on(S3[0] + 6.4, S3[1]))

    # step 4: p over x and theta at D 38 and D 232.6, from the five nodes; p1 is its slope at x = 0
    def surface(s, ox, oy, scale, t_in):
        def P(t, x, p):
            return ox + 280 * t / (2 * np.pi) + 160 * x / 0.7, oy - 165 * x / 0.7 - scale * p
        rows = [(0.0, np.zeros_like(th))] + [(x, wave(co, th)) for x, co in s["nodes"]]
        p1 = wave(s["co"], th)
        body = []
        tr = np.linspace(t_in + 0.4, t_in + 3.4, len(rows))
        for j, ((x, p), tj) in enumerate(zip(rows[::-1], tr)):
            line = [P(t, x, v) for t, v in zip(th, p)]
            grp([draw_in(clock, line, SERIES[0] if x else INK_SOFT, 1.8 if x else 1.2, float(tj), float(tj) + 0.8)], on(float(tj), S4[1]))
            if x:
                body.append(f'<text x="{line[-1][0] + 6:.1f}" y="{line[-1][1] + 4:.1f}" font-size="10" fill="{INK_SOFT}">x {x:.1f}</text>')
        for tc in np.linspace(0, 2 * np.pi, 17):
            vals = [(0.0, 0.0)] + [(x, float(wave(co, np.array([tc]))[0])) for x, co in s["nodes"]]
            near = [P(tc, x, v) for x, v in vals[:2]]
            far = [P(tc, x, v) for x, v in vals[1:]]
            body.append(f'<path d="{pa.path_d(near)}" fill="none" stroke="{SERIES[0]}" stroke-width="0.8" stroke-opacity="0.5" stroke-dasharray="3 3"/>')
            body.append(f'<path d="{pa.path_d(far)}" fill="none" stroke="{SERIES[0]}" stroke-width="0.8" stroke-opacity="0.5"/>')
        x7, co7 = s["nodes"][-1]
        dash = [P(t, x7, x7 * v) for t, v in zip(th, p1)]
        body.append(f'<text x="{P(0, 0, 0)[0] - 6:.1f}" y="{P(0, 0, 0)[1] + 4:.1f}" text-anchor="end" font-size="10" fill="{INK_SOFT}">x 0</text>')
        grp(body, on(t_in + 3.6, S4[1]))
        a0, a1 = P(0, 0, 0), P(2 * np.pi, 0, 0)
        b1 = P(0, 0.7, 0)
        labels = [f'<text x="{(a0[0] + a1[0]) / 2:.1f}" y="{a0[1] + 22:.1f}" text-anchor="middle" font-size="11" fill="{INK}">θ, over one echo period →</text>',
                  f'<text x="{(a0[0] + b1[0]) / 2 - 10:.1f}" y="{(a0[1] + b1[1]) / 2:.1f}" text-anchor="end" font-size="11" fill="{INK}">radius x →</text>',
                  f'<text x="{ox + 220:.1f}" y="{oy - 262:.1f}" text-anchor="middle" font-size="13" font-weight="bold" fill="{INK}">D {s["D"]:.1f}</text>']
        grp(labels, on(t_in + 0.2, S4[1]))
        ratio = float(np.max(np.abs(wave(co7, th))) / (x7 * np.max(np.abs(p1))))
        grp([f'<path d="{pa.path_d(dash)}" fill="none" stroke="{SERIES[1]}" stroke-width="1.8" stroke-dasharray="5 3"/>',
             f'<text x="{ox + 220:.1f}" y="{oy - 246:.1f}" text-anchor="middle" font-size="11" fill="{SERIES[1]}">dashed: p1 × 0.7, the slope carried out to x = 0.7;</text>',
             f'<text x="{ox + 220:.1f}" y="{oy - 232:.1f}" text-anchor="middle" font-size="11" fill="{SERIES[1]}">the surface there has {100 * ratio:.1f}% of its height</text>'],
            on(t_in + 5.0, S4[1]))
        print(f"D {s['D']:.2f}: max|p(0.7)| / (0.7 max|p1|) = {ratio:.4f}")

    scale = 55 / max(float(np.max(np.abs(wave(co, th)))) for s in (reps[0], reps[-1]) for _, co in s["nodes"])
    surface(reps[0], 50, LY + LH - 20, scale, S4[0])
    surface(reps[-1], 545, LY + LH - 20, scale, S4[0] + 1.0)
    grp([f'<text x="{W / 2:.0f}" y="{LY + LH + 26}" text-anchor="middle" font-size="11" fill="{INK_SOFT}">height: p. Rows: p at the solver\'s nodes '
         f'x = 0.3 to 0.7, and p = 0 at the centre. Both surfaces on one vertical scale.</text>'], on(S4[0] + 3.6, S4[1]))

    # the step titles and the progress dots
    for (u, v), (title, text) in zip(STEPS, TITLES):
        clock.group(svg, [f'<text x="16" y="{LY + LH + 66}" font-size="14" font-weight="bold" fill="{INK}">{title}</text>',
                          f'<text x="16" y="{LY + LH + 88}" font-size="12" fill="{INK}">{text}</text>'], u, v - 0.3)
    for i, (u, v) in enumerate(STEPS):
        cx = 16 + 22 * i
        svg.add(f'<circle cx="{cx + 6}" cy="{LY + LH + 112}" r="5" fill="none" stroke="{INK_SOFT}"/>')
        clock.group(svg, [f'<circle cx="{cx + 6}" cy="{LY + LH + 112}" r="5" fill="{SERIES[1]}"/>'], u, v - 0.3)
    hgt = footnote(svg, [f"{sum(s['kind'] == 'full' for s in allst)} full solutions to D 73 (held_fit.py)",
                         "chains: raw T138-win2 chain and chain_o, I461-win2/crawlG"], "", [], LY + LH + 140, REPO)
    out = os.path.join(a.o, "wave_family.svg")
    open(out, "w").write(svg.finish(hgt))
    print(f"wrote {out}")


def draw_in(clock, pts_xy, color, width, t0, t1):
    """a polyline drawn from its start between t0 and t1 (stroke-dashoffset)"""
    L = sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pts_xy, pts_xy[1:])) + 1
    d = pa.path_d(pts_xy)
    if clock.frame is not None:
        f = 0 if clock.frame >= t1 else (L if clock.frame <= t0 else L * (1 - (clock.frame - t0) / (t1 - t0)))
        return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-dasharray="{L:.0f}" stroke-dashoffset="{f:.0f}"/>'
    kt = ";".join(f"{v:.4f}" for v in (0, t0 / clock.T, t1 / clock.T, 1))
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-dasharray="{L:.0f}" stroke-dashoffset="{L:.0f}">'
            f'<animate attributeName="stroke-dashoffset" values="{L:.0f};{L:.0f};0;0" keyTimes="{kt}" dur="{clock.T}s" '
            f'repeatCount="indefinite"/></path>')


if __name__ == "__main__":
    main()
