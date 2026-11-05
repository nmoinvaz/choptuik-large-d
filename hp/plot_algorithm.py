#!/usr/bin/env python3
"""The large-D chain's step (hp/anm_run.py with --jet_q 1 --pred_q 1 --reseed 2 --reseed_q 1), animated, in four SVGs.

  algo_step.svg       the loop: base, quad-double jets, step, prediction, re-seed, corrector, new base or a halved step
  algo_predictor.svg  the Taylor series in D: a toy branch's partial sums (an illustration) beside the chain's terms
                      |z_k| a^k at D 128.56 and 131.06
  algo_reseed.svg     the re-seed: the last node integrated toward the horizon, the horizon series inverted on the rungs
                      u0/32 .. u0 (a schematic), and each rung's residual against the double- and quad-double floors
  algo_corrector.svg  the corrector with the horizon data eliminated: re-seed, Newton step, re-seed of the trial (a
                      schematic), and max |F| per quad-double Newton step

The measured panels read the elimination pilot of notes/QUAD_DOUBLE.md 13.22 from the raw-data repository (clone it
beside this checkout: git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw):

    cd hp && python3 plot_algorithm.py ../../choptuik-large-d-raw [-o .]

The animations are SMIL (they play where the SVG is shown as an image); without animation every frame's content shows.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_graphs import GRID, INK, INK_SOFT, SERIES, SURFACE, Svg, axes, better_hint, callout, caption, esc, footnote, host_specs, legend, marker  # noqa: E402,E501

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
PILOT = "anm-mac/elimq_pilot"
EPS_DD, EPS_QD = 2.0 ** -106, 2.0 ** -212      # unit roundoff of double- and quad-double


# ---- animation: SMIL opacity schedules on one cycle per figure
FRAME = None                                  # --frame T: write the still at time T instead of the animation


class Clock:
    def __init__(self, cycle, fade=0.3):
        self.T, self.fade, self.frame = cycle, fade, FRAME

    def sched(self, pts):
        """<animate> of opacity through (time, value) points, held flat between them, looped every cycle"""
        pts = sorted(pts)
        if pts[0][0] > 0:
            pts = [(0.0, pts[0][1])] + pts
        if pts[-1][0] < self.T:
            pts = pts + [(self.T, pts[-1][1])]
        kt = ";".join(f"{min(max(t / self.T, 0.0), 1.0):.4f}" for t, _ in pts)
        vs = ";".join(f"{v:g}" for _, v in pts)
        return f'<animate attributeName="opacity" values="{vs}" keyTimes="{kt}" dur="{self.T}s" repeatCount="indefinite"/>'

    def pts(self, t_on, t_off=None, dim=None, t_dim=None):
        """visible from t_on (faded in) to t_off (the cycle's end by default); dim: the opacity it drops to at t_dim"""
        f = self.fade
        t_off = self.T - 0.7 if t_off is None else t_off
        pts = [(0.0, 0.0), (t_on, 0.0), (t_on + f, 1.0)]
        if dim is not None and t_dim is not None:
            pts += [(t_dim, 1.0), (t_dim + f, dim), (t_off, dim)]
        else:
            pts += [(t_off, 1.0)]
        pts += [(t_off + f, 0.0)]
        return pts

    def at(self, pts, t):
        """the schedule's opacity at time t (linear between points, as SMIL's default calcMode)"""
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t0 <= t <= t1:
                return v1 if t1 == t0 else v0 + (v1 - v0) * (t - t0) / (t1 - t0)
        return pts[-1][1]

    def group(self, svg, body, t_on, t_off=None, dim=None, t_dim=None):
        body = "".join(body) if isinstance(body, list) else body
        pts = self.pts(t_on, t_off, dim, t_dim)
        if self.frame is None:
            svg.add(f"<g>{body}{self.sched(pts)}</g>")
        elif self.at(pts, self.frame) > 0:
            svg.add(f'<g opacity="{self.at(pts, self.frame):.2f}">{body}</g>')


def capture(svg, fn):
    """the elements fn draws on svg, removed from it and returned as a list"""
    n = len(svg.parts)
    fn()
    out = svg.parts[n:]
    del svg.parts[n:]
    return out


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pow10(e):
    """10 to the integer e as text, 10⁻¹⁶"""
    return "10" + str(int(round(e))).translate(SUP)


def path_d(pts):
    return " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))


def arrow(svg, x1, y1, x2, y2, color, width=1.6, dash=None, head=8):
    """a line with a filled head at (x2, y2)"""
    ln = math.hypot(x2 - x1, y2 - y1) or 1
    ux, uy = (x2 - x1) / ln, (y2 - y1) / ln
    d = f' stroke-dasharray="{dash}"' if dash else ""
    svg.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2 - head * 0.8 * ux:.1f}" y2="{y2 - head * 0.8 * uy:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{d}/>')
    pts = (f"{x2:.1f},{y2:.1f} {x2 - head * ux - head * 0.45 * uy:.1f},{y2 - head * uy + head * 0.45 * ux:.1f} "
           f"{x2 - head * ux + head * 0.45 * uy:.1f},{y2 - head * uy - head * 0.45 * ux:.1f}")
    svg.add(f'<polygon points="{pts}" fill="{color}"/>')


def narrate(svg, clk, x, y, lines):
    """one line of text per stage, (t_on, t_off, text), centered at x"""
    for t_on, t_off, s in lines:
        clk.group(svg, f'<text x="{x:.1f}" y="{y:.1f}" font-size="12" fill="{INK}" text-anchor="middle">{esc(s)}</text>',
                  t_on, t_off)


# ---- data
def load(raw):
    recs = [json.loads(l) for l in open(os.path.join(raw, PILOT, "anm_run.jsonl")) if l.strip()]
    log = open(os.path.join(raw, PILOT + ".log")).read().splitlines()
    # quad-double Newton histories per step: max|F| at each iterate, from the "qn" lines after each "step k:" line
    hist, k = {}, None
    for line in log:
        m = re.match(r"step (\d+): D ", line)
        if m:
            k = int(m.group(1)); hist[k] = []
            continue
        m = re.match(r"\s+qn (\d+)(?: \[\w+\])?: \|F\| ([0-9.e+-]+)", line)
        if m and k is not None:
            hist[k].append(float(m.group(2)))
        m = re.search(r"double-double \|F\| of its first two parts ([0-9.e+-]+)", line)
        if m and k is not None and k not in hist.setdefault("dd", {}):
            hist["dd"][k] = float(m.group(1))
    commit = subprocess.run(["git", "-C", raw, "log", "-1", "--format=%h", "--", PILOT, PILOT + ".log"],
                            capture_output=True, text=True).stdout.strip() or "uncommitted"
    return recs, hist, commit


def sources(commit):
    return [f"{PILOT} (anm_run.jsonl, .log) of the raw-data repository at {commit}", "hp/plot_algorithm.py"]


# ---- 1. the loop
def fig_step(recs, commit, out):
    r0 = recs[0]
    a0 = r0["attempts"][0]
    clk = Clock(16.0)
    svg = Svg(1000)
    y0 = caption(svg, 40, 34, "One step of the large-D chain: predict in D, re-seed the horizon, correct in quad-double",
                 f"hp/anm_run.py --jet_q 1 --pred_q 1 --reseed 2 --reseed_q 1 · times of the step from D {r0['D0']:.2f} "
                 f"to {a0['D']:.2f} on the Mac: jets {r0['t_coef']:.0f} s, re-seed {a0['reseed']['seconds']:.0f} s, "
                 f"corrector {a0['t_corr']:.0f} s")
    bw, bh = 196, 66
    xs = [40, 290, 540, 764]
    top, bot = y0 + 8, y0 + 150
    boxes = {
        "base": (xs[0], top, "base z₀ at D₀", "converged, four double parts"),
        "jets": (xs[1], top, "jets z₁ … z₂₄", f"quad-double, one LU · {r0['t_coef']:.0f} s"),
        "step": (xs[2], top, "step a", "min(a_max, h_max)"),
        "pred": (xs[3], top, "prediction z(D₀ + a)", "Σ z_k aᵏ, Horner in quad-double"),
        "reseed": (xs[3], bot, "re-seed", f"horizon from the last node · {a0['reseed']['seconds']:.0f} s"),
        "corr": (xs[2], bot, "corrector", f"qd Newton, trials re-seeded · {a0['t_corr']:.0f} s"),
        "new": (xs[0], bot, "new base", "D₀ ← D₀ + a, a ← min(2a, h_max)"),
    }
    for key, (x, y, t1, t2) in boxes.items():
        svg.add(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="{SURFACE}" stroke="{INK_SOFT}" stroke-width="1.2"/>')
        svg.text(x + bw / 2, y + 28, t1, size=13, fill=INK, anchor="middle", weight="bold")
        svg.text(x + bw / 2, y + 47, t2, size=11, anchor="middle")
    # decision diamond between the corrector and the new base
    dx, dy, dr = (xs[0] + bw + xs[2]) / 2, bot + bh / 2, 40
    svg.add(f'<polygon points="{dx},{dy - dr} {dx + dr * 1.5},{dy} {dx},{dy + dr} {dx - dr * 1.5},{dy}" '
            f'fill="{SURFACE}" stroke="{INK_SOFT}" stroke-width="1.2"/>')
    svg.text(dx, dy + 4, "converged?", size=12, fill=INK, anchor="middle", weight="bold")
    mid = top + bh / 2
    for i in range(3):
        arrow(svg, xs[i] + bw + 4, mid, xs[i + 1] - 4, mid, INK_SOFT)
    arrow(svg, xs[3] + bw / 2, top + bh + 4, xs[3] + bw / 2, bot - 4, INK_SOFT)
    arrow(svg, xs[3] - 4, bot + bh / 2, xs[2] + bw + 4, bot + bh / 2, INK_SOFT)
    arrow(svg, xs[2] - 4, dy, dx + dr * 1.5 + 4, dy, INK_SOFT)
    arrow(svg, dx - dr * 1.5 - 4, dy, xs[0] + bw + 4, dy, SERIES[2])
    svg.text(dx - dr * 1.5 - 10, dy - 8, "yes", size=11, fill=INK, anchor="end")
    arrow(svg, xs[0] + bw / 2, bot - 4, xs[0] + bw / 2, top + bh + 4, SERIES[2])
    # failure: halve a, the series is kept
    fx = xs[2] + 40
    arrow(svg, dx + 20, dy - dr + 14, fx, top + bh + 4, SERIES[1], dash="5 4")
    svg.text(fx + 14, top + bh + 30, "no: a ← a/2, the series kept", size=11, fill=INK)
    svg.text(fx + 14, top + bh + 45, "(three times at most)", size=11)
    # the walk: each stage's box lit in turn
    order = [("base", 0.3, 1.4, "the converged state at D₀: 769 unknowns in four double parts"),
             ("jets", 1.4, 6.4, "the Taylor coefficients of the branch in D, order by order, through one quad-double LU"),
             ("step", 6.4, 7.4, "the step from the series' tail: its last term ε = 10⁻⁵ of the first, capped at h_max"),
             ("pred", 7.4, 8.6, "the series summed at D₀ + a in quad-double"),
             ("reseed", 8.6, 10.0, "the horizon data replaced by the regular solution's through the last node"),
             ("corr", 10.0, 12.6, "quad-double Newton on the pinned system; each trial point is re-seeded first"),
             ("dec", 12.6, 13.4, "converged to max |F| < 10⁻²⁸, or the step is halved and the prediction redone"),
             ("new", 13.4, 15.0, "the converged state is the next base")]
    for key, t_on, t_off, s in order:
        if key == "dec":
            body = (f'<polygon points="{dx},{dy - dr} {dx + dr * 1.5},{dy} {dx},{dy + dr} {dx - dr * 1.5},{dy}" '
                    f'fill="none" stroke="{SERIES[0]}" stroke-width="3"/>')
        else:
            x, y = boxes[key][:2]
            body = f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="none" stroke="{SERIES[0]}" stroke-width="3"/>'
        clk.group(svg, body, t_on, t_off)
    narrate(svg, clk, 500, bot + bh + 50, [(t0, t1, s) for _, t0, t1, s in order])
    height = footnote(svg, sources(commit), host_specs(), [], bot + bh + 60, REPO)
    open(out, "w").write(svg.finish(height))
    print("wrote", out)


# ---- 2. the predictor
def fig_predictor(recs, commit, out):
    clk = Clock(16.0)
    N = 24
    tk = lambda k: 1.0 + (k - 1) * 8.0 / (N - 1)          # noqa: E731, order k's time
    svg = Svg(1000)
    y = caption(svg, 50, 34, "Predictor: the branch's Taylor series in D, built order by order and summed at the step",
                "left: a toy branch (1 + h/R)^(−1/2) and its partial sums (an illustration) · right: the chain's terms "
                "|z_k| aᵏ (largest over the 769 unknowns) at two bases")
    X, W, H = 70, 330, 300
    # toy: f(x) = (1 + x)^(-1/2), x = h / R; coefficients binom(-1/2, k); step from the chain's rule at eps 1e-5
    c = [1.0]
    for k in range(1, N + 1):
        c.append(c[-1] * (-0.5 - (k - 1)) / k)
    a_toy = (1e-5 * abs(c[1]) / abs(c[N])) ** (1.0 / (N - 1))
    xmax = 1.3 * a_toy
    xm, ym = axes(svg, X, y, W, H, (0, xmax), (0.55, 1.02), "step h in units of the radius R",
                  "toy branch value", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    grid = [xmax * i / 160 for i in range(161)]
    svg.add(f'<path d="{path_d([(xm(x), ym((1 + x) ** -0.5)) for x in grid])}" fill="none" stroke="{SERIES[-1]}" stroke-width="3"/>')
    orders = [1, 2, 3, 4, 6, 10, 24]
    curves = []
    for n in orders:
        pts = [(xm(x), ym(sum(c[k] * x ** k for k in range(n + 1)))) for x in grid]
        curves.append([(px, py) for px, py in pts if y - 4 <= py <= y + H + 4])
    # labels at each curve's right end, pushed apart to 12 px
    ends = sorted(range(len(orders)), key=lambda i: curves[i][-1][1])
    ly = {}
    for i in ends:
        ly[i] = max(curves[i][-1][1], max(ly.values()) + 12 if ly else -1e9)
    for i, n in enumerate(orders):
        pts = curves[i]
        body = [f'<path d="{path_d(pts)}" fill="none" stroke="{SERIES[0]}" stroke-width="1.6"/>']
        lx = pts[-1][0]
        body.append(f'<text x="{lx + 6:.1f}" y="{ly[i] + 4:.1f}" font-size="10" fill="{SERIES[0]}">order {n}</text>')
        t_next = tk(orders[i + 1]) if i + 1 < len(orders) else None
        clk.group(svg, body, tk(n), dim=0.18 if t_next else None, t_dim=t_next)
    clk.group(svg, capture(svg, lambda: marker(svg, "circle", xm(0), ym(1.0), INK, "the base, h = 0")), 0.3)
    step = []
    step.append(f'<line x1="{xm(a_toy):.1f}" y1="{y:.1f}" x2="{xm(a_toy):.1f}" y2="{y + H:.1f}" stroke="{INK}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    step.append(f'<text x="{xm(a_toy) - 5:.1f}" y="{y + 14:.1f}" font-size="11" fill="{INK}" text-anchor="end">step a = {a_toy:.2f} R</text>')
    clk.group(svg, step, 9.6)
    pv = sum(c[k] * a_toy ** k for k in range(N + 1))
    clk.group(svg, capture(svg, lambda: (marker(svg, "diamond", xm(a_toy), ym(pv), SERIES[1], f"prediction at a: {pv:.6f}"),
                                         callout(svg, xm(a_toy), ym(pv), "prediction", dx=-30, dy=-60))), 11.6)
    svg.text(X + 8, y + H - 10, "the nearest singularity is at h = −R, behind the base", size=10)
    legend(svg, X + W, y - 12, [("branch", SERIES[-1]), ("partial sums", SERIES[0])])
    # right: the chain's terms at its two bases
    X2, W2 = 620, 330
    xm2, ye2 = axes(svg, X2, y, W2, H, (0, N), (-16, 1), "order k", "largest term |z_k| aᵏ (log scale)",
                    xfmt=lambda v: f"{v:g}", yfmt=pow10)
    ym2 = lambda v: ye2(math.log10(v))                      # noqa: E731
    for si, r in enumerate(recs[:2]):
        a = r["attempts"][0]["a"]
        col = SERIES[si]
        terms = [nk * a ** k for k, nk in enumerate(r["norms"])]
        line = f'<path d="{path_d([(xm2(k), ym2(v)) for k, v in enumerate(terms)])}" fill="none" stroke="{col}" stroke-width="1.4"/>'
        clk.group(svg, line, tk(N) + 0.3)
        for k, v in enumerate(terms):
            shape = "circle" if si == 0 else "square"
            clk.group(svg, capture(svg, lambda: marker(svg, shape, xm2(k), ym2(v), col,
                                                       f"D {r['D0']:.2f}, a {a}: order {k}, |z_k| a^k {v:.2e}")),
                      0.3 if k == 0 else tk(k))
    r0 = recs[0]
    a0, am = r0["attempts"][0]["a"], r0["a_max"]
    bound = 1e-5 * r0["norms"][1] * a0
    rule = [f'<line x1="{X2}" y1="{ym2(bound):.1f}" x2="{X2 + W2}" y2="{ym2(bound):.1f}" stroke="{INK}" stroke-width="1.2" stroke-dasharray="4 3"/>',
            f'<text x="{X2 + W2 - 4}" y="{ym2(bound) - 6:.1f}" font-size="10" fill="{INK}" text-anchor="end">ε |z₁| a, the step rule’s bound (ε = 10⁻⁵)</text>']
    clk.group(svg, rule, 9.6)
    last = r0["norms"][N] * a0 ** N
    clk.group(svg, capture(svg, lambda: callout(svg, xm2(N), ym2(last), f"a = h_max {a0:g}, the rule allows {am:.2f}", dx=-70, dy=-2)), 10.6)
    r1 = recs[1]
    t5 = r1["norms"][5] * r1["attempts"][0]["a"] ** 5
    ratio = t5 / (r0["norms"][5] * a0 ** 5)
    clk.group(svg, capture(svg, lambda: callout(svg, xm2(5), ym2(t5), f"order 5: {ratio:.0f}× larger at D {r1['D0']:.2f}",
                                                dx=30, dy=-40)), tk(6))
    legend(svg, X2 + W2, y - 12, [(f"base D {recs[0]['D0']:.2f}", SERIES[0]), (f"base D {recs[1]['D0']:.2f}", SERIES[1])])
    better_hint(svg, X2, y, W2, H, where="right", label="smaller", reverse=True)
    narrate(svg, clk, 500, y + H + 62, [
        (0.3, 1.0, "the base: a converged state at D₀"),
        (1.0, 9.6, "order by order: J(z₀) z_k = −r_k, r_k from the quad-double jet of z₀ + z₁h + … + z_(k−1)h^(k−1)"),
        (9.6, 11.6, "the step: a_max where the last term |z_N| a^N falls to ε |z₁| a, then a = min(a_max, h_max)"),
        (11.6, 15.3, "the prediction Σ z_k aᵏ, summed by Horner in quad-double, goes to the re-seed")])
    height = footnote(svg, sources(commit), host_specs(), [], y + H + 72, REPO)
    open(out, "w").write(svg.finish(height))
    print("wrote", out)
    print("| order k | " + " | ".join(f"D {r['D0']:.2f}, a {r['attempts'][0]['a']}" for r in recs[:2]) + " |")
    print("|---|---|---|")
    for k in range(N + 1):
        print(f"| {k} | " + " | ".join(f"{r['norms'][k] * r['attempts'][0]['a'] ** k:.2e}" for r in recs[:2]) + " |")


# ---- 3. the re-seed
def fig_reseed(recs, commit, out):
    clk = Clock(16.0)
    svg = Svg(1000)
    y = caption(svg, 70, 34, "Re-seed: the horizon data of the regular solution through the last node",
                "top: a schematic along u = 1 − x from the last node (u₀ = 0.3) to the horizon (u = 0) · bottom: each "
                "rung's residual in the pilot's re-seeds of the predictions at D 131.06 and 133.56")
    X, W = 110, 760
    ulo, uhi = 0.006, 0.36
    um = lambda u: X + (math.log10(u) - math.log10(ulo)) / (math.log10(uhi) - math.log10(ulo)) * W  # noqa: E731
    u0 = 0.3
    rungs = [u0 / 2 ** j for j in range(5, -1, -1)]
    # top: schematic field values; the regular solution s(u) and each rung's horizon series before and after its Newton
    ty, TH = y + 4, 170
    s = lambda u: ty + 30 + (TH - 60) * (u / u0) ** 0.6       # noqa: E731
    us = [ulo * (uhi / ulo) ** (i / 200) for i in range(201)]
    svg.add(f'<path d="{path_d([(um(u), s(u)) for u in us if u <= u0])}" fill="none" stroke="{SERIES[-1]}" stroke-width="3"/>')
    svg.line(um(ulo), ty + TH, um(uhi), ty + TH, INK_SOFT)
    for u, lab in zip(rungs, ["u₀/32", "u₀/16", "u₀/8", "u₀/4", "u₀/2", "u₀"]):
        svg.line(um(u), ty + TH, um(u), ty + TH + 5, INK_SOFT)
        svg.text(um(u), ty + TH + 18, lab, size=11, anchor="middle")
    svg.text(um(ulo), ty + TH - 8, "← horizon (u = 0)", size=11, anchor="start")
    svg.text(um(u0) - 12, s(u0) + 4, "last node", size=11, fill=INK, anchor="end")
    clk.group(svg, capture(svg, lambda: marker(svg, "circle", um(u0), s(u0), INK, "the last node of the prediction, u0 = 0.3")), 0.3)
    # the stepper: plan points from the node inward, steps of at most 0.05 between the rungs
    steps, u = [], u0
    for target in reversed(rungs[:-1]):
        while u - target > 1e-12:
            u = max(u - 0.05, target); steps.append(u)
    t_step = [1.0 + 3.0 * i / len(steps) for i in range(len(steps))]
    for u, t in zip(steps, t_step):
        clk.group(svg, f'<circle cx="{um(u):.1f}" cy="{s(u):.1f}" r="3.5" fill="{SERIES[2]}"/>', t)
    svg.text(um(0.05), s(0.05) + 32, "Taylor stepper, order 36, steps ≤ 0.05", size=10, fill=SERIES[2], anchor="middle")
    t_r = [4.6 + 1.1 * j for j in range(6)]
    miss = [34, 26, 20, 15, 11, 8]
    for j, u in enumerate(rungs):
        series_u = [v for v in us if v <= u]
        before = [(um(v), s(v) - miss[j] * (v / u) ** 2) for v in series_u]
        after = [(um(v), s(v)) for v in series_u]
        clk.group(svg, [f'<path d="{path_d(before)}" fill="none" stroke="{SERIES[1]}" stroke-width="1.6" stroke-dasharray="5 3"/>',
                        f'<line x1="{um(u):.1f}" y1="{s(u) - miss[j]:.1f}" x2="{um(u):.1f}" y2="{s(u):.1f}" stroke="{SERIES[1]}" stroke-width="1"/>'],
                  t_r[j], t_r[j] + 0.55)
        clk.group(svg, [f'<path d="{path_d(after)}" fill="none" stroke="{SERIES[0]}" stroke-width="2"/>',
                        f'<circle cx="{um(u):.1f}" cy="{s(u):.1f}" r="6" fill="none" stroke="{SERIES[0]}" stroke-width="1.6"/>'],
                  t_r[j] + 0.55, dim=0.25 if j < 5 else None, t_dim=t_r[j] + 1.1 if j < 5 else None)
    legend(svg, X + W, ty - 6, [("regular solution", SERIES[-1]), ("horizon series, a rung's start", SERIES[1]),
                                ("solved", SERIES[0])])
    # bottom: the rungs' residuals and the two floors eps G(u), G(u) = (1 - u)^-(D - 3), the mass mode's gain
    by, BH = ty + TH + 50, 250
    xm, ye = axes(svg, X, by, W, BH, (ulo, uhi, "log"), (-66, -8), "rung u = 1 − x (log scale)",
                  "residual max |R| at the rung (log scale)", xfmt=lambda v: f"{v:g}", yfmt=pow10)
    ym = lambda v: ye(math.log10(v))                           # noqa: E731
    D1 = recs[0]["attempts"][0]["D"]
    for eps, lab, col, t, off in ((EPS_DD, "double-double floor 2⁻¹⁰⁶ G(u)", INK_SOFT, 4.6, 15),
                                  (EPS_QD, "quad-double floor 2⁻²¹² G(u)", INK, 11.6, -7)):
        pts = [(xm(u), ym(eps * (1 - u) ** -(D1 - 3))) for u in us]
        clk.group(svg, [f'<path d="{path_d(pts)}" fill="none" stroke="{col}" stroke-width="1.2" stroke-dasharray="4 3"/>',
                        f'<text x="{xm(0.0065):.1f}" y="{ym(eps * (1 - 0.0065) ** -(D1 - 3)) + off:.1f}" font-size="10" fill="{col}">{lab}, D {D1:.2f}</text>'], t)
    for si, r in enumerate(recs[:2]):
        rs = r["attempts"][0]["reseed"]["rungs"]
        col, shape = SERIES[si], ("circle", "square")[si]
        dd, qd = rs[:-1], rs[-1]
        for j, g in enumerate(dd):
            clk.group(svg, capture(svg, lambda: marker(svg, shape, xm(g["u"]) + (si * 2 - 1) * 4, ym(g["R"]), col,
                                                       f"D {r['attempts'][0]['D']:.2f}, rung u {g['u']:.6f}: |R| {g['R']:.2e} after {g['its']} Newton steps")),
                      t_r[j] + 0.55)
        clk.group(svg, capture(svg, lambda: marker(svg, "diamond", xm(qd["u"]) + (si * 2 - 1) * 4, ym(qd["R"]), col,
                                                   f"D {r['attempts'][0]['D']:.2f}, u0 polished in quad-double: |R| {qd['R']:.2e} after {qd['its']} steps")),
                  11.6)
    q0 = recs[0]["attempts"][0]["reseed"]["rungs"][-1]
    clk.group(svg, capture(svg, lambda: callout(svg, xm(q0["u"]) - 4, ym(q0["R"]), f"u₀ polished in quad-double: {q0['R']:.1e}", dx=-60, dy=-40)), 11.9)
    legend(svg, X + W, by - 12, [(f"prediction at D {recs[0]['attempts'][0]['D']:.2f}", SERIES[0]),
                                 (f"at D {recs[1]['attempts'][0]['D']:.2f}", SERIES[1])])
    better_hint(svg, X, by, W, BH, where="right", label="smaller", reverse=True)
    narrate(svg, clk, X + W / 2, by + BH + 62, [
        (0.3, 1.0, "the prediction's last node, at u₀ = 0.3"),
        (1.0, 4.6, "the node integrated toward the horizon by the kernel's Taylor stepper, to u₀/32"),
        (4.6, 11.6, "each rung: Newton on the 2M horizon unknowns (m_h, a_h) in the m and a rows, from the last rung's data"),
        (11.6, 15.3, "at u₀ the solve is polished in quad-double (residual_q, a QDLu solve) down to its floor")])
    height = footnote(svg, sources(commit), host_specs(), [], by + BH + 72, REPO)
    open(out, "w").write(svg.finish(height))
    print("wrote", out)
    print("| rung u | " + " | ".join(f"D {r['attempts'][0]['D']:.2f}" for r in recs[:2]) + " | dd floor | qd floor |")
    print("|---|---|---|---|---|")
    for j, g in enumerate(recs[0]["attempts"][0]["reseed"]["rungs"]):
        r1 = recs[1]["attempts"][0]["reseed"]["rungs"][j]
        dd_f = "" if j == 6 else f"{EPS_DD * (1 - g['u']) ** -(D1 - 3):.1e}"
        print(f"| {g['u']:.6f}{' qd' if j == 6 else ''} | {g['R']:.2e} | {r1['R']:.2e} | {dd_f} | "
              f"{EPS_QD * (1 - g['u']) ** -(D1 - 3):.1e} |")


# ---- 4. the corrector
def fig_corrector(recs, hist, commit, out):
    clk = Clock(14.0)
    svg = Svg(1000)
    y = caption(svg, 50, 34, "Corrector: quad-double Newton with the horizon data eliminated",
                "left: a schematic of one correction · right: max |F| at each quad-double Newton iterate of the pilot's "
                "corrections at D 131.06 and 133.56")
    X, W, H = 60, 400, 300
    svg.add(f'<rect x="{X}" y="{y}" width="{W}" height="{H}" fill="none" stroke="{GRID}"/>')
    arrow(svg, X + 10, y + H - 10, X + 10, y + 30, INK_SOFT, width=1.2)
    svg.add(f'<text x="{X + 22}" y="{y + 40}" font-size="11" fill="{INK_SOFT}">horizon data (m_h, a_h)</text>')
    arrow(svg, X + 10, y + H - 10, X + W - 20, y + H - 10, INK_SOFT, width=1.2)
    svg.text(X + W - 24, y + H - 18, "the other unknowns (nodes, centre, Δ)", size=11, anchor="end")
    # the re-seed set: horizon data of the regular solution through the last node, a curve over the other unknowns
    cx = lambda t: X + 40 + t * (W - 80)                    # noqa: E731
    cy = lambda t: y + H - 70 - 150 * t + 60 * t * t           # noqa: E731
    svg.add(f'<path d="{path_d([(cx(i / 100), cy(i / 100)) for i in range(101)])}" fill="none" stroke="{SERIES[-1]}" stroke-width="3"/>')
    svg.text(cx(0.0), cy(0.0) + 20, "points with re-seeded horizon data", size=11, anchor="start")
    ts = 0.72
    marker(svg, "circle", cx(ts), cy(ts), SERIES[2], "the solution: F = 0")
    svg.text(cx(ts) + 4, cy(ts) + 22, "solution", size=11, fill=INK)
    P = (cx(0.30), cy(0.30) - 70)
    Q = [(cx(0.30), cy(0.30)), (cx(0.62), cy(0.62)), (cx(ts), cy(ts))]
    Tr = [(cx(0.62), cy(0.62) - 26), (cx(ts) + 2, cy(ts) - 5)]
    clk.group(svg, capture(svg, lambda: (marker(svg, "diamond", P[0], P[1], SERIES[1], "the prediction"),
                                         svg.text(P[0] - 10, P[1] + 4, "prediction", size=11, fill=INK, anchor="end"))), 0.8)
    clk.group(svg, capture(svg, lambda: arrow(svg, P[0], P[1] + 8, Q[0][0], Q[0][1] - 7, SERIES[1], dash="4 3")), 2.0)
    clk.group(svg, capture(svg, lambda: marker(svg, "circle", Q[0][0], Q[0][1], SERIES[0], "re-seeded prediction")), 2.4)
    times = [(3.6, 4.0, 5.2, 5.6), (6.8, 7.2, 8.4, 8.8)]
    for i, (t_step, t_trial, t_rs, t_q) in enumerate(times):
        clk.group(svg, capture(svg, lambda: arrow(svg, Q[i][0] + 6, Q[i][1] - 3, Tr[i][0] - 6, Tr[i][1] + 2, SERIES[0])), t_step)
        clk.group(svg, capture(svg, lambda: marker(svg, "square", Tr[i][0], Tr[i][1], SERIES[0], f"trial point {i + 1}")), t_trial)
        clk.group(svg, capture(svg, lambda: arrow(svg, Tr[i][0], Tr[i][1] + 7, Q[i + 1][0], Q[i + 1][1] - 6, SERIES[1], dash="4 3")), t_rs)
        clk.group(svg, capture(svg, lambda: marker(svg, "circle", Q[i + 1][0], Q[i + 1][1], SERIES[0], f"iterate {i + 1}, re-seeded")), t_q)
    svg.add(f'<g><line x1="{X + W - 150}" y1="{y + 22}" x2="{X + W - 128}" y2="{y + 22}" stroke="{SERIES[1]}" stroke-width="1.6" stroke-dasharray="4 3"/>'
            f'<text x="{X + W - 122}" y="{y + 26}" font-size="11" fill="{INK}">re-seed</text>'
            f'<line x1="{X + W - 150}" y1="{y + 40}" x2="{X + W - 128}" y2="{y + 40}" stroke="{SERIES[0]}" stroke-width="1.6"/>'
            f'<text x="{X + W - 122}" y="{y + 44}" font-size="11" fill="{INK}">Newton step</text></g>')
    # right: the measured histories
    X2, W2 = 620, 330
    xm, ye = axes(svg, X2, y, W2, H, (-0.3, 2.3), (-33, -9), "quad-double Newton iterate",
                  "max |F| of the pinned system (log scale)", xfmt=lambda v: f"{v:g}" if v == int(v) else "", yfmt=pow10)
    ym = lambda v: ye(math.log10(v))                           # noqa: E731
    t_it = [2.4, 5.6, 8.8]
    for si in range(2):
        h = hist[si]
        col, shape = SERIES[si], ("circle", "square")[si]
        D = recs[si]["attempts"][0]["D"]
        for i, v in enumerate(h):
            if i:
                clk.group(svg, f'<line x1="{xm(i - 1):.1f}" y1="{ym(h[i - 1]):.1f}" x2="{xm(i):.1f}" y2="{ym(v):.1f}" stroke="{col}" stroke-width="1.4"/>', t_it[i])
            clk.group(svg, capture(svg, lambda: marker(svg, shape, xm(i) + (si * 2 - 1) * 4, ym(v), col, f"D {D:.2f}, iterate {i}: max|F| {v:.2e}")), t_it[i])
    tol = 1e-28
    svg.add(f'<line x1="{X2}" y1="{ym(tol):.1f}" x2="{X2 + W2}" y2="{ym(tol):.1f}" stroke="{INK}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    svg.text(X2 + 4, ym(tol) - 6, "tolerance 10⁻²⁸ (--tol_qd)", size=10, fill=INK)
    ddf = hist["dd"][0]
    svg.add(f'<line x1="{X2}" y1="{ym(ddf):.1f}" x2="{X2 + W2}" y2="{ym(ddf):.1f}" stroke="{INK_SOFT}" stroke-width="1.2" stroke-dasharray="4 3"/>')
    svg.text(X2 + W2 - 4, ym(ddf) - 6, f"the same state rounded to double-double: {ddf:.1e}", size=10, anchor="end")
    h0 = hist[0]
    clk.group(svg, capture(svg, lambda: callout(svg, xm(1) - 4, ym(h0[1]), f"{h0[1]:.1e} in one step", dx=40, dy=-14)), 6.0)
    legend(svg, X2 + W2, y - 12, [(f"D {recs[0]['attempts'][0]['D']:.2f}", SERIES[0]), (f"D {recs[1]['attempts'][0]['D']:.2f}", SERIES[1])])
    better_hint(svg, X2, y, W2, H, where="right", label="smaller", reverse=True)
    narrate(svg, clk, 500, y + H + 62, [
        (0.8, 2.0, "the prediction at D₀ + a"),
        (2.0, 3.6, "re-seeded: its horizon data replaced by those through its last node"),
        (3.6, 5.2, "a quad-double Newton step on the pinned system"),
        (5.2, 6.8, "the trial point is re-seeded at u₀ before its residual is judged"),
        (6.8, 9.6, "and again, until max |F| is below the tolerance"),
        (9.6, 13.3, "the converged state is the next base")])
    height = footnote(svg, sources(commit), host_specs(), [], y + H + 72, REPO)
    open(out, "w").write(svg.finish(height))
    print("wrote", out)
    print("| iterate | " + " | ".join(f"D {recs[i]['attempts'][0]['D']:.2f}" for i in range(2)) + " |")
    print("|---|---|---|")
    for i in range(max(len(hist[0]), len(hist[1]))):
        print(f"| {i} | " + " | ".join(f"{hist[s][i]:.2e}" if i < len(hist[s]) else "" for s in range(2)) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("-o", default=HERE)
    ap.add_argument("--frame", type=float, default=None, help="write the still at this time (s) instead")
    a = ap.parse_args()
    global FRAME
    FRAME = a.frame
    recs, hist, commit = load(a.raw)
    fig_step(recs, commit, os.path.join(a.o, "algo_step.svg"))
    fig_predictor(recs, commit, os.path.join(a.o, "algo_predictor.svg"))
    fig_reseed(recs, commit, os.path.join(a.o, "algo_reseed.svg"))
    fig_corrector(recs, hist, commit, os.path.join(a.o, "algo_corrector.svg"))


if __name__ == "__main__":
    main()
