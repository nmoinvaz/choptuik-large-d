#!/usr/bin/env python3
"""newton_stall.svg: why the large-D continuation stalls along the weak directions, and what the stall looks like.

Top, a model (no data): the residual along one weak direction, F(t) = F0 - sigma t + c t^2 / 2, against Newton's
straight-line model F0 - sigma t, in units of F0 and F0 / sigma. Newton jumps to the model's zero, t = F0 / sigma; the
curvature has added c F0^2 / (2 sigma^2) there, so the step reduces the residual only while k = c F0 / sigma^2 < 2
(the basin, F0 < 2 sigma^2 / c). Two cases with the same start and slope: k = 0.3 (Newton converges) and k = 2.5
(it lands above its start, and line search ends at the curve's lowest point, 1 - 1 / (2 k), above zero).
Bottom, data: each continuation run's step in D (as plot_point_spacing.py, which supplies the steps up to T48) and
T73's two runs (pin 7 and pin 5, folder T73 of the raw-data repository), with the stalls at 35.2, 60.96 and 67.78 marked.
Standard library only; the primitives are data_graphs.py, vendored unchanged from github.com/nmoinvaz/speedy-gonzales
(skills/graph-data). Prints the plotted numbers.

    git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw   (beside the repository, once)
    cd hp && python3 plot_newton_stall.py ../../choptuik-large-d-raw/T73
"""
import json
import os
import subprocess
import sys

from data_graphs import INK, SERIES, Svg, axes, better_hint, callout, caption, esc, footnote, legend, marker
from plot_point_spacing import MACHINE, nice_hi, nice_lo, steps

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/nmoinvaz/choptuik-large-d"
WIDTH, X, W = 1080, 80, 920
CASES = ((0.3, "converges: k = c·F₀/σ² = 0.3"), (2.5, "stalls: k = 2.5"))


def resid(k, t):
    """F / F0 at t (in units of F0 / sigma) for k = c F0 / sigma^2"""
    return 1.0 - t + 0.5 * k * t * t


def polyline(svg, pts, color, width=2.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    svg.add(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{color}" '
            f'stroke-width="{width}"{d}/>')


def t73_steps(raw):
    out = []
    for run in ("pin7", "pin5"):
        Ds = [json.loads(line)["D"] for line in open(os.path.join(raw, "t73", f"{run}.jsonl")) if line.strip()]
        out += [(D, D - prev, run) for prev, D in zip(Ds, Ds[1:])]
    return sorted(out)


def main():
    if len(sys.argv) < 2 or not os.path.isfile(os.path.join(sys.argv[1], "t73", "pin5.jsonl")):
        sys.exit(__doc__)
    raw = sys.argv[1]
    raw_rev = subprocess.run(["git", "-C", raw, "log", "-1", "--format=%h", "--", ".", ":(exclude)README.md"], capture_output=True,
                             text=True).stdout.strip() or "?"

    svg = Svg(WIDTH)
    svg.text(16, 28, "Why the continuation stalls along the weak directions", size=15, fill=INK, weight="bold")

    # --- top: the model of one Newton step ---
    y = caption(svg, X, 58, "One Newton step along a weak direction (a model, not data)",
                "Same start and slope, two curvatures. Newton jumps to where its straight line reaches zero; along a "
                "weak direction that is far (slope σ small), and the curve has bent back up by then.")
    h, tmax = 300, 1.3
    xm, ym = axes(svg, X, y, W, h, (0.0, tmax), (-0.2, 1.4), "distance along the weak direction (units of F₀/σ)",
                  "residual (units of F₀)", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    better_hint(svg, X, y, W, h, "right", label="better", reverse=True)
    gray = SERIES[-1]
    polyline(svg, [(xm(0), ym(1.0)), (xm(1.2), ym(-0.2))], gray, 1.6, "6 4")
    print("model: k, t, F/F0 (true residual); Newton's line F/F0 = 1 - t")
    for i, (k, label) in enumerate(CASES):
        pts = [(t := tmax * j / 260, resid(k, t)) for j in range(261)]
        polyline(svg, [(xm(t), ym(f)) for t, f in pts if -0.2 <= f <= 1.4], SERIES[i])
        land = resid(k, 1.0)
        marker(svg, "circle", xm(1.0), ym(land), SERIES[i], f"{label}: Newton lands at F = {land:.3g} F₀")
        print(f"  k {k}: lands at t = 1 with F = {land:.4g} F0; lowest point t = {1 / k:.4g}, F = "
              f"{max(0.0, 1 - 1 / (2 * k)):.4g} F0" + ("" if k > 0.5 else " (below zero: a root exists)"))
    marker(svg, "diamond", xm(0), ym(1.0), INK, "start: F = F₀")
    marker(svg, "cross", xm(1.0), ym(0.0), gray, "the zero of Newton's straight line, t = F₀/σ")
    k_bad = CASES[1][0]
    callout(svg, xm(1.0), ym(0.0), "Newton's line reaches zero at t = F₀/σ", dx=-36, dy=30)
    callout(svg, xm(1.0), ym(resid(k_bad, 1.0)), f"k = {k_bad}: lands at {resid(k_bad, 1.0):g} F₀, above its start",
            dx=44, dy=-14)
    callout(svg, xm(1 / k_bad), ym(1 - 1 / (2 * k_bad)),
            f"line search ends at the lowest point, {1 - 1 / (2 * k_bad):g} F₀: the stall", dx=30, dy=40)
    legend(svg, WIDTH - 16, 58, [(CASES[0][1], SERIES[0]), (CASES[1][1], SERIES[1]), ("Newton's straight line", gray)])

    # --- bottom: the data, each run's step in D ---
    old = [g for g in steps() if g[0] >= 20]
    new = t73_steps(raw)
    y2 = y + h + 92
    c_old, c_new = SERIES[2], SERIES[3]
    legend(svg, WIDTH - 16, y2, [("deflated Newton, to T48", c_old), ("pinned + one full step, T73", c_new)])
    yb = caption(svg, X, y2, "The symptom: the step in D collapses where Newton stops converging",
                 "Each point is the distance from a converged point to its run's previous one. After a failed solve "
                 "the step halves, so a stall shows as a dive; each fix let the steps recover.")
    h2 = 300
    gs = [g[1] for g in old] + [g[1] for g in new]
    xm2, ym2 = axes(svg, X, yb, W, h2, (20, 72), (nice_lo(min(gs)), nice_hi(max(gs)), "log"), "dimension D",
                    "step in D (log scale)", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    better_hint(svg, X, yb, W, h2, "right", label="faster")
    for D, g, m, run, jumped in old:
        svg.add(f'<circle cx="{xm2(D):.1f}" cy="{ym2(g):.1f}" r="2.4" fill="{c_old}">'
                f'<title>{esc(f"D = {D:.4f}: a step of {g:.4g} ({MACHINE[m]}, run {run})")}</title></circle>')
    for D, g, run in new:
        svg.add(f'<circle cx="{xm2(D):.1f}" cy="{ym2(g):.1f}" r="2.4" fill="{c_new}">'
                f'<title>{esc(f"D = {D:.4f}: a step of {g:.4g} (T73 {run})")}</title></circle>')
    print("\ndata: the smallest step near each stall")
    for lo, hi, label, dx, dy in ((35.0, 35.5, "D = {D:.2f}: a cloud run stopped; from 35 harmonics above 13 frozen",
                                   34, 0),
                                  (60.9, 61.1, "D = {D:.2f}: cured by holding the near-free harmonics on their trend",
                                   -40, 40),
                                  (67.7, 67.8, "D = {D:.2f}: T48 stopped; T73 pins and corrects", -40, -4)):
        D, g = min(((p[0], p[1]) for p in old if lo <= p[0] <= hi), key=lambda p: p[1])
        callout(svg, xm2(D), ym2(g), label.format(D=D), dx=dx, dy=dy)
        print(f"  D {D:.4f}: step {g:.3g}")
    tail = sorted(g for D, g, _ in new if D > 70)
    print(f"  T73 above D 70: median step {tail[len(tail) // 2]:.4f} over {len(tail)} steps")

    height = footnote(svg, ["top: the model F(t) = F₀ − σt + ½ct², no data",
                            "bottom: hp/data/delta_dd.csv, large_d/up70.jsonl, pilot_cloud/pilotB.jsonl, "
                            f"raw/T73 {raw_rev}"],
                      "runs: " + ", ".join(MACHINE.values()),
                      ["the top panel is a model of one weak direction, not a measurement",
                       "the runs used different settings and machines"], yb + h2 + 46, repo_url=REPO)
    with open(os.path.join(HERE, "newton_stall.svg"), "w") as fh:
        fh.write(svg.finish(height))


if __name__ == "__main__":
    main()
