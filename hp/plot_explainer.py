#!/usr/bin/env python3
"""The diagrams of ALGORITHM.md, each an SVG in hp/ (light and dark mode, data_graphs.py primitives):

  explain_step.svg     one step of the large-D chain, each stage colored by the number type it runs in
  explain_layout.svg   the unknown vector z of the chain's states (M 32, nodes 0.3-0.7, L = 769), drawn to scale
  explain_domain.svg   what the residual F(z) computes along x: centre series, Taylor steps between the nodes,
                       horizon series, and the matching rows
  explain_parts.svg    1/3 as one, two, four and eight doubles: each part's value and the error of their sum
  explain_soa.svg      a quad-double vector in memory: four contiguous float64 arrays, element i across them
  explain_gain.svg     the mass-mode gain G = 0.7^-(D-3) against D, with the chain's walls marked
  explain_newton.svg   what a Newton step is: the tangent's zero for x^2 - 2, and a step too long for arctan x,
                       refused because |f| grows, then halved

    cd hp && python3 plot_explainer.py [-o .]

The parts of 1/3 are computed here with mpmath; the layout reads hpkernel.driver's documented layout (offsets in
reseed.py); the walls' positions are those of notes/QUAD_DOUBLE.md 13.17-13.28 and notes/OCTO_DOUBLE.md.
The script also prints the numbers it draws.
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_graphs import GRID, INK, INK_SOFT, SERIES, SURFACE, Svg, axes, better_hint, callout, esc, footnote  # noqa: E402

REPO = "https://github.com/nmoinvaz/choptuik-large-d"
SRC = ["hp/plot_explainer.py"]
WHITE = "#ffffff"                     # label ink on colored fills (fixed for contrast, not themed)
TYPE = {"f64": SERIES[9], "dd": SERIES[0], "qd": SERIES[1], "od": SERIES[2]}
NAME = {"f64": "double (f64)", "dd": "double-double", "qd": "quad-double", "od": "octo-double"}


def box(svg, x, y, w, h, fill, lines, ink=WHITE, size=11, tip=""):
    svg.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" fill="{fill}">'
            f'<title>{esc(tip or " ".join(lines))}</title></rect>')
    n = len(lines)
    for i, s in enumerate(lines):
        svg.text(x + w / 2, y + h / 2 + 4 + (i - (n - 1) / 2) * (size + 3), s, size=size, fill=ink, anchor="middle",
                 weight="bold" if i == 0 else "normal")


def arrow(svg, x1, y1, x2, y2, color=INK_SOFT, label=None, lx=None, ly=None, anchor="middle"):
    svg.line(x1, y1, x2, y2, color, 1.5)
    ang = math.atan2(y2 - y1, x2 - x1)
    a, b = ang + 2.6, ang - 2.6
    svg.add(f'<polygon points="{x2:.1f},{y2:.1f} {x2 + 8 * math.cos(a):.1f},{y2 + 8 * math.sin(a):.1f} '
            f'{x2 + 8 * math.cos(b):.1f},{y2 + 8 * math.sin(b):.1f}" fill="{color}"/>')
    if label:
        svg.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 6, label, size=10,
                 anchor=anchor)


def type_legend(svg, x, y, keys):
    for k in keys:
        svg.add(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="12" height="12" rx="2" fill="{TYPE[k]}"/>')
        svg.text(x + 17, y + 1, NAME[k], size=11)
        x += 34 + 7 * len(NAME[k])


def step_svg():
    svg = Svg(860)
    svg.text(16, 24, "One step of the large-D chain, by number type", size=13, fill=INK, weight="bold")
    svg.text(16, 42, "the stages of hp/anm_run.py with --jet_o 1 --reseed 2 --reseed_q 1, as the chain runs past D 175",
             size=10)
    type_legend(svg, 16, 66, ["dd", "qd", "od"])
    y, h, w = 96, 70, 150
    xs = [16, 186, 356, 526, 696]
    box(svg, xs[0], y, w, h, TYPE["od"], ["Taylor series", "z_1 .. z_24 in D", "one LU, 24 jets"])
    box(svg, xs[1], y, w, h, TYPE["od"], ["prediction", "z(a) = sum z_k a^k", "Horner's rule"])
    # the re-seed: double-double rungs, then a quad-double polish at u0
    box(svg, xs[2], y, w, h / 2, TYPE["dd"], ["re-seed rungs", "u0/32 .. u0"], size=10)
    box(svg, xs[2], y + h / 2, w, h / 2, TYPE["qd"], ["polish at u0"], size=10)
    # the corrector: quad-double by default, octo-double with --corrector od
    box(svg, xs[3], y, w, h / 2, TYPE["qd"], ["Newton, quad-double", "to 1e-28"], size=10)
    box(svg, xs[3], y + h / 2, w, h / 2, TYPE["od"], ["or octo-double", "--corrector od, 1e-40"], size=10)
    box(svg, xs[4], y, 148, h, TYPE["qd"], ["new base", "state saved", "trend re-based"])
    for i in range(4):
        arrow(svg, xs[i] + w + 2, y + h / 2, xs[i + 1] - 4, y + h / 2)
    # the two outcomes of the corrector
    arrow(svg, xs[3] + w / 2, y + h + 4, xs[3] + w / 2, y + h + 40, label="not converged: halve a, keep the series",
          lx=xs[3] + w / 2 - 6, ly=y + h + 54, anchor="end")
    svg.line(xs[3] + w / 2, y + h + 40, xs[1] + w / 2, y + h + 40, INK_SOFT, 1.5)
    arrow(svg, xs[1] + w / 2, y + h + 40, xs[1] + w / 2, y + h + 4)
    svg.line(xs[4] + 74, y - 4, xs[4] + 74, y - 20, INK_SOFT, 1.5)
    svg.line(xs[4] + 74, y - 20, xs[0] + w / 2, y - 20, INK_SOFT, 1.5)
    arrow(svg, xs[0] + w / 2, y - 20, xs[0] + w / 2, y - 4, label="converged: next step from D + a", lx=430, ly=y - 26)
    callout(svg, xs[3] + w - 8, y + h * 0.75, "removes the 193.77 floor", dx=26, dy=34)
    return svg.finish(footnote(svg, SRC, "", [], y + h + 96, REPO))


def layout_svg(M=32, nodes=(0.3, 0.4, 0.5, 0.6, 0.7)):
    K = len(nodes)
    L = 4 * M * (K + 1) + 1
    svg = Svg(860)
    svg.text(16, 24, f"The unknown vector z, to scale: L = 4M(K + 1) + 1 = {L} numbers", size=13, fill=INK, weight="bold")
    svg.text(16, 42, f"the large-D chain's states: M = {M} points per half period in tau, K = {K} nodes at x = "
             + ", ".join(f"{n:g}" for n in nodes), size=10)
    x0, x1, y, h = 16, 844, 92, 34
    px = (x1 - x0) / L
    segs = [("centre data", "p1, c0", 2 * M, SERIES[9])]
    segs += [(f"node x = {n:g}", "p, m, a, c", 4 * M, SERIES[0]) for n in nodes]
    segs += [("horizon data", "m_h, a_h", 2 * M, SERIES[1]), ("Delta", "", 1, SERIES[4])]
    x = x0
    for name, sub, n, color in segs:
        w = max(n * px, 7)
        tip = f"{name}: {n} numbers" + (f" ({sub})" if sub else "")
        svg.add(f'<rect x="{x:.1f}" y="{y}" width="{max(w - 2, 3):.1f}" height="{h}" rx="2" fill="{color}"><title>{esc(tip)}</title></rect>')
        if n > 1:
            svg.text(x + (w - 2) / 2, y + h / 2 + 4, f"{n}", size=11, fill=WHITE, anchor="middle", tabular=True)
            svg.text(x + (w - 2) / 2, y - 22, name, size=11, fill=INK, anchor="middle")
            svg.text(x + (w - 2) / 2, y - 8, sub, size=10, anchor="middle")
        x += w
    callout(svg, x - 4, y + h / 2, "Delta, the period: 1 number", dx=-120, dy=38)
    # one node's block opened up: four fields of M Fourier coefficients each
    nx = x0 + 2 * M * px
    w4 = 4 * M * px
    yb = y + h + 58
    svg.line(nx, y + h + 2, x0, yb - 2, GRID)
    svg.line(nx + w4 - 2, y + h + 2, x0 + 4 * 200 - 2, yb - 2, GRID)
    for i, (f, what) in enumerate([("p", "scalar field, one null derivative"), ("m", "the other null derivative"),
                                   ("a", "radial metric function"), ("c", "lapse-type function")]):
        bx = x0 + i * 200
        svg.add(f'<rect x="{bx:.1f}" y="{yb}" width="196" height="30" rx="2" fill="{SERIES[0]}"/>')
        svg.text(bx + 98, yb + 19, f"{f}: {M} Fourier coefficients in tau", size=10, fill=WHITE, anchor="middle")
        svg.text(bx + 98, yb + 46, what, size=10, anchor="middle")
    return svg.finish(footnote(svg, SRC + ["hpkernel.driver layout"], "", [], yb + 60, REPO)), L


def domain_svg(nodes=(0.3, 0.4, 0.5, 0.6, 0.7)):
    svg = Svg(860)
    svg.text(16, 24, "What the residual F(z) computes along the radial coordinate x", size=13, fill=INK, weight="bold")
    svg.text(16, 42, "F = 0 exactly when the pieces join smoothly: then z is a self-similar solution with period Delta",
             size=10)
    x0, x1, y = 60, 820, 130
    X = lambda v: x0 + v * (x1 - x0)  # noqa: E731
    svg.line(x0, y, x1, y, INK_SOFT, 1.5)
    for v in (0, 0.25, 0.5, 0.75, 1):
        svg.line(X(v), y, X(v), y + 5, INK_SOFT)
        svg.text(X(v), y + 20, f"{v:g}", size=11, anchor="middle", tabular=True)
    svg.text((x0 + x1) / 2, y + 40, "x (centre at 0, past light cone at 1)", size=12, fill=INK, anchor="middle")
    # centre series and horizon series regions
    svg.add(f'<rect x="{X(0):.1f}" y="{y - 46}" width="{X(nodes[0]) - X(0):.1f}" height="26" rx="3" fill="{SERIES[9]}"/>')
    svg.text((X(0) + X(nodes[0])) / 2, y - 29, "centre series, order 146", size=10, fill=WHITE, anchor="middle")
    svg.add(f'<rect x="{X(nodes[-1]):.1f}" y="{y - 46}" width="{X(1) - X(nodes[-1]):.1f}" height="26" rx="3" fill="{SERIES[1]}"/>')
    svg.text((X(nodes[-1]) + X(1)) / 2, y - 29, "horizon series, order 130", size=10, fill=WHITE, anchor="middle")
    # Taylor steps between the nodes, outward
    for a, b in zip(nodes[:-1], nodes[1:]):
        xa, xb = X(a) + 4, X(b) - 4
        svg.add(f'<path d="M{xa:.1f},{y - 10} Q{(xa + xb) / 2:.1f},{y - 44} {xb:.1f},{y - 10}" fill="none" '
                f'stroke="{SERIES[0]}" stroke-width="1.6"/>')
        svg.add(f'<polygon points="{xb:.1f},{y - 8} {xb - 7:.1f},{y - 14} {xb - 1:.1f},{y - 17}" fill="{SERIES[0]}"/>')
    svg.text((X(nodes[0]) + X(nodes[-1])) / 2, y - 52, "Taylor steps of order 36 from node to node", size=10,
             fill=INK, anchor="middle")
    for v in nodes:
        svg.add(f'<circle cx="{X(v):.1f}" cy="{y}" r="5" fill="{SERIES[0]}" stroke="{SURFACE}" stroke-width="2"/>')
    # where the data enter
    svg.text(X(0), y + 64, "p1, c0 start the centre series", size=10, anchor="start")
    svg.text(X(0.5), y + 64, "each node holds p, m, a, c", size=10, anchor="middle")
    svg.text(X(1), y + 64, "m_h, a_h start the horizon series", size=10, anchor="end")
    # matching rows
    svg.text((x0 + x1) / 2, y + 88, "residual rows: at every node, the incoming piece minus the node's own values",
             size=11, fill=INK, anchor="middle")
    callout(svg, X(nodes[-1]), y, "gain G = 0.7^-(D-3) here", dx=60, dy=40)
    return svg.finish(footnote(svg, SRC, "", [], y + 100, REPO))


def parts_of_third():
    import mpmath as mp
    mp.mp.prec = 600
    v = mp.mpf(1) / 3
    parts, r = [], v
    for _ in range(8):
        p = float(r)
        parts.append(p)
        r -= mp.mpf(p)
    errs = {n: float(abs(sum(mp.mpf(p) for p in parts[:n]) - v) / v) for n in (1, 2, 4, 8)}
    return parts, errs


def parts_svg():
    parts, errs = parts_of_third()
    svg = Svg(860)
    svg.text(16, 24, "One number, 1/3, in the four formats", size=13, fill=INK, weight="bold")
    svg.text(16, 42, "each part is an ordinary 8-byte double; the value is their exact sum, each part far below the "
             "one before", size=10)
    y, bw = 66, 74
    for k, n in (("f64", 1), ("dd", 2), ("qd", 4), ("od", 8)):
        svg.text(16, y + 21, NAME[k], size=11, fill=INK)
        svg.text(16, y + 35, f"{8 * n} bytes", size=10)
        for i in range(n):
            box(svg, 130 + i * (bw + 4), y, bw, 30, TYPE[k], [f"{parts[i]:.3e}"], size=10, tip=f"part {i}: {parts[i]!r}")
        svg.text(130 + 8 * (bw + 4) + 6, y + 21, f"error {errs[n]:.1e}", size=11, anchor="start", tabular=True)
        y += 46
    return svg.finish(footnote(svg, SRC + ["mpmath at 600 bits"], "", [], y, REPO)), parts, errs


def soa_svg(L=769):
    svg = Svg(860)
    svg.text(16, 24, "A quad-double vector in memory: four float64 arrays, not an array of structs", size=13, fill=INK,
             weight="bold")
    svg.text(16, 42, f"z0 .. z3 are each {L} doubles ({8 * L} bytes); element i is z0[i] + z1[i] + z2[i] + z3[i]",
             size=10)
    y, cw, ch = 64, 44, 26
    cols = list(range(10)) + [None, L - 1]
    hi = L - 1
    for q in range(4):
        svg.text(16, y + 18, f"z{q}", size=12, fill=INK)
        x = 50
        for c in cols:
            if c is None:
                svg.text(x + cw / 2, y + 18, "...", size=12, anchor="middle")
            else:
                fill = TYPE["qd"] if c == hi else GRID
                svg.add(f'<rect x="{x}" y="{y}" width="{cw - 3}" height="{ch}" rx="2" fill="{fill}"/>')
                svg.text(x + (cw - 3) / 2, y + 17, f"[{c}]", size=10, fill=WHITE if c == hi else INK_SOFT,
                         anchor="middle", tabular=True)
            x += cw
        svg.text(x + 8, y + 18, f"bytes {8 * L * q} .. {8 * L * (q + 1) - 1}", size=10, tabular=True)
        y += ch + 8
    callout(svg, 50 + 11 * cw + (cw - 3) / 2, 64 + 3 * (ch + 8) + ch / 2, "element 768 is Delta, four parts",
            dx=-210, dy=34)
    return svg.finish(footnote(svg, SRC + ["the .npz states hold zh, zl, z2, z3 (and z4..z7)"], "", [], y + 40, REPO))


WALLS = [(79.27, "dd -> qd"), (152.93, "u0 polish"), (162.31, "a_h, m_h steps"), (171.27, "series -> od"),
         (188.02, "fail-fast"), (193.77, "qd floor")]


def gain_svg():
    G = lambda D: 0.7 ** -(D - 3)  # noqa: E731
    svg = Svg(860)
    svg.text(16, 24, "Why the precision has to grow: the mass-mode gain at the last node", size=13, fill=INK, weight="bold")
    svg.text(16, 42, "an error e in the horizon data a_h reaches the node at x = 0.7 as about e G, with G = 0.7^-(D-3)",
             size=10)
    x, y, w, h = 92, 70, 700, 300
    # the y axis is log10 G, linear, so its ticks fall every few decades
    xm, ym = axes(svg, x, y, w, h, (70, 210), (10, 32), "spacetime dimension D", "gain G",
                  xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"1e{v:g}")
    lg = lambda D: math.log10(G(D))  # noqa: E731
    pts = " ".join(f"{xm(D):.1f},{ym(lg(D)):.1f}" for D in [70 + 0.5 * i for i in range(281)])
    svg.add(f'<polyline points="{pts}" fill="none" stroke="{SERIES[0]}" stroke-width="2"/>')
    below = {79.27, 152.93, 188.02}           # labels right of and below the point; the rest left and above
    for D, label in WALLS:
        px, py = xm(D), ym(lg(D))
        svg.add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{SERIES[1]}" stroke="{SURFACE}" stroke-width="2">'
                f'<title>{esc(f"D {D}: {label}, G = {G(D):.1e}")}</title></circle>')
        if D in below:
            svg.text(px + 10, py + 16, f"{D:g} {label}", size=10, anchor="start")
        else:
            svg.text(px - 8, py - 10, f"{D:g} {label}", size=10, anchor="end")
    better_hint(svg, x, y, w, h, where="right", label="harder")
    return svg.finish(footnote(svg, SRC + ["walls: notes/QUAD_DOUBLE.md 13.17-13.28, notes/OCTO_DOUBLE.md"], "", [],
                               y + h + 40, REPO)), [(D, G(D)) for D, _ in WALLS]


def newton_svg():
    """two panels: one Newton step on f(x) = x^2 - 2 from 3, and a step too long on arctan x from 1.5, refused and halved"""
    def curve(svg, xm, ym, f, lo, hi, n=200):
        pts = [lo + (hi - lo) * i / n for i in range(n + 1)]
        svg.add('<polyline points="' + " ".join(f"{xm(x):.1f},{ym(f(x)):.1f}" for x in pts)
                + f'" fill="none" stroke="{SERIES[0]}" stroke-width="2.5"/>')

    def seg(svg, xm, ym, x0, y0, x1, y1, color, dash=False, width=1.5):
        svg.add(f'<line x1="{xm(x0):.1f}" y1="{ym(y0):.1f}" x2="{xm(x1):.1f}" y2="{ym(y1):.1f}" stroke="{color}" '
                f'stroke-width="{width}"' + (' stroke-dasharray="6 4"' if dash else "") + "/>")
    from data_graphs import caption, marker
    svg = Svg(980)
    svg.text(16, 24, "What a Newton step is", size=14, fill=INK, weight="bold")
    svg.text(16, 42, "To solve f(x) = 0, replace f by its tangent at the current point and jump to where the tangent is "
             "zero: x_new = x − f(x) / f′(x).", size=11)
    f, df = (lambda x: x * x - 2), (lambda x: 2 * x)
    xs = [3.0]
    for _ in range(4):
        xs.append(xs[-1] - f(xs[-1]) / df(xs[-1]))
    x, y, w, h = 70, 110, 380, 330
    caption(svg, x, 80, "One step: follow the tangent to zero", "f(x) = x² − 2, from x₀ = 3; the root is √2")
    xm, ym = axes(svg, x, y, w, h, (0.5, 3.4), (-3, 8), "x", "f(x)", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    seg(svg, xm, ym, 0.5, 0, 3.4, 0, INK_SOFT, width=1)
    curve(svg, xm, ym, f, 0.5, 3.4)
    for k in range(2):
        x0, x1 = xs[k], xs[k + 1]
        seg(svg, xm, ym, x1 - 0.15, df(x0) * (x1 - 0.15 - x0) + f(x0), x0 + 0.15, df(x0) * 0.15 + f(x0), SERIES[1])
        seg(svg, xm, ym, x1, 0, x1, f(x1), INK_SOFT, dash=True)
        marker(svg, "circle", xm(x0), ym(f(x0)), SERIES[1], f"x_{k} = {x0:.6f}, f = {f(x0):.4f}")
        marker(svg, "diamond", xm(x1), ym(0), SERIES[1], f"tangent zero: x_{k + 1} = {x1:.6f}")
    callout(svg, xm(xs[0]), ym(f(xs[0])), "x₀ = 3, f = 7", dx=12, dy=60)
    callout(svg, xm(xs[1]), ym(0), "x₁ = 1.833: where the tangent is zero", dx=60, dy=40)
    svg.text(x + 8, y + 18, f"Then x₂ = {xs[2]:.4f}, x₃ = {xs[3]:.6f}, x₄ = {xs[4]:.10f}:", size=11, fill=INK)
    svg.text(x + 8, y + 34, "the error squares each step (quadratic convergence).", size=11, fill=INK)
    g, dg = math.atan, (lambda x: 1 / (1 + x * x))
    x0 = 1.5
    full = x0 - g(x0) / dg(x0)
    half = x0 + 0.5 * (full - x0)
    x2, w2 = 560, 380
    caption(svg, x2, 80, "A step that is too long: refused, then halved",
            "f(x) = arctan x, from x₀ = 1.5: the slope is small, so the step is long")
    xm2, ym2 = axes(svg, x2, y, w2, h, (-2.5, 2.5), (-1.5, 1.5), "x", "f(x)", xfmt=lambda v: f"{v:g}",
                    yfmt=lambda v: f"{v:g}")
    seg(svg, xm2, ym2, -2.5, 0, 2.5, 0, INK_SOFT, width=1)
    curve(svg, xm2, ym2, g, -2.5, 2.5)
    seg(svg, xm2, ym2, full - 0.2, dg(x0) * (full - 0.2 - x0) + g(x0), x0 + 0.3, dg(x0) * 0.3 + g(x0), SERIES[1])
    seg(svg, xm2, ym2, full, 0, full, g(full), INK_SOFT, dash=True)
    seg(svg, xm2, ym2, half, 0, half, g(half), INK_SOFT, dash=True)
    marker(svg, "circle", xm2(x0), ym2(g(x0)), SERIES[1], f"x₀ = 1.5, |f| = {abs(g(x0)):.3f}")
    marker(svg, "cross", xm2(full), ym2(g(full)), SERIES[7], f"full step: x = {full:.3f}, |f| = {abs(g(full)):.3f}")
    marker(svg, "diamond", xm2(half), ym2(g(half)), SERIES[2], f"half step: x = {half:.3f}, |f| = {abs(g(half)):.3f}")
    callout(svg, xm2(full), ym2(g(full)), f"full step: |f| grows {abs(g(x0)):.2f} → {abs(g(full)):.2f}, refused",
            dx=20, dy=50)
    callout(svg, xm2(half), ym2(g(half)), f"half step (λ = 1/2): |f| {abs(g(half)):.2f}, accepted", dx=20, dy=-70)
    svg.text(x2 + 8, y + 18, "The chain's corrector near D 245 does this in 769 unknowns:", size=11, fill=INK)
    svg.text(x2 + 8, y + 34, "its weakest directions have tiny slopes, so full steps overshoot.", size=11, fill=INK)
    out = svg.finish(footnote(svg, SRC + ["illustration: f(x) = x² − 2 and arctan x"], "", [], y + h + 44, REPO))
    return out, xs, (x0, full, half, abs(g(x0)), abs(g(full)), abs(g(half)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=HERE)
    a = ap.parse_args()
    out = {}
    out["explain_step.svg"] = step_svg()
    out["explain_layout.svg"], L = layout_svg()
    out["explain_domain.svg"] = domain_svg()
    out["explain_parts.svg"], parts, errs = parts_svg()
    out["explain_soa.svg"] = soa_svg(L)
    out["explain_gain.svg"], gw = gain_svg()
    out["explain_newton.svg"], nx, at = newton_svg()
    for name, s in out.items():
        with open(os.path.join(a.o, name), "w") as f:
            f.write(s)
    print(f"L = {L}; z in dd {16 * L} B, qd {32 * L} B, od {64 * L} B; J {8 * L * L / 2**20:.2f} MiB per part")
    print("parts of 1/3:", " ".join(f"{p:.3e}" for p in parts))
    print("relative error by parts:", ", ".join(f"{n}: {e:.2e}" for n, e in errs.items()))
    print("gain at the walls:", ", ".join(f"{D:g}: {g:.2e}" for D, g in gw))
    print("Newton on x^2 - 2 from 3:", ", ".join(f"{v:.12f}" for v in nx))
    print("arctan from 1.5: full step to {:.4f} (|f| {:.4f} -> {:.4f}, refused), half step to {:.4f} (|f| {:.4f})".format(
        at[1], at[3], at[4], at[2], at[5]))
    print("wrote", ", ".join(out))


if __name__ == "__main__":
    main()
