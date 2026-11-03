#!/usr/bin/env python3
"""What collocation is, on a toy equation, in three panels (an illustration computed on the spot, not project data).
Left: the equation p' = -p + cos u on [0, 1] with p(0) = 1 is replaced by its values at N Chebyshev points; the
polynomial through the values supplies p' there (the differentiation matrix), the equation is demanded at every point,
the boundary value at u = 0. A wrong guess leaves a mismatch (the residual) at each point; Newton drives it to zero.
Middle: the error of the solution against the exact one falls geometrically with the number of points. Right: why the
method cannot impose analyticity at a singular point: u^21, the kind of non-analytic solution the horizon equation
admits at D = 40, and the polynomial through 16 Chebyshev points on [0, 0.3] are indistinguishable.

    cd hp && python3 plot_collocation_idea.py [-o collocation_idea.svg]
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_graphs import GRID, INK, INK_SOFT, SERIES, Svg, axes, better_hint, callout, caption, footnote, host_specs, legend, marker  # noqa: E402,E501

REPO = "https://github.com/nmoinvaz/choptuik-large-d"


def cheb(N, a, b):
    """N Chebyshev-Lobatto points on [a, b] (increasing) and d/du on them"""
    j = np.arange(N)
    t = np.cos(np.pi * j / (N - 1))
    c = np.where((j == 0) | (j == N - 1), 2.0, 1.0) * (-1.0) ** j
    T = np.tile(t, (N, 1)).T
    D = np.outer(c, 1 / c) / (T - T.T + np.eye(N))
    D = D - np.diag(D.sum(axis=1))
    u = (a + b) / 2 - (b - a) / 2 * t
    return u, D * (-2 / (b - a))


def exact(u):
    return (np.sin(u) + np.cos(u)) / 2 + np.exp(-u) / 2


def solve(N):
    """collocation for p' = -p + cos u, p(0) = 1: the linear system (D + I) p = cos u at the points u > 0, p(0) = 1"""
    u, D = cheb(N, 0.0, 1.0)
    A = D + np.eye(N)
    rhs = np.cos(u)
    A[0] = 0; A[0, 0] = 1; rhs[0] = 1.0
    return u, D, np.linalg.solve(A, rhs)


def interp(u, p, x):
    """the polynomial through (u, p) evaluated at x (Lagrange)"""
    out = np.zeros_like(x)
    for i in range(len(u)):
        li = np.ones_like(x)
        for j in range(len(u)):
            if j != i:
                li *= (x - u[j]) / (u[i] - u[j])
        out += p[i] * li
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "collocation_idea.svg"))
    a = ap.parse_args()
    svg = Svg(1180)
    y = caption(svg, 50, 40, "Collocation: demand the equation at a few points, let a polynomial fill the gaps",
                "left: p′ = −p + cos u on [0, 1], five points; middle: how the error falls with the number of points; right: what the method cannot tell apart")
    # ---- left panel
    X, W, H = 50, 330, 330
    xm, ym = axes(svg, X, y, W, H, (0, 1), (0.6, 1.05), "u", "p(u)", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    xs = np.linspace(0, 1, 200)
    path = " ".join(f"{'M' if i == 0 else 'L'}{xm(x):.1f},{ym(exact(x)):.1f}" for i, x in enumerate(xs))
    svg.add(f'<path d="{path}" fill="none" stroke="{SERIES[-1]}" stroke-width="2"/>')
    N = 5
    u, D, p = solve(N)
    guess = np.full(N, 1.0); guess[0] = 1.0                      # a wrong guess: p = 1 everywhere
    pg = interp(u, guess, xs)
    path = " ".join(f"{'M' if i == 0 else 'L'}{xm(x):.1f},{ym(v):.1f}" for i, (x, v) in enumerate(zip(xs, pg)))
    svg.add(f'<path d="{path}" fill="none" stroke="{SERIES[1]}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    # the residual of the guess at the points, as bars: r = D guess + guess - cos u (not at u = 0, the boundary row)
    r = D @ guess + guess - np.cos(u)
    for i in range(1, N):
        svg.line(xm(u[i]), ym(guess[i]), xm(u[i]), ym(guess[i] - 0.25 * r[i]), SERIES[1], width=3)
    for i in range(N):
        marker(svg, "diamond", xm(u[i]), ym(guess[i]), SERIES[1], f"guess at u = {u[i]:.3f}: 1 (residual {r[i]:+.2f})")
    ps = interp(u, p, xs)
    path = " ".join(f"{'M' if i == 0 else 'L'}{xm(x):.1f},{ym(v):.1f}" for i, (x, v) in enumerate(zip(xs, ps)))
    svg.add(f'<path d="{path}" fill="none" stroke="{SERIES[0]}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    for i in range(N):
        marker(svg, "circle", xm(u[i]), ym(p[i]), SERIES[0], f"solved value at u = {u[i]:.3f}: {p[i]:.5f} (exact {exact(u[i]):.5f})")
    for i in range(N):
        svg.line(xm(u[i]), ym(0.6), xm(u[i]), ym(0.6) - 8, INK, width=1.2)
    svg.text(xm(0.5), ym(0.6) - 12, "the points (Chebyshev: denser at the ends)", size=9, anchor="middle")
    callout(svg, xm(u[3]), ym(guess[3]), "a wrong guess: the equation fails by this much", dx=-60, dy=95)
    callout(svg, xm(u[3]), ym(p[3]), "after Newton: it holds at every point", dx=-30, dy=80)
    legend(svg, X + W, y - 10, [("exact", SERIES[-1]), ("solved, 5 points", SERIES[0]), ("a guess", SERIES[1])])
    # ---- middle panel: error against N
    X2, W2 = 450, 300
    Ns = list(range(3, 21))
    errs = []
    for n in Ns:
        u, D, p = solve(n)
        errs.append(max(float(np.abs(interp(u, p, xs) - exact(xs)).max()), 1e-16))
    xm2, yl = axes(svg, X2, y, W2, H, (2, 21), (-16, 0), "number of points", "largest error of the solution (log₁₀)", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    path = " ".join(f"{'M' if i == 0 else 'L'}{xm2(n):.1f},{yl(math.log10(e)):.1f}" for i, (n, e) in enumerate(zip(Ns, errs)))
    svg.add(f'<path d="{path}" fill="none" stroke="{SERIES[0]}" stroke-width="1.6"/>')
    for n, e in zip(Ns, errs):
        marker(svg, "circle", xm2(n), yl(math.log10(e)), SERIES[0], f"{n} points: error {e:.1e}")
    callout(svg, xm2(12), yl(math.log10(errs[9])), "one point, one digit", dx=10, dy=-100)
    better_hint(svg, X2, y, W2, H, where="right", label="accurate", reverse=True)
    # ---- right panel: u^21 and its interpolant on [0, 0.3]
    X3, W3 = 870, 270
    s = 21
    u0 = 0.3
    uu, _ = cheb(16, 0.0, u0)
    f = (uu / u0) ** s
    xs3 = np.linspace(0, u0, 300)
    g = interp(uu, f, xs3)
    diff = float(np.abs(g - (xs3 / u0) ** s).max())
    xm3, ym3 = axes(svg, X3, y, W3, H, (0, 0.3), (-0.05, 1.05), "u (distance from the horizon)", "u²¹ scaled to 1 at u = 0.3", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")
    path = " ".join(f"{'M' if i == 0 else 'L'}{xm3(x):.1f},{ym3((x / u0) ** s):.1f}" for i, x in enumerate(xs3))
    svg.add(f'<path d="{path}" fill="none" stroke="{SERIES[-1]}" stroke-width="2.5"/>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{xm3(x):.1f},{ym3(v):.1f}" for i, (x, v) in enumerate(zip(xs3, g)))
    svg.add(f'<path d="{path}" fill="none" stroke="{SERIES[1]}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    for x, v in zip(uu, f):
        marker(svg, "circle", xm3(x), ym3(v), SERIES[1], f"point u = {x:.4f}")
    callout(svg, xm3(0.27), ym3((0.27 / u0) ** s), f"16-point polynomial: within {diff:.0e} of u²¹", dx=-20, dy=-120)
    svg.text(X3 + 6, y + H - 60, "flat to 20 derivatives at u = 0", size=10)
    legend(svg, X3 + W3, y - 10, [("u²¹", SERIES[-1]), ("its interpolant", SERIES[1])])
    height = footnote(svg, ["an illustration computed by hp/plot_collocation_idea.py (not project data)"], host_specs(), [], y + H + 50, REPO)
    open(a.o, "w").write(svg.finish(height))
    print("wrote", a.o)
    print("| points | largest error |")
    print("|---|---|")
    for n, e in zip(Ns, errs):
        print(f"| {n} | {e:.1e} |")
    print(f"u^21 on [0, 0.3] against its 16-point interpolant: largest difference {diff:.1e}")


if __name__ == "__main__":
    main()
