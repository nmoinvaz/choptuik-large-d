#!/usr/bin/env python3
"""The Newton variants of the double-double solver against D, from the local machine's benchmark (T42).

Five ways to build the Newton step, all converging to the same double-double solution (the residual is always
double-double; notes/LARGE_D_MIXED.md):
  dd     the exact double-double Jacobian and LU (the reference)
  f64    the Jacobian in f64 on SIMD lanes, f64 LU
  f64h   f64, with the horizon block's columns in double-double
  f64g   f64, with a GMRES correction loop on the exact double-double Jacobian-vector product
  f64hg  both
Data: data/bench_variants/bench_local.jsonl (bench_variants.py --repeat=3 --threads=4 on an otherwise idle i7-1185G7,
AVX-512, on AC power; 2026-10-23). Each point is the median whole-solve wall time of the three repeats; a variant that
did not converge in any repeat is drawn as a cross on the "failed" row. Left: seconds per solve (log scale). Right:
the speedup over dd. The resolution differs by D: M = 80 at D = 3.7, 48 at 8.25, 32 above.

    cd hp && python3 plot_bench_variants.py [-o bench_variants.svg]
"""
import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "solver"))
from plot_delta import GRID, INK, INK_SOFT, SERIES, SURFACE, Panel, Svg, esc, marker  # noqa: E402

VARIANTS = ["dd", "f64", "f64h", "f64g", "f64hg"]
LABELS = {"dd": "dd (reference)", "f64": "f64", "f64h": "f64 + dd horizon", "f64g": "f64 + GMRES",
          "f64hg": "f64 + dd horizon + GMRES"}


def load(path):
    runs = [json.loads(x) for x in open(path) if x.strip()]
    out = {}
    for r in runs:
        out.setdefault((r["D"], r["M"], r["variant"]), []).append(r)
    table = {}
    for (D, M, v), rs in out.items():
        ok = [r for r in rs if r.get("converged")]
        table[(D, v)] = dict(M=M, n=len(rs), n_ok=len(ok), wall=statistics.median(r["wall"] for r in ok) if ok else None,
                             its=statistics.median(r["its"] for r in ok) if ok else None)
    return table


class LogPanel(Panel):
    """Panel with a log10 y axis: yr in log10 units, ticks at the given values."""

    def __init__(self, svg, px, py, pw, ph, xr, yr, title, xlabel, ylabel, yticks):
        super().__init__(svg, px, py, pw, ph, xr, yr, title, xlabel, ylabel, yfmt="")
        for v in yticks:
            y = self.sy(math.log10(v))
            svg.line(px, y, px + pw, y, GRID)
            svg.text(px - 8, y + 4, f"{v:g}", size=10, anchor="end")


def cross(svg, x, y, color, title, r=4.5):
    svg.add(f'<g><path d="M{x - r:.1f},{y - r:.1f}L{x + r:.1f},{y + r:.1f}M{x - r:.1f},{y + r:.1f}L{x + r:.1f},'
            f'{y - r:.1f}" stroke="{color}" stroke-width="2"/><title>{esc(title)}</title></g>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "bench_variants.svg"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "bench_variants", "bench_local.jsonl"))
    a = ap.parse_args()
    t = load(a.data)
    Ds = sorted({D for D, _ in t})
    color = dict(zip(VARIANTS, SERIES))
    svg = Svg(980)
    xr = (0, 64)
    # left: seconds per solve, log scale, with a "failed" row under the axis range
    lo, hi = math.log10(0.5), math.log10(200)
    L = LogPanel(svg, 70, 145, 380, 300, xr, (lo - 0.25, hi), "Seconds per solve (median of 3)", "D",
                 "seconds (log scale)", [0.5, 1, 2, 5, 10, 20, 50, 100, 200])
    yf_L = L.sy(lo - 0.15)
    svg.text(L.px - 8, yf_L + 4, "failed", size=10, anchor="end")
    # right: speedup over dd
    R = Panel(svg, 560, 145, 380, 300, xr, (-1, 9), "Speedup over the double-double Jacobian", "D", "speedup (x)",
              yfmt="{:g}")
    svg.add(f'<rect x="{R.px:.1f}" y="{R.sy(0) + 1:.1f}" width="{R.pw:.1f}" height="{R.sy(-1) - R.sy(0) - 1:.1f}" '
            f'fill="{SURFACE}"/>')
    svg.text(R.px - 8, R.sy(-0.5) + 4, "failed", size=10, anchor="end")
    R.svg.line(R.px, R.sy(1), R.px + R.pw, R.sy(1), INK_SOFT, dash="4 3")
    for v in VARIANTS:
        c = color[v]
        pts_L, pts_R = [], []
        for D in Ds:
            e = t.get((D, v))
            if e is None:
                continue
            ref = t[(D, "dd")]["wall"]
            if e["wall"] is None:
                jit = (VARIANTS.index(v) - 2) * 1.5      # failures of several variants at one D sit side by side
                cross(svg, L.sx(D + jit), yf_L, c, f"{v} at D = {D} (M = {e['M']}): did not converge ({e['n']} repeats)")
                cross(svg, R.sx(D + jit), R.sy(-0.5), c, f"{v} at D = {D} (M = {e['M']}): did not converge")
                pts_L.append(None)
                pts_R.append(None)
                continue
            sp = ref / e["wall"]
            tip = f"{v} at D = {D} (M = {e['M']}): {e['wall']:.1f} s, {sp:.1f}x dd, {e['its']:g} iterations"
            pts_L.append((L.sx(D), L.sy(math.log10(e["wall"])), tip))
            pts_R.append((R.sx(D), R.sy(sp), tip))
        for pts in (pts_L, pts_R):
            seg = []
            for p in pts + [None]:
                if p is None:
                    if len(seg) > 1:
                        svg.path([(x, y) for x, y, _ in seg], c, width=2)
                    seg = []
                else:
                    seg.append(p)
            for p in pts:
                if p is not None:
                    marker(svg, p[0], p[1], c, p[2], r=4)
    # one legend row for both panels
    kx = 70
    for v in VARIANTS:
        svg.line(kx, 74, kx + 22, 74, color[v], width=2)
        marker(svg, kx + 11, 74, color[v], LABELS[v], r=3.5)
        svg.text(kx + 28, 78, LABELS[v], size=11)
        kx += 28 + 6.3 * len(LABELS[v]) + 26
    svg.text(70, 30, "Newton variants of the double-double solver against D", size=14, fill=INK, weight="bold")
    svg.text(70, 50, "local machine (i7-1185G7, AVX-512, 4 threads), 3 repeats each; M = 80 at D = 3.7, 48 at 8.25, "
             "32 above; every converged run gives the same Delta to 1e-18", size=11)
    open(a.o, "w").write(svg.finish(505))
    print(f"wrote {a.o}")
    for D in Ds:
        print(f"D {D:6.2f}: " + "  ".join(
            f"{v} {'FAIL' if t[(D, v)]['wall'] is None else '%.1fs' % t[(D, v)]['wall']}" for v in VARIANTS))


if __name__ == "__main__":
    main()
