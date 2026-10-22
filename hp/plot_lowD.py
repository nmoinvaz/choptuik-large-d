#!/usr/bin/env python3
"""Below D = 4: the double-double points and the maximum of Delta, the bias of the 47-mode f64 points there, and the
Fourier modes a 1e-10 Delta needs as D falls toward 3 (notes/LOW_D.md).

Data: data/lowD_dd.jsonl and data/lowD_dd_local_max.jsonl (double-double, M = 80), ../solver/data/scan_down_kc47.jsonl
(f64, 47 modes), the maximum from lowD_max.py's fits, Ecker et al. 2026's maximum (SciPost Phys. 20, 176, Eq. 70), and
the mode estimate of lowD_cost.py (its four model variants as a band; an estimate, not a bound).

    cd hp && python3 plot_lowD.py [-o lowD.svg]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "solver"))
sys.path.insert(0, HERE)
from plot_delta import INK, INK_SOFT, SERIES, Panel, Svg, marker  # noqa: E402
import lowD_cost as cost  # noqa: E402

EEGJ_MAX = (3.755726, 3.466772)     # Ecker, Ecker, Grumiller & Jechtl 2026, Eq. (70), each +- 1e-6


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "lowD.svg"))
    args = ap.parse_args()
    dd = {}
    for f in ("lowD_dd.jsonl", "lowD_dd_local_max.jsonl"):
        for r in jl(os.path.join(HERE, "data", f)):
            if r.get("M") == 80:
                dd[round(r["D"], 6)] = r["Delta"]
    f64 = {round(r["D"], 6): r["Delta"] for r in jl(os.path.join(HERE, "..", "solver", "data", "scan_down_kc47.jsonl"))}
    D3 = np.array(sorted(dd))
    sel = D3[(D3 >= 3.72) & (D3 <= 3.80)]              # the quartic of lowD_max.py (3.72-3.80)
    c = np.polyfit(sel - 3.75, [dd[d] for d in sel], 4)
    r0 = [x.real + 3.75 for x in np.roots(np.polyder(c)) if abs(x.imag) < 1e-12 and 3.72 < x.real + 3.75 < 3.78][0]
    dmax = (r0, float(np.polyval(c, r0 - 3.75)))
    cD, cF, cM = SERIES[0], SERIES[1], SERIES[2]

    W = 900
    svg = Svg(W)
    svg.text(78, 26, "Below D = 4: the maximum of Δ, and why the computation stops near D = 3.4", size=15, fill=INK,
             weight="bold")
    svg.text(78, 46, "Double-double points at M = 80 τ modes against the 47-mode f64 points; hover for values", size=11)

    p = Panel(svg, 78, 120, 300, 340, (3.62, 4.04), (3.440, 3.470), "Δ near its maximum", "D", "Δ", yfmt="{:.3f}")
    xs = np.linspace(3.70, 4.0, 200)
    for d, v in sorted(f64.items()):
        marker(svg, p.sx(d), p.sy(v), cF, f"f64, 47 modes: D = {d}, Δ = {v:.9f}", hollow=True, r=4.5)
    for d, v in sorted(dd.items()):
        marker(svg, p.sx(d), p.sy(v), cD, f"double-double, M = 80: D = {d}, Δ = {v:.10f}", r=4)
    marker(svg, p.sx(dmax[0]), p.sy(dmax[1]), cM, f"maximum (quartic): D = {dmax[0]:.6f}, Δ = {dmax[1]:.8f}",
           shape="diamond", r=4)
    svg.text(p.sx(dmax[0]) + 4, p.sy(dmax[1]) - 14, "maximum: D = 3.7557, Δ = 3.46677", size=10.5, fill=INK)
    p.key("dot", cD, "double-double")
    p.key("hollow", cF, "f64, 47 modes")

    q = Panel(svg, 448, 120, 160, 340, (3.68, 3.82), (0, 6), "f64 minus double-double", "D", "ΔΔ (×10⁻⁶)")
    for d in sorted(set(dd) & set(f64)):
        v = (f64[d] - dd[d]) * 1e6
        marker(svg, q.sx(d), q.sy(v), cF, f"D = {d}: f64 − double-double = {v:.2f}×10⁻⁶", r=4.5)
    svg.line(q.px, q.sy(0), q.px + q.pw, q.sy(0), INK_SOFT, width=1.2, dash="6 4")

    r = Panel(svg, 678, 120, 190, 340, (3.0, 4.0), (0, 460), "τ modes for a 10⁻¹⁰ Δ", "D", "modes")
    A4 = cost.calibrate(55, 2.6e-8, cost.BETA[0][1])
    Ds = np.linspace(3.05, 4.0, 40)
    lo, hi = [], []
    for D in Ds:
        f1 = cost.F1_of(D)
        ks = [cost.modes_needed(A, cost.beta_of(D, m), 1e-10) for m in ("linear", "power")
              for A in (A4, A4 * (f1 / 16.23) ** 2)]
        lo.append(min(ks)); hi.append(max(ks))
    pts = [(r.sx(d), r.sy(v)) for d, v in zip(Ds, hi)] + [(r.sx(d), r.sy(v)) for d, v in zip(Ds[::-1], lo[::-1])]
    svg.add('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="{cD}" fill-opacity="0.25"/>')
    for lim, txt in ((63, "f64 kernel: 63"), (128, "double-double kernel: 128")):
        svg.line(r.px, r.sy(lim), r.px + r.pw, r.sy(lim), INK_SOFT, width=1.2, dash="6 4")
        svg.text(r.px + r.pw - 2, r.sy(lim) - 5, txt, size=10, fill=INK, anchor="end")
    svg.text(r.sx(3.08), r.sy(430), "estimate", size=10, fill=INK)
    open(args.o, "w").write(svg.finish(120 + 340 + 56))
    print("wrote", args.o, "max", dmax, "Ecker et al. minus ours:", EEGJ_MAX[0] - dmax[0], EEGJ_MAX[1] - dmax[1], "f64-dd (1e-6):",
          {d: round((f64[d] - dd[d]) * 1e6, 2) for d in sorted(set(dd) & set(f64))})


if __name__ == "__main__":
    main()
