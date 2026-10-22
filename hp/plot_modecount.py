#!/usr/bin/env python3
"""The unstable-mode count as a picture: the lambda plane at D = 61 with the counting boxes, the zeros and poles of
det M(lambda) and the points where it was evaluated; and the two real growth rates against 1/D.

Data: data/modecount_results.jsonl (the box layout, windings, expected zeros and candidate poles of each count),
data/modecount_D61.0264_M32_evals.jsonl (the evaluation points), data/gamma_dd.jsonl and data/local/gamma_dd.jsonl
(lambda_phys = 1/gamma for the line). notes/MODE_COUNT.md has the method.

    cd hp && python3 plot_modecount.py [-o modecount.svg]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "solver"))
from plot_delta import BAND, INK, INK_SOFT, SERIES, Panel, Svg, marker  # noqa: E402

DATA = os.path.join(HERE, "data")


def jl(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name)) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "modecount.svg"))
    args = ap.parse_args()
    res = jl("modecount_results.jsonl")
    r = [x for x in res if x["tag"] == "D61.0264_M32"][-1]
    ev = jl("modecount_D61.0264_M32_evals.jsonl")
    cP, cG = SERIES[0], INK_SOFT

    W = 900
    svg = Svg(W)
    svg.text(78, 26, "Counting the unstable modes: one physical growth rate", size=15, fill=INK, weight="bold")
    svg.text(78, 46, "Zeros of the growth-mode determinant det M(λ), counted by the winding of its phase around each "
             "box (argument principle); hover for values", size=11)

    # panel 1: the lambda plane at D = 61
    y2, y1, w = r["y2"], r["y1"], r["omega"]
    p = Panel(svg, 78, 120, 470, 380, (-0.4, 7.0), (-5.6, 5.6), f"The λ plane at D = {r['D']:.2f} (M = {r['M']})",
              "Re λ (growth rate)", "Im λ")
    for name, b in r["boxes"].items():
        if name in ("full", "B1"):
            continue
        x0, x1, ya, yb = b["rect"]
        for (a, c) in ((ya, yb), (-yb, -ya)) if name in ("U", "U0") else ((ya, yb),):
            X0, X1, Y0, Y1 = p.sx(x0), p.sx(x1), p.sy(c), p.sy(a)
            svg.add(f'<rect x="{X0:.1f}" y="{Y0:.1f}" width="{X1 - X0:.1f}" height="{Y1 - Y0:.1f}" fill="{BAND}" '
                    f'fill-opacity="0.6" stroke="{INK_SOFT}" stroke-width="1"><title>box {name}: winding of det M '
                    f'{round(b.get("winding_det", 0)) if "winding_det" in b else ""}</title></rect>')
    for name, b in r["boxes"].items():
        if name in ("full", "B1", "U0"):
            continue
        x0, x1, ya, yb = b["rect"]
        wd = len(b.get("expected_zeros", [])) - len(b.get("candidate_poles", b.get("poles", [])))
        txt = f"{wd:+d}" if wd else "0"
        xm = (p.sx(x0) + p.sx(x1)) / 2
        ym = p.sy(yb) + 16 if name != "U" else p.sy((ya + yb) / 2) - 14
        svg.text(xm, ym, txt, size=12, fill=INK, anchor="middle", weight="bold")
        if name == "U":                  # its mirror image below the real axis (det M(conj λ) = conj det M(λ))
            svg.text(xm, p.sy(-(ya + yb) / 2) + 26, txt, size=12, fill=INK, anchor="middle", weight="bold")
    svg.add(f'<g opacity="0.55">' + "".join(
        f'<circle cx="{p.sx(e["re"]):.1f}" cy="{p.sy(e["im"]):.1f}" r="1.6" fill="{INK_SOFT}"/>'
        f'<circle cx="{p.sx(e["re"]):.1f}" cy="{p.sy(-e["im"]):.1f}" r="1.6" fill="{INK_SOFT}"/>'
        for e in ev if e["re"] <= r["aR"] + 1e-9 and abs(e["im"]) <= y2 + 1e-9) + "</g>")
    for q in r["boxes"]["full"].get("candidate_poles", r["boxes"]["full"].get("poles", [])):
        X, Y, s = p.sx(q[0]), p.sy(q[1]), 4.5
        svg.add(f'<g><path d="M{X - s:.1f},{Y - s:.1f}L{X + s:.1f},{Y + s:.1f}M{X - s:.1f},{Y + s:.1f}L{X + s:.1f},'
                f'{Y - s:.1f}" stroke="{SERIES[1]}" stroke-width="2"/><title>pole of det M at λ = {q[0]:.3f} '
                f'{q[1]:+.3f}i (a horizon-series resonance)</title></g>')
    marker(svg, p.sx(0), p.sy(0), cG, "λ = 0: shifting the solution in time (marginal, not unstable)", r=5)
    marker(svg, p.sx(r["lam_g"]), p.sy(0), cG, f"λ = {r['lam_g']:.4f}: gauge mode 1 + ⟨c₁⟩", hollow=True, r=5)
    marker(svg, p.sx(r["lam_f"]), p.sy(0), cP, f"λ = {r['lam_f']:.6f}: the physical unstable mode, γ = 1/λ", r=6)
    svg.text(p.sx(r["lam_f"]) + 6, p.sy(0) + 34, "λ_phys", size=11, fill=INK)
    svg.text(p.sx(r["lam_g"]), p.sy(0) + 34, "gauge", size=11, fill=INK, anchor="middle")
    svg.text(p.sx(0) - 4, p.sy(0) + 34, "0", size=11, fill=INK, anchor="end")
    p.key("dot", cP, "physical mode")
    p.key("hollow", cG, "gauge")
    svg.text(p.kx, p.py - 20, "×  pole", size=10, fill=SERIES[1])

    # panel 2: the real rates against 1/D
    lam = {}
    for f in ("gamma_dd.jsonl", os.path.join("local", "gamma_dd.jsonl")):
        for x in jl(f):
            if x.get("lam") and x.get("F", 1) <= 1e-20:
                lam[round(x["D"], 4)] = x["lam"]
    counts = {}
    for x in res:
        if x["M"] == 32 and "_" not in x["tag"].replace("_M32", ""):
            counts[round(x["D"], 4)] = x
    q = Panel(svg, 640, 120, 230, 380, (0, 0.13), (0.5, 2.4), "Real growth rates against 1/D", "1/D", "λ")
    svg.line(q.px, q.sy(2), q.px + q.pw, q.sy(2), INK_SOFT, width=1.2, dash="6 4")
    svg.text(q.px + q.pw - 2, q.sy(2) - 6, "λ∞ = 2 (γ∞ = 1/2)", size=10, fill=INK, anchor="end")
    Ds = sorted(lam)
    q.curve([1 / d for d in Ds], [lam[d] for d in Ds], cP, width=2)
    for d, x in sorted(counts.items()):
        marker(svg, q.sx(1 / d), q.sy(x["lam_f"]), cP, f"D = {d}: λ_phys = {x['lam_f']:.6f}; counted, nothing else "
               "in the box", r=5)
        marker(svg, q.sx(1 / d), q.sy(x["lam_g"]), cG, f"D = {d}: gauge λ = {x['lam_g']:.4f}", hollow=True, r=5)
    svg.text(q.sx(0.122), q.sy(2.2317) - 10, "λ_phys", size=10.5, fill=INK, anchor="end")
    svg.text(q.sx(0.122), q.sy(0.6573) + 20, "gauge", size=10.5, fill=INK, anchor="end")
    open(args.o, "w").write(svg.finish(120 + 380 + 56))
    print("wrote", args.o, "counted D:", sorted(counts))


if __name__ == "__main__":
    main()
