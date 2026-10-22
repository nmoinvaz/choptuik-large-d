#!/usr/bin/env python3
"""The critical exponent gamma(D) for every matter model computed here, against 1/D, with the value 1/2.

Data (all from files in the repository):
- massless scalar, echoing: the double-double growth rates (hp/data/gamma_dd.jsonl and hp/data/local/gamma_dd.jsonl,
  |F| <= 1e-20), the double-double D = 3.7 and 4 points (hp/data/lowD_gamma.jsonl, M = 80) and the f64 points below
  D = 8.25 (solver/rational_fit.gamma_data);
- axion-dilaton, phase-rotating self-similar (CSS): solver/matter/data/css_gamma_k1.jsonl (matter/CSS.md);
- l = 1 scalar multiplet, echoing: D = 4 from l1_kc23_gamma.jsonl, D >= 5 from l1_kc23r.jsonl (matter/ELL.md);
- sigma model, echoing, at couplings eta = 1 and 0.3: solver/matter/data/sigdss_*.jsonl as collected by
  sigdss_tables.py, the highest k_c with a converged growth rate at each D; D >= 10 is marked preliminary
  (matter/SIGMA_DSS.md section 6);
- sigma model, self-similar branch at eta = 0.14: sigma_path_eta0.14.jsonl up to the fold (sigma_fold_eta0.14.jsonl,
  matter/SIGMA.md).

    python3 plot_matter.py [-o matter.svg]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_delta import INK, INK_SOFT, SERIES, Panel, Svg, marker  # noqa: E402

MDATA = os.path.join(HERE, "matter", "data")
HP = os.path.join(HERE, "..", "hp", "data")


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def scalar():
    pts = {}
    cwd = os.getcwd()
    os.chdir(os.path.join(HERE, "data"))
    try:
        from rational_fit import gamma_data
        D, g, _ = gamma_data()
    finally:
        os.chdir(cwd)
    for d, v in zip(D, g):
        if d < 8.25:
            pts[round(float(d), 6)] = float(v)
    for r in jl(os.path.join(HP, "lowD_gamma.jsonl")):
        if r.get("M") == 80:
            pts[round(r["D"], 6)] = r["gamma"]
    for f in ("gamma_dd.jsonl", os.path.join("local", "gamma_dd.jsonl")):
        for r in jl(os.path.join(HP, f)):
            if r.get("lam") and r.get("F", 1) <= 1e-20:
                pts[round(r["D"], 6)] = r["gamma"]
    return sorted(pts.items())


def axion_dilaton():
    return sorted((r["D"], r["gamma"]) for r in jl(os.path.join(MDATA, "css_gamma_k1.jsonl")))


def ell1():
    g4 = [r for r in jl(os.path.join(MDATA, "l1_kc23_gamma.jsonl")) if r["D"] == 4.0][-1]
    pts = {4.0: 1 / g4["lam"]}
    for r in jl(os.path.join(MDATA, "l1_kc23r.jsonl")):
        if r.get("lam"):
            pts[r["D"]] = 1 / r["lam"]
    return sorted(pts.items())


def sigma_dss(eps):
    best = {}
    for f in sorted(glob.glob(os.path.join(MDATA, "sigdss_*.jsonl"))):
        if any(s in f for s in ("parity", "count", "roots", "l1_reference")):
            continue
        for r in jl(f):
            if not r.get("lam") or abs(r.get("eps", -1) - eps) > 1e-4:
                continue
            if r.get("last_step") is not None and r["last_step"] > 1e-3:     # unconverged secant roots (sigdss_tables.py)
                continue
            D = round(r["D"], 6)
            if D not in best or r["kc"] >= best[D]["kc"]:
                best[D] = r
    return sorted((D, 1 / r["lam"]) for D, r in best.items())


def sigma_css():
    pts = [(r["D"], r["gamma"]) for r in jl(os.path.join(MDATA, "sigma_path_eta0.14.jsonl"))
           if "gamma" in r and abs(r.get("eta", 0) - 0.14) < 1e-9]
    fold = jl(os.path.join(MDATA, "sigma_fold_eta0.14.jsonl"))
    top = max(fold, key=lambda r: r["D"])
    for r in fold:                       # the branch from the last path point up to the fold
        if r["D"] > max(p[0] for p in pts) and fold.index(r) <= fold.index(top):
            pts.append((r["D"], r["gamma"]))
    return sorted(pts), (top["D"], top["gamma"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(HERE, "matter.svg"))
    args = ap.parse_args()
    S, AD, L1 = scalar(), axion_dilaton(), ell1()
    s1, s03 = sigma_dss(1.0), sigma_dss(1 / 0.3)
    SC, fold = sigma_css()
    cS, cA, cL, cE = SERIES[0], SERIES[2], SERIES[1], SERIES[4]

    W = 900
    svg = Svg(W)
    svg.text(78, 26, "The critical exponent γ across matter models, against 1/D", size=15, fill=INK, weight="bold")
    svg.text(78, 46, "Infinite dimension is the left edge; hover a mark for its value. Hollow: preliminary (σ-model, "
             "D ≥ 10).", size=11)
    p = Panel(svg, 78, 118, 790, 380, (0, 0.34), (0.1, 0.55), "γ = 1/λ for each model's critical solution",
              "1/D", "γ")
    y = p.sy(0.5)
    svg.line(p.px, y, p.px + p.pw, y, INK_SOFT, width=1.3, dash="6 4")
    svg.text(p.sx(0.335), y + 16, "1/2: the large-D value where the r² shift fixes it (scalar, axion–dilaton)", size=10,
             fill=INK, anchor="end")

    def series(pts, color, name, hollow_from=None, r=3.2, line=True):
        if line:
            p.curve([1 / d for d, _ in pts], [g for _, g in pts], color, width=1.6)
        for d, g in pts:
            if p.inside(1 / d, g):
                marker(svg, p.sx(1 / d), p.sy(g), color, f"{name}, D = {d:g}: γ = {g:.6f}",
                       hollow=hollow_from is not None and d >= hollow_from, r=r)

    series(S, cS, "scalar (echoing)", r=2.6)
    series(AD, cA, "axion–dilaton (self-similar)", r=2.6)
    series(L1, cL, "ℓ = 1 multiplet (echoing)", r=3.6)
    series(s1, cE, "σ-model, η = 1 (echoing)", hollow_from=10, r=3.6)
    series(s03, cE, "σ-model, η = 0.3 (echoing)", hollow_from=10, r=3.6)
    p.curve([1 / d for d, _ in SC], [g for _, g in SC], SERIES[3], width=2, dash="5 3")
    marker(svg, p.sx(1 / fold[0]), p.sy(fold[1]), SERIES[3], f"σ-model self-similar branch ends at a fold: "
           f"D = {fold[0]:.4f}, γ = {fold[1]:.5f}", shape="diamond", r=4)

    def label(pts, text, dy=0, color=INK, at=-1, anchor="start", dx=8):
        d, g = pts[at]
        svg.text(p.sx(1 / d) + dx, p.sy(g) + 4 + dy, text, size=10.5, fill=color, anchor=anchor)

    x4 = p.sx(0.25) + 10
    for g, text in ((0.392, "scalar, echoing (D = 3.7–61)"), (0.2641, "axion–dilaton, self-similar (D = 4–80)"),
                    (0.1878, "σ-model, self-similar, η = 0.14"), (0.1177, "ℓ = 1 multiplet and σ-model, echoing")):
        svg.text(x4, p.sy(g) + 4, text, size=10.5, fill=INK)
    label(s1, "σ-model η = 1", dy=-10, at=-1, anchor="end", dx=-8)
    label(s03, "σ-model η = 0.3", dy=14, at=-1, anchor="end", dx=-8)
    svg.text(p.sx(1 / fold[0]) + 10, p.sy(fold[1]) - 6, f"σ-model self-similar (η = 0.14): ends at a fold, D = "
             f"{fold[0]:.3f}", size=10.5, fill=INK)
    p.key("dot", cS, "scalar")
    p.key("dot", cA, "axion–dilaton")
    p.key("dot", cL, "ℓ = 1")
    p.key("dot", cE, "σ-model, echoing")
    p.key("line", SERIES[3], "σ-model, self-similar", dash="5 3")
    p.key("line", INK_SOFT, "γ = 1/2", dash="6 4")
    open(args.o, "w").write(svg.finish(118 + 380 + 56))
    print("wrote", args.o)
    for name, pts in (("scalar", S), ("axion-dilaton", AD), ("l=1", L1), ("sigma eta=1", s1), ("sigma eta=0.3", s03),
                      ("sigma CSS", SC)):
        print(f"{name:14s} {len(pts):3d} points, D {pts[0][0]:g}-{pts[-1][0]:g}, gamma {pts[0][1]:.5f}-{pts[-1][1]:.5f}")
    print("fold", fold)


if __name__ == "__main__":
    main()
