#!/usr/bin/env python3
"""Markdown table of the CSS solution against D from the data files (matter/CSS.md, section 5).

    python3 css_table.py [BG.jsonl] [GAMMA.jsonl] [D1,D2,...]
"""
import sys, os, json

here = os.path.dirname(os.path.abspath(__file__))
bgf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "data", "css_bg_k1.jsonl")
gmf = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "data", "css_gamma_k1.jsonl")
bg = {r["D"]: r for r in (json.loads(l) for l in open(bgf))}
gm = {r["D"]: r for r in (json.loads(l) for l in open(gmf))} if os.path.exists(gmf) else {}
Ds = [float(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else sorted(bg)
print("| D | ω | Δ = 2π/ω | \\|f(0)\\| | $`x_h`$ | \\|f(x_h)\\| | Re μ | \\|∂J/∂ω\\| | λ | γ = 1/λ | (D − 3)γ | n(λ − 2) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for D in Ds:
    if D not in bg:
        continue
    r = bg[D]; n = D - 2
    g = gm.get(D)
    lam = "%.10f" % g["lam"] if g else ""
    gam = "%.10f" % g["gamma"] if g else ""
    mass = "%.6f" % ((D - 3) * g["gamma"]) if g else ""
    nl = "%.4f" % (n * (g["lam"] - 2)) if g else ""
    print("| %g | %.10f | %.8f | %.10f | %.9f | %.8f | %.2f | %.1e | %s | %s | %s | %s |"
          % (D, r["omega"], 6.283185307179586 / r["omega"], r["f0"], r["X"], r["absfh"], r["mu"][0], r["dJdomega"],
             lam, gam, mass, nl))
