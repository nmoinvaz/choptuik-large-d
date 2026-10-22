"""Markdown table of the l = 1 results across D from the data files (D = 4 from the kc 39 / kc 23 runs, D >= 5 from the
kc 23 refinement l1_kc23r.jsonl, with the continuation's selection sensitivity |dDelta/dF| where available).
usage: python ell_table.py"""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
dat = lambda f: [json.loads(l) for l in open(os.path.join(HERE, "data", f))] if os.path.exists(os.path.join(HERE, "data", f)) else []

rows = {}
d4 = dat("l1_D4_kc39.jsonl")[-1]
g4 = [r for r in dat("l1_kc23_gamma.jsonl") if r["D"] == 4.0][-1]
rows[4.0] = dict(Delta=d4["Delta"], lam=g4["lam"], F=d4["F"], K=d4["constraint"], kc="39 (λ: 23)")
for r in dat("l1_kc23r.jsonl"):
    rows[r["D"]] = dict(Delta=r["Delta"], lam=r.get("lam"), F=r["F"], K=r["constraint"], kc=str(r["kc"]))
sens = {}
for f in ("march_l1_kc23m.jsonl",):
    for r in dat(f):
        sens[round(r["D"], 6)] = r.get("dDelta_dF")
print("| D | n | Δ | λ = 1/γ | γ | (D − 3)γ | resid. abs(F) | ψ-constraint | abs(dΔ/dF) | k_c |")
print("|---|---|---|---|---|---|---|---|---|---|")
for D in sorted(rows):
    r = rows[D]
    lam = r["lam"]
    s = sens.get(round(D, 6))
    print(f"| {D:g} | {D - 2:g} | {r['Delta']:.8f} | " + (f"{lam:.6f} | {1 / lam:.6f} | {(D - 3) / lam:.6f}" if lam else "– | – | –")
          + f" | {r['F']:.0e} | {r['K']:.0e} | " + (f"{s:.1e}" if s else "–") + f" | {r['kc']} |")
