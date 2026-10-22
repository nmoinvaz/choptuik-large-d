"""Collect the sigma-model DSS data (data/sigdss_*.jsonl) into the tables of matter/SIGMA_DSS.md.
usage: python sigdss_tables.py [eps ...]    (prints markdown tables of Delta, lambda, gamma against D for each eps,
and the D = 4 table against eta)."""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

recs = []
for f in sorted(glob.glob(os.path.join(DATA, "sigdss_*.jsonl"))):
    if "parity" in f or "count" in f or "roots" in f:
        continue
    for line in open(f):
        r = json.loads(line)
        r["file"] = os.path.basename(f)
        # growth rates from the first (secant) root finder that stopped at maxit are not converged: drop them
        if r.get("lam") and r.get("last_step") is not None and r["last_step"] > 1e-6 * r["lam"]:
            r["lam"] = None
        recs.append(r)


def best(D, eps, key=None, kc=None):
    c = [r for r in recs if abs(r["D"] - D) < 1e-6 and abs(r["eps"] - eps) < 1e-6 and (key is None or r.get(key))
         and (kc is None or r["kc"] == kc)]
    return c


def eta_s(eps):
    return "∞" if eps == 0 else f"{1 / eps:g}"


# D = 4 against eta
print("D = 4 (kc 15 unless noted)\n")
print("| η | Δ | Δ/2 | λ | γ = 1/λ | max abs u | constraint |")
print("|---|---|---|---|---|---|---|")
e4 = sorted({round(r["eps"], 6) for r in recs if abs(r["D"] - 4) < 1e-9})
for e in e4:
    c = best(4.0, e)
    if not c:
        continue
    lam = [r for r in c if r.get("lam")]
    r0 = c[0]
    ls = "; ".join(f"{r['lam']:.5f} (kc {r['kc']})" for r in lam) if lam else ""
    gs = "; ".join(f"{1 / r['lam']:.5f}" for r in lam) if lam else ""
    Ds = sorted({(r["kc"], round(r["Delta"], 8)) for r in c})
    print(f"| {eta_s(e)} | {', '.join(f'{d:.6f} (kc {k})' for k, d in Ds)} | {Ds[-1][1] / 2:.5f} | {ls} | {gs} | "
          f"{r0.get('umax', 0):.3f} | {r0.get('constraint', 0):.1e} |")

for e in [float(v) for v in sys.argv[1:]] or sorted({round(r["eps"], 6) for r in recs}):
    Ds = sorted({round(r["D"], 6) for r in recs if abs(r["eps"] - e) < 1e-6})
    print(f"\nη = {eta_s(e)}\n")
    print("| D | Δ | λ | γ | kc | constraint | max abs u | λ_gauge |")
    print("|---|---|---|---|---|---|---|---|")
    for D in Ds:
        c = best(D, e)
        lam = [r for r in c if r.get("lam")]
        for r in (lam if lam else c[-1:]):
            print(f"| {D:g} | {r['Delta']:.7f} | {r.get('lam') or '':{'.6f' if r.get('lam') else ''}} | "
                  f"{(1 / r['lam']) if r.get('lam') else '':{'.6f' if r.get('lam') else ''}} | {r['kc']} | "
                  f"{r['constraint']:.1e} | {r.get('umax', 0):.3f} | {r.get('lam_gauge', 0):.5f} |")
