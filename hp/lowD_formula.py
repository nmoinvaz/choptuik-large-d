#!/usr/bin/env python3
"""A formula for Delta(D) and gamma(D) below D = 4, built around D = 3, from Ecker, Ecker, Grumiller & Jechtl's
published values (github.com/EckerChristian/CriticalSpacetimeCrystal, plots/DeltaList.dat and plots/gamma.dat; not
included here) for 3.046 <= D <= 4, checked against our double-double points (data/lowD_dd.jsonl,
data/lowD_dd_local_max.jsonl, data/lowD_gamma.jsonl).

Forms compared, each in x = D - 3, fitted by least squares and judged on points they never saw (fit on D >= 3.10,
predict 3.046-3.10; fit on D <= 3.9, predict 3.9-4.0):
  power:    Delta = x^p (c0 + c1 x + ... + cK x^K), p fitted (nonlinear, by a scan in p);
  sqrt:     Delta = c0 + c1 x^(1/2) + c2 x + ... (a polynomial in sqrt x);
  log:      Delta = c0 + c1 ln x + c2 x + ... ;
  poly:     Delta = c0 + c1 x + ... (no special point at D = 3).
The polynomial in sqrt x wins for both; the chosen degrees (8 for Delta, 6 for gamma) are fitted on all of their
points and compared with our double-double values, which the fit never saw.

    git clone --depth 1 https://github.com/EckerChristian/CriticalSpacetimeCrystal ecc
    python3 lowD_formula.py ecc
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def table(path):
    out = []
    for line in open(path):
        f = line.split()
        if len(f) >= 2:
            try:
                out.append((float(f[0]), float(f[1])))
            except ValueError:
                pass
    d = {}
    for D, v in out:
        d[round(D, 9)] = v
    return np.array(sorted(d)), np.array([d[k] for k in sorted(d)])


def design(form, x, K, p=None):
    if form == "power":
        return np.stack([x ** (p + k) for k in range(K + 1)], 1)
    if form == "sqrt":
        return np.stack([x ** (k / 2) for k in range(K + 1)], 1)
    if form == "log":
        return np.stack([np.ones_like(x), np.log(x)] + [x ** k for k in range(1, K)], 1)
    return np.stack([x ** k for k in range(K + 1)], 1)


def fit(form, x, y, K):
    if form != "power":
        A = design(form, x, K)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        return c, None
    best = None
    for p in np.linspace(0.0, 0.6, 601):
        A = design(form, x, K, p)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = np.abs(A @ c - y).max()
        if best is None or r < best[0]:
            best = (r, c, p)
    return best[1], best[2]


def evalf(form, c, x, p=None):
    return design(form, x, len(c) - 1, p) @ c


def main():
    ecc = sys.argv[1] if len(sys.argv) > 1 else "ecc"
    D, Dl = table(os.path.join(ecc, "plots", "DeltaList.dat"))
    G, Gl = table(os.path.join(ecc, "plots", "gamma.dat"))
    m = D <= 4.0 + 1e-9
    D, Dl = D[m], Dl[m]
    mg = G <= 4.0 + 1e-9
    G, Gl = G[mg], Gl[mg]
    print(f"their Delta: {len(D)} points, D = {D.min():g}-{D.max():g}; gamma: {len(G)} points, D = {G.min():g}-{G.max():g}")

    for name, X, Y in (("Delta", D, Dl), ("gamma", G, Gl)):
        x = X - 3
        lo = X >= 3.10 if name == "Delta" else X >= 3.30
        hi = X <= 3.9
        print(f"\n{name}: held-out max error (fit on D >= {3.10 if name == 'Delta' else 3.30:g}, predict below; "
              "fit on D <= 3.9, predict above); max fit residual")
        for form in ("power", "sqrt", "log", "poly"):
            for K in (3, 4, 5, 6):
                c1, p1 = fit(form, x[lo], Y[lo], K)
                e1 = np.abs(evalf(form, c1, x[~lo], p1) - Y[~lo]).max()
                c2, p2 = fit(form, x[hi], Y[hi], K)
                e2 = np.abs(evalf(form, c2, x[~hi], p2) - Y[~hi]).max()
                c, p = fit(form, x, Y, K)
                r = np.abs(evalf(form, c, x, p) - Y).max()
                ps = f" p = {p:.3f}" if p is not None else ""
                print(f"  {form:5s} K = {K}: below {e1:.1e}, above {e2:.1e}; residual {r:.1e}{ps}")

    # the chosen forms, fitted on all of their points, against our double-double values
    KD, KG = 8, 6
    print(f"\nchosen: a polynomial in sqrt(D - 3), degree {KD} for Delta and {KG} for gamma, on all their points D <= 4")
    cD, _ = fit("sqrt", D - 3, Dl, KD)
    cG, _ = fit("sqrt", G - 3, Gl, KG)
    rD = np.abs(evalf("sqrt", cD, D - 3) - Dl).max()
    rG = np.abs(evalf("sqrt", cG, G - 3) - Gl).max()
    print("  Delta = " + " ".join(f"{v:+.6f}*s^{k}" for k, v in enumerate(cD)) + f"   (s = sqrt(D - 3); max residual {rD:.1e})")
    print("  gamma = " + " ".join(f"{v:+.6f}*s^{k}" for k, v in enumerate(cG)) + f"   (max residual {rG:.1e})")
    ours = {}
    for f in ("lowD_dd.jsonl", "lowD_dd_local_max.jsonl"):
        for line in open(os.path.join(HERE, "data", f)):
            r = json.loads(line)
            if r.get("M") == 80:
                ours[r["D"]] = r["Delta"]
    dif = [evalf("sqrt", cD, np.array([d - 3]))[0] - v for d, v in sorted(ours.items())]
    print("  against our double-double Delta at D = " + ", ".join(f"{d:g}" for d in sorted(ours)) + ": "
          + ", ".join(f"{x:+.1e}" for x in dif))
    for line in open(os.path.join(HERE, "data", "lowD_gamma.jsonl")):
        r = json.loads(line)
        if r.get("M") == 80:
            g = evalf("sqrt", cG, np.array([r["D"] - 3]))[0]
            print(f"  against our double-double gamma at D = {r['D']}: {g - r['gamma']:+.1e}")
    print("  the value at D = 3 (the constant term) is not determined by these data: across degrees "
          + ", ".join(f"{K}: {fit('sqrt', D - 3, Dl, K)[0][0]:.3f}" for K in range(6, 12)) + " (Delta); "
          + ", ".join(f"{K}: {fit('sqrt', G - 3, Gl, K)[0][0]:.3f}" for K in range(5, 10)) + " (gamma)")

if __name__ == "__main__":
    main()
