#!/usr/bin/env python3
"""Fits of lambda(D), omega(D), |f(0)|(D) in powers of 1/n and their D = infinity limits (matter/CSS.md, section 7).

    python3 css_fit.py [BG.jsonl] [GAMMA.jsonl]

For each quantity Q: least-squares fits Q = Q_inf + sum_{k=1..K} c_k n^-k over the points with D >= Dmin, for a grid of
(K, Dmin); a fit is kept when it has at least K + 3 points.  The quoted limit is the median of the kept fits whose
truncation is "converged" (the change from K-1 to K is below the spread), with the full spread over (K, Dmin) as the
error bar.  For lambda the fits with lambda_inf = 2 fixed are compared (rms residual) with the free fits.
"""
import sys, os, json
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
bgf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "data", "css_bg_k1.jsonl")
gmf = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "data", "css_gamma_k1.jsonl")
bg = {r["D"]: r for r in (json.loads(l) for l in open(bgf))}
gm = {r["D"]: r for r in (json.loads(l) for l in open(gmf))} if os.path.exists(gmf) else {}


def fit(Ds, Q, K, fixed=None):
    n = np.asarray(Ds) - 2.0
    cols = [n ** -k for k in range(1, K + 1)]
    y = np.asarray(Q, float)
    if fixed is None:
        cols = [np.ones_like(n)] + cols
    else:
        y = y - fixed
    Amat = np.array(cols).T
    c, *_ = np.linalg.lstsq(Amat, y, rcond=None)
    rms = np.sqrt(np.mean((Amat @ c - y) ** 2))
    return (c[0] if fixed is None else fixed), c, rms


def scan(name, Ds, Q, Ks=(2, 3, 4, 5, 6, 7), Dmins=(5, 6, 8, 10, 12, 14, 16, 20), fixed=None):
    rows = []
    for Dmin in Dmins:
        sel = [i for i, D in enumerate(Ds) if D >= Dmin]
        if len(sel) < 4:
            continue
        for K in Ks:
            if len(sel) < K + 3:
                continue
            q0, c, rms = fit([Ds[i] for i in sel], [Q[i] for i in sel], K, fixed)
            rows.append((Dmin, K, q0, c, rms, len(sel)))
    return rows


def report(name, Ds, Q, fixed_compare=None, digits=8):
    rows = scan(name, Ds, Q)
    print("\n%s: %d points, D = %g .. %g" % (name, len(Ds), min(Ds), max(Ds)))
    print("  Dmin  K  npts   %s_inf          c1          rms" % name)
    for Dmin, K, q0, c, rms, npts in rows:
        print("  %4g  %d  %4d   %.*f   %+.5f   %.1e" % (Dmin, K, npts, digits, q0, c[1] if len(c) > 1 else np.nan, rms))
    vals = np.array([r[2] for r in rows])
    # "converged" rows: K >= 3 and |q0(K) - q0(K-1)| small relative to the spread over all
    best = {}
    for Dmin, K, q0, c, rms, npts in rows:
        best.setdefault(Dmin, {})[K] = q0
    conv = []
    for Dmin, d in best.items():
        for K, q in d.items():
            if K - 1 in d and K >= 3:
                conv.append((abs(q - d[K - 1]), q, Dmin, K))
    conv.sort()
    good = [t[1] for t in conv[: max(3, len(conv) // 3)]]
    med = float(np.median(good)) if good else float("nan")
    lo, hi = (float(min(good)), float(max(good))) if good else (np.nan, np.nan)
    print("  -> %s_inf = %.*f  (range of the best third of K-converged fits: %.*f .. %.*f; all fits: %.*f .. %.*f)"
          % (name, digits, med, digits, lo, digits, hi, digits, vals.min(), digits, vals.max()))
    if fixed_compare is not None:
        print("  with %s_inf = %g fixed (same number of free parameters as the free fit with K-1 powers):" % (name, fixed_compare))
        for Dmin in sorted(best):
            sel = [i for i, D in enumerate(Ds) if D >= Dmin]
            for K in (3, 4, 5, 6):
                if len(sel) < K + 3:
                    continue
                _, cf, rf = fit([Ds[i] for i in sel], [Q[i] for i in sel], K, fixed=fixed_compare)
                q0, c, r = fit([Ds[i] for i in sel], [Q[i] for i in sel], K - 1)
                print("    Dmin %4g: fixed, %d powers: rms %.1e   free, %d powers: rms %.1e  (lambda_inf %.6f)"
                      % (Dmin, K, rf, K - 1, r, q0))
    return med, lo, hi


Ds = sorted(bg)
om = [bg[D]["omega"] for D in Ds]
f0 = [bg[D]["f0"] for D in Ds]
kap = bg[Ds[0]]["kappa"]
A2 = [bg[D]["omega"] ** 2 * bg[D]["f0"] ** 2 / (1 - kap * bg[D]["f0"] ** 2) ** 2 for D in Ds]
xh = [bg[D]["X"] for D in Ds]
print("kappa = %g" % kap)
res = {}
res["omega"] = report("omega", Ds, om)
res["f0"] = report("f0", Ds, f0)
res["A2"] = report("A2", Ds, A2)
res["xh"] = report("x_h", Ds, xh)
if gm:
    Dg = sorted(gm)
    lam = [gm[D]["lam"] for D in Dg]
    res["lambda"] = report("lambda", Dg, lam, fixed_compare=2.0)
    g = [1 / l for l in lam]
    res["gamma"] = report("gamma", Dg, g)
    # GAMMA.md-style table: free-fit lambda_inf - 2 against the number of powers, and the lambda_inf = 2 fits
    print("\nlambda_inf - 2 from free fits (rows: Dmin; columns: number of powers K)")
    Ks = (3, 4, 5, 6, 7)
    print("  Dmin " + " ".join("     K=%d   " % K for K in Ks))
    tab = {}
    for Dmin in (12, 16, 20, 24, 30):
        sel = [i for i, D in enumerate(Dg) if D >= Dmin]
        row = []
        for K in Ks:
            if len(sel) >= K + 3:
                q0, c, rms = fit([Dg[i] for i in sel], [lam[i] for i in sel], K)
                row.append(q0 - 2)
            else:
                row.append(np.nan)
        tab[Dmin] = row
        print("  %4d " % Dmin + " ".join("%+.2e  " % v for v in row))
    print("\nwith lambda_inf = 2 fixed: lambda = 2 + c1/n + c2/n^2 + ...")
    cs = []
    for Dmin in (12, 16, 20, 24):
        sel = [i for i, D in enumerate(Dg) if D >= Dmin]
        for K in (4, 5, 6):
            if len(sel) >= K + 3:
                _, c, rms = fit([Dg[i] for i in sel], [lam[i] for i in sel], K, fixed=2.0)
                cs.append(c[0])
                print("  Dmin %d, %d powers: c1 = %.5f  c2 = %+.3f  c3 = %+.2f  rms %.1e" % (Dmin, K, c[0], c[1], c[2], rms))
    res["c1_fixed2"] = (float(np.median(cs)), float(min(cs)), float(max(cs)))
    res["lam_minus_2_table"] = {str(k): [float(x) for x in v] for k, v in tab.items()}
out = os.path.join(here, "data", "css_fit_k%g.json" % kap)
json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in res.items()}, open(out, "w"), indent=1)
print("\nwritten", out)
