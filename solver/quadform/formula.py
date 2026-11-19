#!/usr/bin/env python3
"""The quadratic formula for Delta(D) with a given number of constants, beside the rational one (post hoc, not one of
the frozen experiments; notes/QUADRATIC_FORMULA.md 5).

    cd solver/quadform && python3 formula.py fit [--N=11] [--dmin=4] [--dmax=245.64] [--weights=sigma|uniform]
                                                 [--spread=3] [--Mmax=4] [--out=data/formula_q11.json]
    cd solver/quadform && python3 formula.py check [--file=data/formula_q11.json] [--digits=K] [--mutate=coef]

fit: every quadratic form (p, q, r) with p + q + r + 2 = N and degrees within 3 of each other, and every rational
[L/M] with M <= Mmax, fitted by algfit.fit on all the points dmin <= D <= dmax; each class enters its form with the
smallest sum of squares. Prints the largest |miss| by range for both and for the README's stored formula, and writes
the quadratic's P, Q, R as decimal coefficients of eps^j, eps = 1 / (D - 2), scaled so that Q(0) = 1, with the sign
of the root.
check: evaluates the written coefficients in float64, Delta = 2 R / (-Q + s sqrt(Q^2 - 4 P R)), on the same points,
and exits 1 unless the largest |miss| per range is within 5e-15 + 5% of the stored one (--mutate=coef scales one
coefficient by 1 + 1e-6: it must exit 1; --digits=K first rounds every coefficient to K significant digits).
"""
import json
import math
import os
import sys

import mpmath as mp
from flint import arb

import algfit as A

ARGS = A.ARGS
RANGES = [(3.04, 4), (4, 8.25), (8.25, 30.2), (30.2, 67.77), (67.77, 130), (130, 245.64)]


def by_range(pts, f):
    """the largest |f(D, y) - y| per range, as floats"""
    out = {}
    for lo, hi in RANGES:
        v = [abs(f(D, y)) for D, y, _, _ in pts if lo - 1e-9 <= D < hi + 1e-9]
        if v:
            out[f"{lo:g}-{hi:g}"] = max(v)
    return out


def eps_monomials(M, k):
    """A_k as coefficients of eps^j (mpmath), from its Chebyshev form on [a, b]"""
    mono_u = M._mono(k)
    a = mp.mpf(M.a.mid().str(A.PREC // 3, radius=False))
    b = mp.mpf(M.b.mid().str(A.PREC // 3, radius=False))
    lin = [-(a + b) / (b - a), 2 / (b - a)]                 # u = lin[0] + lin[1] eps
    out, pw = [mp.mpf(0)] * len(mono_u), [mp.mpf(1)]
    for c in mono_u:
        for i, v in enumerate(pw):
            out[i] += c * v
        nxt = [mp.mpf(0)] * (len(pw) + 1)
        for i, v in enumerate(pw):
            nxt[i] += v * lin[0]
            nxt[i + 1] += v * lin[1]
        pw = nxt
    return out


def f64_value(F, D):
    e = 1.0 / (D - 2.0)
    P, Q, R = (math.fsum(float(c) * e ** j for j, c in enumerate(F[k])) for k in ("P", "Q", "R"))
    return 2 * R / (-Q + F["sign"] * math.sqrt(Q * Q - 4 * P * R))


def fit():
    N, dmin, wt = int(ARGS.get("N", 11)), float(ARGS.get("dmin", 4)), ARGS.get("weights", "sigma")
    dmax, spread = float(ARGS.get("dmax", A.DMAX)), int(ARGS.get("spread", 3))
    pts = [p for p in A.load_points() if dmin - 1e-9 <= p[0] <= dmax + 1e-9]
    print(f"formula, N {N}, weights {wt}: {len(pts)} points, D {pts[0][0]:.4f} to {pts[-1][0]:.2f}", flush=True)
    best, cands = {}, []
    classes = [("rational", A.rational_forms(N, int(ARGS.get("Mmax", 4)))), ("quadratic", A.quadratic_forms(N, spread))]
    for cls, forms in classes:
        for degs in forms:
            M = A.fit(pts, degs, uniform=(wt == "uniform"))
            print(f"  {cls:9s} {A.form_name(degs):>10s} ss {M.info['ss']:.3e} max miss {M.info['max_miss']:.2e} "
                  f"ok {M.info['ok']} defect {A.real_axis_defect(M, dmin)}", flush=True)
            cands.append(dict(cls=cls, form=A.form_name(degs), ss=M.info["ss"], max_miss=M.info["max_miss"],
                              ok=M.info["ok"], defect=bool(A.real_axis_defect(M, dmin))))
            if M.info["ok"] and not A.real_axis_defect(M, dmin) and (cls not in best or M.info["ss"] < best[cls].info["ss"]):
                best[cls] = M
    allp = A.load_points()
    ref = A.plant_fn("T_pole")
    rows = {"README formula (stored)": by_range(allp, lambda D, y: float(ref(D) - y))}
    for cls, M in best.items():
        rows[f"{cls} {A.form_name(M.degs)}"] = by_range(allp, lambda D, y, M=M: float(M.value(D, y)[0] - y))
    for k, v in rows.items():
        print(f"  {k:28s} " + "  ".join(f"{r} {e:.1e}" for r, e in v.items()))
    M = best["quadratic"]
    P, Q, R = (eps_monomials(M, k) for k in (2, 1, 0))
    s = Q[0]
    P, Q, R = ([c / s for c in X] for X in (P, Q, R))
    D0 = pts[len(pts) // 2]
    e = 1 / (mp.mpf(D0[0]) - 2)
    p, q, r = (mp.fsum(c * e ** j for j, c in enumerate(X)) for X in (P, Q, R))
    sq = mp.sqrt(q * q - 4 * p * r)
    y0 = mp.mpf(D0[1].mid().str(40, radius=False))
    sign = 1 if abs(2 * r / (-q + sq) - y0) < abs(2 * r / (-q - sq) - y0) else -1
    vinf, _ = M.value("inf", arb(A.DINF))
    bps = sorted((z.real for z in M.branch_points() if z == z and abs(z) != float("inf") and
                  abs(z.imag) < 1e-9 * max(1.0, abs(z.real))))
    F = dict(form="P Delta^2 + Q Delta + R = 0; Delta = 2 R / (-Q + sign sqrt(Q^2 - 4 P R)); P, Q, R = sum_j c_j eps^j; "
                  "eps = 1 / (D - 2)", degrees_pqr=[M.degs[2], M.degs[1], M.degs[0]], n_const=M.n - 1, sign=sign,
             P=[mp.nstr(c, 22) for c in P], Q=[mp.nstr(c, 22) for c in Q], R=[mp.nstr(c, 22) for c in R],
             fitted_on=[dmin, dmax], weights=wt, chi2dof=M.info["chi2dof"], max_over_sigma=M.info["max_over_sigma"], points=len(pts), Delta_inf=vinf.mid().str(14, radius=False),
             real_branch_points_D=bps, max_miss_by_range=rows, candidates=cands, produced_by="solver/quadform/formula.py fit")
    F["max_miss_f64"] = by_range(allp, lambda D, y: f64_value(F, D) - float(y))
    print("  float64 from the written coefficients: " + "  ".join(f"{r} {e:.1e}" for r, e in F["max_miss_f64"].items()))
    print(f"  Delta_inf {F['Delta_inf']}, real branch points at D = {bps}")
    out = ARGS.get("out", os.path.join(A.HERE, "data", f"formula_q{N}.json"))
    json.dump(F, open(out, "w"), indent=1)
    print(f"wrote {out}")
    return 0


def check():
    path = ARGS.get("file", os.path.join(A.HERE, "data", "formula_q11.json"))
    F = json.load(open(path))
    if "digits" in ARGS:                                   # the coefficients as a table would print them
        for k in ("P", "Q", "R"):
            F[k] = [mp.nstr(mp.mpf(c), int(ARGS["digits"])) for c in F[k]]
    if ARGS.get("mutate") == "coef":
        F["Q"][1] = mp.nstr(mp.mpf(F["Q"][1]) * (1 + mp.mpf("1e-6")), 22)
    got = by_range(A.load_points(), lambda D, y: f64_value(F, D) - float(y))
    key = [k for k in F["max_miss_by_range"] if k.startswith("quadratic")][0]
    ok = True
    for r, e in F["max_miss_by_range"][key].items():
        good = got[r] <= 5e-15 + 1.05 * e
        ok = ok and good
        print(f"  D {r:12s} stored {e:.2e}  float64 from the file {got[r]:.2e}  {'ok' if good else 'FAIL'}")
    print(f"check {os.path.basename(path)}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(dict(fit=fit, check=check)[sys.argv[1]]())
