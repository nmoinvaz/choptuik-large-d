#!/usr/bin/env python3
"""Measurement A of notes/QUAD_DOUBLE.md 13.11: can a one-parameter correction of the predicted horizon datum a_h,
chosen to maximise the horizon series' radius, put perturbed horizon data back inside the series block's basin? On the
stored double-double states the horizon data are perturbed (a_h scaled by 1 + eps; m_h and a_h by a random vector of
relative size eps; every unknown by 1 + eps r), the kernel's horizon series is recomputed and its radius read by
horizon_radius.py's ratio test (pole_test.ratio_radius); then a golden-section search over s in a_h -> a_h (1 + s)
maximises the radius. Recorded per case: R before, s*, R(s*), and for the mass kind the error of s* against the exact
restoring value -eps/(1 + eps) in units of delta_max(D, 0.30) from route1-mac/pole_test.jsonl.

    cd hp && python3 mass_correct.py RAWDIR OUT.jsonl STATE.npz [STATE.npz ...] [--threads N]
"""
import json
import math
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from hpkernel.driver import load, make_solver  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pole_test import ratio_radius  # noqa: E402

MASS_EPS = (1e-4, 1e-3, 1e-2)
RAND_EPS = (1e-6, 1e-4)
U0 = 0.30


def delta_max_of(D, raw):
    recs = [json.loads(l) for l in open(os.path.join(raw, "route1-mac", "pole_test.jsonl")) if l.strip()]
    recs = [r for r in recs if "delta_max" in r and "D" in r]
    return min(recs, key=lambda r: abs(r["D"] - D))["delta_max"]["0.3"]


def golden_max(f, lo, hi, iters=50):
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    n = 2
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = f(d)
        n += 1
    s = (a + b) / 2
    return s, f(s), n + 1


def main():
    args = list(sys.argv[1:])
    threads = 2
    if "--threads" in args:
        i = args.index("--threads"); threads = int(args[i + 1]); del args[i:i + 2]
    raw, out, paths = args[0], args[1], args[2:]
    rng = np.random.default_rng(3)
    with open(out, "a") as f:
        for path in paths:
            zh, zl, m = load(path)
            D, M, nodes = m["D"], m["M"], list(m["nodes"])
            K = len(nodes); n = D - 2; u0 = 1.0 - nodes[-1]
            kf = 13 if D >= 35 else None
            s = make_solver(D, M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130), threads=threads, kfreeze=kf)
            s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
            sig = s.series_scale(1)
            oa, om = 3 * M + 4 * M * K, 2 * M + 4 * M * K
            dmax = delta_max_of(D, raw)

            def radius(zh2, zl2):
                hh, hl = s.series(zh2, zl2, 1, True)
                c = (np.asarray(hh) + np.asarray(hl)).reshape(-1, 4, M)
                R, _ = ratio_radius(c, u0, sig)
                return R if R is not None and np.isfinite(R) else 0.0

            def corrected(zh2, zl2, sc):
                a, b = zh2.copy(), zl2.copy()
                a[oa:oa + M] *= 1 + sc; b[oa:oa + M] *= 1 + sc
                return a, b

            R_true = radius(zh, zl)
            print(f"D {D:.4f}: on the solution R {R_true:.3f}; delta_max(0.30) {dmax:.1e}", flush=True)
            cases = [("mass", e) for e in MASS_EPS] + [("random_h", e) for e in RAND_EPS] + [("random_all", e) for e in RAND_EPS]
            for kind, eps in cases:
                zp, zlp = zh.copy(), zl.copy()
                if kind == "mass":
                    zp[oa:oa + M] *= 1 + eps; zlp[oa:oa + M] *= 1 + eps
                    s_true = -eps / (1 + eps)
                elif kind == "random_h":
                    v = zp[om:om + 2 * M]; r = rng.standard_normal(2 * M)
                    zp[om:om + 2 * M] = v + eps * np.linalg.norm(v) * r / np.linalg.norm(r)
                    s_true = None
                else:
                    r = rng.standard_normal(len(zp))
                    zp = zp * (1 + eps * r); zlp = zlp * (1 + eps * r)
                    s_true = None
                R0 = radius(zp, zlp)
                f_obj = lambda sc: radius(*corrected(zp, zlp, sc))
                s_star, R_star, nev = golden_max(f_obj, -2 * eps, 2 * eps, iters=55)
                rec = dict(state=os.path.basename(path), D=D, kind=kind, eps=eps, R_true=R_true, R_before=R0, s_star=s_star, R_after=R_star,
                           evaluations=nev, delta_max=dmax, s_true=s_true,
                           s_err_over_delta_max=(abs(s_star - s_true) / dmax) if s_true is not None else None,
                           passes=bool(R_star > U0))
                f.write(json.dumps(rec) + "\n"); f.flush()
                extra = f", |s* - s_true|/delta_max {rec['s_err_over_delta_max']:.1e}" if s_true is not None else ""
                print(f"   {kind:10s} eps {eps:.0e}: R before {R0:.3f} -> after {R_star:.3f} (s* {s_star:+.3e}{extra}); {'PASS' if rec['passes'] else 'FAIL'} ({nev} evaluations)", flush=True)


if __name__ == "__main__":
    main()
