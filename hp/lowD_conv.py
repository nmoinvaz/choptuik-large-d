#!/usr/bin/env python3
"""Convergence table of the low-D double-double points (notes/LOW_D.md): for every D in hp/data/lowD_dd.jsonl, Delta
against the tau resolution M (and the other discretisation choices recorded: nodes, tol_step, kt), the differences
between successive M, and the error estimate of the best value.

    python3 lowD_conv.py [hp/data/lowD_dd.jsonl ...] [--selftest] [--mutate]

Error estimate (an estimator, not a bound): with the three highest M at one D (same nodes, tol_step, kt) and successive
differences d1 = Delta(M2) - Delta(M1), d2 = Delta(M3) - Delta(M2), convergence counts as demonstrated when the
contraction r = |d2 / d1| (rescaled to the step M2 -> M3 when the spacing differs) is below 0.5; the quoted error is
then |d2|, the last difference (larger than the geometric tail |d2| r / (1 - r), which is printed too: on planted
sequences with a 1/M prefactor the tail estimate undershot, the last difference did not). Otherwise "not demonstrated".
With two M only, the difference itself is quoted and no estimate.
Other discretisation changes at the best M (nodes, tol_step, kt) are listed as differences from the reference record.
--selftest: the same estimator on planted sequences Delta_inf + A e^(-b M) (1 + c / M) + noise must cover the planted
Delta_inf; --mutate plants a sequence that has not converged (b too small for the M used) and the estimator must refuse.
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def estimate(pts):
    """pts: [(M, Delta)] sorted by M; returns (best, err or None, r or None, note)"""
    if len(pts) < 2:
        return pts[-1][1], None, None, "one M"
    if len(pts) == 2:
        return pts[-1][1], None, None, f"two M: difference {pts[1][1] - pts[0][1]:+.1e}"
    (M1, v1), (M2, v2), (M3, v3) = pts[-3:]
    d1, d2 = v2 - v1, v3 - v2
    if d1 == 0:
        return v3, abs(d2), None, "d1 = 0"
    r_step = abs(d2 / d1)                       # ratio over (M3 - M2) after rescaling below
    r = r_step ** ((M3 - M2) / (M2 - M1)) if M3 - M2 != M2 - M1 else r_step
    # r is the contraction from the step M1->M2 to M2->M3 measured per (M3 - M2); the remainder after M3 is
    # the geometric tail of further steps of size M3 - M2
    if not (0 < r < 0.5):
        return v3, None, r, f"not demonstrated (ratio {r:.2f})"
    # the bar is the last difference |d2| (for r < 0.5 it exceeds the geometric tail |d2| r / (1 - r), which planted
    # sequences with a 1/M prefactor showed to undershoot the true remainder)
    return v3, abs(d2), r, f"geometric, ratio {r:.2g}, tail estimate {abs(d2) * r / (1 - r):.1e}"


def selftest(mutate=False):
    import random
    random.seed(3)
    ok = True
    cases = [(3.4654, 2.0, 0.27, 0.0, (64, 80, 96)), (3.4654, 5.0, 0.24, 3.0, (64, 96, 128)),
             (2.9, 1.0, 0.16, -2.0, (80, 96, 128)), (3.2, 0.5, 0.20, 1.0, (48, 64, 80))]
    if mutate:
        cases = [(3.4654, 2.0, 0.02, 0.0, (64, 80, 96))]
    for Dinf, A, b, c, Ms in cases:
        import math
        pts = [(M, Dinf + A * math.exp(-b * M) * (1 + c / M) + 1e-16 * random.gauss(0, 1)) for M in Ms]
        best, err, r, note = estimate(pts)
        covered = err is not None and abs(best - Dinf) <= err * 1.0001 + 1e-15
        print(f"planted Delta_inf {Dinf} A {A} b {b} c {c} M {Ms}: best {best:.15f} err {err} ({note}) true error "
              f"{best - Dinf:+.1e} -> {'covered' if covered else 'NOT covered'}")
        if mutate:
            ok &= err is None              # must refuse
        else:
            ok &= covered
    if mutate:
        print("mutation control:", "the estimator refused the unconverged sequence (as required)" if ok else "FAILED to refuse")
    else:
        print("selftest", "ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--selftest" in sys.argv:
        selftest("--mutate" in sys.argv)
    files = args or [os.path.join(HERE, "data", "lowD_dd.jsonl")]
    recs = []
    for f in files:
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get("converged", True):
                    recs.append(r)
    byD = defaultdict(list)
    for r in recs:
        byD[round(r["D"], 6)].append(r)
    for D in sorted(byD):
        rs = byD[D]
        key = lambda r: (tuple(r["nodes"]), r.get("tol_step"), r.get("kt"), r.get("kcs"), r.get("ks"))
        ref_key = key(min(rs, key=lambda r: (r["nodes"] != [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6], r.get("tol_step") != 1e-32)))
        main_ = sorted({r["M"]: r for r in rs if key(r) == ref_key}.values(), key=lambda r: r["M"])
        pts = [(r["M"], r["Delta"] + r["Delta_lo"]) for r in main_]
        best, err, ratio, note = estimate(pts)
        print(f"D {D}: " + "  ".join(f"M {M}: {v:.15f}" for M, v in pts))
        for (Ma, va), (Mb, vb) in zip(pts, pts[1:]):
            print(f"     M {Ma} -> {Mb}: {vb - va:+.2e}")
        print(f"     best {best:.15f}  error estimate {err if err is None else f'{err:.1e}'}  ({note})")
        for r in rs:
            if key(r) != ref_key:
                same_M = [p for p in main_ if p["M"] == r["M"]]
                if same_M:
                    dv = (r["Delta"] + r["Delta_lo"]) - (same_M[0]["Delta"] + same_M[0]["Delta_lo"])
                    print(f"     variant M {r['M']} nodes {r['nodes']} tol_step {r.get('tol_step')} kt {r.get('kt')} "
                          f"kcs {r.get('kcs')} ks {r.get('ks')}: {dv:+.2e} from the reference")
        last = main_[-1]
        print(f"     at M {last['M']}: |F| {last['F']:.1e}  max dDelta/dF {last.get('sens_max', float('nan')):.1e}  tails "
              f"{last.get('tail_c', float('nan')):.0e} {last.get('tail_h', float('nan')):.0e}  steps {last.get('steps')}")


if __name__ == "__main__":
    main()
