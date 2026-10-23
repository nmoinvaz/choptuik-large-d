#!/usr/bin/env python3
"""Is the scan's centre-series order kc larger than it needs to be, and what does a smaller one cost or save?

At each saved state the centre series' terms |c_k| x0^k (largest over fields and tau points) are computed with the
state's own kc, and kc_min is the order past which they stay below 1e-45 of the largest for three orders, plus a
margin (bench_variants.kc_auto's rule, notes/LARGE_D_MIXED.md section 7). Then, with kc = the state's and kc = kc_min:
the residual, the double-double Jacobian (hi and lo) and the Newton step (dz hi and lo, through the double-double LU)
are compared bit for bit, and the Jacobian is timed (best of --reps).

    cd hp && python3 kc_check.py [--D=36,40,45,50,55,60,61.03] [--margin=10] [--reps=2] [--threads=2] [--json=out]
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import load, make_solver  # noqa: E402


def states():
    out = {}
    for p in glob.glob(os.path.join(HERE, "data", "states", "up*_D*.npz")) + \
            glob.glob(os.path.join(HERE, "data", "seeds", "local", "up*_D*.npz")):
        out[float(re.search(r"_D([0-9.]+)\.npz$", p).group(1))] = p
    return out


def kc_min(s, zh, zl, m, margin):
    h, l = s.series(zh, zl, 0)
    c = np.abs((np.asarray(h) + np.asarray(l)).reshape(-1, 4, m["M"])).max(axis=(1, 2))
    t = c * m["nodes"][0] ** np.arange(len(c))
    for k in range(20, len(t) - 3):
        if (t[k:k + 3] < 1e-45 * t.max()).all():
            return min(k + margin, m["kc"]), t
    return m["kc"], t


def timed(f, reps):
    best, out = None, None
    for _ in range(reps):
        t = time.time()
        out = f()
        dt = time.time() - t
        best = dt if best is None else min(best, dt)
    return out, best


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    Ds = [float(x) for x in opts.get("D", "36,40,45,50,55,60,61.03").split(",")]
    margin, reps, threads = int(opts.get("margin", 10)), int(opts.get("reps", 2)), int(opts.get("threads", 2))
    avail = states()
    recs = []
    for D0 in Ds:
        p = avail[min(avail, key=lambda d: abs(d - D0))]
        zh, zl, m = load(p)
        kf = 13 if m["D"] >= 35.5 else None     # run up7 on (D >= 36.07) froze the p1 harmonics above 13
        mk = lambda kc: make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=kc, ks=m["ks"], threads=threads,
                                    kfreeze=kf)
        s = mk(m["kc"])
        segs = s.plan(zh, zl, 1e-32, 0.05)
        s.set_segs(segs)
        k, t = kc_min(s, zh, zl, m, margin)
        s2 = mk(k)
        s2.set_segs(segs)
        same = {}
        r1, r2 = s.residual(zh, zl), s2.residual(zh, zl)
        same["residual"] = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(r1, r2))
        (J1, t1) = timed(lambda: s.jacobian(zh, zl), reps)
        (J2, t2) = timed(lambda: s2.jacobian(zh, zl), reps)
        same["jacobian"] = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(J1, J2))
        n1, n2 = s.newton(zh, zl), s2.newton(zh, zl)
        same["newton_step"] = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(n1[:4], n2[:4]))
        # when not bitwise: how far apart (relative to the largest entry), in the Jacobian and the step
        Ja, Jb = np.asarray(J1[0]) + np.asarray(J1[1]), np.asarray(J2[0]) + np.asarray(J2[1])
        dJ = float(np.max(np.abs(Ja - Jb)) / np.max(np.abs(Ja)))
        d1, d2 = np.asarray(n1[0]) + np.asarray(n1[1]), np.asarray(n2[0]) + np.asarray(n2[1])
        dstep = float(np.max(np.abs(d1 - d2)) / max(np.max(np.abs(d1)), 1e-300))
        rec = dict(D=m["D"], state=os.path.basename(p), M=m["M"], kfreeze=kf, kc_state=m["kc"], kc_min=k,
                   term_at_kc_min=float(t[k] / t.max()) if k < len(t) else None,
                   term_at_kc_state=float(t[min(m["kc"], len(t) - 1)] / t.max()),
                   jacobian_s=[round(t1, 2), round(t2, 2)], speedup=round(t1 / t2, 2), bitwise=same,
                   jacobian_rel_diff=dJ, step_rel_diff=dstep)
        recs.append(rec)
        print(f"D = {m['D']:8.4f} ({os.path.basename(p)}, kf {kf}): kc {m['kc']} -> {k} (terms {rec['term_at_kc_min']:.1e} "
              f"at {k}, {rec['term_at_kc_state']:.1e} at {m['kc']}); Jacobian {t1:.2f} s -> {t2:.2f} s ({t1 / t2:.2f}x); "
              f"bitwise: " + ", ".join(f"{a} {'yes' if b else 'NO'}" for a, b in same.items())
              + ("" if all(same.values()) else f" (Jacobian differs by {dJ:.1e} of its largest entry, step by {dstep:.1e})"),
              flush=True)
    if "json" in opts:
        json.dump(recs, open(opts["json"], "w"), indent=1)


if __name__ == "__main__":
    main()
