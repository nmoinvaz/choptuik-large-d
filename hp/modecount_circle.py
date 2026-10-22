#!/usr/bin/env python3
"""Winding number of det M(lambda) on small circles (notes/MODE_COUNT.md): a local test of a single zero or pole.

    python3 modecount_circle.py STATE.npz RE,IM,R [RE,IM,R ...] [--M=N] [--kf=none|13] [--tol=0.785] [--n0=16]
                                [--res=hp/data/modecount_circles.jsonl]

The phase of det M is followed around each circle lambda = c + R e^{i t}, starting from n0 equal steps and bisecting
until every step is at most tol, then every step above tol / 2 is bisected once more and checked against the whole.
Each evaluation goes to hp/data/modecount_<tag>_evals.jsonl (the same cache as modecount_wind.py for that state).
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from modecount_wind import parse, wrap, DATA  # noqa: E402


def main():
    from modecount_common import setup
    args, opts = parse(sys.argv[1:])
    path = args[0]
    kf = "default" if "kf" not in opts else (None if opts["kf"] == "none" else int(opts["kf"]))
    st = setup(path, M=int(opts["M"]) if "M" in opts else None, threads=int(opts.get("threads", 1)), kf=kf)
    tag = opts.get("tag", f"D{st.D:.4f}_M{st.M}")
    tol = float(opts.get("tol", math.pi / 4))
    n0 = int(opts.get("n0", 16))
    cache_path = os.path.join(DATA, f"modecount_{tag}_evals.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            cache[(r["re"], r["im"])] = r
    cf = open(cache_path, "a")
    logf = open(os.path.join(DATA, f"modecount_{tag}.log"), "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    def det(lam):
        key = (float(lam.real), float(lam.imag))
        if key not in cache:
            t = time.time()
            la, arg, pmin, pmax = st.det(key[0], key[1])
            r = dict(re=key[0], im=key[1], lnabs=la, arg=arg, pmin=pmin, pmax=pmax, sec=round(time.time() - t, 2))
            cache[key] = r
            cf.write(json.dumps(r) + "\n")
            cf.flush()
        return cache[key]["lnabs"], cache[key]["arg"]

    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} circles, D {st.D}, M {st.M}, kfreeze {st.kf}, omega {st.omega:.6f}, "
        f"lam_g {st.lam_g:.10f}")
    for spec in args[1:]:
        cr, ci, R = (float(v) for v in spec.split(","))
        c = complex(cr, ci)
        pt = lambda s: c + R * complex(math.cos(2 * math.pi * s), math.sin(2 * math.pi * s))
        vals = {i / n0: det(pt(i / n0)) for i in range(n0)}
        vals[1.0] = vals[0.0]

        def step(a, b):
            return wrap(vals[b][1] - vals[a][1])

        changed = True
        while changed:
            changed = False
            ts = sorted(vals)
            for a, b in zip(ts[:-1], ts[1:]):
                if abs(step(a, b)) > tol and (b - a) * 2 * math.pi * R > 1e-6:
                    m = 0.5 * (a + b)
                    vals[m] = det(pt(m))
                    changed = True
        ts = sorted(vals)
        bad = 0
        for a, b in zip(ts[:-1], ts[1:]):
            d = step(a, b)
            if abs(d) > tol / 2:
                m = 0.5 * (a + b)
                vals[m] = det(pt(m))
                if abs(wrap(vals[m][1] - vals[a][1]) + wrap(vals[b][1] - vals[m][1]) - d) > 1e-6:
                    bad += 1
        ts = sorted(vals)
        steps = [step(a, b) for a, b in zip(ts[:-1], ts[1:])]
        w = sum(steps) / (2 * math.pi)
        rec = dict(D=st.D, M=st.M, kfreeze=st.kf, tag=tag, center=[cr, ci], R=R, winding=w, points=len(ts) - 1,
                   max_step=max(abs(s) for s in steps), verify_mismatch=bad,
                   min_lndet=min(v[0] for v in vals.values()), time=time.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"  circle |lam - ({cr:.6f} {ci:+.6f}i)| = {R}: winding of det M {w:+.6f}, {len(ts) - 1} points, "
            f"max step {rec['max_step']:.3f} rad, verification mismatches {bad}")
        with open(opts.get("res", os.path.join(DATA, "modecount_circles.jsonl")), "a") as fo:
            fo.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
