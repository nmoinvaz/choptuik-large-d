#!/usr/bin/env python3
"""Checks of the large-D mixed-precision Newton (notes/LARGE_D_MIXED.md): the exact Jacobian-vector product
(Solver.jvp), its controls, and the correction loop (driver.gmres_refine) against the double-double Newton step.

    cd hp && python3 largeD_mixed_check.py jvp [--states=...] [--threads=2] [--mutate=1|2|3] [--expect_fail] [--json=F]
    cd hp && python3 largeD_mixed_check.py controls
    cd hp && python3 largeD_mixed_check.py dotcheck
    cd hp && python3 largeD_mixed_check.py steps [--D=20,40,45,50,55,60] [--seeds=1,2] [--out=F.jsonl] [--step_tol=1e-10]

jvp: at each state, J v by Solver.jvp (one forward-mode pass per block, double-double) against the double-double
  Jacobian (Solver.jacobian, hi and lo) times v, the product summed in double-double (dd_matvec below), for a random v
  with nonzero lo parts. Per entry the difference is scaled by (|J| |v|)_i (the size of the terms summed), and the
  largest such ratio is reported, with the normwise ratio max |diff| / max |Jv| and the product's time against the
  Jacobian's. Also: F from the same pass against Solver.residual, bitwise; and a unit vector's product against that
  Jacobian column, bitwise (the same pass).
  The two share the model's equations (model::block_state on Du lanes); what the check tests is the product's own
  part: the tangent seeding of the inputs and Delta, the target node, the frozen horizon rows and the phase row.
  Pass, two tiers. Strict (D = 3.7, M = 48 resampled to 16 and to 32, no freeze): every ratio <= 1e-28 (the criterion
  written before the runs: double-double rounding up to ~100 for the summation length). Large D (D = 20.04, M = 32;
  D = 50.01, M = 32, kfreeze 13): the dual pass's own rounding is amplified there, as the f64 lanes' is
  (notes/LARGE_D_MIXED.md section 2): jvp(v1 + v2) - jvp(v1) - jvp(v2) is 2e-28 (D = 20) and 1.5e-26 (D = 50) in the
  same scaled measure, so the product and the matrix (whose columns carry the same amplified rounding) cannot agree
  to 1e-28. There the pass is: ratio <= 10 x that linearity defect, measured at the same state with two other random
  vectors. The 1e-28 criterion first written for every state failed at D = 50 (9e-27); this tier replaces it there.
  --mutate breaks the product on purpose (1: the Delta tangent scaled by 1 + 1e-20; 2: the horizon block's tangent
  dropped; 3: the phase row dropped); the check must then fail (exit 1; --expect_fail inverts the exit code).
controls: every mutation at the D = 50.01 state and at M = 16; each must fire; exit 0 if all fire.
dotcheck: dd_matvec itself against exact rational arithmetic (fractions) on a few rows of the D = 20.04 Jacobian,
  and with one product term dropped as its control.
steps: at the saved states nearest to --D (perturbed by 1e-12 relative as in hp/mixed_largeD_blocks.py, seeds
  --seeds), the Newton step of each loop variant (f64g: the f64 Jacobian then gmres_refine; f64hg: the horizon fix then
  gmres_refine) against the double-double step: relative difference, products, cycles, time. Pass: <= 1e-5 at every
  state (the criterion of the horizon fix). Records appended to --out as they are done.
"""
import glob
import json
import os
import re
import sys
import time
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from hpkernel.driver import GMRES, dd_add, load, make_solver, newton_step, resample_z  # noqa: E402

LOWD = os.path.join(HERE, "data", "lowD_states", "lowDv_D3.7000_M48.npz")
_SPLIT = 134217729.0  # 2^27 + 1


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def split(a):
    t = _SPLIT * a
    hi = t - (t - a)
    return hi, a - hi


def two_prod(a, b):
    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def dd_mul(ah, al, bh, bl):
    p, e = two_prod(ah, bh)
    e = e + (ah * bl + al * bh)
    h = p + e
    return h, e - (h - p)


def dd_matvec(Jh, Jl, vh, vl, drop=None):
    """(Jh + Jl) (vh + vl) in double-double, column by column (each column's product added with dd_add); drop=(i, j)
    leaves out one term (the control of dotcheck)"""
    n = Jh.shape[0]
    yh, yl = np.zeros(n), np.zeros(n)
    for j in range(Jh.shape[1]):
        ph, pl = dd_mul(Jh[:, j], Jl[:, j], np.full(n, vh[j]), np.full(n, vl[j]))
        if drop is not None and drop[1] == j:
            ph[drop[0]] = pl[drop[0]] = 0.0
        yh, yl = dd_add(yh, yl, ph, pl)
    return yh, yl


def opts_of(argv):
    pos = [a for a in argv if not a.startswith("--")]
    opts = {}
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opts[k] = v if v else True
    return pos, opts


def saved_states():
    return {float(re.search(r"_D([0-9.]+)\.npz$", p).group(1)): p for p in glob.glob(os.path.join(HERE, "data", "states", "up*_D*.npz"))}


def solver_at(spec, threads):
    """spec: 'lowD:M' (the D = 3.7, M = 48 state resampled to M) or a D value (the saved scan state nearest to it,
    kfreeze 13 from D = 35 as the scan ran). Returns (s, zh, zl, label)"""
    if str(spec).startswith("lowD"):
        M = int(str(spec).split(":")[1])
        zh, zl, m = load(LOWD)
        K = len(m["nodes"])
        if M != m["M"]:
            zh, zl = resample_z(zh, m["M"], M, K)
        s = make_solver(m["D"], M, nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads)
        s.set_segs(s.plan(zh, zl, 1e-30, 0.05))
        return s, np.asarray(zh), np.asarray(zl), f"D {m['D']} M {M} (from 48)"
    avail = saved_states()
    p = avail[min(avail, key=lambda d: abs(d - float(spec)))]
    zh, zl, m = load(p)
    kf = 13 if m["D"] >= 35 else None
    s = make_solver(m["D"], m["M"], nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads, kfreeze=kf)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    return s, np.asarray(zh), np.asarray(zl), f"D {m['D']} M {m['M']}{' kfreeze 13' if kf else ''}"


def lin_defect(s, zh, zl, Jh, rng):
    """max over entries of |jvp(v1 + v2) - jvp(v1) - jvp(v2)| / (|J| (|v1| + |v2|)): the rounding of the dual pass"""
    n = s.size
    vs = []
    for _ in range(2):
        vh = rng.standard_normal(n)
        vs.append(two_sum(vh, vh * 1e-17 * rng.standard_normal(n)))
    v3 = dd_add(*vs[0], *vs[1])
    j1, j2, j3 = ([np.asarray(a) for a in s.jvp(zh, zl, *v)] for v in (vs[0], vs[1], v3))
    sh, sl = dd_add(*j1, *j2)
    d = np.abs((j3[0] - sh) + (j3[1] - sl))
    sc = np.abs(Jh) @ (np.abs(vs[0][0]) + np.abs(vs[1][0]))
    return float(np.max(d / np.where(sc > 0, sc, 1.0)))


def jvp_check(opts, quiet=False):
    threads = int(opts.get("threads", 2))
    specs = str(opts.get("states", "lowD:16,lowD:32,20,50")).split(",")
    mut = int(opts.get("mutate", 0))
    tol = float(opts.get("tol", 1e-28))
    rng = np.random.default_rng(7)
    ok, recs = True, []
    for spec in specs:
        s, zh, zl, label = solver_at(spec, threads)
        n = s.size
        vh = rng.standard_normal(n)
        vl = vh * 1e-17 * rng.standard_normal(n)
        vh, vl = two_sum(vh, vl)
        t = time.time()
        Jh, Jl = s.jacobian(zh, zl)
        t_jac = time.time() - t
        Jh, Jl = np.asarray(Jh), np.asarray(Jl)
        t = time.time()
        J64 = np.asarray(s.jacobian_f64(zh, zl))
        t_j64 = time.time() - t
        del J64
        reps = 3
        t = time.time()
        for _ in range(reps):
            yh, yl, Fh, Fl = s.jvp(zh, zl, vh, vl, mutate=mut, with_residual=True)
        t_jvp = (time.time() - t) / reps
        yh, yl = np.asarray(yh), np.asarray(yl)
        rh, rl = dd_matvec(Jh, Jl, vh, vl)
        dh, dl = dd_add(yh, yl, -rh, -rl)
        diff = np.abs(dh + dl)
        scale = np.abs(Jh) @ np.abs(vh)
        ratio = float(np.max(diff / np.where(scale > 0, scale, 1.0)))
        normwise = float(np.max(diff) / np.max(np.abs(rh)))
        R = s.residual(zh, zl)
        F_same = bool(np.array_equal(np.asarray(R[0]), np.asarray(Fh)) and np.array_equal(np.asarray(R[1]), np.asarray(Fl)))
        j = int(rng.integers(0, n))
        e = np.zeros(n)
        e[j] = 1.0
        ch, cl = (np.asarray(a) for a in s.jvp(zh, zl, e, np.zeros(n), mutate=mut))
        col_same = bool(np.array_equal(ch, Jh[:, j]) and np.array_equal(cl, Jl[:, j]))
        strict = str(spec).startswith("lowD")
        lin = None if strict else lin_defect(s, zh, zl, Jh, rng)
        lim = tol if strict else 10 * max(lin, 1e-30)
        good = ratio <= lim and (mut != 0 or (F_same and col_same))
        ok &= good
        rec = dict(state=label, N=n, tier="strict" if strict else "largeD", ratio=ratio, normwise=normwise, lin_defect=lin,
                   limit=lim, F_bitwise=F_same, column_bitwise=col_same, t_jvp=round(t_jvp, 4), t_jac_dd=round(t_jac, 3),
                   t_jac_f64=round(t_j64, 3), mutate=mut, pass_=good)
        recs.append(rec)
        if not quiet:
            print(f"{label} (N {n}){' mutate ' + str(mut) if mut else ''}: max |Jv - J.v| / (|J||v|) {ratio:.1e}, "
                  f"normwise {normwise:.1e}{'' if strict else f', linearity defect {lin:.1e}'} (limit {lim:.0e}); "
                  f"residual from the same pass bitwise {F_same}, unit vector = Jacobian column bitwise {col_same}; "
                  f"jvp {t_jvp * 1e3:.0f} ms, dd Jacobian {t_jac:.2f} s ({t_jac / t_jvp:.0f} products), f64 Jacobian "
                  f"{t_j64:.2f} s ({t_j64 / t_jvp:.1f} products) -> {'pass' if good else 'FAIL'}", flush=True)
    if "json" in opts:
        json.dump(recs, open(opts["json"], "w"), indent=1)
    if opts.get("expect_fail"):
        return 0 if not ok else 1
    return 0 if ok else 1


def controls(opts):
    fired = 0
    cases = [(m, st) for m in (1, 2, 3) for st in ("50", "lowD:16")]
    for m, st in cases:
        o = dict(states=st, mutate=m, threads=opts.get("threads", 2))
        r = jvp_check(o)
        fired += r != 0
        print(f"  control mutate {m} at {st}: {'fires' if r != 0 else 'DOES NOT FIRE'}", flush=True)
    print(f"{fired} of {len(cases)} controls fire")
    return 0 if fired == len(cases) else 1


def dotcheck(opts):
    s, zh, zl, label = solver_at("20", int(opts.get("threads", 2)))
    Jh, Jl = (np.asarray(a) for a in s.jacobian(zh, zl))
    rng = np.random.default_rng(3)
    n = s.size
    vh = rng.standard_normal(n)
    vl = vh * 1e-17 * rng.standard_normal(n)
    vh, vl = two_sum(vh, vl)
    rows = [0, 200, 500, n - 70, n - 1]
    ok = True
    for drop in (None, (rows[2], int(np.argmax(np.abs(Jh[rows[2]]) > 0)))):
        yh, yl = dd_matvec(Jh, Jl, vh, vl, drop=drop)
        worst = 0.0
        for i in rows:
            ex = sum(Fraction(Jh[i, j]) * Fraction(vh[j]) + Fraction(Jh[i, j]) * Fraction(vl[j]) + Fraction(Jl[i, j]) * Fraction(vh[j])
                     + Fraction(Jl[i, j]) * Fraction(vl[j]) for j in range(n) if Jh[i, j] != 0 or Jl[i, j] != 0)
            sc = float(sum(abs(Fraction(Jh[i, j]) * Fraction(vh[j])) for j in range(n))) or 1.0
            worst = max(worst, abs(float(Fraction(yh[i]) + Fraction(yl[i]) - ex)) / sc)
        good = worst <= 1e-30
        print(f"{label}: dd_matvec {'with one term dropped (control)' if drop else ''} vs exact rationals on rows {rows}: "
              f"max error / (|J||v|) {worst:.1e} -> {'pass' if good else 'fails'}")
        ok &= good if drop is None else not good
    return 0 if ok else 1


def steps(opts):
    import scipy.linalg as sla  # noqa: F401
    threads = int(opts.get("threads", 2))
    Ds = [float(x) for x in str(opts.get("D", "20,40,45,50,55,60")).split(",")]
    seeds = [int(x) for x in str(opts.get("seeds", "1,2")).split(",")]
    variants = str(opts.get("variants", "f64,f64h,f64g,f64hg")).split(",")
    g = dict(GMRES)
    for k in ("step_tol", "max_products", "max_outer"):
        if k in opts:
            g[k] = type(GMRES[k])(float(opts[k])) if k == "step_tol" else int(opts[k])
    out = opts.get("out")
    done = set()
    if out and os.path.exists(out):
        done = {(round(r["D"], 4), r["seed"]) for r in map(json.loads, open(out))}
    worst = {v: 0.0 for v in variants}
    for D0 in Ds:
        s, zh, zl, label = solver_at(D0, threads)
        n = s.size
        D = float(label.split()[1])
        for seed in seeds:
            if (round(D, 4), seed) in done:
                continue
            rng = np.random.default_rng(seed)
            ph = zh * (1 + 1e-12 * rng.standard_normal(n))
            pl = zl.copy()
            t = time.time()
            dh, dl, Fh, Fl, *_ = s.newton(ph, pl)
            t_dd = time.time() - t
            ddz = np.asarray(dh) + np.asarray(dl)
            nr = np.linalg.norm(ddz)
            rec = dict(D=D, seed=seed, t_dd=round(t_dd, 2), gmres=g, v={})
            line = []
            for v in variants:
                t = time.time()
                xh, xl, *_r, info = newton_step(s, ph, pl, v, gmres=g)
                tt = time.time() - t
                c = float(np.linalg.norm(np.asarray(xh) + np.asarray(xl) - ddz) / nr)
                worst[v] = max(worst[v], c)
                rec["v"][v] = dict(err=c, t=round(tt, 3), **{k: info[k] for k in info if k != "changes"},
                                   changes=info.get("changes"))
                line.append(f"{v} {c:.1e} ({info.get('products', 0)} products, {tt:.2f} s)")
            print(f"{label} seed {seed} (dd step {t_dd:.2f} s): " + "; ".join(line), flush=True)
            if out:
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
    print("worst step difference: " + ", ".join(f"{v} {worst[v]:.1e}" for v in variants))
    return 0 if all(worst[v] <= 1e-5 for v in variants if v != "f64") else 1


def main():
    pos, opts = opts_of(sys.argv[1:])
    cmd = pos[0] if pos else "jvp"
    return {"jvp": jvp_check, "controls": controls, "dotcheck": dotcheck, "steps": steps}[cmd](opts)


if __name__ == "__main__":
    sys.exit(main())
