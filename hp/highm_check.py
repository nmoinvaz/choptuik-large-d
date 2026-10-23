#!/usr/bin/env python3
"""Checks and measurements for the kernel's larger tau resolutions (M = 160, 192, 256, 320, 384) and the block
storage of the mixed-precision Newton system (notes/HIGH_M.md).

    cd hp && python3 highm_check.py ref --save=REF.json [--Ms=8,16,48,64,80,96,128]    (with a build of main)
    cd hp && python3 highm_check.py ref --compare=REF.json [--Ms=...]                   (with this branch's build)
    cd hp && python3 highm_check.py block [--Ms=16,48,80,128] [--variants=f64x4,f64x1,dd4] [--mutate=pivot|link] [--expect_fail] [--json=F]
    cd hp && python3 highm_check.py controls
    cd hp && python3 highm_check.py converge [--M=80] [--json=F]
    cd hp && python3 highm_check.py lift --M=160 [--start=STATE.npz] [--plan=start|source] [--tol1=1e-14] [--no_replan] [--tol=1e-28] [--out=F.jsonl] [--tag=highm]
    cd hp && python3 highm_check.py profile --M=256 [--start=STATE.npz] [--reps=2] [--out=F.jsonl]

ref: bit-identity at M <= 128 of everything the kernel computes there (residual, the dense double-double Newton step
  and its pivots, chord and lu_solve_t after it, the dense f64 Jacobian, newton_mixed's step, pivots, lu_solve_t
  and chord, the centre and horizon series, the Taylor plan) at the D = 3.7 state resampled to each M (from M = 48
  for M <= 48, the M = 64 and 80 states at their own M, M = 96 and 128 from M = 80). Each output is stored as the
  sha256 of its float64 bytes (so M = 128, where a dense Jacobian is 400-800 MB, costs no disk); --compare exits 1
  on any differing hash. Uses only methods that the parent commit's kernel has, so --save runs on a build of main.
block: at M <= 128, newton_mixed in block storage (block=True) against the dense path (block=False) at the state
  and at a perturbed state (Delta + 1e-6, fields + 1e-8, seed 1): the residuals must be bit-identical (same code);
  the steps dz, the sensitivities y = J^-T e_Delta (lu_solve_t) and a chord step after are compared relative to
  their max. The yardstick is the rounding floor of the dense f64 LU itself: the same step solved with the dense
  double-double LU of the same f64 Jacobian (lu_factor(Jf) against lu_factor(Jf, f64=True)). Pass: every
  block-vs-dense difference <= max(100 x floor, 1e-12). --mutate (pivot: one pivot of the middle step scaled by
  1 + 1e-6; link: the Delta copies untied) must make it fail (--expect_fail inverts the exit code).
controls: block --mutate=pivot and --mutate=link at M = 48, each must fail; exit 0 if both fire.
converge: driver.newton_solve with jac="f64" from the perturbed state with the dense and with the block path, and a
  control (dense, a second start: Delta - 1e-6, seed 2); pass: Delta and max|z| differences <= max(1e-27, 10 x the
  control's), relative.
lift: a converged state (--start, default hp/data/lowD_states/lowDv_D3.7000_M80.npz; its meta has the nodes, kc 100,
  ks 250) resampled to --M and solved with the f64-Jacobian Newton (block storage above M = 128), in two passes. Pass
  1 on the Taylor plan (tol_step 1e-30, hmax 0.05) of the resampled start (--plan=start; --plan=source, the plan of the
  start state at its own M, blows up on the resampled state, |F| NaN at M = 128 from 80) to |F| <= --tol1 (1e-14).
  Pass 2 on the plan of the pass-1 state, which needs 2-4 times fewer steps (the resampled state's new top harmonics
  are aliasing products that the tau-derivative amplifies; near the solution they are small), to |F| <= --tol (1e-28);
  if the re-planned residual is not small, pass 1 continues instead. --no_replan: one pass on the start's plan to --tol.
  Both passes run through lowD_dd.solve_point (records in <out>_points.jsonl; states
  hp/data/lowD_states/<tag>_p1_D<D>_M<M>.npz and <tag>_D<D>_M<M>.npz with their plans; every accepted Newton step
  checkpointed, so a rerun of the same command resumes). Then one Newton step at the solution is timed by part
  (residual, Jacobian, block factorization, and the Jacobian without the tau derivative). The record in --out
  (default hp/data/lowD_highM/lift_D3.7.jsonl): Delta (hi, lo), |F|, iterations and seconds of each, the plans' step
  counts, the series tails (lowD_dd.tails), the timings, the stored LU's bytes, the load average and the peak RSS of
  the process (run one M per process). A rerun skips an M already in --out.
profile: at a state (--start, default the D = 3.7 state resampled to --M): seconds of a residual, of newton_mixed in
  block storage (residual + Jacobian + LU + solve), of its Jacobian build and of the block factorization
  (Solver.lu_info), and of the Jacobian build without the tau derivative (jac_mutate=1; the difference is the tau
  derivative's time), --reps times, 2 threads.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from hpkernel.driver import dd_add, dd_norm, load, make_solver, newton_solve, resample_z, save  # noqa: E402

STATES = os.path.join(HERE, "data", "lowD_states")
STATE = lambda M: os.path.join(STATES, f"lowDv_D3.7000_M{M}.npz")  # noqa: E731
MUT = {"pivot": 1, "link": 2}


def opts_of(argv):
    pos = [a for a in argv if not a.startswith("--")]
    opts = {}
    for a in argv:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            opts[k] = v if v else True
    return pos, opts


def peak_rss_mb():
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def source_for(M):
    """the D = 3.7 state to resample to M: M = 48 for M <= 48, the state itself for 64 and 80, M = 80 above"""
    if M in (64, 80):
        return STATE(M)
    return STATE(48) if M <= 48 else STATE(80)


def solver_at(M, threads=2, path=None, plan=(1e-30, 0.05)):
    """solver and z at the D = 3.7 state resampled to M, with the state's meta settings and a plan for z"""
    path = path or source_for(M)
    zh, zl, m = load(path)
    if M != m["M"]:
        zh, zl = resample_z(zh + zl, m["M"], M, len(m["nodes"]))
    s = make_solver(m["D"], M, nodes=m["nodes"], kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads)
    s.set_segs(m.get("segs") if M == m["M"] and m.get("segs") else s.plan(zh, zl, *plan))
    return s, np.asarray(zh), np.asarray(zl), dict(m, M=M, source=os.path.basename(path))


def perturb(zh, zl, dDelta, dz, seed):
    u = np.random.default_rng(seed).uniform(-1, 1, len(zh))
    u[-1] = 0.0
    e = dz * u
    e[-1] = dDelta
    return dd_add(zh, zl, e, np.zeros_like(e))


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


# ------------------------------------------------------------------ ref
def ref_hashes(M, threads):
    s, zh, zl, m = solver_at(M, threads)
    out = {}
    t0 = time.time()

    def put(name, *arrs):
        if len(arrs) == 1:
            out[name] = sha(arrs[0])
        else:
            out[name + "_hi"], out[name + "_lo"] = sha(arrs[0]), sha(arrs[1])

    put("residual", *s.residual(zh, zl))
    put("segs", np.concatenate([np.asarray(g) for g in s.get_segs()]))
    put("series_centre", *s.series(zh, zl, 0))
    put("series_horizon", *s.series(zh, zl, 1))
    e = np.zeros(s.size)
    e[-1] = 1.0
    # dense double-double Newton step, then chord and lu_solve_t with its LU
    dh, dl, Fh, Fl, pmin, pmax = s.newton(zh, zl)
    put("newton_dz", dh, dl)
    put("newton_piv", np.array([pmin, pmax]))
    th, tl = dd_add(zh, zl, np.asarray(dh), np.asarray(dl))
    ch, cl, _, _ = s.chord(th, tl)
    put("newton_chord", ch, cl)
    put("newton_solve_t", *s.lu_solve_t(e, np.zeros(s.size)))
    # dense f64 Jacobian, mixed Newton step (default: f64x4, f64 LU), chord and lu_solve_t with its LU
    put("jacobian_f64", s.jacobian_f64(zh, zl))
    dh, dl, Fh, Fl, pmin, pmax = s.newton_mixed(zh, zl)
    put("mixed_dz", dh, dl)
    put("mixed_F", Fh, Fl)
    put("mixed_piv", np.array([pmin, pmax]))
    th, tl = dd_add(zh, zl, np.asarray(dh), np.asarray(dl))
    ch, cl, _, _ = s.chord(th, tl)
    put("mixed_chord", ch, cl)
    put("mixed_solve_t", *s.lu_solve_t(e, np.zeros(s.size)))
    print(f"  M = {M} (N = {s.size}, from {m['source']}): {len(out)} outputs in {time.time() - t0:.0f} s", flush=True)
    return out


def ref(opts):
    Ms = [int(v) for v in str(opts.get("Ms", "8,16,48,64,80,96,128")).split(",")]
    threads = int(opts.get("threads", 2))
    allh = {}
    if "save" in opts:
        # incremental: one M at a time, so an interrupted run keeps what it has (M already in the file are skipped)
        if os.path.exists(opts["save"]):
            allh = json.load(open(opts["save"]))
        for M in Ms:
            if str(M) in allh:
                print(f"  M = {M}: already saved")
                continue
            allh[str(M)] = ref_hashes(M, threads)
            json.dump(allh, open(opts["save"] + ".tmp", "w"), indent=1)
            os.replace(opts["save"] + ".tmp", opts["save"])
        print(f"saved {sum(len(v) for v in allh.values())} hashes to {opts['save']}")
        return 0
    for M in Ms:
        allh[str(M)] = ref_hashes(M, threads)
    want = json.load(open(opts["compare"]))
    bad = 0
    n = 0
    for M, h in allh.items():
        if M not in want:
            print(f"  M = {M}: not in the reference")
            bad += 1
            continue
        for k, v in h.items():
            n += 1
            same = want[M].get(k) == v
            bad += not same
            print(f"  M = {M:>3} {k:24s} {v[:16]} {'bit-identical' if same else 'DIFFERS from ' + str(want[M].get(k))[:16]}")
    print(f"PASS: {n} outputs bit-identical" if not bad else f"FAIL: {bad} of {n} outputs differ")
    return 1 if bad else 0


# ------------------------------------------------------------------ block vs dense
def relmax(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


VARIANTS = {"f64x4": dict(tangents=4), "f64x1": dict(tangents=1), "dd4": dict(tangents=4, dd_base=True)}


def block_vs_dense(M, threads, mutate=0, quiet=False, variant="f64x4"):
    kw = VARIANTS[variant]
    s, zh0, zl0, m = solver_at(M, threads)
    e = np.zeros(s.size)
    e[-1] = 1.0
    r = dict(M=M, N=s.size, mutate=mutate, variant=variant, points={})
    for label, (zh, zl) in (("state", (zh0, zl0)), ("perturbed", perturb(zh0, zl0, 1e-6, 1e-8, 1))):
        t = time.time()
        dh, dl, Fh, Fl, p0, p1 = s.newton_mixed(zh, zl, block=False, **kw)
        td = time.time() - t
        dz_d = np.asarray(dh) + np.asarray(dl)
        y_d = np.add(*s.lu_solve_t(e, np.zeros(s.size)))
        th, tl = dd_add(zh, zl, np.asarray(dh), np.asarray(dl))
        c_d = np.add(*s.chord(th, tl)[:2])
        t = time.time()
        bh, bl, Gh, Gl, q0, q1 = s.newton_mixed(zh, zl, block=True, block_mutate=mutate, **kw)
        tb = time.time() - t
        nbytes, isblock, _, _ = s.lu_info()
        dz_b = np.asarray(bh) + np.asarray(bl)
        y_b = np.add(*s.lu_solve_t(e, np.zeros(s.size)))
        c_b = np.add(*s.chord(th, tl)[:2])
        same_F = bool(np.array_equal(np.asarray(Fh).view(np.uint64), np.asarray(Gh).view(np.uint64))
                      and np.array_equal(np.asarray(Fl).view(np.uint64), np.asarray(Gl).view(np.uint64)))
        # rounding floor of the dense f64 LU: the same right-hand side with the dense double-double LU of Jf
        Jf = np.asarray(s.jacobian_f64(zh, zl, **kw))
        s.lu_factor(Jf)
        nF = (-np.asarray(Fh), -np.asarray(Fl))
        dz_dd = np.add(*s.lu_solve(*nF))
        y_dd = np.add(*s.lu_solve_t(e, np.zeros(s.size)))
        floor = max(relmax(dz_d, dz_dd), relmax(y_d, y_dd))
        p = dict(F=dd_norm(Fh, Fl), residual_bit_identical=same_F, dz=relmax(dz_b, dz_d), sens=relmax(y_b, y_d),
                 chord=relmax(c_b, c_d), floor_dz=relmax(dz_d, dz_dd), floor_sens=relmax(y_d, y_dd),
                 dDelta_dense=float(dz_d[-1]), dDelta_block=float(dz_b[-1]), piv_dense=[p0, p1], piv_block=[q0, q1],
                 t_dense=round(td, 2), t_block=round(tb, 2), lu_mb_block=round(nbytes / 2 ** 20, 1),
                 lu_mb_dense=round(16 * (8 * ((s.size + 7) // 8)) * s.size / 2 ** 20, 1))
        lim = max(100 * floor, 1e-12)
        p["limit"] = lim
        p["passed"] = bool(same_F and p["dz"] <= lim and p["sens"] <= lim and p["chord"] <= lim)
        r["points"][label] = p
        if not quiet:
            print(f"  M = {M} (N = {s.size}) {variant} {label:9s}: |F| {p['F']:.1e}, residual bit-identical {same_F}; block vs dense: "
                  f"dz {p['dz']:.1e}, dDelta/dF {p['sens']:.1e}, chord {p['chord']:.1e}; floor (dense f64 vs dd LU) "
                  f"{p['floor_dz']:.1e} / {p['floor_sens']:.1e}; limit {lim:.0e} -> {'pass' if p['passed'] else 'FAIL'}; "
                  f"{td:.1f} s dense, {tb:.1f} s block; LU {p['lu_mb_block']} MB block vs {p['lu_mb_dense']} MB dense",
                  flush=True)
    r["passed"] = all(p["passed"] for p in r["points"].values())
    return r


def block(opts):
    Ms = [int(v) for v in str(opts.get("Ms", "16,48,80,128")).split(",")]
    threads = int(opts.get("threads", 2))
    mut = MUT[opts["mutate"]] if opts.get("mutate") else 0
    rs = [block_vs_dense(M, threads, mut, variant=v) for M in Ms for v in str(opts.get("variants", "f64x4")).split(",")]
    ok = all(r["passed"] for r in rs)
    print(("MUTATED (" + opts["mutate"] + "): " if mut else "") + ("PASS" if ok else "FAIL"))
    if "json" in opts:
        json.dump(rs, open(opts["json"], "w"), indent=1)
    if opts.get("expect_fail"):
        print("  expected to fail:", "it did (the control fires)" if not ok else "it PASSED: the control does not fire")
        return 0 if not ok else 1
    return 0 if ok else 1


def controls(opts):
    fired = []
    for mname in MUT:
        r = block_vs_dense(48, int(opts.get("threads", 2)), MUT[mname], quiet=False)
        fired.append(not r["passed"])
        print(f"  control {mname}: {'fires' if not r['passed'] else 'DOES NOT FIRE'}")
    print(f"controls: {sum(fired)} of {len(fired)} fire", "PASS" if all(fired) else "FAIL")
    return 0 if all(fired) else 1


def zdiff(a, b):
    dh, dl = dd_add(a[0], a[1], -b[0], -b[1])
    d = dh + dl
    return abs(float(d[-1])) / abs(float(a[0][-1])), float(np.abs(d[:-1]).max() / np.abs(a[0][:-1]).max())


def converge(opts):
    M = int(opts.get("M", 80))
    threads = int(opts.get("threads", 2))
    tol = float(opts.get("tol", 1e-28))
    s, zh0, zl0, m = solver_at(M, threads)
    zh, zl = perturb(zh0, zl0, 1e-6, 1e-8, 1)
    out = dict(M=M, N=s.size, tol=tol, runs={})
    sols = {}
    for name, blk, start in (("dense", False, (zh, zl)), ("block", True, (zh, zl)),
                             ("control", False, perturb(zh0, zl0, -1e-6, 1e-8, 2))):
        print(f" {name}:", flush=True)
        t = time.time()
        a, b, hist = newton_solve(s, *start, tol=tol, it_max=12, verbose=True, jac="f64", mixed=dict(block=blk))
        a, b = np.asarray(a), np.asarray(b)
        sols[name] = (a, b)
        out["runs"][name] = dict(newton_calls=len(hist), F=dd_norm(*s.residual(a, b)), seconds=round(time.time() - t, 1),
                                 Delta=float(a[-1]), Delta_lo=float(b[-1]))
    ctl = zdiff(sols["dense"], sols["control"])
    dD, dZ = zdiff(sols["dense"], sols["block"])
    lim = (max(1e-27, 10 * ctl[0]), max(1e-27, 10 * ctl[1]))
    ok = all(r["F"] <= tol for r in out["runs"].values()) and dD <= lim[0] and dZ <= lim[1]
    out.update(block_vs_dense_Delta=dD, block_vs_dense_z=dZ, control_Delta=ctl[0], control_z=ctl[1], passed=ok)
    for k, r in out["runs"].items():
        print(f"  {k:8s}: {r['newton_calls']} Newton calls, |F| {r['F']:.1e}, {r['seconds']} s, Delta {r['Delta']!r} {r['Delta_lo']:+.3e}")
    print(f"  block vs dense: Delta {dD:.1e}, max|dz|/max|z| {dZ:.1e}; control (dense, two starts) {ctl[0]:.1e}, {ctl[1]:.1e}"
          f" -> {'PASS' if ok else 'FAIL'}")
    if "json" in opts:
        json.dump(out, open(opts["json"], "w"), indent=1)
    return 0 if ok else 1


# ------------------------------------------------------------------ lift and profile at large M
class Timed:
    """a Solver proxy that times newton_mixed and residual calls"""

    def __init__(self, s):
        self.s = s
        self.t = {"newton_mixed": [], "residual": []}

    def __getattr__(self, k):
        f = getattr(self.s, k)
        if k in self.t:
            def g(*a, **kw):
                t = time.time()
                r = f(*a, **kw)
                self.t[k].append(time.time() - t)
                return r
            return g
        return f


def lift(opts):
    """two-pass lift through lowD_dd.solve_point (every accepted Newton step checkpointed; a rerun resumes, and skips a
    pass whose converged state exists), then a timing of one Newton step at the solution (prof)"""
    import argparse
    import lowD_dd
    M = int(opts["M"])
    threads = int(opts.get("threads", 2))
    tol = float(opts.get("tol", 1e-28))
    tol1 = float(opts.get("tol1", 1e-14))
    tag = opts.get("tag", "highm")
    start = opts.get("start", STATE(80))
    outp = opts.get("out", os.path.join(HERE, "data", "lowD_highM", "lift_D3.7.jsonl"))
    points = outp.replace(".jsonl", "_points.jsonl")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    zh, zl, m = load(start)
    D, Mo, nodes = m["D"], m["M"], list(m["nodes"])
    if os.path.exists(outp) and any(json.loads(ln).get("M") == M and json.loads(ln).get("tag", "highm") == tag
                                    for ln in open(outp) if ln.strip()):
        print(f"D {D} M {M}: already in {outp}")
        return 0
    kw = dict(nodes=nodes, kt=m["kt"], kc=m["kc"], ks=m["ks"], threads=threads)
    log = lambda t: print(t, flush=True)  # noqa: E731

    def ns(tg, tl):
        return argparse.Namespace(out=points, tag=tg, resume=True, kt=m["kt"], kcs=m["kc"], ks=m["ks"], tol_step=1e-30,
                                  hmax=0.05, solver="newton", it_max=int(opts.get("it_max", 14)), tol=tl, accept=tl, lm=0,
                                  jac="f64", threads=threads, svd=0, dn_its=2, w=24)

    def last_rec(tg_tol):
        r = None
        for ln in open(points):
            x = json.loads(ln)
            if x["M"] == M and abs(x["D"] - D) < 1e-9 and x.get("converged") and (x["F"] <= tg_tol):
                r = x
        return r

    t0 = time.time()
    if M != Mo:
        zh, zl = resample_z(zh + zl, Mo, M, len(nodes))
    s = make_solver(D, M, **kw)
    if opts.get("no_replan"):
        ok, zh, zl = lowD_dd.solve_point(D, M, zh, zl, f"lift from {os.path.basename(start)}, start plan", ns(tag, tol), nodes,
                                         STATES, log, segs=s.plan(zh, zl, 1e-30, 0.05))
        passes = [last_rec(tol)] if ok else []
    else:
        # pass 1 on the resampled start's own plan, to tol1; pass 2 on the plan of the pass-1 state, to tol
        ok, zh, zl = lowD_dd.solve_point(D, M, zh, zl, f"lift from {os.path.basename(start)}, pass 1", ns(tag + "_p1", tol1),
                                         nodes, STATES, log, segs=s.plan(zh, zl, 1e-30, 0.05))
        passes = [last_rec(tol1)] if ok else []
        if ok:
            segs2 = s.plan(zh, zl, 1e-30, 0.05)
            s.set_segs(segs2)
            F2 = dd_norm(*s.residual(zh, zl))
            if not (np.isfinite(F2) and F2 <= 1e-12):
                print(f" re-planned residual {F2:.1e}: pass 2 on the pass-1 plan instead", flush=True)
                segs2 = load(os.path.join(STATES, f"{tag}_p1_D{D:.4f}_M{M}.npz"))[2]["segs"]
            ok, zh, zl = lowD_dd.solve_point(D, M, zh, zl, "pass 2 (re-planned)", ns(tag, tol), nodes, STATES, log, segs=segs2)
            if ok:
                passes.append(last_rec(tol))
    if not ok:
        print(f"D {D} M {M}: NOT converged", flush=True)
        return 1
    fin = passes[-1]
    # one Newton step at the solution, timed by part (the same Jacobian without the tau derivative for its share)
    zh, zl, mm = load(os.path.join(STATES, f"{tag}_D{D:.4f}_M{M}.npz"))
    s.set_segs(mm["segs"])
    t = time.time(); s.residual(zh, zl); t_res = time.time() - t
    t = time.time(); s.newton_mixed(zh, zl); t_nm = time.time() - t
    nbytes, isblock, t_jac, t_fac = s.lu_info()
    t_nodtau = None
    if isblock and not opts.get("no_dtau_timing"):
        s.newton_mixed(zh, zl, jac_mutate=1)
        t_nodtau = s.lu_info()[2]
    rec = dict(D=D, M=M, N=s.size, tag=tag, Delta=fin["Delta"], Delta_lo=fin["Delta_lo"], F=fin["F"], converged=True,
               start=os.path.basename(start), start_M=Mo, nodes=nodes, kt=m["kt"], kc=m["kc"], ks=m["ks"], tol_step=1e-30,
               hmax=0.05, steps=fin["steps"], steps_pass1=passes[0]["steps"] if len(passes) > 1 else None,
               its=[p["its"] for p in passes], tail_c=fin["tail_c"], tail_h=fin["tail_h"],
               t_its=[p.get("t_its") for p in passes], s_residual=round(t_res, 2), s_newton_step=round(t_nm, 2),
               s_jacobian=None if t_jac is None else round(t_jac, 2), s_factor=None if t_fac is None else round(t_fac, 2),
               s_jacobian_no_dtau=None if t_nodtau is None else round(t_nodtau, 2), lu_mb=round(nbytes / 2 ** 20, 1),
               lu_block=isblock, load=round(os.getloadavg()[0], 2), peak_rss_mb=peak_rss_mb(), threads=threads,
               wall=round(time.time() - t0, 1), jac="f64")
    with open(outp, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"D {D} M {M}: Delta {rec['Delta']:.16f} ({rec['Delta_lo']:+.2e}) |F| {rec['F']:.1e}; steps {rec['steps_pass1']} -> "
          f"{rec['steps']}; tails {rec['tail_c']:.0e} {rec['tail_h']:.0e}; Newton step {rec['s_newton_step']} s (residual "
          f"{rec['s_residual']} s, Jacobian {rec['s_jacobian']} s, factor {rec['s_factor']} s, Jacobian without d/dtau "
          f"{rec['s_jacobian_no_dtau']} s); LU {rec['lu_mb']} MB; peak RSS {rec['peak_rss_mb']} MB; load {rec['load']}", flush=True)
    return 0


def profile(opts):
    M = int(opts["M"])
    threads = int(opts.get("threads", 2))
    reps = int(opts.get("reps", 2))
    path = opts.get("start")
    s, zh, zl, m = solver_at(M, threads, path=path)
    out = dict(M=M, N=s.size, threads=threads, state=m["source"], residual=[], newton_mixed=[], jac=[], factor=[])
    for _ in range(reps):
        t = time.time(); s.residual(zh, zl); out["residual"].append(round(time.time() - t, 2))
        t = time.time(); s.newton_mixed(zh, zl, block=True); out["newton_mixed"].append(round(time.time() - t, 2))
        _, _, tj, tf = s.lu_info()
        out["jac"].append(round(tj, 2))
        out["factor"].append(round(tf, 2))
        # the same Jacobian without the tau derivative (Ctx::mutate 1): its time difference is the tau derivative's
        s.newton_mixed(zh, zl, block=True, jac_mutate=1)
        out.setdefault("jac_no_dtau", []).append(round(s.lu_info()[2], 2))
    out["peak_rss_mb"] = peak_rss_mb()
    print(json.dumps(out), flush=True)
    if "out" in opts:
        with open(opts["out"], "a") as f:
            f.write(json.dumps(out) + "\n")
    return 0


def main():
    pos, opts = opts_of(sys.argv[1:])
    cmd = pos[0] if pos else "block"
    return {"ref": ref, "block": block, "controls": controls, "converge": converge, "lift": lift, "profile": profile}[cmd](opts)


if __name__ == "__main__":
    sys.exit(main())
