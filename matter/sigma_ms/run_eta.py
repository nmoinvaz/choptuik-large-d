#!/usr/bin/env python3
"""The sigma-model's Delta at one coupling eta (D = 4 by default) on M tau points: seed, Taylor plan, double-double
Newton, re-plan from the solution and Newton again, optionally quad-double Newton; one record per run.

Seeds (one of):
  --sigdss FILE.npy --kc_seed K     a sigdss state (solver/matter/sigdss_solver.py, Fourier to harmonic K) at this eta,
                                    regridded to the nodes (sigdss's own integration) and to the M points
  --from STATE.npz                  a state of this script (any M: resampled; any eta: zero-order continuation in eps)
  --from STATE1.npz --from2 STATE2.npz
                                    the secant through two states at other couplings, extrapolated linearly in eps

    PYTHONPATH=<site> python3 run_eta.py --eta 7 --M 32 --sigdss <raw>/I631-win/data/...kc39_eta7....npy --kc_seed 39 \
        --out <raw>/I639-mac/data --tag eta7_M32 [--qd] [--threads 4]

Writes <out>/sm_<tag>.npz (the state, double-double, and quad-double parts with --qd) and appends a record to
<out>/sm_records.jsonl: Delta, max |F|, Newton history, Taylor steps, the constraint diagnostic, the series tails, the
centre datum's top harmonics, timings. Exit status 1 when Newton does not reach --tol.
"""
import argparse
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smdriver as sd  # noqa: E402


def eps_of(eta):
    return 0.0 if eta in (None, "inf") or eta == float("inf") else 1.0 / float(eta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", default="inf")
    ap.add_argument("--D", type=float, default=4.0)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--sigdss", default="")
    ap.add_argument("--kc_seed", type=int, default=39)
    ap.add_argument("--from", dest="src", default="")
    ap.add_argument("--from2", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kt", type=int, default=36)
    ap.add_argument("--kc", type=int, default=80)
    ap.add_argument("--ks", type=int, default=80)
    ap.add_argument("--nodes", default="")
    ap.add_argument("--plan_tol", type=float, default=sd.PLAN_TOL)
    ap.add_argument("--hmax", type=float, default=sd.PLAN_HMAX)
    ap.add_argument("--tol", type=float, default=1e-27)
    ap.add_argument("--it", type=int, default=16)
    ap.add_argument("--qd", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, f"sm_{a.tag}.log"), "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    eta = None if a.eta == "inf" else float(a.eta)
    eps = eps_of(eta)
    nodes = [float(x) for x in a.nodes.split(",")] if a.nodes else list(sd.NODES)
    K = len(nodes)
    log(f"start {sd.now()}: eta {a.eta}, D {a.D}, M {a.M}, K {K}, kt {a.kt}, kc {a.kc}, ks {a.ks}, threads {a.threads}")
    t0 = time.time()
    s = sd.make_solver(eta, a.M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads, D=a.D)
    seed = {}
    if a.sigdss:
        zh, zl = sd.seed_from_sigdss(a.sigdss, eps, a.kc_seed, a.M, nodes, D=a.D)
        seed = dict(sigdss=os.path.basename(a.sigdss), kc_seed=a.kc_seed, Delta_seed=float(zh[-1]))
    else:
        (z1, m1) = sd.load(a.src)
        if m1["nodes"] != nodes:
            raise SystemExit("--from: the state's nodes differ")
        zh, zl = np.asarray(z1[0]), np.asarray(z1[1])
        if m1["M"] != a.M:
            zh, zl = sd.resample_z(zh + zl, m1["M"], a.M, K), np.zeros(sd.sizes(a.M, K))
        seed = dict(src=os.path.basename(a.src), eta_src=m1["eta"], M_src=m1["M"])
        if a.from2:
            (z2, m2) = sd.load(a.from2)
            z2h = np.asarray(z2[0]) + np.asarray(z2[1])
            if m2["M"] != a.M:
                z2h = sd.resample_z(z2h, m2["M"], a.M, K)
            e1, e2 = eps_of(m1["eta"]), eps_of(m2["eta"])
            w = (eps - e1) / (e2 - e1)
            zh, zl = (1 - w) * (zh + zl) + w * z2h, np.zeros_like(zh)
            seed.update(src2=os.path.basename(a.from2), eta_src2=m2["eta"], w=w)
    F0 = sd.fnorm(s.residual([zh, zl]))
    log(f"seed {seed}: Delta {zh[-1]:.12f}, |F| (single-step plan) {F0:.2e}")
    tp = time.time()
    s.set_segs(s.plan(zh, zl, a.plan_tol, a.hmax))
    steps0 = [len(g) - 1 for g in s.get_segs()]
    F1 = sd.fnorm(s.residual([zh, zl]))
    log(f"plan from the seed: {sum(steps0)} steps {steps0} ({time.time() - tp:.1f} s); |F| {F1:.2e}")
    zh, zl, h1 = sd.newton_solve(s, zh, zl, tol=a.tol, it_max=a.it, log=log)
    # re-plan from the solution and converge on that plan (the plan depends on the state; hp plans once per point)
    tp = time.time()
    s.set_segs(s.plan(zh, zl, a.plan_tol, a.hmax))
    steps1 = [len(g) - 1 for g in s.get_segs()]
    F2 = sd.fnorm(s.residual([zh, zl]))
    log(f"plan from the solution: {sum(steps1)} steps ({time.time() - tp:.1f} s); |F| {F2:.2e}")
    zh, zl, h2 = sd.newton_solve(s, zh, zl, tol=a.tol, it_max=a.it, log=log)
    conv = bool(h2 and h2[-1].get("converged"))
    Fend = h2[-1]["F"] if h2 else float("nan")
    Delta = mp.mpf(zh[-1]) + mp.mpf(zl[-1])
    rec = dict(tag=a.tag, eta=a.eta, D=a.D, M=a.M, K=K, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, plan_tol=a.plan_tol,
               hmax=a.hmax, seed=seed, F_seed=F0, steps=sum(steps1), steps_seg=steps1, converged=conv, F=Fend,
               Delta=mp.nstr(Delta, 34), Delta_hi=float(zh[-1]), Delta_lo=float(zl[-1]),
               newton=[dict((k, v) for k, v in h.items() if k != "piv") for h in h1 + h2],
               constraint=sd.constraint(s, zh + zl, nodes), tail_centre=sd.series_tail(s, zh, zl, 0, nodes),
               tail_horizon=sd.series_tail(s, zh, zl, 1, nodes),
               q_top=[(k, v) for k, v in sorted(sd.harmonics_q(zh + zl, a.M).items())][-4:],
               t=round(time.time() - t0, 1), when=sd.now(), threads=a.threads)
    log(f"double-double: converged {conv}, |F| {Fend:.2e}, Delta {rec['Delta']}, constraint {rec['constraint']:.2e}, "
        f"tails {rec['tail_centre']:.1e} {rec['tail_horizon']:.1e}")
    z4 = None
    if a.qd and conv:
        z4, hq = sd.newton_qd(s, sd.qparts(zh, zl), tol=1e-55, it_max=6, log=log)
        Dq = sum(mp.mpf(p[-1]) for p in z4)
        rec.update(qd=dict(F=[h["F"] for h in hq], converged=bool(hq[-1].get("converged")), Delta=mp.nstr(Dq, 60),
                           Delta_qd_minus_dd=mp.nstr(Dq - Delta, 6)))
        log(f"quad-double: |F| {hq[-1]['F']:.2e}, Delta {rec['qd']['Delta']}, minus dd {rec['qd']['Delta_qd_minus_dd']}")
    meta = dict(eta=a.eta, D=a.D, M=a.M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, segs=s.get_segs(), tag=a.tag)
    if z4 is not None:
        sd.save(os.path.join(a.out, f"sm_{a.tag}.npz"), z4[0], z4[1], meta, z4[2], z4[3])
    else:
        sd.save(os.path.join(a.out, f"sm_{a.tag}.npz"), zh, zl, meta)
    rec["t"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "sm_records.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"done {sd.now()} ({rec['t']} s)")
    sys.exit(0 if conv else 1)


if __name__ == "__main__":
    main()
