#!/usr/bin/env python3
"""The sigma-model's Delta at one coupling eta (D = 4 by default) on M tau points: seed, Taylor plan, double-double
Newton, re-plan from the solution and Newton again, optionally quad-double Newton; one record per run.

Seeds (one of):
  --sigdss FILE.npy --kc_seed K     a sigdss state (solver/matter/sigdss_solver.py, Fourier to harmonic K) at this eta,
                                    regridded to the nodes (sigdss's own integration) and to the M points
  --from STATE.npz                  a state of this script (any M: resampled; any eta: zero-order continuation in eps)
  --from STATE1.npz --from2 STATE2.npz
                                    the secant through two states at other couplings, extrapolated linearly in eps;
                                    through two states at one coupling and other D, linearly in D (chain.py --Ds)

    PYTHONPATH=<site> python3 run_eta.py --eta 7 --M 32 --sigdss <raw>/I631-win/data/...kc39_eta7....npy --kc_seed 39 \
        --out <raw>/I639-mac/data --tag eta7_M32 [--qd] [--threads 4]

--solver band uses the deflated Newton smdriver.newton_band in place of newton_solve, for a start on the near-null
band of the Jacobian (SIGMA_DSS.md section 6; --band_tol, --radius).

--qd continues a converged double-double state with Newton in quad-double (--qd_jac dd: the exact quad-double
Jacobian). With --qd_accept B the run's exit status is that of the quad-double stage: 0 when the quad-double
residual of its final state is at most B and the Newton's history shows a start inside its basin (qd_basin). This is the hand-over of #928: past D 12.4 at M = 32 the double-double
Newton stalls near 1e-27 on the band, whose rows the double-double residual no longer resolves, so --tol is set to
about 3e-27 and the 1e-28 bound is met in quad-double. The record's qd entry holds the quad-double residuals, the
final one (F_final), the hand-over tolerance, the double-double residual at the hand-over (handover_F), each step's
max |F| and max |dz|, and whether the state was accepted. --handover_max H also hands over a state whose
double-double stage stalled above --tol, when its residual is at most H.

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

# enough digits for the sums of a state's parts (quad-double: about 64 digits); at mpmath's default of 53 bits the
# records' Delta string held the hi part only (the runs of #639 before this line; Delta_hi and Delta_lo are exact)
mp.mp.dps = 80


def qd_basin(hq, Fq, bound, first_rise=1.0):
    """Whether the quad-double Newton's history shows a start inside its basin (#928, the limit (2') of the raw
    folder's qd_protocol.md): max |F| is at most the bound within 4 steps and at the end; it never rises from one
    step to the next; from the second step on it is at most 1e10 times the square of the step before, or below
    1e-55, the Newton's own tolerance; and the step's size max |dz| is at most 1e10 times the square of the step
    before at every step after the first. The first step is exempt in max |F| because a displacement along the
    near-null band shows in the residual only at second order: that step is quadratic in dz, not in |F|. With
    first_rise above 1 the first step may also raise max |F|, by less than that factor.
    Returns (all hold, each one)."""
    taken = [h for h in hq if "F_next" in h and "failed" not in h]
    r = [hq[0]["F"]] + [h["F_next"] for h in taken]
    dz = [h["dz"] for h in taken]
    k = next((i for i, v in enumerate(r) if v <= bound), None)
    c = dict(within_4_steps=k is not None and k <= 4,
             no_rise=all(q <= p * (first_rise if i == 0 else 1.0) for i, (p, q) in enumerate(zip(r, r[1:]))),
             F_quadratic=all(r[i + 1] <= max(1e10 * r[i] ** 2, 1e-55) for i in range(1, len(r) - 1)),
             dz_quadratic=all(dz[i] <= 1e10 * dz[i - 1] ** 2 for i in range(1, len(dz))),
             final=bool(Fq <= bound))
    return all(c.values()), c


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
    ap.add_argument("--tol", type=float, default=1e-28)
    ap.add_argument("--it", type=int, default=16)
    ap.add_argument("--jac", default="dd", help="dd (exact double-double Jacobian) or mixed (f64, staircase)")
    ap.add_argument("--jac_final", default="dd", help="the Jacobian of the Newton on the final plan")
    ap.add_argument("--chord", type=int, default=0)
    ap.add_argument("--replan", type=int, default=1, help="re-plan the Taylor steps after every Newton step")
    ap.add_argument("--solver", default="newton", choices=["newton", "band"],
                    help="newton (newton_solve) or band (newton_band: the near-null band deflated)")
    ap.add_argument("--band_tol", type=float, default=1e-6, help="band: singular values below this times the largest")
    ap.add_argument("--radius", type=float, default=0.0, help="band: the first limit on a band step (0: 1%% of |z|)")
    ap.add_argument("--qd", action="store_true")
    ap.add_argument("--qd_jac", default="mixed", help="dd (exact quad-double Jacobian), mixed or chord")
    ap.add_argument("--qd_it", type=int, default=8)
    ap.add_argument("--qd_accept", type=float, default=0.0,
                    help="with --qd: the exit status is 0 when the final quad-double max |F| is at most this")
    ap.add_argument("--qd_first_rise", type=float, default=1.0,
                    help="with --qd: the first quad-double step may raise max |F| by less than this factor")
    ap.add_argument("--handover_max", type=float, default=0.0,
                    help="with --qd: hand a double-double state over that stalled above --tol but at most here")
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
            if e1 != e2:
                w = (eps - e1) / (e2 - e1)
            else:
                # one coupling, two D (#652): the secant in D
                D1, D2 = float(m1.get("D", 4.0)), float(m2.get("D", 4.0))
                w = (a.D - D1) / (D2 - D1)
                seed.update(D_src=D1, D_src2=D2)
            zh, zl = (1 - w) * (zh + zl) + w * z2h, np.zeros_like(zh)
            seed.update(src2=os.path.basename(a.from2), eta_src2=m2["eta"], w=w)
    F0 = sd.fnorm(s.residual([zh, zl]))
    log(f"seed {seed}: Delta {zh[-1]:.12f}, |F| (single-step plan) {F0:.2e}")
    tp = time.time()
    s.set_segs(s.plan(zh, zl, a.plan_tol, a.hmax))
    steps0 = [len(g) - 1 for g in s.get_segs()]
    F1 = sd.fnorm(s.residual([zh, zl]))
    log(f"plan from the seed: {sum(steps0)} steps {steps0} ({time.time() - tp:.1f} s); |F| {F1:.2e}")
    replans = []

    def replan(zh_, zl_):
        # a seed that is not a solution has singular components near the horizon (x - c small), which need short
        # steps; each Newton iterate is smoother, so the steps are planned again from it
        s.set_segs(s.plan(zh_, zl_, a.plan_tol, a.hmax))
        replans.append(sum(len(g) - 1 for g in s.get_segs()))
        log(f"    re-planned: {replans[-1]} steps")
    ckpt = os.path.join(a.out, f"sm_{a.tag}_ckpt.npz")

    def keep(zh_, zl_):
        if a.solver == "band":
            # a band walk is long: the state after every step, so that a stopped run can be continued (--from)
            sd.save(ckpt, zh_, zl_, dict(eta=a.eta, D=a.D, M=a.M, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks,
                                         segs=s.get_segs(), tag=a.tag + "_ckpt"))

    def after(zh_, zl_):
        if a.replan:
            replan(zh_, zl_)
        keep(zh_, zl_)

    def solve(zh_, zl_, jac, after_step=None):
        if a.solver == "band":
            return sd.newton_band(s, zh_, zl_, tol=a.tol, it_max=a.it, band_tol=a.band_tol, radius=a.radius or None,
                                  log=log, after_step=after_step)
        return sd.newton_solve(s, zh_, zl_, tol=a.tol, it_max=a.it, log=log, jac=jac, chord=a.chord,
                               after_step=after_step)
    zh, zl, h1 = solve(zh, zl, a.jac, after)
    # re-plan from the solution and converge on that plan (the plan depends on the state; hp plans once per point)
    tp = time.time()
    s.set_segs(s.plan(zh, zl, a.plan_tol, a.hmax))
    steps1 = [len(g) - 1 for g in s.get_segs()]
    F2 = sd.fnorm(s.residual([zh, zl]))
    log(f"plan from the solution: {sum(steps1)} steps ({time.time() - tp:.1f} s); |F| {F2:.2e}")
    zh, zl, h2 = solve(zh, zl, a.jac_final, keep)
    conv = bool(h2 and h2[-1].get("converged"))
    Fend = h2[-1]["F"] if h2 else float("nan")
    Delta = mp.mpf(zh[-1]) + mp.mpf(zl[-1])
    rec = dict(tag=a.tag, eta=a.eta, D=a.D, M=a.M, K=K, nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, plan_tol=a.plan_tol,
               hmax=a.hmax, seed=seed, F_seed=F0, steps_seed=sum(steps0), replans=replans, jac=a.jac, solver=a.solver,
               jac_final=a.jac_final, chord=a.chord, steps=sum(steps1), steps_seg=steps1, converged=conv, F=Fend,
               Delta=mp.nstr(Delta, 34), Delta_hi=float(zh[-1]), Delta_lo=float(zl[-1]),
               newton=[dict((k, v) for k, v in h.items() if k != "piv") for h in h1 + h2],
               constraint=sd.constraint(s, zh + zl, nodes), tail_centre=sd.series_tail(s, zh, zl, 0, nodes),
               tail_horizon=sd.series_tail(s, zh, zl, 1, nodes),
               q_top=[(k, v) for k, v in sorted(sd.harmonics_q(zh + zl, a.M).items())][-4:],
               t=round(time.time() - t0, 1), when=sd.now(), threads=a.threads)
    log(f"double-double: converged {conv}, |F| {Fend:.2e}, Delta {rec['Delta']}, constraint {rec['constraint']:.2e}, "
        f"tails {rec['tail_centre']:.1e} {rec['tail_horizon']:.1e}")
    z4 = None
    over = bool(a.qd and not conv and a.handover_max and Fend <= a.handover_max)
    if over:
        log(f"hand-over above --tol: the double-double stage ended at |F| {Fend:.2e} (<= {a.handover_max:g})")
    if a.qd and (conv or over):
        z4, hq = sd.newton_qd(s, sd.qparts(zh, zl), tol=1e-55, it_max=a.qd_it, log=log, jac=a.qd_jac,
                              first_rise=a.qd_first_rise)
        Dq = sum(mp.mpf(p[-1]) for p in z4)
        Fq = sd.fnorm(s.residual(z4))
        basin_ok, basin = qd_basin(hq, Fq, a.qd_accept, a.qd_first_rise) if a.qd_accept else (False, {})
        rec.update(qd=dict(F=[h["F"] for h in hq], converged=bool(hq[-1].get("converged")), Delta=mp.nstr(Dq, 60),
                           Delta_qd_minus_dd=mp.nstr(Dq - Delta, 6), F_final=Fq, jac=a.qd_jac, handover_tol=a.tol,
                           handover_F=Fend, handover_max=a.handover_max, first_rise=a.qd_first_rise, steps=[dict(F=h["F"], F_next=h.get("F_next"),
                                                                                    dz=h.get("dz")) for h in hq],
                           basin=basin, accepted=bool(a.qd_accept and basin_ok), t=[h["t"] for h in hq]))
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
    if a.qd and a.qd_accept:
        sys.exit(0 if (conv or over) and rec["qd"]["accepted"] else 1)
    sys.exit(0 if conv else 1)


if __name__ == "__main__":
    main()
