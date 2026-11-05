#!/usr/bin/env python3
"""The quad-double jet (Solver.residual_jet_q, multidouble 2.0.0's QJet lanes) against the quad-double residual on one
stored large-D state z_0 of the pinned chain (notes/QUAD_DOUBLE.md 13.19), beside the double-double jet
(Solver.residual_jet). The double-double jet's order-1 coefficient is off by 0.03-1.4 times eps_dd G |dF/dD| (13.19,
the horizon series' cancellation amplified by the mass mode G = 0.7^-(n-1)), which the quad-double one should remove.

The quantities, all on the bare kernel (the freeze projection of --kfreeze, no pin offset), max norms:
- e0q, e0d: order 0 of the quad-double and the double-double jet at z_0 against Solver.residual_q at z_0.
- z1d: the chain's z_1 on the bare kernel, the double-double jet's F_1 at fixed z_0 through the double-double LU with
  one refinement step (anm_run.py coefficients). Its errors in dF/dD and J v are common-mode and cancel; through the
  quad-double LU instead (the first version of this check) they do not, and its z1d lay along a direction where the
  double-double jet's order-1 error was below R's resolution (notes/QUAD_DOUBLE.md 13.20).
- R: d/dh F(z_0 + h z1d, D + h) at h = 0 by central differences of residual_q, Richardson-combined at steps 2^-29
  and 2^-30 (anm_run.py order1_r, without the held slope); dR its change at steps 2^-30 and 2^-31, R's own spread.
- e1q, e1d: order 1 of each jet on the state jet (z_0, z1d) against R.
- z1q: -J^-1 F_1 with the quad-double jet's F_1; z1f = z1d - J^-1 R (one pass of anm_run.py --fix1 1 --fix1_qlu 1).

Pass, written before the second run: e0q <= 1e-4 e0d; e1q <= 1e-3 e1d and e1q <= 10 dR + 1e-30; |z1q - z1f| <=
1e-3 |z1d - z1f|. Controls, each must fail the order-1 criterion by a factor of 100 (e1 >= 100 e1q): the
quad-double jet with model.rs JET_MUT_KAPPA (kappa's n frozen) and JET_MUT_FREEZE_N (the dimension frozen). The jet with
its horizon solves unrefined (QD_MUT_NOREFINE) is recorded, not a control: at D 85.75 it changed order 1 by less than
R's resolution (13.20), and its control is the Rust test qsolve_jet_resolves_sub_double_double_changes.

    python3 qjet_check.py RAWDIR OUT.jsonl [--state anm-mac/run_trend_pin3_qd9/anm_pin3_D85.7473.npz] [--threads 2]

RAWDIR is a clone of the raw-data repository (the state's path is relative to it); OUT.jsonl must lie in a task
folder of that repository (hp/raw_out.py). Exit status 1 when a criterion fails or a control does not fire.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from anm_run import coefficients  # noqa: E402
from hpkernel.driver import make_solver, qd_axpy  # noqa: E402
from multidouble import dd_add, dd_scale  # noqa: E402
from raw_out import require_raw  # noqa: E402

JET_MUT_FREEZE_N, JET_MUT_KAPPA, QD_MUT_NOREFINE = 21, 22, 7
mp.mp.dps = 50
# 1/3 in double-double (anm_run.py THIRD)
THIRD = (1.0 / 3.0, float(mp.mpf(1) / 3 - mp.mpf(1.0 / 3.0)))


def c(v):
    return np.ascontiguousarray(np.asarray(v, dtype=float))


def q4(z):
    return tuple(c(p) for p in z)


def qsub(x, y):
    """x - y in quad-double, four parts"""
    return qd_axpy(-1.0, *q4(y), *q4(x))


def qmax(x):
    """max |x| of a four-part vector (to double precision)"""
    return float(np.max(np.abs(np.asarray(x[0]) + np.asarray(x[1]))))


def dd_as_q(h, l):
    z = np.zeros(len(h))
    return (c(h), c(l), z, z.copy())


def richardson(r1, r2):
    """(4 r2 - r1) / 3 in double-double, r1 at step e and r2 at e / 2"""
    th, tl = dd_add(*dd_scale(c(r2[0]), c(r2[1]), 4.0), -c(r1[0]), -c(r1[1]))
    return dd_add(*dd_scale(th, tl, THIRD[0]), *dd_scale(th, tl, THIRD[1]))


def main():
    args = list(sys.argv[1:])
    opt = dict(state="anm-mac/run_trend_pin3_qd9/anm_pin3_D85.7473.npz", threads=2)
    for k in list(opt):
        f = "--" + k
        if f in args:
            i = args.index(f); opt[k] = type(opt[k])(args[i + 1]); del args[i:i + 2]
    raw, out = args[0], args[1]
    require_raw(out)
    d = np.load(os.path.join(raw, opt["state"]))
    m = json.loads(str(d["meta"]))
    z0 = q4((d["zh"], d["zl"], d["z2"] if "z2" in d else np.zeros(len(d["zh"])), d["z3"] if "z3" in d else np.zeros(len(d["zh"]))))
    D, M, nodes = (mp.mpf(m["D_str"]) if "D_str" in m else mp.mpf(float(m["D"]))), int(m["M"]), list(m["nodes"])
    kt, kc, ks, kf = m.get("kt", 36), m.get("kc", 146), m.get("ks", 130), m.get("kfreeze", 3)
    mk = lambda Dx: make_solver(Dx, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=opt["threads"], kfreeze=kf)  # noqa: E731
    s = mk(D)
    segs = s.plan(z0[0], z0[1], 1e-32, 0.05)
    s.set_segs(segs)
    rec = dict(state=opt["state"], D=mp.nstr(D, 15), M=M, kfreeze=kf)
    print(f"{opt['state']}: D {mp.nstr(D, 12)}, M {M}, nodes {nodes}, kfreeze {kf}", flush=True)

    t0 = time.time()
    Fq = s.residual_q(*z0)
    t_rq = time.time() - t0
    t0 = time.time()
    Jq = s.residual_jet_q([z0], 2)
    t_jq = time.time() - t0
    t0 = time.time()
    Hs, Ls = s.residual_jet([z0[0]], [z0[1]], 2)
    t_jd = time.time() - t0
    e0q, e0d = qmax(qsub(Jq[0], Fq)), qmax(qsub(dd_as_q(Hs[0], Ls[0]), Fq))
    print(f"order 0 vs residual_q (max |F_q| {qmax(Fq):.2e}): quad-double jet {e0q:.2e}, double-double jet {e0d:.2e}; "
          f"times: residual_q {t_rq:.1f} s, qd jet (2 orders) {t_jq:.1f} s, dd jet {t_jd:.1f} s", flush=True)
    rec.update(F_q=qmax(Fq), e0q=e0q, e0d=e0d, t_residual_q=t_rq, t_jet_q=t_jq, t_jet_dd=t_jd)

    t0 = time.time()
    s.newton_q(*z0)
    t_lu = time.time() - t0
    zero = np.zeros(len(z0[0]))
    solve = lambda b: q4(s.qlu_solve(*q4(b)))  # noqa: E731
    neg = lambda x: q4(tuple(-c(p) for p in x))  # noqa: E731
    z1q = solve(neg(Jq[1]))
    s.newton(z0[0], z0[1])
    Zh, Zl, _, _ = coefficients(s, z0[0], z0[1], 1)
    z1d = dd_as_q(Zh[1], Zl[1])
    print(f"quad-double LU {t_lu:.1f} s; |z1d| {qmax(z1d):.3e}", flush=True)

    def diff_along(e):
        """central differences of residual_q along (z1d, 1) at step e, to double-double"""
        Ks = []
        for sg in (1.0, -1.0):
            sx = mk(D + sg * mp.mpf(e))
            sx.set_segs(segs)
            Ks.append(q4(sx.residual_q(*qd_axpy(sg * e, *z1d, *z0))))
        dq = qsub(Ks[0], Ks[1])
        return dd_scale(c(dq[0]), c(dq[1]), 0.5 / e)

    t0 = time.time()
    r29, r30, r31 = (diff_along(2.0 ** -k) for k in (29, 30, 31))
    R, R2 = richardson(r29, r30), richardson(r30, r31)
    t_r = time.time() - t0
    dR = float(np.max(np.abs(np.asarray(dd_add(*R, -c(R2[0]), -c(R2[1]))[0]))))
    Rq = dd_as_q(*R)
    zs = [z0, z1d]
    J1q = s.residual_jet_q(zs, 2)
    H1, L1 = s.residual_jet([z0[0], z1d[0]], [z0[1], z1d[1]], 2)
    e1q, e1d = qmax(qsub(J1q[1], Rq)), qmax(qsub(dd_as_q(H1[1], L1[1]), Rq))
    print(f"order 1 along (z1d, 1) vs central differences (max |R| {qmax(Rq):.2e}, spread dR {dR:.2e}, {t_r:.0f} s): "
          f"quad-double jet {e1q:.2e}, double-double jet {e1d:.2e}", flush=True)
    rec.update(t_lu_q=t_lu, z1d=qmax(z1d), R=qmax(Rq), dR=dR, e1q=e1q, e1d=e1d)

    z1f = qd_axpy(1.0, *solve(neg(Rq)), *z1d)
    ez_q, ez_d = qmax(qsub(z1q, z1f)), qmax(qsub(z1d, z1f))
    print(f"z_1 against the fixed one: quad-double jet {ez_q:.2e}, double-double jet {ez_d:.2e} (|z_1| {qmax(z1f):.3e})", flush=True)
    rec.update(ez_q=ez_q, ez_d=ez_d)

    ctl = {}
    for name, mut in (("kappa", JET_MUT_KAPPA), ("freeze_n", JET_MUT_FREEZE_N)):
        Jc = s.residual_jet_q(zs, 2, mut)
        ctl[name] = qmax(qsub(Jc[1], Rq))
    e1_nr = qmax(qsub(s.residual_jet_q(zs, 2, QD_MUT_NOREFINE)[1], Rq))
    print("controls, order 1 against R: " + ", ".join(f"{k} {v:.2e}" for k, v in ctl.items())
          + f"; horizon solves unrefined (recorded) {e1_nr:.2e}", flush=True)
    rec["controls"] = ctl
    rec["e1_norefine"] = e1_nr
    rec["version"] = 2

    checks = {
        "order 0: e0q <= 1e-4 e0d": e0q <= 1e-4 * e0d,
        "order 1: e1q <= 1e-3 e1d": e1q <= 1e-3 * e1d,
        "order 1: e1q <= 10 dR + 1e-30": e1q <= 10 * dR + 1e-30,
        "z_1: |z1q - z1f| <= 1e-3 |z1d - z1f|": ez_q <= 1e-3 * ez_d,
    }
    for k, v in ctl.items():
        checks[f"control {k} fires: e1 >= 100 e1q"] = v >= 100 * e1q
    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  {'pass' if v else 'FAIL'}  {k}", flush=True)
    rec["checks"] = checks
    rec["pass"] = ok
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
