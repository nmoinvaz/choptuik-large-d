#!/usr/bin/env python3
"""The octo-double series (anm_run.py --jet_o 1: Solver.residual_jet_o through the octo-double LU of Solver.newton_o)
against the quad-double one (--jet_q 1: residual_jet_q through the quad-double LU of newton_q) on one stored state z_0
of the pinned chain, order by order (notes/OCTO_DOUBLE.md). Both are built on the bare kernel (the freeze projection
of the state's kfreeze, no held offsets), from the state's four quad-double parts (promoted to eight for the
octo-double series).

The quantities, max norms over the rows, for k = 1..N:
- d_oq(k) = |z_k^o - z_k^q| / |z_k^o|: the two series' difference, formed in octo-double.
- d_qa(k) = |z_k^q - z_k^q1| / |z_k^q|, z^q1 the quad-double series with the horizon series in its alpha form
  (horizon_alpha 1, notes/LARGE_D_MIXED.md 10): the same model with other rounding, so d_qa estimates the quad-double
  series' own error independently of the octo-double code.
- d_dq(k) = |z_k^d - z_k^q| / |z_k^q|, z^d the double-double series (anm_run.py coefficients: the double-double jet
  through the double-double LU, one refinement per order), the control.
- with --alpha_o 1 also d_oa(k) = |z_k^o - z_k^o1| / |z_k^o|, the octo-double series' own error estimated the same way.
- d_oq1(k) = |z_k^o - z_k^q1| / |z_k^o|, recorded from version 2 on: how far the quad-double series in the alpha form is
  from the octo-double one (at D 128.56, version 1, d_oq equalled d_qa to three digits, so nearly all of d_qa was the
  default form's error).

Pass (--check 1), written before the first run: the octo-double series agrees with the quad-double one to about the
latter's own error, d_oq(k) <= 30 d_qa(k) for every k, and the median over k of d_oq(k) / d_qa(k) lies in [1/30, 30].
With --alpha_o 1 also d_oa(k) <= 1e-6 d_qa(k) for every k (the octo-double rounding far below the quad-double one).
Controls, each must fail where the octo-double series passes: the double-double series in its place (d_dq(k) > 30
d_qa(k) for some k), and the octo-double z_1 with the dimension frozen (model.rs JET_MUT_FREEZE_N, order 1 only:
|z_1^frozen - z_1^o| / |z_1^o| > 1e3 d_qa(1)).

--check 0 records the same quantities without a verdict (exit 0): at the chain's highest state, where the quad-double
series loses digits, d_oq(k) is that series' error by order.

    python3 ojet_check.py RAWDIR OUT.jsonl [--state anm-mac/pred_q/A2/anm_pin3_D128.5559.npz] [--N 8] [--threads 4]
        [--check 1] [--alpha_o 0]

RAWDIR is a clone of the raw-data repository (the state's path is relative to it); OUT.jsonl must lie in a task folder
of that repository (hp/raw_out.py). Exit status 1 when a criterion fails or a control does not fire (with --check 1).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from anm_run import coefficients, coefficients_o, coefficients_q, o8, o8_of_q  # noqa: E402
from hpkernel.driver import make_solver  # noqa: E402
from multidouble import od_axpy  # noqa: E402
from raw_out import require_raw  # noqa: E402

JET_MUT_FREEZE_N = 21
mp.mp.dps = 50


def c(v):
    return np.ascontiguousarray(np.asarray(v, dtype=float))


def omax(x):
    """max |x| of a multi-part vector (to double precision, from its first two parts)"""
    return float(np.max(np.abs(np.asarray(x[0]) + np.asarray(x[1]))))


def rel(x, y):
    """|x - y| / |x| in octo-double, x and y given in any number of parts (promoted to eight)"""
    xo = o8(tuple(x) + tuple(np.zeros(len(x[0])) for _ in range(8 - len(x))))
    yo = o8(tuple(y) + tuple(np.zeros(len(y[0])) for _ in range(8 - len(y))))
    return omax(od_axpy(-1.0, *yo, *xo)) / omax(xo)


def main():
    args = list(sys.argv[1:])
    opt = dict(state="anm-mac/pred_q/A2/anm_pin3_D128.5559.npz", N=8, threads=4, check=1, alpha_o=0)
    for k in list(opt):
        f = "--" + k
        if f in args:
            i = args.index(f); opt[k] = type(opt[k])(args[i + 1]); del args[i:i + 2]
    raw, out = args[0], args[1]
    require_raw(out)
    d = np.load(os.path.join(raw, opt["state"]))
    m = json.loads(str(d["meta"]))
    n = len(d["zh"])
    z4 = tuple(c(d[k]) if k in d else np.zeros(n) for k in ("zh", "zl", "z2", "z3"))
    D, M, nodes = (mp.mpf(m["D_str"]) if "D_str" in m else mp.mpf(float(m["D"]))), int(m["M"]), list(m["nodes"])
    kt, kc, ks, kf = m.get("kt", 36), m.get("kc", 146), m.get("ks", 130), m.get("kfreeze", 3)
    N = opt["N"]
    mk = lambda alpha: make_solver(D, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=opt["threads"], kfreeze=kf, horizon_alpha=alpha)  # noqa: E731
    s = mk(0)
    segs = s.plan(z4[0], z4[1], 1e-32, 0.05)
    s.set_segs(segs)
    G = 0.7 ** -(float(D) - 3.0)
    rec = dict(state=opt["state"], D=mp.nstr(D, 15), M=M, kfreeze=kf, N=N, G=G, version=2)
    print(f"{opt['state']}: D {mp.nstr(D, 12)}, M {M}, nodes {nodes}, kfreeze {kf}, N {N}; G = 0.7^-(D-3) = {G:.2e}, "
          f"eps_qd G^2 = {2.0 ** -209 * G * G:.1e}", flush=True)

    def timed(name, f):
        t0 = time.time()
        r = f()
        rec["t_" + name] = time.time() - t0
        print(f"  {name}: {rec['t_' + name]:.0f} s", flush=True)
        return r

    timed("lu_q", lambda: s.newton_q(*z4))
    Zq, _ = timed("series_q", lambda: coefficients_q(s, z4, N))
    s1 = mk(1)
    s1.set_segs(segs)
    timed("lu_q1", lambda: s1.newton_q(*z4))
    Zq1, _ = timed("series_q1", lambda: coefficients_q(s1, z4, N))
    s.newton(z4[0], z4[1])
    Zh, Zl, _, _ = timed("series_d", lambda: coefficients(s, z4[0], z4[1], N))
    zo = o8_of_q(z4)
    timed("lu_o", lambda: s.newton_o(list(zo)))
    Zo, _ = timed("series_o", lambda: coefficients_o(s, zo, N))
    # the control: z_1 with the dimension frozen, through the same LU
    F = s.residual_jet_o([list(zo)], 2, JET_MUT_FREEZE_N)
    z1f = o8(s.olu_solve([-c(p) for p in F[1]]))
    if opt["alpha_o"]:
        timed("lu_o1", lambda: s1.newton_o(list(zo)))
        Zo1, _ = timed("series_o1", lambda: coefficients_o(s1, zo, N))

    d_oq = [rel(Zo[k], Zq[k]) for k in range(1, N + 1)]
    d_qa = [rel(Zq[k], Zq1[k]) for k in range(1, N + 1)]
    d_dq = [rel(Zq[k], (Zh[k], Zl[k])) for k in range(1, N + 1)]
    d_oq1 = [rel(Zo[k], Zq1[k]) for k in range(1, N + 1)]
    d_frozen = rel(Zo[1], z1f)
    rec.update(d_oq=d_oq, d_qa=d_qa, d_dq=d_dq, d_oq1=d_oq1, d_frozen_1=d_frozen, norms_o=[omax(Zo[k]) for k in range(N + 1)])
    if opt["alpha_o"]:
        rec["d_oa"] = [rel(Zo[k], Zo1[k]) for k in range(1, N + 1)]
    print(" k   |z_k^o|     d_oq       d_qa       d_dq       d_oq1" + ("      d_oa" if opt["alpha_o"] else ""), flush=True)
    for k in range(1, N + 1):
        print(f"{k:2d}  {rec['norms_o'][k]:.3e}  {d_oq[k - 1]:.2e}  {d_qa[k - 1]:.2e}  {d_dq[k - 1]:.2e}  {d_oq1[k - 1]:.2e}"
              + (f"  {rec['d_oa'][k - 1]:.2e}" if opt["alpha_o"] else ""), flush=True)
    print(f"control, z_1 with the dimension frozen: {d_frozen:.2e}", flush=True)

    ok = True
    if opt["check"]:
        ratios = sorted(a / b for a, b in zip(d_oq, d_qa))
        med = ratios[len(ratios) // 2]
        checks = {
            "d_oq(k) <= 30 d_qa(k), every k": all(a <= 30 * b for a, b in zip(d_oq, d_qa)),
            "median d_oq / d_qa in [1/30, 30]": 1 / 30 <= med <= 30,
            "control: the double-double series fails, d_dq(k) > 30 d_qa(k) for some k": any(a > 30 * b for a, b in zip(d_dq, d_qa)),
            "control: z_1 with the dimension frozen fails, > 1e3 d_qa(1)": d_frozen > 1e3 * d_qa[0],
        }
        if opt["alpha_o"]:
            checks["d_oa(k) <= 1e-6 d_qa(k), every k"] = all(a <= 1e-6 * b for a, b in zip(rec["d_oa"], d_qa))
        rec["median_ratio"] = med
        for k, v in checks.items():
            print(f"  {'pass' if v else 'FAIL'}  {k}", flush=True)
        rec["checks"] = checks
        ok = all(checks.values())
        rec["pass"] = ok
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if opt["check"]:
        print("PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
