#!/usr/bin/env python3
"""Checks of the sigma-model kernel's mode matrix M(lambda) at a saved state (#918), each with a mutation that it
must catch.

M(lambda) is the linearization of the multiple-shooting residual for perturbations exp(lambda tau) g(tau, x), g of the
background's form, with Delta and the phase row held fixed (multishoot's mode_matrix; model.rs gives every tau
derivative of a perturbation the rate). Its determinant vanishes at the growth rate of a mode.

    PYTHONPATH=<site> python3 matter/sigma_ms/mode_check.py STATE.npz [--threads 2] [--lam 1.7] [--only 1,2,3,4]
        [--out FILE.json] [--mutate sigma_h|real_solve|vtau|rate_horizon|rate_lu|rate_centre|rate_steps]

Checks, each printed with its verdict; the exit status is 1 when one fails. The bounds were fixed before the first
run.
  (1) M(0) is the Jacobian's leading block (every row but the phase row, every unknown but Delta): the largest
      difference is below 1e-28 of the largest entry, and M(0) has no imaginary part. The two come from different
      lanes of the engine, complex tangents and real ones.
  (2) The derivative of M in lambda at --lam agrees between the real rates and a complex one: the central difference
      of M(lam +- 2^-16) against Im M(lam + i 2^-30) / 2^-30, to 1e-7 of the largest entry. The complex rate takes
      the 2M x 2M real form of the horizon solves, the real ones the M x M factors.
  (3) The tau-translation mode: v = d/dtau of the state has |M(0) v|_2 <= 1e-9 |M(0)|_F |v|_2. The bound is far
      above f64 rounding (v is formed in f64) and far below a vector that is not the mode.
  (4) The gauge mode: det M changes sign within lambda_g (1 -+ 1e-2), and the two rates that enclose the change
      both lie within 1e-10 of lambda_g = mean over tau of 1 - c_x(tau, 1) (smdriver.gauge_rate), the exact rate of
      the mode that relabels the light cone. This is the one check that tests the rate in every place it enters.
Mutations (each must give exit status 1, by the check named):
  sigma_h       (1)  M(0) from equations whose a^3 sigma^2 coefficient is scaled by 1 + 2^-30 in the Taylor steps
  real_solve    (2)  the complex rate's horizon solves with the M x M factors of the values
  vtau          (3)  v with the even fields a, c differentiated by the odd fields' operator
  rate_horizon  (4)  the rate left out of the horizon series' tau derivatives
  rate_lu       (4)  the rate left out of the factors of the horizon series' solves
  rate_centre   (4)  the rate left out of the centre series
  rate_steps    (4)  the rate left out of the Taylor steps
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smdriver as sd  # noqa: E402

MUT = dict(sigma_h=2, rate_horizon=11, rate_lu=12, rate_centre=13, rate_steps=14, real_solve=15)
CHECK_OF = dict(sigma_h=1, real_solve=2, vtau=3, rate_horizon=4, rate_lu=4, rate_centre=4, rate_steps=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--lam", type=float, default=1.7)
    ap.add_argument("--only", default="1,2,3,4")
    ap.add_argument("--out", default="")
    ap.add_argument("--mutate", default="", choices=[""] + sorted(CHECK_OF))
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",")}
    if a.mutate:
        only = {CHECK_OF[a.mutate]}

    def log(msg):
        print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)

    z, meta = sd.load(a.state)
    zh, zl = np.ascontiguousarray(z[0], float), np.ascontiguousarray(z[1], float)
    s = sd.solver_of(meta, a.threads)
    n = len(zh) - 1
    F = sd.fnorm(s.residual([zh, zl]))
    log(f"start {sd.now()}: {os.path.basename(a.state)}, eta {meta['eta']}, D {meta.get('D', 4.0)}, M {meta['M']}, "
        f"{n} unknowns before Delta, |F| {F:.2e}, Delta {zh[-1]:.15f}"
        + (f", mutation {a.mutate}" if a.mutate else ""))
    out = dict(state=os.path.basename(a.state), eta=meta["eta"], D=meta.get("D", 4.0), M=meta["M"], F=F,
               Delta=float(zh[-1]), mutate=a.mutate, threads=a.threads, when=sd.now())
    ok = True

    def mm(lr, li=0.0, code=0):
        s.set_mutate(code)
        try:
            return [np.asarray(x) for x in s.mode_matrix(zh, zl, lr, 0.0, li, 0.0)]
        finally:
            s.set_mutate(0)

    m0 = None
    if only & {1, 3}:
        t = time.time()
        m0 = mm(0.0, code=MUT["sigma_h"] if a.mutate == "sigma_h" else 0)
        log(f"M(0): {time.time() - t:.1f} s")
    if 1 in only:
        Jh, Jl = (np.asarray(x) for x in s.jacobian(zh, zl))
        d = float(np.max(np.abs((m0[0] - Jh[:n, :n]) + (m0[1] - Jl[:n, :n]))))
        scale = float(np.max(np.abs(Jh[:n, :n])))
        im = float(max(np.max(np.abs(m0[2])), np.max(np.abs(m0[3]))))
        v = d <= 1e-28 * scale and im == 0.0
        ok &= v
        out["c1"] = dict(diff=d, scale=scale, imag=im, ok=bool(v))
        log(f"(1) M(0) against the Jacobian's leading block: largest difference {d:.2e}, largest entry {scale:.2e} "
            f"(ratio {d / scale:.1e} <= 1e-28), largest imaginary part {im:.1e} (0): {'PASS' if v else 'FAIL'}")
        del Jh, Jl
    if 3 in only:
        v_ = sd.dtau_state(zh + zl, meta, wrong=a.mutate == "vtau")
        mv = sd.dd_matvec(m0[0], m0[1], v_, np.zeros(n))
        mv = np.asarray(mv[0]) + np.asarray(mv[1])
        fro = float(np.linalg.norm(m0[0]))
        rho = float(np.linalg.norm(mv) / (fro * np.linalg.norm(v_)))
        v = rho <= 1e-9
        ok &= v
        out["c3"] = dict(rho=rho, Mv=float(np.linalg.norm(mv)), fro=fro, v=float(np.linalg.norm(v_)), ok=bool(v))
        log(f"(3) the tau-translation mode: |M(0) v|_2 {np.linalg.norm(mv):.2e}, |M(0)|_F {fro:.2e}, |v|_2 "
            f"{np.linalg.norm(v_):.2e}, ratio {rho:.2e} (<= 1e-9): {'PASS' if v else 'FAIL'}")
    m0 = None
    if 2 in only:
        h, e = 2.0 ** -16, 2.0 ** -30
        t = time.time()
        p, q = mm(a.lam + h), mm(a.lam - h)
        cd = ((p[0] - q[0]) + (p[1] - q[1])) / (2 * h)
        del p, q
        c = mm(a.lam, e, code=MUT["real_solve"] if a.mutate == "real_solve" else 0)
        cs = (c[2] + c[3]) / e
        err, scale = float(np.max(np.abs(cd - cs))), float(np.max(np.abs(cs)))
        v = scale > 0 and err <= 1e-7 * scale
        ok &= v
        out["c2"] = dict(lam=a.lam, err=err, scale=scale, ok=bool(v))
        log(f"(2) dM/dlambda at {a.lam}: central difference against the complex rate, largest difference {err:.2e}, "
            f"largest entry {scale:.2e} (ratio {err / max(scale, 1e-300):.1e} <= 1e-7): {'PASS' if v else 'FAIL'} "
            f"({time.time() - t:.1f} s)")
        del cd, cs, c
    if 4 in only:
        lg = float(sd.gauge_rate(zh, zl, meta))
        log(f"(4) the gauge rate lambda_g = {lg:.15f}")
        s.set_mutate(MUT.get(a.mutate, 0) if CHECK_OF.get(a.mutate) == 4 else 0)
        cache = {}
        try:
            root, w = sd.det_root_near(sd.det_fn(s, zh, zl, log, cache), lg, widths=(1e-3, 1e-2))
        finally:
            s.set_mutate(0)
        br = None if root is None else sd.sign_bracket(cache, root)
        far = None if br is None else max(abs(br[0] - lg), abs(br[1] - lg))
        v = br is not None and far <= 1e-10
        ok &= v
        out["c4"] = dict(lam_g=lg, root=root, sign_change=br, farthest=far, bracket=w, evals=len(cache),
                         det=sorted((k, sg, la) for k, (sg, la) in cache.items()), ok=bool(v))
        if br is None:
            log(f"    no sign change of det M within lambda_g (1 -+ {w:g}): FAIL")
        else:
            log(f"    det M changes sign between lambda_g {br[0] - lg:+.2e} and lambda_g {br[1] - lg:+.2e} (both "
                f"<= 1e-10 in size), {len(cache)} determinants: {'PASS' if v else 'FAIL'}")
    out["ok"] = bool(ok)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    log(f"MODE CHECK{' (mutation ' + a.mutate + ')' if a.mutate else ''}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
