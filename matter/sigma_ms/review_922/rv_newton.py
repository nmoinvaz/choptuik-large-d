#!/usr/bin/env python3
"""Review of smdriver.newton_band (#922): is a state it calls converged a root of the kernel's system, with Delta
fixed? One plain Newton step from a stored converged state, computed without newton_band and, in K1, without the
kernel's Jacobian or solver.

    PYTHONPATH=<site> python3 rv_newton.py --state FILE.npz [--fd] [--out FILE.json] [--threads 2]
      The state is loaded with its stored plan (the Taylor steps of its meta), so the residual is the one the stored
      run tested.
      K2  J = Solver.jacobian (dense, double-double), d = -J^-1 F by multidouble.dd_solve: the reviewer's solve.
      K1  (--fd) J by central differences of Solver.residual in double-double, step h = 2^-33 in every unknown
          (2n residuals), the same solve. This shares the residual with newton_band and nothing else of the kernel.
          With --richardson the columns are 4 D(h) - D(2h) = 3 J (4n residuals), which removes the h^2 term of the
          truncation (up to 1.8e-14 in an entry at h = 2^-33, rv_fdcol.py), and the system solved is 3 J d = -3 F.
      K3  (--fd) the two Jacobians entry by entry.
      Also Solver.newton's own step (the one newton_band and newton_solve take), for comparison.
      K4  x = J^-T e_Delta with Solver.jacobian: Newton's step moves Delta by -x.F, so |x|_1 max |F| bounds how far
          Delta can move at a residual of that size (to first order). Reported, with no criterion.
      Criteria, written before the runs: after each step max |F| <= 1e-30, and Delta moves by at most 1e-20.
      Exit 0 only if every step computed meets both.
"""
import argparse
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import smdriver as sd  # noqa: E402
from multidouble import dd_add, dd_solve  # noqa: E402

H = 2.0 ** -33


def dd_delta(zh, zl):
    return mp.mpf(float(zh[-1])) + mp.mpf(float(zl[-1]))


def fd_jacobian(s, zh, zl, log, richardson=False):
    """the Jacobian by central differences of the residual, columns D(h); with richardson, 4 D(h) - D(2h), which is
    3 J with the h^2 term of the truncation removed (no division: every operation is exact or double-double)"""
    n = len(zh)
    Jh, Jl = np.empty((n, n)), np.empty((n, n))
    e, zero = np.zeros(n), np.zeros(n)
    t0 = time.time()

    def column(j, h):
        e[j] = h
        ph, pl = dd_add(zh, zl, e, zero)
        mh, ml = dd_add(zh, zl, -e, zero)
        e[j] = 0.0
        Fp, Fm = s.residual([ph, pl]), s.residual([mh, ml])
        dh, dl = dd_add(np.asarray(Fp[0]), np.asarray(Fp[1]), -np.asarray(Fm[0]), -np.asarray(Fm[1]))
        return np.asarray(dh) / (2 * h), np.asarray(dl) / (2 * h)                    # a power of two: exact

    for j in range(n):
        dh, dl = column(j, H)
        if richardson:
            d2h, d2l = column(j, 2 * H)
            dh, dl = (np.asarray(x) for x in dd_add(4 * dh, 4 * dl, -d2h, -d2l))
        Jh[:, j], Jl[:, j] = dh, dl
        if j % 500 == 0:
            log(f"  finite differences: column {j} of {n} ({time.time() - t0:.0f} s)")
    return Jh, Jl


def step(s, zh, zl, Jh, Jl, F):
    """one Newton step with the given Jacobian: (z1h, z1l, max |F(z1)|, |d|_2, dDelta)"""
    dh, dl = dd_solve(np.ascontiguousarray(Jh), np.ascontiguousarray(Jl), -np.asarray(F[0]), -np.asarray(F[1]))
    dh, dl = np.asarray(dh), np.asarray(dl)
    th, tl = dd_add(zh, zl, dh, dl)
    return th, tl, sd.fnorm(s.residual([th, tl])), float(np.linalg.norm(dh + dl)), float(dh[-1] + dl[-1]), dh + dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--fd", action="store_true")
    ap.add_argument("--richardson", action="store_true", help="with --fd: the Jacobian from two step sizes")
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    mp.mp.dps = 40

    def log(m):
        print(m, flush=True)
    z, meta = sd.load(a.state)
    zh, zl = np.asarray(z[0], float), np.asarray(z[1], float)
    eta = None if meta["eta"] == "inf" else float(meta["eta"])
    s = sd.make_solver(eta, meta["M"], nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=meta["ks"],
                       threads=a.threads, D=meta["D"])
    s.set_segs(meta["segs"])
    F = s.residual([zh, zl])
    F0 = sd.fnorm(F)
    D0 = dd_delta(zh, zl)
    log(f"{os.path.basename(a.state)}: eta {meta['eta']}, D {meta['D']}, M {meta['M']}, n = {len(zh)}; on the stored "
        f"plan max |F| = {F0:.3e}, Delta = {mp.nstr(D0, 28)}")
    res = dict(state=os.path.basename(a.state), n=len(zh), F0=F0, Delta0=mp.nstr(D0, 30), steps={})
    ok = True

    def record(name, t):
        nonlocal ok
        th, tl, F1, nd, dD, d = t
        good = F1 <= 1e-30 and abs(dD) <= 1e-20
        ok = ok and good
        res["steps"][name] = dict(F_after=F1, step_norm=nd, dDelta=dD, Delta_after=mp.nstr(dd_delta(th, tl), 30), ok=good)
        log(f"  {name}: |d|_2 = {nd:.3e}, dDelta = {dD:+.3e}, max |F| after = {F1:.3e}, Delta after = "
            f"{mp.nstr(dd_delta(th, tl), 28)}: {'OK' if good else 'FAIL'}")
        return d
    t0 = time.time()
    # the kernel's own Newton step, for comparison (not part of the verdict's independence)
    dz, Fn, pmin, pmax = s.newton([zh, zl])
    th, tl = dd_add(zh, zl, np.asarray(dz[0]), np.asarray(dz[1]))
    dN = np.asarray(dz[0]) + np.asarray(dz[1])
    dn = record("Solver.newton's step", (th, tl, sd.fnorm(s.residual([th, tl])), float(np.linalg.norm(dN)),
                                         float(dN[-1]), dN))
    log(f"    (pivots {pmin:.1e}..{pmax:.1e}; {time.time() - t0:.0f} s)")
    Jkh, Jkl = (np.asarray(x, float) for x in s.jacobian(zh, zl))
    log(f"  Solver.jacobian: {time.time() - t0:.0f} s")
    d2 = record("K2, Solver.jacobian with the reviewer's solve", step(s, zh, zl, Jkh, Jkl, F))
    # is the F that Solver.newton returns (the one newton_band tests against tol) Solver.residual's, bit for bit?
    same = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(Fn, F)) and len(Fn) == len(F)
    res["newton_F_is_residual"] = bool(same)
    log(f"  the F returned by Solver.newton equals Solver.residual's bit for bit: {same}")
    # K4, how far Delta can move at this residual: x = J^-T e_Delta, so that Newton's dDelta = -x.F and
    # |dDelta| <= |x|_1 max |F| for any residual of that size
    e = np.zeros(len(zh))
    e[-1] = 1.0
    xh, xl = dd_solve(np.ascontiguousarray(Jkh.T), np.ascontiguousarray(Jkl.T), e, np.zeros(len(zh)))
    x = np.asarray(xh) + np.asarray(xl)
    Fv = np.asarray(F[0]) + np.asarray(F[1])
    pred = -float(sum(mp.mpf(float(a_)) * (mp.mpf(float(b_)) + mp.mpf(float(c_)))
                      for a_, b_, c_ in zip(x, np.asarray(F[0]), np.asarray(F[1]))))
    g1, g2 = float(np.sum(np.abs(x))), float(np.linalg.norm(x))
    res["K4"] = dict(g1=g1, g2=g2, ginf=float(np.max(np.abs(x))), W_at_F=g1 * F0, W_at_1e31=g1 * 1e-31,
                     W_at_1e28=g1 * 1e-28, dDelta_predicted=pred, F2=float(np.linalg.norm(Fv)))
    log(f"  K4, x = J^-T e_Delta: |x|_1 = {g1:.3e}, |x|_2 = {g2:.3e}, max |x| = {res['K4']['ginf']:.3e}; -x.F = "
        f"{pred:+.3e} (the step's dDelta {res['steps'][next(k for k in res['steps'] if k.startswith('K2'))]['dDelta']:+.3e}); "
        f"|x|_1 max |F| = {g1 * F0:.2e} here, {g1 * 1e-31:.2e} at max |F| = 1e-31, {g1 * 1e-28:.2e} at 1e-28")
    res["K2_vs_newton"] = dict(max_diff=float(np.max(np.abs(d2 - dn))), dDelta_diff=float(d2[-1] - dn[-1]))
    log(f"    against Solver.newton's step: max difference {res['K2_vs_newton']['max_diff']:.2e}, in Delta "
        f"{res['K2_vs_newton']['dDelta_diff']:+.2e}  ({time.time() - t0:.0f} s)")
    if a.fd:
        Jfh, Jfl = fd_jacobian(s, zh, zl, log, a.richardson)
        Fs = F
        if a.richardson:
            # the columns are 3 J: compared with 3 Solver.jacobian, and solved against 3 F
            Jkh, Jkl = (np.asarray(x) for x in dd_add(Jkh, Jkl, 2 * Jkh, 2 * Jkl))
            Fs = dd_add(np.asarray(F[0]), np.asarray(F[1]), 2 * np.asarray(F[0]), 2 * np.asarray(F[1]))
        diff = sum(np.asarray(x) for x in dd_add(Jfh, Jfl, -Jkh, -Jkl))
        sc = 3.0 if a.richardson else 1.0
        diff = diff / sc
        i, j = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        res["K3"] = dict(max_abs_diff=float(np.abs(diff).max()), where=[int(i), int(j)],
                         J_max=float(np.abs(Jkh).max()) / sc, fro_diff=float(np.linalg.norm(diff)),
                         fro_J=float(np.linalg.norm(Jkh)) / sc)
        log(f"  K3, finite-difference Jacobian against Solver.jacobian: largest entry difference "
            f"{res['K3']['max_abs_diff']:.2e} at ({i}, {j}) (largest entry of J {res['K3']['J_max']:.2e}); Frobenius "
            f"{res['K3']['fro_diff']:.2e} of {res['K3']['fro_J']:.2e}  ({time.time() - t0:.0f} s)")
        res["K3"]["richardson"] = bool(a.richardson)
        d1 = record("K1, Jacobian from the residual alone" + (" (two step sizes)" if a.richardson else ""),
                    step(s, zh, zl, Jfh, Jfl, Fs))
        res["K1_vs_K2"] = dict(max_diff=float(np.max(np.abs(d1 - d2))), dDelta_diff=float(d1[-1] - d2[-1]))
        log(f"    against K2's step: max difference {res['K1_vs_K2']['max_diff']:.2e}, in Delta "
            f"{res['K1_vs_K2']['dDelta_diff']:+.2e}  ({time.time() - t0:.0f} s)")
    res.update(ok=ok, sec=round(time.time() - t0, 1))
    log(f"RV_NEWTON ({os.path.basename(a.state)}): {'PASS' if ok else 'FAIL'}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
