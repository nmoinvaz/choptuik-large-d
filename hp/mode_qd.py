#!/usr/bin/env python3
"""#1078: det M(lam) by a quad-double LU of the double-double mode matrix, equilibrated by powers of two.

    python3 mode_qd.py OUT.jsonl STATE.npz LAM [LAM ...] [--threads=6] [--hmax=0.05] [--control=none|update_dd]
                       [--noequil]
    python3 mode_qd.py selftest

At D 3.044375 (M' 96) the smallest pivot of the equilibrated double-double LU falls like exp(-15 lam): 2.5e-2 at
lam 1, 9.5e-14 at 3, 4.4e-29 at 5.14 (dh370), below the double-double LU's rounding floor (n u max|a|, about 1e-27),
so mode_det's sign there is noise. The matrix's entries are double-double, accurate to about 1e-32 of their size, far
below that pivot; only the LU's rounding swamps it. This script factors the same entries in quad-double
(multidouble's qd_logdet, u about 1e-64) after Ruiz equilibration by powers of two (mode_scale.ruiz; exact), and
corrects ln|det| by the scalings' exact logarithm. --control=update_dd does the trailing updates in double-double
(the control: it must reproduce the double-double jag). Each lambda writes sign, ln|det|, the qd pivot range, and the
scalings' shift; the kernel's mode_det is not called (run it separately to compare).

selftest: multidouble's planted case (python/tests/test_logdet.py at the pinned commit): a float64 matrix
A = [[B, b], [c^T, d]] of order 40 whose Schur complement is +-1e-45 (the sign chosen so that the quad-double LU with
double-double updates gets it wrong), with full-mantissa entries and the reference det from mpmath at 200 digits. The
quad-double LU, plain and equilibrated, must give the sign and ln|det| within 1e-6; with update_dd it must not (the
control). An earlier planted matrix of few-bit dyadic entries could not make the control fire (dh370): every
trailing update was exact in double-double."""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def qd_det(Mh, Ml, equil=True, control="none", par=True):
    from mode_scale import ruiz
    from multidouble import qd_logdet
    Mh, Ml = np.asarray(Mh, float), np.asarray(Ml, float)
    if equil:
        r, c = ruiz(Mh + Ml)
    else:
        r, c = np.zeros(Mh.shape[0]), np.zeros(Mh.shape[1])
    f = np.exp2(r)[:, None] * np.exp2(c)[None, :]                      # powers of two: exact on both parts
    z = np.zeros_like(Mh)
    sg, la, pmin, pmax = qd_logdet(np.ascontiguousarray(Mh * f), np.ascontiguousarray(Ml * f), z, z, par=par,
                                   control=control)
    shift = float(r.sum() + c.sum())
    return dict(sign=float(sg), ln=float(la) - shift * math.log(2.0), pmin=float(pmin), pmax=float(pmax),
                ratio=float(pmin) / float(pmax), shift=shift, equil=bool(equil), control=control)


DPS = 200


def near_singular(n, tau, seed):
    """multidouble's test matrix: B, b, c random; the last three entries of c chosen in mpmath to cancel c^T B^-1 b to
    about 2^-53 each; d = fl(t + tau). Returns A and its Schur complement s exactly (det A = s det B)"""
    import mpmath as mp
    rng = np.random.default_rng(seed)
    m = n - 1
    B, b, c = rng.standard_normal((m, m)), rng.standard_normal(m), rng.standard_normal(m)
    c[m - 3:] = 0.0
    with mp.workdps(DPS):
        y = mp.lu_solve(mp.matrix(B.tolist()), mp.matrix(b.tolist()))
        t = mp.fsum(mp.mpf(c[k]) * y[k] for k in range(m))
        for k in range(m - 3, m):
            c[k] = float(-t / y[k])
            t += mp.mpf(c[k]) * y[k]
        d = float(t + tau)
        sc = mp.mpf(d) - t
    return np.block([[B, b[:, None]], [c[None, :], np.array([[d]])]]), sc


def mp_logdet(A):
    import mpmath as mp
    with mp.workdps(DPS):
        dt = mp.det(mp.matrix([[mp.mpf(float(v)) for v in row] for row in A]))
        return (1.0 if dt > 0 else -1.0), float(mp.log(abs(dt)))


def planted(n=40, tau=1e-45, seed=5):
    """near_singular's matrix with Schur complement +tau or -tau, whichever gives det A the sign opposite to the one the
    quad-double LU with double-double updates finds (as multidouble's test, with its update_dd control as 'wrong')"""
    from multidouble import qd_logdet
    for t in (tau, -tau):
        A, _ = near_singular(n, t, seed)
        z = np.zeros_like(A)
        ref = mp_logdet(A)
        if qd_logdet(A, z, z, z, control="update_dd")[0] != ref[0]:
            return A, ref
    raise AssertionError("the double-double updates' sign follows tau here")


def selftest():
    A, (s0, l0) = planted()
    z = np.zeros_like(A)
    agrees = lambda d: d["sign"] == s0 and abs(d["ln"] - l0) <= 1e-6
    q, qe = qd_det(A, z, equil=False), qd_det(A, z, equil=True)
    c = qd_det(A, z, equil=False, control="update_dd")
    ok1, ok3, ok2 = agrees(q), agrees(qe), not agrees(c)
    print(f"selftest: planted det (mpmath, {DPS} digits) sign {s0:+.0f} ln {l0:.9f}; qd {q['sign']:+.0f} {q['ln']:.9f} "
          f"({'PASS' if ok1 else 'FAIL'}); qd equilibrated {qe['sign']:+.0f} {qe['ln']:.9f} ({'PASS' if ok3 else 'FAIL'}); "
          f"control update_dd {c['sign']:+.0f} {c['ln']:.9f} ({'fires' if ok2 else 'DOES NOT FIRE'})")
    return 0 if ok1 and ok2 and ok3 else 1


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    o = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    if args and args[0] == "selftest":
        return selftest()
    if len(args) < 3:
        print(__doc__)
        return 2
    import lowD_gamma as LG
    from hpkernel.driver import load
    out, state = args[0], args[1]
    zh, zl, meta = load(state)
    D, M = meta["D"], meta["M"]
    s, grid = LG.build_solver(D, M, meta, int(o.get("threads", 6)))
    s.set_segs(s.plan(zh, zl, 1e-32, float(o.get("hmax", 0.05))))
    for lam in [float(x) for x in args[2:]]:
        t = time.time()
        Mh, Ml = s.mode_matrix(zh, zl, lam, 0.0)
        d = qd_det(Mh, Ml, equil="noequil" not in o, control=o.get("control", "none"))
        row = dict(D=D, M=M, lam=lam, n=int(np.asarray(Mh).shape[0]), **d, state=os.path.basename(state), map=grid,
                   hmax=float(o.get("hmax", 0.05)), sec=round(time.time() - t, 1))
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"lam {lam!r}: qd sign {d['sign']:+.0f} ln|det| {d['ln']:.9f} pivots {d['pmin']:.2e}..{d['pmax']:.2e} "
              f"(control {d['control']}, {row['sec']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
