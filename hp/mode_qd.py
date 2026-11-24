#!/usr/bin/env python3
"""#1078: det M(lam) by a quad-double LU of the double-double mode matrix, equilibrated by powers of two.
#1082: --entries=qd forms the entries themselves in quad-double (Solver.mode_matrix_q).

    python3 mode_qd.py OUT.jsonl STATE.npz LAM [LAM ...] [--threads=6] [--hmax=0.05] [--control=none|update_dd]
                       [--noequil] [--entries=dd|qd] [--allow_unraw]
    python3 mode_qd.py selftest

--entries=dd (the default) factors Solver.mode_matrix's double-double entries (hi, lo); --entries=qd factors the four
parts of Solver.mode_matrix_q (multishoot's mode_matrix_real at Quad: the tangents, the Taylor steps and the refined
horizon solves in quad-double, at the double-double state). At D 3.044375 the double-double entries' rounding, over
1269 Taylor steps, flips the determinant's sign every 3e-13 in lam (#1078 comment 7120204793); quad-double entries
carry no such rounding. Solver.mode_matrix_q is built for M = 32, 48, 96 and 128. The output must lie in a task
folder of the raw-data repository (hp/raw_out.py) unless --allow_unraw.

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
trailing update was exact in double-double. A second planted matrix has quad-double entries in four parts (order 40,
Schur complement +-1e-45 against entries of order one): the factorization of all four parts must give the sign and
ln|det| of mpmath's determinant of the parts' exact sums within 1e-6, and the same matrix cut to its first two parts
(the double-double entries) must not (the control: the lower parts reach the LU)."""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _qd_logdet():
    """multidouble's quad-double log-determinant with its control argument: the Rust binding _md.qd_logdet takes
    control, while the Python wrapper of the pinned 8c4360ac (multidouble 5.1.0) does not forward it (dh370, #1078)"""
    import multidouble
    md = getattr(multidouble, "_md", None)
    if md is not None and hasattr(md, "qd_logdet"):
        return md.qd_logdet
    return multidouble.qd_logdet


def qd_det(parts, equil=True, control="none", par=True):
    """(sign, ln|det|, pivot range, ...) of the matrix whose entries are the sums of its 2 (double-double) or 4
    (quad-double) part arrays, by multidouble's qd_logdet after Ruiz equilibration of the leading parts"""
    from mode_scale import ruiz
    qd_logdet = _qd_logdet()
    parts = [np.asarray(p, float) for p in parts]
    if len(parts) not in (2, 4):
        raise ValueError(f"qd_det: 2 or 4 part arrays, not {len(parts)}")
    if equil:
        r, c = ruiz(parts[0] + parts[1])
    else:
        r, c = np.zeros(parts[0].shape[0]), np.zeros(parts[0].shape[1])
    f = np.exp2(r)[:, None] * np.exp2(c)[None, :]                      # powers of two: exact on every part
    z = np.zeros_like(parts[0])
    q = [np.ascontiguousarray(p * f) for p in parts] + [z] * (4 - len(parts))
    sg, la, pmin, pmax = qd_logdet(q[0], q[1], q[2], q[3], par=par, control=control)
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
    qd = _qd_logdet()
    for t in (tau, -tau):
        A, _ = near_singular(n, t, seed)
        z = np.zeros_like(A)
        ref = mp_logdet(A)
        if qd(A, z, z, z, False, "update_dd")[0] != ref[0]:
            return A, ref
    raise AssertionError("the double-double updates' sign follows tau here")


def qd_split(x):
    """the four f64 parts of the mpmath number x by successive rounding: a normalized quad-double expansion"""
    import mpmath as mp
    out = []
    for _ in range(4):
        p = float(x)
        out.append(p)
        x = x - mp.mpf(p)
    return out


def planted_q(n=40, tau=1e-45, seed=7):
    """an n x n matrix of quad-double entries in four part arrays, [[B, b], [c^T, d]], with B, b, c random to all four
    parts and d chosen so that the Schur complement d - c^T B^-1 b is tau to about 1e-64; returns the parts and
    mpmath's (sign, ln|det|) of the parts' exact sums"""
    import mpmath as mp
    rng = np.random.default_rng(seed)
    with mp.workdps(DPS):
        def rand_q():
            a = rng.standard_normal(4)
            return qd_split(mp.mpf(a[0]) * (1 + mp.mpf(a[1]) * mp.mpf(2) ** -53 + mp.mpf(a[2]) * mp.mpf(2) ** -106
                                            + mp.mpf(a[3]) * mp.mpf(2) ** -159))
        P = np.zeros((4, n, n))
        for i in range(n):
            for j in range(n):
                if (i, j) != (n - 1, n - 1):
                    P[:, i, j] = rand_q()
        exact = lambda i, j: mp.fsum(mp.mpf(float(P[q, i, j])) for q in range(4))
        m = n - 1
        B = mp.matrix([[exact(i, j) for j in range(m)] for i in range(m)])
        b = mp.matrix([exact(i, m) for i in range(m)])
        y = mp.lu_solve(B, b)
        t = mp.fsum(exact(m, k) * y[k] for k in range(m))
        P[:, m, m] = qd_split(t + tau)
        A = mp.matrix([[exact(i, j) for j in range(n)] for i in range(n)])
        dt = mp.det(A)
        return [P[q] for q in range(4)], ((1.0 if dt > 0 else -1.0), float(mp.log(abs(dt))))


def selftest():
    A, (s0, l0) = planted()
    z = np.zeros_like(A)
    agrees = lambda d: d["sign"] == s0 and abs(d["ln"] - l0) <= 1e-6
    q, qe = qd_det([A, z], equil=False), qd_det([A, z], equil=True)
    c = qd_det([A, z], equil=False, control="update_dd")
    ok1, ok3, ok2 = agrees(q), agrees(qe), not agrees(c)
    print(f"selftest: planted det (mpmath, {DPS} digits) sign {s0:+.0f} ln {l0:.9f}; qd {q['sign']:+.0f} {q['ln']:.9f} "
          f"({'PASS' if ok1 else 'FAIL'}); qd equilibrated {qe['sign']:+.0f} {qe['ln']:.9f} ({'PASS' if ok3 else 'FAIL'}); "
          f"control update_dd {c['sign']:+.0f} {c['ln']:.9f} ({'fires' if ok2 else 'DOES NOT FIRE'})")
    P, (s1, l1) = planted_q()
    agrees_q = lambda d: d["sign"] == s1 and abs(d["ln"] - l1) <= 1e-6
    f, fe, fc = qd_det(P, equil=False), qd_det(P, equil=True), qd_det(P[:2], equil=True)
    ok4, ok5, ok6 = agrees_q(f), agrees_q(fe), not agrees_q(fc)
    print(f"selftest: four-part planted det sign {s1:+.0f} ln {l1:.9f}; qd {f['sign']:+.0f} {f['ln']:.9f} "
          f"({'PASS' if ok4 else 'FAIL'}); equilibrated {fe['sign']:+.0f} {fe['ln']:.9f} ({'PASS' if ok5 else 'FAIL'}); "
          f"control, first two parts {fc['sign']:+.0f} {fc['ln']:.9f} ({'fires' if ok6 else 'DOES NOT FIRE'})")
    return 0 if ok1 and ok2 and ok3 and ok4 and ok5 and ok6 else 1


def det_rows(out, state, lams, entries="dd", threads=6, hmax=0.05, equil=True, control="none", log=print):
    """det M(lam) at each lam with the entries of `entries` (dd: Solver.mode_matrix; qd: Solver.mode_matrix_q at the
    state's double-double parts), each row appended to out (jsonl) as it is made; returns the rows"""
    if entries not in ("dd", "qd"):
        raise ValueError(f"entries must be dd or qd, not {entries}")
    import lowD_gamma as LG
    from hpkernel.driver import load
    zh, zl, meta = load(state)
    D, M = meta["D"], meta["M"]
    s, grid = LG.build_solver(D, M, meta, int(threads))
    s.set_segs(s.plan(zh, zl, 1e-32, float(hmax)))
    zero = np.zeros_like(zh)
    rows = []
    for lam in lams:
        t = time.time()
        if entries == "qd":
            parts = s.mode_matrix_q(zh, zl, zero, zero, lam, 0.0)
        else:
            parts = s.mode_matrix(zh, zl, lam, 0.0)
        t_m = time.time() - t
        d = qd_det(parts, equil=equil, control=control)
        del parts
        row = dict(D=D, M=M, lam=lam, entries=entries, n=int(len(zh) - 1), **d, state=os.path.basename(state),
                   map=grid, hmax=float(hmax), threads=int(threads), sec_matrix=round(t_m, 1),
                   sec=round(time.time() - t, 1))
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        log(f"lam {lam!r}: {entries} entries, qd LU sign {d['sign']:+.0f} ln|det| {d['ln']:.9f} pivots "
            f"{d['pmin']:.2e}..{d['pmax']:.2e} (control {d['control']}; matrix {row['sec_matrix']} s, all {row['sec']} s)")
        rows.append(row)
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    o = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    if args and args[0] == "selftest":
        return selftest()
    if len(args) < 3:
        print(__doc__)
        return 2
    from raw_out import require_raw
    out, state = args[0], args[1]
    require_raw(out)
    det_rows(out, state, [float(x) for x in args[2:]], entries=o.get("entries", "dd"), threads=o.get("threads", 6),
             hmax=o.get("hmax", 0.05), equil="noequil" not in o, control=o.get("control", "none"),
             log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
