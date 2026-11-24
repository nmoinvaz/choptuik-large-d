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

selftest: a planted n = 48 matrix A = L U, unit lower L and upper U with dyadic entries, one diagonal entry of U
2^-140 (below double-double's floor), so det A = prod U_kk exactly. A is held in two exact parts, hi = L U0 (U with
that entry zero) and lo = L E (E the 2^-140 entry alone), since in one double the tiny part would round away. The
quad-double LU must give det's sign and ln|det| to 1e-12; with update_dd it must not (the control)."""
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


def qd_det(Mh, Ml, equil=True, control="none", par=True):
    from mode_scale import ruiz
    qd_logdet = _qd_logdet()
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


def planted(n=48, seed=3, tiny=-140):
    rng = np.random.default_rng(seed)
    L = np.tril(rng.integers(-8, 9, (n, n)) / 16.0, -1) + np.eye(n)       # unit lower, entries k/16
    U0 = np.triu(rng.integers(-8, 9, (n, n)) / 16.0, 1)
    d = rng.integers(1, 8, n) * np.where(rng.random(n) < 0.5, -1.0, 1.0)  # diagonal: small integers
    k = n // 2
    dk = 2.0 ** tiny                                                      # one pivot below double-double's floor
    d0 = d.copy()
    d0[k] = 0.0
    U0 += np.diag(d0)
    E = np.zeros((n, n))
    E[k, k] = dk
    hi, lo = L @ U0, L @ E                                                # both exact: dyadics of few bits
    d[k] = dk
    return hi, lo, float(np.prod(np.sign(d))), float(np.sum(np.log(np.abs(d))))


def selftest():
    hi, lo, s0, l0 = planted()
    q = qd_det(hi, lo, equil=False)
    ok1 = q["sign"] == s0 and abs(q["ln"] - l0) < 1e-12
    c = qd_det(hi, lo, equil=False, control="update_dd")
    ok2 = not (c["sign"] == s0 and abs(c["ln"] - l0) < 1e-6)
    qe = qd_det(hi, lo, equil=True)
    ok3 = qe["sign"] == s0 and abs(qe["ln"] - l0) < 1e-12
    print(f"selftest: planted det sign {s0:+.0f} ln {l0:.12f}; qd {q['sign']:+.0f} {q['ln']:.12f} "
          f"({'PASS' if ok1 else 'FAIL'}); qd equilibrated {qe['sign']:+.0f} {qe['ln']:.12f} "
          f"({'PASS' if ok3 else 'FAIL'}); control update_dd {c['sign']:+.0f} {c['ln']:.12f} "
          f"({'fires' if ok2 else 'DOES NOT FIRE'})")
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
