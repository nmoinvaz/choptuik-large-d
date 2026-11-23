"""#1078: is det M(lam)'s sign at low D unresolved because of scale, or because M is near singular?

    python3 mode_scale.py OUT.jsonl STATE.npz LAM [LAM ...] [--threads=6] [--hmax=0.05] [--mutate=noscale]
    python3 mode_scale.py selftest

At D 3.044375 the double-double LU of the mode matrix has pivots from 1e-4 to 4e32, a ratio below double-double's
unit roundoff (dh370, #1078), so mode_det's sign is not resolved there. A determinant's sign does not change under
positive diagonal scalings, so this script takes M(lam) (Solver.mode_matrix, hi + lo), equilibrates its rows and
columns by powers of two (Ruiz's iteration: exact in floating point), and factors the scaled matrix in float64 with
partial pivoting. Each lambda writes: the pivot range of the unscaled and of the scaled float64 LU, the scaled LU's
sign and ln|det| (corrected by the scalings' exact logarithms), and kernel mode_det's sign and ln|det| beside them.
If the scaled pivot ratio is far above 1e-16, the scaled sign is resolved; a pivot ratio that stays small after
scaling means M itself is near singular there. --mutate=noscale skips the scaling (the control: its pivot range must
equal the unscaled one). The selftest plants a badly scaled matrix with a known determinant."""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import lu_factor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def ruiz(A, iters=40):
    """power-of-two row and column scalings r, c with diag(2^r) A diag(2^c) of max-norm about 1 in every row and column"""
    r = np.zeros(A.shape[0])
    c = np.zeros(A.shape[1])
    B = np.abs(A).copy()
    for _ in range(iters):
        rm = B.max(axis=1)
        rm[rm == 0] = 1.0
        dr = -np.round(np.log2(np.sqrt(rm)))
        B *= np.exp2(dr)[:, None]
        r += dr
        cm = B.max(axis=0)
        cm[cm == 0] = 1.0
        dc = -np.round(np.log2(np.sqrt(cm)))
        B *= np.exp2(dc)[None, :]
        c += dc
        if np.all(dr == 0) and np.all(dc == 0):
            break
    return r, c


def logdet_lu(A):
    """sign, ln|det| and the pivot range of a float64 LU with partial pivoting"""
    lu, piv = lu_factor(A, check_finite=False)
    d = np.diag(lu)
    sign = np.prod(np.sign(d)) * (-1) ** int(np.sum(piv != np.arange(len(piv))))
    ad = np.abs(d)
    return float(sign), float(np.sum(np.log(ad))), float(ad.min()), float(ad.max())


def scaled_det(Mh, Ml, noscale=False):
    A = Mh + Ml
    s0, l0, p0min, p0max = logdet_lu(A)
    if noscale:
        r, c = np.zeros(A.shape[0]), np.zeros(A.shape[1])
    else:
        r, c = ruiz(A)
    S = A * np.exp2(r)[:, None] * np.exp2(c)[None, :]                    # exact: powers of two
    s1, l1, p1min, p1max = logdet_lu(S)
    l1 -= (r.sum() + c.sum()) * math.log(2.0)                             # ln|det M| = ln|det S| - ln det of the scalings
    return dict(sign=s1, ln=l1, pmin=p1min, pmax=p1max, ratio=p1min / p1max, sign_unscaled=s0, ln_unscaled=l0,
                pmin_unscaled=p0min, pmax_unscaled=p0max, ratio_unscaled=p0min / p0max)


def selftest():
    rng = np.random.default_rng(7)
    n = 60
    B = rng.standard_normal((n, n)) + 4 * np.eye(n)
    sB, lB = np.linalg.slogdet(B)
    r = rng.integers(-60, 60, n)
    c = rng.integers(-60, 60, n)
    A = B * np.exp2(r)[:, None] * np.exp2(c)[None, :]                    # det A = det B 2^(sum r + sum c)
    want_l = lB + (r.sum() + c.sum()) * math.log(2.0)
    d = scaled_det(A, np.zeros_like(A))
    ok1 = d["sign"] == sB and abs(d["ln"] - want_l) < 1e-8 and d["ratio"] > 1e-3 and d["ratio_unscaled"] < 1e-20
    dn = scaled_det(A, np.zeros_like(A), noscale=True)
    ok2 = abs(dn["ratio"] - dn["ratio_unscaled"]) <= 1e-12 * max(dn["ratio_unscaled"], 1e-300)
    print(f"selftest: planted scaled matrix: sign {d['sign']:+.0f} (want {sB:+.0f}), ln|det| {d['ln']:.10f} (want "
          f"{want_l:.10f}), pivot ratio scaled {d['ratio']:.1e} vs unscaled {d['ratio_unscaled']:.1e}: "
          f"{'PASS' if ok1 else 'FAIL'}; noscale control equals unscaled: {'PASS' if ok2 else 'FAIL'}")
    return 0 if ok1 and ok2 else 1


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
    lams = [float(x) for x in args[2:]]
    zh, zl, meta = load(state)
    D, M = meta["D"], meta["M"]
    s, grid = LG.build_solver(D, M, meta, int(o.get("threads", 6)))
    s.set_segs(s.plan(zh, zl, 1e-32, float(o.get("hmax", 0.05))))
    for lam in lams:
        t = time.time()
        Mh, Ml = s.mode_matrix(zh, zl, lam, 0.0)
        d = scaled_det(np.asarray(Mh), np.asarray(Ml), noscale=o.get("mutate") == "noscale")
        sg, la, pmin, pmax = s.mode_det(zh, zl, lam, 0.0)
        row = dict(D=D, M=M, lam=lam, n=int(np.asarray(Mh).shape[0]), **d, kernel_sign=float(sg), kernel_ln=float(la),
                   kernel_pmin=float(pmin), kernel_pmax=float(pmax), state=os.path.basename(state), map=grid,
                   mutate=o.get("mutate"), sec=round(time.time() - t, 1))
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"lam {lam:.12f}: scaled sign {d['sign']:+.0f} ln|det| {d['ln']:.6f} pivot ratio {d['ratio']:.1e} | "
              f"unscaled {d['ratio_unscaled']:.1e} | kernel sign {sg:+.0f} ln|det| {la:.6f} ({row['sec']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
