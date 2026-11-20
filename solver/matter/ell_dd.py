#!/usr/bin/env python3
"""Double-double determinant of the l = 1 mode matrix M(lambda) (#1025, route (c)).

The sign of det M, which ell_refine's root finder reads, is rounding noise past D 17 at k_c 55-63: lambda_1's zero sits
on the bottom member of a ladder of small singular values that falls below rounding (matter/ELL.md section 6). This
module asks first which part needs double-double: the LU or the matrix's entries.

  lu_dd(A, lb, ub)     banded LU with partial pivoting of a real double-double matrix (ddnp.R), giving sign and
                       ln|det|; adapted from complex_imag_dd.lu_factor (wip/xf-checks, #375) to real entries
  lu_double(A, lb, ub) the same elimination in float64 (the control path)
  variants(M, ...)     M, M^T and row/column scalings by random powers of 2: exact in any precision, so det keeps its
                       sign, while partial pivoting takes another elimination path

    python ell_dd.py selftest [--mutate=nopivot]
        (a) a random banded matrix with some diagonal entries 1e-25 against mpmath (60 digits): |d ln|det|| < 1e-12
            (ln|det| sums float64 logarithms of the pivots) and the same sign; with --mutate=nopivot (no row
            exchanges) it must fail;
        (b) a planted ladder (dense, n 100; regular values 1 .. 1e-6 and 20 values 1e-14 .. 1e-26, mixed by random
            orthogonal factors, rounded to double): over 5 variants the dd sign must equal mpmath's exact sign of the
            rounded matrix at every variant, and the float64 path must miss it at one variant at least (else the
            planted case is too easy to test anything).
    python ell_dd.py sens STATE D kc LAMS [--eps 1e-16,1e-15,1e-14] [--seeds 3] [--blocks] [--out FILE]
        E1: dd sign and ln|det| of M at each lambda over 5 variants, beside the float64 path's;
        E2: the dd sign after relative noise eps * xi on M's propagator entries (all blocks, and with --blocks one
            block at a time; the -I blocks stay exact).
Reference: double-double arithmetic after Dekker (Numer. Math. 18, 224 (1971)) and Hida, Li and Bailey (QD, 2001), as
in ddnp.py."""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ddnp as dd                                    # noqa: E402

MUTATE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")


def lu_dd(A, lb=None, ub=None):
    """LU with partial pivoting of a real double-double matrix A (dd.R, N x N), in place. lb, ub: the lower and upper
    bandwidths (None: dense); with pivoting U's upper bandwidth is at most lb + ub. Returns (sign, ln|det|)."""
    N = A.shape[0]
    sign, lnabs = 1.0, 0.0
    ubu = None if (lb is None or ub is None) else lb + ub
    for k in range(N):
        r1 = N if lb is None else min(N, k + lb + 1)
        c1 = N if ubu is None else min(N, k + ubu + 1)
        piv = k if MUTATE == "nopivot" else k + int(np.argmax(np.abs(A.h[k:r1, k])))
        if piv != k:
            for arr in (A.h, A.l):
                arr[[k, piv], k:c1] = arr[[piv, k], k:c1]
            sign = -sign
        ph, pl = float(A.h[k, k]), float(A.l[k, k])
        pv = ph + pl
        if pv == 0.0:
            return 0.0, -math.inf
        sign *= 1.0 if pv > 0 else -1.0
        lnabs += math.log(abs(pv))
        if r1 > k + 1:
            l = dd.div(A[k + 1:r1, k], dd.R(np.full(r1 - k - 1, ph), np.full(r1 - k - 1, pl)))
            A[k + 1:r1, k] = l
            if c1 > k + 1:
                u = A[k, k + 1:c1]
                A[k + 1:r1, k + 1:c1] = dd.sub(A[k + 1:r1, k + 1:c1],
                                               dd.mul(dd.R(l.h[:, None], l.l[:, None]), dd.R(u.h[None, :], u.l[None, :])))
    return sign, lnabs


def lu_double(A, lb=None, ub=None):
    """lu_dd's elimination in float64 (A a float array, copied)."""
    A = np.array(A, dtype=float)
    N = A.shape[0]
    sign, lnabs = 1.0, 0.0
    ubu = None if (lb is None or ub is None) else lb + ub
    for k in range(N):
        r1 = N if lb is None else min(N, k + lb + 1)
        c1 = N if ubu is None else min(N, k + ubu + 1)
        piv = k + int(np.argmax(np.abs(A[k:r1, k])))
        if piv != k:
            A[[k, piv], k:c1] = A[[piv, k], k:c1]
            sign = -sign
        pv = A[k, k]
        if pv == 0.0:
            return 0.0, -math.inf
        sign *= 1.0 if pv > 0 else -1.0
        lnabs += math.log(abs(pv))
        if r1 > k + 1:
            l = A[k + 1:r1, k] / pv
            A[k + 1:r1, k] = l
            if c1 > k + 1:
                A[k + 1:r1, k + 1:c1] -= np.outer(l, A[k, k + 1:c1])
    return sign, lnabs


def bandwidths(M):
    r, c = np.nonzero(M)
    return int(max(0, np.max(r - c))), int(max(0, np.max(c - r)))


def variants(M, nscale=3, seed=7):
    """(name, matrix, lb, ub, ln-scale) with det(matrix) = det(M) * 2^(ln-scale / ln 2), same sign"""
    lb, ub = bandwidths(M)
    out = [("M", M, lb, ub, 0.0), ("M^T", M.T.copy(), ub, lb, 0.0)]
    g = np.random.default_rng(seed)
    for i in range(nscale):
        er = g.integers(-8, 9, M.shape[0]); ec = g.integers(-8, 9, M.shape[1])
        out.append((f"scaled {i}", np.ldexp(np.ldexp(M, er[:, None]), ec[None, :]), lb, ub,
                    float(er.sum() + ec.sum()) * math.log(2.0)))
    return out


def det_both(V, lb, ub):
    t0 = time.time()
    sd, ld = lu_dd(dd.R(V.copy()), lb, ub)
    t1 = time.time()
    sf, lf = lu_double(V, lb, ub)
    return dict(sign_dd=sd, ln_dd=ld, sign_f64=sf, ln_f64=lf, sec_dd=round(t1 - t0, 1), sec_f64=round(time.time() - t1, 1))


def selftest():
    import mpmath as mp
    mp.mp.dps = 60
    ok = True
    g = np.random.default_rng(3)
    n, lb, ub = 80, 6, 5
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - lb), min(n, i + ub + 1)):
            A[i, j] = g.standard_normal()
    A[np.arange(0, n, 7), np.arange(0, n, 7)] *= 1e-25              # tiny diagonal: without row exchanges the growth
                                                                      # (1e25) swamps even double-double
    ex = mp.det(mp.matrix(A.tolist()))
    s, l = lu_dd(dd.R(A.copy()), lb, ub)
    err = abs(l - float(mp.log(abs(ex))))
    good = err < 1e-12 and s == (1.0 if ex > 0 else -1.0)
    print(f"selftest (a) banded n {n}: ln|det| error {err:.1e} (bar 1e-12), sign {s:+.0f} vs exact "
          f"{'+' if ex > 0 else '-'}: {'OK' if good else 'BAD'}" + (" [mutate nopivot]" if MUTATE else ""), flush=True)
    ok &= good
    n = 100
    Q1, Q2 = (np.linalg.qr(g.standard_normal((n, n)))[0] for _ in range(2))
    sig = np.concatenate([np.logspace(0, -6, n - 20), np.logspace(-14, -26, 20)])
    B = (Q1 * sig) @ Q2.T
    exb = mp.det(mp.matrix(B.tolist()))
    sx = 1.0 if exb > 0 else -1.0
    rows, miss_f64 = [], 0
    for name, V, a, b, lsc in variants(B):
        r = det_both(V, None, None)
        rows.append((name, r["sign_dd"], r["sign_f64"]))
        miss_f64 += r["sign_f64"] != sx
        ok &= r["sign_dd"] == sx
    good_b = all(sd == sx for _, sd, _ in rows) and miss_f64 >= 1
    print(f"selftest (b) planted ladder n {n}: exact sign {sx:+.0f}; dd signs {[r[1] for r in rows]}, float64 signs "
          f"{[r[2] for r in rows]}; dd equals exact at every variant: {all(sd == sx for _, sd, _ in rows)}; float64 "
          f"misses at {miss_f64} of {len(rows)} (needed >= 1): {'OK' if good_b else 'BAD'}", flush=True)
    ok &= good_b
    print(f"selftest: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def sens(argv):
    import argparse
    from dss_ell import DSSEll, NODES_STD
    ap = argparse.ArgumentParser()
    ap.add_argument("state"); ap.add_argument("D", type=float); ap.add_argument("kc", type=int); ap.add_argument("lams")
    ap.add_argument("--eps", default="1e-16,1e-15,1e-14"); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--blocks", action="store_true"); ap.add_argument("--out", default=None)
    ap.add_argument("--mutate", default=None)
    o = ap.parse_args([a for a in argv])
    z = np.load(o.state)
    S = DSSEll(kc=o.kc, nodes=NODES_STD, n=o.D - 2, ell=1, Delta_ref=z[-1], Ks=30, u0=0.04)
    S.freeze_from(z, 0.7)
    blocks, offs = S._block_list()
    blk = 3 * S.n_odd + 2 * S.n_even
    out = o.out or os.path.join(HERE, "data", "ell_dd_sens_1025.jsonl")
    for lam in (float(v) for v in o.lams.split(",")):
        t0 = time.time()
        with np.errstate(all="ignore"):
            M = S.mode_matrix(z, lam)
        lb, ub = bandwidths(M)
        print(f"D {o.D} k_c {o.kc} lambda {lam}: n {M.shape[0]}, bandwidths {lb}/{ub}, built in {time.time() - t0:.0f} s",
              flush=True)
        e1 = []
        for name, V, a, b, lsc in variants(M):
            r = det_both(V, a, b)
            r.update(variant=name, ln_dd=r["ln_dd"] - lsc, ln_f64=r["ln_f64"] - lsc)
            e1.append(r)
            print(f"  E1 {name:9s}: dd {r['sign_dd']:+.0f} ln|det| {r['ln_dd']:.10f} ({r['sec_dd']} s); float64 "
                  f"{r['sign_f64']:+.0f} {r['ln_f64']:.10f}", flush=True)
        e2 = []
        mask = np.zeros(M.shape, bool)                            # the propagator entries, not the -I blocks
        for b_, (c0, nf, fun, tgt) in enumerate(blocks):
            mask[b_ * blk:(b_ + 1) * blk, c0:c0 + nf] = True
        sel = [("all", mask)]
        if o.blocks:
            for b_, (c0, nf, fun, tgt) in enumerate(blocks):
                mb = np.zeros_like(mask); mb[b_ * blk:(b_ + 1) * blk, c0:c0 + nf] = True
                sel.append((f"block {b_}", mb))
        for eps in (float(v) for v in o.eps.split(",")):
            for nm, mk in sel:
                sg = []
                for sd in range(o.seeds):
                    xi = np.random.default_rng(100 + sd).standard_normal(M.shape)
                    Mp = M + eps * xi * M * mk
                    s, l = lu_dd(dd.R(Mp), lb, ub)
                    sg.append(s)
                e2.append(dict(eps=eps, sel=nm, signs=sg))
                print(f"  E2 eps {eps:g} {nm:8s}: dd signs {sg}", flush=True)
        rec = dict(D=o.D, kc=o.kc, lam=lam, state=os.path.basename(o.state), n=int(M.shape[0]), lb=lb, ub=ub, e1=e1, e2=e2,
                   e1_dd_agree=len({r["sign_dd"] for r in e1}) == 1, e1_f64_agree=len({r["sign_f64"] for r in e1}) == 1,
                   seconds=round(time.time() - t0))
        print(f"  E1 summary: dd signs agree over the variants: {rec['e1_dd_agree']}; float64 signs agree: "
              f"{rec['e1_f64_agree']}", flush=True)
        open(out, "a").write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        sys.exit(selftest())
    sys.exit(sens([a for a in sys.argv[2:] if not a.startswith("--mutate=")]))
