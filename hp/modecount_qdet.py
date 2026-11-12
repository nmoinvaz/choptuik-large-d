#!/usr/bin/env python3
"""det M(lambda) of the double-double mode matrix in ball arithmetic, beside the double-double LU's (#584).

    cd hp && python3 modecount_qdet.py STATE.npz RE,IM [RE,IM ...] [--M=N] [--kf=none|13] [--data=hp/data]
                                       [--fmax=1e-20] [--threads=N] [--prec=320] [--ulp] [--ulps=1,2,4,8] [--draws=0]
                                       [--json=out.json]
    python3 hp/modecount_qdet.py --selftest [--mutate=no_lo]

At each lambda the matrix that mode_det factors (CSolver.mode_matrix, four double-double parts) is converted exactly
to complex balls and its determinant taken with python-flint's acb_mat.det at --prec bits (Arb's LU, which shares
nothing with multidouble's). That is the exact determinant of the double-double matrix, enclosed, so its difference
from mode_det's is the double-double LU's rounding alone, apart from the matrix's.

--ulp also takes both at Re lambda moved up by one f64 ulp, the step of #585. When the exact determinants of the two
matrices differ as much as mode_det's do, the noise is in the matrix entries (their assembly in double-double), and a
quad-double LU cannot remove it. When they agree and mode_det's differ, it is the LU's.
--ulps=K1,K2,... takes the exact determinant at Re lambda moved by each K f64 ulps up and down (#605): a phase change
that grows like K is a derivative, one that does not is noise in the matrix's assembly at the ulp scale.
--draws=N adds N determinants of the matrix with every entry moved by a uniform relative amount in [-u, u],
u = 2^-106 (double-double's rounding): how much one rounding per entry moves arg det. It is a lower bound on the
matrix's own noise, since the assembly rounds many times per entry.

Every run starts with the planted check (--selftest alone runs only it): a permuted triangular matrix of known
determinant, and a random 30 x 30 double-double matrix against mpmath's LU at 80 digits, where the two must agree to
1e-40 relative. --mutate=no_lo drops the low parts in the conversion, the control, and the check must then fail.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

U_DD = 2.0 ** -106


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def to_acb(rh, rl, ih, il, prec, mutate=""):
    """the double-double matrix as an exact acb_mat (every h + l is exact at prec >= 2 * 53 + 2 for a normalised pair);
    mutate=no_lo drops the low parts (the planted check's control)"""
    from flint import acb, acb_mat, arb, ctx
    ctx.prec = prec
    n = rh.shape[0]
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            if mutate == "no_lo":
                re_, im_ = arb(float(rh[i, j])), arb(float(ih[i, j]))
            else:
                re_ = arb(float(rh[i, j])) + arb(float(rl[i, j]))
                im_ = arb(float(ih[i, j])) + arb(float(il[i, j]))
            row.append(acb(re_, im_))
        rows.append(row)
    A = acb_mat(rows)
    return A


def exact_part(A):
    """True when every entry of A is exact (zero radius)"""
    n = A.nrows()
    return all(A[i, j].real.rad() == 0 and A[i, j].imag.rad() == 0 for i in range(n) for j in range(n))


def ball_det(rh, rl, ih, il, prec, mutate=""):
    """(ln|det|, arg det, radius of ln|det|, radius of arg, exact input, seconds, log det as an acb) of the
    double-double matrix in balls"""
    from flint import ctx
    t = time.time()
    A = to_acb(rh, rl, ih, il, prec, mutate)
    ex = exact_part(A)
    d = A.det()
    ctx.prec = prec
    lg = d.log()
    return (float(lg.real.mid()), float(lg.imag.mid()), float(lg.real.rad()), float(lg.imag.rad()), ex,
            round(time.time() - t, 2), lg)


def ball_det_real(mh, ml, prec):
    """(sign, ln|det|, radius of ln|det|) of a real double-double matrix (mh, ml) in balls, the exact determinant of
    the double-double matrix; sign 0 when the ball contains 0 (gamma_dd.py --ball, #590)"""
    from flint import arb, arb_mat, ctx
    ctx.prec = prec
    n = mh.shape[0]
    A = arb_mat([[arb(float(mh[i, j])) + arb(float(ml[i, j])) for j in range(n)] for i in range(n)])
    d = A.det()
    if d.contains(0):
        return 0.0, float("nan"), float("inf")
    sg = 1.0 if d > 0 else -1.0
    lg = abs(d).log()
    return sg, float(lg.mid()), float(lg.rad())


def ball_diff(b2, b1):
    """(d ln|det|, d arg) between two ball_det results, taken in balls (an f64 difference of the two would floor at
    an ulp of arg); the arg difference wrapped to (-pi, pi]"""
    from flint import arb
    d = b2[6] - b1[6]
    da = d.imag
    k = round(float(da.mid()) / (2 * math.pi))
    return float(d.real.mid()), float((da - 2 * k * arb.pi()).mid())


def split_dd(x):
    """an mpmath value rounded to a double-double pair (hi, lo)"""
    import mpmath as mp
    h = float(x)
    return h, float(x - mp.mpf(h))


def planted_check(mutate="", log=print):
    """(a) a permuted triangular matrix with known determinant; (b) a random 30 x 30 double-double matrix against
    mpmath's LU at 80 digits, relative agreement 1e-40; with mutate=no_lo (b) must fail"""
    import mpmath as mp
    from flint import acb, ctx
    rng = np.random.default_rng(584)
    ok = True
    # (a) U upper triangular with diagonal d (double-double), rows permuted by a 3-cycle (an even permutation)
    n = 12
    mp.mp.dps = 80
    d = [mp.mpc(rng.uniform(0.5, 2), rng.uniform(-1, 1)) * mp.mpf(10) ** int(rng.integers(-8, 8)) for _ in range(n)]
    U = [[mp.mpc(0)] * n for _ in range(n)]
    for i in range(n):
        U[i][i] = d[i]
        for j in range(i + 1, n):
            U[i][j] = mp.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1))
    perm = [1, 2, 0] + list(range(3, n))
    P = [U[perm[i]] for i in range(n)]
    parts = [np.zeros((n, n)) for _ in range(4)]
    for i in range(n):
        for j in range(n):
            parts[0][i, j], parts[1][i, j] = split_dd(P[i][j].real)
            parts[2][i, j], parts[3][i, j] = split_dd(P[i][j].imag)
    # the exact det of the rounded matrix is the product of its rounded diagonal (triangular after the permutation)
    dr = [mp.mpc(mp.mpf(parts[0][perm.index(i), i]) + mp.mpf(parts[1][perm.index(i), i]),
                 mp.mpf(parts[2][perm.index(i), i]) + mp.mpf(parts[3][perm.index(i), i])) for i in range(n)]
    want = mp.fprod(dr)
    la, ar, rla, rar, ex, _, _ = ball_det(*parts, 320, mutate)
    ea = abs(la - float(mp.log(abs(want)))) + abs(wrap(ar - float(mp.arg(want))))
    pa = ea < 1e-12 and ex
    log(f"  planted (a), permuted triangular n {n}: ln|det| {la:.15f} against {float(mp.log(abs(want))):.15f}, "
        f"error {ea:.1e}, exact input {ex}: {'pass' if pa else 'FAIL'}")
    ok = ok and pa
    # (b) random dense, the mpmath LU at 80 digits on the same double-double values
    n = 30
    parts = [rng.standard_normal((n, n)) for _ in range(2)]
    parts = [parts[0], parts[0] * rng.uniform(-1, 1, (n, n)) * 2.0 ** -53,
             parts[1], parts[1] * rng.uniform(-1, 1, (n, n)) * 2.0 ** -53]
    Am = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            Am[i, j] = mp.mpc(mp.mpf(parts[0][i, j]) + mp.mpf(parts[1][i, j]),
                              mp.mpf(parts[2][i, j]) + mp.mpf(parts[3][i, j]))
    want = mp.det(Am)
    ctx.prec = 320
    A = to_acb(*parts, 320, mutate)
    got = A.det()
    w = acb(mp.nstr(want.real, 70), mp.nstr(want.imag, 70))
    rel = float(abs(got - w).mid()) / float(abs(w).mid())
    pb = rel < 1e-40
    log(f"  planted (b), random 30 x 30 against mpmath at 80 digits: relative difference {rel:.1e} (bound 1e-40)"
        f"{', mutate ' + mutate if mutate else ''}: {'pass' if pb else 'FAIL'}")
    return ok and pb


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    mutate = opts.get("mutate", "")
    t0 = time.time()
    if "selftest" in opts:
        ok = planted_check(mutate)
        print("selftest:", "pass" if ok else "FAIL")
        return 0 if ok else 1
    if len(args) < 2:
        sys.exit(__doc__)
    ok = planted_check()
    from modecount_common import setup
    path = args[0]
    kf = "default" if "kf" not in opts else (None if opts["kf"] == "none" else int(opts["kf"]))
    threads = int(opts.get("threads", 1))
    st = setup(path, M=int(opts["M"]) if "M" in opts else None, threads=threads, kf=kf,
               fmax=float(opts.get("fmax", 1e-20)))
    import modecount_cx
    from flint import ctx
    ctx.threads = threads
    prec = int(opts.get("prec", 320))
    draws = int(opts.get("draws", 0))
    n = st.c.size - 1
    print(f"{os.path.basename(path)}: D {st.D}, M {st.M}, kf {st.kf}, n {n}, prec {prec} bits ({time.time() - t0:.0f} s)",
          flush=True)
    rng = np.random.default_rng(int(opts.get("seed", 584)))
    recs = []

    def one(re_, im_):
        p = st.c.mode_det(st.zh, st.zl, re_, im_)
        A = [np.ascontiguousarray(x) for x in st.c.mode_matrix(st.zh, st.zl, re_, im_)]
        c = modecount_cx.cdet(*A)
        b = ball_det(*A, prec)
        return p, A, c, b

    for a in args[1:]:
        re_, im_ = (float(x) for x in a.split(","))
        p, A, c, b = one(re_, im_)
        cons = bool(np.array_equal(np.array(p[:2]), np.array(c[:2])))
        r = dict(lam=[re_, im_], dd=dict(lndet=p[0], arg=p[1], pmin=p[2], pmax=p[3]),
                 cdet_matches_mode_det=cons,
                 ball=dict(lndet=b[0], arg=b[1], rad_lndet=b[2], rad_arg=b[3], exact_input=b[4], sec=b[5]),
                 lu_dlndet=p[0] - b[0], lu_darg=wrap(p[1] - b[1]))
        ok = ok and b[4]
        print(f"lambda {re_!r} {im_:+.17g}i: double-double ln|det| {p[0]:.12f} arg {p[1]:+.12f} (pmin {p[2]:.2e}); "
              f"ball ln|det| {b[0]:.12f} +- {b[2]:.1e}, arg {b[1]:+.12f} +- {b[3]:.1e} ({b[5]:.0f} s); "
              f"the LU's rounding: d ln|det| {r['lu_dlndet']:+.2e}, d arg {r['lu_darg']:+.2e} rad; "
              f"cdet = mode_det: {cons}", flush=True)
        if "ulp" in opts:
            re2 = float(np.nextafter(re_, np.inf))
            p2, A2, c2, b2 = one(re2, im_)
            bdl, bda = ball_diff(b2, b)
            r["ulp"] = dict(re=re2, dd=dict(lndet=p2[0], arg=p2[1]), ball=dict(lndet=b2[0], arg=b2[1], rad_arg=b2[3]),
                            dd_dlndet=p2[0] - p[0], dd_darg=wrap(p2[1] - p[1]),
                            ball_dlndet=bdl, ball_darg=bda,
                            entries_differ=int(sum(int(np.sum(x != y)) for x, y in zip(A, A2))))
            u = r["ulp"]
            print(f"  one ulp up in Re: double-double d arg {u['dd_darg']:+.3e}, d ln|det| {u['dd_dlndet']:+.3e}; "
                  f"exact det of the two matrices d arg {u['ball_darg']:+.3e}, d ln|det| {u['ball_dlndet']:+.3e}; "
                  f"{u['entries_differ']} of {4 * n * n} parts differ", flush=True)
        if "ulps" in opts:
            # the exact determinant at Re lambda moved by k f64 ulps, up and down (#605): a derivative grows like k,
            # noise at the ulp scale does not; ball dets only, each against the point's own
            rows = []
            for k in (int(x) for x in opts["ulps"].split(",")):
                for sgn in (1, -1):
                    re_k = re_
                    for _ in range(k):
                        re_k = float(np.nextafter(re_k, sgn * np.inf))
                    Ak = [np.ascontiguousarray(x) for x in st.c.mode_matrix(st.zh, st.zl, re_k, im_)]
                    bk = ball_det(*Ak, prec)
                    dl, da = ball_diff(bk, b)
                    rows.append(dict(ulps=sgn * k, re=re_k, darg=da, dlndet=dl, rad_arg=bk[3]))
                    print(f"  {sgn * k:+d} ulp in Re: exact det d arg {da:+.3e}, d ln|det| {dl:+.3e}", flush=True)
            r["ulps"] = rows
        if draws:
            ds = []
            for k in range(draws):
                # each value h + l moves by delta h, delta uniform in [-u, u], through its low part
                Bh = [A[0], A[1] + A[0] * rng.uniform(-U_DD, U_DD, A[0].shape),
                      A[2], A[3] + A[2] * rng.uniform(-U_DD, U_DD, A[2].shape)]
                bdl, bda = ball_diff(ball_det(*Bh, prec), b)
                ds.append(dict(darg=bda, dlndet=bdl))
            r["draws"] = ds
            sd = float(np.std([x["darg"] for x in ds])) if draws > 1 else abs(ds[0]["darg"])
            dargs = ", ".join(f"{x['darg']:+.2e}" for x in ds)
            r["floor"] = max(abs(x["darg"]) for x in ds)
            print(f"  {draws} draws of one double-double rounding per entry: d arg {dargs} (spread {sd:.2e} rad, "
                  f"largest {r['floor']:.2e})", flush=True)
        recs.append(r)
        # the record after every point, so a stopped run keeps what it did
        rec = dict(state=path, D=st.D, M=st.M, kf=st.kf, n=n, prec=prec, points=recs, planted=ok)
        if opts.get("json"):
            with open(opts["json"], "w") as fh:
                json.dump(rec, fh, indent=1, default=float)
    print("check:", "pass" if ok else "FAIL", f"({time.time() - t0:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
