#!/usr/bin/env python3
"""Which pivots of det M(lambda)'s LU carry the change of ln|det M| under one ulp of the differentiation matrix (#552).

    cd hp && python3 modecount_pivots.py STATE.npz RE IM [--M=48] [--kf=29] [--fmax=1e-15] [--entry=10,11]
        [--threads=1] [--json=OUT]

The state is set up as hp/modecount_circle.py sets it up (modecount_common.setup with the same --M, --kf and --fmax).
M(lambda) at lambda = RE + i IM is factored by modecount_cx's mode_pivots (multidouble's clu_logdet LU with every
pivot recorded) with the double-double differentiation matrix as it is (shift 0) and with its entry (I, J) moved up and
down by two sizes of perturbation:
  * part hi: one ulp of the high part (np.nextafter, as I506-macm3/ulp_circle.py does), a relative change of about
    u_hi = 2^-53 of the entry, the perturbation of #498, #506 and #520;
  * part lo: one unit in the last place of the double-double value, the low part moved by spacing(hi) 2^-53, a
    relative change of about u_lo = 2^-106, the size of double-double's own rounding.
For each part and shift the change of ln|u_kk| in elimination order is compared with the plain LU: the total change of
ln|det| and of arg det (wrapped to (-pi, pi], the quantity a winding count reads), how many pivots carry 90% of the
summed |change|, their sizes against 1e3 u_part ||M||_F (the perturbation's
own scale) and how many pivot rows differ. Verdict, per part and shift:
  * localized: at most 5% of the pivots carry 90% of the summed |change|, and each of them is below 1e3 u_part ||M||_F;
  * global: more than half of the pivots are needed for 90%;
  * none: no pivot moves by more than 1e-3 in ln;
  * between: anything else, reported with the numbers.

Checks (exit 1 on failure):
  * consistency: mode_pivots' ln|det|, arg det, min and max |u_kk| equal mode_det's bit for bit;
  * determinism: a CSolver built afresh with shift 0 gives the plain run's pivots bit for bit;
  * planted (run first, through cpivots, the same LU): A = L U in mpmath at 60 digits, with |l_ij| < 1 so that
    partial pivoting keeps the diagonal and the pivots are U's diagonal, split into double-double parts. With r of
    the diagonal entries 1e-33 (below double-double's rounding of A), one ulp of one entry's high part must move
    exactly r pivots by more than 1e-3 in ln, and the verdict must be localized; with r = 0 (the control) none may
    move.
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modecount_common import setup  # noqa: E402  (also puts modecount_cx/build on the path)
from hpkernel.driver import dd_from_mp, dmat_odd, phase_weights, plow_odd  # noqa: E402
import modecount_cx  # noqa: E402

U_PART = {"hi": 2.0 ** -53, "lo": 2.0 ** -106}
MOVED = 1e-3


def wrap(x):
    """an angle difference in (-pi, pi]"""
    return float(-((-x + np.pi) % (2 * np.pi) - np.pi))


def compare(p, q, n, mnorm, u=U_PART["hi"]):
    """the change of the LU q against the plain LU p: a dict of the numbers and the verdict"""
    d = np.asarray(q[4]) - np.asarray(p[4])
    total = q[0] - p[0]
    darg = wrap(q[1] - p[1])
    a = np.abs(d)
    order = np.argsort(-a, kind="stable")
    s = a.sum()
    if s == 0.0:
        return dict(total=total, darg=darg, moved=0, carry90=0, rows_differ=int(np.count_nonzero(np.asarray(p[6]) != np.asarray(q[6]))),
                    top=[], verdict="none")
    cum = np.cumsum(a[order]) / s
    k90 = int(np.searchsorted(cum, 0.9) + 1)
    carriers = order[:k90]
    thr = 1e3 * u * mnorm
    small = bool(np.all(np.exp(np.asarray(p[4])[carriers]) < thr))
    moved = int(np.count_nonzero(a > MOVED))
    if moved == 0:
        verdict = "none"
    elif k90 <= 0.05 * n and small:
        verdict = "localized"
    elif k90 > n / 2:
        verdict = "global"
    else:
        verdict = "between"
    top = [dict(k=int(k), row_plain=int(p[6][k]), row_shift=int(q[6][k]), u_plain=float(np.exp(p[4][k])),
                u_shift=float(np.exp(q[4][k])), dln=float(d[k]), darg_u=wrap(q[5][k] - p[5][k])) for k in order[:10]]
    return dict(total=total, darg=darg, moved=moved, carry90=k90, carriers_below_thr=small,
                thr=thr, rows_differ=int(np.count_nonzero(np.asarray(p[6]) != np.asarray(q[6]))), top=top,
                verdict=verdict)


def dd_parts(x):
    hi = float(x)
    return hi, float(x - hi)


def planted(n, r, seed=5):
    """A = L U with |l_ij| < 1 and U's last r diagonal entries 1e-33, as four (n, n) double-double arrays"""
    rng = np.random.default_rng(seed)
    with mp.workdps(60):
        L = [[mp.mpc(0) for _ in range(n)] for _ in range(n)]
        U = [[mp.mpc(0) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            L[i][i] = mp.mpc(1)
            for j in range(i):
                L[i][j] = mp.mpc(*(0.45 * rng.uniform(-1, 1, 2)))
            for j in range(i, n):
                U[i][j] = mp.mpc(*rng.uniform(-1, 1, 2))
            if i < n - r:
                U[i][i] = mp.mpc(1.5 + rng.uniform(0, 0.5), rng.uniform(-0.5, 0.5))
            else:
                U[i][i] = mp.mpc("1e-33", "1e-33")
        out = [np.zeros((n, n)) for _ in range(4)]
        for i in range(n):
            for j in range(n):
                a = mp.fsum(L[i][k] * U[k][j] for k in range(min(i, j) + 1))
                out[0][i, j], out[1][i, j] = dd_parts(a.real)
                out[2][i, j], out[3][i, j] = dd_parts(a.imag)
    return out


def planted_check(n=40, r=3):
    """the planted control: (r moved pivots, localized) for r tiny pivots, and none moved for r = 0"""
    ok = True
    for rr in (r, 0):
        A = planted(n, rr)
        p = modecount_cx.cpivots(*A)
        B = [x.copy() for x in A]
        B[0][5, 0] = np.nextafter(B[0][5, 0], np.inf)
        q = modecount_cx.cpivots(*B)
        mnorm = float(np.sqrt(np.sum(A[0] ** 2 + A[2] ** 2)))
        c = compare(p, q, n, mnorm, U_PART["hi"])
        good = c["moved"] == rr and c["verdict"] == ("localized" if rr else "none")
        print(f"planted n {n}, r {rr}: moved {c['moved']} (want {rr}), carry90 {c['carry90']}, verdict {c['verdict']}"
              f": {'pass' if good else 'FAIL'}", flush=True)
        ok = ok and good
    return ok


def csolver(st, shift, entry, threads, part="hi"):
    """modecount_common.build_pair's CSolver, with the differentiation matrix's entry moved by one unit of the high
    part (part hi) or of the double-double value (part lo) in the direction of shift (0: as it is)"""
    meta, M = st.meta, st.M
    with mp.workdps(40):
        nh, nl = dd_from_mp([mp.mpf(meta["D"]) - 2])
    dh, dl = dmat_odd(M)
    dh, dl = np.array(dh, dtype=float, copy=True), np.array(dl, dtype=float, copy=True)
    if shift:
        i, j = entry
        if part == "hi":
            dh[i, j] = np.nextafter(dh[i, j], np.inf if shift > 0 else -np.inf)
        else:
            dl[i, j] += np.sign(shift) * np.spacing(dh[i, j]) * 2.0 ** -53
    ph, pl = phase_weights(M)
    a = b = None
    if st.kf:
        a, b = plow_odd(M, st.kf)
    return modecount_cx.CSolver(M, float(nh[0]), float(nl[0]), dh, dl, ph, pl, list(meta["nodes"]), st.s.get_segs(),
                                meta.get("kt", 36), meta.get("kc", 420), meta.get("ks", 100), threads, a, b, None)


def bits(x):
    return np.asarray(x, dtype=float).view(np.uint64)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(args) != 3:
        sys.exit(__doc__)
    path, re_, im_ = args[0], float(args[1]), float(args[2])
    threads = int(opts.get("threads", 1))
    entry = tuple(int(x) for x in opts.get("entry", "10,11").split(","))
    t0 = time.time()
    ok = planted_check()
    kf = "default" if "kf" not in opts else (None if opts["kf"] == "none" else int(opts["kf"]))
    st = setup(path, M=int(opts["M"]) if "M" in opts else None, threads=threads, kf=kf,
               fmax=float(opts.get("fmax", 1e-20)))
    n = st.c.size - 1
    print(f"{os.path.basename(path)}: D {st.D}, M {st.M}, kf {st.kf}, n {n}, lambda {re_} {im_:+}i, "
          f"entry dh{list(entry)} ({time.time() - t0:.0f} s)", flush=True)
    rh, rl, ih, il = st.c.mode_matrix(st.zh, st.zl, re_, im_)
    mnorm = float(np.sqrt(np.sum(np.asarray(rh) ** 2 + np.asarray(ih) ** 2)))
    det = st.c.mode_det(st.zh, st.zl, re_, im_)
    p = st.c.mode_pivots(st.zh, st.zl, re_, im_)
    cons = bool(np.array_equal(bits(det), bits(p[:4])))
    print(f"  plain: ln|det| {p[0]:.6f}, arg {p[1]:+.6f}, pmin {p[2]:.3e}, pmax {p[3]:.3e}, ||M||_F {mnorm:.3e}; "
          f"consistency with mode_det: {'pass' if cons else 'FAIL'}", flush=True)
    p0 = csolver(st, 0, entry, threads).mode_pivots(st.zh, st.zl, re_, im_)
    det_ok = bool(np.array_equal(bits(p[:4]), bits(p0[:4])) and np.array_equal(bits(p[4]), bits(p0[4]))
                  and np.array_equal(bits(p[5]), bits(p0[5])) and np.array_equal(p[6], p0[6]))
    print(f"  determinism (a fresh CSolver at shift 0): {'pass' if det_ok else 'FAIL'}", flush=True)
    ok = ok and cons and det_ok
    rec = dict(state=path, D=st.D, M=st.M, kf=st.kf, n=n, lam=[re_, im_], entry=list(entry), mnorm=mnorm,
               plain=dict(lndet=p[0], arg=p[1], pmin=p[2], pmax=p[3]), consistency=cons, determinism=det_ok, shifts={})
    for part, sh in (("hi", 1), ("hi", -1), ("lo", 1), ("lo", -1)):
        q = csolver(st, sh, entry, threads, part).mode_pivots(st.zh, st.zl, re_, im_)
        c = compare(p, q, n, mnorm, U_PART[part])
        rec["shifts"][f"{part}{sh:+d}"] = dict(c, lndet=q[0], arg=q[1], pmin=q[2])
        print(f"  part {part}, shift {sh:+d}: ln|det| {q[0]:.6f} (change {c['total']:+.3f}), arg det {q[1]:+.6f} (change "
              f"{c['darg']:+.4f} rad), pmin {q[2]:.3e}; pivots moved by more "
              f"than {MOVED:g} in ln: {c['moved']} of {n}; 90% of the summed |change| in {c['carry90']} pivots "
              f"(all below 1e3 u_{part} ||M|| = {c.get('thr', 0):.1e}: {c.get('carriers_below_thr')}); rows differ at "
              f"{c['rows_differ']}; verdict {c['verdict']}", flush=True)
        for t in c["top"]:
            print(f"    k {t['k']:4d}: |u| {t['u_plain']:.3e} -> {t['u_shift']:.3e}, d ln {t['dln']:+.3f}, "
                  f"d arg {t['darg_u']:+.3f}, rows "
                  f"{t['row_plain']} -> {t['row_shift']}")
    print("check:", "pass" if ok else "FAIL", f"({time.time() - t0:.0f} s)")
    if opts.get("json"):
        with open(opts["json"], "w") as fh:
            json.dump(rec, fh, indent=1, default=float)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
