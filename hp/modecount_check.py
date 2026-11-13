#!/usr/bin/env python3
"""Checks of the complex-lambda determinant (hp/modecount_cx) before the mode count.

    python3 modecount_check.py STATE.npz [--M=N] [--threads=N] [--quick] [--regress=CACHE] [--regress_n=8]
                               [--mutate=dmat] [--out=hp/data/modecount_check.jsonl]

--quick: item 2 at lambda = 1.7 only, no item 3, and only the gauge zero in item 4.

1. complex LU (modecount_cx.cdet) against numpy.linalg.slogdet on random complex matrices;
2. at real lambda: ln|det| and arg det against hpkernel's Solver.mode_det (sign, ln|det|), and the matrices at 1.7
   entry by entry, bit for bit (hi and lo parts; Im part zero). Control: the complex kernel at the next double above
   1.7 must differ from hpkernel's at 1.7;
3. analyticity: the complex difference quotient of ln det M along i h against the real one along h (Cauchy-Riemann),
   which tests the complex couplings (dtau and the horizon solves);
4. the real zeros nearest the gauge value lam_g = 1 + <c1> and the formula's lam_phys, by brentq on hpkernel's real
   determinant, against lam_g;
5. with --regress=CACHE, a modecount_<tag>_evals.jsonl written on the same state and layout (modecount_wind.py,
   modecount_circle.py): CSolver.mode_det at regress_n of its lambda (spread over the file) bit for bit against the
   cached ln|det|, arg det, min |u_kk| and max |u_kk|. The layout's segment plan uses libm, so a cache from another
   operating system can differ in the last bits (notes/HIGH_M.md 9.5). Control: the first of them with Re lambda one
   ulp up must differ in one of the four.

--mutate=dmat, the control of checks 2 and 5: the complex kernel is built with the differentiation matrix's largest
entry of row 0 moved by one ulp of its high part (modecount_pivots.csolver), so both must fail.
Exit 1 if check 2 or 5 fails or one of their controls passes.
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from modecount_common import setup, timer  # noqa: E402
from gamma_dd import gamma_formula  # noqa: E402
import modecount_cx  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    out = opts.get("out", os.path.join(HERE, "data", "modecount_check.jsonl"))
    log = lambda m: print(m, flush=True)
    rec = {}
    # 1. complex LU
    rng = np.random.default_rng(1)
    errs = []
    for n in (7, 40, 129):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        z = np.zeros((n, n))
        la, arg, _, _ = modecount_cx.cdet(np.ascontiguousarray(A.real), z, np.ascontiguousarray(A.imag), z)
        s, ln = np.linalg.slogdet(A)
        errs.append(max(abs(la - ln), abs(np.angle(np.exp(1j * (arg - np.angle(s)))))))
    rec["cdet_vs_numpy_max_err"] = max(errs)
    log(f"complex LU vs numpy slogdet (n = 7, 40, 129): max error {max(errs):.1e}")

    threads = int(opts.get("threads", 1))
    st = setup(args[0], M=int(opts["M"]) if "M" in opts else None, threads=threads, log=log)
    bad = []
    if opts.get("mutate") == "dmat":
        from modecount_pivots import csolver
        from hpkernel.driver import dmat_odd
        j = int(np.argmax(np.abs(np.asarray(dmat_odd(st.M)[0])[0])))
        st.c = csolver(st, 1, (0, j), threads)
        rec["mutate"] = f"dmat[0, {j}] one ulp up"
        log(f"--mutate=dmat: the complex kernel's dmat[0, {j}] moved one ulp up (checks 2 and 5 must fail)")
    elif "mutate" in opts:
        sys.exit(f"--mutate takes dmat, not {opts['mutate']}")
    D = st.D
    rec.update(D=D, M=st.M, kfreeze=st.kf, F=st.F, state=os.path.basename(args[0]), lam_g=st.lam_g)
    log(f"D {D}: M {st.M}, kfreeze {st.kf}, |F| {st.F:.1e}; omega {st.omega:.6f}; lam_g = 1 + <c1> = {st.lam_g:.12f}")

    # 2. real lambda: complex kernel against hpkernel
    lam0 = 1 / gamma_formula(D)
    rows = []
    quick = "quick" in opts
    for lam in ((1.7,) if quick else (0.5, 1.7, 4.0)):
        t = timer()
        sg, la, pmin, pmax = st.det_real(lam)
        t1 = t()
        t = timer()
        lc, ac, pminc, pmaxc = st.det(lam, 0.0)
        t2 = t()
        ok_sign = (sg > 0) == (abs(ac) < 1e-9)
        rows.append(dict(lam=lam, sign=sg, lndet=la, lndet_c=lc, arg_c=ac, dln=lc - la, sign_ok=ok_sign, t_real=t1, t_cx=t2))
        log(f"  lam {lam}: hp sign {sg:+.0f} ln|det| {la:.12f} ({t1:.1f} s) | cx ln|det| {lc:.12f} arg {ac:+.3e} "
            f"({t2:.1f} s): d ln {lc - la:+.1e}, sign {'ok' if ok_sign else 'MISMATCH'}")
    rec["real_lambda"] = rows
    # matrices entry by entry at lam = 1.7, bit for bit, and the control one ulp up
    bits = lambda a: np.asarray(a, dtype=float).view(np.uint64)  # noqa: E731
    mh, ml = (np.asarray(a) for a in st.s.mode_matrix(st.zh, st.zl, 1.7, 0.0))
    rh, rl, ih, il = (np.asarray(a) for a in st.c.mode_matrix(st.zh, st.zl, 1.7, 0.0))
    ndiff = int(np.sum((bits(rh) != bits(mh)) | (bits(rl) != bits(ml))))
    dm = np.abs((rh - mh) + (rl - ml)).max()
    sc = np.abs(mh).max()
    imx = float(np.abs(ih).max() + np.abs(il).max())
    ch, cl, _, _ = (np.asarray(a) for a in st.c.mode_matrix(st.zh, st.zl, float(np.nextafter(1.7, 2.0)), 0.0))
    nctl = int(np.sum((bits(ch) != bits(mh)) | (bits(cl) != bits(ml))))
    ok2 = ndiff == 0 and imx == 0.0
    rec["matrix_diff_real_lambda"] = dict(entries=int(mh.size), differing=ndiff, max_abs_diff=float(dm),
                                          max_entry=float(sc), max_imag=imx, control_differing=nctl, ok=ok2)
    log(f"  M(1.7): {ndiff} of {mh.size} entries differ from hpkernel's in a bit, max |M_cx - M_hp| = {dm:.1e} "
        f"(max |M| {sc:.1e}), max |Im M_cx| = {imx:.1e}: {'pass' if ok2 else 'FAIL'}; control (cx one ulp above 1.7): "
        f"{nctl} differ, {'fires' if nctl else 'DOES NOT FIRE'}")
    if not ok2:
        bad.append("check 2: the matrices at 1.7 differ")
    if not nctl:
        bad.append("control 2: one ulp of lambda changed no entry")

    # 3. Cauchy-Riemann on ln det
    cr = []
    for lam in (() if quick else (1.3, 3.0)):
        h = 1e-5
        l0, a0, _, _ = st.det(lam, 0.0)
        lx, ax, _, _ = st.det(lam + h, 0.0)
        ly, ay, _, _ = st.det(lam, h)
        dx = complex(lx - l0, math.remainder(ax - a0, 2 * math.pi)) / h
        dy = complex(ly - l0, math.remainder(ay - a0, 2 * math.pi)) / (1j * h)
        cr.append(dict(lam=lam, h=h, dlogdet_real_dir=[dx.real, dx.imag], dlogdet_imag_dir=[dy.real, dy.imag],
                       rel_diff=abs(dx - dy) / abs(dx)))
        log(f"  d ln det / d lam at {lam}: along h {dx:.8f}, along i h {dy:.8f}, rel. diff {abs(dx - dy) / abs(dx):.1e}")
    rec["cauchy_riemann"] = cr

    # 4. real zeros: gauge and physical
    def f(lam):
        sg, la, _, _ = st.det_real(lam)
        return sg * math.exp(la - ref)

    zeros = {}
    for name, c, w in (("gauge", st.lam_g, 0.02),) + ((("phys", lam0, 0.01),) if not quick else ()):
        ref = st.det_real(c)[1]
        a, b = c - w, c + w
        fa, fb = f(a), f(b)
        if fa * fb > 0:
            zeros[name] = None
            log(f"  no sign change in [{a:.4f}, {b:.4f}] around {name} {c:.6f}")
            continue
        z = brentq(f, a, b, xtol=1e-13, rtol=1e-15)
        zeros[name] = z
        log(f"  real zero near {name} value {c:.12f}: {z:.12f} (diff {z - c:+.1e})")
    rec["real_zeros"] = zeros

    # 5. bit for bit against cached determinants of the same state and layout
    if "regress" in opts:
        ev = [json.loads(line) for line in open(opts["regress"])]
        n = min(int(opts.get("regress_n", 8)), len(ev))
        pick = [ev[i] for i in np.unique(np.linspace(0, len(ev) - 1, n).round().astype(int))]
        keys = ("lnabs", "arg", "pmin", "pmax")
        reg = []
        for k, e in enumerate(pick + [dict(pick[0], re=float(np.nextafter(pick[0]["re"], np.inf)), ctl=True)]):
            t = timer()
            got = st.det(e["re"], e["im"])
            same = [bits(g) == bits(e[q]) for g, q in zip(got, keys)]
            ctl = e.get("ctl", False)
            reg.append(dict(re=e["re"], im=e["im"], control=ctl, cached=[e[q] for q in keys], got=list(got),
                            equal=[bool(s) for s in same], sec=t()))
            log(f"  5. {'control, ' if ctl else ''}lambda {e['re']!r}{e['im']:+}i: ln|det| {got[0]!r} (cached "
                f"{e['lnabs']!r}), arg {got[1]!r} ({e['arg']!r}); four floats equal {sum(same)} of 4"
                + ("" if ctl else (": pass" if all(same) else ": FAIL")))
            if ctl and all(same):
                bad.append("control 5: one ulp of Re lambda left the determinant's four floats unchanged")
            elif not ctl and not all(same):
                bad.append(f"check 5 failed at {e['re']}{e['im']:+}i")
        rec["regression"] = dict(cache=os.path.basename(opts["regress"]), points=reg)
    rec["failed"] = bad
    with open(out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
    for b in bad:
        log(f"FAIL: {b}")
    log("all checks pass" if not bad else f"{len(bad)} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
