#!/usr/bin/env python3
"""Checks of the complex-lambda determinant (hp/modecount_cx) before the mode count.

    python3 modecount_check.py STATE.npz [--M=N] [--threads=N] [--out=hp/data/modecount_check.jsonl]

1. complex LU (modecount_cx.cdet) against numpy.linalg.slogdet on random complex matrices;
2. at real lambda: ln|det| and arg det against hpkernel's Solver.mode_det (sign, ln|det|), and the matrices entry by
   entry (Im part zero);
3. analyticity: the complex difference quotient of ln det M along i h against the real one along h (Cauchy-Riemann),
   which tests the complex couplings (dtau and the horizon solves);
4. the real zeros nearest the gauge value lam_g = 1 + <c1> and the formula's lam_phys, by brentq on hpkernel's real
   determinant, against lam_g.
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

    st = setup(args[0], M=int(opts["M"]) if "M" in opts else None, threads=int(opts.get("threads", 1)), log=log)
    D = st.D
    rec.update(D=D, M=st.M, kfreeze=st.kf, F=st.F, state=os.path.basename(args[0]), lam_g=st.lam_g)
    log(f"D {D}: M {st.M}, kfreeze {st.kf}, |F| {st.F:.1e}; omega {st.omega:.6f}; lam_g = 1 + <c1> = {st.lam_g:.12f}")

    # 2. real lambda: complex kernel against hpkernel
    lam0 = 1 / gamma_formula(D)
    rows = []
    for lam in (0.5, 1.7, 4.0):
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
    # matrices entry by entry at lam = 1.7
    mh, ml = st.s.mode_matrix(st.zh, st.zl, 1.7, 0.0)
    rh, rl, ih, il = st.c.mode_matrix(st.zh, st.zl, 1.7, 0.0)
    dm = np.abs((np.asarray(rh) - np.asarray(mh)) + (np.asarray(rl) - np.asarray(ml))).max()
    sc = np.abs(np.asarray(mh)).max()
    imx = np.abs(np.asarray(ih)).max()
    rec["matrix_diff_real_lambda"] = dict(max_abs_diff=float(dm), max_entry=float(sc), max_imag=float(imx))
    log(f"  M(1.7): max |M_cx - M_hp| = {dm:.1e} (max |M| {sc:.1e}), max |Im M_cx| = {imx:.1e}")

    # 3. Cauchy-Riemann on ln det
    cr = []
    for lam in (1.3, 3.0):
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
    for name, c, w in (("gauge", st.lam_g, 0.02), ("phys", lam0, 0.01)):
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
    with open(out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
