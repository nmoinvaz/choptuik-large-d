#!/usr/bin/env python3
"""Checks of hp/modecount_cx after its port to the current kernel (scaled series, multidouble lanes) and of its
test-field sector (notes/GHOST_ADMIXTURE.md), on one saved state, before any count.

    python3 modecount_test_check.py STATE.npz [--threads=N] [--regress=hp/data/modecount_<tag>_evals.jsonl]
                                    [--out=hp/data/modecount_test_check.jsonl]

1. Port: at real lambda (0, 1.7) the complex kernel's mode matrix (test field off) against hpkernel's
   Solver.mode_matrix, entry by entry, bit for bit (hi and lo parts; Im part zero).
   Control: the complex kernel at the next double above 1.7 must differ.
2. Regression (--regress, a cache of the old crate's evaluations on the same state and layout, written by
   modecount_wind.py before the port): ln|det| and arg det at two cached complex lambda.
3. Test field, the identity M_psi(0; z) z_psi = F_psi(z): the p, m equations are linear in p, m with a, c held, so at
   lambda = 0 the test field's matrix applied to the state's own p, m gives the p, m rows of the residual, for any z.
   Checked at the state with every unknown but Delta perturbed by 1e-6 (random; at 1e-4 the Taylor steps planned for
   the state can reach a singular point of the perturbed fields and return NaN): max |difference| / max |F_psi| below
   1e-20. Control: mode 2 (a's tangent kept, the coupling p -> a -> p put back) must miss by more than 1e-8.
4. Test field at the state itself: |M_psi(0) z_psi| / |z_psi| is the residual's size (lambda = 0 is a zero); ln|det|
   and min |u_kk| at lambda = 0 against lambda = +-0.1.
5. Analyticity of det M_psi (Cauchy-Riemann at lambda = 1.3, step 1e-5): the difference quotients of ln det along h
   and along i h agree to 1e-4 relative. Control: the same with det M_psi(conj lambda) along i h (anti-analytic) must
   disagree by more than 0.5.
Exit 1 if a check fails or a control passes.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from modecount_common import setup, timer  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    out = opts.get("out", os.path.join(HERE, "data", "modecount_test_check.jsonl"))
    log = lambda m: print(m, flush=True)  # noqa: E731
    bad = []
    st = setup(args[0], threads=int(opts.get("threads", 1)), log=log)
    rec = dict(state=os.path.basename(args[0]), D=st.D, M=st.M, K=st.K, kfreeze=st.kf, F=st.F)
    log(f"D {st.D}: M {st.M}, K {st.K}, kfreeze {st.kf}, |F| {st.F:.1e}, omega {st.omega:.6f}")
    zh, zl = st.zh, st.zl

    # 1. port: bit for bit at real lambda
    rows = []
    for lam in (0.0, 1.7, "ctl"):
        t = timer()
        lc = np.nextafter(1.7, 2.0) if lam == "ctl" else lam
        mh, ml = (np.asarray(a) for a in st.s.mode_matrix(zh, zl, 1.7 if lam == "ctl" else lam, 0.0))
        rh, rl, ih, il = (np.asarray(a) for a in st.c.mode_matrix(zh, zl, lc, 0.0))
        ndiff = int(np.sum((rh != mh) | (rl != ml)))
        dmax = float(np.abs((rh - mh) + (rl - ml)).max())
        imx = float(np.abs(ih).max() + np.abs(il).max())
        rows.append(dict(lam=str(lam), entries=int(mh.size), differing=ndiff, max_abs_diff=dmax, max_imag=imx, sec=t()))
        log(f"  1. lambda {lam}: {ndiff} of {mh.size} entries differ from hpkernel's (max {dmax:.1e}); max |Im| {imx:.1e}")
        if lam == "ctl":
            if ndiff == 0:
                bad.append("control 1 passed (lambda one ulp off gave the same matrix)")
        elif ndiff or imx:
            bad.append(f"check 1 failed at lambda {lam}")
    rec["port"] = rows

    # 2. regression against the old crate's cached evaluations
    if "regress" in opts:
        ev = [json.loads(line) for line in open(opts["regress"])]
        ev = [e for e in ev if e["im"] != 0.0][:2]
        reg = []
        for e in ev:
            la, arg, _, _ = st.det(e["re"], e["im"])
            dla, darg = la - e["lnabs"], math.remainder(arg - e["arg"], 2 * math.pi)
            reg.append(dict(re=e["re"], im=e["im"], lnabs_old=e["lnabs"], lnabs=la, d_lnabs=dla, d_arg=darg))
            log(f"  2. lambda {e['re']:.6f}{e['im']:+.6f}i: ln|det| {la:.12f} (old {e['lnabs']:.12f}, {dla:+.1e}), "
                f"arg {arg:+.12f} ({darg:+.1e})")
            if abs(dla) > 1e-9 or abs(darg) > 1e-9:
                bad.append("check 2 failed")
        rec["regression"] = reg

    # 3. the identity at lambda = 0, at a perturbed state
    rows_, cols_ = (np.asarray(v) for v in st.c.psi_index())
    rng = np.random.default_rng(3)
    z = np.asarray(zh) + np.asarray(zl)
    pert = 1e-6 * rng.standard_normal(len(z))
    pert[-1] = 0.0
    zph = np.ascontiguousarray(np.asarray(zh) + pert)
    zpl = np.ascontiguousarray(np.asarray(zl))
    fh, fl = (np.asarray(a) for a in st.s.residual(zph, zpl))
    if not np.all(np.isfinite(fh)):
        bad.append("check 3: the residual at the perturbed state is not finite")
    ident = {}
    for mode in (1, 2):
        vh, vl = np.ascontiguousarray(zph[cols_]), np.ascontiguousarray(zpl[cols_])
        reh, rel, imh, iml = (np.asarray(a) for a in st.c.test_matvec(zph, zpl, vh, vl, 0.0, 0.0, mode))
        d = (reh - fh[rows_]) + (rel - fl[rows_])
        ratio = float(np.abs(d).max() / np.abs(fh[rows_]).max())
        ident[mode] = dict(max_F_psi=float(np.abs(fh[rows_]).max()), max_diff=float(np.abs(d).max()), ratio=ratio,
                           max_imag=float(np.abs(imh).max()))
        log(f"  3. mode {mode}: max |M_psi(0) z_psi - F_psi| = {np.abs(d).max():.1e}, max |F_psi| "
            f"{np.abs(fh[rows_]).max():.1e}: ratio {ratio:.1e}")
    if not ident[1]["ratio"] <= 1e-20:
        bad.append("check 3 failed")
    if not ident[2]["ratio"] > 1e-8:
        bad.append("control 3 passed (a's tangent kept)")
    rec["identity"] = ident

    # 4. lambda = 0 at the state
    vh, vl = np.ascontiguousarray(np.asarray(zh)[cols_]), np.ascontiguousarray(np.asarray(zl)[cols_])
    reh, rel, _, _ = (np.asarray(a) for a in st.c.test_matvec(zh, zl, vh, vl, 0.0, 0.0, 1))
    nullity = float(np.abs(reh + rel).max() / np.abs(vh).max())
    dets = {lam: st.c.test_det(zh, zl, lam, 0.0) for lam in (-0.1, 0.0, 0.1)}
    rec["zero"] = dict(null_residual=nullity, dets={str(k): v for k, v in dets.items()})
    log(f"  4. |M_psi(0) z_psi| / |z_psi| = {nullity:.1e} (|F| {st.F:.1e}); " + ", ".join(
        f"lambda {k:+.1f}: ln|det| {v[0]:.3f}, sign {'+' if abs(v[1]) < 1 else '-'}, min |u_kk| {v[2]:.1e}"
        for k, v in dets.items()))
    if nullity > 1e3 * max(st.F, 1e-30):
        bad.append("check 4 failed (psi = the background's p, m is not a null vector)")

    # 5. Cauchy-Riemann
    lam, h = 1.3, 1e-5
    l0, a0, _, _ = st.c.test_det(zh, zl, lam, 0.0)
    lx, ax, _, _ = st.c.test_det(zh, zl, lam + h, 0.0)
    ly, ay, _, _ = st.c.test_det(zh, zl, lam, h)
    lyc, ayc, _, _ = st.c.test_det(zh, zl, lam, -h)
    dx = complex(lx - l0, math.remainder(ax - a0, 2 * math.pi)) / h
    dy = complex(ly - l0, math.remainder(ay - a0, 2 * math.pi)) / (1j * h)
    dyc = complex(lyc - l0, math.remainder(ayc - a0, 2 * math.pi)) / (1j * h)
    cr, crc = abs(dx - dy) / abs(dx), abs(dx - dyc) / abs(dx)
    rec["cauchy_riemann"] = dict(lam=lam, h=h, along_h=[dx.real, dx.imag], along_ih=[dy.real, dy.imag], rel_diff=cr,
                                 control_rel_diff=crc)
    log(f"  5. d ln det M_psi / d lam at {lam}: along h {dx:.8f}, along i h {dy:.8f} (rel. diff {cr:.1e}); control "
        f"(conj lambda) {dyc:.8f} (rel. diff {crc:.1e})")
    if cr > 1e-4:
        bad.append("check 5 failed")
    if crc < 0.5:
        bad.append("control 5 passed")

    rec["fails"] = bad
    with open(out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
    log("PASS" if not bad else "FAIL: " + "; ".join(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
