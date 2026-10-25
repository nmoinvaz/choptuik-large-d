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
   Checked at a perturbed state: unknown j (all but Delta) moved by 1e-8 xi_j / max_i |J_ij| (xi standard normal, J
   the mode matrix at lambda = 0), so that every row of F moves by about 1e-8 whatever the horizon amplifies.
   Pass: max |M_psi z_psi - F_psi| <= 10 nu, where nu = max(the noise of M_psi z_psi - F_psi, eps_dd max |F_psi|) and
   the noise is that difference's largest change when z moves by 1e-31 relative (two random moves): in exact
   arithmetic it does not change, so its change is the two computations' rounding.
   Control: mode 2 (a's tangent kept, the coupling p -> a -> p put back) must miss by more than 1e3 times the bound.
4. Test field at the state itself (lambda = 0 is a zero): max |M_psi(0) z_psi| <= 10 max(nu_0, max |F_psi|), nu_0 the
   noise of check 3 measured at the state. Control: (p, -m) in place of (p, m) must give more than 1e3 times the
   bound. Logged with it: ln|det| and min |u_kk| at lambda = 0 against lambda = +-0.1.
   (Checks 3 and 4 were changed to these yardsticks after D = 45, 4 and 67 failed the earlier ones; see
   notes/GHOST_ADMIXTURE.md.)
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
from multidouble import dd_add  # noqa: E402


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
        if lam == 0.0:
            jac0 = mh + ml
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
    eps_dd = 2.0 ** -104

    def ident_diff(zh_, zl_, mode, vsign=None):
        """M_psi(0; z) z_psi - F_psi(z), and F_psi (z_psi's m part times vsign if given)"""
        zh_, zl_ = np.ascontiguousarray(zh_), np.ascontiguousarray(zl_)
        vh, vl = np.ascontiguousarray(zh_[cols_]), np.ascontiguousarray(zl_[cols_])
        if vsign is not None:
            vh, vl = np.ascontiguousarray(vh * vsign), np.ascontiguousarray(vl * vsign)
        reh, rel, imh, _ = (np.asarray(a) for a in st.c.test_matvec(zh_, zl_, vh, vl, 0.0, 0.0, mode))
        fh, fl = (np.asarray(a) for a in st.s.residual(zh_, zl_))
        return (reh - fh[rows_]) + (rel - fl[rows_]), fh[rows_] + fl[rows_], reh + rel, float(np.abs(imh).max())

    def noise(zh_, zl_):
        """largest change of M_psi z_psi - F_psi under two random moves of z by 1e-31 relative (Delta kept)"""
        d0 = ident_diff(zh_, zl_, 1)[0]
        g = np.random.default_rng(11)
        out = 0.0
        for _ in range(2):
            mv = 1e-31 * np.abs(zh_) * g.standard_normal(len(zh_))
            mv[-1] = 0.0
            th, tl = dd_add(np.asarray(zh_), np.asarray(zl_), mv, np.zeros_like(mv))
            out = max(out, float(np.abs(ident_diff(th, tl, 1)[0] - d0).max()))
        return out

    colmax = np.abs(jac0).max(axis=0)
    rng = np.random.default_rng(3)
    pert = np.zeros(len(zh))
    nz = colmax > 0
    pert[:len(colmax)][nz] = 1e-8 * rng.standard_normal(int(nz.sum())) / colmax[nz]
    zph, zpl = (np.ascontiguousarray(a) for a in dd_add(np.asarray(zh), np.asarray(zl), pert, np.zeros_like(pert)))
    ident = {}
    d1, fpsi, _, im1 = ident_diff(zph, zpl, 1)
    d2, _, _, im2 = ident_diff(zph, zpl, 2)
    if not np.all(np.isfinite(fpsi)):
        bad.append("check 3: the residual at the perturbed state is not finite")
    nu_p = max(noise(zph, zpl), eps_dd * float(np.abs(fpsi).max()))
    bound3 = 10 * nu_p
    for mode, d, im in ((1, d1, im1), (2, d2, im2)):
        ident[mode] = dict(max_F_psi=float(np.abs(fpsi).max()), max_diff=float(np.abs(d).max()), max_imag=im)
        log(f"  3. mode {mode}: max |M_psi(0) z_psi - F_psi| = {np.abs(d).max():.1e}, max |F_psi| "
            f"{np.abs(fpsi).max():.1e}, noise {nu_p:.1e}, bound {bound3:.1e}")
    rec["identity"] = dict(ident, noise=nu_p, bound=bound3, pert_scale=1e-8)
    if not ident[1]["max_diff"] <= bound3:
        bad.append("check 3 failed")
    if not ident[2]["max_diff"] > 1e3 * bound3:
        bad.append("control 3 passed (a's tangent kept)")

    # 4. lambda = 0 at the state
    d0, f0, mz, _ = ident_diff(zh, zl, 1)
    nu_0 = max(noise(zh, zl), eps_dd * float(np.abs(f0).max()))
    null = float(np.abs(mz).max())
    bound4 = 10 * max(nu_0, float(np.abs(f0).max()))
    m = st.M
    # z_psi in psi_index order: p1 (M), each node's p and m (M each), m_h (M); the control flips every m
    sgn = np.ones(len(cols_))
    for i_ in range(st.K):
        sgn[m + 2 * m * i_ + m:m + 2 * m * (i_ + 1)] = -1.0
    sgn[-m:] = -1.0
    ctl = float(np.abs(ident_diff(zh, zl, 1, vsign=sgn)[2]).max())
    dets = {lam: st.c.test_det(zh, zl, lam, 0.0) for lam in (-0.1, 0.0, 0.1)}
    rec["zero"] = dict(null_residual=null, F_psi=float(np.abs(f0).max()), noise=nu_0, bound=bound4,
                       control=ctl, dets={str(k): v for k, v in dets.items()})
    log(f"  4. max |M_psi(0) z_psi| = {null:.1e} (max |F_psi| {np.abs(f0).max():.1e}, noise {nu_0:.1e}, bound "
        f"{bound4:.1e}); control (p, -m): {ctl:.1e}; " + ", ".join(
            f"lambda {k:+.1f}: ln|det| {v[0]:.3f}, sign {'+' if abs(v[1]) < 1 else '-'}, min |u_kk| {v[2]:.1e}"
            for k, v in dets.items()))
    if null > bound4:
        bad.append("check 4 failed (psi = the background's p, m is not a null vector)")
    if not ctl > 1e3 * bound4:
        bad.append("control 4 passed ((p, -m) taken for a null vector)")

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
