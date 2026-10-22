"""Shared set-up for the mode count (notes/MODE_COUNT.md): a saved double-double state, hpkernel's Solver for it (the
layout gamma_dd.py uses: kfreeze-13 from D = 35 on, polished by Newton when its residual is above 1e-20), and the
complex-lambda determinant of hp/modecount_cx (built by hp/modecount_build.sh) on the same layout and segment plan.

    from modecount_common import setup
    st = setup("hp/data/states/up6_D30.0688.npz")       # st.det(lr, li) -> (ln|det|, arg det, pmin, pmax)

Also: the exactly known structure of det M(lambda) in this formulation (see MODE_COUNT.md):
  * lam_g = 1 + <c1>, c1 = (n-1)(1 - a_h^2): a residual gauge mode (time reparametrisation that keeps c(tau, 1) = 1),
    a zero of det M at lam_g + i k omega for every integer k;
  * poles at the eigenvalues of -(omega D + diag kappa_kk), kappa_kk = (c1 + 1)(kk + 1) - n/2, kk = 0..ks (the horizon
    series' periodic solves): Re = n/2 - lam_g (kk + 1) to collocation accuracy, Im = odd multiples of omega.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "modecount_cx", "build"))
from hpkernel.driver import load, make_solver, newton_solve, dmat_odd, phase_weights, plow_odd, dd_from_mp, resample_z, lm_solve  # noqa: E402
import mpmath as mp  # noqa: E402
import modecount_cx  # noqa: E402


class Setup:
    pass


def build_pair(meta, zh, zl, M, kf, threads=1):
    """hpkernel Solver (planned segments) and the complex-lambda CSolver on the same layout."""
    D = meta["D"]
    s = make_solver(D, M, nodes=meta["nodes"], kt=meta.get("kt", 36), kc=meta.get("kc", 420), ks=meta.get("ks", 100),
                    threads=threads, kfreeze=kf)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    with mp.workdps(40):
        nh, nl = dd_from_mp([mp.mpf(D) - 2])
    dh, dl = dmat_odd(M)
    ph, pl = phase_weights(M)
    a = b = None
    if kf:
        a, b = plow_odd(M, kf)
    c = modecount_cx.CSolver(M, float(nh[0]), float(nl[0]), dh, dl, ph, pl, list(meta["nodes"]), s.get_segs(),
                             meta.get("kt", 36), meta.get("kc", 420), meta.get("ks", 100), threads, a, b, None)
    return s, c


def resid(s, zh, zl):
    fh, fl = s.residual(zh, zl)
    return float(np.abs(np.asarray(fh) + np.asarray(fl)).max())


def setup(path, M=None, threads=1, kf="default", fmax=1e-20, log=print):
    """Load a state, rebuild the layout (optionally resampled to M tau points and re-converged), polish."""
    zh, zl, meta = load(path)
    D = meta["D"]
    if kf == "default":
        kf = 13 if D >= 35 else None
    M0 = meta["M"]
    if M is not None and M != M0:
        K = len(meta["nodes"])
        zh2, zl2 = resample_z(np.asarray(zh) + np.asarray(zl), M0, M, K)
        if kf:
            a_, b_ = plow_odd(M, kf)
            zh2[:M] = (a_ + b_) @ zh2[:M]
        zh, zl = zh2, zl2
        meta = dict(meta, M=M)
    M = meta["M"]
    s, c = build_pair(meta, zh, zl, M, kf, threads)
    F = resid(s, zh, zl)
    F_in = F
    if F > fmax:
        log(f"  |F| {F:.1e} in this layout: Newton")
        zh, zl, hist = newton_solve(s, zh, zl, tol=1e-24, it_max=6, verbose=False)
        F = resid(s, zh, zl)
        if F > fmax:
            log(f"  |F| {F:.1e} after Newton: Levenberg-Marquardt")
            zh, zl, hist = lm_solve(s, zh, zl, tol=1e-27, it_max=20, verbose=False, newton_after=1e-22)
            F = resid(s, zh, zl)
    st = Setup()
    st.path, st.meta, st.D, st.M, st.kf = path, meta, D, M, kf
    st.zh, st.zl, st.s, st.c, st.F, st.F_in = np.ascontiguousarray(zh), np.ascontiguousarray(zl), s, c, F, F_in
    n = D - 2
    K = len(meta["nodes"])
    st.n, st.K = n, K
    st.Delta = float(zh[-1] + zl[-1])
    st.omega = 2 * np.pi / st.Delta
    ah = np.asarray(zh[3 * M + 4 * M * K: 4 * M + 4 * M * K]) + np.asarray(zl[3 * M + 4 * M * K: 4 * M + 4 * M * K])
    st.a_h = ah
    st.c1 = (n - 1) * (1 - ah ** 2)
    st.lam_g = 1 + float(np.mean(st.c1))
    st.ks = meta.get("ks", 100)

    def det(lr, li):
        return st.c.mode_det(st.zh, st.zl, float(lr), float(li))

    def det_real(lam):
        return st.s.mode_det(st.zh, st.zl, float(lam), 0.0)

    st.det, st.det_real = det, det_real
    return st


def horizon_poles(st, kk_max=None):
    """Eigenvalues lam of -(omega D + diag kappa_kk) for kk = 0..ks: where the horizon series' tangent solves are
    singular, i.e. the only possible poles of det M(lambda). Returns a list of (kk, lam)."""
    M = st.M
    dh, dl = dmat_odd(M)
    Dm = (np.asarray(dh) + np.asarray(dl)) * st.omega
    out = []
    for kk in range(0, (st.ks if kk_max is None else kk_max) + 1):
        kap = (st.c1 + 1) * (kk + 1) - st.n / 2
        ev = np.linalg.eigvals(Dm + np.diag(kap))
        for e in ev:
            out.append((kk, -complex(e)))
    return out


def timer():
    t0 = time.time()
    return lambda: time.time() - t0
