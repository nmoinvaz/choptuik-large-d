"""Shared set-up for the mode count (notes/MODE_COUNT.md): a saved double-double state, hpkernel's Solver for it (the
layout gamma_dd.py uses: kfreeze-13 from D = 35 on, polished by Newton when its residual is above 1e-20), and the
complex-lambda determinant of hp/modecount_cx (built by hp/modecount_build.sh) on the same layout and segment plan.

    from modecount_common import setup
    st = setup("hp/data/states/up6_D30.0688.npz")       # st.det(lr, li) -> (ln|det|, arg det, pmin, pmax)

Also: the exactly known structure of det M(lambda) in this formulation (see MODE_COUNT.md):
  * lam_g = 1 + <c1>, c1 = (n-1)(1 - a_h^2): a residual gauge mode (time reparametrisation that keeps c(tau, 1) = 1),
    a zero of det M at lam_g + 2 i k omega for every integer k (the folded layout's period);
  * poles at the eigenvalues of -(omega D + diag kappa_kk), kappa_kk = (c1 + 1)(kk + 1) - n/2, kk = 0..ks (the horizon
    series' periodic solves): Re = n/2 - lam_g (kk + 1) to collocation accuracy, Im = odd multiples of omega.

The centre series (x = 0 to the first node) and the horizon series (u = 1 - x = 0 to the last node) take one
d/dtau per order, and their radius of convergence falls as M rises (#141). At D = 30 without the cut the centre
series' radius is 0.43, 0.32, 0.28 and 0.23 at M = 32, 48, 64 and 80, so at M >= 64 it diverges at the first node
0.3 and |F| is the truncated divergent sum (3.4e5 at kc 414). A resampled layout therefore gets a node inside the
centre series' radius and a higher horizon order ks where its series converge too slowly (fit_nodes), and setup
refuses a layout whose series do not converge to SERIES_TAIL (series_gate), at the start and at the solution. A start
can pass while its solution does not (#265). At D = 45, M = 48, kf 21 the start has 7.4e-74 and the solution 2.4e-8.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "modecount_cx", "build"))
from hpkernel.driver import load, save, make_solver, newton_solve, dmat_odd, phase_weights, plow_odd, dd_from_mp, resample_z, lm_solve  # noqa: E402
import mpmath as mp  # noqa: E402

# Largest truncation estimate of either series accepted, relative to its largest term (series_tail). It is the residual
# a count accepts (--fmax 1e-15), so the layouts of the counts so far are kept. D = 30 at M = 48 without the cut, the
# closest, has 4.8e-16 at the first node.
SERIES_TAIL = 1e-15
# fit_nodes adds at most this many inner nodes, and raises ks by KS_FACTOR at most this many times
NODES_ADDED_MAX = 3
# The horizon side gets a higher ks, not a node. A node at 0.85 at D = 61, M = 64 leaves det M nearly singular (pivots
# 1e-23 to 2e3, the two LUs 2 apart in ln|det|, no sign change at lam_phys), while the same state without it gives
# pivots 3e-14 to 5e6 and the sign change. The mass mode, x^-(n-1), shrinks by (0.7/0.85)^(n-1) = 1e-5 over that
# segment at every tau point.
KS_FACTOR = 1.5
# where setup keeps the re-converged resampled states, modecount_state_D<D>_M<M>[_kf<kf>].npz
DATA = os.path.join(HERE, "data")


class Setup:
    pass


def build_solver(meta, zh, zl, M, kf, threads=1):
    """hpkernel Solver on the layout of meta (nodes, kt, kc, ks), with the Taylor segments planned for z."""
    s = make_solver(meta["D"], M, nodes=meta["nodes"], kt=meta.get("kt", 36), kc=meta.get("kc", 420),
                    ks=meta.get("ks", 100), threads=threads, kfreeze=kf)
    s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
    return s


def build_pair(meta, zh, zl, M, kf, threads=1):
    """hpkernel Solver (planned segments) and the complex-lambda CSolver on the same layout."""
    import modecount_cx
    D = meta["D"]
    s = build_solver(meta, zh, zl, M, kf, threads)
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


def resample_state(zh, zl, meta, M, kf):
    """The state resampled to M tau points (lo parts dropped), with p1 projected onto the cut when kf is set."""
    K = len(meta["nodes"])
    zh2, zl2 = resample_z(np.asarray(zh) + np.asarray(zl), meta["M"], M, K)
    if kf:
        a_, b_ = plow_odd(M, kf)
        zh2[:M] = (a_ + b_) @ zh2[:M]
    return zh2, zl2, dict(meta, M=M)


def _series(s, zh, zl, which):
    """scaled coefficients (orders, 4, M) of the centre (which = 0) or horizon (1) series, and the scale sigma"""
    h, l = s.series(zh, zl, which, True)
    return (np.asarray(h) + np.asarray(l)).reshape(-1, 4, s.m), s.series_scale(which)


def series_tail(s, zh, zl, meta, which, n=24):
    """Truncation estimate of the centre series at the first node (which = 0) or of the horizon series at the last
    (which = 1), relative to the series' largest term. With E_k the largest term of orders k - 1 and k (over the four
    fields and the tau points; p and m are odd in x, a and c even) and r = (E_last / E_{last-n})^(1/n), the estimate
    is E_last r / (1 - r), inf for r >= 1 (the series diverges there). Returns dict(tail, ratio, last, x, radius)."""
    C, sig = _series(s, zh, zl, which)
    x = meta["nodes"][0] if which == 0 else 1.0 - meta["nodes"][-1]
    T = np.abs(C).max(axis=(1, 2)) * (x / sig) ** np.arange(C.shape[0])
    E = np.maximum(T[1:], T[:-1])
    n = min(n, len(E) - 1)
    top = float(T.max())
    r = float((E[-1] / E[-1 - n]) ** (1.0 / n)) if E[-1 - n] > 0 else 0.0
    if not np.isfinite(top) or not np.isfinite(r) or r >= 1:
        tail = float("inf")
    else:
        tail = float(E[-1]) * r / (1 - r) / top
    return dict(tail=tail, ratio=r, last=float(E[-1]) / top, x=x, radius=x / r if r > 0 else float("inf"))


def _series_at(s, zh, zl, which, t):
    """the node state (4M, f64) of the centre (which = 0) series at x = t or the horizon (1) series at u = t"""
    C, sig = _series(s, zh, zl, which)
    v = np.zeros(C.shape[1:])
    for k in range(C.shape[0] - 1, -1, -1):
        v = v * (t / sig) + C[k]
    return v.ravel()


def add_node(s, zh, zl, meta, which):
    """z and meta with a node added at half the first node (which = 0) or at half the last node's distance to the
    horizon (1), its state from the series of the layout of s"""
    M, nodes = meta["M"], list(meta["nodes"])
    K = len(nodes)
    zh = np.asarray(zh) + np.asarray(zl)
    if which == 0:
        x = nodes[0] / 2
        v = _series_at(s, zh, np.zeros_like(zh), 0, x)
        zn = np.concatenate([zh[:2 * M], v, zh[2 * M:]])
        nodes = [x] + nodes
    else:
        x = 1.0 - (1.0 - nodes[-1]) / 2
        v = _series_at(s, zh, np.zeros_like(zh), 1, 1.0 - x)
        o = 2 * M + 4 * M * K
        zn = np.concatenate([zh[:o], v, zh[o:]])
        nodes = nodes + [x]
    added = list(meta.get("nodes_added", [])) + [x]
    return zn, np.zeros_like(zn), dict(meta, nodes=nodes, nodes_added=added)


def fit_nodes(meta, zh, zl, kf, threads=1, log=print, tol=SERIES_TAIL, nmax=NODES_ADDED_MAX):
    """While the centre series' truncation estimate exceeds tol, add a node at half the first node (at most nmax); while
    the horizon series' does and it converges (ratio < 1), raise ks by KS_FACTOR (at most nmax times). A divergent
    horizon series is left to setup's gate (see KS_FACTOR). Returns zh, zl, meta and the last tails (centre,
    horizon)."""
    M = meta["M"]
    added = [0, 0]
    while True:
        s = build_solver(meta, zh, zl, M, kf, threads)
        tails = [series_tail(s, zh, zl, meta, w) for w in (0, 1)]
        which = next((w for w in (0, 1) if tails[w]["tail"] > tol and added[w] < nmax
                      and (w == 0 or tails[w]["ratio"] < 1)), None)
        if which is None:
            return zh, zl, meta, tails
        t = tails[which]
        msg = (f"  {('centre', 'horizon')[which]} series at {('x', 'u')[which]} = {t['x']:g}: truncation "
               f"{t['tail']:.1e}, ratio {t['ratio']:.3f} per order (radius {t['radius']:.3f})")
        if which == 0:
            zh, zl, meta = add_node(s, zh, zl, meta, 0)
            log(f"{msg}; nodes {meta['nodes']}")
        else:
            ks = int(np.ceil(meta.get("ks", 100) * KS_FACTOR))
            meta = dict(meta, ks=ks, ks_from=meta.get("ks_from", meta.get("ks", 100)))
            log(f"{msg}; ks {ks}")
        added[which] += 1


def tails_text(tails):
    return ", ".join(f"{n} {t['tail']:.1e} (ratio {t['ratio']:.3f} at {t['x']:g})"
                     for n, t in zip(("centre", "horizon"), tails))


def series_gate(s, zh, zl, meta, when, log=None):
    """Both series' truncation (series_tail) on the layout of s, logged if log is given. RuntimeError if either exceeds
    SERIES_TAIL. when names the state checked, 'start' before the solve or 'solution' after it. Returns the tails."""
    tails = [series_tail(s, zh, zl, meta, w) for w in (0, 1)]
    if log:
        log(f"  series truncation ({when}): {tails_text(tails)}")
    if max(t["tail"] for t in tails) > SERIES_TAIL:
        # a layout whose series do not converge at its end nodes does not represent the equations (#141: the centre
        # series at x = 0.3 diverges at M = 64 and |F| is its truncated sum)
        raise RuntimeError(f"series not converged in this layout ({when}; M = {meta['M']}, nodes {meta['nodes']}): "
                           f"{tails_text(tails)}; more than {SERIES_TAIL:.0e}")
    return tails


def prepare(path, M=None, threads=1, kf="default", log=print, fit=True):
    """The state setup starts from. It is the state as loaded, or resampled to M tau points and fitted (fit_nodes), or
    the re-converged state an earlier setup saved in DATA. Returns zh, zl, meta, kf and the path setup saves to (None
    at the state's own M)."""
    zh, zl, meta = load(path)
    D = meta["D"]
    if kf == "default":
        kf = 13 if D >= 35 else None
    M0 = meta["M"]
    saved = None
    if M is not None and M != M0:
        # a state resampled to M tau points is re-converged once and kept in DATA (hp/data)
        saved = os.path.join(DATA, f"modecount_state_D{D:.4f}_M{M}" + (f"_kf{kf}" if kf else "") + ".npz")
        if os.path.exists(saved):
            zh, zl, meta = load(saved)
            log(f"  resampled state from {os.path.basename(saved)}")
        else:
            zh, zl, meta = resample_state(zh, zl, meta, M, kf)
            if fit:
                zh, zl, meta, _ = fit_nodes(meta, zh, zl, kf, threads, log)
    return zh, zl, meta, kf, saved


def setup(path, M=None, threads=1, kf="default", fmax=1e-20, log=print, fit=True):
    """Load a state, rebuild the layout (optionally resampled to M tau points, fitted where a series does not converge
    (fit_nodes), and re-converged), polish. fit=False keeps the state's nodes and ks (the control of fit_nodes). Both
    series must pass SERIES_TAIL on the start and on the solution (series_gate)."""
    zh, zl, meta, kf, saved = prepare(path, M, threads, kf, log, fit)
    D = meta["D"]
    M = meta["M"]
    s, c = build_pair(meta, zh, zl, M, kf, threads)
    series_gate(s, zh, zl, meta, "start")
    F = resid(s, zh, zl)
    F_in = F
    if F > fmax:
        log(f"  |F| {F:.1e} in this layout: Newton")
        zh, zl, hist = newton_solve(s, zh, zl, tol=min(fmax, 1e-24), it_max=12, verbose=False)
        F = resid(s, zh, zl)
        if F > fmax:
            log(f"  |F| {F:.1e} after Newton: Levenberg-Marquardt")
            zh, zl, hist = lm_solve(s, zh, zl, tol=min(fmax, 1e-27), it_max=20, verbose=False, newton_after=1e-22)
            F = resid(s, zh, zl)
        if saved:
            save(saved, zh, zl, dict(meta, F=F))
            log(f"  |F| {F:.1e}; saved {os.path.basename(saved)}")
    if F > fmax:
        # a count on a background that does not solve the equations of this layout means nothing (the M = 16
        # resampling at D = 61 stalled at |F| = 1.3e-9 and gave a spurious real sign change)
        raise RuntimeError(f"background not converged in this layout: |F| = {F:.1e} > {fmax:.0e} (M = {M})")
    # the solution passes the gate too. A start can pass while its solution does not (#254, D 45 kf 21: the resampled
    # start 7.4e-74, the solution 2.4e-8).
    tails = series_gate(s, zh, zl, meta, "solution", log)
    st = Setup()
    st.path, st.meta, st.D, st.M, st.kf = path, meta, D, M, kf
    st.zh, st.zl, st.s, st.c, st.F, st.F_in = np.ascontiguousarray(zh), np.ascontiguousarray(zl), s, c, F, F_in
    st.tails = tails
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
