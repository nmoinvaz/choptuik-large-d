"""Driver of the sigma-model kernel (sigmams, matter/sigma_ms): set-up, seeds from sigdss states, Newton in double- and
quad-double, a deflated Newton for starts on the near-null band (newton_band), continuation in eps = 1/eta,
diagnostics, and the growth rates of perturbations from the mode matrix's determinant (det_fn, det_root).

Double-double vectors are (hi, lo) pairs of float64 arrays, quad-double ones four parts. Layout of z (M half-grid
points, K nodes; multishoot's, NF = 5, NL = 2, NR = 3):

    q, c0 | node_0 (p, m, a, c, psi) | ... | node_{K-1} | m_h, a_h, psi_h | Delta          length 5M(K+1) + 1

The kernel is imported from the site directory of matter/sigma_ms/build.sh (PYTHONPATH). Copied and adapted from
hp/python/hpkernel/driver.py (dmat_odd, phase_weights, resample, newton_solve), which is not imported.
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

from multidouble import dd_add, dd_from_mp, dd_matvec, dd_norm, qd_axpy  # noqa: F401
from sigmams import Solver

HERE = os.path.dirname(os.path.abspath(__file__))
MATTER = os.path.normpath(os.path.join(HERE, "..", "..", "solver", "matter"))
# the nodes: dense near the centre, where c(0) falls to 0.07 and the centre datum's harmonics decay as e^(-0.35 k),
# the first node close enough for the centre series (k omega x0 / c0 of order 10 at M = 48), the last at u0 = 0.02
NODES = [0.001, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
         0.9, 0.95, 0.98]
PLAN_TOL = 1e-32
PLAN_HMAX = 0.05


# ------------------------------------------------------------------ grid
def dmat_odd(M, dps=40):
    """Folded d/dtheta for odd fields on the half grid of N = 2M points, in double-double (hi, lo), row-major (as
    hp's driver)."""
    with mp.workdps(dps):
        N = 2 * M
        out = []
        for i in range(M):
            for j in range(M):
                v = mp.mpf(0)
                for jj, sgn in ((j, 1), (j + M, -1)):
                    if jj != i:
                        d = (i - jj) * mp.pi / N
                        v += sgn * mp.mpf(0.5) * (-1) ** (i - jj) * mp.cot(d)
                out.append(v)
        hi, lo = dd_from_mp(out)
    return hi.reshape(M, M), lo.reshape(M, M)


def phase_weights(M, dps=40):
    """sin(theta_j) on the half grid: the phase row picks the sin(theta) harmonic of q (sigdss's Im q_1 = 0)"""
    with mp.workdps(dps):
        return dd_from_mp([mp.sin(mp.pi * j / M) for j in range(M)])


def theta(M):
    return np.pi * np.arange(M) / M


def dd_of(x, dps=40):
    """an exact or mpmath number as a double-double (hi, lo)"""
    with mp.workdps(dps):
        h, l = dd_from_mp([mp.mpf(x)])
    return float(h[0]), float(l[0])


def k2_of(eta, D=4, dps=40):
    """kappa^2 = n / eta (0 at eta = inf) as a double-double"""
    if eta is None or eta == float("inf") or eta == "inf":
        return 0.0, 0.0
    with mp.workdps(dps):
        return dd_of((mp.mpf(D) - 2) / mp.mpf(eta), dps)


def make_solver(eta, M, nodes=None, kt=36, kc=80, ks=80, threads=0, D=4, segs=None, match_node=None):
    """A Solver at coupling eta (None or inf: eta = infinity, the l = 1 multiplet) in D dimensions"""
    nodes = list(NODES if nodes is None else nodes)
    nh, nl = dd_of(mp.mpf(D) - 2)
    kh, kl = k2_of(eta, D)
    dh, dl = dmat_odd(M)
    ph, pl = phase_weights(M)
    return Solver(M, nh, nl, kh, kl, dh, dl, ph, pl, nodes, segs or [], kt, kc, ks, threads, match_node)


def sizes(M, K):
    return 5 * M * (K + 1) + 1


# ------------------------------------------------------------------ seeds from sigdss (solver/matter/sigdss_solver.py)
def resample(F, M):
    """Full-grid samples F (any N0) -> half-grid samples of the N = 2M grid (trigonometric interpolation, as hp's)"""
    N0 = len(F)
    c = np.fft.rfft(F) / N0
    th = 2 * np.pi * np.arange(M) / (2 * M)
    k = np.arange(len(c))
    w = np.where((k == 0) | ((N0 % 2 == 0) & (k == N0 // 2)), 1.0, 2.0)
    return (w * c[None, :] * np.exp(1j * k[None, :] * th[:, None])).real.sum(1)


def sigdss_states(path, eps, kc, xs, D=4, kappa=0.5):
    """sigdss's solution (f64, Fourier to harmonic kc, 12 nodes 0.02..0.9) at the points xs: the state (5, N) at each,
    from its centre series and log-RK4 below its first node, its RK4 between nodes, and its horizon series at u0 = 0.01
    with RK4 inward above its last node. Returns (S, z, states, (q, c0), (mS, aS, qS), Delta)."""
    if MATTER not in sys.path:
        sys.path.insert(0, MATTER)
    from sigdss_solver import SigDSS, NODES_STD
    z = np.load(path)
    S = SigDSS(kc=kc, nodes=NODES_STD, n=D - 2, eps=eps, Delta_ref=z[-1])
    S.freeze_from(z, kappa)
    f = S.fields(z)
    om = np.array([2 * np.pi / f["Delta"]])
    nodes = list(S.nodes)
    st = [s[:, None, :] for s in f["states"]]           # (5, 1, N)
    d, c0 = f["d"][None, :], f["c0"][None, :]
    out = []
    for x in xs:
        if x <= nodes[0]:
            Ser = S.centre_series(d, c0, om)
            s = S.trunc(np.stack([S.eval_series(s_, S.x0) for s_ in Ser]))
            if x > S.x0:
                steps = max(20, int(np.ceil(S.steps_map["log"] * np.log(x / S.x0) / np.log(nodes[0] / S.x0))))
                h = (np.log(x) - np.log(S.x0)) / steps
                y = np.log(S.x0)
                fn = lambda y, s: np.exp(y) * S.rhs(np.exp(y), s, om)
                for _ in range(steps):
                    k1 = fn(y, s); k2 = fn(y + h / 2, s + h / 2 * k1); k3 = fn(y + h / 2, s + h / 2 * k2); k4 = fn(y + h, s + h * k3)
                    s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); y += h
        elif x >= 1 - S.u0:
            Ser = S.ssh_series(f["mS"][None, :], f["aS"][None, :], f["qS"][None, :], om)
            s = S.trunc(np.stack([S.eval_series(s_, S.u0) for s_ in Ser]))
            if x < 1 - S.u0:
                s = S.integrate_u(s, 1 - S.u0, x, om)
        elif x > nodes[-1]:
            Ser = S.ssh_series(f["mS"][None, :], f["aS"][None, :], f["qS"][None, :], om)
            s = S.trunc(np.stack([S.eval_series(s_, S.u0) for s_ in Ser]))
            s = S.integrate_u(s, 1 - S.u0, x, om)
        else:
            i = max(j for j in range(len(nodes)) if nodes[j] <= x)
            s = st[i]
            if x > nodes[i]:
                if nodes[i] >= 0.4:
                    s = S.integrate_u(s, nodes[i], x, om)
                else:
                    s = S.integrate(s, nodes[i], x, S.nseg(nodes[i], x), om)
        out.append(s[:, 0, :])
    return S, z, out, (f["d"], f["c0"]), (f["mS"], f["aS"], f["qS"]), f["Delta"]


def seed_from_sigdss(path, eps, kc, M, nodes=None, D=4):
    """z (hi, lo = 0) on the M-point half grid and the nodes from a sigdss state (Fourier to harmonic kc)"""
    nodes = list(NODES if nodes is None else nodes)
    S, zs, states, (q, c0), (mS, aS, qS), Delta = sigdss_states(path, eps, kc, nodes, D)
    blocks = [resample(q, M), resample(c0, M)]
    for s in states:
        blocks += [resample(s[i], M) for i in range(5)]
    blocks += [resample(mS, M), resample(aS, M), resample(qS, M), np.array([Delta])]
    zh = np.concatenate(blocks)
    return zh, np.zeros_like(zh)


def resample_z(zh, M_old, M_new, K):
    """A state's every field block from M_old to M_new half-grid points (trigonometric interpolation in f64, the
    parity of each block kept: the half grid of an odd field is extended by f(theta + pi) = -f(theta)); lo dropped"""
    pars = ["odd", "even"] + ["odd", "odd", "even", "even", "odd"] * K + ["odd", "even", "odd"]
    out = []
    for b, par in enumerate(pars):
        h = zh[b * M_old:(b + 1) * M_old]
        full = np.concatenate([h, -h]) if par == "odd" else np.concatenate([h, h])
        out.append(resample(full, M_new))
    out.append(zh[-1:])
    return np.concatenate(out)


# ------------------------------------------------------------------ Newton
def fnorm(F):
    """max |F| of a residual given as parts"""
    return float(np.max(np.abs(sum(np.asarray(f) for f in F))))


def newton_solve(s, zh, zl, tol=1e-28, it_max=16, max_halvings=10, verbose=True, log=print, jac="dd", chord=0,
                 chord_rate=0.1, after_step=None):
    """Newton in double-double with backtracking on max |F| (halving); stops when max |F| < tol or a step finds no
    decrease. jac: "dd", the exact double-double Jacobian (Solver.newton); "mixed", the f64 Jacobian of Fun<4> lanes in
    staircase storage (Solver.newton_mixed: the same root, linear convergence), with an exact step instead when a
    mixed step finds no decrease. chord: after each Jacobian, up to this many chord steps with its factors
    (Solver.chord) while each lowers max |F| by at least chord_rate. With jac "mixed", once max |F| < tol the step
    computed there is still taken if max |F| stays below tol (the linear convergence leaves an error of about that
    step along the weak directions, which |F| does not show; hp's newton_solve). after_step(zh, zl), if given, is
    called after every accepted step (run_eta.py re-plans the Taylor steps there). Returns zh, zl, history."""
    hist = []
    t0 = time.time()

    def search(dz, nF):
        lam = 1.0
        for h in range(max_halvings + 1):
            th, tl = dd_add(zh, zl, dz[0] * lam, dz[1] * lam)
            n1 = fnorm(s.residual([th, tl]))
            if np.isfinite(n1) and n1 < nF:
                return True, lam, th, tl, n1
            lam *= 0.5
        return False, lam, th, tl, n1

    for it in range(it_max):
        kind = jac
        if kind == "dd":
            # the residual first: a converged state needs no Jacobian
            nF0 = fnorm(s.residual([zh, zl]))
            if nF0 < tol:
                hist.append(dict(it=it, jac=kind, F=nF0, Delta=float(zh[-1]), converged=True, t=round(time.time() - t0, 1)))
                if verbose:
                    log(f"  newton {it}: |F| {nF0:.2e} converged, Delta {zh[-1]:.15f}")
                return zh, zl, hist
        dz, F, pmin, pmax = s.newton([zh, zl]) if kind == "dd" else s.newton_mixed([zh, zl])
        nF = fnorm(F)
        rec = dict(it=it, jac=kind, F=nF, Delta=float(zh[-1]), piv=(pmin, pmax), t_step=getattr(s, "step_timing", lambda: None)(),
                   t=round(time.time() - t0, 1))
        if nF < tol:
            rec["converged"] = True
            if kind != "dd":
                th, tl = dd_add(zh, zl, dz[0], dz[1])
                n1 = fnorm(s.residual([th, tl]))
                if n1 < tol:
                    zh, zl = th, tl
                    rec.update(polished=True, F_polished=n1, dDelta=float(dz[0][-1] + dz[1][-1]))
            hist.append(rec)
            if verbose:
                log(f"  newton {it} ({kind}): |F| {nF:.2e} converged, Delta {zh[-1]:.15f} (step |dz| {dd_norm(*dz):.1e}"
                    f"{', taken' if rec.get('polished') else ''})")
            return zh, zl, hist
        ok, lam, th, tl, n1 = search(dz, nF)
        if not ok and kind != "dd":
            if verbose:
                log(f"  newton {it}: |F| {nF:.2e}: no decrease along the mixed step; retrying with the exact Jacobian")
            dz, F, pmin, pmax = s.newton([zh, zl])
            rec.update(jac="dd", retry_dd=True)
            ok, lam, th, tl, n1 = search(dz, nF)
        rec.update(lam=lam, F_next=n1, dz=dd_norm(*dz), dDelta=float(dz[0][-1] + dz[1][-1]) * lam)
        hist.append(rec)
        if verbose:
            log(f"  newton {it} ({rec['jac']}): |F| {nF:.2e} -> {n1:.2e}, lam {lam:g}, |dz| {rec['dz']:.1e}, "
                f"dDelta {rec['dDelta']:+.2e}, pivots {pmin:.1e}..{pmax:.1e} ({rec['t']} s)")
        if not ok:
            rec["failed"] = "line search"
            return zh, zl, hist
        zh, zl = th, tl
        for c in range(chord):
            if n1 < tol:
                break
            dzc, Fc = s.chord([zh, zl])
            ch, cl = dd_add(zh, zl, dzc[0], dzc[1])
            n2 = fnorm(s.residual([ch, cl]))
            crec = dict(it=it, chord=c, F=n1, F_next=n2, dDelta=float(dzc[0][-1] + dzc[1][-1]), t=round(time.time() - t0, 1))
            hist.append(crec)
            if verbose:
                log(f"    chord {c}: |F| {n1:.2e} -> {n2:.2e}, dDelta {crec['dDelta']:+.2e}")
            if not (np.isfinite(n2) and n2 < chord_rate * n1):
                break
            zh, zl, n1 = ch, cl, n2
        if after_step is not None:
            after_step(zh, zl)
    F = s.residual([zh, zl])
    nF = fnorm(F)
    hist.append(dict(it=it_max, F=nF, Delta=float(zh[-1]), t=round(time.time() - t0, 1), converged=nF < tol))
    return zh, zl, hist


def newton_band(s, zh, zl, tol=1e-28, it_max=40, band_tol=1e-6, radius=None, band=True, kappa=0.1, correctors=10,
                grow=3, max_halvings=30, verbose=True, log=print, after_step=None):
    """Deflated Newton in double-double, for starts on the near-null band of the Jacobian (SIGMA_DSS.md section 6).
    There Newton's step dz_N = -J^-1 F has components of 1e2 to 1e4 along the band, far beyond the reach of the linear
    model, and backtracking on |F| (newton_solve) scales the useful components down with them.

    Each iteration takes the exact step dz_N (Solver.newton) and an f64 SVD of the dense Jacobian. The band is the
    span V_w of the right singular vectors with sig_i < band_tol sig_max, and U_w that of their left vectors. The
    residual splits into the band's part, B = |U_w^T F|_2, and the resolved part, S = |F - U_w U_w^T F|_2, and the step
    into d_b = V_w V_w^T dz_N along the band and d_s = dz_N - d_b, the deflated step: the step of the system bordered
    with U_w and V_w. This is a Lyapunov-Schmidt corrector with a line search (H. B. Keller, Lectures on Numerical
    Methods in Bifurcation Problems, TIFR 79, Springer 1987: Method III and the bordering algorithm of section 4.11;
    design 2 of notes/LIT_REVIEW.md's addendum of #527).

    Inside the band the f64 SVD resolves nothing below 1e-16 sig_max, so the band gets its own basis: J V_w is formed
    in double-double, and the SVD of the reduced Jacobian U_w^T J V_w rotates U_w and V_w within their spans, with
    reduced singular values s_i. The f64 vectors span the band to about 1e-14, so the image of a band move y has a
    part |L y|_2 in the resolved rows, L = J V_w - U_w U_w^T J V_w: about 1.5e-14 |y|, which is 1e-11 for a band step
    of 1e3. A step that holds back the band move y leaves that much of S, whatever else it does: the floor E(y).
      - While S > max(kappa B, 2 E(a)), a = V_w^T dz_N, the state is off the band's valley: the step is d_s alone,
        halved until S falls.
      - Otherwise the step is d_s + V_w y, with y the damped band step of length at most the radius,
            y_i = a_i s_i^2 / (s_i^2 + mu^2)        (Levenberg-Marquardt's damping, here on the reduced problem):
        the band directions that the Jacobian resolves best move first, and the deepest are held back. With Newton's
        band step scaled as a whole they are not: its length sits in the deepest directions, which carry little of B.
        The move leaves a residual of second order in the resolved rows. It is removed by up to `correctors` chord
        steps with the same factors. A chord step c = -J^-1 F proposes the held-back band step again, so V_w (a - y)
        is taken off it exactly, and then the band part of what remains, by V_w V_w^T. The trial is accepted when B
        has fallen and S is at most max(kappa B, S before the step, 2 E(a - y), 0.1 tol), and the radius is halved
        otherwise. So a band step
        is judged by the band's own residual, on the valley.
    When the first trial holds, up to `grow` doublings of the radius are tried in the same iteration, each kept while
    it lowers B further: a trial costs a few residuals and chord steps, an iteration a factorization and an SVD. The
    radius changes only after a step it limited (mu > 0): it becomes the accepted step's length after a halving, and is
    doubled when no halving was needed.
    Nothing is added to the equations: near the root mu = 0 and the steps are Newton's own, and the result has
    max |F| < tol. A step no longer than the radius is tried whole first, with no SVD, and taken if it lowers |F|_2.

    band=False is the control: no band is deflated, which leaves Newton with backtracking on |F|_2.
    radius: the first limit on the length of a band step (default 1% of |z|_2). Returns zh, zl, history."""
    hist = []
    t0 = time.time()

    def vec(parts):
        return sum(np.asarray(p) for p in parts)

    if radius is None:
        radius = 0.01 * float(np.linalg.norm(vec([zh, zl])))
    for it in range(it_max):
        dz, F, pmin, pmax = s.newton([zh, zl])
        Fv = vec(F)
        nF, g0 = float(np.max(np.abs(Fv))), float(np.linalg.norm(Fv))
        rec = dict(it=it, F=nF, F2=g0, Delta=float(zh[-1]), piv=(pmin, pmax), t=round(time.time() - t0, 1))
        hist.append(rec)
        if nF < tol:
            rec["converged"] = True
            if verbose:
                log(f"  band {it}: |F| {nF:.2e} converged, Delta {zh[-1]:.15f}")
            return zh, zl, hist
        dN = vec(dz)
        nN = float(np.linalg.norm(dN))
        if nN <= radius:
            # a short step: Newton's own, if it lowers |F|_2
            th, tl = dd_add(zh, zl, dz[0], dz[1])
            g1 = float(np.linalg.norm(vec(s.residual([th, tl]))))
            if np.isfinite(g1) and g1 < g0:
                rec.update(mode="newton", F2_next=g1, dz=nN, dDelta=float(dN[-1]), t=round(time.time() - t0, 1))
                if verbose:
                    log(f"  band {it} (newton): |F| {nF:.2e}, |F|_2 {g0:.2e} -> {g1:.2e}, |dz| {nN:.1e}, dDelta "
                        f"{dN[-1]:+.2e}, pivots {pmin:.1e}..{pmax:.1e} ({rec['t']} s)")
                zh, zl = th, tl
                if after_step is not None:
                    after_step(zh, zl)
                continue
        Uw = Vw = L = sg = a = r = None
        k = 0
        if band:
            ts = time.time()
            Jh, Jl = s.jacobian(zh, zl)
            U, sig, Vt = np.linalg.svd(np.asarray(Jh) + np.asarray(Jl))
            weak = sig < band_tol * sig[0]
            k = int(weak.sum())
            # the count of singular values below 1e-3, 1e-4, ..., 1e-12 of the largest: where the band ends
            rec.update(sig_max=float(sig[0]), k=k,
                       sig_counts=[int(np.sum(sig < 10.0 ** -e * sig[0])) for e in range(3, 13)])
            if k:
                Uw, Vw = np.ascontiguousarray(U[:, weak]), np.ascontiguousarray(Vt[weak].T)
                JV = np.empty_like(Vw)
                zero = np.zeros(len(dN))
                for j in range(k):
                    JV[:, j] = vec(dd_matvec(Jh, Jl, np.ascontiguousarray(Vw[:, j]), zero))
                # the band's own basis, from the reduced Jacobian
                P, sg, Qt = np.linalg.svd(Uw.T @ JV)
                Uw, Vw, JV = Uw @ P, Vw @ Qt.T, JV @ Qt.T
                L = JV - Uw @ (Uw.T @ JV)
                a, r = Vw.T @ dN, Uw.T @ Fv
                M = getattr(s, "m", 0)
                if M:
                    # where the band lives: its share in the centre data q and c0, and its largest Delta component
                    rec.update(band_q=float(np.sum(Vw[:M] ** 2) / k), band_c0=float(np.sum(Vw[M:2 * M] ** 2) / k),
                               band_Delta=float(np.max(np.abs(Vw[-1]))))
                rec.update(band_sig=[float(x) for x in sg], band_r=[float(abs(x)) for x in r],
                           band_a=[float(abs(x)) for x in a])
            rec["t_svd"] = round(time.time() - ts, 1)
            del U, Vt, Jh, Jl

        def split(v):
            if not k:
                return float(np.linalg.norm(v)), 0.0
            rv = Uw.T @ v
            return float(np.linalg.norm(v - Uw @ rv)), float(np.linalg.norm(rv))

        S0, B0 = split(Fv)
        E = 0.0
        if k:
            db = Vw @ a
            nb = float(np.linalg.norm(a))
            ds = dd_add(dz[0], dz[1], -db, np.zeros_like(db))
            E = float(np.linalg.norm(L @ a))
        else:
            nb, ds = 0.0, (dz[0], dz[1])
        mode = "band" if k and S0 <= max(kappa * B0, 2 * E, 0.1 * tol) else "resolved"

        def damped(rad):
            """the weights w_i = s_i^2 / (s_i^2 + mu^2) for which |w a| = rad, and mu (bisection in log mu, the length
            falling as mu grows); all 1 and mu = 0 when the whole band step is no longer than rad"""
            if nb <= rad:
                return np.ones(k), 0.0
            lo, hi = np.log(max(sg.min(), 1e-300)) - 40.0, np.log(sg.max()) + 40.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if np.linalg.norm(a * sg ** 2 / (sg ** 2 + np.exp(2 * mid))) > rad:
                    lo = mid
                else:
                    hi = mid
            return sg ** 2 / (sg ** 2 + np.exp(2 * hi)), float(np.exp(hi))

        def trial(x):
            """the trial state: for a resolved step, d_s scaled by x; for a band step, the damped band step of length
            x. Returns (accepted, zh, zl, S, B, chord steps taken, S before them, the linear model's B, mu, length)"""
            if mode == "resolved":
                th, tl = dd_add(zh, zl, ds[0] * x, ds[1] * x)
                S1, B1 = split(vec(s.residual([th, tl])))
                return np.isfinite(S1) and S1 < S0, th, tl, S1, B1, 0, float("nan"), B0, 0.0, 0.0
            w, mu = damped(x)
            y = Vw @ (w * a)
            held = Vw @ ((1 - w) * a)
            Eh = float(np.linalg.norm(L @ ((1 - w) * a)))
            Bp = float(np.linalg.norm(r * (1 - w)))
            th, tl = dd_add(zh, zl, ds[0], ds[1])
            th, tl = dd_add(th, tl, y, np.zeros_like(y))
            S1, B1 = split(vec(s.residual([th, tl])))
            Sm, nc = S1, 0
            while nc < correctors and np.isfinite(S1) and S1 > max(kappa * B1, 1.5 * Eh, 0.1 * tol):
                c, _ = s.chord([th, tl])
                c = dd_add(c[0], c[1], -held, np.zeros_like(held))
                cb = Vw @ (Vw.T @ vec(c))
                ch, cl = dd_add(th, tl, c[0], c[1])
                ch, cl = dd_add(ch, cl, -cb, np.zeros_like(cb))
                S2, B2 = split(vec(s.residual([ch, cl])))
                if not (np.isfinite(S2) and S2 < S1):
                    break
                th, tl, S1, B1, nc = ch, cl, S2, B2, nc + 1
            ok = np.isfinite(S1) and B1 < B0 - 1e-4 * (B0 - Bp) and S1 <= max(kappa * B1, S0, 2 * Eh, 0.1 * tol)
            return ok, th, tl, S1, B1, nc, Sm, Bp, mu, float(np.linalg.norm(w * a))

        x = radius if mode == "band" else 1.0
        for h in range(max_halvings + 1):
            t = trial(x)
            if t[0]:
                break
            x *= 0.5
        grown = 0
        if t[0] and mode == "band" and h == 0:
            while grown < grow and t[8] > 0:
                t2 = trial(2 * x)
                if not (t2[0] and t2[4] < t[4]):
                    break
                t, x, grown = t2, 2 * x, grown + 1
        ok, th, tl, S1, B1, nc, Sm, Bp, mu, ny = t
        dzt = vec([th, tl]) - vec([zh, zl])
        rec.update(mode=mode, S=S0, B=B0, E=E, S_move=Sm, S_next=S1, B_next=B1, B_model=Bp, mu=mu, halvings=h,
                   grown=grown, correctors=nc, dz_newton=nN, dz_band=nb, dz_band_taken=ny,
                   dz=float(np.linalg.norm(dzt)), radius=radius, dDelta=float(dzt[-1]), t=round(time.time() - t0, 1))
        if verbose:
            if mode == "band":
                how = (f"band step {ny:.1e} of {nb:.1e} (mu {mu:.1e}, {h} halvings, {grown} doublings, model B "
                       f"{Bp:.2e}), {nc} correctors from S {Sm:.1e}")
            else:
                how = f"lam {x:.3g} ({h} halvings), band step {nb:.1e} held"
            log(f"  band {it} ({mode}, {k} deflated): |F| {nF:.2e}, S {S0:.2e} -> {S1:.2e} (E {E:.1e}), "
                f"B {B0:.2e} -> {B1:.2e}, {how}, |dz| {rec['dz']:.1e}, dDelta {rec['dDelta']:+.2e}, pivots "
                f"{pmin:.1e}..{pmax:.1e} ({rec['t']} s)")
        if not ok:
            rec["failed"] = "line search"
            return zh, zl, hist
        zh, zl = th, tl
        if mode == "band" and mu > 0:
            radius = x * (2 if h == 0 and B0 - B1 > 0.5 * (B0 - Bp) else 1)
        if after_step is not None:
            after_step(zh, zl)
    Fv = vec(s.residual([zh, zl]))
    nF = float(np.max(np.abs(Fv)))
    hist.append(dict(it=it_max, F=nF, Delta=float(zh[-1]), t=round(time.time() - t0, 1), converged=nF < tol))
    return zh, zl, hist


def newton_qd(s, z, tol=1e-55, it_max=8, verbose=True, log=print, jac="dd", first_rise=1.0):
    """Newton in quad-double: full steps while max |F| falls. jac "dd": the exact quad-double Jacobian (Solver.newton
    with four parts); "mixed": the f64 Jacobian in staircase storage with the quad-double residual (Solver.newton_mixed,
    linear convergence to the same root); "chord": the first step's factors (exact) kept for the later steps.
    first_rise: the first step is also taken when max |F| rises by less than this factor (1: only when it falls). On
    the near-null band the first step removes a displacement that the residual shows only at second order, and its
    remainder can exceed the start by a factor of order one (#928).
    Returns z, history."""
    z = [np.asarray(p, dtype=float) for p in z]
    hist = []
    t0 = time.time()
    for it in range(it_max):
        if jac == "mixed":
            dz, F, _, _ = s.newton_mixed(z)
        elif jac == "chord" and it > 0:
            dz, F = s.chord(z)
        else:
            dz, F, _, _ = s.newton(z)
        nF = fnorm(F)
        rec = dict(it=it, F=nF, Delta=float(z[0][-1]), t=round(time.time() - t0, 1))
        hist.append(rec)
        if nF < tol:
            rec["converged"] = True
            if verbose:
                log(f"  newton_qd {it}: |F| {nF:.2e} converged")
            break
        zn = list(qd_axpy(1.0, *dz, *z))
        Fn = s.residual(zn)
        n1 = fnorm(Fn)
        rec.update(F_next=n1, dDelta=float(dz[0][-1] + dz[1][-1]), dz=float(np.max(np.abs(dz[0] + dz[1]))))
        if verbose:
            log(f"  newton_qd {it}: |F| {nF:.2e} -> {n1:.2e}, max |dz| {rec['dz']:.2e}, dDelta {rec['dDelta']:+.2e} "
                f"({rec['t']} s)")
        if not (np.isfinite(n1) and (n1 < nF or (it == 0 and n1 < first_rise * nF))):
            rec["failed"] = "no decrease"
            break
        z = zn
    return z, hist


def qparts(zh, zl):
    return [np.asarray(zh, float), np.asarray(zl, float), np.zeros(len(zh)), np.zeros(len(zh))]


# ------------------------------------------------------------------ diagnostics
def fields(s, zh, K):
    """the node states (K, 5, M), centre data (q, c0) and horizon data from z (hi parts)"""
    M = s.m
    q, c0 = zh[:M], zh[M:2 * M]
    st = zh[2 * M:2 * M + 5 * M * K].reshape(K, 5, M)
    o = 2 * M + 5 * M * K
    return q, c0, st, zh[o:o + 3 * M].reshape(3, M)


def constraint(s, zh, nodes):
    """max over the nodes of |psi_tau + (p + m)/2 - c (p - m)/(2x)| relative to max |p| (zero for an exact solution;
    sigdss_solver.py's constraint, here a resolution diagnostic of the collocation in tau)"""
    M = s.m
    dh, dl = dmat_odd(M)
    D = dh + dl
    om = 2 * np.pi / zh[-1]
    _, _, st, _ = fields(s, zh, len(nodes))
    worst, pm = 0.0, 0.0
    for x, (p, m, a, c, q) in zip(nodes, st):
        K = om * (D @ q) + 0.5 * (p + m) - c * (p - m) / (2 * x)
        worst = max(worst, float(np.max(np.abs(K))))
        pm = max(pm, float(np.max(np.abs(p))))
    return worst / pm


def series_tail(s, zh, zl, which, nodes):
    """max over fields and points of the last three terms of the centre (0) or horizon (1) series at its evaluation
    radius, relative to the largest term"""
    v, nk, sig = s.series(zh, zl, which)
    M = s.m
    c = np.abs(np.asarray(v).reshape(nk, 5, M))
    r = (nodes[0] if which == 0 else 1 - nodes[-1]) / sig
    t = c.max(axis=(1, 2)) * r ** np.arange(nk)
    return float(t[-3:].max() / t.max())


def harmonics_q(zh, M):
    """|harmonic k| of the centre datum q (odd k), from its half-grid samples"""
    q = zh[:M]
    full = np.concatenate([q, -q])
    c = np.fft.rfft(full) / (2 * M)
    return {k: float(2 * abs(c[k])) for k in range(1, M, 2)}


# ------------------------------------------------------------------ growth rates (the mode matrix, #918)
def solver_of(meta, threads=2):
    """the Solver of a saved state: its coupling, D, tau points, nodes, series orders and planned steps"""
    eta = None if meta["eta"] in ("inf", None) else float(meta["eta"])
    s = make_solver(eta, int(meta["M"]), nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=meta["ks"],
                    threads=threads, D=float(meta.get("D", 4.0)))
    s.set_segs(meta["segs"])
    return s


def gauge_rate(zh, zl, meta, dps=40):
    """lambda_g = mean over tau of 1 - c_x(tau, 1), the growth rate of the mode that relabels the light cone: a
    neighbouring ingoing null ray x = 1 + d(tau) leaves the cone as d' = (1 - c_x(tau, 1)) d. From the horizon data
    a_h, psi_h with the kernel's own x c_x at c = 1 (model.rs, horizon_c1): (n-1)(a^2 - 1) - 2 n a^2 sigma^2,
    sigma = sin(kappa psi)/kappa. Returns an mpmath number."""
    M, K = int(meta["M"]), len(meta["nodes"])
    o = 2 * M + 5 * M * K
    with mp.workdps(dps):
        n = mp.mpf(float(meta.get("D", 4.0))) - 2
        k2 = mp.mpf(0) if meta["eta"] in ("inf", None) else n / mp.mpf(float(meta["eta"]))
        tot = mp.mpf(0)
        for j in range(M):
            a = mp.mpf(float(zh[o + M + j])) + mp.mpf(float(zl[o + M + j]))
            psi = mp.mpf(float(zh[o + 2 * M + j])) + mp.mpf(float(zl[o + 2 * M + j]))
            sg = psi if k2 == 0 else mp.sin(mp.sqrt(k2) * psi) / mp.sqrt(k2)
            tot += 1 - (n - 1) * (a * a - 1) + 2 * n * a * a * sg * sg
        return tot / M


def dmat_even(M):
    """d/dtheta of a field of period pi (the even fields a, c) from its M half-grid samples, in f64"""
    k = np.fft.fftfreq(M, 1.0 / M)
    if M % 2 == 0:
        k[M // 2] = 0.0
    eye = np.eye(M)
    return np.real(np.fft.ifft(2j * k[:, None] * np.fft.fft(eye, axis=0), axis=0))


def dtau_state(zh, meta, wrong=False):
    """d/dtau of every unknown before Delta (f64): omega d/dtheta with the odd operator on q, p, m, psi, m_h, psi_h
    and the even one on c0, a, c, a_h. This is the tau-translation mode, which the mode matrix at rate 0 annihilates
    up to the tau truncation. wrong=True takes the odd operator for the even fields too (the check's control)."""
    M, K = int(meta["M"]), len(meta["nodes"])
    om = 2 * np.pi / zh[-1]
    dh, dl = dmat_odd(M)
    Do = dh + dl
    De = Do if wrong else dmat_even(M)
    even = [False, True] + [False, False, True, True, False] * K + [False, True, False]
    v = np.zeros(len(zh) - 1)
    for i, ev in enumerate(even):
        v[i * M:(i + 1) * M] = om * ((De if ev else Do) @ zh[i * M:(i + 1) * M])
    return v


def det_fn(s, zh, zl, log=None, cache=None):
    """lam -> (sign, ln|det M(lam)|) at the state, by the double-double LU (Solver.mode_det); memoized in cache"""
    cache = {} if cache is None else cache

    def sd(lam):
        lam = float(lam)
        if lam not in cache:
            t = time.time()
            sg, la, pmin, pmax, im = s.mode_det(zh, zl, lam)
            assert im == 0.0, "a real rate gave an imaginary part"
            cache[lam] = (sg, la)
            if log:
                log(f"    lam {lam:.15f}: sign {sg:+.0f} ln|det| {la:.6f} pivots {pmin:.2e}..{pmax:.2e} "
                    f"({time.time() - t:.1f} s)")
        return cache[lam]
    return sd


def det_root(sd, a, b, xtol=1e-13):
    """the root of det M in [a, b], where its sign changes, by Brent's method on sign * exp(ln|det| - ref) (as
    hp/gamma_dd.py; scipy's brentq, whose reference is R. P. Brent, Algorithms for Minimization Without Derivatives,
    Prentice-Hall, 1973, ch. 3-4); None without a sign change"""
    from scipy.optimize import brentq
    if sd(a)[0] == sd(b)[0]:
        return None
    ref = 0.5 * (sd(a)[1] + sd(b)[1])
    return brentq(lambda x: sd(x)[0] * np.exp(sd(x)[1] - ref), a, b, xtol=xtol, rtol=1e-15)


def det_root_near(sd, guess, widths=(1e-3, 1e-2, 5e-2), xtol=1e-13):
    """the root of det M nearest guess: the first of the brackets guess (1 -+ w) with a sign change, the half that
    holds it. Returns (root or None, the bracket's w)."""
    for w in widths:
        lo, hi = guess * (1 - w), guess * (1 + w)
        if sd(lo)[0] != sd(hi)[0]:
            if sd(guess)[0] != sd(lo)[0]:
                return det_root(sd, lo, guess, xtol), w
            return det_root(sd, guess, hi, xtol), w
    return None, widths[-1]


def sign_bracket(cache, root):
    """the two neighbouring rates of the determinants in cache (det_fn's) that hold the root between them, with
    opposite signs: the root's enclosure by the sign of det M"""
    ks = sorted(cache)
    for p, q in zip(ks, ks[1:]):
        if p <= root <= q and cache[p][0] != cache[q][0]:
            return p, q
    return None


# ------------------------------------------------------------------ files
def save(path, zh, zl, meta, z2=None, z3=None):
    extra = {} if z2 is None else dict(z2=np.asarray(z2), z3=np.asarray(z3))
    np.savez(path, zh=np.asarray(zh), zl=np.asarray(zl), meta=json.dumps(meta), **extra)


def load(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    z = [d["zh"], d["zl"]] + ([d["z2"], d["z3"]] if "z2" in d else [])
    return z, meta


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC"
