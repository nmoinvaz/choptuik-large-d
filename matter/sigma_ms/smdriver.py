"""Driver of the sigma-model kernel (sigmams, matter/sigma_ms): set-up, seeds from sigdss states, Newton in double- and
quad-double, continuation in eps = 1/eta, diagnostics.

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

from multidouble import dd_add, dd_from_mp, dd_norm, qd_axpy  # noqa: F401
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


def newton_qd(s, z, tol=1e-55, it_max=8, verbose=True, log=print, jac="dd"):
    """Newton in quad-double: full steps while max |F| falls. jac "dd": the exact quad-double Jacobian (Solver.newton
    with four parts); "mixed": the f64 Jacobian in staircase storage with the quad-double residual (Solver.newton_mixed,
    linear convergence to the same root); "chord": the first step's factors (exact) kept for the later steps.
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
        rec.update(F_next=n1, dDelta=float(dz[0][-1] + dz[1][-1]))
        if verbose:
            log(f"  newton_qd {it}: |F| {nF:.2e} -> {n1:.2e}, dDelta {rec['dDelta']:+.2e} ({rec['t']} s)")
        if not (np.isfinite(n1) and n1 < nF):
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
