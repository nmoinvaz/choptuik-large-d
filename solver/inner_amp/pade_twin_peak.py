#!/usr/bin/env python3
"""The smallest chord twin by peak density (issue #672).

#669 (pade_twin_size.py size; notes/PADE_FAR_END.md 8) minimised the chord twin's coefficient norm subject to chi <= 1,
and that measure failed its calibration: chord2's smallest twin by norm peaks higher than chord2. Here the size is the
peak density on #548's grid, P(c) = max_j |phi(s_j)|, phi(s) = (1-s)^ex sum_{m<M} c_m T_m(2s-1), s_j = j/200, j < 200,
for the chord twin of pade_twin_size.py (M = 40 terms, exact Beta moments, the planted zero factor).

Problem. With A the n x M matrix of the twin's columns and b the target, rows weighted by 1/err_k (#670's errors,
F = 1), and Phi the J x M matrix of phi on the grid,
    P* = min_c max_j |(Phi c)_j|  subject to  |(A c - b)_k| <= 1 for every k < n,
a second-order cone program (each |.| of a complex number is a cone in R^3).

Coordinates. A = U S V (mpmath svd_c, full V). w = V c. For i < n with s_i >= 1 the unknown is y_i = s_i (w - w0)_i, for
i < n with s_i < 1 and for the null directions i >= n it is (w - w0)_i, with w0 = (U^H b)_i / s_i the least-norm
interpolant. In these unknowns x the residual is U D x (D_i = 1 or s_i) and the density a + G x with bounded entries,
so the cone program is solved in double precision (numpy) by a log-barrier Newton method (Boyd and Vandenberghe 2004,
Convex Optimization, Cambridge University Press, ch. 11), with the radius 1 - 1e-9.

Bracket. P_hi is the peak of an explicit twin c = V^H (w0 + E x), rebuilt at 150 digits, whose chi is recomputed from c
(at 150 digits, and at 250 with the 250-digit basis). P_lo is a dual bound (ibid. ch. 5): for any lambda in C^J with
||lambda||_1 = 1 and Phi^H lambda = A^H mu, every feasible c has max_j |(Phi c)_j| >= Re(lambda^H Phi c) =
Re(mu^H (b + r)) >= Re(mu^H b) - ||mu||_1. lambda comes from the barrier's central path, lambda_j ~ (Phi c)_j / (tau^2 -
|(Phi c)_j|^2), projected at 150 digits onto the complement of Phi V^H_i for the null directions and the directions with
s_i < thr, so that mu = U (v / s) exists, v = V Phi^H lambda; P_lo is the largest bound over thr in THRESH and over the
last three centred stages. The smallest twin lies in [P_lo, P_hi].

Context. The grid of 200 points leaves room between its points: the minimiser can bulge between s = 0 and 0.005, and
its polynomial factor can grow near s = 1, where (1-s)^ex hides it. So the same minimisation is repeated on s = j/1000,
and each twin's peak is also given on s = j/2000. Neither is judged.

    python3 pade_twin_peak.py --selftest [--mutate=radius|noproj|dual]
    python3 pade_twin_peak.py peak OUTDIR [--mutate=scale] [--allow_unraw]
    python3 pade_twin_peak.py peak_real OUTDIR [--allow_unraw]        (reads OUTDIR/peak_plants.json)

Exit 1 when a frozen control fails, or when a frozen rule forbids the real stage. Numerical evidence, not a bound on
the real series.
"""
import json
import math
import os
import random
import sys
import time

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pade_twin_size as ts                                            # noqa: E402

tc, pg = ts.tc, ts.pg
NMAX = ts.NMAX
NS = ts.NS_SIZE                     # 23, 31
M_PEAK = ts.M_SIZE                  # 40
M_CONV = ts.M_CONV                  # 50, convergence context at wf 1.00
GRID = 200                          # #548's grid, s = j/200, j < 200 (judged)
GRID_CTX = 1000                     # context: the same minimisation on s = j/1000
FINE = 2000                         # context: the explicit twin's peak on s = j/2000
DPS_FIT, DPS_CHECK = ts.DPS_FIT, ts.DPS_CHECK
R_SOLVE = 1 - 1e-9                  # the barrier's polydisc radius, so the explicit twin has chi < 1
TOL_GAP = 1e-9                      # barrier stops when (2J + n)/t <= TOL_GAP tau
THRESH = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 0)
SCALE_MUT = mp.mpf("0.75")
MUT_DUAL = None                     # set by selftest(--mutate=dual)
FROZEN = {                          # written before the peak and peak_real stages were first run (#672)
    "size": "P = max_j |phi(s_j)| on s_j = j/200, j < 200, of the chord twin with M = 40 terms (pade_twin_size.py's "
            "chord basis and #670's errors, F = 1), minimised exactly subject to chi <= 1; bracket [P_lo, P_hi] from "
            "an explicit twin and a dual bound; targets chord2, arcjump, arc; n = 23, 31; every node",
    "planted": "selftest: (1) a rotated separable problem with weights 1e-30..1e30, P* = max_k (|b_k| - 1)/s_k = 3.1; "
               "(2) the Chebyshev problem on a grid holding the extrema of T_10, P* = (1 - 1/S) 2^-9 with S = 100; "
               "each with P_hi and P_lo within 1e-6 relative of P*, P_lo <= P* (1 + 1e-12), P_hi >= P* (1 - 1e-12), and the "
               "explicit twin's chi <= 1. (3) #670's planted 8-term chord density at wf 1.00, n = 23: P_lo <= P_hi <= "
               "its own peak, P_hi / P_lo - 1 <= 0.01, chi <= 1 + 1e-6 at 250 digits. --mutate=radius (polydisc "
               "radius 1.05), --mutate=noproj (lambda not projected) and --mutate=dual (the bound's residual term "
               "dropped) each make the selftest fail",
    "P_gap": "P_hi / P_lo - 1 <= 0.01 for every target, n and node",
    "P_valid": "P_lo <= P_hi (1 + 1e-12); the explicit twin's chi recomputed at 250 digits with the 250-digit basis "
               "<= 1 + 1e-6; and P_lo recomputed at 250 digits from the same lambda within 1e-6 relative",
    "P0_calibration": "chord2's smallest twin within 10% of chord2's own peak P_own (its 6-term density): P_lo >= 0.9 "
                      "P_own and P_hi <= 1.1 P_own, at n = 23 and 31 and every node",
    "mutation_peak": "--mutate=scale (chord2's sigma_k x 0.75, its errors kept) must make P0 fail at every n and node; "
                     "the run exits 1",
    "separation": "the factor-3 rule of 8.2: at n = 31, rho_hi = P_hi(arcjump) / P_lo(chord2) and rho_lo = "
                  "P_lo(arcjump) / P_hi(chord2). If rho_hi <= 3 at every node, no peak prior separates them (the "
                  "answer); if rho_lo > 3 at every node, a peak prior of factor 3 separates them; otherwise mixed. "
                  "n = 23 is reported the same way",
    "peak_real": "only if P_gap, P_valid and P0 pass and rho_lo > 3 at every node at n = 23: the real series' smallest "
                 "peak chord twin at n = 23, rho_real = P(real) / P(chord2) per node, read against 3; otherwise no "
                 "real run",
    "context": "the same minimisation on s = j/1000 with the separation rule and chord2's ratio to its own peak there; "
               "each twin's peak on s = j/2000, coefficient norm and largest |c_m|; the number of active constraints; "
               "the plant arc's ratio; arcjump against chord2's own peak; M = 50 at wf 1.00; reported, not judged",
}


# ---------------------------------------------------------------- the grid and the reduced problem
def grid_matrix(ex, M, J, dps):
    """Phi[j, m] = (1 - s_j)^ex T_m(2 s_j - 1), s_j = j/J, j < J (the chord twin, w = 1)"""
    with mp.workdps(dps):
        ex = mp.mpc(ex)
        P = mp.matrix(J, M)
        for j in range(J):
            s = mp.mpf(j) / J
            x = 2 * s - 1
            wt = mp.power(1 - s, ex)
            t0, t1 = mp.mpf(1), x
            for m in range(M):
                P[j, m] = wt * t0
                t0, t1 = t1, 2 * x * t1 - t0
        return P


def to_np(Mx):
    return np.array([[complex(Mx[i, j]) for j in range(Mx.cols)] for i in range(Mx.rows)])


class Reduced:
    """the cone program in the coordinates of the module docstring, at the working precision dps"""

    def __init__(self, Phi, A, b, dps):
        self.dps = dps
        with mp.workdps(dps):
            n, M = A.rows, A.cols
            self.n, self.M, self.J = n, M, Phi.rows
            U, S, V = mp.svd_c(A, full_matrices=True)
            self.U, self.V, self.Phi, self.A, self.b = U, V, Phi, A, b
            self.S = S = [mp.re(S[i]) for i in range(n)]
            self.beta = beta = U.H * b
            # start: the ridge twin w0_i = s_i beta_i / (s_i^2 + lam) with ||S w0 - beta||_2 = 1/2, so |r_k| <= 1/2
            ab = [abs(beta[i]) for i in range(n)]

            def ynorm(L):
                lam = mp.mpf(10) ** L
                return mp.sqrt(mp.fsum((lam * ab[i] / (S[i] ** 2 + lam)) ** 2 for i in range(n)))

            if mp.sqrt(mp.fsum(x * x for x in ab)) <= mp.mpf(1) / 2:
                lam = mp.inf
                w0 = [mp.mpc(0)] * M
            else:
                lo, hi = mp.mpf(-400), mp.mpf(400)
                for _ in range(200):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if ynorm(mid) < mp.mpf(1) / 2 else (lo, mid)
                lam = mp.mpf(10) ** lo
                w0 = [S[i] * beta[i] / (S[i] ** 2 + lam) if i < n else mp.mpc(0) for i in range(M)]
            self.w0 = mp.matrix(w0)
            y0 = mp.matrix([S[i] * w0[i] - beta[i] for i in range(n)])
            self.E = [1 / S[i] if i < n and S[i] >= 1 else mp.mpf(1) for i in range(M)]
            D = [mp.mpf(1) if S[i] >= 1 else S[i] for i in range(n)]
            PV = Phi * V.H                                  # J x M: the density of each right singular vector
            self.PV = PV
            a = PV * self.w0
            r0 = U * y0
            G = mp.matrix(self.J, M)
            for j in range(self.J):
                for i in range(M):
                    G[j, i] = PV[j, i] * self.E[i]
            H = mp.matrix(n, n)
            for k in range(n):
                for i in range(n):
                    H[k, i] = U[k, i] * D[i]
            self.a, self.G, self.H = np.array([complex(a[j]) for j in range(self.J)]), to_np(G), to_np(H)
            self.r0 = np.array([complex(r0[k]) for k in range(n)])
            self.ridge_log10 = float(lo) if lam != mp.inf else None

    def coeffs(self, x):
        """c = V^H (w0 + E x) at the working precision, x a numpy complex vector"""
        with mp.workdps(self.dps):
            w = mp.matrix([self.w0[i] + self.E[i] * mp.mpc(complex(x[i])) for i in range(self.M)])
            return self.V.H * w


def ipm(a, G, H, R, r0, tol=TOL_GAP, mu=10.0, log=None):
    """min tau s.t. |a_j + (G x)_j| <= tau, |r0_k + (H x[:n])_k| <= R, started at x = 0 (so |r0| < R is required).
    The unknowns are whitened first: x = T xi with [G; H 0] T orthonormal (numpy SVD), dropping the directions whose
    singular value is below 1e-13 of the largest (they move neither the density nor a residual). Then a log-barrier
    method in (Re xi, Im xi, tau): damped Newton steps 1/(1 + sqrt(dec)) for these self-concordant barriers (Boyd and
    Vandenberghe 2004, 9.6), full steps once sqrt(dec) < 1/4, with t x mu per stage until
    (2J + n)/t <= tol tau. Returns x, tau, lambda (the density duals on the last centring point) and stats"""
    J, M = G.shape
    n = H.shape[0]
    R = float(R)
    Hpad = np.zeros((n, M), dtype=complex)
    Hpad[:, :n] = H
    P, sv, Qh = np.linalg.svd(np.vstack([G, Hpad]), full_matrices=False)
    keep = sv > 1e-13 * sv[0]
    T = Qh.conj().T[:, keep] / sv[keep]
    Gw, Hw = G @ T, Hpad @ T
    r = Gw.shape[1]
    Lu, Lv = np.hstack([Gw.real, -Gw.imag]), np.hstack([Gw.imag, Gw.real])
    Hp, Hq = np.hstack([Hw.real, -Hw.imag]), np.hstack([Hw.imag, Hw.real])
    au, av, pu, pv = a.real, a.imag, r0.real, r0.imag
    N = 2 * r + 1
    z = np.zeros(N)
    z[-1] = 1.1 * float(np.max(np.abs(a))) + 1e-6
    mpar = 2 * J + n
    t = mpar / z[-1]

    def parts(z):
        x, tau = z[:-1], z[-1]
        u, v, p, q = au + Lu @ x, av + Lv @ x, pu + Hp @ x, pv + Hq @ x
        return u, v, p, q, tau * tau - u * u - v * v, R * R - p * p - q * q

    def feasible(z):
        u, v, p, q, D1, D2 = parts(z)
        return z[-1] > 0 and np.min(D1) > 0 and np.min(D2) > 0

    def duals(z, t):
        u, v, p, q, D1, D2 = parts(z)
        return 2 * (u + 1j * v) / (t * D1)

    stages = newton = 0
    stalled = False
    snaps = []                      # the duals of every stage whose centring converged
    while True:
        centred = False
        for _ in range(100):
            u, v, p, q, D1, D2 = parts(z)
            tau = z[-1]
            g = np.empty(N)
            g[:-1] = Lu.T @ (2 * u / D1) + Lv.T @ (2 * v / D1) + Hp.T @ (2 * p / D2) + Hq.T @ (2 * q / D2)
            g[-1] = t - np.sum(2 * tau / D1)
            i1, i2 = 1 / D1, 1 / D2
            huu, hvv, huv = 2 * i1 + 4 * u * u * i1 ** 2, 2 * i1 + 4 * v * v * i1 ** 2, 4 * u * v * i1 ** 2
            htu, htv = -4 * tau * u * i1 ** 2, -4 * tau * v * i1 ** 2
            hpp, hqq, hpq = 2 * i2 + 4 * p * p * i2 ** 2, 2 * i2 + 4 * q * q * i2 ** 2, 4 * p * q * i2 ** 2
            Hs = np.empty((N, N))
            Hs[:-1, :-1] = (Lu.T @ (huu[:, None] * Lu) + Lu.T @ (huv[:, None] * Lv) + Lv.T @ (huv[:, None] * Lu)
                            + Lv.T @ (hvv[:, None] * Lv) + Hp.T @ (hpp[:, None] * Hp) + Hp.T @ (hpq[:, None] * Hq)
                            + Hq.T @ (hpq[:, None] * Hp) + Hq.T @ (hqq[:, None] * Hq))
            Hs[:-1, -1] = Hs[-1, :-1] = Lu.T @ htu + Lv.T @ htv
            Hs[-1, -1] = np.sum(-2 * i1 + 4 * tau * tau * i1 ** 2)
            dz = np.linalg.solve(Hs, -g)
            dec = max(-float(g @ dz), 0.0)
            newton += 1
            if dec <= 1e-14:
                centred = True
                break
            s = 1.0 if math.sqrt(dec) < 0.25 else 1 / (1 + math.sqrt(dec))
            while not feasible(z + s * dz):
                s /= 2
                if s < 1e-14:
                    break
            if s < 1e-14:
                stalled = True
                break
            z = z + s * dz
            if s == 1.0 and dec <= 1e-10:
                centred = True
                break
        stages += 1
        if centred:
            snaps.append({"t": t, "tau": float(z[-1]), "lam": duals(z, t)})
        if log:
            log(f"      stage {stages}: t {t:.2e}, tau {z[-1]:.12f}, Newton decrement {dec:.1e}, steps {newton}"
                + ("" if centred else ", not centred") + (", stalled" if stalled else ""))
        if stalled or not centred or mpar / t <= tol * z[-1]:
            break
        t *= mu
    return {"x": T @ (z[:r] + 1j * z[r:2 * r]), "tau": float(z[-1]), "snaps": snaps[-3:], "t": t, "stages": stages,
            "newton": newton, "stalled": stalled, "centred": centred, "gap_est": mpar / t, "rank": int(r)}


def lower_bound(red, lam, R, thr, project=True):
    """the dual bound of the module docstring for one threshold, at red.dps; None when it is not finite"""
    with mp.workdps(red.dps):
        n, M, J = red.n, red.M, red.J
        lm = mp.matrix([mp.mpc(complex(z)) for z in lam])
        sel = [i for i in range(M) if i >= n or red.S[i] < thr]
        if project and sel:
            Q = mp.matrix(J, len(sel))
            for j in range(J):
                for c, i in enumerate(sel):
                    Q[j, c] = red.PV[j, i]
            lm = lm - Q * mp.lu_solve(Q.H * Q, Q.H * lm)
        l1 = mp.fsum(abs(lm[j]) for j in range(J))
        v = red.V * (red.Phi.H * lm)
        big = [i for i in range(n) if i not in sel]
        mu = mp.matrix(n, 1)
        for i in big:
            for k in range(n):
                mu[k] += red.U[k, i] * v[i] / red.S[i]
        val = mp.re(mp.fsum(mp.conj(v[i]) * red.beta[i] / red.S[i] for i in big))
        if MUT_DUAL != "dual":      # --mutate=dual drops the residual term, which makes the bound invalid
            val -= R * mp.fsum(abs(mu[k]) for k in range(n))
        resid = max([abs(v[i]) for i in sel], default=mp.mpf(0))
        return val / l1, lm, float(resid)


def peak_of(Phi, c, dps):
    with mp.workdps(dps):
        d = Phi * c
        return max(abs(d[j]) for j in range(Phi.rows))


def peak_min(Phi, A, b, dps, radius=1, project=True, log=None):
    """the smallest peak over the twins with chi <= radius: P_lo, P_hi, the explicit twin c and the duals"""
    t0 = time.time()
    red = Reduced(Phi, A, b, dps)
    res = ipm(red.a, red.G, red.H, R_SOLVE * radius, red.r0, log=log)
    c = red.coeffs(res["x"])
    with mp.workdps(dps):
        r = A * c - b
        chi = max(abs(r[k]) for k in range(A.rows))
        p_hi = peak_of(Phi, c, dps)
        d = Phi * c
        active_p = sum(1 for j in range(Phi.rows) if abs(d[j]) >= p_hi * (1 - mp.mpf("1e-6")))
        active_r = sum(1 for k in range(A.rows) if abs(r[k]) >= radius * (1 - mp.mpf("1e-6")))
    lbs, best = {}, None             # every (stage, thr) bound is valid, so the largest is taken
    for si, snap in enumerate(res["snaps"]):
        for thr in THRESH:
            try:
                lb, _, _ = lower_bound(red, snap["lam"], radius, thr, project)
            except (ZeroDivisionError, ValueError):
                lb = None
            lbs[f"{si}:{thr:g}"] = float(lb) if lb is not None else None
            if lb is not None and (best is None or lb > best[0]):
                best = (lb, si, thr)
    p_lo, si, thr_best = best
    out = {"P_lo": float(p_lo), "P_hi": float(p_hi), "gap": float(p_hi / p_lo - 1) if p_lo > 0 else math.inf,
           "chi": float(chi), "tau_ipm": res["tau"], "thr": thr_best, "snap_t": res["snaps"][si]["t"],
           "stages": res["stages"], "newton": res["newton"], "stalled": res["stalled"], "centred": res["centred"],
           "rank": res["rank"], "active_peak": active_p, "active_resid": active_r, "lb_by_stage_thr": lbs,
           "seconds": time.time() - t0}
    res["lam"] = res["snaps"][si]["lam"]
    if log:
        log(f"    P in [{out['P_lo']:.8f}, {out['P_hi']:.8f}] (gap {out['gap']:.1e}; dual from t {out['snap_t']:.1e}, "
            f"thr {thr_best:g}); chi {out['chi']:.9f}; active: {active_p} grid points, {active_r} residuals; "
            f"{res['stages']} stages, {res['newton']} Newton steps, rank {res['rank']} ({out['seconds']:.0f}s)")
    return out, c, red, res


# ---------------------------------------------------------------- selftest
def unitary(M, seed, dps):
    rng = random.Random(seed)
    with mp.workdps(dps):
        Z = mp.matrix(M, M)
        for i in range(M):
            for j in range(M):
                Z[i, j] = mp.mpc(rng.gauss(0, 1), rng.gauss(0, 1))
        Q, _ = mp.qr(Z)
        return Q


def planted_separable(dps, radius=1):
    """Phi = R^H, A = [diag(s) 0] R^H with R unitary, so in c' = R^H c the problem separates: P* = max_k (|b_k| - 1)_+ /
    s_k. Weights 1e-30..1e30; the active constraint has s = 1"""
    M, n = 12, 8
    with mp.workdps(dps):
        s = [mp.mpf(v) for v in ("1e-30", "1e-10", "1", "1", "1e10", "1e20", "1e30", "1e-3")]
        q = [None, None, mp.mpf("3.1"), mp.mpf("1.7"), mp.mpf("2.5"), mp.mpf("0.9"), mp.mpf("2.0"), None]
        mag = [mp.mpf("0.5") if q[k] is None else 1 + s[k] * q[k] for k in range(n)]
        b = mp.matrix([mag[k] * mp.expjpi(mp.mpf(k) / 5) for k in range(n)])
        Rm = unitary(M, 672, dps)
        Phi = Rm.H
        A = mp.matrix(n, M)
        for k in range(n):
            for j in range(M):
                A[k, j] = s[k] * Rm.H[k, j]
        exact = max(q[k] for k in range(n) if q[k] is not None)
    return Phi, A, b, exact


def planted_cheb(dps, N=10, S=100, extra=60):
    """Phi[j, m] = T_m(x_j) on a grid with the N + 1 extrema of T_N, one constraint |S 2^(N-1) c_N - S| <= 1: the
    smallest max |p| is (1 - 1/S) 2^(1-N) (equioscillation)"""
    with mp.workdps(dps):
        xs = [mp.cos(mp.pi * j / N) for j in range(N + 1)] + [-1 + 2 * (mp.mpf(j) + mp.mpf("0.37")) / extra
                                                              for j in range(extra)]
        M = N + 1
        Phi = mp.matrix(len(xs), M)
        for j, x in enumerate(xs):
            for m in range(M):
                Phi[j, m] = mp.chebyt(m, x)
        A = mp.matrix(1, M)
        A[0, N] = S * mp.mpf(2) ** (N - 1)
        b = mp.matrix([mp.mpf(S)])
        exact = (1 - mp.mpf(1) / S) * mp.mpf(2) ** (1 - N)
    return Phi, A, b, exact


def selftest(mut=None):
    global MUT_DUAL
    MUT_DUAL = mut
    ok = True
    rad = mp.mpf("1.05") if mut == "radius" else 1
    proj = mut != "noproj"
    for name, maker in (("separable, weights 1e-30..1e30", planted_separable),
                        ("Chebyshev T_10, S = 100", planted_cheb)):
        Phi, A, b, exact = maker(DPS_FIT)
        out, c, _, _ = peak_min(Phi, A, b, DPS_FIT, radius=rad, project=proj)
        with mp.workdps(DPS_FIT):
            chi = max(abs((A * c - b)[k]) for k in range(A.rows))
        e_hi, e_lo = abs(out["P_hi"] / float(exact) - 1), abs(out["P_lo"] / float(exact) - 1)
        valid = out["P_lo"] <= float(exact) * (1 + 1e-12) and out["P_hi"] >= float(exact) * (1 - 1e-12)
        good = e_hi <= 1e-6 and e_lo <= 1e-6 and valid and chi <= 1
        ok &= good
        print(f"selftest planted {name}: P* = {float(exact):.10e}; P_hi {out['P_hi']:.10e} (rel {e_hi:.1e}), P_lo "
              f"{out['P_lo']:.10e} (rel {e_lo:.1e}), both within 1e-6 and on their side of P*: "
              f"{e_hi <= 1e-6 and e_lo <= 1e-6 and valid}; twin chi {float(chi):.9f} (<= 1); "
              f"{'PASS' if good else 'FAIL'}")
    # 3: the real chord basis, #670's planted 8-term density at n = 23: its smallest twin is no larger than itself
    wf = 1.0
    c2, _ = tc.plant(wf, "chord2")
    rel, _ = tc.err_model(wf)
    target = tc.planted_target(c2, 8)
    with mp.workdps(DPS_FIT):
        cm = [mp.mpf("0.8") ** m * mp.expjpi(mp.mpf(m) / mp.pi) for m in range(8)]
        s0 = mp.fsum(cm[m] * c2.moments(8)[m][0] for m in range(8))
        own = mp.matrix([cm[m] / s0 for m in range(8)] + [mp.mpc(0)] * (M_PEAK - 8))
    out, cc, chk_ok = run_one(c2, target, rel, ts.N23, M_PEAK, rad, proj)
    with mp.workdps(DPS_FIT):
        p_own = float(peak_of(grid_matrix(c2.ex, M_PEAK, GRID, DPS_FIT), own, DPS_FIT))
    good = (out["P_hi"] <= p_own * (1 + 1e-9) and out["P_lo"] <= out["P_hi"] * (1 + 1e-12) and out["gap"] <= 0.01
            and out["chi_check"] <= 1 + 1e-6)
    ok &= good
    print(f"selftest planted 8-term chord density, n {ts.N23}: own peak {p_own:.6f}; P in [{out['P_lo']:.6f}, "
          f"{out['P_hi']:.6f}] (P_lo <= P_hi <= own, gap {out['gap']:.1e} <= 0.01), chi at 250 digits "
          f"{out['chi_check']:.9f} (<= 1): {'PASS' if good else 'FAIL'}")
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- stages
def run_one(base, target, rel, n, M, radius=1, project=True, log=None, mut=None, fit=None, chk=None, J=GRID):
    """the smallest peak chord twin of target (sigma_0..sigma_(n-1)) at n and M on s = j/J, with its checks at
    DPS_CHECK"""
    fit = fit or ts.chord_cols(base, DPS_FIT, M, NMAX)
    chk = chk or ts.chord_cols(base, DPS_CHECK, M, NMAX)
    sig = target[:n]
    err = tc.weights(sig, rel, n, 1.0, DPS_FIT)
    errc = tc.weights(sig, rel, n, 1.0, DPS_CHECK)
    if mut == "scale":
        sig = [z * SCALE_MUT for z in sig]
    A, b = ts.weighted(fit, sig, err, n, M)
    Phi = grid_matrix(base.ex, M, J, DPS_FIT)
    out, c, red, res = peak_min(Phi, A, b, DPS_FIT, radius, project, log)
    with mp.workdps(DPS_CHECK):
        A2, b2 = ts.weighted(chk, sig, errc, n, M)
        c2 = mp.matrix([mp.mpc(c[m]) for m in range(M)])
        out["chi_check"] = float(max(abs((A2 * c2 - b2)[k]) for k in range(n)))
        Phi2 = grid_matrix(base.ex, M, J, DPS_CHECK)
        red2 = Reduced(Phi2, A2, b2, DPS_CHECK)
        lb2, _, _ = lower_bound(red2, res["lam"], radius, out["thr"], project)
        out["P_lo_check"] = float(lb2)
        out["P_lo_check_rel"] = abs(float(lb2) / out["P_lo"] - 1)
    cl = [c[m] for m in range(M)]
    out["c_norm"] = float(mp.sqrt(mp.fsum(abs(z) ** 2 for z in cl)))
    out["c_max"] = max(float(abs(z)) for z in cl)
    out["c_abs"] = [float(abs(z)) for z in cl]
    s = np.arange(FINE) / FINE
    v = np.polynomial.chebyshev.chebval(2 * s - 1, np.array([complex(z) for z in cl])) * (1 - s) ** complex(base.ex)
    out["P_fine"] = float(np.max(np.abs(v)))
    out["argmax_fine"] = float(s[int(np.argmax(np.abs(v)))])
    if log:
        log(f"    250 digits: chi {out['chi_check']:.9f}, P_lo {out['P_lo_check']:.8f} (rel "
            f"{out['P_lo_check_rel']:.1e}); context: peak on s = j/{FINE} {out['P_fine']:.6f} at s "
            f"{out['argmax_fine']:.4f}, ||c|| {out['c_norm']:.3e}, max|c| {out['c_max']:.3e}")
    return out, cl, True


def own_peaks(base, own_c):
    """chord2's own density (6 terms) on the judged grid, the context grid and s = j/FINE"""
    with mp.workdps(DPS_FIT):
        own = mp.matrix(list(own_c) + [mp.mpc(0)] * (M_PEAK - len(own_c)))
        out = {J: float(peak_of(grid_matrix(base.ex, M_PEAK, J, DPS_FIT), own, DPS_FIT)) for J in (GRID, GRID_CTX)}
    s = np.arange(FINE) / FINE
    v = np.polynomial.chebyshev.chebval(2 * s - 1, np.array([complex(z) for z in own_c])) * (1 - s) ** complex(base.ex)
    out[FINE] = float(np.max(np.abs(v)))
    return out


def separation(per, own, n):
    """the factor-3 rule of 8.2 on the peak, per node: rho_hi = P_hi(arcjump) / P_lo(chord2), rho_lo = P_lo(arcjump) /
    P_hi(chord2)"""
    sep = {}
    for wf, tw in per.items():
        c2, aj = tw["chord2"][n], tw["arcjump"][n]
        sep[wf] = {"rho_hi": aj["P_hi"] / c2["P_lo"], "rho_lo": aj["P_lo"] / c2["P_hi"],
                   "arcjump_over_own": aj["P_hi"] / own[wf], "arc_over_chord2": tw["arc"][n]["P_hi"] / c2["P_hi"]}
    hi_ok = all(x["rho_hi"] <= 3 for x in sep.values())
    lo_ok = all(x["rho_lo"] > 3 for x in sep.values())
    return {"nodes": sep, "verdict": "no peak prior separates" if hi_ok else
            ("a peak prior of factor 3 separates" if lo_ok else "mixed")}


def log_sep(log, label, s):
    log(f"{label}: " + "; ".join(f"wf {wf} rho {x['rho_lo']:.3f}..{x['rho_hi']:.3f}, arcjump / own "
                                 f"{x['arcjump_over_own']:.3f}, arc / chord2 {x['arc_over_chord2']:.3f}"
                                 for wf, x in s["nodes"].items()) + f" -> {s['verdict']}")


def run_peak(outdir, mut=None):
    log = ts.Log(outdir, "peak_plants", mut)
    rec = {"frozen": FROZEN, "mutate": mut, "nodes": {}}
    for wf in pg.NODES:
        pl = ts.size_targets(wf)
        base = pl["chord2"][0]
        fit, chk = ts.chord_cols(base, DPS_FIT, M_CONV, NMAX), ts.chord_cols(base, DPS_CHECK, M_CONV, NMAX)
        rel, info = tc.err_model(wf)
        node = rec["nodes"].setdefault(wf, {"errors": info})
        for n in NS:
            own_c, own_chi, _ = tc.twin_fit(fit, pl["chord2"][1][:n], rel, n, ts.M_CONTROL, 1.0)
            po = own_peaks(base, own_c)
            node.setdefault("own", {})[n] = {"P_own": po[GRID], "P_own_ctx": po[GRID_CTX], "P_own_fine": po[FINE],
                                             "chi": own_chi}
            log(f"wf {wf} n {n}: chord2's own density (M 6): peak {po[GRID]:.6f} on s = j/{GRID} ({po[GRID_CTX]:.6f} "
                f"on j/{GRID_CTX}, {po[FINE]:.6f} on j/{FINE}), chi {own_chi:.1e}")
            for kind in ("chord2",) if mut else ("chord2", "arcjump", "arc"):     # a mutation run judges P0 only
                log(f"  wf {wf} n {n} {kind}:")
                out, _, _ = run_one(base, pl[kind][1], rel, n, M_PEAK, log=log, mut=mut, fit=fit, chk=chk)
                node.setdefault("twins", {}).setdefault(kind, {})[n] = out
                if not mut:
                    log(f"    context, grid s = j/{GRID_CTX}:")
                    cx, _, _ = run_one(base, pl[kind][1], rel, n, M_PEAK, log=log, fit=fit, chk=chk, J=GRID_CTX)
                    node.setdefault("twins_ctx", {}).setdefault(kind, {})[n] = cx
                if wf == 1.0 and not mut:
                    cv, _, _ = run_one(base, pl[kind][1], rel, n, M_CONV, fit=fit, chk=chk)
                    out["M50"] = {k: cv[k] for k in ("P_lo", "P_hi", "gap", "chi_check")}
                    out["M50"]["rel_change"] = cv["P_hi"] / out["P_hi"] - 1
                    log(f"    M {M_CONV} (context): P in [{cv['P_lo']:.8f}, {cv['P_hi']:.8f}], "
                        f"{out['M50']['rel_change']:+.2e} relative to M {M_PEAK}")
                log.dump(rec)
    v = {"P_gap": True, "P_valid": True, "P0": {}, "separation": {}, "context_grid": {}}
    for wf, node in rec["nodes"].items():
        for kind, per in node["twins"].items():
            for n, o in per.items():
                v["P_gap"] &= o["gap"] <= 0.01
                v["P_valid"] &= (o["chi_check"] <= 1 + 1e-6 and o["P_lo_check_rel"] <= 1e-6
                                 and o["P_lo"] <= o["P_hi"] * (1 + 1e-12))
    for n in NS:
        p0 = {}
        for wf, node in rec["nodes"].items():
            c2, own = node["twins"]["chord2"][n], node["own"][n]["P_own"]
            p0[wf] = {"lo_ratio": c2["P_lo"] / own, "hi_ratio": c2["P_hi"] / own,
                      "pass": c2["P_lo"] >= 0.9 * own and c2["P_hi"] <= 1.1 * own}
        v["P0"][n] = {"nodes": p0, "pass": all(x["pass"] for x in p0.values())}
        log(f"n {n}: P0 " + "; ".join(f"wf {wf} P/P_own {x['lo_ratio']:.4f}..{x['hi_ratio']:.4f} "
                                      f"{'OK' if x['pass'] else 'BAD'}" for wf, x in p0.items()))
        if mut:
            continue
        own = {wf: node["own"][n]["P_own"] for wf, node in rec["nodes"].items()}
        v["separation"][n] = separation({wf: node["twins"] for wf, node in rec["nodes"].items()}, own, n)
        log_sep(log, f"n {n}: separation", v["separation"][n])
        own_x = {wf: node["own"][n]["P_own_ctx"] for wf, node in rec["nodes"].items()}
        v["context_grid"][n] = separation({wf: node["twins_ctx"] for wf, node in rec["nodes"].items()}, own_x, n)
        v["context_grid"][n]["P0_ratio"] = {wf: node["twins_ctx"]["chord2"][n]["P_lo"] / own_x[wf]
                                            for wf, node in rec["nodes"].items()}
        log_sep(log, f"n {n}: context, grid j/{GRID_CTX}: separation", v["context_grid"][n])
        log(f"n {n}: context, grid j/{GRID_CTX}: chord2's P_lo / own " + ", ".join(
            f"wf {wf} {x:.4f}" for wf, x in v["context_grid"][n]["P0_ratio"].items()))
    v["controls_pass"] = v["P_gap"] and v["P_valid"] and all(v["P0"][n]["pass"] for n in NS)
    log(f"P_gap {v['P_gap']}, P_valid {v['P_valid']}, P0 {[v['P0'][n]['pass'] for n in NS]}; controls pass: "
        f"{v['controls_pass']}")
    rec["verdict"] = v
    log.dump(rec)
    return v["controls_pass"]


def run_peak_real(outdir):
    plants = json.load(open(os.path.join(outdir, "peak_plants.json")))
    v = plants["verdict"]
    sep23 = v["separation"]["23"]["nodes"]
    if not (v["controls_pass"] and all(x["rho_lo"] > 3 for x in sep23.values())):
        print("frozen rule: the peak controls or the separation at n = 23 do not hold, so no real peak run")
        return False
    log = ts.Log(outdir, "peak_real", None)
    rec = {"frozen": FROZEN, "nodes": {}}
    with mp.workdps(DPS_FIT):                   # the files' strings at full precision (tc.real_sig() may hold 53 bits)
        real = pg.load_scan(pg.REAL["J1250"])
    for wf in pg.NODES:
        base = tc.plant(wf, "chord2")[0]
        rel, _ = tc.err_model(wf)
        sig = real[wf][:ts.N23]
        out, _, _ = run_one(base, sig, rel, ts.N23, M_PEAK, log=log)
        c2 = plants["nodes"][str(wf)]["twins"]["chord2"]["23"]
        out["rho_real"] = [out["P_lo"] / c2["P_hi"], out["P_hi"] / c2["P_lo"]]
        rec["nodes"][wf] = out
        log(f"wf {wf}: real smallest peak / chord2's: {out['rho_real'][0]:.3f}..{out['rho_real'][1]:.3f}")
    log.dump(rec)
    return True


def main():
    argv = sys.argv[1:]
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    if "--selftest" in argv:
        sys.exit(0 if selftest(mut) else 1)
    argv = [a for a in argv if not a.startswith("--")]
    if len(argv) != 2 or argv[0] not in ("peak", "peak_real"):
        raise SystemExit(__doc__)
    stage, outdir = argv
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    os.makedirs(outdir, exist_ok=True)
    fn = {"peak": lambda: run_peak(outdir, mut), "peak_real": lambda: run_peak_real(outdir)}[stage]
    sys.exit(0 if fn() else 1)


if __name__ == "__main__":
    main()
