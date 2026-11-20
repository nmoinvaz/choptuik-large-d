#!/usr/bin/env python3
"""Eigenvalues of the l-multiplet's linearised operator near the tracked growth rate, to see an exchange (#1025).

An eigenvalue is a lam with det M(lam) = 0, M = DSSEll.mode_matrix (perturbations ~ exp(lam tau), Delta fixed). M is
real for real lam (complex step). For complex lam, DSSEll.mode_matrix_c forms it by central differences.
  1. Real axis (--dlam > 0): the sign of det M on a grid of lam (default 0.05 to max(3, lam1 + 1), step 0.1), each
     sign change refined by brentq. These are the real eigenvalues, the gauge mode (lam_gauge) included.
  2. Every eigenvalue inside the circle |lam - lam1| = r by Beyn's contour integral (Beyn 2012): with p real probe
     vectors V, A_k = (1/2 pi i) int lam^k M(lam)^-1 V dlam (k = 0, 1) by the trapezoid rule on --N points
     (offset from the real axis, so that M(conj lam) = conj M(lam) halves the solves). The rank of A_0 is the
     number of eigenvalues inside, and they are the eigenvalues of V0^H A_1 W0 S0^-1 (A_0 = V0 S0 W0^H truncated).
  3. Each one polished by successive linear problems (Ruhe 1973: lam <- lam + mu, mu the eigenvalue of
     M(lam) v = -mu M'(lam) v nearest 0, M' by central differences in lam), on the real axis with mode_matrix.
W.-J. Beyn, An integral method for solving nonlinear eigenvalue problems, Linear Algebra Appl. 436 (2012) 3839.
A. Ruhe, Algorithms for the nonlinear eigenvalue problem, SIAM J. Numer. Anal. 10 (1973) 674.

usage: python ell_eigs.py STATE D kc lam1 [--l 1] [--kappa 0.7] [--u0 0.04] [--Ks 30] [--r 1.2] [--N 48] [--p 10]
                          [--eps 1e-6] [--skip_b] [--no_polish_complex]
--eps is mode_matrix_c's relative differencing step. At k_c 47, D 17.5, check (a) gives 2.1e-6 at 1e-7 (roundoff,
  ~1/eps), 1.9e-7 at 1e-6 and 1.0e-6 at 1e-5 (truncation, ~eps^2) (#1025, raw I1025-mac eigs_check_*.log).
--r 0 takes r = max(1, 1.25 (lam1 - lam_gauge)), so the gauge mode lies inside: Beyn must find it there.
--no_polish_complex leaves complex eigenvalues at Beyn's values; half_diff (the distance to the N/2 rule's nearest
  eigenvalue) is recorded for every eigenvalue as an error estimate.
                          [--lo 0.05] [--hi H] [--dlam 0.1] [--out FILE] [--check | --check_only] [--mutate=step]
       python ell_eigs.py planted [--mutate=beyn]
--check, on the state, before the rest (--check_only: the checks and the timings alone):
  (a) mode_matrix_c at real lam1 against mode_matrix: every column within 1e-6 of the column's max;
  (b) mode_matrix_c(conj lam) = conj mode_matrix_c(lam) at lam = lam1 + 0.3i, relative 1e-12.
  --mutate=step scales one column's differencing denominator by 1.001, and (a) must fail (exit 1).
planted: the same contour and polish code on T(lam) = A - lam B - lam^2 C (n 40, seed 1), whose eigenvalues come
  from the companion linearisation, on a circle whose radius lies midway between two distinct eigenvalue distances:
  every true eigenvalue inside found after the polish (1e-10), none extra. Eigenvalues near the contour outside leak
  into A_0 at about (r/R)^N; they come out outside the circle and are listed, not counted. --mutate=beyn drops the lam weight of A_1,
  and the check must fail (exit 1).
       python ell_eigs.py planted_ill
planted_ill: the production contour (N 24, p 12) on T(lam) joined to m = 8 directions eps expm(lam K) diag(s), small
  at every lam (rel 1e-13 and 1e-17 of |T|, as M's smallest singular values at k_c 31 and its rcond at k_c 47-55),
  turning with lam and holding no eigenvalue, mixed by fixed random orthogonal P, Q; a circle with no true eigenvalue
  inside and one with one (radius a quarter of the gap); rel 1e-2 is the well-conditioned control. Pass: the polished
  eigenvalues inside, merged within 1e-8, are the true ones inside (1e-8). Beyn on M^-1 V fails it at 1e-13 and 1e-17 (#1025: winb's D 17.5 scans saturated A_0's rank).
The ARPACK start vector and the probes are fixed, so a rerun gives the same numbers.
Records (one per call): data/ell_eigs.jsonl, or --out.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.linalg as sl
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, eigs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mutate=")), "")


def nearest_mu(M, Mp, k=1):
    """the k eigenvalues mu of M v = -mu M' v nearest 0: shift-invert Arnoldi on -M^-1 M' (eigenvalues 1/mu)"""
    n = M.shape[0]
    lu = sl.lu_factor(M)
    dt = np.result_type(M, Mp)
    kk = max(k, 6)                                   # several at once: k = 1 is unreliable for close |nu|
    if kk >= n - 2:
        nu = np.linalg.eigvals(-sl.lu_solve(lu, Mp))
    else:
        op = LinearOperator((n, n), matvec=lambda x: -sl.lu_solve(lu, Mp @ x), dtype=dt)
        nu = eigs(op, k=kk, which="LM", return_eigenvectors=False, tol=1e-13, ncv=min(n, max(2 * kk + 1, 30)),
                  v0=np.ones(n, dtype=dt) / np.sqrt(n))
    mu = 1 / nu
    return mu[np.argsort(np.abs(mu))][:k]


def ruhe(Mf, lam0, tol=1e-9, maxit=15, real=False, dl=1e-4):
    """successive linear problems from lam0; Mf(lam) -> M (real lam when real)"""
    lam, hist = lam0, []
    for it in range(maxit):
        try:
            mu = nearest_mu(Mf(lam), (Mf(lam + dl) - Mf(lam - dl)) / (2 * dl))[0]
        except Exception as e:                        # ARPACK breakdown: reported as not converged
            print(f"   polish from {lam0}: {type(e).__name__} at iteration {it}", flush=True)
            return lam, it, hist + [float("inf")]
        if real:
            mu = mu.real
        lam = lam + mu
        hist.append(float(abs(mu)))
        if abs(mu) < tol * max(1.0, abs(lam)):
            return lam, it + 1, hist
    return lam, maxit, hist


LIFT_TAU = 1e-14                                      # below this times sigma_max a singular value's sign is not resolved


def floor_lift(Mref, tau=LIFT_TAU):
    """W = sigma_max U_f V_f^H from the singular directions of Mref below tau sigma_max (#1025: centre modes decaying
    outward, which the outer conditions see only below rounding). M + W keeps M's regular eigenvalues when the floor is
    exact and lifts the floor to sigma_max, so det(M + W) has a resolved sign. Mref is taken at a real reference lambda
    away from the roots, and W is fixed (M + W stays analytic in lambda). --mutate=nolift returns W = 0."""
    U, s, Vh = np.linalg.svd(Mref)
    k = int(np.sum(s < tau * s[0]))
    info = dict(k=k, sigma_max=float(s[0]), floor_top=float(s[-k] / s[0]) if k else None,
                above=float(s[-k - 1] / s[0]) if k < len(s) else None)
    if k == 0:
        return None, info
    W = s[0] * (U[:, -k:] @ Vh[-k:])
    return (0.0 * W if "nolift" in MUT else W), info


def beyn_eigs(A0, A1, rank_tol, abs_floor=0.0):
    U, s, Wh = np.linalg.svd(A0, full_matrices=False)
    k = int(np.sum(s > max(rank_tol * s[0], abs_floor))) if s[0] > 0 else 0
    if k == 0:
        return np.array([]), s
    return np.linalg.eigvals((U[:, :k].conj().T @ A1 @ Wh[:k].conj().T) / s[:k]), s


def beyn(Mf, c, r, N, p, rank_tol=1e-9, seed=7, half=False, abs_tol=1e-10):
    """eigenvalues of M inside |lam - c| = r (c real, M(conj lam) = conj M(lam)) and A_0's singular values; with
    half, also the eigenvalues from the rule on every other point (N/2, a free error estimate). The rank counts only
    singular values above abs_tol times the integrand's size (N max |X_j|), so an empty circle, where A_0 is
    quadrature noise, gives none."""
    V = None
    A = np.zeros(4, object)                           # A0, A1 on all points; A0, A1 on the even points
    xs = 0.0
    for j in range(N // 2):
        th = 2 * np.pi * (j + 0.5) / N                # upper half; the lower half by conjugation
        lam = c + r * np.exp(1j * th)
        M = Mf(lam)
        if V is None:
            V = np.random.default_rng(seed).standard_normal((M.shape[0], p))
        X = sl.solve(M, V) * (r * np.exp(1j * th) / N)
        xs = max(xs, float(np.linalg.norm(X, 2)))
        a0, a1 = 2 * X.real, 2 * (X if "beyn" in MUT else lam * X).real
        A[0] = A[0] + a0; A[1] = A[1] + a1
        if j % 2 == 0:
            A[2] = A[2] + 2 * a0; A[3] = A[3] + 2 * a1
    fl = abs_tol * N * xs
    ev, s = beyn_eigs(A[0], A[1], rank_tol, fl)
    if not half:
        return ev, s
    return ev, s, beyn_eigs(A[2], A[3], rank_tol, fl)[0]


def polish(Mr, Mc, ev, c, r, nopolish_complex=False):
    """Ruhe-polish each Beyn eigenvalue within 1.5 r (complex ones in the upper half, their conjugates by symmetry)"""
    out = []
    for e in sorted(ev, key=lambda x: (abs(x - c), -x.imag)):
        real = abs(e.imag) < 1e-7 * max(1.0, abs(e))
        if not real and e.imag < 0:
            continue                                  # the conjugate of an upper-half one, added below
        if abs(e - c) > 1.5 * r or (not real and nopolish_complex):
            lam, it, h = e, 0, []
        else:
            lam, it, h = ruhe(Mr, float(e.real), real=True) if real else ruhe(Mc, complex(e))
        conv = (h[-1] < 1e-9 * max(1.0, abs(lam))) if h else None   # None: not polished
        rec = dict(beyn=[float(e.real), float(e.imag)], lam=[float(np.real(lam)), float(np.imag(lam))], its=it,
                   last_mu=h[-1] if h else None, moved=float(abs(lam - e)), polished=bool(h), converged=conv,
                   inside=bool(abs(lam - c) < r))
        out.append(rec)
        if not real:
            out.append(dict(rec, beyn=[rec["beyn"][0], -rec["beyn"][1]], lam=[rec["lam"][0], -rec["lam"][1]]))
    return out


def planted():
    rng = np.random.default_rng(1)
    n = 40
    A, B, C = (rng.standard_normal((n, n)) for _ in range(3))
    T = lambda lam: A - lam * B - lam ** 2 * C
    L0 = np.block([[np.zeros((n, n)), np.eye(n)], [A, -B]])
    L1 = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), C]])
    true = sl.eigvals(L0, L1)
    true = true[np.isfinite(true)]
    c = float(true[np.argmin(np.abs(true - 0.5))].real)
    d = np.unique(np.round(np.abs(true - c), 12))     # distinct distances (a conjugate pair shares one)
    r = 0.5 * (d[3] + d[4])                           # the contour midway between two of them
    inside = [t for t in true if abs(t - c) < r]
    ev, s = beyn(T, c, r, 64, 10)
    got = [g for g in polish(T, T, ev, c, r) if g["inside"]]   # leakage from outside lands outside
    good = True
    print(f"planted: circle |lam - {c:.6f}| = {r:.6f} holds {len(inside)} true eigenvalues; A_0's rank {len(ev)} "
          f"(singular values {', '.join(f'{v:.1e}' for v in s[:8])})" + (f"  [MUTATION {MUT}]" if MUT else ""))
    for g in got:
        lam = complex(*g["lam"])
        e = min(abs(lam - t) for t in inside) if inside else np.inf
        ok = e < 1e-10
        good &= ok
        print(f"planted: {complex(*g['beyn']):.8f} -> {lam:.10f}, nearest true eigenvalue inside {e:.1e} away "
              f"({'OK' if ok else 'BAD'})")
    found = sum(min((abs(complex(*g["lam"]) - t) for g in got), default=np.inf) < 1e-10 for t in inside)
    ok = found == len(inside) and len(got) == len(inside)
    good &= ok
    print(f"planted: {found} of {len(inside)} true eigenvalues inside found, {len(got)} reported "
          f"({'OK' if ok else 'BAD'})")
    return good


def ill_operator(rel, m=8, seed=2, coupled=False):
    """S(lam) = P [T0(lam) + E(lam)] Q (direct sum; T0 = A - lam B - lam^2 C as in planted, n0 40), E(lam) = eps
    expm(lam K) diag(s): m directions eps = rel |T0(0.5)|_2 small at every lam, turning with lam (K real skew), det E != 0
    everywhere; P, Q fixed random orthogonal. Decoupled: S itself. Coupled (the shooting structure of M): Z = [[T, F],
    [G, S + G T^-1 F]] with a well-conditioned chain block T = 3 I + 0.1 lam R (nb 60) and O(1) couplings F, G, so
    det Z = det T det S: the eigenvalues and the floor sit in the Schur complement S, and the floor's singular vectors
    carry chain components, as M's centre modes carry node 0 and 1 parts. Returns the operator and T0's eigenvalues."""
    rng = np.random.default_rng(1)
    n0 = 40
    A, B, C = (rng.standard_normal((n0, n0)) for _ in range(3))
    T0 = lambda lam: A - lam * B - lam ** 2 * C
    L0 = np.block([[np.zeros((n0, n0)), np.eye(n0)], [A, -B]])
    L1 = np.block([[np.eye(n0), np.zeros((n0, n0))], [np.zeros((n0, n0)), C]])
    true = sl.eigvals(L0, L1)
    true = true[np.isfinite(true)]
    g = np.random.default_rng(seed)
    K = g.standard_normal((m, m)); K = K - K.T
    s = np.exp(g.uniform(np.log(0.5), np.log(2.0), m))
    eps = rel * np.linalg.norm(T0(0.5), 2)
    n = n0 + m
    P = np.linalg.qr(g.standard_normal((n, n)))[0]
    Q = np.linalg.qr(g.standard_normal((n, n)))[0]
    nb = 60
    R = g.standard_normal((nb, nb)) / np.sqrt(nb)
    F, G = g.standard_normal((nb, n)), g.standard_normal((n, nb))

    def S(lam):
        Z = np.zeros((n, n), complex if np.iscomplexobj(lam) else float)
        Z[:n0, :n0] = T0(lam)
        Z[n0:, n0:] = eps * (sl.expm(lam * K) * s[None, :])
        return P @ Z @ Q
    if not coupled:
        return S, true

    def Zf(lam):
        T = 3 * np.eye(nb) + 0.1 * lam * R
        return np.block([[T, F], [G, S(lam) + G @ np.linalg.solve(T, F)]])
    return Zf, true


def planted_ill(N=24, p=12, lift=False):
    """the production contour (N 24, p 12) on ill_operator at rel 1e-2 (a well-conditioned control), 1e-13 (M's
    smallest singular values at k_c 31, D 13) and 1e-17 (rcond at k_c 47 and 55, D 17.5), decoupled and coupled, with
    no true eigenvalue inside the circle and with one, each circle a quarter of the gap to the nearest eigenvalue
    outside (leakage about 4^-24). Pass, on what the scan reports: after the polish, with values within 1e-8 merged,
    the eigenvalues inside are the true ones inside, each within 1e-8 (fixed before any fix, #1025, coordinator
    2026-11-20). The fixed scan reports whether each polish converged, and only converged ones count as eigenvalues.
    lift: the floor lift (floor_lift) from the operator at the real reference c + r."""
    allgood = True
    for coupled in (False, True):
        for rel in (1e-2, 1e-13, 1e-17):
            Tt, true = ill_operator(rel, coupled=coupled)
            re = np.sort(true[np.abs(true.imag) < 1e-12].real)
            c1 = re[np.argmin(np.abs(re - 0.5))]
            dd = np.sort(np.abs(true - c1))
            g = np.linspace(-2, 2, 4001)                # the 'none' circle: the real point farthest from the spectrum
            dist = np.array([np.min(np.abs(true - x)) for x in g])
            c0 = g[np.argmax(dist)]
            for case, c, r in (("none", c0, 0.25 * dist.max()), ("one", c1, 0.25 * dd[1])):
                inside = [t for t in true if abs(t - c) < r]
                Mf, k = Tt, 0
                if lift:
                    W, info = floor_lift(Tt(c + r))
                    k = info["k"]
                    if W is not None:
                        Mf = (lambda W: (lambda lam: Tt(lam) + W))(W)
                ev, sv = beyn(Mf, c, r, N, p)
                got = []
                for q in polish(Mf, Mf, ev, c, r):
                    lq = complex(*q["lam"])
                    if q["inside"] and q["converged"] is not False and all(abs(lq - u) >= 1e-8 for u in got):
                        got.append(lq)
                found = sum(min((abs(u - t) for u in got), default=np.inf) < 1e-8 for t in inside)
                ok = found == len(inside) and len(got) == len(inside)
                allgood &= ok
                print(f"planted_ill: {'coupled' if coupled else 'decoupled'}, rel {rel:.0e}, {case}: circle "
                      f"|lam - {c:.6f}| = {r:.6f} holds {len(inside)}; lifted {k}; A_0's rank {len(ev)} of p {p} "
                      f"(singular values {sv[0]:.1e} .. {sv[-1]:.1e}); {found} found, {len(got)} reported inside "
                      f"({'OK' if ok else 'BAD'})" + (f"  [MUTATION {MUT}]" if MUT else ""), flush=True)
    print(f"planted_ill{' --lift' if lift else ''}: {'OK' if allgood else 'BAD'}")
    return allgood


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state"); ap.add_argument("D", type=float); ap.add_argument("kc", type=int)
    ap.add_argument("lam1", type=float)
    ap.add_argument("--l", type=int, default=1); ap.add_argument("--kappa", type=float, default=0.7)
    ap.add_argument("--u0", type=float, default=0.04); ap.add_argument("--Ks", type=int, default=30)
    ap.add_argument("--r", type=float, default=1.2); ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--lo", type=float, default=0.05); ap.add_argument("--hi", type=float, default=None)
    ap.add_argument("--dlam", type=float, default=0.1); ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--check_only", action="store_true")
    ap.add_argument("--eps", type=float, default=1e-6); ap.add_argument("--skip_b", action="store_true")
    ap.add_argument("--no_polish_complex", action="store_true")
    ap.add_argument("--mutate", default="")
    a = ap.parse_args()
    from dss_ell import DSSEll, NODES_STD
    z = np.load(a.state)
    S = DSSEll(kc=a.kc, nodes=NODES_STD, n=a.D - 2, ell=a.l, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0)
    S.freeze_from(z, a.kappa)
    t0 = time.time()
    rec = dict(D=a.D, kc=a.kc, l=a.l, state=os.path.basename(a.state), lam1=a.lam1, lam_gauge=S.gauge_lambda(z),
               Delta=float(z[-1]), omega=float(2 * np.pi / z[-1]))
    Mr = lambda lam: S.mode_matrix(z, float(np.real(lam)))
    Mc = lambda lam: S.mode_matrix_c(z, lam, eps=a.eps)
    if a.check or a.check_only:
        tm = time.time()
        M0 = Mr(a.lam1)
        rec["t_mode_matrix"] = round(time.time() - tm, 1)
        sc = None
        if "step" in MUT:
            sc = np.ones(len(z)); sc[7] = 1.001
        tm = time.time()
        M1 = S.mode_matrix_c(z, a.lam1, eps=a.eps, step_scale=sc)
        rec["t_mode_matrix_c"] = round(time.time() - tm, 1)
        tm = time.time()
        sl.solve(M1 + np.eye(M1.shape[0]), np.ones((M1.shape[0], a.p)))   # timing only (M1 is singular)
        rec["t_solve_c"] = round(time.time() - tm, 1)
        print(f"n {M0.shape[0]}; seconds: mode_matrix {rec['t_mode_matrix']}, mode_matrix_c {rec['t_mode_matrix_c']}, "
              f"complex solve with {a.p} probes {rec['t_solve_c']}", flush=True)
        colmax = np.maximum(np.abs(M0).max(axis=0), 1e-300)
        ea = float((np.abs(M1 - M0).max(axis=0) / colmax).max())
        lc = a.lam1 + 0.3j
        if a.skip_b:
            eb = 0.0
        else:
            Ma, Mb = Mc(lc), Mc(np.conj(lc))
            eb = float(np.abs(Mb - np.conj(Ma)).max() / np.abs(Ma).max())
        oka, okb = ea < 1e-6, eb < 1e-12
        print(f"check (a) mode_matrix_c vs mode_matrix at lam1: max column-relative error {ea:.1e} (bar 1e-6) "
              f"{'OK' if oka else 'BAD'}" + (f"  [MUTATION {MUT}]" if MUT else ""), flush=True)
        print(f"check (b) conjugate symmetry at lam1 + 0.3i: " + ("skipped" if a.skip_b else
              f"{eb:.1e} (bar 1e-12) {'OK' if okb else 'BAD'}"), flush=True)
        ia = int(np.argmax(np.abs(M1 - M0).max(axis=0) / colmax))
        print(f"   worst column {ia}: max |M| there {colmax[ia]:.2e} (matrix max {np.abs(M0).max():.2e}), "
              f"eps {a.eps:g}", flush=True)
        rec.update(check_a=ea, check_b=eb)
        if not (oka and okb) or a.check_only:
            print(json.dumps(rec), flush=True)
            sys.exit(0 if oka and okb else 1)
    if a.dlam > 0:
        hi = a.hi if a.hi is not None else max(3.0, a.lam1 + 1.0)
        grid = np.arange(a.lo, hi + 1e-12, a.dlam)
        cache = {}

        def sd(lam):
            if lam not in cache:
                cache[lam] = np.linalg.slogdet(Mr(lam))
            return cache[lam]
        sg = [sd(l)[0] for l in grid]
        roots = []
        for i in range(len(grid) - 1):
            if sg[i] != sg[i + 1]:
                ref = 0.5 * (sd(grid[i])[1] + sd(grid[i + 1])[1])
                roots.append(brentq(lambda l: sd(l)[0] * np.exp(sd(l)[1] - ref), grid[i], grid[i + 1], xtol=1e-11,
                                    rtol=1e-13))
        rec.update(real_grid=[float(grid[0]), float(grid[-1]), a.dlam], real_roots=[float(x) for x in roots],
                   t_real=round(time.time() - t0))
        print(f"D {a.D} k_c {a.kc}: real roots in [{grid[0]:.2f}, {grid[-1]:.2f}]: "
              + ", ".join(f"{x:.8f}" for x in roots) + f"  (lam_gauge {rec['lam_gauge']:.6f})", flush=True)
    t1 = time.time()
    r = a.r if a.r > 0 else max(1.0, 1.25 * (a.lam1 - rec["lam_gauge"]))   # --r 0: the gauge mode inside
    ev, s, evh = beyn(Mc, a.lam1, r, a.N, a.p, half=True)
    a.r = r
    rec.update(circle=[a.lam1, a.r, a.N, a.p], beyn_sv=[float(v) for v in s], t_beyn=round(time.time() - t1),
               beyn_half=[[float(e.real), float(e.imag)] for e in evh])
    print(f"   circle |lam - {a.lam1:.6f}| = {a.r}: {len(ev)} eigenvalues (A_0 singular values "
          + ", ".join(f"{v:.1e}" for v in s) + ")" + ("  [rank = p: raise --p]" if len(ev) == a.p else ""), flush=True)
    got = polish(Mr, Mc, ev, a.lam1, a.r, nopolish_complex=a.no_polish_complex)
    for g in got:                                     # distance to the N/2 rule's nearest eigenvalue
        g["half_diff"] = float(min((abs(complex(*g["beyn"]) - e) for e in evh), default=np.inf))
    rec.update(eigs=got, n_inside=sum(g["inside"] for g in got), seconds=round(time.time() - t0))
    for g in got:
        lm = "none" if g["last_mu"] is None else f"{g['last_mu']:.1e}"
        print(f"   {complex(*g['beyn']):.8f} -> {complex(*g['lam']):.10f} ({g['its']} its, last |mu| {lm}, "
              f"moved {g['moved']:.1e}){'' if g['inside'] else ', outside: leakage'}", flush=True)
    print(json.dumps(rec), flush=True)
    out = a.out or os.path.join(HERE, "data", "ell_eigs.jsonl")
    open(out, "a").write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "planted":
        sys.exit(0 if planted() else 1)
    if sys.argv[1] == "planted_ill":
        sys.exit(0 if planted_ill(lift="--lift" in sys.argv) else 1)
    main()
