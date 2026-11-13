#!/usr/bin/env python3
"""Twins both ways, twins of the real S_1 series, and the smallest twin by size (issue #669).

#548 and #670 (pade_chord_arc.py twin, pade_twin_count.py; notes/PADE_FAR_END.md 6.3, 7) fitted a chord twin, a density
on the straight chord P0 -> kappa_* with exactly two branch points, to the arc plant arcjump, and found that 19-20
Chebyshev terms reproduce its sigma_0..sigma_22 within the measured errors. Here the twin runs the other way and on the
real series (part 1), and the twin is minimised by size instead of term count (part 2).

Twin classes. Each is pref(x) L(x) with pref the planted zero factor and L(x) = int_0^1 w(s) (1-s)^ex
sum_{m<M} c_m T_m(2s-1) exp(x gamma(s)) gamma'(s) ds, ex = beta - 1, as the plants (pade_geometry.Plant).
  chord    gamma the chord P0 -> kappa_*, w = 1. Two branch points (the class of chord2). Exact Beta moments, #670.
  arc      gamma the cubic arc of planted_plants.Arc, w = 1, 2, 1, 2 on the quarters of s. Each step is a log branch
           point on the arc (the class of arcjump).
  smooth   the same arc with w = 1 (the class of the plant arc). Its density continues off the arc, so it has two branch
           points only. Context.
Arc moments are exact: gamma^k gamma' T_m(2s-1) is a polynomial in u = 1 - s, and int u^(ex+j) du is elementary on each
piece (arc_moments). The chord moments of pade_twin_count are exact too, so neither basis shares the plants' quadrature.

Part 1, twins (stage `twins`, plants, then `twins_real`). Weighted least squares (weights 1/err_k) of each twin class
against sigma_0..sigma_22. chi = max_k |sigma_k(twin) - sigma_k| / err_k, err_k = rel_k |sigma_k|, rel_k the measured
error (sigma_err_fit.py, as pade_geometry.noise_eps; floor 1e-30). M_min = smallest M with chi <= 1, M_lo = smallest M
with l2 <= sqrt(n) (no twin of fewer terms has chi <= 1), as #670. C1 compares chi at 150 and 250 digits at M_min - 1
and M_min. At M = n the twin interpolates and chi is rounding noise, so there C1 requires chi <= 1e-30 at both (#672).

Part 2, size (stage `size`, plants, then `size_real`). The size of a chord twin is N = ||c||_2 = (sum_m |c_m|^2)^(1/2),
the norm of its Chebyshev coefficients, at M_SIZE terms. With A the n x M matrix of the twin's columns and b the target,
rows weighted by 1/err_k, and rho = A c - b, the least-norm c with residual rho is A^H (A A^H)^(-1) (b + rho) when
M >= n. So the smallest twin solves
    N^2 = min over the polydisc |rho_k| <= 1 of (b + rho)^H (A A^H)^(-1) (b + rho),
a convex quadratic problem in n complex unknowns, solved by a log-barrier Newton method (Boyd and Vandenberghe 2004,
Convex Optimization, Cambridge University Press, ch. 11). N_hi is the norm of an explicit twin whose chi is recomputed
from c. N_lo = g(mu)^(1/2), the Lagrange dual at the barrier's multipliers, a lower bound for every mu >= 0 (weak
duality, ibid. ch. 5). Context: the ridge route (Hoerl and Kennard 1970, Technometrics 12, 55), c(lambda) minimising
||A c - b||^2 + lambda ||c||^2, whose smallest ||c|| with chi <= 1 is an upper bound; and the peak density and largest
|c_m| of the smallest twin.

    python3 pade_twin_size.py --selftest [--mutate=upoly|radius]
    python3 pade_twin_size.py twins OUTDIR [--mutate=arccol|chordcol] [--allow_unraw]
    python3 pade_twin_size.py twins_real OUTDIR [--mutate=c1dup] [--allow_unraw]   (reads OUTDIR/twins_plants.json)
    python3 pade_twin_size.py twins_jdiag OUTDIR [--allow_unraw]      (post-freeze diagnostic, #672, not judged)
    python3 pade_twin_size.py size OUTDIR [--mutate=scale] [--allow_unraw]
    python3 pade_twin_size.py size_real OUTDIR [--allow_unraw]        (reads OUTDIR/size_plants.json)

Exit 1 when a frozen control fails, or when a frozen rule forbids the real stage. Numerical evidence, not a bound.
"""
import json
import math
import os
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pade_twin_count as tc                                           # noqa: E402  (sets pg.KM = 31)

F, pg, pp = tc.F, tc.pg, tc.pp
N23 = 23                            # sigma_0..sigma_22, the measured orders
NMAX = tc.NMAX                      # 31: sigma_0..sigma_30 (orders 23-30 with extrapolated errors, #670)
NS_SIZE = (23, 31)
M_SIZE = 40                         # terms of the size twin (M >= n, so every residual vector is reachable)
M_CONV = 50                         # convergence context at wf 1.00
DPS_FIT, DPS_CHECK = 150, 250
M_CONTROL = 6                       # every plant has M = 6
CHI_CONTROL = 1e-30
C1_INTERP = 1e-30                   # C1 at M = n, where the twin interpolates (#672)
TOL_QP = 1e-10                      # barrier stops when n/t <= TOL_QP Q
PIECES = F.PIECES
FROZEN = {                          # written before the twins, size, twins_real and size_real stages were first run
    "classes": "chord (two branch points, chord2's class), arc (arcjump's stepped arc, w = 1,2,1,2, log branch points "
               "on the arc), smooth (the arc with w = 1, the plant arc's class, context)",
    "errors": "part 1: measured rel_k, k <= 22 (floor 1e-30), F = 1; part 2: #670's model (measured for k <= 22, "
              "log-linear extrapolation on k = 18..22 beyond), F = 1",
    "twin_size": "M_min (least squares, chi <= 1) and M_lo (l2 <= sqrt(n)); the smallest twin lies in [M_lo, M_min]",
    "C0_same_class": "chord2 in the chord basis, arcjump in the arc basis, the plant arc in the smooth basis, each at "
                     "M = 6, n = 23: chi <= 1e-30 at every node",
    "C1_precision": "at M_min - 1 and M_min of every scan, chi at 250 digits agrees with 150 digits within 1e-3 relative",
    "C1_amended_672": "#672, the coordinator's amendment, frozen before the rerun of twins and twins_real: at M = n, "
                      "C1 passes when chi <= 1e-30 at both 150 and 250 digits; below n it is unchanged. Why: at M = n the "
                      "twin is the interpolant and chi is rounding noise at both precisions, so their relative "
                      "difference is undefined. A check fit that mpmath finds singular counts as chi = inf",
    "mutation_C1_672": "--mutate=c1dup (twins_real only): column n - 1 of the 250-digit bases is replaced by column "
                       "n - 2, so the 250-digit fit cannot interpolate at M = n. It must make the amended C1 fire on "
                       "every scan with M_min = n and leave the scans with M_min < n passing; the run does not stop on "
                       "a firing, judges C1 only, and exits 1",
    "C2_reproduce": "the chord twin of arcjump at n = 23 has M_min = 19 at every node (#670); a failure is a "
                    "discrepancy, reported",
    "both_ways": "the arc twin of chord2 counts as a twin (not an interpolant) if M_min <= n - 1 = 22; its M_min and "
                 "gap n - M_min are reported per node",
    "mutation_twins": "--mutate=arccol (arc column m = 0, order 20, x (1 + 1e-8)) must make C0 arc fail at every node, "
                      "--mutate=chordcol the same for C0 chord; each run exits 1",
    "twins_real": "only if C0 and C1 pass on the plants; per node M_min and M_lo of the chord, arc and smooth twins of "
                  "the real sigma_0..sigma_22 (J1250); C1 on the real fits stops the work if it fires; a class is read "
                  "as preferred by term count only if its M_min is smaller by at least 3 at every node (reported); "
                  "J1500 at wf 1.00 is context",
    "size": "N = ||c||_2 of the chord twin with M_SIZE = 40 terms, minimised exactly subject to chi <= 1; bracket "
            "[N_lo, N_hi] from the barrier (explicit twin) and its dual; targets chord2, arcjump, arc; n = 23, 31; "
            "every node",
    "S_gap": "N_hi / N_lo - 1 <= 1e-4 for every target, n and node",
    "S_valid": "the explicit twin's chi recomputed at 250 digits with the 250-digit basis <= 1 + 1e-6, and N_lo "
               "recomputed at 250 digits with the same multipliers within 1e-6 relative",
    "S0_calibration": "chord2's smallest twin within 10% of chord2's own size N_own (its 6-term coefficients): "
                      "N_lo >= 0.9 N_own and N_hi <= 1.1 N_own at n = 23 and 31 and every node",
    "mutation_size": "--mutate=scale (chord2's sigma_k x 1.25, its errors kept) must make S0 fail at every n and node; "
                     "the run exits 1",
    "separation": "at n = 31, rho_hi = N_hi(arcjump) / N_lo(chord2) and rho_lo = N_lo(arcjump) / N_hi(chord2): if "
                  "rho_hi <= 3 at every node, no size prior separates them (the answer); if rho_lo > 3 at every node, a "
                  "size prior of factor 3 separates them; otherwise mixed. n = 23 is reported the same way",
    "size_real": "only if S_gap, S_valid and S0 pass and rho_lo > 3 at every node at n = 23: the real series' smallest "
                 "chord twin at n = 23, rho_real = N(real) / N(chord2) per node, read against 3; otherwise no real "
                 "size run",
    "context": "the ridge route's smallest ||c|| with chi <= 1, the peak density and largest |c_m| of each smallest twin, "
               "N at M = 50 (wf 1.00) against M = 40, and the plant arc's N; reported, not judged",
}


# ---------------------------------------------------------------- exact moments on the arc
def pmul(a, b):
    out = [mp.mpc(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def arc_poly_u(arc, mut=None):
    """gamma and dgamma/ds as polynomials in u = 1 - s (low order first):
    gamma = P0 + D s + s (1-s)(e0 + e1 s) = P1 + (E - D) u - (E + e1) u^2 + e1 u^3, E = e0 + e1, D = P1 - P0"""
    P0, P1 = mp.mpc(arc.P0), mp.mpc(arc.P1)
    D = P1 - P0
    e0, e1 = (mp.mpc(0), mp.mpc(0)) if arc.kind == "chord" else (mp.mpc(arc.e0), mp.mpc(arc.e1))
    E = e0 + e1
    g = [P1, E - D, -(E + e1), e1]
    dg = [D - E, 2 * (E + e1), -3 * e1]
    if mut == "upoly":
        dg[0] *= 1 + mp.mpf("1e-10")
    return g, dg


def arc_moments(arc, ex, pieces, M, K, mut=None):
    """mom[m][k] = 1/k! sum_pieces w int_a^b (1-s)^ex T_m(2s-1) gamma(s)^k gamma'(s) ds, m < M, k < K, exactly, at the
    working precision: in u = 1 - s the integrand is u^ex times a polynomial, and T_m(2s-1) = (-1)^m T_m(2u-1)"""
    g, dg = arc_poly_u(arc, mut)
    ex = mp.mpc(ex)
    T = tc.shifted_cheb(M)
    Tu = [[(-1) ** m * c for c in T[m]] for m in range(M)]
    P, gk = [], [mp.mpc(1)]
    for k in range(K):
        P.append([c / mp.factorial(k) for c in pmul(gk, dg)])
        gk = pmul(gk, g)
    deg = M - 1 + max(len(p) for p in P)
    I = [mp.mpc(0)] * (deg + 1)
    for a, b, w in pieces:
        lo, hi = 1 - mp.mpf(b), 1 - mp.mpf(a)
        for j in range(deg + 1):
            e = ex + j + 1
            I[j] += w * ((mp.power(hi, e) if hi > 0 else 0) - (mp.power(lo, e) if lo > 0 else 0)) / e
    J = [[mp.fsum(p[j] * I[i + j] for j in range(len(p))) for i in range(M)] for p in P]
    return [[mp.fsum(Tu[m][i] * J[k][i] for i in range(m + 1)) for k in range(K)] for m in range(M)]


class Cols:
    """twin columns (pref * mom_m)_k, m < M, k < K, at precision dps; pieces for the density's step weight"""

    def __init__(self, mom, pref, ex, dps, pieces):
        self.dps, self.ex, self.pieces = dps, ex, pieces
        K = len(mom[0])
        with mp.workdps(dps):
            pr = [mp.mpc(z) for z in pref]
            self.cols = [pp.polymul(pr, mom[m], K) for m in range(len(mom))]


def chord_cols(base, dps, M, K, mut=None):
    with mp.workdps(dps):
        b = Cols(tc.chord_moments(base.P0, base.P1, base.ex, M, K - 1), base.pref, base.ex, dps, PIECES["chord2"])
        if mut == "chordcol":
            b.cols[0][20] *= 1 + mp.mpf("1e-8")
    return b


def arc_cols(pl, pieces, dps, M, K, mut=None):
    with mp.workdps(dps):
        b = Cols(arc_moments(pl.arc, pl.ex, pieces, M, K), pl.pref, pl.ex, dps, pieces)
        if mut == "arccol" and pieces == PIECES["arcjump"]:
            b.cols[0][20] *= 1 + mp.mpf("1e-8")
    return b


# ---------------------------------------------------------------- least squares twins (part 1)
def density_peak(c, ex, pieces):
    """largest |w(s) (1-s)^ex sum c_m T_m(2s-1)| on s = j/200, j < 200"""
    out = 0.0
    cc = [complex(z) for z in c]
    for j in range(200):
        s = j / 200
        w = next(wt for a, b, wt in pieces if a <= s < b or (b == 1 and s == 1))
        v = sum(z * math.cos(m * math.acos(2 * s - 1)) for m, z in enumerate(cc)) * complex(1 - s) ** complex(ex)
        out = max(out, abs(w * v))
    return out


def row(basis, sig, rel, n, M):
    c, chi, l2 = tc.twin_fit(basis, sig, rel, n, M, 1.0)
    return {"M": M, "chi": chi, "l2": l2, "c_max": max(float(abs(z)) for z in c),
            "c_norm": float(mp.sqrt(mp.fsum(abs(z) ** 2 for z in c))),
            "density_max": density_peak(c, basis.ex, basis.pieces)}


def c1_row(M, n, chi, chi_check):
    """C1 on one precision row. Below n: 150 and 250 digits agree within 1e-3 relative. At M = n (amended, #672): the
    twin is the interpolant, chi is rounding noise, and both must be <= C1_INTERP"""
    rd = abs(chi_check - chi) / chi_check if math.isfinite(chi_check) and chi_check > 0 else math.inf
    if M == n:
        return {"rule": "interpolant", "rel_diff": rd, "pass": chi <= C1_INTERP and chi_check <= C1_INTERP}
    return {"rule": "relative", "rel_diff": rd, "pass": rd <= 1e-3}


def scan(basis, check, sig, rel, n, log, label):
    t0 = time.time()
    rows = [row(basis, sig, rel, n, M) for M in range(M_CONTROL, n + 1)]
    m_min, m_lo = tc.sizes(rows, n)
    prec = []
    for M in ((m_min - 1, m_min) if m_min else ()):
        r = next((x for x in rows if x["M"] == M), None)
        if r is None:
            continue
        try:
            chi_c = row(check, sig, rel, n, M)["chi"]
        except ValueError:          # mpmath's "matrix is numerically singular": the check cannot fit
            chi_c = math.inf
        prec.append({"M": M, "chi": r["chi"], "chi_check": chi_c, **c1_row(M, n, r["chi"], chi_c)})
    c1 = bool(prec) and all(p["pass"] for p in prec)
    at = next((x for x in rows if x["M"] == m_min), None)
    log(f"  {label}: M_min {m_min} M_lo {m_lo}; chi at M_min - 1, M_min: "
        + ", ".join(f"{x['chi']:.2e}" for x in rows if m_min and x["M"] in (m_min - 1, m_min))
        + (f"; at M_min max|c| {at['c_max']:.2e}, peak density {at['density_max']:.2f}" if at else "")
        + "; at 250 digits " + ", ".join(f"{p['chi_check']:.2e} ({p['rule']})" for p in prec)
        + f"; C1 {'OK' if c1 else 'BAD'} ({time.time() - t0:.0f}s)")
    return {"rows": rows, "M_min": m_min, "M_lo": m_lo, "precision": prec, "C1_pass": c1}


# ---------------------------------------------------------------- the smallest twin by size (part 2)
def weighted(basis, target, err, n, M):
    with mp.workdps(basis.dps):
        A, b = mp.matrix(n, M), mp.matrix(n, 1)
        for k in range(n):
            for m in range(M):
                A[k, m] = basis.cols[m][k] / err[k]
            b[k] = mp.mpc(target[k]) / err[k]
    return A, b


def herm_form(G, y):
    Gy = G * y
    return mp.re(mp.fsum(mp.conj(y[i]) * Gy[i] for i in range(y.rows)))


def dual_bound(G, b, mu, R2=1):
    """g(mu) = min_rho (b + rho)^H G (b + rho) + sum mu_k (|rho_k|^2 - R2): a lower bound on the polydisc minimum"""
    n = G.rows
    GD = G.copy()
    for i in range(n):
        GD[i, i] += mu[i]
    rho = mp.lu_solve(GD, -(G * b))
    return herm_form(G, b + rho) + mp.fsum(mu[i] * (abs(rho[i]) ** 2 - R2) for i in range(n))


def polydisc_min(G, b, radius=1, tol=TOL_QP, log=None):
    """min over |rho_k| <= radius of Q(rho) = (b + rho)^H G (b + rho), G Hermitian positive definite: log-barrier
    Newton in the real form x = (Re rho, Im rho), started at rho = 0, t x 10 per stage until n/t <= tol Q"""
    n = G.rows
    R2 = mp.mpf(radius) ** 2
    N2 = 2 * n
    GG = mp.matrix(N2, N2)
    for i in range(n):
        for j in range(n):
            g = G[i, j]
            GG[i, j] = GG[n + i, n + j] = mp.re(g)
            GG[i, n + j], GG[n + i, j] = -mp.im(g), mp.im(g)
    beta = mp.matrix(N2, 1)
    for i in range(n):
        beta[i], beta[n + i] = mp.re(b[i]), mp.im(b[i])

    def quad(x):
        y = beta + x
        Gy = GG * y
        return mp.fsum(y[i] * Gy[i] for i in range(N2)), Gy

    def slack(x):
        return [R2 - x[i] ** 2 - x[n + i] ** 2 for i in range(n)]

    x = mp.matrix(N2, 1)
    Q, _ = quad(x)
    t = n / Q
    stages = newton = 0
    while True:
        for _ in range(100):
            Q, Gy = quad(x)
            d = slack(x)
            grad = mp.matrix(N2, 1)
            H = GG * (2 * t)
            for i in range(n):
                p, q, di = x[i], x[n + i], d[i]
                grad[i] = 2 * t * Gy[i] + 2 * p / di
                grad[n + i] = 2 * t * Gy[n + i] + 2 * q / di
                H[i, i] += 2 / di + 4 * p * p / di ** 2
                H[n + i, n + i] += 2 / di + 4 * q * q / di ** 2
                H[i, n + i] += 4 * p * q / di ** 2
                H[n + i, i] += 4 * p * q / di ** 2
            dx = mp.lu_solve(H, -grad)
            dec = -mp.fsum(grad[i] * dx[i] for i in range(N2))
            newton += 1
            if dec <= mp.mpf("1e-24"):
                break
            F0 = t * Q - mp.fsum(mp.log(v) for v in d)
            s = mp.mpf(1)
            while True:
                xn = x + dx * s
                dn = slack(xn)
                if min(dn) > 0 and t * quad(xn)[0] - mp.fsum(mp.log(v) for v in dn) <= F0 - s * dec / 4:
                    break
                s /= 2
                if s < mp.mpf("1e-40"):
                    raise RuntimeError("polydisc_min: line search failed")
            x = xn
        stages += 1
        Q, _ = quad(x)
        if log:
            log(f"      stage {stages}: t {mp.nstr(t, 3)}, Q {mp.nstr(Q, 12)}, Newton steps {newton}")
        if n / t <= tol * Q:
            break
        t *= 10
    d = slack(x)
    mu = [1 / (t * v) for v in d]
    rho = mp.matrix([mp.mpc(x[i], x[n + i]) for i in range(n)])
    return {"rho": rho, "Q": Q, "mu": mu, "dual": dual_bound(G, b, mu, R2), "stages": stages, "newton": newton,
            "t": t}


def smallest_twin(basis, check, target, err, err_check, n, M, radius=1, log=None):
    """the least-norm chord twin with chi <= radius: N_lo, N_hi, the explicit twin c, and the checks at check.dps"""
    t0 = time.time()
    with mp.workdps(basis.dps):
        A, b = weighted(basis, target, err, n, M)
        G = mp.inverse(A * A.H)
        res = polydisc_min(G, b, radius)
        c = A.H * (G * (b + res["rho"]))
        r = A * c - b
        chi = float(max(abs(r[k]) for k in range(n)))
        n_hi = mp.sqrt(mp.fsum(abs(c[m]) ** 2 for m in range(M)))
        n_lo = mp.sqrt(max(res["dual"], 0))
        cl = [c[m] for m in range(M)]
    with mp.workdps(check.dps):
        A2, b2 = weighted(check, target, err_check, n, M)
        c2 = mp.matrix([mp.mpc(z) for z in cl])
        r2 = A2 * c2 - b2
        chi2 = float(max(abs(r2[k]) for k in range(n)))
        G2 = mp.inverse(A2 * A2.H)
        n_lo2 = mp.sqrt(max(dual_bound(G2, b2, [mp.mpf(v) for v in res["mu"]], mp.mpf(radius) ** 2), 0))
    out = {"n": n, "M": M, "N_lo": float(n_lo), "N_hi": float(n_hi), "gap": float(n_hi / n_lo - 1),
           "chi": chi, "chi_check": chi2, "N_lo_check": float(n_lo2), "N_lo_check_rel": float(abs(n_lo2 / n_lo - 1)),
           "stages": res["stages"], "newton": res["newton"], "c_max": max(float(abs(z)) for z in cl),
           "density_max": density_peak(cl, basis.ex, basis.pieces), "c_abs": [float(abs(z)) for z in cl],
           "active": sum(1 for k in range(n) if abs(r[k]) > 0.999), "seconds": time.time() - t0}
    if log:
        log(f"    smallest twin n {n} M {M}: N in [{out['N_lo']:.6e}, {out['N_hi']:.6e}] (gap {out['gap']:.1e}); "
            f"chi {chi:.6f} (250 digits {chi2:.6f}); active constraints {out['active']}; max|c| {out['c_max']:.3e}, "
            f"peak density {out['density_max']:.3f}; {res['stages']} stages, {res['newton']} Newton steps "
            f"({out['seconds']:.0f}s)")
    return out, cl


def ridge_route(basis, target, err, n, M):
    """context: c(lambda) = A^H (A A^H + lambda)^(-1) b; the smallest ||c|| on the path with chi <= 1 (first crossing of
    chi = 1 on a grid of one point per decade, then bisection in log lambda)"""
    with mp.workdps(basis.dps):
        A, b = weighted(basis, target, err, n, M)
        E, W = mp.eighe(A * A.H)
        beta = W.H * b
        Er = [mp.re(E[i]) for i in range(n)]

        def at(lam):
            v = mp.matrix([beta[i] / (Er[i] + lam) for i in range(n)])
            r = -lam * (W * v)
            return float(max(abs(r[k]) for k in range(n))), mp.sqrt(mp.fsum(Er[i] * abs(v[i]) ** 2 for i in range(n)))

        lo = int(math.floor(float(mp.log10(min(Er))))) - 4
        hi = int(math.ceil(float(mp.log10(max(Er))))) + 4
        last = None
        for j in range(lo, hi + 1):
            chi, nrm = at(mp.mpf(10) ** j)
            if chi > 1:
                break
            last = j
        if last is None:
            return {"N_ridge": None}
        a, z = mp.mpf(last), mp.mpf(last + 1)
        for _ in range(30):
            mid = (a + z) / 2
            if at(mp.mpf(10) ** mid)[0] <= 1:
                a = mid
            else:
                z = mid
        chi, nrm = at(mp.mpf(10) ** a)
        return {"N_ridge": float(nrm), "chi": chi, "log10_lambda": float(a), "eig_min": float(min(Er)),
                "eig_max": float(max(Er))}


# ---------------------------------------------------------------- selftest
def planted_qp(n, M, scales, mags, dps, radius=1):
    """A = diag(scales) U, U with orthonormal rows, so A A^H = diag(scales^2) and the polydisc problem separates:
    Q* = sum_k (|b_k| - 1)_+^2 / scales_k^2"""
    with mp.workdps(dps):
        import random
        rng = random.Random(669)
        Z = mp.matrix(M, n)
        for i in range(M):
            for j in range(n):
                Z[i, j] = mp.mpc(rng.gauss(0, 1), rng.gauss(0, 1))
        Qm, _ = mp.qr(Z)                   # M x M unitary; its first n columns are orthonormal
        U = mp.matrix(n, M)
        for i in range(n):
            for j in range(M):
                U[i, j] = mp.conj(Qm[j, i])
        A = mp.matrix(n, M)
        for i in range(n):
            for j in range(M):
                A[i, j] = scales[i] * U[i, j]
        b = mp.matrix([mags[k] * mp.expjpi(mp.mpf(k) / 7) for k in range(n)])
        exact = mp.fsum(max(mags[k] - 1, 0) ** 2 / scales[k] ** 2 for k in range(n))
        G = mp.inverse(A * A.H)
        res = polydisc_min(G, b, radius)
        c = A.H * (G * (b + res["rho"]))
        chi = max(abs((A * c - b)[k]) for k in range(n))
        return exact, res, chi


def selftest(mut=None):
    ok = True
    wf = 1.0
    c2, _ = tc.plant(wf, "chord2")
    aj, _ = tc.plant(wf, "arcjump")
    # 1: the u-polynomial moments on the chord against the exact Beta moments of pade_twin_count
    with mp.workdps(DPS_FIT):
        a = arc_moments(c2.arc, c2.ex, PIECES["chord2"], M_CONV, NMAX, mut)
        b = tc.chord_moments(c2.P0, c2.P1, c2.ex, M_CONV, NMAX - 1)
        e1 = max(abs(a[m][k] - b[m][k]) / abs(b[m][k]) for m in range(M_CONV) for k in range(NMAX))
    print(f"selftest 1: u-polynomial chord moments against the Beta moments, m < {M_CONV}, k < {NMAX}: max rel diff "
          f"{float(e1):.1e} (pass < 1e-90)")
    ok &= e1 < mp.mpf("1e-90")
    # 2: the exact stepped-arc moments against the plant's tanh-sinh moments
    mp.mp.dps = pg.DPS_PLANT
    qm = aj.moments(N23)
    with mp.workdps(DPS_FIT):
        ex = arc_moments(aj.arc, aj.ex, PIECES["arcjump"], N23, N23, mut)
        e2 = max(abs(ex[m][k] - qm[m][k]) / abs(ex[m][k]) for m in range(N23) for k in range(N23))
    print(f"selftest 2: exact stepped-arc moments against the plant's tanh-sinh moments, m, k < {N23}: max rel diff "
          f"{float(e2):.1e} (pass < 1e-50)")
    ok &= e2 < mp.mpf("1e-50")
    # 3: the polydisc minimum on a planted problem with a closed-form answer, eigenvalues of A A^H over 1e-40..1e40
    rad = mp.mpf("1.05") if mut == "radius" else 1
    n = 8
    scales = [mp.mpf(v) for v in ("1e-20", "1", "1e-12", "0.5", "1e12", "2", "1e20", "1e-3")]
    mags = [mp.mpf(v) for v in ("0.4", "2.5", "0.9", "1.7", "0.2", "3.1", "0.7", "1.0005")]
    exact, res, chi = planted_qp(n, 11, scales, mags, DPS_FIT, rad)
    rel = abs(res["Q"] - exact) / exact
    lo_ok = res["dual"] <= exact * (1 + mp.mpf("1e-30"))
    print(f"selftest 3: planted polydisc problem, Q* = {mp.nstr(exact, 10)}: barrier {mp.nstr(res['Q'], 10)}, dual "
          f"{mp.nstr(res['dual'], 10)}, rel diff {float(rel):.1e} (pass < 1e-8), dual <= Q* {lo_ok}, twin chi "
          f"{float(chi):.6f} (pass <= 1)")
    ok &= rel < 1e-8 and lo_ok and chi <= 1
    # 4: the same machinery on the real twin basis: a planted chord density's smallest twin is no larger than the
    #    planted density (it is itself a twin), and the bracket closes
    rel_k, _ = tc.err_model(wf)
    target = tc.planted_target(c2, 8)
    fit, chk = chord_cols(c2, DPS_FIT, M_SIZE, N23), chord_cols(c2, DPS_CHECK, M_SIZE, N23)
    with mp.workdps(DPS_FIT):
        own = mp.mpf(1) / abs(mp.fsum(mp.mpf("0.8") ** m * mp.expjpi(mp.mpf(m) / mp.pi) * c2.moments(8)[m][0]
                                      for m in range(8)))
        n_own = own * mp.sqrt(mp.fsum(mp.mpf("0.64") ** m for m in range(8)))
    err = tc.weights(target, rel_k, N23, 1.0, DPS_FIT)
    errc = tc.weights(target, rel_k, N23, 1.0, DPS_CHECK)
    out, _ = smallest_twin(fit, chk, target, err, errc, N23, M_SIZE, rad, log=print)
    good = out["N_hi"] <= float(n_own) * (1 + 1e-12) and out["gap"] <= 1e-4 and out["chi_check"] <= 1 + 1e-6
    print(f"selftest 4: planted 8-term chord density, own norm {float(n_own):.6e}: smallest twin N_hi "
          f"{out['N_hi']:.6e} (pass <= own), gap {out['gap']:.1e} (pass <= 1e-4), chi at 250 digits "
          f"{out['chi_check']:.6f} (pass <= 1)")
    ok &= good
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- stages
class Log:
    def __init__(self, outdir, name, mut):
        suf = f"_mutate_{mut}" if mut else ""
        self.f = open(os.path.join(outdir, f"{name}{suf}.log"), "a")
        self.json = os.path.join(outdir, f"{name}{suf}.json")
        self("run " + time.strftime("%Y-%m-%d %H:%M", time.gmtime()) + f" UTC, mutate {mut}")

    def __call__(self, s):
        print(s)
        self.f.write(s + "\n")
        sys.stdout.flush()
        self.f.flush()

    def dump(self, rec):
        json.dump(pg.jsonable(rec), open(self.json, "w"), indent=1)


def bases_part1(wf, dps, mut=None):
    c2, _ = tc.plant(wf, "chord2")
    aj, _ = tc.plant(wf, "arcjump")
    ar, _ = tc.plant(wf, "arc")
    out = {"chord": chord_cols(c2, dps, N23, N23, mut), "arc": arc_cols(aj, PIECES["arcjump"], dps, N23, N23, mut),
           "smooth": arc_cols(ar, PIECES["arc"], dps, N23, N23, mut)}
    if mut == "c1dup" and dps == DPS_CHECK:     # the 250-digit bases only; columns m < n - 1 are untouched
        for b in out.values():
            b.cols[N23 - 1] = list(b.cols[N23 - 2])
    return out


def run_twins(outdir, mut=None):
    log = Log(outdir, "twins_plants", mut)
    rec = {"frozen": FROZEN, "mutate": mut, "controls": {}, "scans": {}}
    ok = True
    for wf in pg.NODES:
        fit, chk = bases_part1(wf, DPS_FIT, mut), bases_part1(wf, DPS_CHECK, mut)
        rel, _ = tc.err_model(wf)
        sig = {k: tc.plant(wf, k)[1][:N23] for k in ("chord2", "arcjump", "arc")}
        for cls, kind in (("chord", "chord2"), ("arc", "arcjump"), ("smooth", "arc")):
            r = row(fit[cls], sig[kind], rel, N23, M_CONTROL)
            good = r["chi"] <= CHI_CONTROL
            ok &= good
            rec["controls"].setdefault(wf, {})[cls] = {**r, "pass": good}
            log(f"wf {wf}: C0 {cls} basis, {kind} at M 6: chi {r['chi']:.1e} {'OK' if good else 'BAD'}")
        if mut:                     # a mutation run judges C0 only
            continue
        for cls, kind in (("arc", "chord2"), ("smooth", "chord2"), ("chord", "arcjump")):
            s = scan(fit[cls], chk[cls], sig[kind], rel, N23, log, f"wf {wf} {cls} twin of {kind}")
            ok &= s["C1_pass"]
            rec["scans"].setdefault(wf, {})[f"{cls}_of_{kind}"] = s
    if mut:
        rec["controls_pass"] = ok
        log(f"controls C0 pass: {ok}")
        log.dump(rec)
        return ok
    c2r = {wf: rec["scans"][wf]["chord_of_arcjump"]["M_min"] for wf in pg.NODES}
    both = {wf: rec["scans"][wf]["arc_of_chord2"]["M_min"] for wf in pg.NODES}
    rec["C2_reproduce"] = {"M_min": c2r, "pass": all(v == 19 for v in c2r.values())}
    rec["both_ways"] = {"M_min": both, "gap": {wf: (N23 - v) if v else None for wf, v in both.items()},
                        "holds": all(v is not None and v <= N23 - 1 for v in both.values())}
    rec["controls_pass"] = ok
    log(f"C2 chord twin of arcjump M_min {c2r}: {'OK' if rec['C2_reproduce']['pass'] else 'DISCREPANCY'}")
    log(f"both ways: arc twin of chord2 M_min {both}, a twin at every node: {rec['both_ways']['holds']}")
    log(f"controls C0 and C1 pass: {ok}")
    log.dump(rec)
    return ok


def run_twins_real(outdir, mut=None):
    plants = json.load(open(os.path.join(outdir, "twins_plants.json")))
    if not plants["controls_pass"]:
        print("the plant controls failed: no real run (frozen rule)")
        return False
    log = Log(outdir, "twins_real", mut)
    rec = {"frozen": FROZEN, "mutate": mut, "scans": {}}
    ok = True
    real = tc.real_sig()
    for wf in pg.NODES:
        fit, chk = bases_part1(wf, DPS_FIT), bases_part1(wf, DPS_CHECK, mut)
        rel, _ = tc.err_model(wf)
        sig = real[wf][:N23]
        for cls in ("chord", "arc", "smooth"):
            s = scan(fit[cls], chk[cls], sig, rel, N23, log, f"wf {wf} {cls} twin of the real series (J1250)")
            ok &= s["C1_pass"]
            rec["scans"].setdefault(wf, {})[cls] = s
            if not s["C1_pass"] and not mut:
                log("C1 fired on real data: stop (frozen rule)")
                log.dump(rec)
                return False
        if mut:                     # a mutation run judges C1 only
            continue
        if wf == 1.0:
            j15 = pg.load_scan(pg.REAL["J1500"])[1.0][:N23]
            with mp.workdps(DPS_FIT):
                err = tc.weights(sig, rel, N23, 1.0, DPS_FIT)
                chij = max(float(abs(mp.mpc(j15[k]) - mp.mpc(sig[k])) / err[k]) for k in range(N23))
                perk = [float(abs(mp.mpc(j15[k]) - mp.mpc(sig[k])) / err[k]) for k in range(N23)]
            rec["J1500_vs_J1250_chi"] = {"chi": chij, "per_k": perk}
            log(f"  context: J1500 against J1250 at wf 1.00 in units of err_k: chi {chij:.2e}; per k "
                + " ".join(f"{v:.1e}" for v in perk))
            for cls in ("chord", "arc"):
                rec["scans"].setdefault("J1500", {})[cls] = scan(fit[cls], chk[cls], j15, rel, N23, log,
                                                                 f"wf 1.0 {cls} twin of J1500 (context)")
    mm = {wf: {cls: rec["scans"][wf][cls]["M_min"] for cls in ("chord", "arc", "smooth")} for wf in pg.NODES}
    if mut:
        at_n = [(wf, c) for wf in pg.NODES for c, s in rec["scans"][wf].items() if s["M_min"] == N23]
        below = [(wf, c) for wf in pg.NODES for c, s in rec["scans"][wf].items() if s["M_min"] != N23]
        fired = bool(at_n) and all(not rec["scans"][wf][c]["C1_pass"] for wf, c in at_n)
        isolated = all(rec["scans"][wf][c]["C1_pass"] for wf, c in below)
        rec.update({"M_min": mm, "C1_pass": ok, "mutation": {"scans_at_n": at_n, "fired_at_n": fired,
                                                            "below_n_pass": isolated}})
        log(f"mutation {mut}: scans with M_min = n {at_n}; amended C1 fired on each: {fired}; scans with M_min < n "
            f"still pass: {isolated}; C1 pass overall: {ok}")
        log.dump(rec)
        return ok
    pref = {cls: all(mm[wf][cls] is not None and mm[wf][o] is not None and mm[wf][o] - mm[wf][cls] >= 3
                     for wf in pg.NODES) for cls, o in (("chord", "arc"), ("arc", "chord"))}
    rec.update({"M_min": mm, "preferred_by_count": pref, "C1_pass": ok})
    log(f"real M_min per node {mm}; preferred by term count (>= 3 fewer at every node): {pref}")
    log.dump(rec)
    return ok


def run_twins_jdiag(outdir):
    """post-freeze diagnostic (#672), not judged, at wf 1.00. (a) #672's twins_real held the J1250 series at 53 bits;
    here it is read at 53 bits and from its strings at full precision, and the scans are repeated on both. (b) Where
    the misfit at M = n - 1 sits, per order. (c) The scans with err_k widened to max(err_k, |sigma_k(J1500) -
    sigma_k(J1250)|), both read at full precision"""
    log = Log(outdir, "twins_jdiag", None)
    rec = {"classes": {}}
    wf = 1.0
    s53 = pg.sl.load_scan(pg.REAL["J1250"], prec=53)[wf][:N23]
    with mp.workdps(DPS_FIT):
        sig = pg.load_scan(pg.REAL["J1250"])[wf][:N23]
        j15 = pg.load_scan(pg.REAL["J1500"])[wf][:N23]
    fn = [f for f in os.listdir(pg.REAL["J1250"]) if f.startswith(f"sigF_w{wf:.4f}")][0]
    strs = json.load(open(os.path.join(pg.REAL["J1250"], fn)))["sigma_str"][:N23]
    digits = [len(a.replace("-", "").replace(".", "").lstrip("0")) for a, _ in strs]
    rel, _ = tc.err_model(wf)
    fit = bases_part1(wf, DPS_FIT)
    with mp.workdps(DPS_FIT):
        err = tc.weights(sig, rel, N23, 1.0, DPS_FIT)
        r53 = [float(abs(mp.mpc(s53[k]) - sig[k]) / abs(sig[k])) for k in range(N23)]
        dj = [abs(j15[k] - sig[k]) for k in range(N23)]
        err_j = [max(err[k], dj[k]) for k in range(N23)]
    rec.update({"rel_53bit": r53, "string_digits_re": digits, "rel_k": rel[:N23],
                "J_diff_over_err": [float(dj[k] / err[k]) for k in range(N23)]})
    log(f"wf {wf}: J1250 as tc.real_sig() holds it against its strings, relative, per k: "
        + " ".join(f"{v:.1e}" for v in r53))
    log(f"  significant digits of Re sigma_k in the J1250 file: {digits}")
    log(f"  measured rel_k: " + " ".join(f"{v:.1e}" for v in rel[:N23]))
    log(f"  |J1500 - J1250| / err_k per k, both read at {DPS_FIT} digits: "
        + " ".join(f"{float(dj[k] / err[k]):.1e}" for k in range(N23)))
    for cls, b in fit.items():
        out = {}
        for lab, target, e in (("53bit", s53, err), ("strings", sig, err), ("widened", sig, err_j)):
            rows = []
            for M in range(M_CONTROL, N23 + 1):
                c, chi, l2 = tc.lsq(b, target, e, N23, M)
                rows.append({"M": M, "chi": chi, "l2": l2})
            m_min, m_lo = tc.sizes(rows, N23)
            out[lab] = {"rows": rows, "M_min": m_min, "M_lo": m_lo}
            log(f"  {cls}, {lab}: M_min {m_min}, M_lo {m_lo}; chi at M = 18..22: "
                + ", ".join(f"{r['chi']:.3e}" for r in rows if 18 <= r["M"] <= 22))
        c, chi, _ = tc.lsq(b, sig, err, N23, N23 - 1)
        with mp.workdps(DPS_FIT):
            per = [float(abs(mp.fsum(c[m] * b.cols[m][k] for m in range(N23 - 1)) - sig[k]) / err[k])
                   for k in range(N23)]
        out["per_k_at_n_minus_1"] = per
        rec["classes"][cls] = out
        log(f"  {cls}: misfit at M = {N23 - 1} per k, in units of err_k: " + " ".join(f"{v:.1e}" for v in per))
    log.dump(rec)
    return True


def size_targets(wf):
    return {k: tc.plant(wf, k) for k in ("chord2", "arcjump", "arc")}


def run_size(outdir, mut=None):
    log = Log(outdir, "size_plants", mut)
    rec = {"frozen": FROZEN, "mutate": mut, "nodes": {}}
    for wf in pg.NODES:
        pl = size_targets(wf)
        base = pl["chord2"][0]
        Mb = M_CONV if wf == 1.0 else M_SIZE
        fit, chk = chord_cols(base, DPS_FIT, Mb, NMAX), chord_cols(base, DPS_CHECK, Mb, NMAX)
        rel, info = tc.err_model(wf)
        node = rec["nodes"].setdefault(wf, {"errors": info})
        for n in NS_SIZE:
            sig_c2 = pl["chord2"][1][:n]
            own_c, own_chi, _ = tc.twin_fit(fit, sig_c2, rel, n, M_CONTROL, 1.0)
            n_own = float(mp.sqrt(mp.fsum(abs(z) ** 2 for z in own_c)))
            node.setdefault("own", {})[n] = {"N_own": n_own, "chi": own_chi, "c_max": max(float(abs(z)) for z in own_c),
                                             "density_max": density_peak(own_c, fit.ex, fit.pieces)}
            log(f"wf {wf} n {n}: chord2's own twin (M 6): N_own {n_own:.6e}, chi {own_chi:.1e}, max|c| "
                f"{node['own'][n]['c_max']:.2f}, peak density {node['own'][n]['density_max']:.3f}")
            for kind in ("chord2",) if mut else ("chord2", "arcjump", "arc"):     # a mutation run judges S0 only
                sig = pl[kind][1][:n]
                err = tc.weights(sig, rel, n, 1.0, DPS_FIT)
                errc = tc.weights(sig, rel, n, 1.0, DPS_CHECK)
                if mut == "scale" and kind == "chord2":
                    sig = [z * mp.mpf("1.25") for z in sig]
                log(f"  wf {wf} n {n} {kind}:")
                out, _ = smallest_twin(fit, chk, sig, err, errc, n, M_SIZE, log=log)
                node.setdefault("twins", {}).setdefault(kind, {})[n] = out
                if mut:
                    continue
                out["ridge"] = ridge_route(fit, sig, err, n, M_SIZE)
                rg = out["ridge"]
                log(f"    ridge route (context): smallest ||c|| with chi <= 1 on the path "
                    + (f"{rg['N_ridge']:.6e} = {rg['N_ridge'] / out['N_hi']:.3f} x N_hi (log10 lambda "
                       f"{rg['log10_lambda']:.2f}); eig(A A^H) {rg['eig_min']:.1e}..{rg['eig_max']:.1e}"
                       if rg["N_ridge"] else "none"))
                if wf == 1.0:
                    cv, _ = smallest_twin(fit, chk, sig, err, errc, n, M_CONV)
                    out["M50"] = {k: cv[k] for k in ("N_lo", "N_hi", "gap", "chi_check")}
                    out["M50"]["rel_change"] = cv["N_hi"] / out["N_hi"] - 1
                    log(f"    M {M_CONV} (context): N_hi {cv['N_hi']:.6e}, {out['M50']['rel_change']:+.2e} relative "
                        f"to M {M_SIZE}")
    # verdicts
    v = {"S_gap": True, "S_valid": True, "S0": {}, "separation": {}}
    for wf, node in rec["nodes"].items():
        for kind, per in node["twins"].items():
            for n, o in per.items():
                v["S_gap"] &= o["gap"] <= 1e-4
                v["S_valid"] &= o["chi_check"] <= 1 + 1e-6 and o["N_lo_check_rel"] <= 1e-6
    for n in NS_SIZE:
        s0, sep = {}, {}
        for wf, node in rec["nodes"].items():
            c2, own = node["twins"]["chord2"][n], node["own"][n]["N_own"]
            s0[wf] = {"lo_ratio": c2["N_lo"] / own, "hi_ratio": c2["N_hi"] / own,
                      "pass": c2["N_lo"] >= 0.9 * own and c2["N_hi"] <= 1.1 * own}
            if mut:
                continue
            aj = node["twins"]["arcjump"][n]
            sep[wf] = {"rho_hi": aj["N_hi"] / c2["N_lo"], "rho_lo": aj["N_lo"] / c2["N_hi"],
                       "arc_over_chord2": node["twins"]["arc"][n]["N_hi"] / c2["N_hi"]}
        v["S0"][n] = {"nodes": s0, "pass": all(x["pass"] for x in s0.values())}
        log(f"n {n}: S0 " + "; ".join(f"wf {wf} N/N_own {x['lo_ratio']:.4f}..{x['hi_ratio']:.4f} "
                                      f"{'OK' if x['pass'] else 'BAD'}" for wf, x in s0.items()))
        if mut:
            continue
        hi_ok = all(x["rho_hi"] <= 3 for x in sep.values())
        lo_ok = all(x["rho_lo"] > 3 for x in sep.values())
        v["separation"][n] = {"nodes": sep, "verdict": "no size prior separates" if hi_ok else
                              ("a size prior of factor 3 separates" if lo_ok else "mixed")}
        log(f"n {n}: separation " + "; ".join(f"wf {wf} rho {x['rho_lo']:.3f}..{x['rho_hi']:.3f}, arc/chord2 "
                                              f"{x['arc_over_chord2']:.3f}" for wf, x in sep.items())
            + f" -> {v['separation'][n]['verdict']}")
    v["controls_pass"] = v["S_gap"] and v["S_valid"] and all(v["S0"][n]["pass"] for n in NS_SIZE)
    log(f"S_gap {v['S_gap']}, S_valid {v['S_valid']}, S0 {[v['S0'][n]['pass'] for n in NS_SIZE]}; controls pass: "
        f"{v['controls_pass']}")
    rec["verdict"] = v
    log.dump(rec)
    return v["controls_pass"]


def run_size_real(outdir):
    plants = json.load(open(os.path.join(outdir, "size_plants.json")))
    v = plants["verdict"]
    sep23 = v["separation"]["23"]["nodes"]
    if not (v["controls_pass"] and all(x["rho_lo"] > 3 for x in sep23.values())):
        print("frozen rule: the size controls or the separation at n = 23 do not hold, so no real size run")
        return False
    log = Log(outdir, "size_real", None)
    rec = {"frozen": FROZEN, "nodes": {}}
    real = tc.real_sig()
    for wf in pg.NODES:
        base = tc.plant(wf, "chord2")[0]
        fit, chk = chord_cols(base, DPS_FIT, M_SIZE, N23), chord_cols(base, DPS_CHECK, M_SIZE, N23)
        rel, _ = tc.err_model(wf)
        sig = real[wf][:N23]
        out, _ = smallest_twin(fit, chk, sig, tc.weights(sig, rel, N23, 1.0, DPS_FIT),
                               tc.weights(sig, rel, N23, 1.0, DPS_CHECK), N23, M_SIZE, log=log)
        c2 = plants["nodes"][str(wf)]["twins"]["chord2"]["23"]
        out["rho_real"] = [out["N_lo"] / c2["N_hi"], out["N_hi"] / c2["N_lo"]]
        rec["nodes"][wf] = out
        log(f"wf {wf}: real smallest chord twin / chord2's: {out['rho_real'][0]:.3f}..{out['rho_real'][1]:.3f}")
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
    if len(argv) != 2 or argv[0] not in ("twins", "twins_real", "twins_jdiag", "size", "size_real"):
        raise SystemExit(__doc__)
    stage, outdir = argv
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    os.makedirs(outdir, exist_ok=True)
    fn = {"twins": lambda: run_twins(outdir, mut), "twins_real": lambda: run_twins_real(outdir, mut),
          "twins_jdiag": lambda: run_twins_jdiag(outdir),
          "size": lambda: run_size(outdir, mut), "size_real": lambda: run_size_real(outdir)}[stage]
    sys.exit(0 if fn() else 1)


if __name__ == "__main__":
    main()
