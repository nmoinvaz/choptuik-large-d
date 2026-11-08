"""Zero of S(x) = sum_n sigma_n x^n by continuation in the Laplace-conjugate plane, without Borel-Pade (issue #203).

Method.  The coefficients sigma_n fall like 1/n! (sigma_{n-1}/sigma_n grows linearly in n), so S is entire of exponential
type, and the series with c_n = n! sigma_n,
    F(t) = sum_n c_n t^n = int_0^inf e^{-s} S(s t) ds          (the Laplace transform of S at 1/t),
has a finite radius of convergence.  F is continued beyond its 23 known coefficients by an algebraic approximant, the
approximant's own Taylor coefficients c~_n (n <= N) replace the missing ones, and
    S~(x) = sum_{n <= N} c~_n x^n / n!                          (the inverse transform, term by term)
is summed until its terms are below 1e-30 of its largest one.  The zero x1 of S~ near the starting point follows by
Newton's method on the same series.

Approximants of F (both reproduce c_0..c_{L+M} or c_0..c_{p+q+r+1} of the data):
  qhp   quadratic Hermite-Pade, P F^2 + Q F + R = O(t^{p+q+r+2}) with deg P, Q, R = p, q, r; F~ is the branch of
        P y^2 + Q y + R = 0 through the data's series (Shafer, BIT 14, 293 (1974); Baker & Graves-Morris, Pade
        Approximants, 2nd ed., CUP 1996, ch. 8).  Primary.
  pade  linear Pade [L/M] of F in t (Baker & Graves-Morris ch. 1); the cross-check.
Neither is Borel-Pade: the transform multiplies by n! (Laplace plane) instead of dividing by n! (Borel plane), and the
qhp approximant is algebraic of degree 2, not rational.

Assumptions.  F has a finite radius, set by a singularity of the Laplace-conjugate function; the approximant's
singularities that matter for S~(x) at |x| ~ 13.5 are near the true ones; the coefficients beyond the data are those
of the approximant.  Approximants are admitted by the fixed rules of `admissible` (degenerate or with a singularity
well inside the data's disc of convergence are rejected).
"""
import json

import mpmath as mp

DPS = 80


def load_sigma(path):
    """sigma_0..sigma_K from a sigF_*.json's sigma_str (pairs of decimal strings)."""
    d = json.load(open(path))
    return [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in d["sigma_str"]], d


def laplace_coeffs(sig):
    return [s * mp.factorial(n) for n, s in enumerate(sig)]


def domb_sykes(c, n0):
    """r_n = c_n/c_{n-1} = z (1 + g/n + h/n^2) solved at n0, n0+1, n0+2; returns (z, g/z... as gamma)."""
    ns = [n0, n0 + 1, n0 + 2]
    A = mp.matrix([[1, mp.mpf(1) / n, mp.mpf(1) / n ** 2] for n in ns])
    b = mp.matrix([c[n] / c[n - 1] for n in ns])
    z, zg, _ = mp.lu_solve(A, b)
    return z, zg / z


def _nullvec(A):
    """right null vector of a complex m x (m+1) matrix, and the ratio of its two smallest singular values"""
    U, S, V = mp.svd_c(A, full_matrices=True)
    n = A.cols
    v = [mp.conj(V[n - 1, j]) for j in range(n)]
    s = sorted([S[i] for i in range(len(S))])
    return v, (s[0] / s[1] if len(s) > 1 and s[1] != 0 else mp.mpf(0))


def qhp_polys(c, p, q, r):
    """P, Q, R (coefficient lists, low order first) with [P F^2 + Q F + R]_k = 0 for k = 0..p+q+r+1."""
    m = p + q + r + 2
    assert m <= len(c), (p, q, r, len(c))
    F2 = [mp.fsum(c[i] * c[k - i] for i in range(k + 1)) for k in range(m)]
    A = mp.matrix(m, m + 1)
    for k in range(m):
        for i in range(p + 1):
            if k - i >= 0:
                A[k, i] = F2[k - i]
        for i in range(q + 1):
            if k - i >= 0:
                A[k, p + 1 + i] = c[k - i]
        if k <= r:
            A[k, p + q + 2 + k] = 1
    v, cond = _nullvec(A)
    P, Q, R = v[:p + 1], v[p + 1:p + q + 2], v[p + q + 2:]
    sc = max(abs(x) for x in v)
    return [x / sc for x in P], [x / sc for x in Q], [x / sc for x in R], cond


def qhp_extend(c, p, q, r, N):
    """Taylor coefficients f_0..f_N of the branch of P y^2 + Q y + R = 0 that starts at c_0."""
    P, Q, R, cond = qhp_polys(c, p, q, r)
    f = [c[0]]
    f2 = [c[0] * c[0]]                  # (f^2)_k for the k already known
    lead = 2 * P[0] * f[0] + Q[0]
    for n in range(1, N + 1):
        # [P f^2 + Q f + R]_n = 0 is linear in f_n, which enters only as lead * f_n
        s = P[0] * mp.fsum(f[j] * f[n - j] for j in range(1, n))
        for i in range(1, min(p, n) + 1):
            s += P[i] * f2[n - i]
        for i in range(1, min(q, n) + 1):
            s += Q[i] * f[n - i]
        if n <= r:
            s += R[n]
        f.append(-s / lead)
        f2.append(mp.fsum(f[j] * f[n - j] for j in range(n + 1)))
    return f, {"cond": cond, "lead": abs(lead) / max(abs(x) for x in P + Q), "P": P, "Q": Q, "R": R}


def pade_extend(c, L, M, N):
    p, qq = mp.pade(c[:L + M + 1], L, M)
    f = list(c[:L + M + 1])
    for n in range(L + M + 1, N + 1):
        f.append(-mp.fsum(qq[k] * f[n - k] for k in range(1, M + 1)) / qq[0])
    return f, {"P": p, "Q": qq}


def series_S(ct, x):
    """S~(x) = sum c~_n x^n/n! and S~'(x), with the largest |term| and the last |term|"""
    s = ds = mp.mpc(0)
    big = mp.mpf(0)
    tn = mp.mpc(1)          # x^n/n!
    last = mp.mpf(0)
    for n, cn in enumerate(ct):
        if n > 0:
            ds += cn * tn * n / x
        term = cn * tn
        s += term
        a = abs(term)
        big = max(big, a)
        last = a
        tn = tn * x / (n + 1)
    return s, ds, big, last


def find_zero(ct, x0, tol=mp.mpf("1e-25"), maxit=60):
    x = mp.mpc(x0)
    for it in range(maxit):
        s, ds, big, last = series_S(ct, x)
        dx = s / ds
        x -= dx
        if abs(dx) < tol * abs(x):
            s, ds, big, last = series_S(ct, x)
            return x, {"it": it + 1, "big": big, "last_rel": last / big, "converged": True}
    return x, {"it": maxit, "converged": False, "big": big, "last_rel": last / big}


def growth(ct, n):
    """|c~_n / c~_{n-1}| at n (the approximant's own growth rate, ~ 1/radius)"""
    return abs(ct[n] / ct[n - 1])


def admissible(c, ct, zeta_ds, x0, info, n_data):
    """Fixed admission rules (identical for planted and real data).
    A1 the approximant reproduces the data's c_n, n < n_data, to 1e-8 relative;
    A2 its growth rate |c~_N/c~_{N-1}| at N = len(ct)-1 is at most 1.25 |zeta_DS| (no singularity well inside the
       data's disc of convergence); zeta_DS is the data's Domb-Sykes estimate at n = 20..22;
    A3 its continued terms are not larger than 30 times the largest data term at |x| = |x0|
       (a spurious singularity with tiny weight but large |zeta| shows here first);
    A4 for qhp, the branch is regular at t = 0 (|2 P_0 c_0 + Q_0| >= 1e-12 of the largest coefficient).
    Returns (ok, reasons)."""
    why = []
    rep = max(abs(ct[n] - c[n]) / abs(c[n]) for n in range(n_data))
    if not rep <= mp.mpf("1e-8"):
        why.append(f"A1 rep {mp.nstr(rep, 3)}")
    g = growth(ct, len(ct) - 1)
    if not g <= mp.mpf("1.25") * abs(zeta_ds):
        why.append(f"A2 growth {mp.nstr(g, 4)}")
    X = abs(mp.mpf(x0))
    tn, bigd, bigt = mp.mpf(1), mp.mpf(0), mp.mpf(0)
    for n, cn in enumerate(ct):
        a = abs(cn) * tn
        if n < n_data:
            bigd = max(bigd, a)
        else:
            bigt = max(bigt, a)
        tn = tn * X / (n + 1)
    if not bigt <= 30 * bigd:
        why.append(f"A3 tail/data {mp.nstr(bigt / bigd, 3)}")
    if "lead" in info and not info["lead"] >= mp.mpf("1e-12"):
        why.append(f"A4 lead {mp.nstr(info['lead'], 3)}")
    return (not why), why


def qhp_triples(total, lo=5, hi=9):
    return [(p, q, total - p - q) for p in range(lo, hi + 1) for q in range(lo, hi + 1) if lo <= total - p - q <= hi]


def pade_pairs(total, lo=8):
    return [(L, total - L) for L in range(lo, total - lo + 1)]


def run_node(sig, x0=13.5, N=220, orders=(21, 20, 19), pade_orders=(22, 21, 20)):
    """All approximants at one node; returns a list of dicts (kind, degrees, order, x1, ok, why, diagnostics)."""
    mp.mp.dps = DPS
    c = laplace_coeffs(sig)
    zeta_ds, gamma_ds = domb_sykes(c, len(c) - 3)
    out = []
    for tot in orders:
        nd = tot + 2
        for (p, q, r) in qhp_triples(tot):
            try:
                ct, info = qhp_extend(c[:nd], p, q, r, N)
            except Exception as e:  # noqa: BLE001
                out.append({"kind": "qhp", "deg": [p, q, r], "nd": nd, "ok": False, "why": [f"error {e}"]})
                continue
            ok, why = admissible(c, ct, zeta_ds, x0, info, nd)
            rec = {"kind": "qhp", "deg": [p, q, r], "nd": nd, "ok": ok, "why": why, "cond": float(info["cond"]),
                   "growth": float(growth(ct, N))}
            if ok:
                x1, zi = find_zero(ct, x0)
                rec.update({"x1": [float(x1.real), float(x1.imag)], "x1_str": [mp.nstr(x1.real, 15), mp.nstr(x1.imag, 15)],
                            "newton": zi["converged"], "last_rel": float(zi["last_rel"])})
                if not zi["converged"] or abs(x1 - x0) > 1.5:
                    rec["ok"] = False
                    rec["why"] = rec["why"] + ["Z newton or far zero"]
            out.append(rec)
    for tot in pade_orders:
        nd = tot + 1
        for (L, M) in pade_pairs(tot):
            try:
                ct, info = pade_extend(c[:nd], L, M, N)
            except Exception as e:  # noqa: BLE001
                out.append({"kind": "pade", "deg": [L, M], "nd": nd, "ok": False, "why": [f"error {e}"]})
                continue
            ok, why = admissible(c, ct, zeta_ds, x0, info, nd)
            rec = {"kind": "pade", "deg": [L, M], "nd": nd, "ok": ok, "why": why, "growth": float(growth(ct, N))}
            if ok:
                x1, zi = find_zero(ct, x0)
                rec.update({"x1": [float(x1.real), float(x1.imag)], "x1_str": [mp.nstr(x1.real, 15), mp.nstr(x1.imag, 15)],
                            "newton": zi["converged"], "last_rel": float(zi["last_rel"])})
                if not zi["converged"] or abs(x1 - x0) > 1.5:
                    rec["ok"] = False
                    rec["why"] = rec["why"] + ["Z newton or far zero"]
            out.append(rec)
    return out, {"zeta_ds": [float(zeta_ds.real), float(zeta_ds.imag)], "gamma_ds": [float(gamma_ds.real), float(gamma_ds.imag)]}
