#!/usr/bin/env python3
"""The rational generalization of arXiv:2611.07007v1 Eq. (16), fitted to the D = infinity tuned profile's harmonics
(notes/EEG_PROFILE.md section 8).

Model, in the paper's gauge with z = e^{i theta}:  bt = A_* Re[z P_L(z^2)/Q_N(z^2)],  Q_N(0) = 1.
Its positive-frequency harmonics are C_{2k+1} = (A_*/2) g_k, with g_k the Taylor coefficients of G(u) = P_L(u)/Q_N(u).
Eq. (16) is (L, N) = (0, 1), G = 1/(1 + w u). Its one-zero extension is (1, 1). The data are c_k = C_{2k+1}/C_1 with
C_nu = i^nu F_nu (notes/EEG_PROFILE.md 3.1), F_nu the tuned harmonics summed at A as in tuned_profile.py, at 60 digits.
A fit has L + N + 1 complex parameters, p_0 = A_* e^{i phi}/2C_1 included.

Fits, on k = 0..9 (nu <= 19):
  pade     [L/N] exact on c_0 .. c_{L+N} [Pade 1892]. L + N <= 8, so nu = 19 at least is out of sample.
  ls       least squares of the relative residuals (m_k - c_k)/c_k, every weight 1, so each harmonic counts alike.
           Levenberg-Marquardt from the pade fit [Levenberg 1944; Marquardt 1963].
  minimax  the largest relative residual minimized, Lawson's reweighted least squares from the ls fit [Lawson 1961].
Error: e_k = |m_k/c_k - 1|, complex, so it bounds the modulus and the phase. A fit reaches the target when
max_{k <= 9} e_k <= 1e-6.
Poles: roots u_p of Q_N, theta_p = arg(u_p)/2 (mod pi) - (i/2) ln|u_p|, the lower member of each conjugate pair.

Data precision. The f coefficients are doubles and the A series stops at K = 45. Its remainder, estimated as a
geometric tail of the last two terms (an estimator, not a bound), is 5e-7 of F_19 and 7e-8 of F_17. --tail adds it.

    python3 rational_fit.py [--A 2.72] [--tail] [--kcut 43] [--out FILE.json] [JSON ...]
    python3 rational_fit.py --plant=rational [--mutate=gauge|drop]
    python3 rational_fit.py --plant=branch | --plant=poles
    python3 rational_fit.py --mutate=gauge | --mutate=A                  (real data)

Checks (criteria fixed before the first run; exit 1 when one fails):
  real data
    D  the harmonics reproduce data/tuned_profile_A2.72.txt, |F_nu/F_1| to 2e-6 relative and arg to 2e-5, nu <= 25,
       for every input JSON. Control: --mutate=A sums at A = 2.7203, which must fail it.
    S  every fit with L + N >= 5 (pade, ls, minimax) has a pole within 1e-3, in Re and in Im, of the profile's nearest
       singularity 1.5344 - 1.4284i (notes/EEG_PROFILE.md 3.3, from r_19). Control: --mutate=gauge (C_nu = F_nu,
       the factor i^nu dropped) must fail it.
  --plant=rational: a planted [2/3], whose poles are 1.5344 - 1.4284i, 1.47 - 1.72i, 1.68 - 1.95i and zeros
    1.60 - 2.40i, 0.70 - 2.10i (paper's gauge), entered as f coefficients of the A series (rounded to doubles) and
    passed through the same loader, gauge map and fits, scanned for L + N <= 5.
    P1 for pade and for ls, no family with L + N <= 4 reaches 1e-6, and at L + N = 5 only (2, 3) does;
    P2 the (2, 3) fits' maximum error is at most 1e-12;
    P3 their poles and zeros agree with the planted ones to 1e-8 in theta.
    Controls: --mutate=gauge must fail P3 (and keep P1, P2); --mutate=drop (nu = 11 dropped, the higher harmonics
    moved down) must fail P1 or P2.
  --plant=branch, --plant=poles: the pole-march test. For the pade [n/n], n = 2, 3, 4, theta_2(n) is the pole second
    nearest to the real axis.
      converged: |theta_2(4) - theta_2(3)| < 0.02;
      marching:  |Im theta_2(2)| > |Im theta_2(3)| > |Im theta_2(4)| and |theta_2(4) - theta_2(3)| > 0.05.
    branch: G = (1 + a_1 u + a_2 u^2)/(1 - u/u_0) + b (1 - u/u_0) log(1 - u/u_0), a pole with a logarithm at the same
      point. It must be marching.
    poles: G = sum_j R_j/(1 - u/u_j), the same leading pole plus five simple poles, the nearest at 1.56 - 1.75i. It
      must be converged, not marching, with theta_2(4) within 0.02 of 1.56 - 1.75i.
    The real data's class is printed, not checked.

References: H. Pade, Ann. Sci. Ec. Norm. Sup. (3) 9 (1892), suppl. 1-93; K. Levenberg, Q. Appl. Math. 2 (1944) 164;
D. W. Marquardt, J. SIAM 11 (1963) 431; C. L. Lawson, PhD thesis, UCLA (1961); H. Stahl, J. Approx. Theory 91 (1997)
139 (the poles of Pade approximants to a function with branch points accumulate on a cut).
"""
import argparse
import json
import math
import os
import sys

import mpmath as mp

mp.mp.dps = 60
HERE = os.path.dirname(os.path.abspath(__file__))
AMP = os.path.join(HERE, "..", "inner_amp")
DEFAULT = [os.path.join(AMP, "scan45J1250", "sigF_w1.0000_K45_J1250_p1280.json"),
           os.path.join(AMP, "scan45J1500", "sigF_w1.0000_K45_J1500_p1280.json")]
DATAFILE = os.path.join(HERE, "data", "tuned_profile_A2.72.txt")
TARGET = mp.mpf("1e-6")
NFIT = 10                       # k = 0..9, nu <= 19
NMAX = 8                        # L + N <= 8 leaves nu = 19 out of sample for pade
KMAX = 12                       # nu <= 25 printed out of sample
THETA_REF = (1.5344, -1.4284)   # notes/EEG_PROFILE.md 3.3, nearest singularity from r_19, paper's gauge


# ---------------------------------------------------------------------------------------------------- data
def load_f(path):
    d = json.load(open(path))
    f = {}
    for key, (re_, im_) in d["f"].items():
        k, nu = map(int, key.split("_"))
        f[(k, nu)] = mp.mpc(re_, im_)
    return d["K"], d["J"], f


def harmonics(f, K, A, numax=25, kcut=None, tail=False):
    """F_nu = sum_k f[k_nu] A^k (as tuned_profile.py, at 60 digits) and the A series' remainder estimate per nu."""
    A = mp.mpf(A)
    Kc = kcut or K
    F, trunc = {1: mp.mpc(0, -0.5) * A}, {1: mp.mpf(0)}
    for nu in range(3, numax + 1, 2):
        t = [f.get((k, nu), 0) * A ** k for k in range(nu, Kc + 1, 2)]
        while len(t) > 1 and t[-1] == 0:
            t.pop()
        s = mp.fsum(t)
        if len(t) < 2:          # a single term (planted profiles): no remainder
            rem = mp.mpc(0)
        elif nu >= 11:          # smooth late terms: geometric tail t_n q/(1 - q), q = t_n/t_{n-1}
            q = t[-1] / t[-2]
            rem = t[-1] * q / (1 - q)
        else:                   # tails below 1e-10, dominated by late-order fit noise: the last term
            rem = t[-1]
        trunc[nu] = abs(rem) / abs(s)
        F[nu] = s + rem if tail and nu >= 11 else s
    return F, trunc


def coeffs(F, gauge=True, drop=None):
    """c_k = C_{2k+1}/C_1, C_nu = i^nu F_nu. gauge=False drops the factor i^nu; drop removes harmonic nu = drop."""
    nus = [nu for nu in sorted(F) if nu != drop]
    return [mp.mpc((1j) ** (nu - 1) if gauge else 1) * F[nu] / F[1] for nu in nus]


# ---------------------------------------------------------------------------------------------------- series
def s_mul(a, b, n):
    return [mp.fsum([a[i] * b[k - i] for i in range(k + 1) if i < len(a) and k - i < len(b)]) for k in range(n)]


def s_inv(q, n):
    s = []
    for k in range(n):
        s.append((1 if k == 0 else 0) - mp.fsum([q[j] * s[k - j] for j in range(1, min(k, len(q) - 1) + 1)]))
    return s


def expand(p, q, n):
    S = s_inv(q, n)
    return s_mul(p, S, n), S


# ---------------------------------------------------------------------------------------------------- fits
def pade(c, L, N):
    if N == 0:
        q = [mp.mpc(1)]
    else:
        M, rhs = mp.matrix(N, N), mp.matrix(N, 1)
        for i in range(N):
            k = L + 1 + i
            for j in range(1, N + 1):
                M[i, j - 1] = c[k - j] if k - j >= 0 else 0
            rhs[i] = -c[k]
        sol = mp.lu_solve(M, rhs)
        q = [mp.mpc(1)] + [sol[j] for j in range(N)]
    p = [mp.fsum([q[j] * c[i - j] for j in range(min(i, N) + 1)]) for i in range(L + 1)]
    return p, q


def ls_fit(c, L, N, p0, q0, wts=None, iters=80):
    """min sum_k wts_k^2 |m_k/c_k - 1|^2, k < NFIT. Complex Gauss-Newton with Levenberg-Marquardt damping."""
    ks = range(NFIT)
    wts = wts or [mp.mpf(1)] * NFIT
    w = [wts[k] / abs(c[k]) for k in ks]
    p, q, nx = list(p0), list(q0), L + 1 + N

    def resid(p, q):
        m, S = expand(p, q, NFIT)
        return [w[k] * (m[k] - c[k]) for k in ks], m, S

    r, m, S = resid(p, q)
    cost, lam = mp.fsum([abs(x) ** 2 for x in r]), mp.mpf("1e-6")
    for _ in range(iters):
        mS = s_mul(m, S, NFIT)
        J = mp.matrix(NFIT, nx)
        for k in ks:
            for l in range(L + 1):
                J[k, l] = w[k] * (S[k - l] if k >= l else 0)
            for j in range(1, N + 1):
                J[k, L + j] = -w[k] * (mS[k - j] if k >= j else 0)
        A, g = J.H * J, J.H * mp.matrix(r)
        step = False
        for _ in range(40):
            Ad = A.copy()
            for d in range(nx):
                Ad[d, d] = A[d, d] * (1 + lam) + mp.mpf("1e-80")
            dx = mp.lu_solve(Ad, -g)
            pn = [p[l] + dx[l] for l in range(L + 1)]
            qn = [q[0]] + [q[j] + dx[L + j] for j in range(1, N + 1)]
            rn, mn, Sn = resid(pn, qn)
            cn = mp.fsum([abs(x) ** 2 for x in rn])
            if cn < cost:
                gain = (cost - cn) / cost
                p, q, r, m, S, cost = pn, qn, rn, mn, Sn, cn
                lam, step = max(lam / 10, mp.mpf("1e-40")), True
                break
            lam *= 10
        if not step or gain < mp.mpf("1e-24"):
            break
    return p, q


def max_err(c, p, q, n=NFIT):
    m, _ = expand(p, q, n)
    return max(abs(m[k] / c[k] - 1) for k in range(n))


def minimax_fit(c, L, N, p, q, outer=40):
    """Lawson: weights v_k <- v_k e_k / sum v e, each step a weighted ls fit; the best iterate is kept."""
    best = (max_err(c, p, q), p, q)
    v = [mp.mpf(1) / NFIT] * NFIT
    for _ in range(outer):
        m, _ = expand(p, q, NFIT)
        e = [abs(m[k] / c[k] - 1) for k in range(NFIT)]
        s = mp.fsum([v[k] * e[k] for k in range(NFIT)])
        if s == 0:
            break
        v = [v[k] * e[k] / s for k in range(NFIT)]
        p, q = ls_fit(c, L, N, p, q, wts=[mp.sqrt(x) for x in v], iters=15)
        me = max_err(c, p, q)
        if me < best[0]:
            best = (me, p, q)
    return best[1], best[2]


# ---------------------------------------------------------------------------------------------------- poles
def to_theta(u):
    """theta (paper's gauge) with e^{2 i theta} = u: Re in [0, pi), Im = -(1/2) ln|u| (negative for |u| > 1)."""
    return complex(float(mp.arg(u) / 2 % mp.pi), float(-mp.log(abs(u)) / 2))


def roots(poly):
    if len(poly) < 2:
        return []
    return mp.polyroots(poly[::-1], maxsteps=400, extraprec=400)


def poles_zeros(p, q):
    """poles sorted by |Im theta|, each with its residue in theta relative to the nearest pole's, and the zeros."""
    P = lambda u: mp.polyval(p[::-1], u)
    dQ = [j * q[j] for j in range(1, len(q))]
    out = []
    for u in roots(q):
        Ru = P(u) / mp.polyval(dQ[::-1], u)
        out.append((to_theta(u), float(abs(Ru) / mp.sqrt(abs(u)))))
    out.sort(key=lambda t: abs(t[0].imag))
    if out:
        r0 = out[0][1]
        out = [(t, r / r0) for t, r in out]
    zs = sorted((to_theta(u) for u in roots(p)), key=lambda t: abs(t.imag))
    return out, zs


def fmt_t(t):
    return f"{t.real:.5f}{t.imag:+.5f}i"


# ---------------------------------------------------------------------------------------------------- scan
def scan(c, nmax, methods=("pade", "ls", "minimax")):
    """every (L, N) with 1 <= L + N <= nmax: fits, errors at k <= KMAX, poles."""
    res = {}
    for n in range(1, nmax + 1):
        for N in range(0, n + 1):
            L = n - N
            fits = {}
            try:
                p, q = pade(c, L, N)
            except ZeroDivisionError:
                continue
            fits["pade"] = (p, q)
            if "ls" in methods:
                fits["ls"] = ls_fit(c, L, N, p, q)
            if "minimax" in methods:
                fits["minimax"] = minimax_fit(c, L, N, *fits["ls"])
            for meth, (p, q) in fits.items():
                m, _ = expand(p, q, len(c))
                e = [abs(m[k] / c[k] - 1) for k in range(len(c))]
                res[(L, N, meth)] = dict(p=p, q=q, e=e, emax=max(e[:NFIT]), pz=poles_zeros(p, q))
    return res


def smallest(res, meth):
    """the smallest L + N at which some (L, N) reaches TARGET, and every (L, N) at that L + N that does."""
    for n in range(1, NMAX + 1):
        hit = [(L, N) for (L, N, m), v in res.items() if m == meth and L + N == n and v["emax"] <= TARGET]
        if hit:
            return n, sorted(hit)
    return None, []


def print_table(res, meth, nmax):
    print(f"\nmax_k<=9 |m_k/c_k - 1|, {meth} (rows L, columns N; '-' outside L + N <= {nmax})")
    print("  L\\N " + "".join(f"{N:>10d}" for N in range(nmax + 1)))
    for L in range(nmax + 1):
        row = []
        for N in range(nmax + 1):
            v = res.get((L, N, meth))
            row.append(f"{float(v['emax']):10.2e}" if v else "         -")
        print(f"  {L:3d} " + "".join(row))


def march(res):
    """theta_2(n) of pade [n/n], n = 2, 3, 4, and the class: converged, marching or neither."""
    t2 = []
    for n in (2, 3, 4):
        pz = res[(n, n, "pade")]["pz"][0]
        t2.append(pz[1][0])
    d43 = abs(t2[2] - t2[1])
    conv = d43 < 0.02
    mar = abs(t2[0].imag) > abs(t2[1].imag) > abs(t2[2].imag) and d43 > 0.05
    return t2, d43, conv, mar


# ---------------------------------------------------------------------------------------------------- plants
def plant_rational():
    """the planted [2/3] in u = e^{2 i theta}, G(0) = 1"""
    pol = [complex(1.5344, -1.4284), complex(1.47, -1.72), complex(1.68, -1.95)]
    zer = [complex(1.60, -2.40), complex(0.70, -2.10)]
    us = [mp.exp(2j * mp.mpc(t)) for t in pol]
    vs = [mp.exp(2j * mp.mpc(t)) for t in zer]
    q, p = [mp.mpc(1)], [mp.mpc(1)]
    for u in us:
        q = s_mul(q, [1, -1 / u], len(q) + 1)
    for v in vs:
        p = s_mul(p, [1, -1 / v], len(p) + 1)
    return p, q, pol, zer


def plant_branch(n):
    """(1 + a1 u + a2 u^2)/(1 - u/u0) + b (1 - u/u0) log(1 - u/u0): coefficients to order n"""
    u0 = mp.exp(2j * mp.mpc(1.53439, -1.428385))
    a1, a2, b = mp.mpc(-0.0040, 0.0021), mp.mpc(0.00010, -0.00005), mp.mpc(-0.012, 0.024)
    s = 1 / u0
    e = [mp.mpc(0), -s] + [s ** k / (k * (k - 1)) for k in range(2, n)]
    pol = [s ** k for k in range(n)]
    a = [mp.mpc(1), a1, a2]
    return [mp.fsum([a[i] * pol[k - i] for i in range(min(k, 2) + 1)]) + b * e[k] for k in range(n)]


def plant_poles(n):
    """sum_j R_j/(1 - u/u_j): the leading pole and five more, normalized to G(0) = 1"""
    th = [complex(1.53439, -1.428385), complex(1.56, -1.75), complex(1.50, -2.05), complex(1.62, -2.35),
          complex(1.45, -2.65), complex(1.66, -2.95)]
    R = [mp.mpc(1), mp.mpc(0.020, -0.010), mp.mpc(-0.015, 0.008), mp.mpc(0.010, 0.006), mp.mpc(-0.006, -0.004),
         mp.mpc(0.004, 0.002)]
    us = [mp.exp(2j * mp.mpc(t)) for t in th]
    g = [mp.fsum([R[j] * us[j] ** (-k) for j in range(len(th))]) for k in range(n)]
    return [x / g[0] for x in g], th[1]


def planted_f(c, A, K):
    """f coefficients (doubles) of an A series whose sum at A is the profile C_nu = C_1 c_k, F_nu = (-i)^{nu-1} ..."""
    A = mp.mpf(A)
    F1 = mp.mpc(0, -0.5) * A
    f = {}
    for k, ck in enumerate(c):
        nu = 2 * k + 1
        if nu == 1:
            continue
        Fnu = ck * F1 / (1j) ** (nu - 1)
        x = complex(Fnu / A ** nu)
        f[(nu, nu)] = mp.mpc(x.real, x.imag)
    return f


# ---------------------------------------------------------------------------------------------------- report
def report_fits(res, c, trunc, label):
    print(f"\n== {label}")
    print(" nu   |c_k|            arg c_k          A-series remainder estimate (relative)")
    for k in range(len(c)):
        nu = 2 * k + 1
        print(f" {nu:2d}   {mp.nstr(abs(c[k]), 12):16s} {mp.nstr(mp.arg(c[k]), 12):16s} {float(trunc.get(nu, 0)):.1e}")
    for meth in ("pade", "ls", "minimax"):
        if any(m == meth for (_, _, m) in res):
            print_table(res, meth, NMAX)
    print("\nsmallest L + N reaching max error <= 1e-6 at every nu <= 19:")
    for meth in ("pade", "ls", "minimax"):
        if any(m == meth for (_, _, m) in res):
            n, hit = smallest(res, meth)
            print(f"  {meth:8s}: " + (f"L + N = {n} ({n + 1} complex parameters): {hit}" if n else
                                      f"none with L + N <= {NMAX}"))


def report_poles(res, nmin):
    print(f"\npoles (theta, paper's gauge, |residue| relative to the nearest pole's) and zeros, L + N >= {nmin}:")
    for (L, N, meth), v in sorted(res.items(), key=lambda t: (t[0][0] + t[0][1], t[0][1], t[0][2])):
        if L + N < nmin or N == 0:
            continue
        pz, zs = v["pz"]
        print(f"  [{L}/{N}] {meth:7s} max {float(v['emax']):.2e}  poles " +
              ", ".join(f"{fmt_t(t)} ({r:.1e})" for t, r in pz) + "  zeros " + ", ".join(fmt_t(t) for t in zs))


def check_datafile(paths, A):
    """D: |F_nu/F_1| and arg F_nu/F_1 against data/tuned_profile_A2.72.txt, for each input"""
    blocks = open(DATAFILE).read().split("\n== ")
    ok, worst = True, (0.0, 0.0)
    for path, blk in zip(paths, blocks):
        K, J, f = load_f(path)
        F, _ = harmonics(f, K, A)
        for line in blk.splitlines():
            t = line.split()
            if len(t) > 3 and t[0].isdigit() and t[1][0].isdigit() and "e" in t[1]:
                nu, mag, ang = int(t[0]), float(t[1]), float(t[2])
                x = F[nu] / F[1]
                dm, da = abs(float(abs(x)) / mag - 1), abs(float(mp.arg(x)) - ang)
                worst = (max(worst[0], dm), max(worst[1], da))
                ok &= dm < 2e-6 and da < 2e-5
    return ok, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="*")
    ap.add_argument("--A", default="2.72")
    ap.add_argument("--tail", action="store_true")
    ap.add_argument("--kcut", type=int, default=None)
    ap.add_argument("--plant", choices=["rational", "branch", "poles"], default=None)
    ap.add_argument("--mutate", choices=["gauge", "drop", "A"], default=None)
    ap.add_argument("--fast", action="store_true", help="pade and ls only (no minimax)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gauge = a.mutate != "gauge"
    drop = 11 if a.mutate == "drop" else None
    methods = ("pade", "ls") if (a.fast or a.plant) else ("pade", "ls", "minimax")
    fails = []

    def check(name, ok, msg):
        print(f"check {name}: {msg} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)

    if a.plant == "rational":
        p0, q0, pol, zer = plant_rational()
        c0, _ = expand(p0, q0, KMAX + 1 + (1 if drop else 0))
        f = planted_f(c0, a.A, 45)
        F, trunc = harmonics(f, 45, a.A, numax=2 * len(c0) - 1)
        c = coeffs(F, gauge, drop)[:KMAX + 1]
        res = scan(c, 5, methods)
        report_fits(res, c, trunc, "planted [2/3]" + (f", mutation {a.mutate}" if a.mutate else ""))
        report_poles(res, 4)
        for meth in ("pade", "ls"):
            low = [(L, N) for (L, N, m), v in res.items() if m == meth and L + N <= 4 and v["emax"] <= TARGET]
            at5 = sorted((L, N) for (L, N, m), v in res.items() if m == meth and L + N == 5 and v["emax"] <= TARGET)
            check(f"P1 {meth}", not low and at5 == [(2, 3)], f"L + N <= 4 reaching 1e-6: {low or 'none'}; at 5: {at5}")
            v = res[(2, 3, meth)]
            check(f"P2 {meth}", v["emax"] <= mp.mpf("1e-12"), f"[2/3] max error {float(v['emax']):.2e}")
            pz, zs = v["pz"]
            dp = max(min(abs(t - x) for t, _ in pz) for x in pol)
            dz = max(min(abs(t - x) for t in zs) for x in zer)
            check(f"P3 {meth}", dp < 1e-8 and dz < 1e-8, f"poles off by {dp:.1e}, zeros by {dz:.1e}")
    elif a.plant in ("branch", "poles"):
        if a.plant == "branch":
            c0 = plant_branch(KMAX + 1)
        else:
            c0, t2p = plant_poles(KMAX + 1)
        f = planted_f(c0, a.A, 45)
        F, trunc = harmonics(f, 45, a.A)
        c = coeffs(F, gauge, drop)[:KMAX + 1]
        res = scan(c, NMAX, ("pade",))
        report_fits(res, c, trunc, f"planted {a.plant}")
        report_poles(res, 4)
        t2, d43, conv, mar = march(res)
        print(f"\ntheta_2 of pade [n/n], n = 2, 3, 4: " + ", ".join(fmt_t(t) for t in t2) +
              f";  |theta_2(4) - theta_2(3)| = {d43:.3f}")
        if a.plant == "branch":
            check("M branch", mar, f"marching {mar}, converged {conv}")
        else:
            d = abs(t2[2] - t2p)
            check("M poles", conv and not mar and d < 0.02,
                  f"converged {conv}, marching {mar}, |theta_2(4) - planted| = {d:.3f}")
    else:
        paths = a.json or DEFAULT
        if a.mutate:
            paths = paths[:1]
        Ause = "2.7203" if a.mutate == "A" else a.A
        do_d = not a.json and (a.A == "2.72" or a.mutate == "A")
        ok, worst = check_datafile(paths, Ause) if do_d else (True, None)
        out = {}
        for i, path in enumerate(paths):
            K, J, f = load_f(path)
            F, trunc = harmonics(f, K, Ause, kcut=a.kcut, tail=a.tail)
            c = coeffs(F, gauge, drop)[:KMAX + 1]
            if i > 0 and a.mutate is None:      # the second J: only the spread of the errors and poles
                res2 = scan(c, NMAX, ("pade", "ls"))
                de = max(abs(res2[key]["emax"] - res[key]["emax"]) / res[key]["emax"] for key in res2)
                dp = max(abs(res2[key]["pz"][0][0][0] - res[key]["pz"][0][0][0]) for key in res2
                         if key[1] > 0 and key[0] + key[1] >= 5)
                print(f"\n{os.path.relpath(path, HERE)} (J {J}): largest relative change of any max error "
                      f"{float(de):.1e}, of a nearest pole (L + N >= 5) {dp:.1e}")
                continue
            res = scan(c, NMAX, methods)
            report_fits(res, c, trunc, f"{os.path.relpath(path, HERE)} (K {K}, J {J}), A = {Ause}" +
                        (", with the A-series tail estimate" if a.tail else "") +
                        (f", K cut at {a.kcut}" if a.kcut else "") + (f", mutation {a.mutate}" if a.mutate else ""))
            report_poles(res, 4)
            # out of sample: nu = 21..25 against the data's remainder estimate
            print("\nout of sample, e_k at nu = 21, 23, 25 (A-series remainder estimate of the data: " +
                  ", ".join(f"{float(trunc[nu]):.1e}" for nu in (21, 23, 25)) + "):")
            for key in sorted(res, key=lambda t: (t[0] + t[1], t[1], t[2])):
                v = res[key]
                if key[0] + key[1] >= 6 and v["emax"] <= 10 * TARGET:
                    print(f"  [{key[0]}/{key[1]}] {key[2]:7s} max(nu<=19) {float(v['emax']):.2e}   " +
                          " ".join(f"{float(x):.1e}" for x in v["e"][10:13]))
            # ratio differences: d_k = r_{k+1} - r_k, r_k = c_{k+1}/c_k; (k + 1.5)^4 |d_k| is flat for a k^-3 approach
            r = [c[k + 1] / c[k] for k in range(len(c) - 1)]
            print("\n(k + 1.5)^4 |r_{k+1} - r_k|, nu = 2k + 1 = 3 .. 17: " +
                  " ".join(f"{float(abs(r[k + 1] - r[k]) * (k + 1.5) ** 4):.5f}" for k in range(1, 9)))
            t2, d43, conv, mar = march(res)
            print(f"pole-march test: theta_2 of pade [n/n], n = 2, 3, 4: " + ", ".join(fmt_t(t) for t in t2) +
                  f";  |theta_2(4) - theta_2(3)| = {d43:.3f}; class: " +
                  ("marching" if mar else "converged" if conv else "neither"))
            near = [(key, v["pz"][0][0][0]) for key, v in res.items() if key[1] > 0 and key[0] + key[1] >= 5]
            re_ = [t.real for _, t in near]
            im_ = [t.imag for _, t in near]
            print(f"nearest pole over the {len(near)} fits with L + N >= 5: Re {min(re_):.5f} .. {max(re_):.5f}, "
                  f"Im {min(im_):.5f} .. {max(im_):.5f}")
            bad = [key for key, t in near if abs(t.real - THETA_REF[0]) > 1e-3 or abs(t.imag - THETA_REF[1]) > 1e-3]
            check("S", not bad, f"fits whose nearest pole is off 1.5344 - 1.4284i by more than 1e-3: "
                                f"{len(bad)} of {len(near)}")
            out = {f"{L}_{N}_{m}": dict(p=[mp.nstr(x, 30) for x in v["p"]], q=[mp.nstr(x, 30) for x in v["q"]],
                                         e=[float(x) for x in v["e"]], emax=float(v["emax"]),
                                         poles=[[t.real, t.imag, rr] for t, rr in v["pz"][0]],
                                         zeros=[[t.real, t.imag] for t in v["pz"][1]])
                   for (L, N, m), v in res.items()}
            out["_data"] = dict(path=os.path.relpath(path, HERE), A=Ause, tail=a.tail, kcut=a.kcut,
                                mutate=a.mutate, c=[mp.nstr(x, 30) for x in c],
                                trunc={str(k): float(v) for k, v in trunc.items()})
        if do_d:
            check("D", ok, f"against {os.path.relpath(DATAFILE, HERE)}: worst |F|/|F_file| - 1 = {worst[0]:.1e}, "
                           f"arg {worst[1]:.1e}")
        else:
            print("check D: skipped (it compares the default inputs at A = 2.72 with the data file)")
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(out, fh, indent=1)
            print(f"wrote {a.out}")
    print("\nall checks pass" if not fails else f"\nFAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
