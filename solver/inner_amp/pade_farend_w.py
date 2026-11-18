#!/usr/bin/env python3
"""An error-weighted far-end estimator for the Borel support of S_1, recalibrated on #547's plants (issue #668).

Same question as pade_farend.py (#547): the farthest singularity of f(u) = sum_k a_k u^k, a_k = k! sigma_k, k <= 22,
from second-order inhomogeneous differential approximants (Guttmann and Joyce 1972, J. Phys. A 5, L81; Hunter and
Baker 1979, Phys. Rev. B 19, 3808). Here the approximant is fitted by weighted least squares: a member with degrees
(d0, d1, d2), Q_2(0) = 1 and P of degree lp has the equations
    R_n(q) = sum_j sum_i q_{j,i} c_j(n - i) = 0,   c_j(m) = (m+1)_j a_{m+j},   n = lp + 1 .. nmax,
(P absorbs the orders n <= lp), and the fit minimizes sum_n |R_n|^2 / V_n with V_n the error of equation n propagated
from the measured relative errors eps_k of sigma_k (sigma_err_fit.py via pade_geometry.noise_eps):
    V_n(q) = sum_{j,i} |q_{j,i}|^2 ((n - i + 1)_j s_{n-i+j})^2,   s_k = eps_k |a_k|.
V depends on q, so the fit is solved three times: with the proxy V_n = ((n+1)_2 s_{n+2})^2, then twice with V at the
previous solution (generalized least squares, reweighted). A square system is #547's exact solve. The window of an
admissible zero is (1.2, 3.0) rho / RHO_REF with rho = (|a_12| / |a_6|)^(1/6), so that the estimator is equivariant
under sigma_k -> c^k sigma_k. Method, presets, protocol and frozen criteria: notes/PADE_FAR_END.md section 19.
Prior art for least-squares Pade approximants: P. Gonnet, S. Guettel and L. N. Trefethen, SIAM Rev. 55, 101 (2013),
doi:10.1137/110853236 (record checked, paper not read).

    python3 pade_farend_w.py selftest [--mutate=...]
    python3 pade_farend_w.py timing                          (prints times only)
    python3 pade_farend_w.py develop OUTDIR [--allow_unraw]  (D1-D5, presets E1 and E2, chooses one)
    python3 pade_farend_w.py plants OUTDIR [--mutate=...] [--allow_unraw]
    python3 pade_farend_w.py real OUTDIR [--mutate=scale1.03|sigma7] [--allow_unraw]

Exit 0 all pass. 1 a check failed, or a mutation fired as it must. 2 a mutation did not fire. Error bars as in
pade_farend.py: noise is the largest shift of the estimate over NSEED draws of the measured per-order noise, spread
the median absolute deviation of the members, calibration the largest error on the plants. None is a bound.
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "hp"))
import pade_farend as pf                                               # noqa: E402
import pade_geometry as pg                                             # noqa: E402

N_COEF = 23
NMAX = 20                           # the last equation uses a_22
DPS = pf.DPS_DA
NSEED = 10
SEED_547 = 547000                   # the draws of #547's C3
SEED_FRESH = 668000                 # the fresh draws of C3'
RHO_REF = 1.2370                    # rho of the real series at wf 1.00 (J1250) is 1.23696
WINDOW = (1.2, 3.0)
FROZEN = {                          # written in notes/PADE_FAR_END.md 19.3 before any plant was judged
    "preset": None,                 # set after the development on D1-D5 (19.2), before the 30 validation plants
    "estimator": "median over the preset's members of the farthest zero of Q2 in (1.2, 3.0) rho / RHO_REF, Im t > 0, "
                 "from the weighted least-squares fit of 19.1",
    "criteria": "C1-C3, B1 and the real range as in pade_farend.FROZEN; C3' on the fresh draws 668000 + i",
    "mutation": "scale1.03 on the real series: 1.03 x the estimate within 1e-6, moved by more than 0.03",
}


# ---------------------------------------------------------------- the members
def members(preset):
    """[(degs, lp, nmax)] of a preset. E0: #547's members (square). E1, E2: 19.1."""
    out = []
    if preset == "E0":
        for degs, lp in pf.members(2):
            nq = sum(d + 1 for d in degs) - 1
            out.append((degs, lp, lp + nq))
        return out
    for d2 in {"E1": (4, 5, 6), "E2": (3, 4, 5)}[preset]:
        for drop in (0, 1):
            degs = [d2 - drop, d2 - drop, d2]
            nq = sum(d + 1 for d in degs) - 1
            for lp in (0, 2):
                if NMAX - lp >= nq:
                    out.append((degs, lp, NMAX))
    return out


def eqs(a, degs, lp, nmax):
    """rows n = lp+1..nmax, the columns (j, i) of the unknowns (q_{2,0} = 1 is the right side), A and b"""
    cols = [(j, i) for j, d in enumerate(degs) for i in range(d + 1) if not (j == 2 and i == 0)]
    rows = list(range(lp + 1, nmax + 1))
    A = mp.matrix(len(rows), len(cols))
    b = mp.matrix(len(rows), 1)
    for r, n in enumerate(rows):
        for c, (j, i) in enumerate(cols):
            m = n - i
            A[r, c] = mp.rf(m + 1, j) * a[m + j] if m >= 0 else 0
        b[r] = -mp.rf(n + 1, 2) * a[n + 2]
    return rows, cols, A, b


def row_var(rows, cols, degs, s, x):
    """V_n(q): the error of equation n propagated from the errors s_k of a_k, at the coefficients x (q_{2,0} = 1)"""
    q = {(2, 0): mp.mpf(1)}
    q.update({cj: x[c] for c, cj in enumerate(cols)})
    V = []
    for n in rows:
        v = mp.mpf(0)
        for (j, i), qq in q.items():
            m = n - i
            if m >= 0:
                v += abs(qq) ** 2 * (mp.rf(m + 1, j) * s[m + j]) ** 2
        V.append(v)
    return V


def qr_ls(A, b):
    """the least-squares solution of Ax = b (A m x n complex, m >= n) by Householder QR with the unitary complex
    reflector. mpmath's qr_solve is not used: it refuses a column whose squared norm is below the absolute eps, which a
    series consistent to 60 digits at 100 digits triggers."""
    m, n = A.rows, A.cols
    M = [[A[r, c] for c in range(n)] + [b[r]] for r in range(m)]
    for j in range(n):
        nrm = mp.sqrt(mp.fsum(abs(M[i][j]) ** 2 for i in range(j, m)))
        if nrm == 0:
            raise ValueError("zero column")
        x0 = M[j][j]
        alpha = -(x0 / abs(x0) if x0 != 0 else mp.mpf(1)) * nrm
        v = [M[i][j] for i in range(j, m)]
        v[0] = v[0] - alpha
        vv = mp.fsum(abs(t) ** 2 for t in v)
        if vv == 0:
            continue
        for k in range(j, n + 1):
            f = 2 * mp.fsum(mp.conj(v[i]) * M[j + i][k] for i in range(m - j)) / vv
            for i in range(m - j):
                M[j + i][k] -= f * v[i]
    x = [mp.mpc(0)] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - mp.fsum(M[i][k] * x[k] for k in range(i + 1, n))) / M[i][i]
    return mp.matrix(x)


def lsq(A, b, V):
    """the minimizer of sum_n |(Ax - b)_n|^2 / V_n (V None: unit weights). The columns are scaled to unit norm first, a
    reparametrization that leaves the minimizer alone."""
    As, bs = mp.matrix(A.rows, A.cols), mp.matrix(A.rows, 1)
    for r in range(A.rows):
        w = 1 if V is None else 1 / mp.sqrt(V[r])
        bs[r] = b[r] * w
        for c in range(A.cols):
            As[r, c] = A[r, c] * w
    cn = [mp.sqrt(mp.fsum(abs(As[r, c]) ** 2 for r in range(A.rows))) for c in range(A.cols)]
    for c in range(A.cols):
        for r in range(A.rows):
            As[r, c] = As[r, c] / cn[c]
    x = qr_ls(As, bs)
    return mp.matrix([x[c] / cn[c] for c in range(A.cols)])


def fit(a, s, degs, lp, nmax, mode="weighted", mut=None):
    """(rows, cols, x, V used in the last solve) or None when there are fewer equations than unknowns"""
    rows, cols, A, b = eqs(a, degs, lp, nmax)
    if A.rows < A.cols:
        return None
    if mut == "e0_row" and A.rows == A.cols:
        b[2] = b[2] * (1 + mp.mpf("1e-6"))
    if mode == "unweighted" or A.rows == A.cols:
        return rows, cols, lsq(A, b, None), None
    V = [(mp.rf(n + 1, 2) * s[n + 2]) ** 2 for n in rows]
    x = lsq(A, b, V)
    for _ in range(2):
        V = row_var(rows, cols, degs, s, x)
        x = lsq(A, b, V)
    return rows, cols, x, V


def zeros_of(degs, cols, x):
    """the zeros t = 1/u of Q_2 (as pade_farend.da_zeros)"""
    qK = [mp.mpc(1)] + [x[c] for c, (j, i) in enumerate(cols) if j == 2]
    return [1 / u for u in mp.polyroots(qK[::-1], maxsteps=400, extraprec=4 * mp.mp.prec) if u != 0]


# ---------------------------------------------------------------- the estimator
def weight_eps(eps, mut):
    """the errors the weights are built from (the noise itself is never changed here), as mpf"""
    e = [mp.mpf(float(x)) for x in eps[:N_COEF]]
    if mut == "err_reversed":
        return [e[0]] + [e[N_COEF - k] for k in range(1, N_COEF)]
    if mut == "err_x1e3_tail":
        return [x * 1000 if k >= 12 else x for k, x in enumerate(e)]
    if mut == "err_x1e3":
        return [x * 1000 for x in e]
    if mut == "err_flat":
        return [mp.mpf(0)] + [mp.mpf(pg.NOISE_REL)] * (N_COEF - 1)
    return e


def scale_of(a, mut=None):
    """rho / RHO_REF, the factor of the window (1 under the mutation abs_window)"""
    if mut == "abs_window":
        return mp.mpf(1)
    return (abs(a[12]) / abs(a[6])) ** (mp.mpf(1) / 6) / mp.mpf(str(RHO_REF))


def farthest_w(ts, scale):
    ok = [complex(t) for t in ts if WINDOW[0] * scale < abs(t) < WINDOW[1] * scale and mp.im(t) > 0]
    return max(ok, key=abs) if ok else None


def member_zeros(sig, eps, preset, mut=None):
    """[(degs, lp, zeros or None)] of every member of the preset, for the series sig (23 coefficients)"""
    mp.mp.dps = DPS
    sig = [mp.mpc(z) for z in sig[:N_COEF]]
    a = [mp.factorial(k) * sig[k] for k in range(N_COEF)]
    ew = weight_eps(eps, mut)
    s = [ew[k] * abs(a[k]) for k in range(N_COEF)]
    mode = "unweighted" if mut == "unweighted" else "weighted"
    out = []
    for degs, lp, nmax in members(preset):
        try:
            f = fit(a, s, degs, lp, nmax, mode, mut if preset == "E0" else None)
            ts = zeros_of(degs, f[1], f[2]) if f else None
        except (ZeroDivisionError, ValueError, mp.NoConvergence):
            ts = None
        out.append((degs, lp, ts))
    return out, scale_of(a, mut)


def estimate(sig, eps, preset, mut=None):
    """(median of the members' farthest admissible zero, [member records], MAD); (None, ..., None) if none admissible"""
    mz, scale = member_zeros(sig, eps, preset, mut)
    got = [{"degs": d, "lp": lp, "t": farthest_w(ts, scale) if ts else None} for d, lp, ts in mz]
    fs = np.array([g["t"] for g in got if g["t"] is not None])
    if len(fs) == 0:
        return None, got, None
    med = complex(np.median(fs.real), np.median(fs.imag))
    return med, got, float(np.median(np.abs(fs - med)))


_EPS = {}


def eps_of(wf):
    pg.NOISE_MODE = "measured"
    if wf not in _EPS:
        _EPS[wf] = pg.noise_eps(wf)
    return _EPS[wf]


def noise_shift(sig, wf, preset, mut=None, seed0=SEED_547, e0=None):
    """the largest shift of the estimate over NSEED draws at the measured errors (x 1000 under noise1000)"""
    eps = eps_of(wf)
    fac = 1000.0 if mut == "noise1000" else 1.0
    if e0 is None:
        e0 = estimate(sig, eps, preset, mut)[0]
    d = 0.0
    for i in range(NSEED):
        mp.mp.dps = DPS
        ns = pg.noisy([mp.mpc(z) for z in sig[:N_COEF]], seed0 + i, [x * fac for x in eps])
        en = estimate(ns, eps, preset, mut)[0]
        d = max(d, abs(en - e0) if (en is not None and e0 is not None) else float("inf"))
    return d


# ---------------------------------------------------------------- selftest
def _rel(za, zb):
    """largest relative distance from a zero of za to the nearest zero of zb (inf if the lengths differ)"""
    if len(za) != len(zb):
        return mp.inf
    return max(min(abs(x - y) for y in zb) / abs(x) for x in za)


def selftest(mut=None):
    """B-regression, S1 (solver against the normal equations), S2 (gauge), S3 (equivariance), S4 (planted exactness).
    Each has a mutation that must fire; mut breaks the check named by it, so the selftest then fails."""
    ok = True
    mp.mp.dps = DPS
    real = pg.load_scan(pg.REAL["J1250"])
    eps = eps_of(1.0)
    d1 = pf.FarPlant(1.0, "chord2", real[1.0], None).sigma()[:N_COEF]
    mp.mp.dps = DPS
    # B-regression: E0 through this code against pade_farend.da_zeros, member by member
    for lab, sig in (("plant D1", d1), ("real, wf 1.00", real[1.0][:N_COEF])):
        sig = [mp.mpc(z) for z in sig]
        worst = mp.mpf(0)
        for mutated in (None, "e0_row"):
            worst = 0
            mz, _ = member_zeros(sig, eps, "E0", mut if mut == "e0_row" and mutated is None else mutated)
            for (degs, lp, ts), (degs0, lp0) in zip(mz, pf.members(2)):
                worst = max(worst, _rel(ts, pf.da_zeros(sig, degs0, lp0)))
            if mutated is None:
                print(f"B-regression, {lab}: E0 zeros against da_zeros, largest relative difference {float(worst):.1e} "
                      f"(pass < 1e-30)")
                ok &= worst < mp.mpf("1e-30")
            else:
                print(f"B-regression mutation e0_row (one right side x (1 + 1e-6)): {float(worst):.1e} (must exceed "
                      f"1e-30)")
                ok &= worst > mp.mpf("1e-30")
    # S1: three E1 members, the QR solution against the normal equations at 250 digits with the same V. On D1 plus the
    # measured noise (draw 547000), where the weights matter: the noise-free plant is consistent to 60 digits
    mp.mp.dps = DPS
    dn = pg.noisy([mp.mpc(z) for z in d1], SEED_547, eps)
    sig = [mp.mpc(z) for z in dn]
    a = [mp.factorial(k) * sig[k] for k in range(N_COEF)]
    s = [mp.mpf(eps[k]) * abs(a[k]) for k in range(N_COEF)]
    mem = members("E1")
    for idx in (0, 5, 9):
        degs, lp, nmax = mem[idx]
        rows, cols, x, V = fit(a, s, degs, lp, nmax)
        for mutated in (False, True):
            mp.mp.dps = 250
            _, _, A, b = eqs(a, degs, lp, nmax)
            W = [1 / v for v in V]
            if mutated or (mut == "selftest_row"):
                W[3] = W[3] * mp.mpf("1.01")
            AH = mp.matrix(A.cols, A.rows)
            for r in range(A.rows):
                for c in range(A.cols):
                    AH[c, r] = mp.conj(A[r, c]) * W[r]
            xn = mp.lu_solve(AH * A, AH * b)
            rel = mp.norm(xn - x) / mp.norm(x)
            mp.mp.dps = DPS
            if not mutated:
                print(f"S1, member {degs} lp {lp} ({A.rows} equations, {A.cols} unknowns): QR against normal "
                      f"equations, relative difference {float(rel):.1e} (pass < 1e-60)")
                ok &= rel < mp.mpf("1e-60")
            else:
                print(f"S1 mutation (one weight x 1.01 in the normal equations): {float(rel):.1e} (must exceed 1e-60)")
                ok &= rel > mp.mpf("1e-60")
    # S2: a uniform scale of the errors changes no zero, a non-uniform one does
    base, _ = member_zeros(dn, eps, "E1")
    gm = "err_x1e3_tail" if mut == "err_x1e3_tail" else "err_x1e3"
    sc, _ = member_zeros(dn, eps, "E1", gm)
    worst = max(_rel(t1, t0) for (_, _, t1), (_, _, t0) in zip(sc, base))
    print(f"S2 gauge, errors x 1e3 in the weights: largest relative change of a member's zeros {float(worst):.1e} "
          f"(pass < 1e-30)")
    ok &= worst < mp.mpf("1e-30")
    tl, _ = member_zeros(dn, eps, "E1", "err_x1e3_tail")
    worst = max(_rel(t1, t0) for (_, _, t1), (_, _, t0) in zip(tl, base))
    print(f"S2 mutation (errors x 1e3 for k >= 12 only): {float(worst):.1e} (must exceed 1e-9)")
    ok &= worst > mp.mpf("1e-9")
    # S3: equivariance under sigma_k -> c^k sigma_k, on D1 and on the real series, c = 1.03 and 1.9
    for lab, sg in (("plant D1", d1), ("real, wf 1.00", real[1.0][:N_COEF])):
        sg = [mp.mpc(z) for z in sg]
        e0 = estimate(sg, eps, "E1")[0]
        for c in (1.03, 1.9):
            cs = mp.mpf(str(c))
            e1 = estimate([z * cs ** k for k, z in enumerate(sg)], eps, "E1",
                          "abs_window" if mut == "abs_window" else None)[0]
            dev = abs(e1 - c * e0) if (e1 is not None and e0 is not None) else float("inf")
            print(f"S3 equivariance, {lab}, c = {c}: |estimate(scaled) - c x estimate| = {dev:.1e} (pass < 1e-12)")
            ok &= dev < 1e-12
    sg = [mp.mpc(z) for z in real[1.0][:N_COEF]]
    e0 = estimate(sg, eps, "E1")[0]
    e1 = estimate([z * mp.mpf("1.9") ** k for k, z in enumerate(sg)], eps, "E1", "abs_window")[0]
    dev = abs(e1 - 1.9 * e0) if (e1 is not None and e0 is not None) else float("inf")
    print(f"S3 mutation abs_window (the window not scaled), real, c = 1.9: deviation {dev:.1e} (must exceed 1e-12)")
    ok &= dev > 1e-12
    # S4: a density with an exact low-degree equation, through an overdetermined E1-type member at lp = 4
    mp.mp.dps = 50
    P0, P1, ex = mp.mpc("0.749", "0.155"), mp.mpc("1.40", "1.20"), mp.mpc("2.5", "-1.2")
    sg = pf.chord_function(P0, P1, ex, lambda t: mp.exp(mp.mpc(-2, 1) * t))
    for mutated in (False, True):
        sm = list(sg)
        if mutated or mut == "selftest_sigma18":
            sm[18] *= mp.mpf("1.01")
        mp.mp.dps = DPS
        sm = [mp.mpc(z) for z in sm]
        am = [mp.factorial(k) * sm[k] for k in range(N_COEF)]
        sm_s = [mp.mpf(eps[k]) * abs(am[k]) for k in range(N_COEF)]
        f = fit(am, sm_s, [4, 4, 4], 4, NMAX)
        t = pf.farthest(zeros_of([4, 4, 4], f[1], f[2]))
        err = abs(t - complex(P1)) if t is not None else float("inf")
        if not mutated:
            print(f"S4 planted exactness, member (4,4,4) lp 4, {len(f[0])} equations, 14 unknowns: far end error "
                  f"{err:.1e} (pass < 1e-10)")
            ok &= err < 1e-10
        else:
            print(f"S4 mutation sigma_18 x 1.01: error {err:.1e} (must exceed 1e-10)")
            ok &= err > 1e-10
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- stages
def jsonable(o):
    return pg.jsonable(o)


def plant_record(name, real, preset, mut, with_fresh=True):
    kind, wf, R, dex, stretch, st = pf.PLANTS[name]
    t0 = time.time()
    pl = pf.FarPlant(wf, kind, real[wf], R, dex, stretch)
    sig = pl.sigma()
    mp.mp.dps = DPS
    P1 = complex(pl.P1)
    eps = eps_of(wf)
    est, got, spr = estimate(sig, eps, preset, mut)
    nz = noise_shift(sig, wf, preset, mut, SEED_547, est)
    nzf = noise_shift(sig, wf, preset, mut, SEED_FRESH, est) if with_fresh else None
    rl = pf.ratio12_21(sig)
    return {"kind": kind, "wf": wf, "R": float(abs(P1)), "dex": dex, "stretch": stretch, "set": st, "P1": P1,
            "M": pl.M, "fit_resid_max": max(pl.resid), "estimate": est,
            "error": abs(est - P1) if est is not None else None, "spread": spr, "noise": nz, "noise_fresh": nzf,
            "members": got, "ratio12_21": rl, "ratio_error": abs(rl - P1), "seconds": time.time() - t0}


def timing():
    real = pg.load_scan(pg.REAL["J1250"])
    for preset in ("E1", "E2"):
        t0 = time.time()
        pl = pf.FarPlant(1.0, "chord2", real[1.0], None)
        sig = pl.sigma()
        t1 = time.time()
        estimate(sig, eps_of(1.0), preset)
        t2 = time.time()
        print(f"{preset}: plant {t1 - t0:.1f} s, one estimate ({len(members(preset))} members) {t2 - t1:.1f} s")
    return True


def develop(outdir):
    """19.2 step 1: D1-D5 with E1 and E2, error and noise (draws 547000 + i), score = max of error/0.02, noise/0.01"""
    real = pg.load_scan(pg.REAL["J1250"])
    names = [n for n, v in pf.PLANTS.items() if v[5] == "D"]
    rec = {"names": names, "presets": {}}
    path = os.path.join(outdir, "development.json")
    for preset in ("E1", "E2"):
        rec["presets"][preset] = {}
        for name in names:
            r = plant_record(name, real, preset, None, with_fresh=False)
            sc = max(r["error"] / 0.02 if r["error"] is not None else float("inf"), r["noise"] / 0.01)
            r["score"] = sc
            rec["presets"][preset][name] = r
            print(f"{preset} {name:18s} est {r['estimate']} err {r['error']} spread {r['spread']:.4f} noise "
                  f"{r['noise']:.4f}  score {sc:.2f}  ({r['seconds']:.0f} s)")
            sys.stdout.flush()
            json.dump(jsonable(rec), open(path, "w"), indent=1)
    tot = {p: max(r["score"] for r in rec["presets"][p].values()) for p in rec["presets"]}
    chosen = "E1" if tot["E1"] <= tot["E2"] else "E2"
    rec["scores"], rec["chosen"] = tot, chosen
    json.dump(jsonable(rec), open(path, "w"), indent=1)
    print(f"development scores {tot}; chosen {chosen}" + ("; BOTH ABOVE 1" if min(tot.values()) > 1 else ""))
    return True


def run_plants(outdir, mut):
    preset = FROZEN["preset"]
    if preset is None:
        raise SystemExit("FROZEN['preset'] is not set: the development (19.2) comes first")
    real = pg.load_scan(pg.REAL["J1250"])
    rec = {"frozen": FROZEN, "preset": preset, "mutate": mut, "plants": {}}
    path = os.path.join(outdir, "plants" + (f"_mutate_{mut}" if mut else "") + ".json")
    for name in pf.PLANTS:
        r = plant_record(name, real, preset, mut)
        rec["plants"][name] = r
        print(f"{name:26s} |P1| {r['R']:.4f}  est {r['estimate']} err {r['error']}  spread {r['spread']}  noise "
              f"{r['noise']:.1e}  fresh {r['noise_fresh']:.1e}  | ratio12-21 err {r['ratio_error']:.4f} "
              f"(M {r['M']}, fit {r['fit_resid_max']:.1e}, {r['seconds']:.0f} s)")
        sys.stdout.flush()
        json.dump(jsonable(rec), open(path, "w"), indent=1)
    P = rec["plants"]
    inf = float("inf")
    c1 = {n: P[n]["error"] is not None and P[n]["error"] <= 0.02 for n in P}
    c2 = {f"{a} -> {b}": (abs(P[b]["estimate"] - P[a]["estimate"]) if P[a]["estimate"] is not None and
                          P[b]["estimate"] is not None else None) for a, b in pf.PAIRS}
    c3 = {n: P[n]["noise"] <= 0.01 for n in P}
    c3f = {n: P[n]["noise_fresh"] <= 0.01 for n in P}
    b1 = {n: P[n]["ratio_error"] > 0.05 for n in P if pf.PLANTS[n][2] is None and pf.PLANTS[n][3] == 0}
    errs = [P[n]["error"] if P[n]["error"] is not None else inf for n in P]
    v = {"C1": all(c1.values()), "C1_fail": [n for n in c1 if not c1[n]], "C1_max_error": max(errs),
         "C2": all(x is not None and x > 0.03 for x in c2.values()), "C2_moves": c2,
         "C3": all(c3.values()), "C3_fail": [n for n in c3 if not c3[n]], "C3_max_noise": max(P[n]["noise"] for n in P),
         "C3f": all(c3f.values()), "C3f_fail": [n for n in c3f if not c3f[n]],
         "C3f_max_noise": max(P[n]["noise_fresh"] for n in P),
         "B1": all(b1.values()), "B1_errors": {n: P[n]["ratio_error"] for n in b1}}
    rec["verdict"] = v
    json.dump(jsonable(rec), open(path, "w"), indent=1)
    print(f"C1 recovery <= 0.02: {v['C1']} (max error {v['C1_max_error']:.4f}; {len(v['C1_fail'])} fail: {v['C1_fail']})")
    print("C2 response > 0.03:", v["C2"], {k: (None if x is None else f"{x:.4f}") for k, x in c2.items()})
    print(f"C3 noise <= 0.01: {v['C3']} (max {v['C3_max_noise']:.1e}; {len(v['C3_fail'])} fail: {v['C3_fail']})")
    print(f"C3' fresh draws <= 0.01: {v['C3f']} (max {v['C3f_max_noise']:.1e}; {len(v['C3f_fail'])} fail)")
    print("B1 ratio-law baseline misses kappa_* plants by > 0.05:", v["B1"])
    if mut in ("unweighted", "err_reversed"):
        fired = not (v["C1"] and v["C2"] and v["C3"])
        print(f"mutation {mut}: must fail C1, C2 or C3: {'FIRED' if fired else 'DID NOT FIRE'}")
        return "fired" if fired else "nofire"
    if mut == "noise1000":
        fired = not v["C3"]
        print(f"mutation {mut}: must fail C3: {'FIRED' if fired else 'DID NOT FIRE'}")
        return "fired" if fired else "nofire"
    if mut == "err_flat":
        print("mutation err_flat (the measured profile replaced by a constant): reported, not judged")
        return "reported"
    return "pass" if (v["C1"] and v["C2"] and v["C3"] and v["C3f"] and v["B1"]) else "fail"


def run_real(outdir, mut):
    preset = FROZEN["preset"]
    if preset is None:
        raise SystemExit("FROZEN['preset'] is not set: the development (19.2) comes first")
    rec = {"frozen": FROZEN, "preset": preset, "mutate": mut, "real": {}}
    for lab, d in pg.REAL.items():
        for wf, sig in pg.load_scan(d).items():
            mp.mp.dps = DPS
            sig = pf.mutate(sig[:N_COEF], mut)
            est, got, spr = estimate(sig, eps_of(wf), preset)
            nz = nzf = None
            if lab == "J1250" and not mut:
                nz = noise_shift(sig, wf, preset, None, SEED_547, est)
                nzf = noise_shift(sig, wf, preset, None, SEED_FRESH, est)
            r = {"estimate": est, "abs": abs(est) if est is not None else None, "spread": spr, "noise": nz,
                 "noise_fresh": nzf, "members": got, "ratio12_21": pf.ratio12_21(sig)}
            rec["real"].setdefault(lab, {})[wf] = r
            print(f"real {lab} wf {wf}: far end {est} |.| {r['abs']}  spread {spr}  noise {nz}  fresh {nzf}  ratio12-21 "
                  f"{r['ratio12_21']:.4f}")
            for g in got:
                t = "none" if g["t"] is None else f"{g['t']:.4f} |t| {abs(g['t']):.4f}"
                print(f"     {g['degs']} lp {g['lp']}: {t}")
            sys.stdout.flush()
    a, b = rec["real"]["J1250"][1.0]["estimate"], rec["real"]["J1500"][1.0]["estimate"]
    rec["J_spread"] = abs(a - b) if a is not None and b is not None else None
    name = "real" + (f"_mutate_{mut}" if mut else "") + ".json"
    json.dump(jsonable(rec), open(os.path.join(outdir, name), "w"), indent=1)
    if mut and mut.startswith("scale"):
        c = float(mut[5:])
        base = complex(*json.load(open(os.path.join(outdir, "real.json")))["real"]["J1250"]["1.0"]["estimate"])
        dev, moved = abs(a - c * base), abs(a - base)
        print(f"mutation {mut}: |estimate - {c} x unmutated| = {dev:.1e} (pass < 1e-6), moved by {moved:.4f} (must "
              f"exceed 0.03)")
        return dev < 1e-6 and moved > 0.03
    if mut:
        base = complex(*json.load(open(os.path.join(outdir, "real.json")))["real"]["J1250"]["1.0"]["estimate"])
        print(f"mutation {mut}: the estimate moves by {abs(a - base):.4f} (reported, not judged)")
    return a is not None


SELFTEST_MUTS = ("e0_row", "selftest_row", "err_x1e3_tail", "abs_window", "selftest_sigma18")


def main():
    argv = sys.argv[1:]
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    argv = [a for a in argv if not a.startswith("--mutate=")]
    pg.NOISE_MODE = "measured"
    if not argv or argv[0] == "selftest":
        ok = selftest(mut)
        if mut is None:
            sys.exit(0 if ok else 1)
        print(f"mutation {mut}: the selftest {'FAILED, the mutation fired' if not ok else 'PASSED, it did not fire'}")
        sys.exit(1 if not ok else 2)
    if argv[0] == "timing":
        timing()
        return
    stage, outdir = argv[0], argv[1]
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    os.makedirs(outdir, exist_ok=True)
    if stage == "develop":
        develop(outdir)
        sys.exit(0)
    if stage == "plants":
        r = run_plants(outdir, mut)
        sys.exit({"pass": 0, "fail": 1, "fired": 1, "nofire": 2, "reported": 0}[r])
    ok = run_real(outdir, mut)
    sys.exit((0 if ok else 1) if not mut or mut.startswith("scale") else 0)


if __name__ == "__main__":
    main()
