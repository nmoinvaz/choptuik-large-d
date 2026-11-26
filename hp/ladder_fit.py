#!/usr/bin/env python3
"""#1098: Delta_inf from a ladder of full form-S direct solves alone (D about 250-4000), with an error bar calibrated
on planted truths of different analytic structure.

The rule (RULE below; chosen on the planted truths alone, before any real point was fitted):
- fits: weighted least squares, polynomials of degree K in x, x in {1/(D-2), 1/(D-1), 1/D} (Chebyshev basis over the
  fitted x range, arb at 256 bits), on the ladder points with D >= D_min, D_min at level lev = 0 .. NLEV (the lowest
  lev points dropped) with at least NMIN points kept; K from 1 to min(KMAX, N - 3) (dof >= 2);
- accepted: chi^2 <= the chi^2(dof) 0.99 quantile and every |residual| / sigma <= RMAX;
- nested test of two fits p, q (q with the larger variance; a degree more, or one low point fewer): |v_p - v_q| <=
  ZNEST sd, sd^2 = s_q^2 - s_p^2 (for nested least squares the variance of the difference of the values at x = 0),
  floored at SDFLOOR s_q;
- K*(x, lev): the smallest K with K and K + 1 both accepted and passing the nested test;
- a level is stable when in every variable its K* fit passes the nested test against the same degree one level up;
  D0 is the lowest stable level among the first NLEV (at most NLEV - 1 low points dropped);
- candidates: the K* and K* + 1 fits of every variable at D0 and at D0 + 1, up to 12 fits (--cand=pair; --cand=kstar
  keeps the K* fits only, --cand=one those at D0 only: planted F 5.2 and 10.2 on ladder A, against 3.0, because
  T4's contamination of the low points then passes unseen);
- value: the candidates' median (the lower middle one); h half their range; s the central fit's propagated data error
  (1 sigma, from the inverse normal matrix); E = F max(h, s), F from the planted test of the same ladder (`planted`;
  about 3, the largest of about 1000 calibration cases, so E is close to a 3-sigma bar);
- no D0: no quote (exit 1);
- leave-one-out check: each point with D >= D0 against the rule run without it, z = (y_i - prediction) / sd, the
  prediction from that run's central fit (var, lev, K), sd^2 = sigma_i^2 + the prediction's propagated variance +
  (the degree K + 1 prediction less it)^2; |z| > ZLOO (10) is an alarm. Without it a shifted interior point was
  absorbed by a degree that interpolates it (leverage 1). Planted (raw I1098-mac loocheck): no alarm on T1-T3 in 304
  quoted cases; a +30 sigma shift at D 1000 reads 17-22.

Planted truths (mpmath, 60 digits; limits exact), each a function of n = D - 2, e = 1/n:
  T1 readme   track_c.readme_formula: rational in e, poles at e -0.0769 and -0.191 (radius 0.077, n -13)
  T2 near     T1 + a complex pair of poles at |e| = 0.02 (n 50, a radius 3.8 times smaller), angle +-2 pi/3, from
              order e^4 on (A 2 Re[(e/z0)^4 / (1 - e/z0)], A = 6.77e-4: |e^4 coefficient| = T1's, 4229)
  T3 branch   T1 + a square-root branch point at e = -0.03 (n -33), from order e^4 on (A [sqrt(1 + e/r) - its cubic
              Taylor polynomial], A = 0.0877: |e^4 coefficient| = T1's)
  T4 exp-osc  T1 + B e^(-0.05 n) cos(1.3 n + 0.4) (track_c P2's rate), B so that it is 1e-16 at D 245.64, the edge of
              the dd/qd/chain_fit data (the size those data allow)
  T5 exp-mono T1 + B e^(-0.05 n), the same B
  L  probe    T1 + c e^5 log e (not analytic at e = 0: outside the rule's hypothesis; `probe` scans c)
Noise per point: Gaussian of sigma (seeds), and three systematic patterns of size sigma: all +sigma, all -sigma, and a
ramp from -sigma to +sigma linear in 1/(D - 2).

    cd hp && python3 ladder_fit.py planted [--ladders=A,B,...] [--sigmas=1e-19,...] [--seeds=40] [--out=FILE]
    cd hp && python3 ladder_fit.py planted --D=D1,D2,... [--rel=r1,r2,...]   a given ladder (sigma_i = r_i sigma)
    cd hp && python3 ladder_fit.py budget [--ladders=...]          noise gain and truncation bias per degree (an oracle)
    cd hp && python3 ladder_fit.py controls --ladder=A --F=x --sigma=1e-23   shift and tight (exit 1 if one does not fire)
    cd hp && python3 ladder_fit.py controls --mutate=lowdeg|var --ladder=A --F=x --sigma=1e-23   (exit 1 if not fired)
    cd hp && python3 ladder_fit.py probe --ladder=A --F=x          the log term's coefficient scan
    cd hp && python3 ladder_fit.py real RAW/I1088-dh610 ... [--F=x] [--mutate=shift] [--out=FILE]

`planted` calibrates F per ladder on seeds 1-20 (T1-T4 and the systematic patterns, every sigma), checks coverage on
seeds 21-40, and writes the table: per ladder and truth, the largest sigma of the grid at which the 90th percentile of
F max(h, s) over the seeds is <= 1e-20 with at least 90% of the seeds quoted, coverage on seeds 21-40 >= 90% and the
systematic patterns covered, and the K* the rule picks there.
Controls (each must fire): shift (D 1000 moved by +30 sigma: no quote, or D 1000 dropped), tight (E / 10: coverage
fails), lowdeg (KMAX 3: no quote or outside E), var (x = (D-2)^-1.5, in which the truths are not analytic: the same).
`real`: the full-system records in direct_s.jsonl under the raw folders given (I1088-*), one per D (the lowest check
residual), sigma = export_delta_all.direct_err's err (its largest recorded term; no floor). Its checks: the central
fit's residuals |fit - data| / sigma <= 3.5, and the leave-one-out check (exit 1 if either fails); --mutate=shift
moves the middle point by +30 sigma. `mkreal --ladder=E20 --truth=T1 --sigma=1e-23 --dir=DIR` writes planted records
in the same format, for a test of `real` through its own reader.
With fewer than NMIN points it quotes nothing and prints the three-point check: the K = 1 and K = 2 fits in each
variable, beside the same fits to T1 at the same D, and the real values against Track C's Delta_inf (#1095).
"""
import glob
import json
import math
import os
import random
import sys

import mpmath as mp
import numpy as np
from flint import arb, arb_mat, ctx
from scipy.stats import chi2 as CHI2

sys.argv, _argv = [sys.argv[0], "real"], sys.argv      # track_c parses argv[2:]; it gets none
import track_c as T  # noqa: E402
sys.argv = _argv

ctx.prec = 256
mp.mp.dps = 60
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:] if
            a.startswith("--"))
MUT = args.get("mutate")

RULE = dict(NLEV=4, NMIN=5, KMAX=16, PCHI=0.99, RMAX=4.0, ZNEST=3.0, SDFLOOR=0.3, CAND=args.get("cand", "pair"))
VARS = {"1/(D-2)": lambda D: 1 / (D - 2), "1/(D-1)": lambda D: 1 / (D - 1), "1/D": lambda D: 1 / D}
if MUT == "var":                                       # control: a variable the truths are not analytic in
    VARS = {"(D-2)^-1.5": lambda D: (D - 2) ** mp.mpf("-1.5")}
if MUT == "lowdeg":                                    # control: degrees too low for the sigma
    RULE["KMAX"] = 3
TARGET = 1e-20
SIGMAS = [1e-18, 3e-19, 1e-19, 3e-20, 1e-20, 3e-21, 1e-21, 3e-22, 1e-22, 3e-23, 1e-23, 3e-24, 1e-24]
RLIM = mp.mpf(T.RA[0])                                 # T1's limit, exact (its e^0 coefficient)


def geom(lo, hi, n, members=()):
    """n points geometric in D from lo to hi; each member replaces the nearest point (in log D)"""
    pts = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]
    for m in members:
        if lo <= m <= hi:
            j = min(range(n), key=lambda i: abs(math.log(pts[i] / m)))
            pts[j] = m
    return sorted(set(round(p, 9) for p in pts))


def cheb_e(lo, hi, n, members=()):
    """n Chebyshev points in e = 1/(D - 2) over [1/(hi - 2), 1/(lo - 2)]; members replace the nearest"""
    a, b = 1 / (hi - 2), 1 / (lo - 2)
    es = [(a + b) / 2 - (b - a) / 2 * math.cos(math.pi * i / (n - 1)) for i in range(n)]
    pts = [2 + 1 / e for e in es]
    for m in members:
        if lo <= m <= hi:
            j = min(range(n), key=lambda i: abs(1 / (pts[i] - 2) - 1 / (m - 2)))
            pts[j] = m
    return sorted(set(round(p, 9) for p in pts))


R2 = 2828.4271247461903
HAVE = (300.0, 1000.0, 2000.0, R2, 4000.0)            # the full points that exist or are running
LADDERS = {
    "A": ("D 250-4000, 15 geometric", geom(250, 4000, 15, HAVE)),
    "A20": ("D 250-4000, 20 geometric", geom(250, 4000, 20, HAVE)),
    "Ac": ("D 250-4000, 15 Chebyshev in 1/n", cheb_e(250, 4000, 15, HAVE)),
    "A25": ("D 250-4000, 25 geometric", geom(250, 4000, 25, HAVE)),
    "B": ("D 500-4000, 11 geometric", geom(500, 4000, 11, HAVE)),
    "B15": ("D 500-4000, 15 geometric", geom(500, 4000, 15, HAVE)),
    "B20": ("D 500-4000, 20 geometric", geom(500, 4000, 20, HAVE)),
    "C": ("D 300-2000, 11 geometric", geom(300, 2000, 11, HAVE)),
    "C15": ("D 300-2000, 15 geometric", geom(300, 2000, 15, HAVE)),
    "E": ("D 250-8000, 15 geometric", geom(250, 8000, 15, HAVE + (8000.0,))),
    "E20": ("D 250-8000, 20 geometric", geom(250, 8000, 20, HAVE + (8000.0,))),
    "E25": ("D 250-8000, 25 geometric", geom(250, 8000, 25, HAVE + (8000.0,))),
    "J20": ("D 250-16000, 20 geometric", geom(250, 16000, 20, HAVE + (8000.0,))),
    "F15": ("D 500-8000, 15 geometric", geom(500, 8000, 15, HAVE + (8000.0,))),
    "F20": ("D 500-8000, 20 geometric", geom(500, 8000, 20, HAVE + (8000.0,))),
    "G": ("D 1000-8000, 9 geometric", geom(1000, 8000, 9, HAVE + (8000.0,))),
    "H": ("the five: D 300, 1000, 2000, 2828, 4000", list(HAVE)),
}
RELS = {}                                              # per ladder: sigma_i / sigma (default 1 at every point)
if "D" in args:                                        # a ladder of one's own: --D=D1,D2,... [--rel=r1,r2,...]
    _D = [float(v) for v in args["D"].split(",")]
    _r = [float(v) for v in args["rel"].split(",")] if "rel" in args else [1.0] * len(_D)
    _o = sorted(range(len(_D)), key=lambda i: _D[i])
    LADDERS["X"] = ("given: D " + ", ".join("%g" % _D[i] for i in _o), [_D[i] for i in _o])
    RELS["X"] = [_r[i] for i in _o]


# ---- planted truths -------------------------------------------------------------------------------------------------
Z0 = mp.mpf("0.02") * mp.expjpi(mp.mpf(2) / 3)
A2 = mp.mpf("6.77e-4")
RB3, A3 = mp.mpf("0.03"), mp.mpf("0.0877")
B4 = mp.mpf("1e-16") * mp.exp(mp.mpf("0.05") * (mp.mpf("245.64") - 2))


def truth(kind, D, c_log=None):
    D = mp.mpf(D)
    n = D - 2
    e = 1 / n
    t = T.readme_formula(D)
    if kind == "T2":
        u = e / Z0
        t += A2 * 2 * mp.re(u ** 4 / (1 - u))
    elif kind == "T3":
        u = e / RB3
        t += A3 * (mp.sqrt(1 + u) - (1 + u / 2 - u ** 2 / 8 + u ** 3 / 16))
    elif kind == "T4":
        t += B4 * mp.exp(-mp.mpf("0.05") * n) * mp.cos(mp.mpf("1.3") * n + mp.mpf("0.4"))
    elif kind == "T5":
        t += B4 * mp.exp(-mp.mpf("0.05") * n)
    elif kind == "L":
        t += c_log * e ** 5 * mp.log(e)
    return t


TRUTHS = ("T1", "T2", "T3", "T4", "T5")
CAL = ("T1", "T2", "T3", "T4")                         # F is calibrated on these; T5 and L are reported


def a(v):
    return arb(mp.nstr(mp.mpf(v), 75))


# ---- fit operators --------------------------------------------------------------------------------------------------
def cheb_rows(ts, lo, hi, K):
    rows = []
    for t in ts:
        u = (2 * t - (lo + hi)) / (hi - lo)
        r = [arb(1), u]
        for _ in range(K - 1):
            r.append(2 * u * r[-1] - r[-2])
        rows.append(r[:K + 1])
    return rows


def ops(Ds, w, var, K, Dt=None):
    """weighted least squares of degree K in var on the points Ds with weights w (1/sigma, relative): g (the weights of
    the value at x = 0 on the data), P (the weighted residual projector, I - H), both arb; with Dt also gt, the
    weights of the fit's value at D = Dt"""
    f = VARS[var]
    ts = [a(f(mp.mpf(d))) for d in Ds]
    lo = min(ts, key=float)
    hi = max(ts, key=float)
    X = arb_mat([[v * w[i] for v in r] for i, r in enumerate(cheb_rows(ts, lo, hi, K))])
    Ni = (X.transpose() * X).inv()
    r0 = arb_mat([cheb_rows([arb(0)], lo, hi, K)[0]])
    G = r0 * Ni * X.transpose()                      # 1 x N: value at 0 = G (W y)
    H = X * Ni * X.transpose()
    N = len(Ds)
    P = arb_mat([[(1 if i == j else 0) - H[i, j] for j in range(N)] for i in range(N)])
    if Dt is None:
        return [G[0, i] * w[i] for i in range(N)], P
    rt = arb_mat([cheb_rows([a(f(mp.mpf(Dt)))], lo, hi, K)[0]])
    Gt = rt * Ni * X.transpose()
    return [G[0, i] * w[i] for i in range(N)], P, [Gt[0, i] * w[i] for i in range(N)]


def ladder_ops(L, w=None, Dt=None, OT=None):
    """{(var, lev, K): (idx, g arb list, P arb_mat, g float array, P float array)} for the ladder L (sorted D list);
    with Dt, OT[key] = (gt arb list, gt float array), the weights of each fit's value at D = Dt"""
    w = w or [arb(1)] * len(L)
    out = {}
    for var in VARS:
        for lev in range(RULE["NLEV"] + 1):
            idx = list(range(lev, len(L)))
            if len(idx) < RULE["NMIN"]:
                continue
            for K in range(1, min(RULE["KMAX"], len(idx) - 3) + 1):
                if Dt is None:
                    g, P = ops([L[i] for i in idx], [w[i] for i in idx], var, K)
                else:
                    g, P, gt = ops([L[i] for i in idx], [w[i] for i in idx], var, K, Dt)
                    OT[(var, lev, K)] = (gt, np.array([float(x) for x in gt]))
                gf = np.array([float(x) for x in g])
                Pf = np.array([[float(P[i, j]) for j in range(len(idx))] for i in range(len(idx))])
                out[(var, lev, K)] = (idx, g, P, gf, Pf)
    return out


def loo_ops(L, rel):
    """the leave-one-out check's fits: per point i, the ladder without it, its fits and their weights at D_i"""
    out = []
    for i in range(len(L)):
        Li = L[:i] + L[i + 1:]
        ri = rel[:i] + rel[i + 1:]
        OT = {}
        Oi = ladder_ops(Li, [arb(1) / a(r) for r in ri], Dt=L[i], OT=OT)
        out.append((i, L[i], Li, ri, Oi, OT))
    return out


ZLOO = 10.0                                            # the leave-one-out check's bar (planted: #1098)


# ---- the rule -------------------------------------------------------------------------------------------------------
def chi2max(dof):
    return CHI2.ppf(RULE["PCHI"], dof)


def accepted(r):
    return r is not None and r["chi2"] <= chi2max(r["dof"]) and r["maxres"] <= RULE["RMAX"]


def nested_ok(p, q):
    """|v_p - v_q| <= ZNEST sd for two nested least-squares fits, q the one with the larger variance:
    sd^2 = s_q^2 - s_p^2 (the variance of the difference), floored at SDFLOOR s_q"""
    sd = max(math.sqrt(max(q["s"] ** 2 - p["s"] ** 2, 0.0)), RULE["SDFLOOR"] * q["s"])
    return abs(p["d"] - q["d"]) <= RULE["ZNEST"] * sd


def select(recs, L):
    """recs {(var, lev, K): dict(d, s, chi2, dof, maxres)}, d the value at x = 0 less a common reference.
    K*(var, lev): the smallest K with K and K + 1 accepted and nested-consistent. A level is stable when, in every
    variable, it has a K* and that fit agrees (nested_ok) with the same degree one level up (one more low point dropped).
    D0: the lowest stable level of the first NLEV. Returns None (no D0) or dict(d, h, s, E0, D0, K, cands)"""
    kstar = {}
    for var in VARS:
        for lev in range(RULE["NLEV"] + 1):
            for K in range(1, RULE["KMAX"]):
                p, q = recs.get((var, lev, K)), recs.get((var, lev, K + 1))
                if accepted(p) and accepted(q) and nested_ok(p, q):
                    kstar[(var, lev)] = K
                    break

    def stable(lev):
        for var in VARS:
            K = kstar.get((var, lev))
            if K is None:
                return False
            q = recs.get((var, lev + 1, K))
            if q is None or not nested_ok(recs[(var, lev, K)], q):
                return False
        return True
    D0 = next((lev for lev in range(RULE["NLEV"]) if stable(lev)), None)
    if D0 is None:
        return None
    cands = []
    for lev in ((D0,) if RULE["CAND"] == "one" else (D0, D0 + 1)):
        for var in VARS:
            if (var, lev) in kstar:
                for K in (kstar[(var, lev)], kstar[(var, lev)] + 1)[:2 if RULE["CAND"] == "pair" else 1]:
                    if (var, lev, K) in recs:
                        cands.append(((var, lev, K), recs[(var, lev, K)]))
    cands.sort(key=lambda t: t[1]["d"])
    cen = cands[len(cands) // 2]
    h = (cands[-1][1]["d"] - cands[0][1]["d"]) / 2
    s = cen[1]["s"]
    return dict(d=cen[1]["d"], h=h, s=s, E0=max(h, s), D0=L[D0], lev=D0, central=cen[0],
                K=cen[0][2], cands=[c[0] for c in cands], kstar=kstar)


# ---- planted --------------------------------------------------------------------------------------------------------
def truth_parts(O, L, kind, c_log=None, rel=None):
    """per fit: the truncation bias at x = 0 (float, value - limit) and the weighted residual vector (float, in units
    of sigma) of the noiseless truth; rel the ladder's sigma profile (O built with weights 1/rel)"""
    tv = [a(truth(kind, d, c_log)) for d in L]
    rel = rel or [1.0] * len(L)
    lim = a(RLIM)
    out = {}
    for key, (idx, g, P, gf, Pf) in O.items():
        val = sum((g[i] * tv[j] for i, j in enumerate(idx)), arb(0))
        y = arb_mat([[tv[j] / a(rel[j])] for j in idx])
        r = P * y
        out[key] = (float(val - lim), np.array([float(r[i, 0]) for i in range(len(idx))]))
    return out


def patterns(L, seeds):
    """noise patterns z (len(L) x S): Gaussian seeds, then all +1, all -1, a ramp in 1/(D-2)"""
    zs, names = [], []
    for sd in seeds:
        rng = random.Random(1000 + sd)
        zs.append([rng.gauss(0, 1) for _ in L])
        names.append("g%d" % sd)
    e = [1 / (d - 2) for d in L]
    lo, hi = min(e), max(e)
    zs += [[1.0] * len(L), [-1.0] * len(L), [2 * (x - lo) / (hi - lo) - 1 for x in e]]
    names += ["sys+", "sys-", "ramp"]
    return np.array(zs).T, names


def recs_from(O, TP, sigma, z, rel):
    """the fit records of one noise column z (point i's noise sigma rel_i z_i)"""
    recs = {}
    for key, (idx, g, P, gf, Pf) in O.items():
        bias, r0 = TP[key]
        zz = z[idx]
        rr = r0 / sigma + Pf @ zz
        recs[key] = dict(d=bias + sigma * float(gf @ (rel[idx] * zz)), s=sigma * float(np.linalg.norm(gf * rel[idx])),
                         chi2=float(rr @ rr), dof=len(idx) - key[2] - 1, maxres=float(np.max(np.abs(rr))))
    return recs


def loo_prep(loo, L, kind, rel):
    """per point i: the truth parts of the fits without it, and per fit t_i less the fit's noiseless value at D_i"""
    tv = {d: a(truth(kind, d)) for d in L}
    out = []
    for i, Di, Li, ri, Oi, OT in loo:
        TPi = truth_parts(Oi, Li, kind, rel=ri)
        dev0 = {}
        for key, (idx, g, P, gf, Pf) in Oi.items():
            gt = OT[key][0]
            pred = sum((gt[k] * tv[Li[j]] for k, j in enumerate(idx)), arb(0))
            dev0[key] = float(tv[Di] - pred)
        out.append((TPi, dev0))
    return out


def loo_planted(loo, preps, sigma, z, rel, D0):
    """(z, D): the largest |z_i| over the points with D_i >= D0, z_i = (y_i - prediction) / sd, the prediction the
    central fit (var, lev, K) of the rule run without point i, sd^2 = sigma_i^2 + the prediction's propagated variance
    + (the prediction of degree K + 1 less it)^2 (its truncation, as the nested test reads it); (None, None) when no
    point could be tested"""
    worst = (None, None)
    for (i, Di, Li, ri, Oi, OT), (TPi, dev0) in zip(loo, preps):
        if Di < D0:
            continue
        zr = np.delete(z, i)
        rr_ = np.delete(rel, i)
        est = select(recs_from(Oi, TPi, sigma, zr, rr_), Li)
        if est is None:
            continue
        key = est["central"]
        idx = Oi[key][0]
        gtf = OT[key][1]
        dev = dev0[key] / sigma + rel[i] * z[i] - float(gtf @ (rr_[idx] * zr[idx]))
        k1 = (key[0], key[1], key[2] + 1)
        tr = 0.0
        if k1 in OT:
            g1 = OT[k1][1]
            tr = dev - (dev0[k1] / sigma + rel[i] * z[i] - float(g1 @ (rr_[idx] * zr[idx])))
        sd = math.sqrt(rel[i] ** 2 + float(np.sum((gtf * rr_[idx]) ** 2)) + tr ** 2)
        if worst[0] is None or abs(dev / sd) > abs(worst[0]):
            worst = (dev / sd, Di)
    return worst


def run_planted(O, L, TP, sigma, Z, shift=None, rel=None, loo=None, preps=None):
    """the rule on every column of Z at this sigma (point i's noise sigma rel_i z_i): list of select() results (None
    where nothing is quoted), with d = value - limit; with loo (loo_ops) and preps (loo_prep), each result also
    carries the leave-one-out check's largest |z| and its D"""
    S = Z.shape[1]
    rel = np.array(rel or [1.0] * len(L))
    res = []
    for k in range(S):
        z = Z[:, k].copy()
        if shift is not None:
            z[shift[0]] += shift[1]
        r = select(recs_from(O, TP, sigma, z, rel), L)
        if r is not None and loo is not None:
            r["loo"], r["loo_D"] = loo_planted(loo, preps, sigma, z, rel, r["D0"])
        res.append(r)
    return res


def cmd_planted():
    lads = args.get("ladders", ",".join(LADDERS)).split(",")
    sigmas = [float(v) for v in args["sigmas"].split(",")] if "sigmas" in args else SIGMAS
    S = int(args.get("seeds", "40"))
    cal_seeds, val_seeds = list(range(1, S // 2 + 1)), list(range(S // 2 + 1, S + 1))
    out = {"rule": RULE, "vars": list(VARS), "target": TARGET, "ladders": {}}
    for lk in lads:
        name, L = LADDERS[lk]
        rel = RELS.get(lk, [1.0] * len(L))
        O = ladder_ops(L, [arb(1) / a(r) for r in rel])
        Z, names = patterns(L, cal_seeds + val_seeds)
        ncal = len(cal_seeds)
        icals = list(range(ncal)) + [len(names) - 3, len(names) - 2, len(names) - 1]
        ivals = list(range(ncal, ncal + len(val_seeds)))
        print("ladder %s (%s): %d points, D %s" % (lk, name, len(L), ", ".join("%g" % d for d in L)), flush=True)
        rows = {}
        for kind in TRUTHS:
            TP = truth_parts(O, L, kind, rel=rel)
            for sg in sigmas:
                rs = run_planted(O, L, TP, sg, Z, rel=rel)
                rows[(kind, sg)] = rs
        # F: the largest |err| / E0 over the calibration columns of T1-T4 at every sigma, floored at 1
        ratios = [abs(r["d"]) / r["E0"] for (kind, sg), rs in rows.items() if kind in CAL
                  for i, r in enumerate(rs) if i in icals and r is not None and r["E0"] > 0]
        F = max([1.0] + ratios)
        worst = sorted(((abs(r["d"]) / r["E0"], kind, sg, names[i], r) for (kind, sg), rs in rows.items()
                        if kind in CAL for i, r in enumerate(rs) if i in icals and r is not None and r["E0"] > 0),
                       key=lambda t: -t[0])[:6]
        for rt, kind, sg, nm, r in worst:
            print("    worst: %.2f %s sigma %.0e %s: err %+.2e h %.2e s %.2e K %d D0 %g central %s; candidates %s" % (
                rt, kind, sg, nm, r["d"], r["h"], r["s"], r["K"], r["D0"], r["central"][0],
                " ".join("%s/%d/%d" % c for c in r["cands"])))
        rec = {"name": name, "D": L, "rel": rel, "F": F, "truths": {}}
        print("  F = %.2f (largest |err| / max(h, s) over %d calibration cases of %s)" % (F, len(ratios), "+".join(CAL)))
        for kind in TRUTHS:
            per = []
            for sg in sigmas:
                rs = rows[(kind, sg)]
                q = [r for i, r in enumerate(rs) if i < ncal + len(val_seeds) and r is not None]
                nq = len(q)
                ngauss = ncal + len(val_seeds)
                Es = sorted(F * r["E0"] for r in q)
                E90 = Es[min(len(Es) - 1, int(math.ceil(0.9 * len(Es))) - 1)] if Es else float("nan")
                Emed = Es[len(Es) // 2] if Es else float("nan")
                errs = [abs(r["d"]) for r in q]
                val = [rs[i] for i in ivals]
                cov = [abs(r["d"]) <= F * r["E0"] for r in val if r is not None]
                sysr = [rs[i] for i in (len(names) - 3, len(names) - 2, len(names) - 1)]
                sysok = all(r is None or abs(r["d"]) <= F * r["E0"] for r in sysr)
                Ks = sorted(r["K"] for r in q)
                D0s = sorted(r["D0"] for r in q)
                row = dict(sigma=sg, quoted=nq / ngauss, E_med=Emed, E_90=E90,
                           err_med=sorted(errs)[len(errs) // 2] if errs else float("nan"),
                           err_max=max(errs) if errs else float("nan"),
                           cover_val=(sum(cov) / len(cov)) if cov else float("nan"), sys_ok=sysok,
                           K_med=Ks[len(Ks) // 2] if Ks else None, D0_med=D0s[len(D0s) // 2] if D0s else None,
                           ratio_max=max((abs(r["d"]) / r["E0"] for r in q if r["E0"] > 0), default=float("nan")))
                per.append(row)
                print("  %s sigma %.0e: quoted %3.0f%%  F*E med %.1e  90%% %.1e  |err| med %.1e max %.1e  "
                      "cover(val) %3.0f%%  sys %s  K* %s  D0 %s" % (
                          kind, sg, 100 * row["quoted"], Emed, E90, row["err_med"], row["err_max"],
                          100 * row["cover_val"] if cov else float("nan"), "ok" if sysok else "MISS", row["K_med"],
                          "%g" % row["D0_med"] if row["D0_med"] else "-"), flush=True)
            need = [r for r in per if r["quoted"] >= 0.9 and r["E_90"] <= TARGET
                    and (math.isnan(r["cover_val"]) or r["cover_val"] >= 0.9) and r["sys_ok"]]
            best = max(need, key=lambda r: r["sigma"]) if need else None
            rec["truths"][kind] = dict(rows=per, sigma_needed=best["sigma"] if best else None,
                                       K=best["K_med"] if best else None, D0=best["D0_med"] if best else None)
            print("  %s: sigma needed for F*E (90%%) <= %.0e: %s (K* %s, D0 %s)" % (
                kind, TARGET, "%.0e" % best["sigma"] if best else "none on the grid",
                best["K_med"] if best else "-", "%g" % best["D0_med"] if best else "-"), flush=True)
        out["ladders"][lk] = rec
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)
    print("table: sigma per point needed for Delta_inf to %.0e (F max(h, s), 90th percentile over seeds)" % TARGET)
    print("  %-4s %-40s %5s  %s" % ("lad", "ladder", "F", "  ".join("%-12s" % k for k in TRUTHS)))
    for lk, rec in out["ladders"].items():
        cells = []
        for kind in TRUTHS:
            t = rec["truths"][kind]
            cells.append("%-12s" % (("%.0e K%s" % (t["sigma_needed"], t["K"])) if t["sigma_needed"] else "none"))
        print("  %-4s %-40s %5.2f  %s" % (lk, rec["name"], rec["F"], "  ".join(cells)))


def cmd_budget():
    """per ladder, at the lowest D_min level: per variable and degree K the noise gain ||g|| (s = ||g|| sigma for a
    uniform sigma) and each truth's truncation bias at x = 0 (noiseless fit - limit); and the best achievable
    max(|bias|, s) at each sigma over K (an oracle that knows the truth; not the rule)"""
    lads = args.get("ladders", ",".join(LADDERS)).split(",")
    out = {}
    for lk in lads:
        name, L = LADDERS[lk]
        O = ladder_ops(L)
        TPs = {k: truth_parts(O, L, k) for k in TRUTHS}
        print("ladder %s (%s)" % (lk, name), flush=True)
        rec = {}
        for var in VARS:
            Ks = sorted(K for (v, lev, K) in O if v == var and lev == 0)
            rows = []
            for K in Ks:
                key = (var, 0, K)
                gain = float(np.linalg.norm(O[key][3]))
                rows.append(dict(K=K, gain=gain, gain1=float(np.sum(np.abs(O[key][3]))),
                                 bias={k: TPs[k][key][0] for k in TRUTHS}))
            rec[var] = rows
            print("  %-8s K: " % var + "  ".join("%d" % r["K"] for r in rows))
            print("  %-8s gain: " % var + "  ".join("%.0e" % r["gain"] for r in rows))
            print("  %-8s gain1: " % var + "  ".join("%.0e" % r["gain1"] for r in rows) +
                  "   (sum of |g_i|: the worst case for errors bounded by sigma of any sign)")
            for k in TRUTHS:
                print("  %-8s %s |bias|: " % (var, k) + "  ".join("%.0e" % abs(r["bias"][k]) for r in rows))
            for k in TRUTHS:
                best = []
                for sg in SIGMAS:
                    m = min(rows, key=lambda r: max(abs(r["bias"][k]), r["gain"] * sg))
                    best.append((sg, max(abs(m["bias"][k]), m["gain"] * sg), m["K"]))
                need = [b for b in best if b[1] <= TARGET]
                print("  %-8s %s oracle: sigma needed for max(|bias|, s) <= %.0e: %s" % (
                    var, k, TARGET, ("%.0e (K %d)" % (max(need)[0], max(need)[2])) if need else "none on the grid"))
                rec.setdefault("oracle", {}).setdefault(var, {})[k] = best
        out[lk] = rec
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)


# ---- controls -------------------------------------------------------------------------------------------------------
def flagged(r):
    """the rule's checks on one planted result: no quote, or the leave-one-out check over ZLOO"""
    return r is None or (r.get("loo") is not None and abs(r["loo"]) > ZLOO)


def cmd_controls():
    """on T1 at one ladder and sigma, Gaussian seeds 21-30 (not the calibration seeds 1-20): each control must fire.
    shift: the D 1000 point moved by +30 sigma; the rule must quote nothing, drop D 1000, or fail the leave-one-out
    check (|z| > ZLOO at some point), in every seed (exit 1 if not).
    tight: E divided by 10; the coverage check must fail in some seed. Informational: the unmutated false alarms, and
    the shift size the checks catch, at D 1000 and at the ladder's lowest point."""
    lk = args.get("ladder", "A")
    sg = float(args.get("sigma", "1e-22"))
    name, L = LADDERS[lk]
    Fc = float(args.get("F", "1"))
    out = {}
    O = ladder_ops(L)
    TP = truth_parts(O, L, "T1")
    loo = loo_ops(L, [1.0] * len(L))
    preps = loo_prep(loo, L, "T1", [1.0] * len(L))
    Z, names = patterns(L, list(range(21, 31)))
    Z = Z[:, :10]
    base = run_planted(O, L, TP, sg, Z, loo=loo, preps=preps)
    q = [r for r in base if r is not None]
    fa = sum(1 for r in q if flagged(r))
    print("controls on T1, ladder %s, sigma %.0e, seeds 21-30; F %.2f; leave-one-out bar %.1f" % (lk, sg, Fc, ZLOO))
    print("  unmutated: %d of 10 quoted; |err| / (F E0) max %.2f; leave-one-out |z| max %.2f (false alarms %d)" % (
        len(q), max(abs(r["d"]) / (Fc * r["E0"]) for r in q) if q else float("nan"),
        max((abs(r["loo"]) for r in q if r["loo"] is not None), default=float("nan")), fa))
    out["base"] = dict(quoted=len(q), ratio_max=max(abs(r["d"]) / (Fc * r["E0"]) for r in q) if q else None,
                       loo_max=max((abs(r["loo"]) for r in q if r["loo"] is not None), default=None), false_alarms=fa)
    j = L.index(1000.0)
    sh = run_planted(O, L, TP, sg, Z, shift=(j, 30.0), loo=loo, preps=preps)
    fired = sum(1 for r in sh if flagged(r) or r["D0"] > 1000.0)
    miss = sum(1 for r in sh if not flagged(r) and r["D0"] <= 1000.0 and abs(r["d"]) > Fc * r["E0"])
    print("  shift (D 1000 by +30 sigma): fired (no quote, D 1000 dropped, or leave-one-out) in %d of 10 (%s); "
          "passed and |err| > F E in %d" % (fired, ", ".join(
              "none" if r is None else "D0 %g loo %.1f at D %g" % (r["D0"], r["loo"] or 0, r["loo_D"] or 0)
              for r in sh), miss))
    out["shift"] = dict(fired=fired, passed_wrong=miss)
    tight = sum(1 for r in q if abs(r["d"]) > 0.1 * Fc * r["E0"])
    print("  tight (E / 10): coverage fails in %d of %d" % (tight, len(q)))
    out["tight"] = dict(fired=tight)
    out["shift_scan"] = []
    for at in (1000.0, L[0]):
        jj = L.index(at)
        for k in (3.0, 5.0, 10.0, 20.0):
            rs = run_planted(O, L, TP, sg, Z, shift=(jj, k), loo=loo, preps=preps)
            nq = sum(1 for r in rs if r is None)
            lf = sum(1 for r in rs if r is not None and flagged(r))
            dropped = sum(1 for r in rs if r is not None and not flagged(r) and r["D0"] > at)
            wrong = sum(1 for r in rs if r is not None and not flagged(r) and r["D0"] <= at
                        and abs(r["d"]) > Fc * r["E0"])
            print("  shift scan: D %g by %+g sigma: no quote %d, leave-one-out fires %d, point dropped %d, passed "
                  "and |err| > F E %d (of 10)" % (at, k, nq, lf, dropped, wrong))
            out["shift_scan"].append(dict(D=at, k=k, no_quote=nq, loo_fired=lf, dropped=dropped, wrong=wrong))
    ok = fired == 10 and tight > 0
    print("controls in this process (shift, tight): %s; lowdeg and var run as separate processes "
          "(--mutate=lowdeg, --mutate=var)" % ("FIRE" if ok else "DID NOT FIRE"))
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)
    sys.exit(0 if ok else 1)


def cmd_loocheck():
    """the leave-one-out check on clean planted data (seeds 21-40, every truth): how often it fires (on T1-T3 a false
    alarm; on T4 and T5 it may see the low points' exponential term), and the coverage of what passes"""
    lads = args.get("ladders", "A,A20,E20").split(",")
    sigmas = [float(v) for v in args.get("sigmas", "1e-22,1e-23").split(",")]
    Fs = dict(kv.split(":") for kv in args.get("Fs", "").split(",") if kv)
    out = {}
    for lk in lads:
        name, L = LADDERS[lk]
        rel = [1.0] * len(L)
        Fc = float(Fs.get(lk, "1"))
        O = ladder_ops(L)
        loo = loo_ops(L, rel)
        Z, names = patterns(L, list(range(21, 41)))
        Z = Z[:, :20]
        for kind in TRUTHS:
            TP = truth_parts(O, L, kind)
            preps = loo_prep(loo, L, kind, rel)
            for sg in sigmas:
                rs = run_planted(O, L, TP, sg, Z, loo=loo, preps=preps)
                q = [r for r in rs if r is not None]
                fl = [r for r in q if flagged(r)]
                ps = [r for r in q if not flagged(r)]
                cov = sum(1 for r in ps if abs(r["d"]) <= Fc * r["E0"])
                zs = sorted(abs(r["loo"]) for r in q if r["loo"] is not None)
                rec = dict(quoted=len(q), fired=len(fl), passed=len(ps), covered=cov,
                           z_med=zs[len(zs) // 2] if zs else None, z_max=zs[-1] if zs else None,
                           fired_at=sorted(set(r["loo_D"] for r in fl)))
                out["%s %s %.0e" % (lk, kind, sg)] = rec
                print("%s %s sigma %.0e: quoted %d of 20; leave-one-out fires %d (at D %s), |z| med %s max %s; "
                      "passed %d, covered at F %.2f: %d" % (
                          lk, kind, sg, len(q), len(fl), ",".join("%g" % d for d in rec["fired_at"]) or "-",
                          "%.2f" % rec["z_med"] if zs else "-", "%.2f" % rec["z_max"] if zs else "-", len(ps), Fc, cov),
                      flush=True)
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)


def cmd_mutated():
    """--mutate=lowdeg or var on T1, ladder A, sigma 1e-22: the rule must quote nothing, or the coverage fail"""
    lk = args.get("ladder", "A")
    sg = float(args.get("sigma", "1e-22"))
    Fc = float(args.get("F", "1"))
    name, L = LADDERS[lk]
    O = ladder_ops(L)
    TP = truth_parts(O, L, "T1")
    Z, names = patterns(L, list(range(21, 31)))
    rs = run_planted(O, L, TP, sg, Z[:, :10])
    q = [r for r in rs if r is not None]
    wrong = sum(1 for r in q if abs(r["d"]) > Fc * r["E0"])
    fired = (10 - len(q)) + wrong
    print("mutation %s on T1, ladder %s, sigma %.0e: %d of 10 quoted, %d of those outside F E; fired in %d of 10: %s" % (
        MUT, lk, sg, len(q), wrong, fired, "FIRES" if fired == 10 else "DOES NOT FIRE in every seed"))
    for r in q[:3]:
        print("  quoted: err %.2e  E0 %.2e  K %d  D0 %g" % (r["d"], r["E0"], r["K"], r["D0"]))
    sys.exit(0 if fired == 10 else 1)


def cmd_probe():
    """the log term c e^5 log e (outside the hypothesis): per c, on ladder A at sigma 1e-22, the share of seeds quoted,
    and of those the share whose |err| > F E (a silent miss)"""
    lk = args.get("ladder", "A")
    sg = float(args.get("sigma", "1e-22"))
    Fc = float(args.get("F", "1"))
    name, L = LADDERS[lk]
    O = ladder_ops(L)
    Z, names = patterns(L, list(range(1, 21)))
    Z = Z[:, :20]
    out = []
    for c in ("1e-12", "1e-10", "1e-8", "1e-6", "1e-4", "1e-2", "1"):
        TP = truth_parts(O, L, "L", mp.mpf(c))
        rs = run_planted(O, L, TP, sg, Z)
        q = [r for r in rs if r is not None]
        miss = sum(1 for r in q if abs(r["d"]) > Fc * r["E0"])
        out.append(dict(c=c, quoted=len(q), miss=miss, err_max=max((abs(r["d"]) for r in q), default=None),
                        E_med=sorted(Fc * r["E0"] for r in q)[len(q) // 2] if q else None))
        print("log probe c %-6s: quoted %2d of 20, outside F E %2d; |err| max %s, F E med %s" % (
            c, len(q), miss, "%.1e" % out[-1]["err_max"] if q else "-", "%.1e" % out[-1]["E_med"] if q else "-"),
            flush=True)
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)


# ---- real -----------------------------------------------------------------------------------------------------------
def full_points(folders):
    """{D: (Delta mpf, sigma float, src)}: full-system direct records, one per D by the lowest check residual"""
    dps = mp.mp.dps
    import export_delta_all as X
    mp.mp.dps = dps
    recs = []
    for fo in folders:
        for f in sorted(glob.glob(os.path.join(fo, "**", "direct_s.jsonl"), recursive=True)):
            for n, line in enumerate(open(f), 1):
                if line.strip():
                    r = json.loads(line)
                    if r.get("valid") is False or r.get("method") == "fine":
                        continue
                    recs.append((f"{f}:{n}", r))
    best = {}
    for src, r in recs:
        if r["system"] != "full":
            continue
        e, ek = X.direct_err(r, src, recs)
        if not e:
            print("full record %s (D %s) left out: %s" % (src, r["D_str"], ek))
            continue
        D = round(float(r["D_str"]), 9)
        key = r.get("F_check") if r.get("F_check") is not None else (r.get("F_solve") or 1.0)
        if D not in best or key < best[D][0]:
            v = mp.fsum(mp.mpf(float(x)) for x in r["Delta_parts"])     # exact; mpf(repr(x)) is the decimal, off by up to 1e-17
            best[D] = (key, (v, float(e), src, ek, r["D_str"]))
    return {D: b[1] for D, b in sorted(best.items())}


def real_recs(L, ys, sig, ref=None, O=None):
    """the fit records of real data (ys arb, sig floats); d = value - ref as a float, ref (when not given) the value
    of the highest degree at the lowest level in the first variable, so that d resolves differences far below the
    data's sigma (a reference of order 1e-2 from the limit would leave d a float resolution of about 1e-18)"""
    w = [arb(1) / arb(s) for s in sig]
    O = O or ladder_ops(L, w)
    vals = {key: sum((g[i] * ys[j] for i, j in enumerate(idx)), arb(0)) for key, (idx, g, P, gf, Pf) in O.items()}
    if ref is None:
        v0 = next(iter(VARS))
        ref = vals[max((k for k in vals if k[0] == v0 and k[1] == 0), key=lambda k: k[2])]
    recs = {}
    for key, (idx, g, P, gf, Pf) in O.items():
        val = vals[key]
        yw = arb_mat([[ys[j] * w[j]] for j in idx])
        r = P * yw
        rr = np.array([float(r[i, 0]) for i in range(len(idx))])
        s = math.sqrt(sum(float(g[i]) ** 2 * sig[j] ** 2 for i, j in enumerate(idx)))
        recs[key] = dict(d=float(val - ref), s=s, chi2=float(rr @ rr), dof=len(idx) - key[2] - 1,
                         maxres=float(np.max(np.abs(rr))), resid=rr.tolist(), idx=idx, val=val)
    return recs


def three_point(L, ys, sig, out):
    """the K = 1 (dof 1) and K = 2 (exact) fits in each variable on the real points and on T1 at the same D and sigma"""
    tc = mp.mpf("1.64389374512558206")                # Track C's Delta_inf with the direct solves, raw I1095-mac
    tcE = 3.21e-12                                    # tcd_real4.log; a spread over fit choices
    print("three-point check (a check of the machinery, not a result): Track C's Delta_inf %s +- %.2e" % (
        mp.nstr(tc, 18), tcE))
    tv = [a(truth("T1", d)) for d in L]
    rows = []
    for var in VARS:
        for K in (1, 2):
            w = [arb(1) / arb(s) for s in sig]
            g, P = ops(L, w, var, K)
            v = sum((g[i] * ys[i] for i in range(len(L))), arb(0))
            vt = sum((g[i] * tv[i] for i in range(len(L))), arb(0))
            s = math.sqrt(sum(float(g[i]) ** 2 * sig[i] ** 2 for i in range(len(L))))
            yw = arb_mat([[ys[i] * w[i]] for i in range(len(L))])
            r = P * yw
            chi = sum(float(r[i, 0]) ** 2 for i in range(len(L)))
            d_tc = float(v - a(tc))
            d_t1 = float(vt - a(RLIM))
            d_sub = d_tc - d_t1                       # the fit to real - T1, plus T1's limit, less Track C
            rows.append(dict(var=var, K=K, Delta=v.mid().str(20, radius=False), s=s, chi2=chi, minus_trackC=d_tc,
                             T1_err=d_t1, T1_subtracted_minus_trackC=d_sub))
            print("  %-8s K %d: Delta_inf %s  s %.1e  chi2 %.2g  minus Track C %+.2e;  T1 at the same D: fit - "
                  "limit %+.2e;  with T1 subtracted, minus Track C %+.2e" % (
                      var, K, v.mid().str(18, radius=False), s, chi, d_tc, d_t1, d_sub))
    out["three_point"] = rows


def cmd_real():
    folders = [x for x in sys.argv[2:] if not x.startswith("--")]
    pts = full_points(folders)
    print("full points: " + "; ".join("D %s %s sigma %.1e (%s) %s" % (v[4], mp.nstr(v[0], 18), v[1], v[3], v[2])
                                      for d, v in pts.items()))
    L = sorted(pts)
    ys = [a(pts[d][0]) for d in L]
    sig = [pts[d][1] for d in L]
    if MUT == "shift":
        j = len(L) // 2
        ys[j] = ys[j] + a(mp.mpf(30 * sig[j]))
        print("MUTATION: D %g moved by +30 sigma (%.1e)" % (L[j], 30 * sig[j]))
    out = {"points": {pts[d][4]: dict(Delta=mp.nstr(pts[d][0], 20), sigma=pts[d][1], err_kind=pts[d][3],
                                      src=pts[d][2]) for d in L}, "rule": RULE, "F": float(args.get("F", "1"))}
    Fc = out["F"]
    if len(L) < RULE["NMIN"]:
        print("%d points: fewer than NMIN = %d, so the rule quotes nothing" % (len(L), RULE["NMIN"]))
        three_point(L, ys, sig, out)
        if "out" in args:
            json.dump(out, open(args["out"], "w"), indent=1, default=str)
        sys.exit(1)
    recs = real_recs(L, ys, sig)
    est = select(recs, L)
    if est is None:
        print("no D0: no variable has a stable degree at any D_min level; nothing quoted")
        sys.exit(1)
    cen = recs[est["central"]]
    v = cen["val"]
    E = Fc * est["E0"]
    print("Delta_inf = %s +- %.2e (F %.2f x max(h %.2e, s %.2e)); D0 %g, central %s K %d; %d candidates" % (
        v.mid().str(25, radius=False), E, Fc, est["h"], est["s"], est["D0"], est["central"][0], est["K"],
        len(est["cands"])))
    worst = max(abs(x) / 1.0 for x in cen["resid"])
    print("  check 1: central fit's residuals (fit - data)/sigma: %s; max %.2f: %s" % (
        ", ".join("%g: %+.2f" % (L[j], x) for j, x in zip(cen["idx"], cen["resid"])), worst,
        "PASS" if worst <= 3.5 else "FAIL"))
    zs = {}
    for i, Di, Li, ri, Oi, OT in loo_ops(L, sig):          # weights 1/sigma_i
        if Di < est["D0"]:
            continue
        yi = ys[:i] + ys[i + 1:]
        ei = select(real_recs(Li, yi, ri, O=Oi), Li)
        if ei is None:
            zs[Di] = None
            continue
        key = ei["central"]
        idx = Oi[key][0]
        gt = OT[key][0]
        pred = sum((gt[k] * yi[j] for k, j in enumerate(idx)), arb(0))
        k1 = (key[0], key[1], key[2] + 1)
        tr = 0.0
        if k1 in OT:
            tr = float(sum((OT[k1][0][k] * yi[j] for k, j in enumerate(idx)), arb(0)) - pred)
        sd = math.sqrt(sig[i] ** 2 + sum(float(gt[k]) ** 2 * ri[j] ** 2 for k, j in enumerate(idx)) + tr ** 2)
        zs[Di] = float(ys[i] - pred) / sd
    zmax = max((abs(z) for z in zs.values() if z is not None), default=0.0)
    print("  check 2 (leave-one-out, bar %.1f): each point against the rule run without it, (data - prediction)/sd: "
          "%s; max %.2f: %s" % (ZLOO, ", ".join("%g: %s" % (d, "-" if z is None else "%+.2f" % z)
                                                for d, z in sorted(zs.items())), zmax,
                                "PASS" if zmax <= ZLOO else "FAIL"))
    ok = worst <= 3.5 and zmax <= ZLOO
    out.update(Delta_inf=v.mid().str(25, radius=False), E=E, h=est["h"], s=est["s"], D0=est["D0"],
               central=list(est["central"]), cands=[list(c) for c in est["cands"]], check_max_resid=worst,
               check_loo={str(d): z for d, z in zs.items()}, ok=ok)
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1, default=str)
    sys.exit(0 if ok else 1)


def cmd_mkreal():
    """planted records in the direct_s.jsonl format, for a test of `real` through its own reader:
    --ladder=E20 --truth=T1 --sigma=1e-23 --seed=1 --dir=DIR writes DIR/direct_s.jsonl (full system, the noise's sigma
    as a discretization bound, Delta in eight double parts)"""
    lk = args.get("ladder", "E20")
    kind = args.get("truth", "T1")
    sg = float(args.get("sigma", "1e-23"))
    rng = random.Random(5000 + int(args.get("seed", "1")))
    name, L = LADDERS[lk]
    os.makedirs(args["dir"], exist_ok=True)
    with open(os.path.join(args["dir"], "direct_s.jsonl"), "w") as fh:
        for d in L:
            v = truth(kind, d) + mp.mpf(rng.gauss(0, 1)) * mp.mpf(sg)
            parts = []
            for _ in range(8):
                parts.append(float(v - mp.fsum(mp.mpf(x) for x in parts)))
            fh.write(json.dumps(dict(D_str=repr(float(d)), system="full", valid=True, Delta_parts=parts,
                                     F_check=1e-60, disc={"steps halved": {"dDelta_bound": sg}},
                                     planted=dict(truth=kind, sigma=sg, ladder=lk))) + "\n")
    print("wrote %d planted %s records (sigma %.0e, ladder %s) to %s; the truth's limit %s" % (
        len(L), kind, sg, lk, args["dir"], mp.nstr(RLIM, 25)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "planted":
        cmd_planted()
    elif cmd == "controls" and MUT in ("lowdeg", "var"):
        cmd_mutated()
    elif cmd == "controls":
        cmd_controls()
    elif cmd == "probe":
        cmd_probe()
    elif cmd == "budget":
        cmd_budget()
    elif cmd == "loocheck":
        cmd_loocheck()
    elif cmd == "mkreal":
        cmd_mkreal()
    elif cmd == "real":
        cmd_real()
    else:
        raise SystemExit(__doc__)
