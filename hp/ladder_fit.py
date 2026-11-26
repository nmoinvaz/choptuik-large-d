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
- no D0: no quote (exit 1).

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
residual), sigma = export_delta_all.direct_err's err (its largest recorded term; no floor). Its check: the central
fit's residuals |fit - data| / sigma <= 3.5 (exit 1 if not); --mutate=shift moves the middle point by +30 sigma.
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


def ops(Ds, w, var, K):
    """weighted least squares of degree K in var on the points Ds with weights w (1/sigma, relative): g (the weights of
    the value at x = 0 on the data), P (the weighted residual projector, I - H), both arb"""
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
    return [G[0, i] * w[i] for i in range(N)], P


def ladder_ops(L, w=None):
    """{(var, lev, K): (idx, g arb list, P arb_mat, g float array, P float array)} for the ladder L (sorted D list)"""
    w = w or [arb(1)] * len(L)
    out = {}
    for var in VARS:
        for lev in range(RULE["NLEV"] + 1):
            idx = list(range(lev, len(L)))
            if len(idx) < RULE["NMIN"]:
                continue
            for K in range(1, min(RULE["KMAX"], len(idx) - 3) + 1):
                g, P = ops([L[i] for i in idx], [w[i] for i in idx], var, K)
                gf = np.array([float(x) for x in g])
                Pf = np.array([[float(P[i, j]) for j in range(len(idx))] for i in range(len(idx))])
                out[(var, lev, K)] = (idx, g, P, gf, Pf)
    return out


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


def run_planted(O, L, TP, sigma, Z, shift=None, rel=None):
    """the rule on every column of Z at this sigma (point i's noise sigma rel_i z_i): list of select() results (None
    where nothing is quoted), with d = value - limit"""
    S = Z.shape[1]
    rel = np.array(rel or [1.0] * len(L))
    res = []
    for k in range(S):
        z = Z[:, k].copy()
        if shift is not None:
            z[shift[0]] += shift[1]
        recs = {}
        for key, (idx, g, P, gf, Pf) in O.items():
            bias, r0 = TP[key]
            zz = z[idx]
            rr = r0 / sigma + Pf @ zz
            dof = len(idx) - key[2] - 1
            recs[key] = dict(d=bias + sigma * float(gf @ (rel[idx] * zz)),
                             s=sigma * float(np.linalg.norm(gf * rel[idx])),
                             chi2=float(rr @ rr), dof=dof, maxres=float(np.max(np.abs(rr))))
        res.append(select(recs, L))
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
                rows.append(dict(K=K, gain=gain, bias={k: TPs[k][key][0] for k in TRUTHS}))
            rec[var] = rows
            print("  %-8s K: " % var + "  ".join("%d" % r["K"] for r in rows))
            print("  %-8s gain: " % var + "  ".join("%.0e" % r["gain"] for r in rows))
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
def cmd_controls():
    """on T1 at one ladder and sigma, Gaussian seeds 21-30 (not the calibration seeds 1-20): each control must fire.
    shift: the D 1000 point moved by +30 sigma; the rule must quote nothing or drop D 1000 (exit 1 if not).
    tight: E divided by 10; the coverage check must fail in some seed. Informational: the shift size the rule always
    catches, at D 1000 and at the ladder's lowest point."""
    lk = args.get("ladder", "A")
    sg = float(args.get("sigma", "1e-22"))
    name, L = LADDERS[lk]
    Fc = float(args.get("F", "1"))
    out = {}
    O = ladder_ops(L)
    TP = truth_parts(O, L, "T1")
    Z, names = patterns(L, list(range(21, 31)))
    Z = Z[:, :10]
    base = run_planted(O, L, TP, sg, Z)
    q = [r for r in base if r is not None]
    print("controls on T1, ladder %s, sigma %.0e, seeds 21-30; F %.2f" % (lk, sg, Fc))
    print("  unmutated: %d of 10 quoted; |err| / (F E0) max %.2f" % (
        len(q), max(abs(r["d"]) / (Fc * r["E0"]) for r in q) if q else float("nan")))
    out["base"] = dict(quoted=len(q), ratio_max=max(abs(r["d"]) / (Fc * r["E0"]) for r in q) if q else None)
    j = L.index(1000.0)
    sh = run_planted(O, L, TP, sg, Z, shift=(j, 30.0))
    fired = sum(1 for r in sh if r is None or (r["D0"] > 1000.0))
    miss = sum(1 for r in sh if r is not None and abs(r["d"]) > Fc * r["E0"])
    print("  shift (D 1000 by +30 sigma): no quote (or D 1000 dropped) in %d of 10; quoted but |err| > F E in %d" % (
        fired, miss))
    out["shift"] = dict(fired=fired, quoted_wrong=miss)
    tight = sum(1 for r in q if abs(r["d"]) > 0.1 * Fc * r["E0"])
    print("  tight (E / 10): coverage fails in %d of %d" % (tight, len(q)))
    out["tight"] = dict(fired=tight)
    out["shift_scan"] = []
    for at in (1000.0, L[0]):
        jj = L.index(at)
        for k in (3.0, 5.0, 10.0, 20.0):
            rs = run_planted(O, L, TP, sg, Z, shift=(jj, k))
            nq = sum(1 for r in rs if r is None)
            dropped = sum(1 for r in rs if r is not None and r["D0"] > at)
            wrong = sum(1 for r in rs if r is not None and r["D0"] <= at and abs(r["d"]) > Fc * r["E0"])
            print("  shift scan: D %g by %+g sigma: no quote %d, point dropped %d, kept and |err| > F E %d (of 10)" % (
                at, k, nq, dropped, wrong))
            out["shift_scan"].append(dict(D=at, k=k, no_quote=nq, dropped=dropped, wrong=wrong))
    ok = fired == 10 and tight > 0
    print("controls in this process (shift, tight): %s; lowdeg and var run as separate processes "
          "(--mutate=lowdeg, --mutate=var)" % ("FIRE" if ok else "DID NOT FIRE"))
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1)
    sys.exit(0 if ok else 1)


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
            v = mp.fsum(mp.mpf(repr(float(x))) for x in r["Delta_parts"])
            best[D] = (key, (v, float(e), src, ek, r["D_str"]))
    return {D: b[1] for D, b in sorted(best.items())}


def real_recs(L, ys, sig, ref):
    w = [arb(1) / arb(s) for s in sig]
    O = ladder_ops(L, w)
    recs = {}
    for key, (idx, g, P, gf, Pf) in O.items():
        val = sum((g[i] * ys[j] for i, j in enumerate(idx)), arb(0))
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
            rows.append(dict(var=var, K=K, Delta=v.mid().str(20, radius=False), s=s, chi2=chi, minus_trackC=d_tc,
                             T1_err=d_t1))
            print("  %-8s K %d: Delta_inf %s  s %.1e  chi2 %.2g  minus Track C %+.2e;  T1 at the same D: fit - "
                  "limit %+.2e" % (var, K, v.mid().str(18, radius=False), s, chi, d_tc, d_t1))
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
    ref = ys[0]
    out = {"points": {pts[d][4]: dict(Delta=mp.nstr(pts[d][0], 20), sigma=pts[d][1], err_kind=pts[d][3],
                                      src=pts[d][2]) for d in L}, "rule": RULE, "F": float(args.get("F", "1"))}
    Fc = out["F"]
    if len(L) < RULE["NMIN"]:
        print("%d points: fewer than NMIN = %d, so the rule quotes nothing" % (len(L), RULE["NMIN"]))
        three_point(L, ys, sig, out)
        if "out" in args:
            json.dump(out, open(args["out"], "w"), indent=1, default=str)
        sys.exit(1)
    recs = real_recs(L, ys, sig, ref)
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
    print("  check: central fit's residuals (fit - data)/sigma: %s; max %.2f: %s" % (
        ", ".join("%g: %+.2f" % (L[j], x) for j, x in zip(cen["idx"], cen["resid"])), worst,
        "PASS" if worst <= 3.5 else "FAIL"))
    out.update(Delta_inf=v.mid().str(25, radius=False), E=E, h=est["h"], s=est["s"], D0=est["D0"],
               central=list(est["central"]), cands=[list(c) for c in est["cands"]], check_max_resid=worst)
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1, default=str)
    sys.exit(0 if worst <= 3.5 else 1)


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
    elif cmd == "real":
        cmd_real()
    else:
        raise SystemExit(__doc__)
