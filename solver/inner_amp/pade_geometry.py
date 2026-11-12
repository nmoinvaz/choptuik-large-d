#!/usr/bin/env python3
"""Pade geometry of the Borel transform of S_1 against planted two-branch-point and arc functions (issue #541).

B(t) = sum_k k! sigma_k t^(-k-1) (proof/D_entire.md section 7). Its [k/k] Pade approximant in u = 1/t, of
f(u) = sum_k a_k u^k with a_k = k! sigma_k, uses a_0..a_2k, so k = 6..11 needs sigma_0..sigma_22. For a function with
finitely many branch points the Pade poles accumulate on the compact set of minimal capacity outside which the function
is single valued (Stahl 1997, J. Approx. Theory 91, 139), the straight segment for two branch points. Poles that stay on
a bent arc as k grows need singularities along it.

Plants, to the conventions of notes/PLANTED_SELECTION.md 2.2 (sigma_0 = 1; the planted zero factor (1 - x/x_ref),
x_ref = x0(wf; omega_t = 1); density (1-s)^(beta-1) sum_{m<M} c_m T_m(2s-1) along gamma(s); c_m by least squares with
weights 1/|sigma_k| against the real sigma_0..sigma_12 of the node; M = 6, or 8 if the largest relative residual
exceeds 1e-2). Moments by tanh-sinh quadrature at the working precision, so each plant is the continuous Laplace
integral to every digit used, not a finite exponential sum.

  chord2   gamma = the chord P0 -> kappa_*. Exactly two branch points, P0 and kappa_*.              plant (a)
  arcjump  gamma = the cubic arc of planted_plants.Arc, density times a step weight (1, 2, 1, 2) on four equal
           pieces of s. Each step is a log branch point on the arc, so the singular support is the arc.  plant (b)
  arcmid   the arc with one step at s = 1/2 (a third branch point at the apex). Context.
  arc      the arc with no step (P-ARC as a continuous integral; its density continues off the arc). Context.
  repo-arc, repo-chord   P-ARC and P-ARC-B chord of planted_plants.py itself (64-node exponential sums). Context.

Geometry. Chord coordinates at each node relative to P0 = D_entire 7.1's first [6/6] pole and P1 = kappa_* (D_entire
section 1), the end points of every plant: s_p = Re((t - P0)/(P1 - P0)), d_p = Im((t - P0)/(P1 - P0)) |P1 - P0|. The
arc fraction R_k = sum d_p d_arc(s_p) / sum d_arc(s_p)^2 over the geometry poles with 0.05 <= s_p <= 0.95 projects the
pole offsets on the arc's offset profile: 0 on the chord, 1 on the arc.

    python3 pade_geometry.py selftest
    python3 pade_geometry.py plants OUTDIR [--mutate=sigma7] [--allow_unraw]
    python3 pade_geometry.py real OUTDIR [--mutate=sigma7] [--allow_unraw]

`real` reads OUTDIR/plants.json (the frozen tolerance comes from the plants). Exit 1 when a check fails.
"""
import json
import os
import random
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "hp"))
import planted_plants as pp                                            # noqa: E402

NODES = (0.99, 1.0, 1.01)
KS = range(6, 12)                   # [k/k], k = 6..11, from sigma_0..sigma_2k
KM = 24                             # plant coefficients kept (0..23)
DPS_PLANT = 60                      # plant construction
DPS_PADE = 120                      # Pade; >= 2x the 50-60 digits of the plants and 6x the 20 of the real strings
TOL_DOUBLET = mp.mpf("1e-3")        # a pole with a numerator zero closer than this (in t) is a Froissart doublet
NOISE_REL = 2e-12                   # relative noise per sigma_k, k >= 1 (above the largest plateau error, 1.7e-12)
NSEED = 10
REAL = {"J1250": os.path.join(HERE, "scan45J1250"), "J1500": os.path.join(HERE, "scan45J1500")}
FROZEN = {                          # written before any plant or real Pade pole of this script was computed
    "R_window": [0.05, 0.95],
    "geometry_window": "non-doublet, 0.3 < |t| < 3, -0.1 <= s <= 1.1, |d| <= 0.5",
    "T1_chord2": "R_k decreasing over k = 8..11, each step larger than the noise spread, and R_11 <= 0.6",
    "T2_arcjump": "|R_k - 1| <= 0.25 for k = 8..11",
    "T3_separation": "S = R_11(arcjump) - R_11(chord2) >= 0.3, else the test cannot discriminate at k <= 11",
    "T4_branch_points": "end-pole extrapolation t(k) = t_inf + c/k^2 over k = 7..11 recovers P0 and kappa_* within 0.05; "
                        "ratio law over k = 12..21 recovers kappa_* within 0.02",
    "match": "real matches plant X if max_{k=9..11} |R_k(real) - R_k(X)| <= tau, tau = max(S/3, 3 nu), nu the largest "
             "noise spread of R at k = 9..11; it must not match the other plant at k = 11",
    "converge": "real R_k, k = 8..11: every |R_{k+1} - R_k| <= 0.15 and |R_11 - R_10| <= |R_9 - R_8| + 2 nu",
    "mutation": "--mutate=sigma7 (sigma_7 x 1.01) must move R_11 by more than tau or a pole by more than 100x the noise",
}


# ---------------------------------------------------------------- input
def load_scan(d):
    out = {}
    for wf in NODES:
        fn = [f for f in os.listdir(d) if f.startswith(f"sigF_w{wf:.4f}") and f.endswith(".json")]
        if not fn:
            continue
        r = json.load(open(os.path.join(d, fn[0])))
        out[wf] = [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in r["sigma_str"]]
    return out


def mutate(sig, spec):
    if not spec:
        return sig
    k = int(spec.replace("sigma", ""))
    s = list(sig)
    s[k] = s[k] * mp.mpf("1.01")
    return s


# ---------------------------------------------------------------- plants (continuous Laplace integrals)
def ts_rule(h=mp.mpf(1) / 32, tmax=mp.mpf("4.5")):
    """tanh-sinh nodes on [0, 1]: (x, 1 - x, weight)"""
    out = []
    n = int(tmax / h)
    for j in range(-n, n + 1):
        tau = j * h
        y = mp.pi / 2 * mp.sinh(tau)
        out.append((1 / (1 + mp.exp(-2 * y)), 1 / (1 + mp.exp(2 * y)), h * mp.pi / 4 * mp.cosh(tau) / mp.cosh(y) ** 2))
    return out


class Plant:
    """S(x) = (1 - x/x_ref) L(x) / L(0), L = int_0^1 w(s) (1-s)^ex sum c_m T_m(2s-1) exp(x gamma(s)) gamma'(s) ds"""

    def __init__(self, wf, kind, real_sig):
        mp.mp.dps = DPS_PLANT
        self.wf, self.kind = wf, kind
        self.arc = pp.Arc(wf, "chord" if kind == "chord2" else "arc")
        self.P0, self.P1 = self.arc.P0, self.arc.P1
        self.ex = pp.mpc(pp.BETA[wf]) - 1
        pieces = {"chord2": [(0, 1, 1)], "arc": [(0, 1, 1)], "arcmid": [(0, 0.5, 1), (0.5, 1, 2)],
                  "arcjump": [(0, 0.25, 1), (0.25, 0.5, 2), (0.5, 0.75, 1), (0.75, 1, 2)]}[kind]
        self.pieces = pieces
        rule = ts_rule()
        nodes = []
        for a, b, w in pieces:
            a, b = mp.mpf(a), mp.mpf(b)
            for x, omx, wt in rule:
                s = a + (b - a) * x
                nodes.append((s, (1 - b) + (b - a) * omx, (b - a) * wt * w))
        self.nodes = nodes
        xref = pp.x0_planted(wf, "1.0000")
        self.pref = pp.factor([xref])
        for M in (6, 8):
            self.mom = self.moments(M)
            res = self.fit(real_sig, M)
            if max(res) <= 1e-2:
                break
        self.M, self.resid = M, res

    def moments(self, M):
        mom = [[mp.mpc(0)] * (KM + 1) for _ in range(M)]
        for s, oms, w in self.nodes:
            g, dg = self.arc.g(s), self.arc.dg(s)
            base = w * mp.power(oms, self.ex) * dg
            tm = [mp.chebyt(m, 2 * s - 1) for m in range(M)]
            gk = mp.mpc(1)
            for k in range(KM + 1):
                for m in range(M):
                    mom[m][k] += base * tm[m] * gk
                gk *= g
        return [[mom[m][k] / mp.factorial(k) for k in range(KM + 1)] for m in range(M)]

    def fit(self, sig_real, M, K=12):
        """pp.Laplace.fit: least squares with weights 1/|sigma_k| of (pref L)_k against sigma_k, k = 0..K"""
        cols = [pp.polymul(self.pref, self.mom[m], K + 1) for m in range(M)]
        A = np.array([[complex(cols[m][k] / abs(sig_real[k])) for m in range(M)] for k in range(K + 1)])
        b = np.array([complex(sig_real[k] / abs(sig_real[k])) for k in range(K + 1)])
        Ar = np.block([[A.real, -A.imag], [A.imag, A.real]])
        sol, *_ = np.linalg.lstsq(Ar, np.concatenate([b.real, b.imag]), rcond=None)
        self.c = [mp.mpc(complex(z)) for z in sol[:M] + 1j * sol[M:]]
        sig = self.sigma(M)
        return [float(abs(sig[k] - sig_real[k]) / abs(sig_real[k])) for k in range(K + 1)]

    def sigma(self, M=None):
        M = M or self.M
        lc = [sum(self.c[m] * self.mom[m][k] for m in range(M)) for k in range(KM + 1)]
        s = pp.polymul(self.pref, lc, KM + 1)
        return [z / s[0] for z in s]


def repo_plant(kind):
    """planted_plants.ArcFamily (64-node exponential sums, fitted to scan25phys), the planted zero at omega_t = 1"""
    fam = pp.ArcFamily(kind)
    return {wf: s[:KM + 1] for wf, s in fam.nodes("1.0000").items()}, fam


# ---------------------------------------------------------------- Pade
def pade(sig, k):
    """poles of [k/k] in t, with the distance to the nearest numerator zero and the residue of f at u_p"""
    a = [mp.factorial(j) * sig[j] for j in range(2 * k + 1)]
    p, q = mp.pade(a, k, k)
    while len(p) > 1 and p[-1] == 0:
        p = p[:-1]
    us = mp.polyroots(q[::-1], maxsteps=500, extraprec=2 * mp.mp.prec)
    uz = mp.polyroots(p[::-1], maxsteps=500, extraprec=2 * mp.mp.prec) if len(p) > 1 else []
    tz = [1 / u for u in uz if u != 0]
    dq = [j * q[j] for j in range(1, len(q))]
    out = []
    for u in us:
        t = 1 / u
        dz = min((abs(t - z) for z in tz), default=mp.inf)
        res = mp.polyval(p[::-1], u) / mp.polyval(dq[::-1], u)
        out.append({"t": t, "dz": dz, "res": res})
    return out


class Frame:
    """chord coordinates and the arc's offset profile at one node"""

    def __init__(self, wf):
        mp.mp.dps = DPS_PLANT
        arc = pp.Arc(wf, "arc")
        self.P0, self.P1 = complex(arc.P0), complex(arc.P1)
        self.D = self.P1 - self.P0
        ss = np.linspace(0, 1, 4001)
        z = np.array([complex(arc.g(mp.mpf(s))) for s in ss])
        w = (z - self.P0) / self.D
        self.prof_s, self.prof_d = w.real, w.imag * abs(self.D)
        assert np.all(np.diff(self.prof_s) > 0), "arc chord coordinate not monotone"

    def coords(self, t):
        w = (complex(t) - self.P0) / self.D
        return w.real, w.imag * abs(self.D)

    def d_arc(self, s):
        return float(np.interp(s, self.prof_s, self.prof_d))


def geometry(poles, fr):
    geo, dbl, out = [], [], []
    for p in poles:
        t = complex(p["t"])
        s, d = fr.coords(t)
        rec = {"t": [t.real, t.imag], "s": s, "d": d, "dz": float(p["dz"]), "abs_res": float(abs(p["res"]))}
        if p["dz"] < TOL_DOUBLET:
            rec["class"] = "doublet"
            dbl.append(rec)
        elif 0.3 < abs(t) < 3 and -0.1 <= s <= 1.1 and abs(d) <= 0.5:
            rec["class"] = "geometry"
            geo.append(rec)
        else:
            rec["class"] = "outside"
        out.append(rec)
    lo, hi = FROZEN["R_window"]
    num = sum(r["d"] * fr.d_arc(r["s"]) for r in geo if lo <= r["s"] <= hi)
    den = sum(fr.d_arc(r["s"]) ** 2 for r in geo if lo <= r["s"] <= hi)
    R = num / den if den > 0 else None
    geo.sort(key=lambda r: r["s"])
    return {"poles": sorted(out, key=lambda r: r["s"]), "R": R, "n_geo": len(geo), "n_doublet": len(dbl),
            "bow": max((r["d"] for r in geo), default=None), "near": geo[0]["t"] if geo else None,
            "far": geo[-1]["t"] if geo else None, "min_dz_geo": min((r["dz"] for r in geo), default=None)}


def noisy(sig, seed, eps=NOISE_REL):
    rng = random.Random(seed)
    return [sig[0]] + [z * (1 + eps * mp.mpc(rng.gauss(0, 1), rng.gauss(0, 1)) / mp.sqrt(2)) for z in sig[1:]]


def analyse(sig, fr, noise=True):
    """[k/k] geometry for k in KS, with the noise spread of R and of each geometry pole"""
    mp.mp.dps = DPS_PADE
    sig = [mp.mpc(z) for z in sig]
    out = {}
    for k in KS:
        g = geometry(pade(sig, k), fr)
        if noise:
            dR, dmax = 0.0, 0.0
            for seed in range(NSEED):
                gn = geometry(pade(noisy(sig, 1000 * k + seed), k), fr)
                if g["R"] is not None and gn["R"] is not None:
                    dR = max(dR, abs(gn["R"] - g["R"]))
                tn = [complex(*r["t"]) for r in gn["poles"]]
                for r in g["poles"]:
                    if r["class"] == "geometry":
                        dmax = max(dmax, min(abs(complex(*r["t"]) - z) for z in tn))
            g["noise_dR"], g["noise_dpole"] = dR, dmax
        out[k] = g
    return out


def extrapolate(res, key):
    """t(k) = t_inf + c/k^2, least squares over k = 7..11"""
    ks = [k for k in range(7, 12) if res[k][key] is not None]
    A = np.array([[1, 1 / k ** 2] for k in ks], dtype=complex)
    b = np.array([complex(*res[k][key]) for k in ks])
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    return complex(c[0])


def ratio_kappa(sig, lo=12, hi=21):
    kap, beta = pp.ratio_law([mp.mpc(z) for z in sig], lo, hi)
    return complex(kap), complex(beta)


def summary(res):
    return {k: {"R": res[k]["R"], "n_geo": res[k]["n_geo"], "n_doublet": res[k]["n_doublet"], "bow": res[k]["bow"],
                "noise_dR": res[k].get("noise_dR"), "noise_dpole": res[k].get("noise_dpole"),
                "min_dz_geo": res[k]["min_dz_geo"]} for k in res}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, complex):
        return [o.real, o.imag]
    if isinstance(o, (mp.mpf, mp.mpc)):
        return jsonable(complex(o))
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


def table(name, res):
    print(f"  {name}: k  R_k  (noise dR)  n_geo  doublets  bow  noise_dpole")
    for k in KS:
        r = res[k]
        R = "  -  " if r["R"] is None else f"{r['R']:+.3f}"
        print(f"     {k:2d}  {R}  ({r.get('noise_dR', 0):.1e})  {r['n_geo']:2d}  {r['n_doublet']}  "
              f"{(r['bow'] or 0):+.3f}  {r.get('noise_dpole', 0):.1e}")


# ---------------------------------------------------------------- stages
def selftest():
    """(1) tanh-sinh moments of the chord plant equal the exact Beta-function moments; (2) the Pade code returns the
    poles of a planted rational function; (3) each check fires on a mutation."""
    mp.mp.dps = DPS_PLANT
    ok = True
    wf = 1.0
    real = load_scan(REAL["J1250"])[wf]
    pl = Plant(wf, "chord2", real)
    D = pl.P1 - pl.P0
    # T_m(2s - 1) as exact polynomials in s
    T = [[mp.mpf(1)], [mp.mpf(-1), mp.mpf(2)]]
    while len(T) < pl.M:
        a, b = T[-1], T[-2]
        n = [mp.mpf(0)] * (len(a) + 1)
        for i, ci in enumerate(a):
            n[i] -= 2 * ci
            n[i + 1] += 4 * ci
        for i, ci in enumerate(b):
            n[i] -= ci
        T.append(n)

    def beta_err(mom):
        err = 0
        for m in (0, 3, pl.M - 1):
            for k in (0, 7, 20):
                ex = 0
                for j in range(k + 1):
                    for i, ci in enumerate(T[m]):
                        ex += mp.binomial(k, j) * pl.P0 ** (k - j) * D ** j * ci * mp.beta(j + i + 1, pl.ex + 1)
                ex *= D / mp.factorial(k)
                err = max(err, abs(mom[m][k] - ex) / abs(ex))
        return err
    err = beta_err(pl.mom)
    print(f"selftest 1: tanh-sinh vs Beta moments, max rel diff {float(err):.1e} (pass < 1e-40)")
    ok &= err < mp.mpf("1e-40")
    mom = [list(r) for r in pl.mom]
    mom[3][7] *= 1 + mp.mpf("1e-30")
    em = beta_err(mom)
    print(f"selftest 1, mutation moment(3, 7) x (1 + 1e-30): rel diff {float(em):.1e} (must exceed 1e-40)")
    ok &= em > mp.mpf("1e-40")
    mp.mp.dps = DPS_PADE
    tp = [mp.mpc("0.8", "0.2"), mp.mpc("1.0", "0.6"), mp.mpc("1.2", "1.0")]
    rp = [mp.mpc(1), mp.mpc("0.5", "-0.3"), mp.mpc("0.2", "0.1")]
    sig = [sum(r * t ** j for r, t in zip(rp, tp)) / mp.factorial(j) for j in range(2 * 3 + 1)]
    got = sorted((p["t"] for p in pade(sig, 3)), key=lambda z: float(mp.re(z)))
    e2 = max(abs(a - b) for a, b in zip(got, tp))
    print(f"selftest 2: [3/3] of a 3-pole sum returns its poles to {float(e2):.1e} (pass < 1e-60)")
    ok &= e2 < mp.mpf("1e-60")
    sig_m = list(sig)
    sig_m[4] *= mp.mpf("1.01")
    got = sorted((p["t"] for p in pade(sig_m, 3)), key=lambda z: float(mp.re(z)))
    e3 = max(abs(a - b) for a, b in zip(got, tp))
    print(f"selftest 2, mutation sigma_4 x 1.01: poles move by {float(e3):.1e} (must exceed 1e-6)")
    ok &= e3 > 1e-6
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


def run_plants(outdir, mut):
    real = load_scan(REAL["J1250"])
    rec = {"frozen": FROZEN, "mutate": mut, "plants": {}}
    frames = {wf: Frame(wf) for wf in NODES}
    kinds = ("chord2", "arcjump", "arcmid", "arc")
    for kind in kinds:
        for wf in (NODES if kind in ("chord2", "arcjump") else (1.0,)):
            pl = Plant(wf, kind, real[wf])
            sig = mutate(pl.sigma(), mut)
            mp.mp.dps = DPS_PADE
            res = analyse(sig, frames[wf])
            kap, beta = ratio_kappa(sig)
            near, far = extrapolate(res, "near"), extrapolate(res, "far")
            P0, P1 = complex(pl.P0), complex(pl.P1)
            info = {"M": pl.M, "fit_resid_max": max(pl.resid), "kappa_ratio": kap, "beta_ratio": beta,
                    "near_inf": near, "far_inf": far, "err_near": abs(near - P0), "err_far": abs(far - P1),
                    "err_kappa_ratio": abs(kap - P1), "P0": P0, "kappa_star": P1,
                    "sigma13_22_rel_vs_real": [float(abs(sig[k] - real[wf][k]) / abs(real[wf][k]))
                                               for k in range(13, 23)]}
            rec["plants"].setdefault(kind, {})[wf] = {"info": info, "summary": summary(res), "orders": res}
            print(f"{kind} wf {wf}: M {pl.M}, fit resid {max(pl.resid):.1e}; kappa(ratio 12-21) {kap:.4f} "
                  f"(err {abs(kap - P1):.3f}); near_inf {near:.3f} (err {abs(near - P0):.3f}), far_inf {far:.3f} "
                  f"(err {abs(far - P1):.3f})")
            table(f"{kind} {wf}", res)
            sys.stdout.flush()
    for kind in ("arc", "chord"):
        sigs, fam = repo_plant(kind)
        sig = mutate(sigs[1.0], mut)
        res = analyse(sig, frames[1.0])
        rec["plants"].setdefault("repo-" + kind, {})[1.0] = {"info": {"fit_resid_max": max(fam.resid[1.0])},
                                                             "summary": summary(res), "orders": res}
        table(f"repo-{kind} 1.0", res)
    # criteria T1-T4 at every node of the two plants
    v = {}
    for wf in NODES:
        c, a = rec["plants"]["chord2"][wf], rec["plants"]["arcjump"][wf]
        Rc = [c["orders"][k]["R"] for k in range(8, 12)]
        nc = [c["orders"][k]["noise_dR"] for k in range(8, 12)]
        t1 = all(Rc[i + 1] < Rc[i] - nc[i + 1] for i in range(3)) and Rc[-1] <= 0.6
        t2 = all(abs(a["orders"][k]["R"] - 1) <= 0.25 for k in range(8, 12))
        S = a["orders"][11]["R"] - Rc[-1]
        t3 = S >= 0.3
        t4 = all(p["info"]["err_near"] <= 0.05 and p["info"]["err_far"] <= 0.05 and p["info"]["err_kappa_ratio"] <= 0.02
                 for p in (c, a))
        nu = max(max(p["orders"][k]["noise_dR"] for k in (9, 10, 11)) for p in (c, a))
        v[wf] = {"T1": t1, "T2": t2, "T3": t3, "S": S, "T4": t4, "nu": nu, "tau": max(S / 3, 3 * nu)}
        print(f"wf {wf}: T1 {t1}  T2 {t2}  T3 {t3} (S = {S:.3f})  T4 {t4}  nu {nu:.1e}  tau {v[wf]['tau']:.3f}")
    rec["verdict"] = v
    name = "plants.json" if not mut else f"plants_mutate_{mut}.json"
    json.dump(jsonable(rec), open(os.path.join(outdir, name), "w"), indent=1)
    return all(x["T1"] and x["T2"] and x["T3"] and x["T4"] for x in v.values())


def run_real(outdir, mut):
    plants = json.load(open(os.path.join(outdir, "plants.json")))
    frames = {wf: Frame(wf) for wf in NODES}
    rec = {"frozen": FROZEN, "mutate": mut, "real": {}}
    for lab, d in REAL.items():
        data = load_scan(d)
        for wf, sig in data.items():
            sig = mutate(sig[:2 * max(KS) + 1], mut)
            res = analyse(sig, frames[wf], noise=(lab == "J1250"))
            kap, beta = ratio_kappa(sig)
            rec["real"].setdefault(lab, {})[wf] = {"summary": summary(res), "orders": res, "kappa_ratio": kap,
                                                   "beta_ratio": beta, "near_inf": extrapolate(res, "near"),
                                                   "far_inf": extrapolate(res, "far")}
            table(f"real {lab} {wf}", res)
            print(f"     kappa (ratio 12-21) {kap:.4f}, near_inf {extrapolate(res, 'near'):.3f}, "
                  f"far_inf {extrapolate(res, 'far'):.3f}")
    # J1250 against J1500 at wf 1.00: largest geometry-pole displacement
    a, b = rec["real"]["J1250"][1.0]["orders"], rec["real"]["J1500"][1.0]["orders"]
    dj = {}
    for k in KS:
        tb = [complex(*r["t"]) for r in b[k]["poles"]]
        dj[k] = max(min(abs(complex(*r["t"]) - z) for z in tb) for r in a[k]["poles"] if r["class"] == "geometry")
    rec["J_difference"] = dj
    print("J1250 vs J1500 at wf 1.00, largest geometry-pole shift:", {k: f"{v:.1e}" for k, v in dj.items()})
    verdict = {}
    for wf in NODES:
        r = rec["real"]["J1250"][wf]["orders"]
        pv = plants["verdict"][str(wf)]
        tau, nu = pv["tau"], pv["nu"]
        Rr = {k: r[k]["R"] for k in KS}
        dist = {}
        for kind in ("chord2", "arcjump"):
            pk = plants["plants"][kind][str(wf)]["orders"]
            dist[kind] = {k: abs(Rr[k] - pk[str(k)]["R"]) for k in KS}
        match = {kind: max(dist[kind][k] for k in (9, 10, 11)) <= tau for kind in dist}
        steps = [abs(Rr[k + 1] - Rr[k]) for k in range(8, 11)]
        conv = all(s <= 0.15 for s in steps) and steps[2] <= steps[0] + 2 * nu
        one = [kd for kd in match if match[kd]]
        other_ok = len(one) == 1 and all(dist[o][11] > tau for o in dist if o != one[0])
        verdict[wf] = {"R": Rr, "dist": dist, "tau": tau, "match": match, "converge": conv, "steps": steps,
                       "PASS": conv and other_ok, "matched": one}
        print(f"wf {wf}: R_k {' '.join(f'{Rr[k]:+.3f}' for k in KS)}; tau {tau:.3f}; match chord2 {match['chord2']} "
              f"(max {max(dist['chord2'][k] for k in (9, 10, 11)):.3f}), arcjump {match['arcjump']} "
              f"(max {max(dist['arcjump'][k] for k in (9, 10, 11)):.3f}); converge {conv}; "
              f"{'PASS' if verdict[wf]['PASS'] else 'FAIL'}")
    rec["verdict"] = verdict
    name = "real.json" if not mut else f"real_mutate_{mut}.json"
    json.dump(jsonable(rec), open(os.path.join(outdir, name), "w"), indent=1)
    return verdict[1.0]["PASS"]


def main():
    argv = sys.argv[1:]
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    argv = [a for a in argv if not a.startswith("--mutate=")]
    if not argv or argv[0] == "selftest":
        sys.exit(0 if selftest() else 1)
    stage, outdir = argv[0], argv[1]
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    os.makedirs(outdir, exist_ok=True)
    ok = run_plants(outdir, mut) if stage == "plants" else run_real(outdir, mut)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
