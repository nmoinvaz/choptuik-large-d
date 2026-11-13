#!/usr/bin/env python3
"""The far end of the Borel singular support of S_1, from sigma_0..sigma_22, calibrated on plants (issue #547).

B(t) = sum_k a_k t^(-k-1), a_k = k! sigma_k (proof/D_entire.md section 7). In u = 1/t, f(u) = sum_k a_k u^k has its
nearest singularity at u = 1/kappa_far, the far end of the singular support, which sets the type of S_1.

Estimator (DA2). Second-order inhomogeneous differential approximants, Q2 f'' + Q1 f' + Q0 f = P with polynomial
Q_j and P matched to a_0..a_n (Guttmann and Joyce 1972, J. Phys. A 5, L81; Hunter and Baker 1979, Phys. Rev. B 19,
3808). The singular points of the approximant are the zeros of Q2. For each member of a fixed ensemble the farthest zero
t = 1/u with 1.2 < |t| < 3 and Im t > 0 is taken, and the estimate is the componentwise median over the ensemble. The
ensemble is ENSEMBLE: degrees (d0, d1, d2) with d2 = 4, 5, 6 and d0 = d1 = d2 or d2 - 1, and the degree of P chosen so
that the conditions use a_0..a_22 or a_0..a_20. A function whose Borel transform is a density with an algebraic end
point times an analytic factor satisfies such an equation approximately, with the analytic background in P.

Plants. As pade_geometry.py (notes/PLANTED_SELECTION.md 2.2): density (1-s)^ex sum_m c_m T_m(2s-1) on a path from
P0 (D_entire 7.1's first [6/6] pole) to P1, the planted zero factor, c_m fitted to the real sigma_0..sigma_12 of the
node. Here P1 is moved: radially, to |P1| = R, from kappa_* (D_entire section 1). Kinds chord2, arc, arcjump, arcmid,
arcsteps as in pade_geometry.py; exponent variants ex = beta - 1 +- 1; and 'stretch' plants, which keep the density
fitted at kappa_* and only stretch the chord to |P1| = R, so their low orders no longer match the data.

    python3 pade_farend.py selftest [--mutate=...]
    python3 pade_farend.py plants OUTDIR [--allow_unraw]
    python3 pade_farend.py real OUTDIR [--mutate=scale1.03|sigma7] [--allow_unraw]

Exit 1 when a frozen criterion fails. Error bars: 'noise' is the largest shift over NSEED seeds of relative complex
Gaussian noise of the measured per-order errors (sigma_err_fit.py, as pade_geometry.py --noise=measured); 'spread' is
the median absolute deviation of the ensemble members from the median; 'calibration' is the largest error on the plants.
None of them is a bound.
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
import pade_geometry as pg                                             # noqa: E402
import planted_plants as pp                                            # noqa: E402

N_COEF = 23                         # sigma_0..sigma_22
DPS_DA = 100
NSEED = 10
WINDOW = (1.2, 3.0)                 # |t| range of an admissible zero of Q2, with Im t > 0
ENSEMBLE = [(d2 - drop, d2 - drop, d2) for d2 in (4, 5, 6) for drop in (0, 1)]
FROZEN = {                          # written after the development plants D1-D5 and before the validation plants and
                                    # any real run of this script
    "estimator": "DA2: median over ENSEMBLE of the farthest zero of Q2 with 1.2 < |t| < 3, Im t > 0",
    "C1_recovery": "every plant: |estimate - P1| <= 0.02",
    "C2_response": "every move pair (end moved by 0.05): |estimate(moved) - estimate(base)| > 0.03",
    "C3_noise": "every plant: noise spread of the estimate <= 0.01 at the measured per-order errors",
    "B1_baseline": "the ratio law over k = 12..21 misses every plant with P1 = kappa_* by more than 0.05 (#541's "
                   "0.10-0.14; checks that these plants are #541's)",
    "real_quote": "kappa_far = estimate at wf 1.00 (J1250), with the calibration (largest plant error), the noise "
                  "spread, the ensemble spread, the J spread (J1250 vs J1500) and the three nodes listed apart",
    "real_range": "the real estimate counts as calibrated only if |kappa_far| lies inside the planted range of |P1|, "
                  "1.62..1.90",
    "mutation": "--mutate=scale1.03 (sigma_k x 1.03^k, which scales the Borel plane by 1.03) must move the estimate "
                "to 1.03 x estimate within 1e-6 and by more than 0.03; --mutate=sigma7 (sigma_7 x 1.01, which leaves "
                "every singularity where it is) is reported, not judged",
}


# ---------------------------------------------------------------- plants with a moved far end
class FarArc(pp.Arc):
    """pp.Arc with the far end P1 given: the cubic through D_entire 7.1's poles, end points held at P0 and P1"""

    def __init__(self, wf, kind, P1):
        pts = [pp.mpc(p) for p in pp.POLES[wf]] + [mp.mpc(P1)]
        P0, P1 = pts[0], pts[-1]
        ch = [mp.mpf(0)]
        for a, b in zip(pts, pts[1:]):
            ch.append(ch[-1] + abs(b - a))
        s = [c / ch[-1] for c in ch]
        A = mp.matrix([[si * (1 - si), si * si * (1 - si)] for si in s[1:-1]])
        b = mp.matrix([pt - P0 - (P1 - P0) * si for pt, si in zip(pts[1:-1], s[1:-1])])
        e = mp.lu_solve(A.H * A, A.H * b)
        self.P0, self.P1, self.e0, self.e1, self.kind = P0, P1, e[0], e[1], kind
        self.fit_pts = list(zip(s, pts))


PIECES = {"chord2": [(0, 1, 1)], "arc": [(0, 1, 1)], "arcmid": [(0, 0.5, 1), (0.5, 1, 2)],
          "arcjump": [(0, 0.25, 1), (0.25, 0.5, 2), (0.5, 0.75, 1), (0.75, 1, 2)],
          "arcsteps": [(0, 0.25, 1), (0.25, 0.5, 2), (0.5, 0.75, 3), (0.75, 1, 4)]}


def kstar(wf):
    return mp.mpc(complex(pp.KSTAR[wf]))


def far_point(wf, R):
    """kappa_* moved radially to modulus R (R None: kappa_* itself)"""
    k = kstar(wf)
    return k if R is None else k * mp.mpf(str(R)) / abs(k)


class FarPlant(pg.Plant):
    """pade_geometry.Plant with the far end at P1 = far_point(wf, R) and the end exponent ex (default beta - 1).
    stretch: keep c_m from the plant at kappa_* (same kind, wf, ex) and only move the end, without refitting."""

    def __init__(self, wf, kind, real_sig, R=None, dex=0, stretch=False):
        mp.mp.dps = pg.DPS_PLANT
        self.wf, self.kind = wf, kind
        P1 = far_point(wf, R)
        self.arc = FarArc(wf, "chord" if kind == "chord2" else "arc", P1)
        self.P0, self.P1 = self.arc.P0, self.arc.P1
        self.ex = pp.mpc(pp.BETA[wf]) - 1 + dex
        self.pieces = PIECES[kind]
        rule = pg.ts_rule()
        self.nodes = []
        for a, b, w in self.pieces:
            a, b = mp.mpf(a), mp.mpf(b)
            for x, omx, wt in rule:
                self.nodes.append((a + (b - a) * x, (1 - b) + (b - a) * omx, (b - a) * wt * w))
        self.pref = pp.factor([pp.x0_planted(wf, "1.0000")])
        if stretch:
            base = FarPlant(wf, kind, real_sig, None, dex)
            mp.mp.dps = pg.DPS_PLANT
            self.M, self.c, self.resid = base.M, base.c, None
            self.mom = self.moments(self.M)
            sig = self.sigma()
            self.resid = [float(abs(sig[k] - real_sig[k]) / abs(real_sig[k])) for k in range(13)]
            return
        for M in (6, 8):
            self.mom = self.moments(M)
            res = self.fit(real_sig, M)
            if max(res) <= 1e-2:
                break
        self.M, self.resid = M, res


# name: (kind, wf, R, dex, stretch, set). D = development (used to choose ENSEMBLE), V = validation.
K0 = None
PLANTS = {
    "D1 chord2": ("chord2", 1.0, K0, 0, False, "D"),
    "D2 arcjump": ("arcjump", 1.0, K0, 0, False, "D"),
    "D3 chord2 R1.80": ("chord2", 1.0, 1.80, 0, False, "D"),
    "D4 chord2 R1.85": ("chord2", 1.0, 1.85, 0, False, "D"),
    "D5 arc R1.85": ("arc", 1.0, 1.85, 0, False, "D"),
    "V arc": ("arc", 1.0, K0, 0, False, "V"),
    "V arcmid": ("arcmid", 1.0, K0, 0, False, "V"),
    "V arcsteps": ("arcsteps", 1.0, K0, 0, False, "V"),
    "V arc R1.80": ("arc", 1.0, 1.80, 0, False, "V"),
    "V arcjump R1.80": ("arcjump", 1.0, 1.80, 0, False, "V"),
    "V arcjump R1.85": ("arcjump", 1.0, 1.85, 0, False, "V"),
    "V chord2 wf0.99": ("chord2", 0.99, K0, 0, False, "V"),
    "V chord2 wf1.01": ("chord2", 1.01, K0, 0, False, "V"),
    "V arcjump wf0.99": ("arcjump", 0.99, K0, 0, False, "V"),
    "V arcjump wf1.01": ("arcjump", 1.01, K0, 0, False, "V"),
    "V chord2 R1.85 wf0.99": ("chord2", 0.99, 1.85, 0, False, "V"),
    "V chord2 R1.85 wf1.01": ("chord2", 1.01, 1.85, 0, False, "V"),
    "V chord2 ex-1": ("chord2", 1.0, K0, -1, False, "V"),
    "V chord2 ex+1": ("chord2", 1.0, K0, 1, False, "V"),
    "V chord2 R1.85 ex-1": ("chord2", 1.0, 1.85, -1, False, "V"),
    "V chord2 R1.85 ex+1": ("chord2", 1.0, 1.85, 1, False, "V"),
    "V chord2 stretch R1.85": ("chord2", 1.0, 1.85, 0, True, "V"),
    "V arc stretch R1.85": ("arc", 1.0, 1.85, 0, True, "V"),
}
for _kind in ("chord2", "arc", "arcjump"):
    for _R in ("-0.05", "+0.05", 1.75, 1.90):
        _r = float(abs(complex(pp.KSTAR[1.0]))) + float(_R) if isinstance(_R, str) else _R
        PLANTS[f"V {_kind} R{_R}"] = (_kind, 1.0, _r, 0, False, "V")
# end moved by 0.05: (base, moved)
PAIRS = []
for _kind, _d in (("chord2", "D1 chord2"), ("arc", "V arc"), ("arcjump", "D2 arcjump")):
    PAIRS += [(_d, f"V {_kind} R+0.05"), (_d, f"V {_kind} R-0.05")]
PAIRS += [("D3 chord2 R1.80", "D4 chord2 R1.85"), ("V arc R1.80", "D5 arc R1.85"),
          ("V arcjump R1.80", "V arcjump R1.85")]


# ---------------------------------------------------------------- differential approximants
def da_zeros(sig, degs, lp, n_coef=N_COEF):
    """zeros t = 1/u of Q_K for the K-th order inhomogeneous DA sum_j Q_j f^(j) = P, deg Q_j = degs[j], deg P = lp,
    Q_K(0) = 1, matched to the coefficients u^0..u^(m-1) with m the number of unknowns. None if a_(m-1+K) is not
    available."""
    a = [mp.factorial(k) * sig[k] for k in range(n_coef)]
    K = len(degs) - 1
    m = sum(d + 1 for d in degs) + lp

    def c(j, n):                    # coefficient of u^n in f^(j)
        return mp.rf(n + 1, j) * a[n + j] if n >= 0 else 0
    if m - 1 + K >= n_coef:
        return None
    rows, rhs = [], []
    for n in range(m):
        row = []
        for j, d in enumerate(degs):
            row += [c(j, n - i) for i in range(d + 1) if not (j == K and i == 0)]
        row += [-1 if n == i else 0 for i in range(lp + 1)]
        rows.append(row)
        rhs.append(-c(K, n))
    x = mp.lu_solve(mp.matrix(rows), mp.matrix(rhs))
    off = sum(d + 1 for d in degs[:K])
    qK = [mp.mpc(1)] + [x[off + i] for i in range(degs[K])]
    return [1 / u for u in mp.polyroots(qK[::-1], maxsteps=400, extraprec=4 * mp.mp.prec) if u != 0]


def members(order=2):
    """(degs, deg P) of the ensemble: deg P such that the conditions use a_0..a_22 and a_0..a_20"""
    out = []
    for d in ENSEMBLE:
        degs = list(d[:order]) + [d[2]] if order == 2 else [d[1], d[2]]
        top = N_COEF - order - sum(x + 1 for x in degs)
        out += [(degs, lp) for lp in (top - 2, top) if lp >= 0]
    return out


def farthest(ts):
    ok = [complex(t) for t in ts if WINDOW[0] < abs(t) < WINDOW[1] and mp.im(t) > 0]
    return max(ok, key=abs) if ok else None


def estimate(sig, order=2):
    """median over the ensemble of the farthest admissible zero, the members' zeros, and the spread (MAD)"""
    mp.mp.dps = DPS_DA
    sig = [mp.mpc(z) for z in sig[:N_COEF]]
    got = []
    for degs, lp in members(order):
        try:
            ts = da_zeros(sig, degs, lp)
        except ZeroDivisionError:
            ts = None
        f = farthest(ts) if ts else None
        got.append({"degs": degs, "lp": lp, "t": f})
    fs = np.array([g["t"] for g in got if g["t"] is not None])
    if len(fs) == 0:
        return None, got, None
    med = complex(np.median(fs.real), np.median(fs.imag))
    return med, got, float(np.median(np.abs(fs - med)))


def noise_spread(sig, wf, order=2, eps=None):
    """largest shift of the estimate over NSEED seeds of the measured per-order noise"""
    pg.NOISE_MODE = "measured"
    eps = eps or pg.noise_eps(wf)
    e0 = estimate(sig, order)[0]
    d = 0.0
    for seed in range(NSEED):
        en = estimate(pg.noisy([mp.mpc(z) for z in sig[:N_COEF]], 547000 + seed, eps), order)[0]
        d = max(d, abs(en - e0) if (en is not None and e0 is not None) else float("inf"))
    return d


def ratio12_21(sig):
    return complex(pp.ratio_law([mp.mpc(z) for z in sig], 12, 21)[0])


# ---------------------------------------------------------------- selftest
def chord_function(P0, P1, ex, dens, n=N_COEF):
    """sigma_k of B(t) = int_0^1 (1-s)^ex dens(s) / (t - g(s)) g' ds on the chord g = P0 + (P1 - P0) s, moments by
    mpmath.quad (independent of the plants' tanh-sinh rule)"""
    D = P1 - P0
    a = [mp.quad(lambda s: (1 - s) ** ex * dens(s) * (P0 + D * s) ** k * D, [0, 0.5, 1]) for k in range(n)]
    return [a[k] / mp.factorial(k) / a[0] for k in range(n)]


def selftest(mut=None):
    """(1) one DA2 member recovers the end of a chord density exp(c s)(1-s)^ex exactly: its Stieltjes function satisfies
    a first-order linear equation with polynomial coefficients (a semi-classical weight), so the code's linear system
    must reproduce it to rounding; (2) the ensemble on a density outside that class, (1 + sin(4s)/2 + 0.3i cos(3s));
    (3) the moved-end plants' moments against mpmath.quad. Each with a mutation that must fire."""
    ok = True
    mp.mp.dps = 50
    P0, P1, ex = mp.mpc("0.749", "0.155"), mp.mpc("1.40", "1.20"), mp.mpc("2.5", "-1.2")
    sig = chord_function(P0, P1, ex, lambda s: mp.exp(mp.mpc(-2, 1) * s))
    if mut == "selftest_sigma18":
        sig[18] *= mp.mpf("1.01")

    def member(s):
        mp.mp.dps = DPS_DA
        return farthest(da_zeros([mp.mpc(z) for z in s], [4, 4, 4], 4))
    e1 = abs(member(sig) - complex(P1))
    print(f"selftest 1: DA2 [4,4,4] on the semi-classical chord density: error {e1:.1e} (pass < 1e-10)")
    ok &= e1 < 1e-10
    sm = list(sig)
    sm[18] *= mp.mpf("1.01")
    em = abs(member(sm) - complex(P1))
    print(f"selftest 1, mutation sigma_18 x 1.01: error {em:.1e} (must exceed 1e-10)")
    ok &= em > 1e-10
    mp.mp.dps = 50
    sig = chord_function(P0, P1, ex, lambda s: 1 + mp.sin(4 * s) / 2 + mp.mpc(0, "0.3") * mp.cos(3 * s))
    if mut == "selftest_ensemble":
        sig = [z * mp.mpf("1.03") ** k for k, z in enumerate(sig)]
    est, got, spr = estimate(sig)
    err = abs(est - complex(P1))
    print(f"selftest 2: DA2 ensemble on a non-semi-classical chord density: {est:.6f} against {complex(P1):.6f}, "
          f"error {err:.1e} (pass < 5e-3), spread {spr:.1e}")
    ok &= err < 5e-3
    sm = [z * mp.mpf("1.03") ** k for k, z in enumerate(sig)]
    em = estimate(sm)[0]
    print(f"selftest 2, mutation sigma_k x 1.03^k (end moved to 1.03 P1): error {abs(em - complex(P1)):.1e} "
          f"(must exceed 5e-3)")
    ok &= abs(em - complex(P1)) > 5e-3
    # 3: the moved-end plant's tanh-sinh moments against mpmath.quad of the same density
    mp.mp.dps = DPS_DA
    real = pg.load_scan(pg.REAL["J1250"])[1.0]
    pl = FarPlant(1.0, "chord2", real, 1.85)
    mp.mp.dps = pg.DPS_PLANT
    D = pl.P1 - pl.P0

    def quad_mom(m, k):
        return mp.quad(lambda s: (1 - s) ** pl.ex * mp.chebyt(m, 2 * s - 1) * (pl.P0 + D * s) ** k * D,
                       [0, 0.5, 1]) / mp.factorial(k)
    mom = [list(r) for r in pl.mom]
    if mut == "selftest_moment":
        mom[2][11] *= 1 + mp.mpf("1e-25")
    e2 = max(abs(mom[m][k] - quad_mom(m, k)) / abs(quad_mom(m, k)) for m in (0, 2, pl.M - 1) for k in (0, 11, 22))
    print(f"selftest 3: plant R = 1.85 moments, tanh-sinh against mpmath.quad, max rel diff {float(e2):.1e} "
          f"(pass < 1e-30); |P1| = {float(abs(pl.P1)):.12f}")
    ok &= e2 < mp.mpf("1e-30") and abs(abs(pl.P1) - mp.mpf("1.85")) < mp.mpf("1e-40")
    mom[2][11] *= 1 + mp.mpf("1e-25")
    e3 = abs(mom[2][11] - quad_mom(2, 11)) / abs(quad_mom(2, 11))
    print(f"selftest 3, mutation moment(2, 11) x (1 + 1e-25): rel diff {float(e3):.1e} (must exceed 1e-30)")
    ok &= e3 > mp.mpf("1e-30")
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- stages
def jsonable(o):
    return pg.jsonable(o)


def run_plants(outdir):
    mp.mp.dps = DPS_DA
    real = pg.load_scan(pg.REAL["J1250"])
    rec = {"frozen": FROZEN, "ensemble": ENSEMBLE, "window": WINDOW, "plants": {}}
    path = os.path.join(outdir, "plants.json")
    for name, (kind, wf, R, dex, stretch, st) in PLANTS.items():
        t0 = time.time()
        pl = FarPlant(wf, kind, real[wf], R, dex, stretch)
        sig = pl.sigma()
        P1 = complex(pl.P1)
        est, got, spr = estimate(sig)
        est1, _, spr1 = estimate(sig, order=1)
        nz = noise_spread(sig, wf)
        rl = ratio12_21(sig)
        r = {"kind": kind, "wf": wf, "R": float(abs(P1)), "dex": dex, "stretch": stretch, "set": st, "P1": P1,
             "M": pl.M, "fit_resid_max": max(pl.resid), "estimate": est, "error": abs(est - P1) if est else None,
             "spread": spr, "noise": nz, "members": got, "da1_estimate": est1, "da1_error": abs(est1 - P1) if est1
             else None, "ratio12_21": rl, "ratio_error": abs(rl - P1), "seconds": time.time() - t0}
        rec["plants"][name] = r
        print(f"{name:26s} |P1| {abs(P1):.4f}  est {est:.4f} |est| {abs(est):.4f} err {r['error']:.4f}  spread "
              f"{spr:.4f}  noise {nz:.1e}  | DA1 err {r['da1_error']:.4f}  ratio12-21 err {r['ratio_error']:.4f}  "
              f"(M {pl.M}, fit {max(pl.resid):.1e}, {r['seconds']:.0f}s)")
        sys.stdout.flush()
        json.dump(jsonable(rec), open(path, "w"), indent=1)
    P = rec["plants"]
    c1 = {n: P[n]["error"] <= 0.02 for n in P}
    c2 = {f"{a} -> {b}": abs(P[b]["estimate"] - P[a]["estimate"]) for a, b in PAIRS}
    c3 = {n: P[n]["noise"] <= 0.01 for n in P}
    b1 = {n: P[n]["ratio_error"] > 0.05 for n in P if PLANTS[n][2] is None and PLANTS[n][3] == 0}
    v = {"C1": all(c1.values()), "C1_fail": [n for n in c1 if not c1[n]],
         "C1_max_error": max(P[n]["error"] for n in P), "C1_max_error_D": max(P[n]["error"] for n in P if
                                                                               P[n]["set"] == "D"),
         "C2": all(x > 0.03 for x in c2.values()), "C2_moves": c2,
         "C3": all(c3.values()), "C3_max_noise": max(P[n]["noise"] for n in P),
         "B1": all(b1.values()), "B1_errors": {n: P[n]["ratio_error"] for n in b1}}
    rec["verdict"] = v
    json.dump(jsonable(rec), open(path, "w"), indent=1)
    print(f"C1 recovery <= 0.02: {v['C1']} (max error {v['C1_max_error']:.4f}; fails: {v['C1_fail']})")
    print("C2 response > 0.03:", v["C2"], {k: f"{x:.4f}" for k, x in c2.items()})
    print(f"C3 noise <= 0.01: {v['C3']} (max {v['C3_max_noise']:.1e})")
    print("B1 ratio-law baseline misses kappa_* plants by > 0.05:", v["B1"], {k: f"{x:.3f}" for k, x in
                                                                              v["B1_errors"].items()})
    return v["C1"] and v["C2"] and v["C3"] and v["B1"]


def mutate(sig, spec):
    if not spec:
        return sig
    if spec.startswith("scale"):
        r = mp.mpf(spec[5:])
        return [z * r ** k for k, z in enumerate(sig)]
    return pg.mutate(sig, spec)


def run_real(outdir, mut):
    rec = {"frozen": FROZEN, "mutate": mut, "real": {}}
    for lab, d in pg.REAL.items():
        mp.mp.dps = DPS_DA
        for wf, sig in pg.load_scan(d).items():
            sig = mutate(sig[:N_COEF], mut)
            est, got, spr = estimate(sig)
            est1, _, spr1 = estimate(sig, order=1)
            nz = noise_spread(sig, wf) if lab == "J1250" else None
            r = {"estimate": est, "abs": abs(est), "spread": spr, "noise": nz, "members": got,
                 "da1_estimate": est1, "da1_spread": spr1, "ratio12_21": ratio12_21(sig)}
            rec["real"].setdefault(lab, {})[wf] = r
            print(f"real {lab} wf {wf}: kappa_far {est:.4f} |kappa_far| {abs(est):.4f}  spread {spr:.4f}  noise "
                  f"{nz if nz is None else f'{nz:.1e}'}  | DA1 {est1:.4f} (spread {spr1:.4f})  ratio12-21 "
                  f"{r['ratio12_21']:.4f}")
            for g in got:
                t = "none" if g["t"] is None else f"{g['t']:.4f} |t| {abs(g['t']):.4f}"
                print(f"     {g['degs']} lp {g['lp']}: {t}")
            sys.stdout.flush()
    a, b = rec["real"]["J1250"][1.0]["estimate"], rec["real"]["J1500"][1.0]["estimate"]
    rec["J_spread"] = abs(a - b)
    rec["node_spread"] = max(abs(rec["real"]["J1250"][w]["estimate"] - a) for w in pg.NODES)
    print(f"J spread (J1250 vs J1500, wf 1.00): {rec['J_spread']:.1e}; largest node difference from wf 1.00: "
          f"{rec['node_spread']:.4f}")
    name = "real" + (f"_mutate_{mut}" if mut else "") + ".json"
    json.dump(jsonable(rec), open(os.path.join(outdir, name), "w"), indent=1)
    ok = a is not None
    if mut and mut.startswith("scale"):
        base = json.load(open(os.path.join(outdir, "real.json")))["real"]["J1250"]["1.0"]["estimate"]
        base = complex(*base) * float(mut[5:])
        dev = abs(a - base)
        moved = abs(a - base / float(mut[5:]))
        print(f"mutation {mut}: |estimate - {mut[5:]} x unmutated| = {dev:.1e} (pass < 1e-6), moved by {moved:.4f} "
              f"(must exceed 0.03)")
        ok = dev < 1e-6 and moved > 0.03
    elif mut:
        base = complex(*json.load(open(os.path.join(outdir, "real.json")))["real"]["J1250"]["1.0"]["estimate"])
        print(f"mutation {mut}: the estimate moves by {abs(a - base):.4f} (reported, not judged)")
    return ok


def main():
    argv = sys.argv[1:]
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    argv = [a for a in argv if not a.startswith("--mutate=")]
    if not argv or argv[0] == "selftest":
        sys.exit(0 if selftest(mut) else 1)
    stage, outdir = argv[0], argv[1]
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    os.makedirs(outdir, exist_ok=True)
    ok = run_plants(outdir) if stage == "plants" else run_real(outdir, mut)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
