#!/usr/bin/env python3
"""The near end and the zero factor of the real S_1 twins (issue #689).

#684 (pade_twin_end.py; notes/PADE_FAR_END.md 11) found that neither the far end nor the end exponent alone gives the
real sigma_0..sigma_22 a twin of fewer than 23 terms. Every twin class so far had a density regular at the near end P0
(D_entire 7.1's first [6/6] pole) and the plants' zero factor 1 - x/x_ref. Here the chord twins are refitted with the
near end changed, and separately with the zero factor moved or removed.

Twin. pref(x) L(x), L(x) = int_0^1 s^alpha (1-s)^ex sum_{m<M} c_m T_m(2s-1) exp(x gamma(s)) gamma'(s) ds on the chord
gamma = P0 + (P1 - P0) s, ex = beta - 1, and pref = 1 - x/x_zero or 1. x_ref = planted_plants.x0_planted(wf, "1.0000"),
the plants' zero. The moments are exact, by int_0^1 s^(alpha+p) (1-s)^ex ds = B(alpha+p+1, ex+1) (chord_moments_a);
at alpha = 0 they are pade_twin_count.chord_moments, so kstar is #672's chord basis bit for bit.

Variants, per node, with P1 = kappa_*, alpha = 0, P0 and x_ref unless stated (JUDGED):
  kstar          the baseline (#672's chord twin)
  a-0.5 .. a+1   alpha = -0.5, -0.25, +0.25, +0.5, +1
  near541        P0 -> #541's extrapolated near end (0.732+0.056i at wf 1.00, pade_geometry.extrapolate of its
                 nearest geometry pole over k = 7..11)
  nozero         pref = 1, the zero factor removed
  zpade          x_ref -> the real series' own zero nearest x_ref, from its [11/11] Pade approximant in x
  zfit           x_ref located by fit: the least-squares l2 at M = 20 = n - 3 minimised over the zero (locate())
Context (CONTEXT): near541 with each alpha; nozero with each alpha and with ex = beta - 2; x_ref moved by +-0.01,
+-0.01i, +-0.1, +-0.1i; the far end at #541's ratio-law kappa (#684's best far end) alone and with each alpha, near541,
nozero and zpade; zfitratio (the zero located with that far end); p0fit (P0 located by l2 at M = 20); afit (alpha
located by l2 at M = 20).

Fits, sizes and C1 are #672's (pade_twin_size.scan): weights 1/err_k with the measured errors (floor 1e-30), chi =
max_k |dsigma_k|/err_k, M_min the smallest M with chi <= 1, M_lo the smallest with l2 <= sqrt(n), n = 23, fits at 150
digits checked at 250, C1 amended at M = n. The scans' logged peak density is that of (1-s)^ex sum c_m T_m, without
s^alpha. Plants (NearPlant) are pade_farend.FarPlant's: the density above, times pref, c_m fitted to the node's real
sigma_0..sigma_12 (M = 6, or 8 if a relative residual exceeds 1e-2), moments by tanh-sinh at 60 digits (tmax 6 when
alpha < 0, so the rule's end truncation stays below 1e-100).

    python3 pade_twin_near.py --selftest [--mutate=alphamom|p0mom]
    python3 pade_twin_near.py control OUTDIR [--mutate=alpha|zero|locate] [--allow_unraw]
    python3 pade_twin_near.py real OUTDIR [--mutate=c1dup] [--allow_unraw]     (reads OUTDIR/control.json)

Gate as amended by #702 (AMENDED; notes/PADE_FAR_END.md 13.8): the variants that change no near end run in the real
stage when K0, every C0 and Z1 pass; the near-end variants need K2 as well. #689's control.json is read unchanged.

Exit 1 when a frozen control fails or a frozen rule forbids the real stage. Numerical evidence, not a bound.
"""
import json
import math
import os
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pade_twin_end as te                                             # noqa: E402

tz, tc, F, pg, pp = te.tz, te.tc, te.F, te.pg, te.pp
N23 = te.N23
DPS_FIT, DPS_CHECK = te.DPS_FIT, te.DPS_CHECK
CHI_C0 = 1e-30
M_FIT = 20                          # a twin of at most this many terms counts as short
M_K2 = 19                           # K2(b): the alpha = 0.5 plant needs at least this many terms with alpha = 0
M_LOC = 20                          # the locators minimise l2 at n - 3 terms (#684's diagnostic, notes 11.5)
ALPHAS = ("-0.5", "-0.25", "+0.25", "+0.5", "+1")
ZMOVES = ("+0.01", "-0.01", "+0.01i", "-0.01i", "+0.1", "-0.1", "+0.1i", "-0.1i")
Z1_DELTA = ("0.003", "0.002")       # Z1's plant: its zero at x_ref + 0.003 + 0.002i
Z1_TOL = 1e-4
SCALE_FLOOR = mp.mpf("1e-20")       # the locators' last offset scale
PADE_ZERO = (11, 11)
# I541-mac/real.json (raw-data repository): near_inf of the J1250 series per node; its real.log prints them to 3
# decimals as 0.773+0.146i, 0.732+0.056i, 0.728+0.035i
NEAR541 = {0.99: 0.7733624141264912 + 0.14599529705211858j, 1.0: 0.7320698315186889 + 0.05591959983765087j,
           1.01: 0.7279591102672306 + 0.03486781605302231j}
JUDGED = {"kstar": {}, **{f"a{a}": {"alpha": a} for a in ALPHAS}, "near541": {"P0": "near541"},
          "nozero": {"zero": None}, "zpade": {"zero": "pade"}, "zfit": {"locate": "zero"}}
CONTEXT = {**{f"near541a{a}": {"P0": "near541", "alpha": a} for a in ALPHAS},
           **{f"nozeroa{a}": {"zero": None, "alpha": a} for a in ALPHAS},
           "nozeroex-1": {"zero": None, "dex": "-1"},
           **{f"z{d}": {"zero": ("move", d)} for d in ZMOVES},
           "ratio": {"P1": "ratio"}, **{f"ratioa{a}": {"P1": "ratio", "alpha": a} for a in ALPHAS},
           "rationear541": {"P1": "ratio", "P0": "near541"}, "rationozero": {"P1": "ratio", "zero": None},
           "ratiozpade": {"P1": "ratio", "zero": "pade"}, "zfitratio": {"P1": "ratio", "locate": "zero"},
           "p0fit": {"locate": "P0"}, "afit": {"locate": "alpha"}}
REF672_CHORD = {wf: te.REF672[wf]["chord"] for wf in pg.NODES}
FROZEN = {                          # written before the control and real stages were first run (notes/PADE_FAR_END.md 13.2)
    "variants_judged": "chord twins, P1 = kappa_*, ex = beta - 1, per node: kstar (alpha 0, P0, the plants' zero "
                       "factor at x_ref; the baseline); near-end exponent alpha = -0.5, -0.25, +0.25, +0.5, +1 (density "
                       "s^alpha (1-s)^ex sum c_m T_m(2s-1)); near541 (P0 -> #541's extrapolated near end); nozero (zero "
                       "factor removed); zpade (the zero at the real series' [11/11] Pade zero nearest x_ref); zfit (the "
                       "zero located by the least-squares l2 at M = 20 = n - 3)",
    "variants_context": "near541 x each alpha; nozero x each alpha; nozero with ex = beta - 2; x_ref + (+-0.01, +-0.01i, "
                        "+-0.1, +-0.1i); the far end at the ratio-law kappa alone and with each alpha, near541, nozero, "
                        "zpade; zfitratio; p0fit (P0 located by l2 at M = 20); afit (alpha located by l2 at M = 20); "
                        "same rules, reported, not judged",
    "errors_sizes_C1": "#672's: measured rel_k (floor 1e-30), F = 1; M_min (least squares, chi <= 1), M_lo (l2 <= "
                       "sqrt(n)), n = 23, M = 6..23; 150 digits checked at 250; C1 amended at M = n, on every scan; if "
                       "C1 fires on real data the work stops. Context: errors widened to the strings' rounding (#683), "
                       "J1500 at wf 1.00",
    "locator": "the zero (or P0) minimising log l2 at M = 20 = n - 3 (never chi at n - 1, #684): Nelder-Mead in offsets "
               "x = centre + scale (p + iq), scale 0.01, then rounds re-centred at the best point with scale / 100 "
               "until a round gains < 1e-3 in log l2 or scale < 1e-20; from each start (zero: x_ref and the Pade "
               "zero; P0: P0 and #541's near end), the lower l2 kept. afit: alpha on -0.9..2.0 step 0.1 by l2 at "
               "M = 20, then golden section to 1e-6",
    "K0_ends": "the near end of #541 reproduced: |near_inf - NEAR541| <= 1e-9 at every node; the far ends as #684's K0 "
               "(the ratio law within 1e-4 of #541, the poles past kappa_* within 1e-9)",
    "K2_control": "the issue's same-class control, at every node: the chord2 plant with alpha = 0.5 (P0, kappa_*, x_ref) "
                  "(a) refitted in its own basis at its own M: chi <= 1e-30; (b) in the kstar basis (alpha = 0): "
                  "M_min >= 19",
    "C0_variants": "every non-located variant basis, judged and context, at every node: the chord2 plant of the same "
                   "spec, refitted in the variant's basis at its own M: chi <= 1e-30. A located variant: the same at "
                   "its located value, in the real stage; a failure there is reported and that variant is not judged",
    "Z1_locator": "a chord2 plant with its zero at x_ref + 0.003 + 0.002i, at every node: (a) own basis chi <= 1e-30; "
                  "(b) the zero locator, started as for zfit, returns the plant's zero within 1e-4. Z1 does not gate "
                  "the real stage. If it fails, zfit and zfitratio are reported as uncalibrated, not judged",
    "gate": "the real stage runs only if K0, K2 and every C0 pass",
    "C2_baseline": "kstar reproduces #672's chord twin: M_min = 23 and chi at 22 terms within 1e-3 relative of REF672 at "
                   "every node; a failure is a discrepancy, reported",
    "mutation_control": "--mutate=alpha adds 0.01 to every basis' alpha: K2(a), Z1(a) and every C0 must fail. "
                        "--mutate=zero moves the zero of every basis that has one by +0.01: K2(a), Z1(a) and every C0 "
                        "with a zero factor must fail, every C0 without one must pass. --mutate=locate makes the "
                        "locator return its first start (x_ref) unsearched: Z1(b) must fail at every node (C0 is "
                        "skipped). The scans and Z1(b) are skipped under alpha and zero. Each run exits 1",
    "mutation_real": "--mutate=c1dup (#672): column n - 1 of the 250-digit bases replaced by column n - 2, on the "
                     "non-located variants; the amended C1 must fire on every scan with M_min = n and pass on every "
                     "scan with M_min < n; exit 1",
    "fits": "a variant fits if M_min <= 20 at all three nodes (chord class)",
    "verdict": "names the fitting judged variants by ingredient: near-end exponent (a*), near-end position (near541), "
               "zero factor (nozero, zpade, zfit if Z1 passed); if none fits: none of the tested near ends or zero "
               "factors gives a short twin",
    "context": "chi at M = 20..22 and l2 at M = 20 per scan; the sensitivity (each judged variant's chord2 plant in the "
               "kstar basis; the kstar plant in the nozero and a+0.5 bases; K2's plant in the a+0.25 and a+1 bases; "
               "Z1's plant in the kstar basis); the Pade zero's error on the K2 and Z1 plants and its spread over "
               "[10/12], [12/10], [10/10], [11/10], [10/11] on the real series; widened errors (a changed judged "
               "verdict is a discrepancy, reported); J1500 at wf 1.00 for the judged variants",
}
# #702 amends FROZEN["gate"] (notes/PADE_FAR_END.md 13.6, 13.8), written before any real run. No threshold changes.
AMENDED = {
    "gate_zero": "the variants that change no near end (kstar, nozero, zpade, zfit; context z*, nozeroex-1, ratio, "
                 "rationozero, ratiozpade, zfitratio) run if K0, every C0 and Z1 pass",
    "gate_near": "the near-end variants (a*, near541, their combinations near541a*, nozeroa*, ratioa*, rationear541, "
                 "and p0fit, afit) run only if K2 passes as well",
    "verdict": "names the fitting judged variants of the groups that ran; if none fits, none of the tested zero factors "
               "gives a short twin, and the near ends are reported as untested when K2 failed",
    "unchanged": "C1, C2, --mutate=c1dup (on the non-located variants that run), the fits rule (M_min <= 20 at all "
                 "three nodes), zfit's location by l2 at M = 20 and its C0 at the located zero (chi <= 1e-30)",
}


def near_end_variant(v):
    """a variant that changes the near end: its exponent alpha, its position P0, or locates one of them"""
    m = JUDGED.get(v, CONTEXT.get(v))
    return "alpha" in m or "P0" in m or m.get("locate") in ("P0", "alpha")


def gates(verdict):
    """the amended gate (AMENDED): (zero-factor group runs, near-end group runs) from control.json's verdict"""
    base = verdict["K0"] and verdict["C0"]
    return base and verdict["Z1_a"] and verdict["Z1_b"], base and verdict["K2"]


# ---------------------------------------------------------------- variants
def xref(wf):
    with mp.workdps(pg.DPS_PLANT):
        return mp.mpc(pp.x0_planted(wf, "1.0000"))


def spec(v, wf, ctx):
    """the variant's twin: P0 (None = the plants' P0), P1, ex, alpha (a string), zero (None = no factor) and what it
    locates. ctx: {"ends": {wf: far ends}, "near": {wf: near end}, "zpade": {wf: zero}, "located": {(v, wf): value}}"""
    m = JUDGED.get(v, CONTEXT.get(v))
    s = {"P0": None, "P1": te.kstar(wf), "ex": te.ex0(wf), "alpha": "0", "zero": xref(wf), "locate": m.get("locate")}
    with mp.workdps(pg.DPS_PLANT):
        if m.get("P1") == "ratio":
            s["P1"] = mp.mpc(ctx["ends"][wf]["ratio"])
        if m.get("P0") == "near541":
            s["P0"] = mp.mpc(ctx["near"][wf])
        if "alpha" in m:
            s["alpha"] = m["alpha"]
        if "dex" in m:
            s["ex"] = s["ex"] + mp.mpf(m["dex"])
        if "zero" in m:
            z = m["zero"]
            if z is None:
                s["zero"] = None
            elif z == "pade":
                s["zero"] = mp.mpc(ctx["zpade"][wf])
            else:
                s["zero"] = s["zero"] + te.shift(z[1])
        loc = ctx.get("located", {}).get((v, wf))
        if loc is not None:
            s[s["locate"]] = loc
    return s


def mutated(sp, mut):
    """the spec with a control mutation applied (alpha: alpha + 0.01; zero: the zero + 0.01 if it has one)"""
    s = dict(sp)
    if mut == "alpha":
        s["alpha"] = mp.nstr(mp.mpf(s["alpha"]) + mp.mpf("0.01"), 30)
    if mut == "zero" and s["zero"] is not None:
        with mp.workdps(pg.DPS_PLANT):
            s["zero"] = mp.mpc(s["zero"]) + mp.mpf("0.01")
    return s


def describe(sp):
    P0 = "plants'" if sp["P0"] is None else f"{complex(sp['P0']):.4f}"
    z = "none" if sp["zero"] is None else f"{complex(sp['zero']):.6f}"
    return f"P0 {P0}, P1 {complex(sp['P1']):.4f}, ex {complex(sp['ex']):.3f}, alpha {sp['alpha']}, zero {z}"


# ---------------------------------------------------------------- exact moments and bases
def chord_moments_a(P0, P1, ex, alpha, M, K, mut=None):
    """mom[m][k] = (P1 - P0)/k! int_0^1 s^alpha (1-s)^ex T_m(2s-1) (P0 + (P1 - P0) s)^k ds, k <= K, by
    int_0^1 s^(alpha+p) (1-s)^ex ds = B(alpha+p+1, ex+1), at the working precision. alpha = 0 is
    pade_twin_count.chord_moments itself. mut: alphamom (alpha + 1e-10), p0mom (P0 (1 + 1e-10))"""
    a = mp.mpf(alpha) + (mp.mpf("1e-10") if mut == "alphamom" else 0)
    P0 = mp.mpc(P0) * (1 + mp.mpf("1e-10") if mut == "p0mom" else 1)
    if a == 0:
        return tc.chord_moments(P0, P1, ex, M, K)
    P1, ex = mp.mpc(P1), mp.mpc(ex)
    D = P1 - P0
    T = tc.shifted_cheb(M)
    beta = [mp.beta(a + 1, ex + 1)]
    for p in range(M + K):
        beta.append(beta[-1] * (p + a + 1) / (p + a + ex + 2))
    q = [[mp.fsum(c * beta[i + j] for i, c in enumerate(T[m])) for j in range(K + 1)] for m in range(M)]
    pw0 = [P0 ** j for j in range(K + 1)]
    pwd = [D ** j for j in range(K + 1)]
    return [[D / mp.factorial(k) * mp.fsum(mp.binomial(k, j) * pw0[k - j] * pwd[j] * q[m][j] for j in range(k + 1))
             for k in range(K + 1)] for m in range(M)]


def geometry(wf, sp):
    """P0, P1, ex, alpha and pref of a spec, at the plants' 60 digits (as pade_twin_end.end_basis)"""
    with mp.workdps(pg.DPS_PLANT):
        arc = F.FarArc(wf, "chord", mp.mpc(sp["P1"]))
        P0 = arc.P0 if sp["P0"] is None else mp.mpc(sp["P0"])
        pref = [mp.mpc(1)] if sp["zero"] is None else pp.factor([mp.mpc(sp["zero"])])
        return P0, arc.P1, mp.mpc(sp["ex"]), mp.mpf(sp["alpha"]), pref


def near_basis(wf, sp, dps, M=N23, K=N23, mut=None):
    """the chord twin's columns (pref * mom_m)_k, m < M, k < K, at dps digits"""
    P0, P1, ex, alpha, pref = geometry(wf, sp)
    with mp.workdps(dps):
        mom = chord_moments_a(P0, P1, ex, alpha, M, K - 1, mut)
        return tz.Cols(mom, pref, ex, dps, F.PIECES["chord2"])


class ZeroBasis:
    """the moments of a spec kept at DPS_FIT, with the zero factor applied per call (the zero locator)"""

    def __init__(self, wf, sp, M):
        P0, P1, ex, alpha, _ = geometry(wf, sp)
        self.ex = ex
        with mp.workdps(DPS_FIT):
            self.mom = chord_moments_a(P0, P1, ex, alpha, M, N23 - 1)

    def cols(self, zero):
        with mp.workdps(pg.DPS_PLANT):
            pref = pp.factor([mp.mpc(zero)])
        return tz.Cols(self.mom, pref, self.ex, DPS_FIT, F.PIECES["chord2"])


class NearPlant(pg.Plant):
    """pade_farend.FarPlant on the chord with the near end P0, far end P1, exponent ex, near-end exponent alpha and zero
    factor of a spec: density s^alpha (1-s)^ex sum c_m T_m(2s-1), c_m fitted to the real sigma_0..sigma_12 (M = 6, or 8
    if the largest relative residual exceeds 1e-2), moments by tanh-sinh at the plants' 60 digits"""

    def __init__(self, wf, sp, real_sig):
        mp.mp.dps = pg.DPS_PLANT
        self.wf, self.kind = wf, "chord2"
        self.arc = F.FarArc(wf, "chord", mp.mpc(sp["P1"]))
        if sp["P0"] is not None:
            self.arc.P0 = mp.mpc(sp["P0"])
        self.P0, self.P1 = self.arc.P0, self.arc.P1
        self.ex, self.alpha = mp.mpc(sp["ex"]), mp.mpf(sp["alpha"])
        self.pieces = F.PIECES["chord2"]
        rule = pg.ts_rule(tmax=mp.mpf(6)) if self.alpha < 0 else pg.ts_rule()
        self.nodes = [(x, omx, wt) for x, omx, wt in rule]          # one piece, s = x
        self.pref = [mp.mpc(1)] if sp["zero"] is None else pp.factor([mp.mpc(sp["zero"])])
        for M in (6, 8):
            self.mom = self.moments(M)
            res = self.fit(real_sig, M)
            if max(res) <= 1e-2:
                break
        self.M, self.resid = M, res

    def moments(self, M):
        mom = [[mp.mpc(0)] * (pg.KM + 1) for _ in range(M)]
        for s, oms, w in self.nodes:
            g, dg = self.arc.g(s), self.arc.dg(s)
            base = w * mp.power(oms, self.ex) * dg
            if self.alpha != 0:
                base *= mp.power(s, self.alpha)
            tm = [mp.chebyt(m, 2 * s - 1) for m in range(M)]
            gk = mp.mpc(1)
            for k in range(pg.KM + 1):
                for m in range(M):
                    mom[m][k] += base * tm[m] * gk
                gk *= g
        return [[mom[m][k] / mp.factorial(k) for k in range(pg.KM + 1)] for m in range(M)]


def pade_zero(sig, near, L=PADE_ZERO[0], M=PADE_ZERO[1]):
    """the zero nearest `near` of the [L/M] Pade approximant of sum sigma_k x^k, at pade_geometry.DPS_PADE digits"""
    with mp.workdps(pg.DPS_PADE):
        p, _ = mp.pade([mp.mpc(z) for z in sig[:L + M + 1]], L, M)
        while len(p) > 1 and p[-1] == 0:
            p = p[:-1]
        roots = mp.polyroots(p[::-1], maxsteps=500, extraprec=2 * mp.mp.prec)
        return complex(min(roots, key=lambda r: abs(r - near)))


def near_end(wf, sig):
    """#541's near_inf: the nearest geometry pole of the [k/k] approximants, k = 7..11, extrapolated in 1/k^2"""
    with mp.workdps(pg.DPS_PADE):
        fr = pg.Frame(wf)
        res = pg.analyse(sig[:N23], fr, noise=False)
        return pg.extrapolate(res, "near")


# ---------------------------------------------------------------- selftest
def selftest(mut=None):
    """the exact columns of every kind of variant basis against the plants' tanh-sinh columns, m, k < 23"""
    ok = True
    wf = 1.0
    real = tc.real_sig()[wf]
    ctx = {"near": NEAR541, "zpade": {wf: complex(xref(wf)) + 0.01j}}   # a stand-in zero; the real one is control's
    for v in ("a-0.5", "a-0.25", "a+0.25", "a+0.5", "a+1", "near541a+0.5", "nozero", "zpade", "nozeroex-1"):
        sp = spec(v, wf, ctx)
        pl = NearPlant(wf, sp, real)
        mp.mp.dps = pg.DPS_PLANT
        qm = pl.moments(N23)
        b = near_basis(wf, sp, DPS_FIT, mut=mut)
        with mp.workdps(DPS_FIT):
            qc = [pp.polymul([mp.mpc(z) for z in pl.pref], qm[m], N23) for m in range(N23)]
            e = max(abs(b.cols[m][k] - qc[m][k]) / abs(qc[m][k]) for m in range(N23) for k in range(N23))
        print(f"selftest {v:13s}: {describe(sp)}; exact columns against the plant's tanh-sinh columns, m, k < {N23}: "
              f"max rel diff {float(e):.1e} (pass < 1e-40); plant M {pl.M}, fit to sigma_0..12 {max(pl.resid):.1e}")
        ok &= e < mp.mpf("1e-40")
    # alpha = 0: the same moments as pade_twin_count.chord_moments, bit for bit
    sp = spec("kstar", wf, ctx)
    b0 = near_basis(wf, sp, DPS_FIT)
    b1 = te.end_basis(wf, "chord", te.kstar(wf), te.ex0(wf), DPS_FIT)
    same = all(b0.cols[m][k] == b1.cols[m][k] for m in range(N23) for k in range(N23))
    print(f"selftest kstar: the columns equal pade_twin_end.end_basis's bit for bit: {same}")
    ok &= same
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- locators
def nelder_mead(f, x0, step, iters, ftol=1e-12):
    """minimise f from x0 (Nelder and Mead 1965, Comput. J. 7, 308): reflection, expansion, contraction, shrink; stops
    when the simplex's values agree within ftol"""
    import numpy as np
    n = len(x0)
    S = np.array([x0] + [np.array(x0, dtype=float) + step * np.eye(n)[i] for i in range(n)], dtype=float)
    v = np.array([f(p) for p in S])
    for _ in range(iters):
        o = np.argsort(v)
        S, v = S[o], v[o]
        if v[-1] - v[0] < ftol:
            break
        c = S[:-1].mean(axis=0)
        xr = c + (c - S[-1])
        fr = f(xr)
        if fr < v[0]:
            xe = c + 2 * (c - S[-1])
            fe = f(xe)
            S[-1], v[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < v[-2]:
            S[-1], v[-1] = xr, fr
        else:
            xc = c + 0.5 * (S[-1] - c)
            fc = f(xc)
            if fc < v[-1]:
                S[-1], v[-1] = xc, fc
            else:
                S[1:] = S[0] + 0.5 * (S[1:] - S[0])
                v[1:] = [f(p) for p in S[1:]]
    o = np.argsort(v)
    return S[o[0]], float(v[o[0]])


def locate_complex(f, start, scale=mp.mpf("0.01"), iters=60, log=None, label=""):
    """minimise f(z) (z mpc, f a float) by Nelder-Mead in offsets z = centre + scale (p + iq): rounds re-centred at the
    best point, scale / 100 per round, until a round gains < 1e-3 or scale < SCALE_FLOOR"""
    with mp.workdps(pg.DPS_PLANT):
        centre = mp.mpc(start)
        best = f(centre)
        rounds = []
        sc = mp.mpf(scale)
        while sc > SCALE_FLOOR:
            c0 = centre
            p, v = nelder_mead(lambda q: f(c0 + sc * mp.mpc(q[0], q[1])), [0.0, 0.0], 1.0, iters)
            gain = best - v
            if v < best:
                centre, best = c0 + sc * mp.mpc(p[0], p[1]), v
            rounds.append({"scale": float(sc), "value": v, "gain": gain})
            if log:
                log(f"    {label} round scale {float(sc):.0e}: log l2 {v:.4f} (gain {gain:.2e}), at "
                    f"{mp.nstr(centre, 20)}")
            if gain < 1e-3:
                break
            sc /= 100
        return centre, best, rounds


def l2_fit(cols, sig, rel, M=M_LOC):
    return max(tc.twin_fit(cols, sig, rel, N23, M, 1.0)[2], 1e-300)


def locate(wf, sp, sig, rel, mut=None, log=None):
    """the located value of sp["locate"] (zero, P0 or alpha) by l2 at M = 20, with the start values tried"""
    what = sp["locate"]
    out = {"what": what, "starts": []}
    if what == "zero":
        zb = ZeroBasis(wf, sp, M_LOC)

        def f(z):
            return math.log(l2_fit(zb.cols(z), sig, rel))
        starts = [("x_ref", xref(wf)), ("pade", mp.mpc(pade_zero(sig, complex(xref(wf)))))]
    elif what == "P0":
        def f(z):
            s = dict(sp, P0=z)
            return math.log(l2_fit(near_basis(wf, s, DPS_FIT, M=M_LOC), sig, rel))
        with mp.workdps(pg.DPS_PLANT):
            starts = [("P0", F.FarArc(wf, "chord", mp.mpc(sp["P1"])).P0), ("near541", mp.mpc(NEAR541[wf]))]
    else:
        return locate_alpha(wf, sp, sig, rel, log)
    if mut == "locate":                 # the control's mutation: the first start, unsearched
        out.update({"value": starts[0][1], "log_l2": f(starts[0][1]), "start": starts[0][0], "mutated": True})
        return out
    for lab, z0 in starts:
        z, v, rounds = locate_complex(f, z0, log=log, label=f"{what} from {lab}")
        out["starts"].append({"start": lab, "start_value": complex(z0), "value": z, "value_str": mp.nstr(z, 40),
                              "log_l2": v, "rounds": rounds})
    b = min(out["starts"], key=lambda r: r["log_l2"])
    out.update({"value": b["value"], "value_str": b["value_str"], "log_l2": b["log_l2"], "start": b["start"]})
    return out


def locate_alpha(wf, sp, sig, rel, log=None):
    def f(a):
        s = dict(sp, alpha=repr(float(a)))
        return math.log(l2_fit(near_basis(wf, s, DPS_FIT, M=M_LOC), sig, rel))
    grid = [round(-0.9 + 0.1 * i, 10) for i in range(30)]
    vals = [f(a) for a in grid]
    i = min(range(len(grid)), key=lambda j: vals[j])
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = f(x1), f(x2)
    while hi - lo > 1e-6:
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = f(x2)
    a, v = (x1, f1) if f1 < f2 else (x2, f2)
    if vals[i] < v:
        a, v = grid[i], vals[i]
    if log:
        log(f"    alpha on the grid -0.9..2.0: best {grid[i]} (log l2 {vals[i]:.4f}); golden section: alpha {a:.7f}, "
            f"log l2 {v:.4f}")
    return {"what": "alpha", "grid": list(zip(grid, vals)), "value": repr(float(a)), "value_str": repr(float(a)),
            "log_l2": v, "start": "grid"}


# ---------------------------------------------------------------- control stage
def c0(wf, sp, real, rel, mut=None):
    """the spec's plant refitted in the (mutated) basis at its own M"""
    pl = NearPlant(wf, sp, real)
    r = tz.row(near_basis(wf, mutated(sp, mut), DPS_FIT, M=pl.M), pl.sigma()[:N23], rel, N23, pl.M)
    return {"chi": r["chi"], "M_plant": pl.M, "fit_resid": max(pl.resid), "pass": r["chi"] <= CHI_C0}, pl


def plant_scan(wf, sp, sig, rel, log, label):
    s = tz.scan(near_basis(wf, sp, DPS_FIT), near_basis(wf, sp, DPS_CHECK), sig, rel, N23, log, label)
    return summary(s)


def summary(s):
    return {"M_min": s["M_min"], "M_lo": s["M_lo"], "C1_pass": s["C1_pass"], "precision": s["precision"],
            "chi": {r["M"]: r["chi"] for r in s["rows"]}, "l2": {r["M"]: r["l2"] for r in s["rows"]}}


def run_control(outdir, mut=None):
    log = tz.Log(outdir, "control", mut)
    real = tc.real_sig()
    rec = {"frozen": FROZEN, "mutate": mut, "K2": {}, "Z1": {}, "C0": {}, "sensitivity": {}, "zpade_spread": {}}
    ends = {wf: te.far_ends(wf) for wf in pg.NODES}
    k0e, rec["K0_far"] = te.check_ends(ends, log)
    near = {wf: near_end(wf, real[wf]) for wf in pg.NODES}
    k0n = True
    for wf in pg.NODES:
        d = abs(near[wf] - NEAR541[wf])
        k0n &= d <= 1e-9
        log(f"K0 wf {wf}: near_inf {near[wf]:.6f}, off #541 by {d:.1e} (pass <= 1e-9)")
    k0 = k0e and k0n
    zp = {wf: pade_zero(real[wf], complex(xref(wf))) for wf in pg.NODES}
    for wf in pg.NODES:
        alt = {f"[{a}/{b}]": pade_zero(real[wf], complex(xref(wf)), a, b)
               for a, b in ((10, 12), (12, 10), (10, 10), (11, 10), (10, 11))}
        rec["zpade_spread"][wf] = {"zero": zp[wf], "others": alt, "spread": max(abs(z - zp[wf]) for z in alt.values()),
                                   "x_ref": complex(xref(wf))}
        log(f"zpade wf {wf}: [11/11] zero {zp[wf]:.10f} (x_ref {complex(xref(wf)):.6f}, off by "
            f"{abs(zp[wf] - complex(xref(wf))):.2e}); spread over the other orders {rec['zpade_spread'][wf]['spread']:.1e}")
    ctx = {"ends": ends, "near": near, "zpade": zp}
    rec.update({"ends": ends, "near": near, "zpade": zp, "K0": k0})
    k2 = z1a = z1b = True
    c0_ok, c0_fail = True, []
    for wf in pg.NODES:
        rel, _ = tc.err_model(wf)
        # K2: the alpha = 0.5 plant
        sp = spec("a+0.5", wf, ctx)
        r, pl = c0(wf, sp, real[wf], rel, mut)
        sig = pl.sigma()[:N23]
        out = {"a": r, "pade_zero_error": abs(pade_zero(sig, complex(xref(wf))) - complex(xref(wf)))}
        log(f"K2(a) wf {wf}: alpha 0.5 plant (M {pl.M}, fit {r['fit_resid']:.1e}) in its own basis"
            f"{' (mutated)' if mut else ''}: chi {r['chi']:.1e} (pass <= 1e-30) {'OK' if r['pass'] else 'BAD'}; its "
            f"Pade zero off x_ref by {out['pade_zero_error']:.1e}")
        if not mut:
            s = plant_scan(wf, spec("kstar", wf, ctx), sig, rel, log, f"K2(b) wf {wf}: alpha 0.5 plant, kstar basis")
            out.update({"kstar": s, "b_pass": s["M_min"] is not None and s["M_min"] >= M_K2})
            log(f"K2(b) wf {wf}: M_min {s['M_min']} (pass >= {M_K2}) {'OK' if out['b_pass'] else 'BAD'}")
            for v in ("a+0.25", "a+1"):
                out[v] = plant_scan(wf, spec(v, wf, ctx), sig, rel, log, f"context wf {wf}: alpha 0.5 plant, {v} basis")
        else:
            out["b_pass"] = True
        rec["K2"][wf] = out
        k2 &= r["pass"] and out["b_pass"]
        # Z1: the zero locator on a plant with a moved zero
        with mp.workdps(pg.DPS_PLANT):
            zt = xref(wf) + mp.mpc(*Z1_DELTA)
        sp = dict(spec("kstar", wf, ctx), zero=zt)
        r, pl = c0(wf, sp, real[wf], rel, mut)
        sig = pl.sigma()[:N23]
        out = {"a": r, "zero_true": complex(zt), "pade_zero_error": abs(pade_zero(sig, complex(xref(wf))) - complex(zt))}
        log(f"Z1(a) wf {wf}: plant with its zero at {complex(zt):.4f} (M {pl.M}) in its own basis"
            f"{' (mutated)' if mut else ''}: chi {r['chi']:.1e} {'OK' if r['pass'] else 'BAD'}; its Pade zero off by "
            f"{out['pade_zero_error']:.1e}")
        z1a &= r["pass"]
        if mut in (None, "locate"):
            loc = locate(wf, dict(spec("zfit", wf, ctx)), sig, rel, mut, log)
            dz = abs(complex(loc["value"]) - complex(zt))
            out.update({"locator": loc, "dev": dz, "b_pass": dz <= Z1_TOL})
            log(f"Z1(b) wf {wf}: located zero {mp.nstr(loc['value'], 15)} (from {loc['start']}), off the plant's by "
                f"{dz:.1e} (pass <= {Z1_TOL}), log l2 at 20 terms {loc['log_l2']:.3f} {'OK' if out['b_pass'] else 'BAD'}")
            z1b &= out["b_pass"]
            if not mut:
                out["kstar"] = plant_scan(wf, spec("kstar", wf, ctx), sig, rel, log,
                                          f"context wf {wf}: Z1's plant in the kstar basis")
        rec["Z1"][wf] = out
        if mut == "locate":
            continue
        # C0 of every non-located variant basis
        for v in list(JUDGED) + list(CONTEXT):
            sp = spec(v, wf, ctx)
            if sp["locate"]:
                continue
            r, pl = c0(wf, sp, real[wf], rel, mut)
            r["has_zero"] = sp["zero"] is not None
            rec["C0"].setdefault(wf, {})[v] = r
            c0_ok &= r["pass"]
            if not r["pass"]:
                c0_fail.append((wf, v))
            log(f"C0 wf {wf} {v:13s}: {describe(sp)}: chi {r['chi']:.1e} (M {r['M_plant']}, fit {r['fit_resid']:.1e})")
            if not mut and v in JUDGED and v != "kstar":
                rec["sensitivity"].setdefault(wf, {})[v] = plant_scan(
                    wf, spec("kstar", wf, ctx), pl.sigma()[:N23], rel, log,
                    f"context wf {wf}: {v}'s plant in the kstar basis")
        if not mut:
            _, pl = c0(wf, spec("kstar", wf, ctx), real[wf], rel)
            for v in ("nozero", "a+0.5"):
                rec["sensitivity"].setdefault(wf, {})[f"kstar plant in {v}"] = plant_scan(
                    wf, spec(v, wf, ctx), pl.sigma()[:N23], rel, log, f"context wf {wf}: kstar's plant in the {v} basis")
    rec["verdict"] = {"K0": k0, "K2": k2, "Z1_a": z1a, "Z1_b": z1b, "C0": c0_ok, "C0_fail": c0_fail}
    if mut:
        k2a_fired = all(not rec["K2"][wf]["a"]["pass"] for wf in pg.NODES)
        z1a_fired = all(not rec["Z1"][wf]["a"]["pass"] for wf in pg.NODES)
        if mut == "alpha":
            fired = k2a_fired and z1a_fired and all(not x["pass"] for d in rec["C0"].values() for x in d.values())
        elif mut == "zero":
            fired = k2a_fired and z1a_fired and all(x["pass"] != x["has_zero"] for d in rec["C0"].values()
                                                    for x in d.values())
        else:
            fired = all(not rec["Z1"][wf]["b_pass"] for wf in pg.NODES)
        rec["verdict"]["mutation_fired_as_frozen"] = fired
        log(f"mutation {mut}: K2(a) failed at every node {k2a_fired}; Z1(a) {z1a_fired}; C0 failures {len(c0_fail)} "
            f"of {sum(len(d) for d in rec['C0'].values())}; Z1(b) {z1b}; as frozen: {fired}")
    rec["gate_pass"] = k0 and k2 and c0_ok
    rec["controls_pass"] = rec["gate_pass"] and z1a and z1b
    rec["gate_amended"] = dict(zip(("zero", "near"), gates(rec["verdict"])))
    log(f"K0 {k0}, K2 {k2}, C0 {c0_ok} ({len(c0_fail)} failures), Z1 {z1a and z1b}; the real stage may run: "
        f"{rec['gate_pass']} as first frozen; as amended (#702), zero-factor group {rec['gate_amended']['zero']}, "
        f"near-end group {rec['gate_amended']['near']}")
    log.dump(rec)
    return rec["controls_pass"]


# ---------------------------------------------------------------- real stage
def run_real(outdir, mut=None):
    ctl = json.load(open(os.path.join(outdir, "control.json")))
    g_zero, g_near = gates(ctl["verdict"])
    if not (g_zero or g_near):
        print("the controls failed: no real run (frozen rule, as amended by #702)")
        return False
    z1 = ctl["verdict"]["Z1_a"] and ctl["verdict"]["Z1_b"]

    def runs(v):
        return g_near if near_end_variant(v) else g_zero
    unrun = [v for v in list(JUDGED) + list(CONTEXT) if not runs(v)]
    log = tz.Log(outdir, "real", mut)
    log(f"amended gate (#702): K0 {ctl['verdict']['K0']}, C0 {ctl['verdict']['C0']}, Z1 {z1}, K2 "
        f"{ctl['verdict']['K2']}; zero-factor group runs {g_zero}, near-end group runs {g_near}; not run: "
        + (", ".join(unrun) or "none"))
    rec = {"frozen": FROZEN, "amended": AMENDED, "mutate": mut, "scans": {}, "widened": {}, "J1500": {},
           "located": {}, "C0_located": {}, "Z1_pass": z1, "gate_zero": g_zero, "gate_near": g_near, "unrun": unrun}
    real = tc.real_sig()
    j15 = pg.load_scan(pg.REAL["J1500"])

    def cplx(v):
        return complex(*v) if isinstance(v, list) and len(v) == 2 and not isinstance(v[0], list) else v
    ends = {float(wf): {k: cplx(v) for k, v in e.items()} for wf, e in ctl["ends"].items()}
    ctx = {"ends": ends, "near": {float(w): cplx(v) for w, v in ctl["near"].items()},
           "zpade": {float(w): cplx(v) for w, v in ctl["zpade"].items()}, "located": {}}
    ok = True
    errs = {wf: tc.err_model(wf)[0] for wf in pg.NODES}
    for group in (JUDGED, CONTEXT):
        for wf in pg.NODES:
            rel, sig = errs[wf], real[wf][:N23]
            relw = te.widened(wf, rel)
            log.dump(rec)
            for v in group:
                if not runs(v):
                    continue
                sp = spec(v, wf, ctx)
                if sp["locate"]:
                    if mut:
                        continue
                    t0 = time.time()
                    loc = locate(wf, sp, sig, rel, None, log)
                    ctx["located"][(v, wf)] = loc["value"]
                    sp = spec(v, wf, ctx)
                    r, _ = c0(wf, sp, real[wf], rel)
                    rec["located"].setdefault(v, {})[wf] = {k: loc[k] for k in ("what", "value_str", "log_l2", "start")}
                    rec["located"][v][wf]["starts"] = loc.get("starts", loc.get("grid"))
                    rec["C0_located"].setdefault(v, {})[wf] = r
                    log(f"located wf {wf} {v}: {loc['what']} = {loc['value_str']} (from {loc['start']}), l2 at 20 "
                        f"terms {math.exp(loc['log_l2']):.3e}; C0 there chi {r['chi']:.1e} "
                        f"{'OK' if r['pass'] else 'BAD'} ({time.time() - t0:.0f}s)")
                fit = near_basis(wf, sp, DPS_FIT)
                chk = near_basis(wf, sp, DPS_CHECK)
                if mut == "c1dup":
                    chk.cols[N23 - 1] = list(chk.cols[N23 - 2])
                s = tz.scan(fit, chk, sig, rel, N23, log, f"wf {wf} {v} ({describe(sp)})")
                rec["scans"].setdefault(wf, {})[v] = summary(s) | {"spec": describe(sp)}
                ok &= s["C1_pass"]
                if not s["C1_pass"] and not mut:
                    log("C1 fired on real data: stop (frozen rule)")
                    log.dump(rec)
                    return False
                if mut or group is not JUDGED:
                    continue
                sw = tz.scan(fit, chk, sig, relw, N23, log, f"wf {wf} {v}, widened errors (context)")
                rec["widened"].setdefault(wf, {})[v] = summary(sw)
                if wf == 1.0:
                    sj = tz.scan(fit, chk, j15[wf][:N23], rel, N23, log, f"wf {wf} {v}, J1500 (context)")
                    rec["J1500"][v] = summary(sj)
    scans = rec["scans"]
    if mut:
        at_n = [(wf, v) for wf in pg.NODES for v, s in scans[wf].items() if s["M_min"] == N23]
        below = [(wf, v) for wf in pg.NODES for v, s in scans[wf].items() if s["M_min"] != N23]
        fired = bool(at_n) and all(not scans[wf][v]["C1_pass"] for wf, v in at_n)
        iso = all(scans[wf][v]["C1_pass"] for wf, v in below)
        rec["mutation"] = {"n_at_n": len(at_n), "n_below": len(below), "fired_at_n": fired, "below_n_pass": iso}
        log(f"mutation {mut}: {len(at_n)} scans with M_min = n, the amended C1 fired on each: {fired}; {len(below)} "
            f"scans with M_min < n, all still pass: {iso}; C1 pass overall: {ok}")
        log.dump(rec)
        return ok

    def table(src, vs):
        out = {}
        for v in vs:
            mm = [src[wf][v]["M_min"] for wf in pg.NODES]
            out[v] = {"M_min": mm, "M_lo": [src[wf][v]["M_lo"] for wf in pg.NODES],
                      "chi22": [src[wf][v]["chi"][22] for wf in pg.NODES],
                      "l2_20": [src[wf][v]["l2"][20] for wf in pg.NODES],
                      "fits": all(m is not None and m <= M_FIT for m in mm)}
        return out
    judged = table(scans, [v for v in JUDGED if runs(v)])
    context = table(scans, [v for v in CONTEXT if runs(v)])
    wide = table(rec["widened"], [v for v in JUDGED if v in rec["widened"][1.0]])
    c0loc = {v: all(r["pass"] for r in d.values()) for v, d in rec["C0_located"].items()}
    for t in (judged, context):
        for v in ("zfit", "zfitratio"):
            if v in t:
                t[v]["calibrated"] = z1 and c0loc.get(v, False)
    zfit_ok = "zfit" in judged and z1 and c0loc.get("zfit", False)
    groups = {"near-end exponent": [v for v in judged if v.startswith("a") and judged[v]["fits"]],
              "near-end position": [v for v in ("near541",) if v in judged and judged[v]["fits"]],
              "zero factor": [v for v in ("nozero", "zpade", "zfit") if v in judged and judged[v]["fits"]
                              and (v != "zfit" or zfit_ok)]}
    if "zfit" not in judged or not judged["zfit"]["fits"] or zfit_ok:
        zfit_note = None
    else:
        zfit_note = "zfit fits, but its locator is uncalibrated (Z1 or its C0 failed): not judged"
    named = [g for g, vs in groups.items() if vs]
    tested = " or ".join(x for x, g in (("near ends", g_near), ("zero factors", g_zero)) if g)
    verdict = ("the fitting variants change: " + "; ".join(f"{g} ({', '.join(groups[g])})" for g in named) if named
               else f"none of the tested {tested} gives a short twin")
    if not g_near:
        verdict += "; the near ends are untested (K2 failed, so the near-end variants did not run)"
    c2 = {wf: {"M_min": scans[wf]["kstar"]["M_min"], "chi22": scans[wf]["kstar"]["chi"][22], "ref": REF672_CHORD[wf]}
          for wf in pg.NODES}
    c2_pass = all(x["M_min"] == N23 and abs(x["chi22"] / x["ref"] - 1) <= 1e-3 for x in c2.values())
    changed = [v for v in wide if wide[v]["fits"] != judged[v]["fits"]]
    rec.update({"judged": judged, "context": context, "widened_table": wide, "groups": groups, "verdict": verdict,
                "zfit_note": zfit_note, "C2_baseline": {"rows": c2, "pass": c2_pass}, "widened_changes_verdict": changed,
                "C0_located_pass": c0loc, "C1_pass": ok})
    for v in list(judged) + list(context):
        t = judged.get(v) or context[v]
        log(f"{'judged ' if v in JUDGED else 'context'} {v:13s}: M_min {t['M_min']} M_lo {t['M_lo']}; chi at 22 terms "
            + ", ".join(f"{x:.2e}" for x in t["chi22"]) + "; l2 at 20 terms " + ", ".join(f"{x:.2e}" for x in t["l2_20"])
            + f" -> fits: {t['fits']}")
    log(f"C2 baseline (kstar against #672): {'OK' if c2_pass else 'DISCREPANCY'}; widened errors change the verdict of "
        f"{changed or 'no variant'}; located variants' C0: {c0loc}; Z1 passed: {z1}")
    if zfit_note:
        log(zfit_note)
    log(f"verdict: {verdict}")
    log.dump(rec)
    return ok


def main():
    argv = sys.argv[1:]
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    if "--selftest" in argv:
        if mut not in (None, "alphamom", "p0mom"):
            raise SystemExit(f"unknown selftest mutation {mut}")
        sys.exit(0 if selftest(mut) else 1)
    argv = [a for a in argv if not a.startswith("--")]
    if len(argv) != 2 or argv[0] not in ("control", "real"):
        raise SystemExit(__doc__)
    stage, outdir = argv
    if mut not in {"control": (None, "alpha", "zero", "locate"), "real": (None, "c1dup")}[stage]:
        raise SystemExit(f"unknown {stage} mutation {mut}")
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    os.makedirs(outdir, exist_ok=True)
    sys.exit(0 if (run_control(outdir, mut) if stage == "control" else run_real(outdir, mut)) else 1)


if __name__ == "__main__":
    main()
