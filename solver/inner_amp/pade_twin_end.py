#!/usr/bin/env python3
"""Which ingredient keeps the real S_1 series from a short twin: the far end of its Borel singular support, or the end
exponent (issue #684).

#672 (pade_twin_size.py twins_real; notes/PADE_FAR_END.md 9.2) found that no chord, stepped-arc or smooth-arc density
with the plants' end points P0 and kappa_*, their end exponent ex = beta - 1 and their zero factor reproduces the real
sigma_0..sigma_22 with fewer than 23 terms, at any node. Here the same twins are refitted with one ingredient changed.

Variants, per node wf, from that node's own J1250 series. P1 is the far end and ex the exponent of (1 - s)^ex.
  kstar          P1 = kappa_*, ex = beta - 1 (#672's twins, the baseline)
  ratio          P1 = the ratio law over k = 12..21 (1.4574+1.1119i at wf 1.00, #541)
  pole8..pole10  P1 = the real [k/k] pole past kappa_*, k = 8, 9, 10 (|t| 1.715, 1.762, 1.806 at wf 1.00, #541)
  ex-0.3..ex+0.3 P1 = kappa_*, ex = beta - 1 + delta, delta = -0.3, -0.1, +0.1, +0.3
Context, not judged: radial far ends R1.75..R2.00 on kappa_*'s ray (pade_farend.far_point); each moved far end with each
exponent shift; ratiopair, (P1, ex) = (kappa, beta - 1) of the same ratio fit; imaginary shifts ex+-0.1i, ex+-0.3i.
Classes, as #669: chord P0 -> P1 (pade_twin_count.chord_moments, exact Beta moments); arc, the stepped arc (w = 1, 2, 1,
2) and smooth, the smooth arc (w = 1), on the cubic of pade_farend.FarArc through D_entire 7.1's poles and P1
(pade_twin_size.arc_moments, exact). Each column is the planted zero factor times the moments.

Fits, sizes and C1 are #672's (pade_twin_size.scan): weighted least squares, weights 1/err_k, err_k = rel_k |sigma_k|
with rel_k the measured errors (floor 1e-30), chi = max_k |dsigma_k| / err_k, M_min the smallest M with chi <= 1,
M_lo the smallest with l2 <= sqrt(n), n = 23, fits at 150 digits checked at 250, C1 as amended at M = n.

    python3 pade_twin_end.py --selftest [--mutate=endmom|upoly]
    python3 pade_twin_end.py control OUTDIR [--mutate=farend|exshift] [--allow_unraw]
    python3 pade_twin_end.py real OUTDIR [--mutate=c1dup] [--allow_unraw]      (reads OUTDIR/control.json)

The selftest checks the moved-end, shifted-exponent moments against the plants' tanh-sinh quadrature. `control` runs
the far ends' reproduction of #541 (K0), the same-class control on #547's far-end-1.85 plant (K1) and a same-class check
of every variant basis (C0). Exit 1 when a frozen control fails or a frozen rule forbids the real stage. Numerical
evidence, not a bound.
"""
import json
import os
import sys
import time

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pade_twin_size as tz                                            # noqa: E402  (sets pg.KM = 31)
import sigma_load as sl                                                # noqa: E402

tc, F, pg, pp = tz.tc, tz.F, tz.pg, tz.pp
N23 = tz.N23
DPS_FIT, DPS_CHECK = tz.DPS_FIT, tz.DPS_CHECK
CHI_C0 = 1e-30
M_FIT = 20                          # a twin of at most this many terms counts as short
M_K1 = 19                           # K1(b): the 1.85 plant needs at least this many terms in the kappa_* basis
DELTAS = ("-0.3", "-0.1", "+0.1", "+0.3")
ENDS = ("ratio", "pole8", "pole9", "pole10")
CLASSES = (("chord", "chord2"), ("arc", "arcjump"), ("smooth", "arc"))
JUDGED = ("kstar",) + ENDS + tuple(f"ex{d}" for d in DELTAS)
CONTEXT = (tuple(f"R{r}" for r in ("1.75", "1.80", "1.85", "1.90", "2.00"))
           + tuple(f"{e}ex{d}" for e in ENDS for d in DELTAS) + ("ratiopair",)
           + tuple(f"ex{d}i" for d in ("-0.3", "-0.1", "+0.1", "+0.3")))
RESOLUTION = ("1.80", "1.84", "1.86", "1.90")
# I541-mac/real.json (raw-data repository): the J1250 [k/k] pole past kappa_* per node, k = 8, 9, 10; and the ratio law
# over k = 12..21 from its real.log, to 4 decimals
REF541 = {0.99: {8: 1.355474512907339 + 1.0506977251883904j, 9: 1.4050935307815564 + 1.0628863629673637j,
                 10: 1.458551637205871 + 1.0719351389146154j},
          1.0: {8: 1.3549378318196221 + 1.0507596283369365j, 9: 1.4066172412863676 + 1.061286490489844j,
                10: 1.4566034199399178 + 1.0685122989873004j},
          1.01: {8: 1.3548989680915362 + 1.0509058447285036j, 9: 1.407585411585267 + 1.0599294998036761j,
                 10: 1.456259830569657 + 1.0676136922555628j}}
REF541_RATIO = {0.99: 1.4598 + 1.1113j, 1.0: 1.4574 + 1.1119j, 1.01: 1.4551 + 1.1126j}
# I672-mac/twins_real.json: chi of the real twins at 22 terms (M_min = 23 everywhere)
REF672 = {0.99: {"chord": 96978.58, "arc": 124166.66, "smooth": 126775.40},
          1.0: {"chord": 91162.72, "arc": 115299.66, "smooth": 117634.68},
          1.01: {"chord": 86025.38, "arc": 106438.08, "smooth": 108482.81}}
FROZEN = {                          # written before the control and real stages were first run (notes/PADE_FAR_END.md 11.2)
    "variants_judged": "far end: ratio (ratio law over k = 12..21), pole8, pole9, pole10 (the real [k/k] pole past "
                       "kappa_*), each with ex = beta - 1; exponent: ex = beta - 1 + delta, delta = -0.3, -0.1, +0.1, "
                       "+0.3, with P1 = kappa_*; per node, from the node's own J1250 series; kstar is the baseline",
    "variants_context": "R1.75..R2.00 (radial far ends), each far end with each delta, ratiopair (kappa and beta - 1 of "
                        "the same 12..21 ratio fit), imaginary shifts +-0.1i, +-0.3i; same rules, reported, not judged",
    "classes": "chord P0 -> P1; arc, the stepped arc of FarArc (w = 1, 2, 1, 2); smooth, the same arc with w = 1",
    "errors": "measured rel_k, k <= 22, floor 1e-30, F = 1 (#672's twins_real); context: rel_k widened to max(rel_k, "
              "the strings' rounding_k) (#683)",
    "sizes_C1": "#672's: M_min (least squares, chi <= 1) and M_lo (l2 <= sqrt(n)), n = 23, M = 6..23; C1 amended at "
                "M = n, on every scan; if C1 fires on real data the work stops",
    "K0_ends": "the far ends reproduce #541: the ratio law within 1e-4 of REF541_RATIO at every node; exactly one real "
               "[k/k] pole with |kappa_*| + 0.01 < |t| < 2.2 and Im t > 0 for k = 8, 9, 10 at every node, within "
               "1e-9 of REF541",
    "K1_control_547": "#547's plant chord2 with its far end at R = 1.85 (pade_farend.FarPlant) at every node: (a) in "
                      "the chord basis with its own far end, at its own M: chi <= 1e-30; (b) in the kappa_* chord "
                      "basis: M_min >= 19",
    "C0_variants": "every variant basis (judged and context), class and node: the plant of the same class (chord2, "
                   "arcjump, arc) with that far end and exponent, c_m fitted to the node's real sigma_0..sigma_12 "
                   "as #547, refitted at its own M: chi <= 1e-30",
    "gate": "the real stage runs only if K0, K1 and every C0 pass",
    "C2_baseline": "kstar reproduces #672: M_min = 23 and chi at 22 terms within 1e-3 relative of REF672 at every node "
                   "and class; a failure is a discrepancy, reported",
    "mutation_control": "--mutate=farend moves every moved-end basis' P1 radially by +0.01 (K1's own-end basis to "
                        "1.86): K1(a) and every C0 of a moved far end must fail, the C0 of kstar and of the exponent "
                        "shifts must pass; --mutate=exshift adds 0.01 to every basis' ex: every C0 and K1(a) must "
                        "fail. Each run exits 1",
    "mutation_real": "--mutate=c1dup (#672): column n - 1 of the 250-digit bases replaced by column n - 2; the amended "
                     "C1 must fire on every scan with M_min = n and pass on every scan with M_min < n; exit 1",
    "fits": "a (variant, class) fits if M_min <= 20 at all three nodes; a variant fits if some class does",
    "verdict": "the far end is named if some far-end variant fits and no exponent variant does; the exponent if the "
               "reverse; both if both; if none fits: none of the tested moves gives a short twin",
    "context": "chi at M = 20..22 per scan; the sensitivity (each judged variant's chord2 plant in the kappa_*, beta - 1 "
               "chord basis); the far end's resolution (K1's plant in chord bases at |P1| = 1.80, 1.84, 1.86, 1.90); "
               "K1's plant in the kappa_* arc bases; the widened errors (a changed judged verdict is reported as a "
               "discrepancy); J1500 at wf 1.00 for the judged variants, chord and arc",
}


# ---------------------------------------------------------------- variants
def kstar(wf):
    return F.kstar(wf)


def ex0(wf):
    with mp.workdps(pg.DPS_PLANT):
        return pp.mpc(pp.BETA[wf]) - 1


def shift(s):
    """'+0.3' -> 0.3, '-0.1i' -> -0.1i, at the plants' precision"""
    with mp.workdps(pg.DPS_PLANT):
        return mp.mpc(0, mp.mpf(s[:-1])) if s.endswith("i") else mp.mpc(mp.mpf(s))


def far_ends(wf):
    """the node's ratio law over k = 12..21 (kappa, beta) and its [k/k] poles past kappa_*, k = 8, 9, 10"""
    sig = tc.real_sig()[wf][:N23]
    lim = float(abs(kstar(wf))) + 0.01
    out = {}
    with mp.workdps(pg.DPS_PADE):
        s = [mp.mpc(z) for z in sig]
        kap, beta = pp.ratio_law(s, 12, 21)
        out["ratio"], out["ratio_beta"] = complex(kap), complex(beta)
        for k in (8, 9, 10):
            ts = [complex(p["t"]) for p in pg.pade(s, k) if lim < abs(p["t"]) < 2.2 and mp.im(p["t"]) > 0]
            out[f"pole{k}_all"] = ts
            out[f"pole{k}"] = ts[0] if len(ts) == 1 else None
    return out


def spec(v, wf, ends):
    """(P1, ex, moves_end) of variant v at node wf"""
    e0 = ex0(wf)
    if v == "kstar":
        return kstar(wf), e0, False
    if v == "ratiopair":
        return mp.mpc(ends["ratio"]), mp.mpc(ends["ratio_beta"]) - 1, True
    if v.startswith("R"):
        return F.far_point(wf, float(v[1:])), e0, True
    end, _, d = v.partition("ex")
    P1 = kstar(wf) if not end else mp.mpc(ends[end])
    with mp.workdps(pg.DPS_PLANT):
        ex = e0 + shift(d) if d else e0
    return P1, ex, bool(end)


def mutate_spec(P1, ex, moves, mut):
    if mut == "farend" and moves:
        with mp.workdps(pg.DPS_PLANT):
            P1 = mp.mpc(P1) * (abs(mp.mpc(P1)) + mp.mpf("0.01")) / abs(mp.mpc(P1))
    if mut == "exshift":
        with mp.workdps(pg.DPS_PLANT):
            ex = mp.mpc(ex) + mp.mpf("0.01")
    return P1, ex


# ---------------------------------------------------------------- bases and plants
def end_basis(wf, cls, P1, ex, dps, M=N23, K=N23, mut=None):
    """twin columns of class cls with far end P1 and exponent ex, at dps digits (the geometry at the plants' 60)"""
    with mp.workdps(pg.DPS_PLANT):
        pref = pp.factor([pp.x0_planted(wf, "1.0000")])
        arc = F.FarArc(wf, "chord" if cls == "chord" else "arc", mp.mpc(P1))
        ex = mp.mpc(ex)
    pieces = F.PIECES[{"chord": "chord2", "arc": "arcjump", "smooth": "arc"}[cls]]
    with mp.workdps(dps):
        if cls == "chord":
            P1b = arc.P1 * (1 + mp.mpf("1e-10")) if mut == "endmom" else arc.P1
            mom = tc.chord_moments(arc.P0, P1b, ex, M, K - 1)
        else:
            mom = tz.arc_moments(arc, ex, pieces, M, K, mut)
        return tz.Cols(mom, pref, ex, dps, pieces)


class EndPlant(pg.Plant):
    """pade_farend.FarPlant with the far end P1 and the exponent ex given: density (1-s)^ex sum c_m T_m(2s-1) on the
    chord or FarArc from P0 to P1, times the planted zero factor, c_m fitted to the real sigma_0..sigma_12 (M = 6, or 8
    if the largest relative residual exceeds 1e-2), moments by tanh-sinh at the plants' 60 digits"""

    def __init__(self, wf, kind, real_sig, P1, ex):
        mp.mp.dps = pg.DPS_PLANT
        self.wf, self.kind = wf, kind
        self.arc = F.FarArc(wf, "chord" if kind == "chord2" else "arc", mp.mpc(P1))
        self.P0, self.P1 = self.arc.P0, self.arc.P1
        self.ex = mp.mpc(ex)
        self.pieces = F.PIECES[kind]
        rule = pg.ts_rule()
        self.nodes = []
        for a, b, w in self.pieces:
            a, b = mp.mpf(a), mp.mpf(b)
            for x, omx, wt in rule:
                self.nodes.append((a + (b - a) * x, (1 - b) + (b - a) * omx, (b - a) * wt * w))
        self.pref = pp.factor([pp.x0_planted(wf, "1.0000")])
        for M in (6, 8):
            self.mom = self.moments(M)
            res = self.fit(real_sig, M)
            if max(res) <= 1e-2:
                break
        self.M, self.resid = M, res


# ---------------------------------------------------------------- selftest
def selftest(mut=None):
    """the exact moments with a moved far end and a shifted exponent against the plants' tanh-sinh quadrature"""
    ok = True
    wf = 1.0
    real = tc.real_sig()[wf]
    cases = (("chord", "chord2", REF541[wf][10], "+0.3"), ("arc", "arcjump", REF541_RATIO[wf], "-0.3"),
             ("smooth", "arc", REF541[wf][8], "+0.1i"))
    for cls, kind, P1, d in cases:
        with mp.workdps(pg.DPS_PLANT):
            ex = ex0(wf) + shift(d)
        pl = EndPlant(wf, kind, real, P1, ex)
        mp.mp.dps = pg.DPS_PLANT
        qm = pl.moments(N23)
        b = end_basis(wf, cls, P1, ex, DPS_FIT, mut=mut)
        with mp.workdps(DPS_FIT):
            with mp.workdps(pg.DPS_PLANT):
                pref = pp.factor([pp.x0_planted(wf, "1.0000")])
            qc = [pp.polymul([mp.mpc(z) for z in pref], qm[m], N23) for m in range(N23)]
            e = max(abs(b.cols[m][k] - qc[m][k]) / abs(qc[m][k]) for m in range(N23) for k in range(N23))
        end_ok = abs(pl.arc.g(mp.mpf(1)) - mp.mpc(P1)) < mp.mpf("1e-50")
        print(f"selftest {cls}: P1 {P1:.4f}, ex beta - 1 {d}: exact columns against the {kind} plant's tanh-sinh "
              f"columns, m, k < {N23}: max rel diff {float(e):.1e} (pass < 1e-40); gamma(1) = P1: {end_ok}")
        ok &= e < mp.mpf("1e-40") and end_ok
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- control stage
def check_ends(ends, log):
    rec, ok = {}, True
    for wf, e in ends.items():
        dr = abs(e["ratio"] - REF541_RATIO[wf])
        good = dr <= 1e-4
        poles = {}
        for k in (8, 9, 10):
            t = e[f"pole{k}"]
            d = abs(t - REF541[wf][k]) if t is not None else float("inf")
            poles[k] = {"t": t, "n_candidates": len(e[f"pole{k}_all"]), "dev": d, "pass": d <= 1e-9}
            good &= poles[k]["pass"]
        rec[wf] = {"ratio": e["ratio"], "ratio_beta": e["ratio_beta"], "ratio_dev": dr, "poles": poles, "pass": good}
        ok &= good
        log(f"K0 wf {wf}: ratio law 12-21 {e['ratio']:.6f} (beta {e['ratio_beta']:.4f}), off #541 by {dr:.1e} "
            f"(pass <= 1e-4); poles past kappa_* " + "; ".join(
                f"k {k} {p['t']:.6f} |t| {abs(p['t']):.4f} ({p['n_candidates']} candidate), off #541 by {p['dev']:.1e}"
                if p["t"] is not None else f"k {k}: {p['n_candidates']} candidates" for k, p in poles.items())
            + f" -> {'OK' if good else 'BAD'}")
    return ok, rec


_BASES = {}


def cached_basis(wf, cls, P1, ex, dps):
    """end_basis, kept per (node, class, far end, exponent, precision) for the control's repeated plant scans"""
    key = (wf, cls, complex(P1), complex(ex), dps)
    if key not in _BASES:
        _BASES[key] = end_basis(wf, cls, P1, ex, dps)
    return _BASES[key]


def plant_scan(wf, P1, ex, sig, rel, log, label):
    fit, chk = cached_basis(wf, "chord", P1, ex, DPS_FIT), cached_basis(wf, "chord", P1, ex, DPS_CHECK)
    s = tz.scan(fit, chk, sig, rel, N23, log, label)
    return {k: s[k] for k in ("M_min", "M_lo", "C1_pass")} | {"chi_20_22": [r["chi"] for r in s["rows"]
                                                                          if 20 <= r["M"] <= 22]}


def run_control(outdir, mut=None):
    log = tz.Log(outdir, "control", mut)
    real = tc.real_sig()
    rec = {"frozen": FROZEN, "mutate": mut, "K1": {}, "C0": {}, "sensitivity": {}, "resolution": {}, "K1_arc": {}}
    ends = {wf: far_ends(wf) for wf in pg.NODES}
    k0, rec["K0"] = check_ends(ends, log)
    rec["ends"] = ends
    k1 = True
    c0_ok, c0_fail = True, []
    for wf in pg.NODES:
        rel, _ = tc.err_model(wf)
        # K1: #547's chord2 plant with the far end at 1.85
        pl = F.FarPlant(wf, "chord2", real[wf], 1.85)
        sig = pl.sigma()[:N23]
        own = F.far_point(wf, 1.86) if mut == "farend" else pl.P1
        exb = mutate_spec(own, pl.ex, True, mut)[1]
        r = tz.row(end_basis(wf, "chord", own, exb, DPS_FIT), sig, rel, N23, pl.M)
        a = r["chi"] <= CHI_C0
        log(f"K1(a) wf {wf}: #547's chord2 plant at R = 1.85 (M {pl.M}, fit to sigma_0..12 {max(pl.resid):.1e}) in its "
            f"own-end chord basis{' (mutated)' if mut else ''}: chi {r['chi']:.1e} (pass <= 1e-30) "
            f"{'OK' if a else 'BAD'}")
        out = {"M_plant": pl.M, "fit_resid": max(pl.resid), "chi_own": r["chi"], "a_pass": a}
        if not mut:
            s = plant_scan(wf, kstar(wf), ex0(wf), sig, rel, log, f"K1(b) wf {wf}: 1.85 plant in the kappa_* chord basis")
            out.update({"kstar": s, "b_pass": s["M_min"] is not None and s["M_min"] >= M_K1})
            log(f"K1(b) wf {wf}: M_min {s['M_min']} (pass >= {M_K1}) {'OK' if out['b_pass'] else 'BAD'}")
            for cls in ("arc", "smooth"):
                fit = end_basis(wf, cls, kstar(wf), ex0(wf), DPS_FIT)
                chk = end_basis(wf, cls, kstar(wf), ex0(wf), DPS_CHECK)
                sc = tz.scan(fit, chk, sig, rel, N23, log, f"context wf {wf}: 1.85 plant in the kappa_* {cls} basis")
                rec["K1_arc"].setdefault(wf, {})[cls] = {k: sc[k] for k in ("M_min", "M_lo", "C1_pass")}
            for R in RESOLUTION:
                rec["resolution"].setdefault(wf, {})[R] = plant_scan(
                    wf, F.far_point(wf, float(R)), ex0(wf), sig, rel, log,
                    f"context wf {wf}: 1.85 plant in the chord basis with |P1| = {R}")
        else:
            out["b_pass"] = True
        rec["K1"][wf] = out
        k1 &= out["a_pass"] and out["b_pass"]
        # C0 of every variant basis
        for v in JUDGED + CONTEXT:
            P1, ex, moves = spec(v, wf, ends)
            Pb, exb = mutate_spec(P1, ex, moves, mut)
            for cls, kind in CLASSES:
                plv = EndPlant(wf, kind, real[wf], P1, ex)
                sv = plv.sigma()[:N23]
                r = tz.row(end_basis(wf, cls, Pb, exb, DPS_FIT, M=plv.M), sv, rel, N23, plv.M)
                good = r["chi"] <= CHI_C0
                rec["C0"].setdefault(wf, {}).setdefault(v, {})[cls] = {"chi": r["chi"], "M_plant": plv.M,
                                                                      "fit_resid": max(plv.resid), "moves_end": moves,
                                                                      "pass": good}
                c0_ok &= good
                if not good:
                    c0_fail.append((wf, v, cls))
            log(f"C0 wf {wf} {v:12s}: P1 {complex(P1):.4f} |P1| {float(abs(P1)):.4f}, ex {complex(ex):.4f}; chi "
                + ", ".join(f"{cls} {rec['C0'][wf][v][cls]['chi']:.1e} (M {rec['C0'][wf][v][cls]['M_plant']})"
                            for cls, _ in CLASSES))
            # sensitivity: the judged variant's chord2 plant in the kappa_*, beta - 1 chord basis
            if not mut and v in JUDGED and v != "kstar":
                plv = EndPlant(wf, "chord2", real[wf], P1, ex)
                rec["sensitivity"].setdefault(wf, {})[v] = plant_scan(
                    wf, kstar(wf), ex0(wf), plv.sigma()[:N23], rel, log,
                    f"context wf {wf}: {v}'s chord2 plant in the kappa_* chord basis")
    rec["verdict"] = {"K0": k0, "K1": k1, "C0": c0_ok, "C0_fail": c0_fail}
    if mut:
        moved = [(wf, v, c) for wf in pg.NODES for v, d in rec["C0"][wf].items() for c, x in d.items() if x["moves_end"]]
        fixed = [(wf, v, c) for wf in pg.NODES for v, d in rec["C0"][wf].items() for c, x in d.items()
                 if not x["moves_end"]]
        failed = set(map(tuple, c0_fail))
        k1a_fired = all(not rec["K1"][wf]["a_pass"] for wf in pg.NODES)
        if mut == "farend":
            fired = k1a_fired and all(x in failed for x in moved) and not any(x in failed for x in fixed)
        else:
            fired = k1a_fired and len(failed) == len(moved) + len(fixed)
        rec["verdict"]["mutation_fired_as_frozen"] = fired
        log(f"mutation {mut}: K1(a) failed at every node: {k1a_fired}; C0 failed on {len(failed)} of "
            f"{len(moved) + len(fixed)} ({len(moved)} with a moved far end); as frozen: {fired}")
    rec["controls_pass"] = k0 and k1 and c0_ok
    log(f"K0 {k0}, K1 {k1}, C0 {c0_ok} ({len(c0_fail)} failures); controls pass: {rec['controls_pass']}")
    log.dump(rec)
    return rec["controls_pass"]


# ---------------------------------------------------------------- real stage
def widened(wf, rel):
    """max(rel_k, the strings' rounding_k) for k < 23, the J1250 file at the node (#683)"""
    _, rows = sl.rounding(json.load(open(sl.node_file(pg.REAL["J1250"], wf))))
    return [max(rel[k], float(rows[k][2])) for k in range(N23)] + list(rel[N23:])


def summary(s):
    return {"M_min": s["M_min"], "M_lo": s["M_lo"], "C1_pass": s["C1_pass"], "precision": s["precision"],
            "chi": {r["M"]: r["chi"] for r in s["rows"]}, "l2": {r["M"]: r["l2"] for r in s["rows"]}}


def run_real(outdir, mut=None):
    ctl = json.load(open(os.path.join(outdir, "control.json")))
    if not ctl["controls_pass"]:
        print("the controls failed: no real run (frozen rule)")
        return False
    log = tz.Log(outdir, "real", mut)
    rec = {"frozen": FROZEN, "mutate": mut, "scans": {}, "widened": {}, "J1500": {}}
    real = tc.real_sig()
    j15 = pg.load_scan(pg.REAL["J1500"])
    ends = {float(wf): {k: (complex(*v) if isinstance(v, list) and len(v) == 2 and not isinstance(v[0], list) else v)
                        for k, v in e.items()} for wf, e in ctl["ends"].items()}
    ok = True
    errs = {wf: tc.err_model(wf)[0] for wf in pg.NODES}
    for group, wf in [(g, w) for g in (JUDGED, CONTEXT) for w in pg.NODES]:
        rel = errs[wf]
        relw = widened(wf, rel)
        sig = real[wf][:N23]
        log.dump(rec)
        for v in group:
            P1, ex, _ = spec(v, wf, ends)
            for cls, _ in CLASSES:
                fit = end_basis(wf, cls, P1, ex, DPS_FIT)
                chk = end_basis(wf, cls, P1, ex, DPS_CHECK)
                if mut == "c1dup":
                    chk.cols[N23 - 1] = list(chk.cols[N23 - 2])
                s = tz.scan(fit, chk, sig, rel, N23, log, f"wf {wf} {v} {cls}")
                rec["scans"].setdefault(wf, {}).setdefault(v, {})[cls] = summary(s)
                ok &= s["C1_pass"]
                if not s["C1_pass"] and not mut:
                    log("C1 fired on real data: stop (frozen rule)")
                    log.dump(rec)
                    return False
                if mut or v not in JUDGED:
                    continue
                sw = tz.scan(fit, chk, sig, relw, N23, log, f"wf {wf} {v} {cls}, widened errors (context)")
                rec["widened"].setdefault(wf, {}).setdefault(v, {})[cls] = summary(sw)
                if wf == 1.0 and cls != "smooth":
                    sj = tz.scan(fit, chk, j15[wf][:N23], rel, N23, log, f"wf {wf} {v} {cls}, J1500 (context)")
                    rec["J1500"].setdefault(v, {})[cls] = summary(sj)
    scans = rec["scans"]
    if mut:
        at_n = [(wf, v, c) for wf in pg.NODES for v, d in scans[wf].items() for c, s in d.items() if s["M_min"] == N23]
        below = [(wf, v, c) for wf in pg.NODES for v, d in scans[wf].items() for c, s in d.items() if s["M_min"] != N23]
        fired = bool(at_n) and all(not scans[wf][v][c]["C1_pass"] for wf, v, c in at_n)
        iso = all(scans[wf][v][c]["C1_pass"] for wf, v, c in below)
        rec["mutation"] = {"n_at_n": len(at_n), "n_below": len(below), "fired_at_n": fired, "below_n_pass": iso}
        log(f"mutation {mut}: {len(at_n)} scans with M_min = n, the amended C1 fired on each: {fired}; {len(below)} "
            f"scans with M_min < n, all still pass: {iso}; C1 pass overall: {ok}")
        log.dump(rec)
        return ok

    def table(src, vs):
        out = {}
        for v in vs:
            per = {cls: [src[wf][v][cls]["M_min"] for wf in pg.NODES] for cls, _ in CLASSES}
            fit_cls = [cls for cls, mm in per.items() if all(m is not None and m <= M_FIT for m in mm)]
            out[v] = {"M_min": per, "M_lo": {cls: [src[wf][v][cls]["M_lo"] for wf in pg.NODES] for cls, _ in CLASSES},
                      "fits_classes": fit_cls, "fits": bool(fit_cls)}
        return out
    judged = table(scans, JUDGED)
    context = table(scans, CONTEXT)
    wide = table(rec["widened"], JUDGED)
    far = [v for v in ENDS if judged[v]["fits"]]
    exv = [v for v in JUDGED if v.startswith("ex") and judged[v]["fits"]]
    verdict = ("both" if far and exv else "the far end" if far else "the exponent" if exv else
               "none of the tested moves gives a short twin")
    c2 = {wf: {cls: {"M_min": scans[wf]["kstar"][cls]["M_min"], "chi22": scans[wf]["kstar"][cls]["chi"][22],
                     "ref": REF672[wf][cls]} for cls, _ in CLASSES} for wf in pg.NODES}
    c2_pass = all(x["M_min"] == N23 and abs(x["chi22"] / x["ref"] - 1) <= 1e-3 for d in c2.values() for x in d.values())
    changed = [v for v in JUDGED if wide[v]["fits"] != judged[v]["fits"]]
    rec.update({"judged": judged, "context": context, "widened_table": wide, "fitting_far_end": far,
                "fitting_exponent": exv, "verdict": verdict, "C2_baseline": {"rows": c2, "pass": c2_pass},
                "widened_changes_verdict": changed, "C1_pass": ok})
    for v in JUDGED + CONTEXT:
        t = judged.get(v) or context[v]
        log(f"{'judged ' if v in JUDGED else 'context'} {v:14s}: M_min " + "; ".join(
            f"{cls} {t['M_min'][cls]}" for cls, _ in CLASSES) + f" -> fits: {t['fits_classes'] or 'no'}")
    log(f"C2 baseline (kstar against #672): {'OK' if c2_pass else 'DISCREPANCY'}; widened errors change the verdict of "
        f"{changed or 'no variant'}")
    log(f"verdict: far-end variants that fit {far or 'none'}; exponent variants that fit {exv or 'none'} -> {verdict}")
    log.dump(rec)
    return ok


def main():
    argv = sys.argv[1:]
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    if "--selftest" in argv:
        if mut not in (None, "endmom", "upoly"):
            raise SystemExit(f"unknown selftest mutation {mut}")
        sys.exit(0 if selftest(mut) else 1)
    argv = [a for a in argv if not a.startswith("--")]
    if len(argv) != 2 or argv[0] not in ("control", "real"):
        raise SystemExit(__doc__)
    stage, outdir = argv
    if mut not in ({"control": (None, "farend", "exshift"), "real": (None, "c1dup")}[stage]):
        raise SystemExit(f"unknown {stage} mutation {mut}")
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    os.makedirs(outdir, exist_ok=True)
    sys.exit(0 if (run_control if stage == "control" else run_real)(outdir, mut) else 1)


if __name__ == "__main__":
    main()
