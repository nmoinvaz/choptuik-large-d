#!/usr/bin/env python3
"""#667's two-set plants and #684's far-end plants with the zero factor at the real series' own zero (issue #703).

Every plant of #541-#684 and #667 had its zero factor 1 - x/x_ref at x_ref = planted_plants.x0_planted(wf, "1.0000"),
5-7e-3 from the real series' zero. #702 (pade_twin_near.py zfit; notes/PADE_FAR_END.md 14) located that zero by the l2 of
the least-squares kappa_* chord twin at M = 20 = n - 3, and with the twin's zero there the real chi at 22 terms falls from
9e4 to 402-427. Here the plants are rebuilt with their zero factor at that located zero, and judged against the real series
in twin bases with the same zero factor, so that plant and real differ only in the tested ingredient.

Zero. ZLOC[wf]: #702's located zero (zfit, 40 digits, I702-mac/real.json, raw-data ff0920c). Every plant's pref and every
basis' zero factor is planted_plants.factor([ZLOC]) at the plants' 60 digits, so a plant and a basis share it exactly.

Plants. #667's (pade_second_set.make_plant): pref (L_main + lam L_2) / S_0 with L_main FarPlant's chord2 or arc density at
kappa_*, L_2 the density s^4 (1-s)^alpha on a second curve, lam set by the second set's share eta e^(i phi) of the real
sigma_22, c_m fitted to the real sigma_0..sigma_12 minus the second set's part (M = 6, or 8). The only change: pref at ZLOC
(ZMain, rezero). #684's: #547's chord2 plant with its far end at R = 1.85 on kappa_*'s ray (K1) and the far-end variants'
chord2 plants (pade_twin_end.EndPlant), each with pref at ZLOC and c_m refitted.

Bases (pade_second_set.Proj, one Householder QR per basis, 150 digits, C1 at 250): the judged basis zk/chord is the kappa_*
chord twin with the zero at ZLOC, #702's zfit basis. Context: #667's other 14 bases (kstar, ratio, pole8..10 x chord,
stepped arc, smooth arc) with the zero at ZLOC, and nozero/chord (pref = 1). The errors are the real series' absolute
errors, err_k = rel_k |sigma_k(real)|, the same for every plant.

Judge, per node: (a) m_DA <= 5 and (b) d_P <= 0.03, as #667; (c') in zk/chord, M_min = 23 and 0.1 <= chi_22 /
chi_22(real) <= 10, with chi_22(real) the real series' in the same basis (#702's 411, 402, 427). A plant reproduces the
real series if (a), (b) and (c') hold at all three nodes.

Relocation, judged apart as (c''). The real series' zero was located by l2 at 20 terms, so its 402 is a misfit after that
location. The same locator applied to a plant may move its zero and lower its misfit. (c'') is (c') in the kappa_* chord
basis with the zero at the plant's own located zero, against the same chi_22(real). gn_locate: Gauss-Newton in w = 1/z with
the coefficients projected out (variable projection; Golub and Pereyra 1973, SIAM J. Numer. Anal. 10, 413): the
least-squares fit at M = 20 against the columns plus the model's derivative in w, sum_m c_m d col_m / dw, whose
coefficient is the step, halved until l2 falls. From ZLOC every accepted step lowers l2 at 20 terms, so at the located zero
chi_22 <= l2_22 <= l2_20 <= l2_20 at ZLOC, and a plant with l2_20(ZLOC) < 0.1 chi_22(real) fails (c'') unrelocated.

    python3 pade_second_zero.py --selftest [--mutate=gn]
    python3 pade_second_zero.py control OUTDIR [--mutate=zero|bzero|judge] [--allow_unraw]
    python3 pade_second_zero.py far OUTDIR [--mutate=farend] [--allow_unraw]        (reads OUTDIR/control.json)
    python3 pade_second_zero.py grid OUTDIR [--allow_unraw]       (reads OUTDIR/control.json; resumes from grid.jsonl)

Exit 1 when a frozen control fails or the gate forbids a stage. Numerical evidence on plants, not a bound.
"""
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pade_second_set as ss                                           # noqa: E402
import pade_twin_near as tn                                            # noqa: E402

te, tz, tc, F, pg, pp = ss.te, ss.tz, ss.tc, ss.F, ss.pg, ss.pp
N23 = ss.N23
NODES = pg.NODES
WF0 = 1.0
DPS_FIT, DPS_CHECK = ss.DPS_FIT, ss.DPS_CHECK
M_LOC = 20
JUDGED_SCAN = "kstar/chord"
SCANS = ss.SCANS + ("nozero/chord",)
CHI_FACTOR = ss.CHI_FACTOR
NEED_DA, TOL_POLE = ss.NEED_DA, ss.TOL_POLE
# I702-mac/real.json (raw-data repository, ff0920c): zfit's zero, located by l2 at 20 terms in the kappa_* chord class
ZLOC_STR = {0.99: ("7.352744280003409465950514289706773142584", "0.07652437116245846414756161651571977476334"),
            1.0: ("7.394084293132979441658267161201297381704", "-0.00001385933440798451687001279813429531307197"),
            1.01: ("7.434373901519574659899775929945604199063", "-0.07724371070105616539462708075304154710774")}
# the same file: the real chi at 22 terms and l2 at 20 terms of zfit (the zero located) and nozero (no zero factor)
REF702 = {0.99: {"zfit": 411.2214920221311, "zfit_l2_20": 7152.104881061936, "nozero": 268.7022134913358,
                 "nozero_l2_20": 107239.80075228636},
          1.0: {"zfit": 401.7976584156973, "zfit_l2_20": 7171.764443903141, "nozero": 256.1066224304467,
                "nozero_l2_20": 95950.71127804488},
          1.01: {"zfit": 427.4915095089004, "zfit_l2_20": 9143.054496074992, "nozero": 259.9752391907437,
                 "nozero_l2_20": 140962.18919773618}}
# the grid: #667's (pade_second_set: kinds, curves, alphas, eta 1e-9..1e-1 in decades, phases), the near part first
NEAR_CFGS = tuple(c for c in ss.CONFIGS if c != "low")      # seg(a, b) near P0 + kappa_*, and the sum set
NEAR_ETAS = (1e-2, 2e-2, 5e-2, 1e-1)
N_CLOSEST = 20
N_AC_MAX = 60
TOL_R5 = 1e-8
S1_MAX = 0.1                        # the far-end plant accounts for the real misfit only within the factor 10 of (c')
S2_FACTOR = 2.3                     # #684: the far end at the ratio-law kappa and the k = 10 pole lowers chi_22 2.3-2.7x
S3_FLOOR = 3e4                      # #684: the real misfit stays above 3e4 error units at every far end
S4_MAX = 1 / 290                    # #667: a set near P0 + kappa_* misfits at least 290 times less, in every basis
FROZEN = {                          # written before the control, far and grid stages were first run (PADE_FAR_END.md 15.2)
    "zero": "ZLOC = #702's zfit zero per node (I702-mac/real.json, ff0920c), not the [11/11] Pade zero (2e-5 away, where "
            "the real chi_22 is 636-720); every plant's pref and every basis' zero factor = planted_plants.factor([ZLOC]) "
            "at 60 digits",
    "plants": "#667's plants (pade_second_set.make_plant) with pref at ZLOC and the main c_m refitted (M = 6, or 8 if the "
              "largest relative residual to sigma_0..sigma_12 exceeds 1e-2); nothing else changed",
    "grid": "part near: kind chord2, arc; curve seg(a, b) (9) and sum; alpha 0 or beta - 1; eta 1e-2, 2e-2, 5e-2, 1e-1; "
            "phi 0, 90, 180, 270 (640 plants). Part rest: #667's 1584 plants (eta 1e-9..1e-1 in decades, curves seg, low, "
            "sum) not in part near (1264). All at wf 1.00",
    "nodes": "wf 0.99 and 1.01 for every plant that passes (a), (b) and (c') or (c'') at wf 1.00, for the plants that pass "
             "(a) and (c') there (at most 60, smallest D first) and (a) and (c'') (at most 60, smallest D''), and for the "
             "20 with the smallest D",
    "bases": "judged: zk/chord, the kappa_* chord twin (P0, kappa_*, beta - 1) with the zero at ZLOC; context: #667's "
             "other 14 bases with the zero at ZLOC, and nozero/chord; projections at 150 digits, C1 at 250",
    "a_b": "#667's (a) m_DA <= 5 and (b) d_P <= 0.03, unchanged",
    "c_prime": "in zk/chord: M_min = 23 and 0.1 <= chi_22 / chi_22(real) <= 10, chi_22(real) the real series' in the "
               "same basis; d_T = |log10(chi_22 / chi_22(real))|",
    "c_reloc": "(c''): (c') in the kappa_* chord basis with the zero at the plant's own zero located by gn_locate (l2 at "
               "20 terms, from ZLOC), against the same chi_22(real); evaluated when R5 passes; a plant whose l2 at 20 "
               "terms in zk/chord is below 0.1 chi_22(real) fails (c'') unrelocated, since from ZLOC chi_22 <= l2_22 <= "
               "l2_20 at the located zero <= l2_20 at ZLOC",
    "reproduces": "(a), (b) and (c') at all three nodes; and apart, (a), (b) and (c'') at all three nodes",
    "question": "the coordinator's: does a plant's zk/chord chi_22 come within a factor 10 of the real's ((c')), and does "
                "it also meet (a) and (b); counts of (c'), (a)+(c'), (b)+(c') and all three at wf 1.00, and at all three "
                "nodes for the plants run there; the same with (c'')",
    "closest": "D = max(m_DA / 5, d_P / 0.03, d_T) per node, D_all its max over the nodes; smallest D_all among the plants "
               "run at every node. D'' is D with d_T from (c'')'s ratio (from the bound when not relocated)",
    "R_controls": "R1 the real m_DA <= 5 at every node; R2 #667's R2 (pade_twin_end.check_ends); R3z the real chi_22 by "
                  "projection in zk/chord and nozero/chord within 1e-6 relative of REF702, M_min = M_lo = 23 and C1 in "
                  "both; R4z the eta = 0 plants: chord2 equals pade_twin_near.NearPlant with its zero at ZLOC, sigma_0.."
                  "sigma_31 exactly, and arc equals its own FarPlant refitted at ZLOC exactly, at every node; R5 "
                  "gn_locate on the real series, from x_ref, returns #702's zfit zero within 1e-8 at every node",
    "J_controls": "J+ the real series judged as a plant passes (a), (b), (c') at every node; J- the eta = 0 chord2 and "
                  "arc plants at wf 1.00 fail the judge, each with M_min = 6 in its own class's kappa_* basis at ZLOC "
                  "(chord2: zk/chord, arc: kstar/smooth)",
    "gate": "the far and grid stages run only if R1-R4z, J+ and J- pass; (c'') is evaluated only if R5 passes too",
    "mutations": "control --mutate=zero (the plants keep their zero at x_ref, the bases at ZLOC): J- and R4z must fail. "
                 "--mutate=bzero (every basis' zero ZLOC + 1e-6, the plants' unchanged): R3z and J- must fail. "
                 "--mutate=judge (REF541 + 0.05, the reference chi_22 x 100, the DA2 radius 0.5): each of J+'s parts must "
                 "fail. far --mutate=farend (K1's own-end basis at 1.86): K1z(a) must fail. --selftest --mutate=gn "
                 "(the locator returns its start): the planted locator check must fail. Each run exits 1",
    "relocation": "gn_locate from ZLOC on every plant, at every node it runs, unless l2_20(ZLOC) < 0.1 chi_22(real); "
                  "its zero, the distance from ZLOC, chi at 20..22 terms, M_min and l2 at 20 terms at the located zero",
    "far_684": "K1z: #547's chord2 plant at R = 1.85 with pref at ZLOC: (a) chi <= 1e-30 in its own-end chord basis at "
               "ZLOC at its own M; (b) M_min >= 19 in zk/chord, with chi_22 / chi_22(real). Sensitivity: the ratio, "
               "pole8..10 far-end chord2 plants at ZLOC in zk/chord. Real (context): chi_22 in the ratio, pole8..10 and "
               "ex +- 0.3 chord bases at ZLOC",
    "statements": "S1 (#684) 'the far end alone does not account for the real misfit' survives if K1z's and every "
                  "sensitivity plant's chi_22 / chi_22(real) < 0.1 at every node. S2 (#684) 'moving the far end to the "
                  "ratio-law kappa or the k = 10 pole lowers the real chi_22 2.3-2.7 times' survives if both are at most "
                  "chi_22(zk/chord) / 2.3 at every node, at ZLOC. S2b 'the far end matters far more than the exponent' "
                  "survives if |log chi(ratio)/chi(kstar)| exceeds |log chi(ex +- 0.3)/chi(kstar)| at every node. S3 "
                  "(#684) 'the real misfit stays above 3e4 error units' survives if the real chi_22 >= 3e4 in all five "
                  "far-end chord bases at ZLOC. S4 (#667) 'a set near P0 + kappa_* misfits at least 290 times less than "
                  "the real series in every basis, even at 10% of sigma_22' survives if every plant of part near and "
                  "of #667's near curves has chi_22 / chi_22(real) <= 1/290 in zk/chord at wf 1.00, and apart with "
                  "(c'')'s chi_22. S5 (#667) 'only the low set reaches the real misfit' survives if some plant passes "
                  "(c') at wf 1.00 and every one that does has the low curve, and apart for (c''). Each failure is a "
                  "discrepancy, reported",
    "C1": "#672's amended C1 on every plant scan; a failure is listed and the stage exits 1",
    "context": "per plant: the other 15 scans' chi_22 / chi_22(real) and #667's (c) on the 15 bases at ZLOC; chi at 20 "
               "and 21 terms and l2 at 20 terms in zk/chord, and l2_20 / l2_20(real); the second set's share; the DA2 "
               "median; the [11/11] Pade zero of the eta = 0 plants against ZLOC",
}


# ---------------------------------------------------------------- the zero
def zloc(wf, mut=None):
    with mp.workdps(pg.DPS_PLANT):
        z = mp.mpc(mp.mpf(ZLOC_STR[wf][0]), mp.mpf(ZLOC_STR[wf][1]))
        return z + (mp.mpf("1e-6") if mut == "bzero" else 0)


def rezero(pl, zero, real_sig):
    """the plant pl (pade_geometry.Plant) with its zero factor at zero and its c_m refitted as FarPlant does"""
    mp.mp.dps = pg.DPS_PLANT
    pl.pref = [mp.mpc(1)] if zero is None else pp.factor([mp.mpc(zero)])
    for M in (6, 8):
        pl.mom = pl.moments(M)
        res = pl.fit(real_sig, M)
        if max(res) <= 1e-2:
            break
    pl.M, pl.resid = M, res
    return pl


class ZMain(ss.Main):
    """pade_second_set.Main with the main density's zero factor at zero (None: kept at x_ref, the #667 plants)"""

    def __init__(self, wf, kind, real_sig, zero):
        self.wf, self.kind, self.real = wf, kind, real_sig
        self.base = F.FarPlant(wf, kind, real_sig)
        if zero is not None:
            rezero(self.base, zero, real_sig)
        mp.mp.dps = pg.DPS_PLANT
        self.mom8 = self.base.moments(8)
        self.cols = [pp.polymul(self.base.pref, self.mom8[m], pg.KM + 1) for m in range(8)]


def zbasis(wf, cls, P1, ex, dps, zero, M=N23, K=N23):
    """pade_twin_end.end_basis with the zero factor at zero (None: no zero factor)"""
    with mp.workdps(pg.DPS_PLANT):
        pref = [mp.mpc(1)] if zero is None else pp.factor([mp.mpc(zero)])
        arc = F.FarArc(wf, "chord" if cls == "chord" else "arc", mp.mpc(P1))
        ex = mp.mpc(ex)
    pieces = F.PIECES[{"chord": "chord2", "arc": "arcjump", "smooth": "arc"}[cls]]
    with mp.workdps(dps):
        if cls == "chord":
            mom = tc.chord_moments(arc.P0, arc.P1, ex, M, K - 1)
        else:
            mom = tz.arc_moments(arc, ex, pieces, M, K)
        return tz.Cols(mom, pref, ex, dps, pieces)


def projections(wf, mut=None):
    """{scan: (Proj at 150 digits, Proj at 250)}: #667's 15 bases and nozero/chord, the zero at ZLOC (bzero: + 1e-6)"""
    real = tc.real_sig()[wf][:N23]
    rel, _ = tc.err_model(wf)
    ends = te.far_ends(wf)
    z = zloc(wf, mut)
    out = {}
    specs = [(f"{v}/{c}", te.spec(v, wf, ends)[:2], c, z) for v in ss.VARIANTS for c, _ in ss.CLASSES]
    specs.append(("nozero/chord", (te.kstar(wf), te.ex0(wf)), "chord", None))
    for key, (P1, ex), cls, zero in specs:
        out[key] = tuple(ss.Proj(zbasis(wf, cls, P1, ex, dps, zero), tc.weights(real, rel, N23, 1.0, dps))
                         for dps in (DPS_FIT, DPS_CHECK))
    return out, ends


def twin_record(projs, sig):
    """pade_second_set.twin_record, with l2 at 20..22 terms kept"""
    out = {}
    n = N23
    for key, (pf, pc) in projs.items():
        rows = pf.rows(sig)
        m_min = next((M for M in sorted(rows) if rows[M][0] <= 1), None)
        m_lo = next((M for M in sorted(rows) if rows[M][1] <= math.sqrt(n)), None)
        prec = []
        if m_min:
            rc = pc.rows(sig, Ms=(m_min - 1, m_min))
            for M in (m_min - 1, m_min):
                if M in rows and M in rc:
                    prec.append({"M": M, "chi": rows[M][0], "chi_check": rc[M][0],
                                 **tz.c1_row(M, n, rows[M][0], rc[M][0])})
        out[key] = {"M_min": m_min, "M_lo": m_lo, "chi": {M: rows[M][0] for M in (20, 21, 22)},
                    "l2": {M: rows[M][1] for M in (20, 21, 22)},
                    "C1_pass": bool(prec) and all(p["pass"] for p in prec), "precision": prec}
    return out


# ---------------------------------------------------------------- the judge
def judge(rec, ref):
    """ref: {scan: (chi_22, l2_20)} of the real series in the same bases"""
    tw = rec["twin"]
    a = rec["da2"]["m_DA"] <= NEED_DA - 1
    d_P = max(rec["pade"][k]["dist"] for k in (8, 9, 10))
    ratios = {s: tw[s]["chi"][22] / ref[s][0] for s in SCANS}
    r = ratios[JUDGED_SCAN]
    d_T = abs(math.log10(r)) if r > 0 else math.inf
    mm = tw[JUDGED_SCAN]["M_min"] == N23
    c = mm and 1 / CHI_FACTOR <= r <= CHI_FACTOR
    c667 = (all(tw[s]["M_min"] == N23 for s in ss.SCANS)
            and all(1 / CHI_FACTOR <= ratios[s] <= CHI_FACTOR for s in ss.SCANS))
    D = max(rec["da2"]["m_DA"] / (NEED_DA - 1), d_P / TOL_POLE, d_T)
    return {"a": a, "b": d_P <= TOL_POLE, "c": c, "all": a and d_P <= TOL_POLE and c, "m_DA": rec["da2"]["m_DA"],
            "d_P": d_P, "d_T": d_T, "ratio": r, "M_min": tw[JUDGED_SCAN]["M_min"], "D": D,
            "l2_ratio": tw[JUDGED_SCAN]["l2"][20] / ref[JUDGED_SCAN][1], "ratios": ratios, "c667_ZLOC": c667,
            "C1_pass": all(tw[s]["C1_pass"] for s in SCANS)}


def evaluate(sig, wf, projs, ref, rad=ss.RAD_DA, ref541=None):
    rec = {"da2": ss.da2_record(sig, wf, rad), "pade": ss.pade_record(sig, wf, ref541), "twin": twin_record(projs, sig)}
    rec["judge"] = judge(rec, ref)
    return rec


def real_ref(projs, wf):
    tw = twin_record(projs, tc.real_sig()[wf][:N23])
    return {s: (tw[s]["chi"][22], tw[s]["l2"][20]) for s in SCANS}, tw


# ---------------------------------------------------------------- the zero locator (relocation context)
class ZeroFit:
    """the kappa_* chord moments at 150 digits, m < M_LOC, k < 23; columns factor([z]) * mom"""

    def __init__(self, wf):
        with mp.workdps(pg.DPS_PLANT):
            arc = F.FarArc(wf, "chord", mp.mpc(te.kstar(wf)))
            ex = te.ex0(wf)
        with mp.workdps(DPS_FIT):
            self.mom = tc.chord_moments(arc.P0, arc.P1, ex, M_LOC, N23 - 1)

    def cols(self, z):
        with mp.workdps(DPS_FIT):
            w = 1 / mp.mpc(z)
            return [[m[0]] + [m[k] - w * m[k - 1] for k in range(1, N23)] for m in self.mom]


def gn_locate(zf, sig, rel, start, maxit=40, tol=mp.mpf("1e-30"), mut=None):
    """the zero minimising the least-squares l2 at M = 20 (as pade_twin_near.locate), by Gauss-Newton in w = 1/z with
    the coefficients projected out; each step halved until l2 falls. mut gn: returns the start, unsearched"""
    err = tc.weights(sig, rel, N23, 1.0, DPS_FIT)
    steps = []
    with mp.workdps(DPS_FIT):
        z = mp.mpc(start)

        def fit(z, extra=None):
            cols = zf.cols(z) + ([extra] if extra is not None else [])
            return tc.lsq(SimpleNamespace(cols=cols, dps=DPS_FIT), sig, err, N23, len(cols))
        c, _, l2 = fit(z)
        if mut == "gn":
            return {"zero": z, "l2_20": l2, "steps": steps, "converged": False, "mutated": True}
        conv = False
        for it in range(maxit):
            J = [mp.mpc(0)] + [-mp.fsum(c[m] * zf.mom[m][k - 1] for m in range(M_LOC)) for k in range(1, N23)]
            c2, _, _ = fit(z, J)
            dw = c2[M_LOC]
            w = 1 / z
            t = mp.mpf(1)
            while t > mp.mpf("1e-12"):
                zn = 1 / (w + t * dw)
                cn, _, l2n = fit(zn)
                if l2n <= l2:
                    break
                t /= 2
            dz = abs(zn - z)
            steps.append({"it": it, "t": float(t), "dz": float(dz), "l2_20": l2n})
            if l2n > l2:
                break
            z, c, l2 = zn, cn, l2n
            if dz < tol:
                conv = True
                break
        return {"zero": z, "zero_str": mp.nstr(z, 40), "l2_20": l2, "steps": steps, "converged": conv}


def relocated(wf, zf, sig, rel, start):
    """gn_locate from start, then the kappa_* chord scan at the located zero (chi at 20..22, M_min)"""
    t0 = time.time()
    loc = gn_locate(zf, sig, rel, start)
    with mp.workdps(pg.DPS_PLANT):
        z = mp.mpc(loc["zero"])
    real = tc.real_sig()[wf][:N23]
    p = ss.Proj(zbasis(wf, "chord", te.kstar(wf), te.ex0(wf), DPS_FIT, z), tc.weights(real, rel, N23, 1.0, DPS_FIT))
    rows = p.rows(sig)
    m_min = next((M for M in sorted(rows) if rows[M][0] <= 1), None)
    return {"zero": complex(loc["zero"]), "zero_str": loc.get("zero_str"), "dz_start": float(abs(loc["zero"] - start)),
            "l2_20": loc["l2_20"], "converged": loc["converged"], "n_steps": len(loc["steps"]), "M_min": m_min,
            "chi": {M: rows[M][0] for M in (20, 21, 22)}, "seconds": time.time() - t0}


# ---------------------------------------------------------------- selftest
def selftest(mut=None):
    ok = True
    wf = WF0
    real = tc.real_sig()
    rel, _ = tc.err_model(wf)
    # 1: zbasis at x_ref is pade_twin_end.end_basis, bit for bit, in each class
    with mp.workdps(pg.DPS_PLANT):
        xr = pp.x0_planted(wf, "1.0000")
    for cls, P1 in (("chord", te.kstar(wf)), ("arc", te.REF541[wf][9]), ("smooth", te.kstar(wf))):
        a = zbasis(wf, cls, P1, te.ex0(wf), DPS_FIT, xr)
        b = te.end_basis(wf, cls, P1, te.ex0(wf), DPS_FIT)
        same = all(a.cols[m][k] == b.cols[m][k] for m in range(N23) for k in range(N23))
        print(f"selftest 1: zbasis({cls}, zero x_ref) equals pade_twin_end.end_basis bit for bit: {same}")
        ok &= same
    # 2: the eta = 0 chord2 plant at ZLOC equals pade_twin_near.NearPlant with that zero
    z = zloc(wf)
    m = ZMain(wf, "chord2", real[wf], z)
    sig, _ = ss.make_plant(m, None, 0, 0)
    sp = dict(tn.spec("kstar", wf, {}), zero=z)
    npl = tn.NearPlant(wf, sp, real[wf])
    mp.mp.dps = pg.DPS_PLANT
    d = max(abs(x - y) for x, y in zip(sig, npl.sigma()))
    print(f"selftest 2: the eta = 0 chord2 plant at ZLOC against NearPlant at ZLOC, sigma_0..31: max |diff| {float(d):.1e} "
          f"(pass = 0)")
    ok &= d == 0
    # 3: the locator on a planted zero: the chord2 plant with its zero at ZLOC + 0.003 + 0.002i, from ZLOC
    with mp.workdps(pg.DPS_PLANT):
        zt = z + mp.mpc("0.003", "0.002")
    pl = rezero(F.FarPlant(wf, "chord2", real[wf]), zt, real[wf])
    mp.mp.dps = pg.DPS_PLANT
    loc = gn_locate(ZeroFit(wf), pl.sigma()[:N23], rel, z, mut=mut)
    dz = float(abs(loc["zero"] - zt))
    print(f"selftest 3: gn_locate on a plant whose zero is ZLOC + 0.003 + 0.002i, from ZLOC: off by {dz:.1e} (pass < "
          f"1e-12), l2 at 20 terms {loc['l2_20']:.1e}, {len(loc['steps'])} steps")
    ok &= dz < 1e-12
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- control stage
def run_control(outdir, mut=None):
    log = tz.Log(outdir, "control", mut)
    rec = {"frozen": FROZEN, "mutate": mut, "nodes": {}, "zloc": {wf: ZLOC_STR[wf] for wf in NODES}}
    real = tc.real_sig()
    ok = {k: True for k in ("R1", "R2", "R3z", "R4z", "R5", "J+", "J-")}
    jplus = {"a": True, "b": True, "c": True}
    ends_all = {}
    for wf in NODES:
        t0 = time.time()
        projs, ends = projections(wf, mut if mut == "bzero" else None)
        ends_all[wf] = ends
        node = rec["nodes"].setdefault(wf, {})
        ref, tw = real_ref(projs, wf)
        node["real_ref"] = ref
        node["real_twin"] = tw
        dev = {"zfit": abs(ref[JUDGED_SCAN][0] / REF702[wf]["zfit"] - 1),
               "nozero": abs(ref["nozero/chord"][0] / REF702[wf]["nozero"] - 1)}
        r3 = (all(d <= 1e-6 for d in dev.values())
              and all(tw[s]["M_min"] == N23 == tw[s]["M_lo"] and tw[s]["C1_pass"] for s in (JUDGED_SCAN, "nozero/chord")))
        ok["R3z"] &= r3
        node["R3z"] = {"dev": dev, "pass": r3}
        log(f"R3z wf {wf}: the real chi_22 in zk/chord {ref[JUDGED_SCAN][0]:.6f} (#702 zfit {REF702[wf]['zfit']:.6f}), "
            f"in nozero/chord {ref['nozero/chord'][0]:.6f} (#702 {REF702[wf]['nozero']:.6f}): rel dev "
            f"{max(dev.values()):.1e} (pass <= 1e-6); M_min, M_lo 23 and C1 in both -> {'OK' if r3 else 'BAD'} "
            f"({time.time() - t0:.0f}s)")
        log(f"context wf {wf}: the real chi_22 at ZLOC per basis: " + ", ".join(f"{s} {ref[s][0]:.1f}" for s in SCANS)
            + f"; M_min " + ", ".join(f"{s} {tw[s]['M_min']}" for s in SCANS if tw[s]["M_min"] != N23)
            + " (others 23)")
        # R1, J+: the real series judged as a plant
        refj = {s: (v[0] * (100 if mut == "judge" else 1), v[1]) for s, v in ref.items()}
        r541 = {k: z + (0.05 if mut == "judge" else 0) for k, z in ss.REF541[wf].items()}
        rr = evaluate(real[wf], wf, projs, refj, rad=0.5 if mut == "judge" else ss.RAD_DA, ref541=r541)
        if mut != "judge":
            ok["R1"] &= rr["da2"]["m_DA"] <= NEED_DA - 1
        ok["J+"] &= rr["judge"]["all"]
        for p in jplus:
            jplus[p] &= rr["judge"][p]
        node["real"] = rr
        log(f"R1/J+ wf {wf}: real m_DA {rr['da2']['m_DA']} (R1 pass <= 5); d_P {rr['judge']['d_P']:.1e}; ratio "
            f"{rr['judge']['ratio']:.3g}; judge (a) {rr['judge']['a']} (b) {rr['judge']['b']} (c') {rr['judge']['c']} "
            f"-> J+ {'OK' if rr['judge']['all'] else 'BAD'}")
        # R4z, J-: the eta = 0 plants at ZLOC
        zp = None if mut == "zero" else zloc(wf)
        for kind, own in (("chord2", "kstar/chord"), ("arc", "kstar/smooth")):
            main = ZMain(wf, kind, real[wf], zp)
            sig, info = ss.make_plant(main, None, 0, 0)
            mp.mp.dps = pg.DPS_PLANT
            if kind == "chord2":
                sp = dict(tn.spec("kstar", wf, {}), zero=zloc(wf))
                ref_sig = tn.NearPlant(wf, sp, real[wf]).sigma()
            else:
                ref_sig = rezero(F.FarPlant(wf, kind, real[wf]), zloc(wf), real[wf]).sigma()
            mp.mp.dps = pg.DPS_PLANT
            d4 = max(abs(a - b) for a, b in zip(sig, ref_sig))
            r4 = d4 == 0
            ok["R4z"] &= r4
            pz = tn.pade_zero(sig, complex(zloc(wf)))
            out = {"R4_maxdiff": float(d4), "R4_pass": r4, "M": info["M"], "fit_resid": info["fit_resid"],
                   "pade_zero": pz, "pade_zero_off": abs(pz - complex(zloc(wf)))}
            if wf == WF0:
                pr = evaluate(sig, wf, projs, ref)
                jm = (not pr["judge"]["all"]) and pr["twin"][own]["M_min"] == 6
                ok["J-"] &= jm
                out.update({"plant": pr, "J-_pass": jm})
                log(f"J- wf {wf} {kind} eta 0: m_DA {pr['da2']['m_DA']}; d_P {pr['judge']['d_P']:.3f}; M_min in {own} "
                    f"{pr['twin'][own]['M_min']} (chi at 22 terms {pr['twin'][own]['chi'][22]:.1e}); ratio in zk/chord "
                    f"{pr['judge']['ratio']:.1e}; judge (a) {pr['judge']['a']} (b) {pr['judge']['b']} (c') "
                    f"{pr['judge']['c']} -> J- {'OK' if jm else 'BAD'}")
            node.setdefault("eta0", {})[kind] = out
            log(f"R4z wf {wf} {kind}: eta 0 plant at {'x_ref' if zp is None else 'ZLOC'} against the reference at ZLOC, "
                f"sigma_0..31 max |diff| {float(d4):.1e} (M {info['M']}, fit {info['fit_resid']:.1e}) -> "
                f"{'OK' if r4 else 'BAD'}; context: its [11/11] Pade zero off ZLOC by {out['pade_zero_off']:.1e}")
        # R5: the locator on the real series, from x_ref
        if not mut:
            rel, _ = tc.err_model(wf)
            with mp.workdps(pg.DPS_PLANT):
                xr = pp.x0_planted(wf, "1.0000")
            t1 = time.time()
            loc = gn_locate(ZeroFit(wf), real[wf][:N23], rel, xr)
            d5 = float(abs(loc["zero"] - zloc(wf)))
            r5 = d5 <= TOL_R5
            ok["R5"] &= r5
            node["R5"] = {"zero": loc["zero_str"], "dev": d5, "l2_20": loc["l2_20"], "steps": loc["steps"],
                          "converged": loc["converged"], "pass": r5}
            log(f"R5 wf {wf}: gn_locate on the real series from x_ref: {loc['zero_str'][:30]}, off #702's zfit by "
                f"{d5:.1e} (pass <= {TOL_R5:.0e}); l2 at 20 terms {loc['l2_20']:.6e} (#702 "
                f"{REF702[wf]['zfit_l2_20']:.6e}); {len(loc['steps'])} steps, converged {loc['converged']} -> "
                f"{'OK' if r5 else 'BAD'} ({time.time() - t1:.0f}s)")
        log.dump(rec)
    k0, k0rec = te.check_ends(ends_all, log)
    ok["R2"] = k0
    rec["R2"] = k0rec
    rec["J+_parts"] = jplus
    rec["checks"] = ok
    if mut == "zero":
        fired = not ok["J-"] and not ok["R4z"]
    elif mut == "bzero":
        fired = not ok["R3z"] and not ok["J-"]
    elif mut == "judge":
        fired = not any(jplus.values())
    else:
        fired = None
    rec["mutation_fired"] = fired
    if mut:
        log(f"mutation {mut}: fired as frozen: {fired} (must)")
    gate = all(ok[k] for k in ("R1", "R2", "R3z", "R4z", "J+", "J-"))
    rec["controls_pass"] = gate and not mut
    rec["relocation_pass"] = rec["controls_pass"] and ok["R5"]
    log(f"controls {ok} -> gate {rec['controls_pass']}, relocation context {rec['relocation_pass']}")
    log.dump(rec)
    return rec["controls_pass"] and rec["relocation_pass"]


# ---------------------------------------------------------------- #684's comparison
def run_far(outdir, mut=None):
    ctl = json.load(open(os.path.join(outdir, "control.json")))
    if not ctl["controls_pass"]:
        print("the controls failed: no far stage (frozen rule)")
        return False
    log = tz.Log(outdir, "far", mut)
    real = tc.real_sig()
    rec = {"frozen": FROZEN, "mutate": mut, "nodes": {}}
    k1a = k1b = c1 = True
    s1 = s2 = s2b = s3 = True
    for wf in NODES:
        rel, _ = tc.err_model(wf)
        z = zloc(wf)
        ends = te.far_ends(wf)
        sig_r = real[wf][:N23]
        node = rec["nodes"].setdefault(wf, {})
        fit_k = zbasis(wf, "chord", te.kstar(wf), te.ex0(wf), DPS_FIT, z)
        chk_k = zbasis(wf, "chord", te.kstar(wf), te.ex0(wf), DPS_CHECK, z)
        # the real series in the far-end and exponent-shifted chord bases at ZLOC
        realrow = {}
        for v in ("kstar", "ratio", "pole8", "pole9", "pole10", "ex-0.3", "ex+0.3"):
            P1, ex, _ = te.spec(v, wf, ends)
            s = tz.scan(zbasis(wf, "chord", P1, ex, DPS_FIT, z), zbasis(wf, "chord", P1, ex, DPS_CHECK, z), sig_r, rel,
                        N23, log, f"wf {wf} real, {v} chord at ZLOC")
            realrow[v] = {"M_min": s["M_min"], "chi22": next(r["chi"] for r in s["rows"] if r["M"] == 22),
                          "C1_pass": s["C1_pass"]}
            c1 &= s["C1_pass"] or mut is not None
        node["real"] = realrow
        rk = realrow["kstar"]["chi22"]
        # K1z: #547's chord2 plant at R = 1.85 with its zero at ZLOC
        pl = rezero(F.FarPlant(wf, "chord2", real[wf], 1.85), z, real[wf])
        mp.mp.dps = pg.DPS_PLANT
        sig = pl.sigma()[:N23]
        own = F.far_point(wf, 1.86) if mut == "farend" else pl.P1
        r = tz.row(zbasis(wf, "chord", own, pl.ex, DPS_FIT, z, M=pl.M), sig, rel, N23, pl.M)
        a = r["chi"] <= tn.CHI_C0
        k1a &= a
        log(f"K1z(a) wf {wf}: #547's chord2 plant at R = 1.85, zero at ZLOC (M {pl.M}, fit {max(pl.resid):.1e}), in its "
            f"own-end chord basis at ZLOC{' (mutated)' if mut else ''}: chi {r['chi']:.1e} (pass <= 1e-30) "
            f"{'OK' if a else 'BAD'}")
        out = {"M_plant": pl.M, "fit_resid": max(pl.resid), "chi_own": r["chi"], "a_pass": a}
        if not mut:
            s = tz.scan(fit_k, chk_k, sig, rel, N23, log, f"K1z(b) wf {wf}: the 1.85 plant in zk/chord")
            chi22 = next(x["chi"] for x in s["rows"] if x["M"] == 22)
            b = s["M_min"] is not None and s["M_min"] >= te.M_K1
            k1b &= b
            c1 &= s["C1_pass"]
            out.update({"M_min": s["M_min"], "M_lo": s["M_lo"], "chi22": chi22, "ratio": chi22 / rk, "b_pass": b,
                        "C1_pass": s["C1_pass"], "chi_20_22": [x["chi"] for x in s["rows"] if x["M"] >= 20]})
            log(f"K1z(b) wf {wf}: M_min {s['M_min']} (pass >= {te.M_K1}) {'OK' if b else 'BAD'}; chi at 22 terms "
                f"{chi22:.2e}, real {rk:.1f}: ratio {chi22 / rk:.1e}")
            sens = {}
            for v in te.ENDS:
                P1, ex, _ = te.spec(v, wf, ends)
                pv = rezero(te.EndPlant(wf, "chord2", real[wf], P1, ex), z, real[wf])
                mp.mp.dps = pg.DPS_PLANT
                sv = tz.scan(fit_k, chk_k, pv.sigma()[:N23], rel, N23, log,
                             f"sensitivity wf {wf}: {v}'s chord2 plant at ZLOC in zk/chord")
                c22 = next(x["chi"] for x in sv["rows"] if x["M"] == 22)
                c1 &= sv["C1_pass"]
                sens[v] = {"M_min": sv["M_min"], "chi22": c22, "ratio": c22 / rk, "C1_pass": sv["C1_pass"]}
            out["sensitivity"] = sens
            node_s1 = out["ratio"] < S1_MAX and all(x["ratio"] < S1_MAX for x in sens.values())
            node_s2 = all(realrow[v]["chi22"] <= rk / S2_FACTOR for v in ("ratio", "pole10"))
            lr = abs(math.log(realrow["ratio"]["chi22"] / rk))
            le = max(abs(math.log(realrow[v]["chi22"] / rk)) for v in ("ex-0.3", "ex+0.3"))
            node_s2b = lr > le
            node_s3 = all(realrow[v]["chi22"] >= S3_FLOOR for v in ("kstar", "ratio", "pole8", "pole9", "pole10"))
            s1 &= node_s1
            s2 &= node_s2
            s2b &= node_s2b
            s3 &= node_s3
            node["statements"] = {"S1": node_s1, "S2": node_s2, "S2b": node_s2b, "S3": node_s3,
                                  "far_over_kstar": {v: realrow[v]["chi22"] / rk for v in realrow}}
            log(f"wf {wf}: real chi_22 at ZLOC, kstar {rk:.1f}; over kstar: " + ", ".join(
                f"{v} {realrow[v]['chi22'] / rk:.3f}" for v in realrow if v != "kstar")
                + f"; plant/real: K1z {out['ratio']:.1e}, " + ", ".join(f"{v} {x['ratio']:.1e}" for v, x in sens.items())
                + f" -> S1 {node_s1}, S2 {node_s2}, S2b {node_s2b}, S3 {node_s3}")
        node["K1z"] = out
        log.dump(rec)
    if mut == "farend":
        rec["mutation_fired"] = not k1a
        log(f"mutation farend: K1z(a) fails at every node: {not k1a} (must)")
        log.dump(rec)
        return False
    rec["verdict"] = {"K1z_a": k1a, "K1z_b": k1b, "C1": c1, "S1": s1, "S2": s2, "S2b": s2b, "S3": s3}
    log(f"K1z(a) {k1a}, K1z(b) {k1b}, C1 {c1}; statements: S1 (far end alone does not account for it) "
        f"{'survives' if s1 else 'FAILS'}; S2 (far end lowers chi_22 2.3x or more) {'survives' if s2 else 'FAILS'}; S2b "
        f"(far end matters more than the exponent) {'survives' if s2b else 'FAILS'}; S3 (above 3e4) "
        f"{'survives' if s3 else 'FAILS'}")
    log.dump(rec)
    return k1a and k1b and c1


# ---------------------------------------------------------------- grid stage
def grid_plants():
    near = [(k, c, a, e, p) for k in ss.KINDS for c in NEAR_CFGS for a in ss.ALPHAS for e in NEAR_ETAS
            for p in ss.PHASES]
    ids = {ss.plant_id(*g) for g in near}
    rest = [(k, c, a, e, p) for k in ss.KINDS for c in ss.CONFIGS for a in ss.ALPHAS for e in ss.ETAS
            for p in ss.PHASES if ss.plant_id(k, c, a, e, p) not in ids]
    return near, rest


def judge_reloc(j, rl, ref):
    """(c'') from a relocation record, or from the bound when the plant was not relocated"""
    if rl.get("skipped"):
        return False, None
    ratio = rl["chi"][22] / ref[JUDGED_SCAN][0]
    return rl["M_min"] == N23 and 1 / CHI_FACTOR <= ratio <= CHI_FACTOR, ratio


def run_grid(outdir):
    ctl = json.load(open(os.path.join(outdir, "control.json")))
    if not ctl["controls_pass"]:
        print("the controls failed: no grid run (frozen rule)")
        return False
    reloc_ok = ctl["relocation_pass"]
    log = tz.Log(outdir, "grid", None)
    path = os.path.join(outdir, "grid.jsonl")
    done = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done[(r["id"], r["wf"])] = r
    jl = open(path, "a")
    real = tc.real_sig()
    cache, parts, zfs = {}, {}, {}

    def node_setup(wf):
        if wf not in cache:
            projs, _ = projections(wf)
            ref, _ = real_ref(projs, wf)
            mains = {k: ZMain(wf, k, real[wf], zloc(wf)) for k in ss.KINDS}
            cache[wf] = (projs, ref, mains)
        return cache[wf]

    def run_one(wf, kind, cfg, a, eta, ph):
        pid = ss.plant_id(kind, cfg, a, eta, ph)
        if (pid, wf) in done:
            return done[(pid, wf)]
        projs, ref, mains = node_setup(wf)
        t0 = time.time()
        main = mains[kind]
        cur = ss.second_curve(wf, kind, cfg, main.base.arc)
        if (wf, kind, cfg, a) not in parts:
            with mp.workdps(pg.DPS_PLANT):
                al = ss.alpha_of(wf, a)
            parts[(wf, kind, cfg, a)] = ss.second_part(main, cur, al)
        sig, info = ss.make_plant(main, parts[(wf, kind, cfg, a)], eta, ph)
        r = evaluate(sig, wf, projs, ref)
        r.update({"id": pid, "wf": wf, "kind": kind, "cfg": cfg, "alpha": a, "eta": eta, "phase": ph,
                  "Q": complex(cur.P1), "plant": info})
        j = r["judge"]
        if reloc_ok:
            l2z = r["twin"][JUDGED_SCAN]["l2"][20]
            if l2z < 0.1 * ref[JUDGED_SCAN][0]:
                rl = {"skipped": True, "bound_ratio": l2z / ref[JUDGED_SCAN][0]}
            else:
                if wf not in zfs:
                    zfs[wf] = ZeroFit(wf)
                rl = relocated(wf, zfs[wf], sig[:N23], tc.err_model(wf)[0], zloc(wf))
            j["c2"], j["ratio2"] = judge_reloc(j, rl, ref)
            j["all2"] = j["a"] and j["b"] and j["c2"]
            if j["ratio2"] is not None:
                j["D2"] = max(j["m_DA"] / (NEED_DA - 1), j["d_P"] / TOL_POLE, abs(math.log10(j["ratio2"])))
            else:
                j["D2"] = max(j["m_DA"] / (NEED_DA - 1), j["d_P"] / TOL_POLE, abs(math.log10(rl["bound_ratio"])))
            r["reloc"] = rl
        r["seconds"] = time.time() - t0
        r = json.loads(json.dumps(pg.jsonable(r)))
        jl.write(json.dumps(r) + "\n")
        jl.flush()
        done[(pid, wf)] = r
        rl = r.get("reloc")
        rtxt = ""
        if rl is not None:
            rtxt = (f" | reloc skipped (l2_20 / real {rl['bound_ratio']:.1e})" if rl.get("skipped") else
                    f" | reloc dz {rl['dz_start']:.1e} Mmin {rl['M_min']} ratio {j['ratio2']:.2e} c'' {j['c2']!s:5s}")
        log(f"wf {wf} {pid:34s} M {info['M']} share22 {info['share'][22]:.1e} | m_DA {j['m_DA']:2d} d_P {j['d_P']:.3f} "
            f"Mmin {j['M_min']} ratio {j['ratio']:.2e} -> a {j['a']!s:5s} b {j['b']!s:5s} c' {j['c']!s:5s} D "
            f"{j['D']:.2f}{rtxt}{'' if j['C1_pass'] else ' C1 BAD'} ({r['seconds']:.1f}s)")
        return r

    near, rest = grid_plants()
    log(f"grid: part near {len(near)} plants, part rest {len(rest)}, at wf {WF0}; {len(done)} records already in "
        f"grid.jsonl; (c'') {'on' if reloc_ok else 'off (R5 failed)'}")
    for g in near:
        run_one(WF0, *g)
    log("part near done")
    for g in rest:
        run_one(WF0, *g)
    log("part rest done")
    allg = {ss.plant_id(*g): g for g in near + rest}
    at0 = {pid: done[(pid, WF0)]["judge"] for pid in allg}
    k2 = "c2" if reloc_ok else "c"
    passing = [p for p, j in at0.items() if j["all"] or j.get("all2")]
    ac = sorted([p for p, j in at0.items() if j["a"] and j["c"]], key=lambda p: at0[p]["D"])[:N_AC_MAX]
    ac2 = sorted([p for p, j in at0.items() if j["a"] and j.get("c2")], key=lambda p: at0[p]["D2"])[:N_AC_MAX]
    closest0 = sorted(at0, key=lambda p: at0[p]["D"])[:N_CLOSEST]
    more = list(dict.fromkeys(passing + ac + ac2 + closest0))
    log(f"wf {WF0}: {len(passing)} pass (a), (b) and (c') or (c''); {len(ac)} pass (a) and (c'), {len(ac2)} (a) and "
        f"(c'') (cap {N_AC_MAX} each); the other nodes for {len(more)} plants")
    for pid in more:
        for wf in NODES:
            run_one(wf, *allg[pid])
    # verdicts
    full = {pid: {wf: done[(pid, wf)]["judge"] for wf in NODES} for pid in more}

    def every(p, f):
        return all(f(full[p][wf]) for wf in NODES)
    out3 = {}
    for lab, cc, aa in (("c'", "c", "all"), ("c''", k2, "all2" if reloc_ok else "all")):
        out3[lab] = {"c": [p for p in full if every(p, lambda j: j[cc])],
                     "ac": [p for p in full if every(p, lambda j: j["a"] and j[cc])],
                     "reproduces": [p for p in full if every(p, lambda j: j[aa])]}
    dall = {p: max(full[p][wf]["D"] for wf in NODES) for p in full}
    closest = min(dall, key=dall.get)
    c1_fail = [k for k, r in done.items() if not r["judge"]["C1_pass"]]
    counts = {}
    for lab, cc, aa in (("c'", "c", "all"), ("c''", k2, "all2" if reloc_ok else "all")):
        counts[lab] = {"c": sum(1 for j in at0.values() if j[cc]),
                       "ac": sum(1 for j in at0.values() if j["a"] and j[cc]),
                       "bc": sum(1 for j in at0.values() if j["b"] and j[cc]),
                       "all": sum(1 for j in at0.values() if j[aa])}
    counts.update({"a": sum(1 for j in at0.values() if j["a"]), "b": sum(1 for j in at0.values() if j["b"]),
                   "ab": sum(1 for j in at0.values() if j["a"] and j["b"]),
                   "c667_ZLOC": sum(1 for j in at0.values() if j["c667_ZLOC"])})
    near667 = [p for p in allg if allg[p][1] in NEAR_CFGS]
    s4 = {}
    for lab, rk in (("c'", "ratio"), ("c''", "ratio2")):
        vals = {p: at0[p][rk] if at0[p].get(rk) is not None else
                done[(p, WF0)].get("reloc", {}).get("bound_ratio", 0.0) for p in near667}
        arg = max(vals, key=vals.get)
        s4[lab] = {"survives": vals[arg] <= S4_MAX, "max_ratio": vals[arg], "at": arg}
    s5 = {}
    for lab, cc in (("c'", "c"), ("c''", k2)):
        cp = [p for p, j in at0.items() if j[cc]]
        s5[lab] = {"survives": bool(cp) and all(allg[p][1] == "low" for p in cp),
                   "curves": sorted({allg[p][1] for p in cp})}
    rl_change = [f"{p} wf {wf}" for (p, wf), r in done.items() if "c2" in r["judge"] and r["judge"]["c2"] != r["judge"]["c"]]
    by = {}
    for p, j in at0.items():
        k, c, a, e, ph = allg[p]
        b = by.setdefault(f"{'near' if c in NEAR_CFGS else 'low'}|{e:.0e}",
                          {"n": 0, "a": 0, "b": 0, "c": 0, "ac": 0, "c2": 0, "ac2": 0, "ratio_min": math.inf,
                           "ratio_max": 0.0, "ratio2_max": 0.0, "d_P_min": math.inf})
        b["n"] += 1
        b["a"] += j["a"]
        b["b"] += j["b"]
        b["c"] += j["c"]
        b["ac"] += j["a"] and j["c"]
        b["c2"] += bool(j.get("c2"))
        b["ac2"] += j["a"] and bool(j.get("c2"))
        b["ratio_min"] = min(b["ratio_min"], j["ratio"])
        b["ratio_max"] = max(b["ratio_max"], j["ratio"])
        b["ratio2_max"] = max(b["ratio2_max"], j.get("ratio2") or 0.0)
        b["d_P_min"] = min(b["d_P_min"], j["d_P"])
    verdict = {"all_nodes": out3, "closest": closest, "closest_D_all": dall[closest],
               "closest_per_node": {wf: full[closest][wf] for wf in NODES}, "counts_wf1": counts, "n_wf1": len(at0),
               "n_three_nodes": len(full), "by_set_eta": by, "C1_fail": c1_fail, "S4": s4, "S5": s5,
               "reloc_changes_c": rl_change, "c2_judged": reloc_ok}
    for key, b in sorted(by.items()):
        log(f"{key:11s}: n {b['n']:3d}, (a) {b['a']:3d}, (b) {b['b']}, (c') {b['c']:3d}, (a)+(c') {b['ac']:3d}, (c'') "
            f"{b['c2']:3d}, (a)+(c'') {b['ac2']:3d}; ratio {b['ratio_min']:.1e}..{b['ratio_max']:.1e}, relocated up to "
            f"{b['ratio2_max']:.1e}; smallest d_P {b['d_P_min']:.3f}")
    for lab in ("c'", "c''"):
        c = counts[lab]
        log(f"wf {WF0}, with ({lab}): of {len(at0)}: ({lab}) {c['c']}, (a)+({lab}) {c['ac']}, (b)+({lab}) {c['bc']}, all "
            f"three {c['all']}; at all three nodes, of {len(full)} run there: ({lab}) {len(out3[lab]['c'])}, (a)+({lab}) "
            f"{len(out3[lab]['ac'])}, all three {len(out3[lab]['reproduces'])}")
    log(f"wf {WF0}: (a) {counts['a']}, (b) {counts['b']}, (a)+(b) {counts['ab']}; #667's (c) on 15 bases at ZLOC "
        f"{counts['c667_ZLOC']}")
    rp1, rp2 = out3["c'"]["reproduces"], out3["c''"]["reproduces"]
    log(f"plants that reproduce the real series: with (c') {rp1 or 'none'}; with (c'') {rp2 or 'none'}")
    log(f"closest by D: {closest}, D_all {dall[closest]:.2f}; " + "; ".join(
        f"wf {wf}: m_DA {j['m_DA']}, d_P {j['d_P']:.3f}, ratio {j['ratio']:.2f}"
        for wf, j in verdict["closest_per_node"].items()))
    for lab in ("c'", "c''"):
        log(f"S4 with ({lab}) (#667: a set near P0 + kappa_* misfits at least 290 times less): largest ratio "
            f"{s4[lab]['max_ratio']:.3g} at {s4[lab]['at']} -> {'survives' if s4[lab]['survives'] else 'FAILS'}")
        log(f"S5 with ({lab}) (#667: only the low set reaches the real misfit): passes for curves {s5[lab]['curves']} -> "
            f"{'survives' if s5[lab]['survives'] else 'FAILS'}")
    log(f"(c') and (c'') differ on {len(rl_change)} records; C1 failures: {c1_fail or 'none'}")
    log.dump({"frozen": FROZEN, "verdict": verdict})
    return not c1_fail


def main():
    argv = sys.argv[1:]
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    if "--selftest" in argv:
        if mut not in (None, "gn"):
            raise SystemExit(f"unknown selftest mutation {mut}")
        sys.exit(0 if selftest(mut) else 1)
    argv = [a for a in argv if not a.startswith("--")]
    if len(argv) != 2 or argv[0] not in ("control", "far", "grid"):
        raise SystemExit(__doc__)
    stage, outdir = argv
    if mut not in {"control": (None, "zero", "bzero", "judge"), "far": (None, "farend"), "grid": (None,)}[stage]:
        raise SystemExit(f"unknown {stage} mutation {mut}")
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    os.makedirs(outdir, exist_ok=True)
    if stage == "control":
        ok = run_control(outdir, mut)
        if mut:
            rec = json.load(open(os.path.join(outdir, f"control_mutate_{mut}.json")))
            print(f"mutation {mut} fired as frozen: {rec['mutation_fired']}; exit 1")
            ok = False
    elif stage == "far":
        ok = run_far(outdir, mut)
    else:
        ok = run_grid(outdir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
