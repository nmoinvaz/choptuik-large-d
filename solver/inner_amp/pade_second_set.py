#!/usr/bin/env python3
"""Plants with a weak second singular set past S_1's arc: do they reproduce the real series (issue #667)?

On the real sigma_0..sigma_22, (a) no 6 of #547's 11 DA2 members agree on a zero past |t| = 1.65 (pade_farend_diag.py),
(b) the [k/k] Pade approximants, k = 8, 9, 10, have a pole past kappa_* (#541, REF541), and (c) no chord, stepped-arc or
smooth-arc twin with the far end at kappa_*, the ratio-law kappa or the k = 8..10 poles has fewer than 23 terms, with
chi 3.4e4..1.3e5 at 22 terms (#672, #684). Every single-curve plant behaves otherwise. Here the plants are a single
curve (chord2 or arc, pade_farend.FarPlant) plus a weak second set past it, and each is judged on (a), (b) and (c).

Plant. S(x) = pref(x) (L_main(x) + lam L_2(x)) / S_0, pref the planted zero factor. L_main is FarPlant's density
(1-s)^(beta-1) sum_m c_m T_m(2s-1) on the chord or the arc P0 -> kappa_*. L_2(x) = int_0^1 s^P (1-s)^alpha
exp(x g2(s)) g2'(s) ds on a second curve g2, so the second set's far end g2(1) is a branch point with exponent alpha and
its near end g2(0) a weak one (exponent P = 4). Its strength is fixed by its share of sigma_22: lam is set so that
(pref L_2)_22 lam = eta e^(i phi) sigma_22(real). Then c_m are fitted to real sigma_0..sigma_12 minus the second set's
part, as FarPlant (M = 6, or 8 if the largest relative residual exceeds 1e-2), so each plant matches sigma_0..sigma_12.
At eta = 0 it is FarPlant's plant. The second set's moments are exact (a polynomial in u = 1 - s times u^alpha).

Second curves (per node, from the node's P0 and kappa_*):
  seg(a, b)  the segment from Q - 0.3 Q/|Q| to Q, Q = P0 + kappa_* + 0.25 (a + i b), a, b in {-1, 0, 1}
  low        the same segment with Q = 1.69 - 0.74i, the mean of the five real DA2 zeros at 1.53..1.95 - 0.65..0.85i
             at wf 1.00 (#547)
  sum        P0 + the main curve, from 2 P0 to P0 + kappa_* (the sum set the issue names)

Judge, per node. (a) m_DA, the largest number of DA2 members with a zero in 1.65 < |t| < 6 within 0.05 of one member's
zero (pade_farend_diag.clusters' rule): pass if m_DA <= 5. (b) d_P = max over k = 8, 9, 10 of the distance from #541's
pole past kappa_* to the plant's nearest [k/k] pole (pade_geometry.pade at 120 digits): pass if d_P <= 0.03. (c) the
twin test of pade_twin_end.py on the 15 bases kstar, ratio, pole8, pole9, pole10 x chord, arc, smooth, with the far ends
the real series' own (te.far_ends), so the bases are #684's: pass if every scan has M_min = 23 and chi at 22 terms within
a factor 10 of the real series' in the same basis; d_T = max |log10(chi_22 / chi_22(real))|. The errors are the real
series' absolute errors, err_k = rel_k |sigma_k(real)|, the same for every plant. The least-squares residuals of all
M come from one Householder QR per basis (Proj), at 150 digits and, for C1, at 250.

    python3 pade_second_set.py --selftest [--mutate=secmom|nest|agree]
    python3 pade_second_set.py control OUTDIR [--mutate=col|judge] [--allow_unraw]
    python3 pade_second_set.py grid OUTDIR [--allow_unraw]              (reads OUTDIR/control.json)
    python3 pade_second_set.py report OUTDIR [--allow_unraw]   (post-freeze summary of OUTDIR/grid.jsonl, not judged)

Exit 1 when a frozen control fails or the gate forbids the grid. Numerical evidence on plants, not a bound.
"""
import cmath
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pade_twin_end as te                                             # noqa: E402  (sets pg.KM = 31)
import pade_farend_diag as FD                                          # noqa: E402

tz, tc, F, pg, pp = te.tz, te.tc, te.F, te.pg, te.pp
N23 = te.N23
NODES = pg.NODES
WF0 = 1.0
DPS_FIT, DPS_CHECK = tz.DPS_FIT, tz.DPS_CHECK
VARIANTS = ("kstar", "ratio", "pole8", "pole9", "pole10")
CLASSES = te.CLASSES                # (twin class, its own plant kind)
SCANS = tuple(f"{v}/{c}" for v in VARIANTS for c, _ in CLASSES)
REF541 = te.REF541

# ---------------------------------------------------------------- the frozen grid
KINDS = ("chord2", "arc")
SEG_STEP, SEG_LEN, P_INNER = 0.25, 0.3, 4
Q_LOW = 1.69 - 0.74j
CONFIGS = tuple(f"seg{a:+d}{b:+d}" for a in (-1, 0, 1) for b in (-1, 0, 1)) + ("low", "sum")
ALPHAS = ("log", "ex")              # alpha = 0, or beta - 1 (the main far end's exponent)
ETAS = tuple(10.0 ** -e for e in range(9, 0, -1))
PHASES = (0, 90, 180, 270)          # degrees
RAD_DA, NEED_DA, RMIN, RMAX = 0.05, 6, FD.RMIN, FD.RMAX       # pade_farend_diag.clusters' radius and window
TOL_POLE = 0.03
CHI_FACTOR = 10.0
N_CLOSEST = 20
NEAR541 = 0.17                      # context: DA2 zeros within this of #541's far poles (the real series has none)
# I684-mac/real.json (raw-data repository, d444df1): chi at 22 terms of the real twins, M_min = 23 in every scan
REF684 = {
    0.99: {"kstar/chord": 96978.58249, "kstar/arc": 124166.6643, "kstar/smooth": 126775.4006, "ratio/chord": 34642.72998,
           "ratio/arc": 48227.82219, "ratio/smooth": 50425.10782, "pole8/chord": 62800.11972,
           "pole8/arc": 84028.47271, "pole8/smooth": 87401.41651, "pole9/chord": 47737.50549, "pole9/arc": 64822.1451,
           "pole9/smooth": 67994.86128, "pole10/chord": 35598.54121, "pole10/arc": 49303.04491,
           "pole10/smooth": 51950.13615},
    1.0: {"kstar/chord": 91162.72005, "kstar/arc": 115299.6611, "kstar/smooth": 117634.6851, "ratio/chord": 34290.18442,
          "ratio/arc": 47233.34276, "ratio/smooth": 49255.52324, "pole8/chord": 61748.31146, "pole8/arc": 82238.17247,
          "pole8/smooth": 85063.407, "pole9/chord": 46420.47231, "pole9/arc": 62694.89689, "pole9/smooth": 65566.93733,
          "pole10/chord": 35276.98216, "pole10/arc": 48239.59289, "pole10/smooth": 50996.39847},
    1.01: {"kstar/chord": 86025.38344, "kstar/arc": 106438.082, "kstar/smooth": 108482.8184, "ratio/chord": 34827.52253,
           "ratio/arc": 47115.17176, "ratio/smooth": 48824.21206, "pole8/chord": 61617.79517,
           "pole8/arc": 80945.69521, "pole8/smooth": 83183.98749, "pole9/chord": 46242.92687, "pole9/arc": 61869.02597,
           "pole9/smooth": 64171.422, "pole10/chord": 35478.7731, "pole10/arc": 48049.64067,
           "pole10/smooth": 50389.44846}}
FROZEN = {                          # written before the control and grid stages were first run (notes/PADE_FAR_END.md 12.2)
    "plants": "pref (L_main + lam L_2) / S_0; L_main is FarPlant's chord2 or arc density at kappa_*; L_2 the density "
              "s^4 (1-s)^alpha on a second curve; lam set so that the second set's part of sigma_22 is eta e^(i phi) "
              "sigma_22(real); c_m fitted to real sigma_0..sigma_12 minus the second set's part (M = 6, or 8 if the "
              "largest relative residual exceeds 1e-2)",
    "grid": "kind chord2, arc; curve seg(a, b) (Q = P0 + kappa_* + 0.25 (a + ib), a, b in {-1, 0, 1}, segment of "
            "length 0.3 radially inward ending at Q), low (Q = 1.69 - 0.74i, same segment), sum (P0 + the main curve); "
            "alpha = 0 or beta - 1; eta = 1e-9, 1e-8, ..., 1e-1; phi = 0, 90, 180, 270 degrees: 1584 plants",
    "nodes": "every plant at wf 1.00; wf 0.99 and 1.01 for every plant that passes (a), (b) and (c) at wf 1.00 and for "
             "the 20 plants with the smallest D at wf 1.00",
    "a_DA2": "m_DA = the largest number of the 11 DA2 members (pade_farend.members) with a zero t of Q2, 1.65 < |t| < 6, "
             "within 0.05 of one member's zero (pade_farend_diag's cluster rule); pass if m_DA <= 5",
    "b_pade": "d_P = max over k = 8, 9, 10 of the distance from REF541[wf][k] to the plant's nearest [k/k] pole "
              "(pade_geometry.pade, 120 digits); pass if d_P <= 0.03",
    "c_twin": "15 scans per node: variants kstar, ratio, pole8, pole9, pole10 (far ends from the real series, "
              "pade_twin_end.far_ends) x classes chord, arc, smooth (pade_twin_end.end_basis), n = 23, M = 6..23; "
              "err_k = rel_k |sigma_k(real)| for every plant; pass if every scan has M_min = 23 and "
              "0.1 <= chi_22 / chi_22(real) <= 10; d_T = max over scans of |log10(chi_22 / chi_22(real))|",
    "reproduces": "a plant reproduces the real series only if (a), (b) and (c) pass at all three nodes",
    "closest": "D = max(m_DA / 5, d_P / 0.03, d_T) per node, D_all = its max over the nodes evaluated; the closest "
               "plant is the one with the smallest D_all among those evaluated at all three nodes",
    "C1": "#672's amended C1 on every plant scan (150 against 250 digits at M_min - 1 and M_min; at M = n, chi <= 1e-30 "
          "at both); a failure is listed and the grid run exits 1",
    "R_controls": "R1: the real series' m_DA <= 5 at every node; R2: the real [k/k] poles past kappa_* within 1e-9 of "
                  "REF541 and the ratio law within 1e-4 (pade_twin_end.check_ends); R3: the fast twin path gives "
                  "M_min = M_lo = 23 and chi_22 within 1e-6 relative of REF684 in all 45 real scans; R4: at eta = 0 "
                  "the plant's sigma_0..sigma_31 equal FarPlant's exactly, for chord2 and arc at every node",
    "J_controls": "J+: the real series judged as a plant passes (a), (b), (c) at every node; J-: the eta = 0 chord2 "
                  "and arc plants fail the judge at wf 1.00, with M_min = 6 in their own class's kstar basis",
    "gate": "the grid runs only if R1-R4, J+ and J- pass",
    "mutations": "control --mutate=col (every basis' column 0 at order 20 x (1 + 1e-8)) must make R3 fail; "
                 "control --mutate=judge (REF541 + 0.05, the reference chi_22 x 100, the DA2 radius 0.5) must make "
                 "each of J+'s three parts fail; each exits 1",
    "context": "per plant: the frozen DA2 median (pade_farend.estimate's rule), the number of DA2 zeros within 0.17 of "
               "#541's far poles, the ratio law over k = 12..21, the second set's share of sigma_12, sigma_17, "
               "sigma_22, the largest |sigma_k / sigma_k(real) - 1| over k = 13..22, chi at 20 and 21 terms, M_lo, "
               "and the main fit's M and residual; reported, not judged",
}


# ---------------------------------------------------------------- the second set
def sec_moments(curve, alpha, p, K, mut=None):
    """mom[k] = 1/k! int_0^1 s^p (1-s)^alpha g(s)^k g'(s) ds, k < K, exactly at the working precision: in u = 1 - s,
    g and g' are polynomials (pade_twin_size.arc_poly_u), s^p = (1-u)^p, and int_0^1 u^(alpha+j) du = 1/(alpha+j+1)"""
    g, dg = tz.arc_poly_u(curve)
    a = mp.mpc(alpha) + (mp.mpf("1e-10") if mut == "secmom" else 0)
    dens = [mp.mpf((-1) ** j * math.comb(p, j)) for j in range(p + 1)]
    base = tz.pmul(dg, dens)
    out, gk = [], [mp.mpc(1)]
    for k in range(K):
        poly = tz.pmul(gk, base)
        out.append(mp.fsum(c / (a + j + 1) for j, c in enumerate(poly)) / mp.factorial(k))
        gk = tz.pmul(gk, g)
    return out


def sec_moments_quad(curve, alpha, p, k):
    """the same moment by mpmath.quad of the integral written in s (independent of arc_poly_u)"""
    P0, P1 = mp.mpc(curve.P0), mp.mpc(curve.P1)
    e0, e1 = (mp.mpc(0), mp.mpc(0)) if curve.kind == "chord" else (mp.mpc(curve.e0), mp.mpc(curve.e1))

    def g(s):
        return P0 + (P1 - P0) * s + s * (1 - s) * (e0 + e1 * s)

    def dg(s):
        return (P1 - P0) + (1 - 2 * s) * (e0 + e1 * s) + s * (1 - s) * e1
    return mp.quad(lambda s: s ** p * (1 - s) ** mp.mpc(alpha) * g(s) ** k * dg(s), [0, 0.5, 1]) / mp.factorial(k)


def second_curve(wf, kind, cfg, main):
    """the second curve of configuration cfg at node wf; main is the main plant's FarArc"""
    with mp.workdps(pg.DPS_PLANT):
        P0, ks = mp.mpc(main.P0), mp.mpc(main.P1)
        if cfg == "sum":
            return SimpleNamespace(P0=2 * P0, P1=P0 + ks, e0=main.e0, e1=main.e1,
                                   kind="chord" if kind == "chord2" else "arc")
        if cfg == "low":
            Q = mp.mpc(Q_LOW)
        else:
            a, b = int(cfg[3:5]), int(cfg[5:7])
            Q = P0 + ks + mp.mpf(SEG_STEP) * mp.mpc(a, b)
        return SimpleNamespace(P0=Q - mp.mpf(SEG_LEN) * Q / abs(Q), P1=Q, e0=mp.mpc(0), e1=mp.mpc(0), kind="chord")


def alpha_of(wf, a):
    return mp.mpc(0) if a == "log" else te.ex0(wf)


class Main:
    """FarPlant's main density at one node, with its moments for M = 8 kept for the refits"""

    def __init__(self, wf, kind, real_sig):
        self.wf, self.kind, self.real = wf, kind, real_sig
        self.base = F.FarPlant(wf, kind, real_sig)
        mp.mp.dps = pg.DPS_PLANT
        self.mom8 = self.base.moments(8)
        self.cols = [pp.polymul(self.base.pref, self.mom8[m], pg.KM + 1) for m in range(8)]


def second_part(main, curve, alpha, p=P_INNER, mut=None):
    """(pref L_2)_k, k <= KM, unscaled"""
    mp.mp.dps = pg.DPS_PLANT
    return pp.polymul(main.base.pref, sec_moments(curve, alpha, p, pg.KM + 1, mut), pg.KM + 1)


def make_plant(main, w, eta, phase, K=12):
    """the plant's sigma_0..sigma_KM and its record, with w = second_part(...); eta = 0 is FarPlant's own fit
    (pade_geometry.Plant.fit)"""
    mp.mp.dps = pg.DPS_PLANT
    real = main.real
    KM = pg.KM
    if eta:
        lam = mp.mpf(eta) * mp.expjpi(mp.mpf(phase) / 180) * real[22] / w[22]
        s2 = [lam * z for z in w]
    else:
        s2 = [mp.mpc(0)] * (KM + 1)
    for M in (6, 8):
        A = np.array([[complex(main.cols[m][k] / abs(real[k])) for m in range(M)] for k in range(K + 1)])
        b = np.array([complex((real[k] - s2[k]) / abs(real[k])) for k in range(K + 1)])
        Ar = np.block([[A.real, -A.imag], [A.imag, A.real]])
        sol, *_ = np.linalg.lstsq(Ar, np.concatenate([b.real, b.imag]), rcond=None)
        c = [mp.mpc(complex(z)) for z in sol[:M] + 1j * sol[M:]]
        lc = [sum(c[m] * main.mom8[m][k] for m in range(M)) for k in range(KM + 1)]
        s = pp.polymul(main.base.pref, lc, KM + 1)
        s = [x + y for x, y in zip(s, s2)]
        sig = [z / s[0] for z in s]
        res = [float(abs(sig[k] - real[k]) / abs(real[k])) for k in range(K + 1)]
        if max(res) <= 1e-2:
            break
    share = {k: float(abs(s2[k]) / abs(s[k])) for k in (12, 17, 22)}
    dev = max(float(abs(sig[k] / real[k] - 1)) for k in range(13, N23))
    return sig, {"M": M, "fit_resid": max(res), "share": share, "dev_13_22": dev, "lam_abs": float(abs(s2[22]))}


# ---------------------------------------------------------------- (a) DA2
def da2(sig):
    """each member's zeros of Q2 in RMIN < |t| < RMAX (pade_farend_diag.zeros' window) and its farthest zero in
    pade_farend's frozen window; one da_zeros call per member"""
    mp.mp.dps = F.DPS_DA
    s = [mp.mpc(z) for z in sig[:N23]]
    mem = []
    for degs, lp in F.members(2):
        err = None
        try:
            ts = F.da_zeros(s, degs, lp)
        except (ZeroDivisionError, ValueError, mp.libmp.NoConvergence) as e:
            ts, err = None, type(e).__name__
        mem.append({"degs": degs, "lp": lp, "error": err,
                    "t": sorted((complex(t) for t in ts or [] if RMIN < abs(t) < RMAX), key=abs),
                    "far": F.farthest(ts) if ts else None})
    return mem


def max_agree(mem, rad=RAD_DA):
    """the largest number of distinct members with a zero within rad of one member's zero, and that zero (the rule of
    pade_farend_diag.clusters: a cluster of `need` members exists exactly when this is >= need)"""
    pts = [(i, t) for i, m in enumerate(mem) for t in m["t"]]
    best, at = 0, None
    for _, t0 in pts:
        n = len({i for i, t in pts if abs(t - t0) < rad})
        if n > best:
            best, at = n, t0
    return best, at


def da2_record(sig, wf, rad=RAD_DA):
    mem = da2(sig)
    m, at = max_agree(mem, rad)
    fs = np.array([x["far"] for x in mem if x["far"] is not None])
    med = complex(np.median(fs.real), np.median(fs.imag)) if len(fs) else None
    near = sum(1 for x in mem for t in x["t"] if min(abs(t - REF541[wf][k]) for k in (8, 9, 10)) < NEAR541)
    return {"m_DA": m, "at": at, "median": med, "near541": near, "members": mem}


# ---------------------------------------------------------------- (b) the far Pade poles
def pade_record(sig, wf, ref=None):
    ref = ref or REF541[wf]
    out = {}
    with mp.workdps(pg.DPS_PADE):
        s = [mp.mpc(z) for z in sig[:N23]]
        for k in (8, 9, 10):
            try:
                ts = [complex(p["t"]) for p in pg.pade(s, k)]
            except (ZeroDivisionError, ValueError, mp.libmp.NoConvergence) as e:
                out[k] = {"dist": math.inf, "nearest": None, "error": type(e).__name__}
                continue
            t = min(ts, key=lambda z: abs(z - ref[k]))
            out[k] = {"dist": abs(t - ref[k]), "nearest": t}
    return out


# ---------------------------------------------------------------- (c) the twin test by projection
class Proj:
    """the least-squares residuals of a twin basis for every M from one Householder QR of the weighted real form. The
    columns are ordered Re c_0, Im c_0, Re c_1, ..., so Q's first 2M columns span the M-term twin, and the residual is
    (I - Q_2M Q_2M^T) W sigma. The weights are fixed (err_k), so one QR serves every target"""

    def __init__(self, basis, err, n=N23, mut=None):
        self.dps, self.n = basis.dps, n
        with mp.workdps(self.dps):
            self.err = [mp.mpf(e) for e in err]
            A = mp.matrix(2 * n, 2 * n)
            for k in range(n):
                for m in range(n):
                    z = basis.cols[m][k] / self.err[k]
                    i, j = (m, n + m) if mut == "nest" else (2 * m, 2 * m + 1)
                    A[k, i], A[k, j], A[n + k, i], A[n + k, j] = z.real, -z.imag, z.imag, z.real
            self.Q, _ = mp.qr(A)

    def rows(self, sig, Ms=range(tz.M_CONTROL, N23 + 1)):
        """{M: (chi, l2)}"""
        n, Q, out = self.n, self.Q, {}
        with mp.workdps(self.dps):
            v = mp.matrix(2 * n, 1)
            for k in range(n):
                z = mp.mpc(sig[k]) / self.err[k]
                v[k], v[n + k] = z.real, z.imag
            y = Q.T * v
            cur = v.copy()
            for j in range(2 * n):
                yj = y[j]
                for i in range(2 * n):
                    cur[i] -= Q[i, j] * yj
                M = (j + 1) // 2
                if j % 2 == 1 and M in Ms:
                    r = [mp.sqrt(cur[k] ** 2 + cur[n + k] ** 2) for k in range(n)]
                    out[M] = (float(max(r)), float(mp.sqrt(mp.fsum(x * x for x in r))))
        return out


def projections(wf, mut=None):
    """{scan: (Proj at 150 digits, Proj at 250)} for the 15 bases at node wf, with the real series' errors"""
    real = tc.real_sig()[wf][:N23]
    rel, _ = tc.err_model(wf)
    ends = te.far_ends(wf)
    out = {}
    for v in VARIANTS:
        P1, ex, _ = te.spec(v, wf, ends)
        for cls, _ in CLASSES:
            pair = []
            for dps in (DPS_FIT, DPS_CHECK):
                b = te.end_basis(wf, cls, P1, ex, dps)
                if mut == "col":
                    with mp.workdps(dps):
                        b.cols[0][20] *= 1 + mp.mpf("1e-8")
                pair.append(Proj(b, tc.weights(real, rel, N23, 1.0, dps), mut=None))
            out[f"{v}/{cls}"] = tuple(pair)
    return out, ends


def twin_record(projs, sig):
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
                    "l2_22": rows[22][1], "C1_pass": bool(prec) and all(p["pass"] for p in prec), "precision": prec}
    return out


# ---------------------------------------------------------------- the judge
def judge(rec, ref_chi):
    a = rec["da2"]["m_DA"] <= NEED_DA - 1
    d_P = max(rec["pade"][k]["dist"] for k in (8, 9, 10))
    ratios = {s: rec["twin"][s]["chi"][22] / ref_chi[s] for s in SCANS}
    d_T = max(abs(math.log10(r)) if r > 0 else math.inf for r in ratios.values())
    mmin = all(rec["twin"][s]["M_min"] == N23 for s in SCANS)
    c = mmin and all(1 / CHI_FACTOR <= r <= CHI_FACTOR for r in ratios.values())
    D = max(rec["da2"]["m_DA"] / (NEED_DA - 1), d_P / TOL_POLE, d_T)
    return {"a": a, "b": d_P <= TOL_POLE, "c": c, "all": a and d_P <= TOL_POLE and c, "m_DA": rec["da2"]["m_DA"],
            "d_P": d_P, "d_T": d_T, "M_min_23": mmin, "ratio_min": min(ratios.values()),
            "ratio_max": max(ratios.values()), "D": D, "C1_pass": all(rec["twin"][s]["C1_pass"] for s in SCANS)}


def evaluate(sig, wf, projs, ref_chi, rad=RAD_DA, ref541=None):
    rec = {"da2": da2_record(sig, wf, rad), "pade": pade_record(sig, wf, ref541), "twin": twin_record(projs, sig)}
    with mp.workdps(pg.DPS_PADE):
        rec["ratio12_21"] = complex(pp.ratio_law([mp.mpc(z) for z in sig[:N23]], 12, 21)[0])
    rec["judge"] = judge(rec, ref_chi)
    return rec


def real_chi(projs, wf):
    """the real series' chi at 22 terms per scan, by the same projections"""
    tw = twin_record(projs, tc.real_sig()[wf][:N23])
    return {s: tw[s]["chi"][22] for s in SCANS}, tw


# ---------------------------------------------------------------- selftest
def selftest(mut=None):
    ok = True
    wf = WF0
    real = tc.real_sig()
    # 1: the second set's exact moments against mpmath.quad
    main = Main(wf, "arc", real[wf])
    for cfg, a in (("seg+0+0", "ex"), ("sum", "log"), ("low", "ex")):
        cur = second_curve(wf, "arc", cfg, main.base.arc)
        with mp.workdps(pg.DPS_PLANT):
            al = alpha_of(wf, a)
            ex = sec_moments(cur, al, P_INNER, 32, mut)
            e = max(abs(ex[k] - sec_moments_quad(cur, al, P_INNER, k)) / abs(ex[k]) for k in (0, 7, 15, 22, 31))
        print(f"selftest 1: {cfg} ({cur.kind}, g(1) = {complex(cur.P1):.4f}), alpha {a}: exact moments against "
              f"mpmath.quad, k = 0, 7, 15, 22, 31: max rel diff {float(e):.1e} (pass < 1e-40)")
        ok &= e < mp.mpf("1e-40")
    # 2: the projection's residuals against pade_twin_count.lsq, the scans' own fit
    rel, _ = tc.err_model(wf)
    ends = te.far_ends(wf)
    for v, cls in (("kstar", "chord"), ("pole10", "arc")):
        P1, ex, _ = te.spec(v, wf, ends)
        b = te.end_basis(wf, cls, P1, ex, DPS_FIT)
        err = tc.weights(real[wf][:N23], rel, N23, 1.0, DPS_FIT)
        pr = Proj(b, err, mut=mut).rows(real[wf][:N23])
        worst = 0.0
        for M in (6, 14, 20, 22):
            _, chi, l2 = tc.lsq(b, real[wf][:N23], err, N23, M)
            worst = max(worst, abs(pr[M][0] / chi - 1), abs(pr[M][1] / l2 - 1))
        print(f"selftest 2: {v} {cls} basis, real series at wf {wf}: projected chi and l2 against tc.lsq at M = 6, 14, "
              f"20, 22: max rel diff {worst:.1e} (pass < 1e-10)")
        ok &= worst < 1e-10
    # 3: the agreement count on planted zero sets
    rad = 0.005 if mut == "agree" else RAD_DA
    for n_close in (6, 5):
        mem = [{"t": [2.0 + 1.2j + 0.008 * cmath.exp(1j * i)]} for i in range(n_close)]
        mem += [{"t": [1.7 + 0.2j * i + 0.3]} for i in range(11 - n_close)]
        m, _ = max_agree(mem, rad)
        cl = FD.clusters(mem, rad=rad, need=NEED_DA)
        good = m == n_close and (len(cl) > 0) == (n_close >= NEED_DA)
        print(f"selftest 3: {n_close} planted members within 0.016 of 2.0+1.2i: m_DA {m} (pass = {n_close}), "
              f"clusters of 6 {len(cl)} (agree: {good})")
        ok &= good
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- control stage
def run_control(outdir, mut=None):
    log = tz.Log(outdir, "control", mut)
    rec = {"frozen": FROZEN, "mutate": mut, "nodes": {}}
    real = tc.real_sig()
    ok = {"R1": True, "R2": True, "R3": True, "R4": True, "J+": True, "J-": True}
    jplus_parts = {"a": True, "b": True, "c": True}
    ends_all = {}
    for wf in NODES:
        t0 = time.time()
        projs, ends = projections(wf, mut if mut == "col" else None)
        ends_all[wf] = ends
        node = rec["nodes"].setdefault(wf, {})
        # R3: the fast twin path on the real series against #684
        ref, tw = real_chi(projs, wf)
        dev = {s: abs(ref[s] / REF684[wf][s] - 1) for s in SCANS}
        r3 = all(dev[s] <= 1e-6 and tw[s]["M_min"] == N23 and tw[s]["M_lo"] == N23 and tw[s]["C1_pass"] for s in SCANS)
        ok["R3"] &= r3
        node["R3"] = {"chi22": ref, "rel_dev": dev, "twin": tw, "pass": r3}
        log(f"R3 wf {wf}: real chi_22 by projection against #684, largest rel dev {max(dev.values()):.1e} (pass "
            f"<= 1e-6); M_min, M_lo all 23: {all(tw[s]['M_min'] == N23 == tw[s]['M_lo'] for s in SCANS)}; C1 "
            f"{all(tw[s]['C1_pass'] for s in SCANS)} -> {'OK' if r3 else 'BAD'} ({time.time() - t0:.0f}s)")
        # R1, J+: the real series judged as a plant
        refj = {s: v * (100 if mut == "judge" else 1) for s, v in ref.items()}
        r541 = {k: z + (0.05 if mut == "judge" else 0) for k, z in REF541[wf].items()}
        rr = evaluate(real[wf], wf, projs, refj, rad=0.5 if mut == "judge" else RAD_DA, ref541=r541)
        r1 = rr["da2"]["m_DA"] <= NEED_DA - 1
        if mut != "judge":
            ok["R1"] &= r1
        jp = rr["judge"]["all"]
        ok["J+"] &= jp
        for p in jplus_parts:
            jplus_parts[p] &= rr["judge"][p]
        node["real"] = rr
        log(f"R1/J+ wf {wf}: real m_DA {rr['da2']['m_DA']} (R1 pass <= 5); d_P {rr['judge']['d_P']:.1e}; d_T "
            f"{rr['judge']['d_T']:.1e}; judge (a) {rr['judge']['a']} (b) {rr['judge']['b']} (c) {rr['judge']['c']} -> "
            f"J+ {'OK' if jp else 'BAD'}; DA2 median {rr['da2']['median']}, zeros within 0.17 of #541's poles "
            f"{rr['da2']['near541']}")
        # R4, J-: the eta = 0 plants
        for kind, own in (("chord2", "kstar/chord"), ("arc", "kstar/smooth")):
            main = Main(wf, kind, real[wf])
            sig, info = make_plant(main, None, 0, 0)
            mp.mp.dps = pg.DPS_PLANT
            ref_sig = main.base.sigma()
            d4 = max(abs(a - b) for a, b in zip(sig, ref_sig))
            r4 = d4 == 0 and info["M"] == main.base.M
            ok["R4"] &= r4
            out = {"R4_maxdiff": float(d4), "R4_pass": r4, "M": info["M"]}
            if wf == WF0:
                pr = evaluate(sig, wf, projs, ref)
                jm = (not pr["judge"]["all"]) and pr["twin"][own]["M_min"] == 6
                ok["J-"] &= jm
                out.update({"plant": pr, "J-_pass": jm})
                log(f"J- wf {wf} {kind} eta 0: m_DA {pr['da2']['m_DA']} at {pr['da2']['at']}; d_P "
                    f"{pr['judge']['d_P']:.3f}; M_min in {own} {pr['twin'][own]['M_min']}; d_T {pr['judge']['d_T']:.2f}; "
                    f"judge (a) {pr['judge']['a']} (b) {pr['judge']['b']} (c) {pr['judge']['c']} -> J- "
                    f"{'OK' if jm else 'BAD'}")
            node.setdefault("eta0", {})[kind] = out
            log(f"R4 wf {wf} {kind}: eta 0 plant against FarPlant, sigma_0..31 max |diff| {float(d4):.1e}, M "
                f"{info['M']} -> {'OK' if r4 else 'BAD'}")
        log.dump(rec)
    k0, k0rec = te.check_ends(ends_all, log)
    ok["R2"] = k0
    rec["R2"] = k0rec
    rec["J+_parts"] = jplus_parts
    rec["checks"] = ok
    if mut == "col":
        fired = not ok["R3"]
        rec["mutation_fired"] = fired
        log(f"mutation col: R3 fails: {fired} (must)")
    elif mut == "judge":
        fired = not any(jplus_parts.values())
        rec["mutation_fired"] = fired
        log(f"mutation judge: J+ parts (a) {jplus_parts['a']} (b) {jplus_parts['b']} (c) {jplus_parts['c']}; each fails: "
            f"{fired} (must)")
    rec["controls_pass"] = all(ok.values()) and not mut
    log(f"controls {ok} -> pass {rec['controls_pass']}")
    log.dump(rec)
    return rec["controls_pass"]


# ---------------------------------------------------------------- grid stage
def plant_id(kind, cfg, a, eta, ph):
    return f"{kind}|{cfg}|{a}|eta{eta:.0e}|ph{ph}"


def run_grid(outdir):
    ctl = json.load(open(os.path.join(outdir, "control.json")))
    if not ctl["controls_pass"]:
        print("the controls failed: no grid run (frozen rule)")
        return False
    log = tz.Log(outdir, "grid", None)
    jl = open(os.path.join(outdir, "grid.jsonl"), "a")
    real = tc.real_sig()
    rec = {"frozen": FROZEN, "nodes": {}, "plants": {}}
    cache, parts = {}, {}

    def node_setup(wf):
        if wf not in cache:
            projs, _ = projections(wf)
            ref, _ = real_chi(projs, wf)
            mains = {k: Main(wf, k, real[wf]) for k in KINDS}
            cache[wf] = (projs, ref, mains)
            rec["nodes"][wf] = {"real_chi22": ref}
        return cache[wf]

    def run_one(wf, kind, cfg, a, eta, ph):
        projs, ref, mains = node_setup(wf)
        t0 = time.time()
        main = mains[kind]
        cur = second_curve(wf, kind, cfg, main.base.arc)
        if (wf, kind, cfg, a) not in parts:
            with mp.workdps(pg.DPS_PLANT):
                al = alpha_of(wf, a)
            parts[(wf, kind, cfg, a)] = second_part(main, cur, al)
        sig, info = make_plant(main, parts[(wf, kind, cfg, a)], eta, ph)
        r = evaluate(sig, wf, projs, ref)
        r.update({"id": plant_id(kind, cfg, a, eta, ph), "wf": wf, "kind": kind, "cfg": cfg, "alpha": a, "eta": eta,
                  "phase": ph, "Q": complex(cur.P1), "Q_in": complex(cur.P0), "plant": info,
                  "seconds": time.time() - t0})
        jl.write(json.dumps(pg.jsonable(r)) + "\n")
        jl.flush()
        j = r["judge"]
        log(f"wf {wf} {r['id']:34s} M {info['M']} fit {info['fit_resid']:.0e} share22 {info['share'][22]:.1e} "
            f"dev {info['dev_13_22']:.1e} | m_DA {j['m_DA']:2d} d_P {j['d_P']:.3f} d_T {j['d_T']:6.2f} "
            f"Mmin23 {j['M_min_23']!s:5s} -> a {j['a']!s:5s} b {j['b']!s:5s} c {j['c']!s:5s} D {j['D']:.2f}"
            f"{'' if j['C1_pass'] else ' C1 BAD'} ({r['seconds']:.1f}s)")
        return {"judge": j, "da2_at": r["da2"]["at"], "da2_median": r["da2"]["median"],
                "pade": {k: r["pade"][k]["nearest"] for k in (8, 9, 10)}, "plant": info,
                "chi22_ratio": {s: r["twin"][s]["chi"][22] / ref[s] for s in SCANS}}

    grid = [(kind, cfg, a, eta, ph) for kind in KINDS for cfg in CONFIGS for a in ALPHAS for eta in ETAS
            for ph in PHASES]
    log(f"grid: {len(grid)} plants at wf {WF0}")
    for g in grid:
        rec["plants"].setdefault(plant_id(*g), {})[WF0] = run_one(WF0, *g)
    log.dump(rec)
    at0 = {pid: d[WF0]["judge"] for pid, d in rec["plants"].items()}
    passing0 = [pid for pid, j in at0.items() if j["all"]]
    closest0 = sorted(at0, key=lambda pid: at0[pid]["D"])[:N_CLOSEST]
    more = list(dict.fromkeys(passing0 + closest0))
    log(f"wf {WF0}: {len(passing0)} plants pass (a), (b) and (c); the other nodes for those and the {N_CLOSEST} "
        f"closest: {len(more)} plants")
    lookup = {plant_id(*g): g for g in grid}
    for pid in more:
        for wf in NODES:
            if wf != WF0:
                rec["plants"][pid][wf] = run_one(wf, *lookup[pid])
        log.dump(rec)
    # verdicts
    full = {pid: rec["plants"][pid] for pid in more}
    repro = [pid for pid, d in full.items() if all(d[wf]["judge"]["all"] for wf in NODES)]
    dall = {pid: max(d[wf]["judge"]["D"] for wf in NODES) for pid, d in full.items()}
    closest = min(dall, key=dall.get)
    c1_fail = [(pid, wf) for pid, d in rec["plants"].items() for wf, x in d.items() if not x["judge"]["C1_pass"]]
    counts = {p: sum(1 for j in at0.values() if j[p]) for p in ("a", "b", "c", "all")}
    pairs = {"ab": sum(1 for j in at0.values() if j["a"] and j["b"]),
             "ac": sum(1 for j in at0.values() if j["a"] and j["c"]),
             "bc": sum(1 for j in at0.values() if j["b"] and j["c"])}
    best = {"m_DA": min(at0, key=lambda p: (at0[p]["m_DA"], at0[p]["D"])),
            "d_P": min(at0, key=lambda p: at0[p]["d_P"]), "d_T": min(at0, key=lambda p: at0[p]["d_T"])}
    groups = {}
    for pid, j in at0.items():
        kind, cfg, a = pid.split("|")[:3]
        key = f"{kind}|{cfg}|{a}"
        if key not in groups or j["D"] < groups[key]["D"]:
            groups[key] = {"D": j["D"], "id": pid, "m_DA": j["m_DA"], "d_P": j["d_P"], "d_T": j["d_T"]}
    rec["verdict"] = {"reproduces": repro, "closest": closest, "closest_D_all": dall[closest],
                      "closest_per_node": {wf: rec["plants"][closest][wf]["judge"] for wf in NODES},
                      "counts_wf1": counts, "pairs_wf1": pairs, "best_wf1": {k: (v, at0[v]) for k, v in best.items()},
                      "groups_wf1": groups, "D_all": dall, "C1_fail": c1_fail}
    for key, gq in sorted(groups.items(), key=lambda x: x[1]["D"]):
        log(f"group {key:22s}: smallest D {gq['D']:.2f} at {gq['id']} (m_DA {gq['m_DA']}, d_P {gq['d_P']:.3f}, "
            f"d_T {gq['d_T']:.2f})")
    log(f"wf {WF0}: pass (a) {counts['a']}, (b) {counts['b']}, (c) {counts['c']}, all three {counts['all']} of "
        f"{len(at0)}; (a)+(b) {pairs['ab']}, (a)+(c) {pairs['ac']}, (b)+(c) {pairs['bc']}")
    for k, (pid, j) in rec["verdict"]["best_wf1"].items():
        log(f"best {k} at wf {WF0}: {pid}: m_DA {j['m_DA']}, d_P {j['d_P']:.3f}, d_T {j['d_T']:.2f}, D {j['D']:.2f}")
    log(f"plants that reproduce the real series at all three nodes: {repro or 'none'}")
    log(f"closest (smallest D_all among the {len(full)} evaluated at every node): {closest}, D_all {dall[closest]:.2f}; "
        + "; ".join(f"wf {wf}: m_DA {j['m_DA']}, d_P {j['d_P']:.3f}, d_T {j['d_T']:.2f}"
                    for wf, j in rec["verdict"]["closest_per_node"].items()))
    log(f"C1 failures on plant scans: {c1_fail or 'none'}")
    log.dump(rec)
    return not c1_fail


def run_report(outdir):
    """post-freeze summary of grid.jsonl at wf 1.00, judged by nothing: per eta and per (kind, curve, alpha), the pass
    counts and the smallest m_DA, d_P and d_T, and the median of log10(chi_22 / chi_22(real)) over the 15 scans"""
    rows = [json.loads(x) for x in open(os.path.join(outdir, "grid.jsonl"))]
    ctl = json.load(open(os.path.join(outdir, "control.json")))
    ref = {float(w): n["R3"]["chi22"] for w, n in ctl["nodes"].items()}
    at0 = [r for r in rows if r["wf"] == WF0]
    log = tz.Log(outdir, "report", None)
    out = {"note": "post-freeze summary of grid.jsonl, judged by nothing", "by_eta": {}, "by_group": {}}

    def lr(r):
        return float(np.median([math.log10(r["twin"][s]["chi"]["22"] / ref[r["wf"]][s]) for s in SCANS]))

    def stats(rs):
        j = [r["judge"] for r in rs]
        return {"n": len(rs), "a": sum(x["a"] for x in j), "b": sum(x["b"] for x in j), "c": sum(x["c"] for x in j),
                "ab": sum(x["a"] and x["b"] for x in j), "m_DA_min": min(x["m_DA"] for x in j),
                "d_P_min": min(x["d_P"] for x in j), "d_T_min": min(x["d_T"] for x in j),
                "log_ratio_median": float(np.median([lr(r) for r in rs])),
                "log_ratio_range": [min(lr(r) for r in rs), max(lr(r) for r in rs)],
                "M_min_23": sum(x["M_min_23"] for x in j), "D_min": min(x["D"] for x in j)}
    log("eta     n   (a) (b) (c) (a+b) M_min=23  min m_DA  min d_P  min d_T  median log10(chi22/real) [range]")
    for eta in ETAS:
        s = stats([r for r in at0 if abs(r["eta"] / eta - 1) < 1e-9])
        out["by_eta"][eta] = s
        log(f"{eta:.0e} {s['n']:4d} {s['a']:4d} {s['b']:3d} {s['c']:3d} {s['ab']:4d} {s['M_min_23']:6d} {s['m_DA_min']:9d} "
            f"{s['d_P_min']:8.3f} {s['d_T_min']:8.2f}  {s['log_ratio_median']:+.2f} [{s['log_ratio_range'][0]:+.2f}, "
            f"{s['log_ratio_range'][1]:+.2f}]")
    log("group (kind|curve|alpha): (a) (b) (c) (a+b); min m_DA, d_P, d_T, D; eta of the smallest d_P")
    for kind in KINDS:
        for cfg in CONFIGS:
            for a in ALPHAS:
                rs = [r for r in at0 if (r["kind"], r["cfg"], r["alpha"]) == (kind, cfg, a)]
                s = stats(rs)
                bp = min(rs, key=lambda r: r["judge"]["d_P"])
                s["eta_best_d_P"] = bp["eta"]
                out["by_group"][f"{kind}|{cfg}|{a}"] = s
                log(f"{kind:6s} {cfg:8s} {a:3s}: {s['a']:3d} {s['b']:3d} {s['c']:3d} {s['ab']:3d}; {s['m_DA_min']:2d} "
                    f"{s['d_P_min']:.3f} {s['d_T_min']:.2f} {s['D_min']:.2f}; d_P best at eta {bp['eta']:.0e} "
                    f"phase {bp['phase']}")
    both = [r for r in at0 if r["judge"]["a"] and r["judge"]["b"]]
    for r in sorted(both, key=lambda r: r["judge"]["D"])[:20]:
        j = r["judge"]
        log(f"(a)+(b): {r['id']}: m_DA {j['m_DA']}, d_P {j['d_P']:.3f}, d_T {j['d_T']:.2f}, ratio "
            f"{j['ratio_min']:.1e}..{j['ratio_max']:.1e}, share22 {r['plant']['share']['22']:.1e}, dev "
            f"{r['plant']['dev_13_22']:.1e}")
    out["a_and_b"] = [r["id"] for r in both]
    log.dump(out)
    return True


def main():
    argv = sys.argv[1:]
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    if "--selftest" in argv:
        if mut not in (None, "secmom", "nest", "agree"):
            raise SystemExit(f"unknown selftest mutation {mut}")
        sys.exit(0 if selftest(mut) else 1)
    argv = [a for a in argv if not a.startswith("--")]
    if len(argv) != 2 or argv[0] not in ("control", "grid", "report"):
        raise SystemExit(__doc__)
    stage, outdir = argv
    if mut not in {"control": (None, "col", "judge"), "grid": (None,), "report": (None,)}[stage]:
        raise SystemExit(f"unknown {stage} mutation {mut}")
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    os.makedirs(outdir, exist_ok=True)
    if stage == "control":
        ok = run_control(outdir, mut)
        if mut:
            rec = json.load(open(os.path.join(outdir, f"control_mutate_{mut}.json")))
            print(f"mutation {mut} fired as frozen: {rec['mutation_fired']}; exit 1")
            ok = False
    elif stage == "grid":
        ok = run_grid(outdir)
    else:
        ok = run_report(outdir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
