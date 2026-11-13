#!/usr/bin/env python3
"""Twin size against the number of S_1 coefficients: could more orders separate a chord from an arc (issue #670)?

#548 (pade_chord_arc.py twin; notes/PADE_FAR_END.md 6.3) fitted a chord twin, the density
(1-s)^ex sum_{m<M} c_m T_m(2s-1) on the straight chord P0 -> kappa_* times the planted zero factor, to the arc plant
arcjump's sigma_0..sigma_22, and found chi <= 1 from M = 20 on. Here the same twin is fitted to n = 23, 25, 27 and 31
coefficients, and the smallest twin size is reported per n and node.

Coefficients. The plants are known functions, so their sigma_0..sigma_30 are computed from them (pade_geometry.Plant's
tanh-sinh moments, kept to order KM = NMAX; checked against the rule with half the step).
Errors. err_k = F rel_k |sigma_k|. For k <= 22, rel_k is measured (sigma_err_fit.py on the scan45J1250 file with a1
from scan39J1000, as pade_geometry.noise_eps), with a floor of 1e-30. For k >= 23 it is an estimate, extrapolated
from log10 rel_k = a + b k fitted on k = 18..22 at the node (b = 0.157 dex per order). F = 1 is the model, F = 0.1 the
errors 10x smaller (what more precise runs, #549, would buy), F = 0.01 context.
Twin basis. The chord moments are exact (Beta functions), not quadrature, so the same-class control (chord2 refitted at
M = 6) also tests the plants' extra orders. The twin is the weighted least-squares fit (weights 1/err_k) by Householder
QR on the real form at DPS_FIT digits; DPS_CHECK repeats the rows around each M_min.
Sizes. chi = max_k |sigma_k(twin) - sigma_k| / err_k and l2 = (sum_k (|sigma_k(twin) - sigma_k| / err_k)^2)^(1/2).
  M_min  smallest M with chi <= 1 for the least-squares twin, an upper bound on the size of the smallest twin.
  M_lo   smallest M with l2 <= sqrt(n), a lower bound: a twin with chi <= 1 has l2 <= sqrt(n), the least-squares twin
         minimises l2, and the sizes are nested, so no twin smaller than M_lo has chi <= 1.
  d_m    visibility of the m-th Chebyshev term: the weighted distance of its column from the span of the lower ones,
         in error units per unit coefficient (context).

    python3 pade_twin_count.py --selftest [--mutate=moment|planted|errtail|quad]
    python3 pade_twin_count.py run OUTDIR [--mutate=moment] [--allow_unraw]

The selftest checks the exact moments against quadrature, the plants' extra orders against half the step, the error
model's tail fit, the precision of the fit and the size finder on planted chord densities of known size; each
mutation must make its check fail. `run` exits 1 when a control fails. Numerical evidence on plants, not a bound on
the real series.
"""
import json
import math
import os
import sys
import time

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pade_farend as F                                                # noqa: E402

pg, pp = F.pg, F.pp
NMAX = 31                           # sigma_0..sigma_30
pg.KM = NMAX                        # plants keep sigma_0..sigma_31; their fit to sigma_0..sigma_12 is unchanged
N_MEAS = 23                         # measured errors: sigma_0..sigma_22
NS = (23, 25, 27, 31)
SCALES = (1.0, 0.1, 0.01)           # judged: 1 and 0.1; 0.01 context
TAIL = (18, 22)                     # orders of the log-linear error fit
FLOOR = 1e-30
DPS_FIT = 150
DPS_CHECK = 250
M_CONTROL = 6
PLANTED = (8, 12, 16)               # sizes whose top term is visible (24 is not: notes/PADE_FAR_END.md 7.2)
REF548 = {0.99: {18: 2.931, 20: 1.666e-3, 22: 2.696e-7}, 1.0: {18: 3.094, 20: 1.527e-3, 22: 2.449e-7},
          1.01: {18: 2.594, 20: 2.079e-3, 22: 1.818e-7}}   # I548-mac/twin.json (raw edcf1b1), even M only
FROZEN = {                          # written before the run stage was first executed (notes/PADE_FAR_END.md 7.2)
    "twin": "#548's chord twin, fitted to arcjump's sigma_0..sigma_(n-1), n = 23, 25, 27, 31, at wf 0.99, 1.00, 1.01",
    "errors": "measured rel_k for k <= 22 (floor 1e-30); k >= 23 from the log-linear fit on k = 18..22 (an estimate); "
              "all orders times F = 1, 0.1 (judged) and 0.01 (context)",
    "sizes": "M_min (chi <= 1, least squares) and M_lo (l2 <= sqrt(n)), the smallest twin lies in [M_lo, M_min]",
    "C0_control": "chord2's own twin at M = 6 has chi <= 1e-3 at every n, node and F",
    "C1_precision": "at M_min - 1 and M_min, chi at DPS_CHECK agrees with DPS_FIT within 1e-3 relative",
    "C2_reproduce": "at n = 23, F = 1: chi at M = 18, 20, 22 within a factor 2 of #548's values (REF548); a failure is "
                    "a discrepancy, reported",
    "near_n_minus_3": "at F = 1, every node: M_min >= n - 4 at n = 27 and 31",
    "saturates": "at F = 1, every node: M_min(31) <= M_min(23) + 2",
    "verdict_549": "if M stays near n - 3 or saturates, at F = 1 and at F = 0.1, #549's extra orders cannot separate a "
                   "chord from an arc without a prior on the density's size, and the prior they would need moves by "
                   "at most M_min(25) - M_min(23) terms; otherwise the gap n - M_min is reported and the verdict is open",
    "context": "the arc plant (a smooth density on the arc, wf 1.00), F = 0.01 and the visibility d_m of each Chebyshev "
               "term (weights from arcjump) are reported, not judged",
}


# ---------------------------------------------------------------- coefficients and errors
_REAL = {}


def real_sig():
    """the J1250 sigma_k per node, parsed from the strings at sigma_load.DPS_LOAD digits whatever the precision"""
    if not _REAL:
        _REAL.update(pg.load_scan(pg.REAL["J1250"]))
    return _REAL


def plant(wf, kind):
    pl = F.FarPlant(wf, kind, real_sig()[wf])
    mp.mp.dps = pg.DPS_PLANT
    return pl, pl.sigma()[:NMAX]


def resampled(pl, h):
    """the plant's sigma_0..sigma_(NMAX-1) with its tanh-sinh rule at step h, c_m kept"""
    mp.mp.dps = pg.DPS_PLANT
    rule = pg.ts_rule(h=h)
    nodes = []
    for a, b, w in pl.pieces:
        a, b = mp.mpf(a), mp.mpf(b)
        for x, omx, wt in rule:
            nodes.append((a + (b - a) * x, (1 - b) + (b - a) * omx, (b - a) * wt * w))
    keep = pl.nodes, pl.mom
    pl.nodes = nodes
    pl.mom = pl.moments(pl.M)
    sig = pl.sigma()[:NMAX]
    pl.nodes, pl.mom = keep
    return sig


def err_model(wf, mut=None):
    """relative error per order k < NMAX and the tail fit (slope, intercept, largest residual in dex)"""
    pg.NOISE_MODE = "measured"
    rel = [float(x) for x in pg.noise_eps(wf)[:N_MEAS]]
    if mut == "errtail":
        rel[20] *= 3
    ks = np.arange(TAIL[0], TAIL[1] + 1)
    y = np.log10(np.array(rel[TAIL[0]:TAIL[1] + 1]))
    b, a = np.polyfit(ks, y, 1)
    resid = float(np.max(np.abs(y - (a + b * ks))))
    out = [max(r, FLOOR) for r in rel] + [float(10 ** (a + b * k)) for k in range(N_MEAS, NMAX)]
    q = np.polyfit(ks, y, 2)        # context: the tail's slope falls with k; a quadratic continues that
    return out, {"slope_dex": float(b), "intercept": float(a), "tail_resid_dex": resid,
                 "rel_22": rel[22], "rel_30": out[NMAX - 1],
                 "linear_over_quadratic_30": float(10 ** (a + b * (NMAX - 1) - np.polyval(q, NMAX - 1)))}


# ---------------------------------------------------------------- exact chord moments
def shifted_cheb(M):
    """integer coefficients in s of T_m(2s - 1), m < M"""
    T = [[1], [-1, 2]]
    while len(T) < M:
        a, b = T[-1], T[-2]
        n = [0] * (len(a) + 1)
        for i, c in enumerate(a):
            n[i] -= 2 * c
            n[i + 1] += 4 * c
        for i, c in enumerate(b):
            n[i] -= c
        T.append(n)
    return T[:M]


def chord_moments(P0, P1, ex, M, K):
    """mom[m][k] = (P1 - P0)/k! int_0^1 (1-s)^ex T_m(2s-1) (P0 + (P1 - P0) s)^k ds, k <= K, by
    int_0^1 s^p (1-s)^ex ds = B(p+1, ex+1), at the working precision"""
    P0, P1, ex = mp.mpc(P0), mp.mpc(P1), mp.mpc(ex)
    D = P1 - P0
    T = shifted_cheb(M)
    beta = [1 / (ex + 1)]
    for p in range(M + K):
        beta.append(beta[-1] * (p + 1) / (p + ex + 2))
    q = [[mp.fsum(c * beta[i + j] for i, c in enumerate(T[m])) for j in range(K + 1)] for m in range(M)]
    pw0 = [P0 ** j for j in range(K + 1)]
    pwd = [D ** j for j in range(K + 1)]
    return [[D / mp.factorial(k) * mp.fsum(mp.binomial(k, j) * pw0[k - j] * pwd[j] * q[m][j] for j in range(k + 1))
             for k in range(K + 1)] for m in range(M)]


class Basis:
    """the twin's columns (pref * mom_m)_k, m < NMAX, k < NMAX, at one precision"""

    def __init__(self, base, dps, mut=None):
        self.dps, self.ex = dps, base.ex
        with mp.workdps(dps):
            mom = chord_moments(base.P0, base.P1, base.ex, NMAX, NMAX - 1)
            if mut == "moment":
                mom[0][26] *= 1 + mp.mpf("1e-8")
            pref = [mp.mpc(z) for z in base.pref]
            self.cols = [pp.polymul(pref, mom[m], NMAX) for m in range(NMAX)]


# ---------------------------------------------------------------- twin fit
def lsq(basis, target, err, n, M):
    """least-squares fit of columns 0..M-1 to target_0..target_(n-1), weights 1/err_k: (c, chi, l2)"""
    with mp.workdps(basis.dps):
        A, b = mp.matrix(2 * n, 2 * M), mp.matrix(2 * n, 1)
        for k in range(n):
            for m in range(M):
                z = basis.cols[m][k] / err[k]
                A[k, m], A[k, M + m], A[n + k, m], A[n + k, M + m] = z.real, -z.imag, z.imag, z.real
            z = mp.mpc(target[k]) / err[k]
            b[k], b[n + k] = z.real, z.imag
        x, _ = mp.qr_solve(A, b)
        c = [mp.mpc(x[m], x[M + m]) for m in range(M)]
        r = [abs(mp.fsum(c[m] * basis.cols[m][k] for m in range(M)) - target[k]) / err[k] for k in range(n)]
        return c, float(max(r)), float(mp.sqrt(mp.fsum(v * v for v in r)))


def weights(sig, rel, n, scale, dps):
    with mp.workdps(dps):
        return [mp.mpf(scale * rel[k]) * abs(mp.mpc(sig[k])) for k in range(n)]


def twin_fit(basis, sig, rel, n, M, scale):
    """least-squares twin of size M against sigma_0..sigma_(n-1), err_k = scale rel_k |sigma_k|: (c, chi, l2)"""
    return lsq(basis, sig, weights(sig, rel, n, scale, basis.dps), n, M)


def visibility(basis, sig, rel, n, scale):
    """d_m, m = 1..n-1: the weighted distance of column m from the span of columns 0..m-1, in error units per unit c_m
    (a term of size |c_m| moves sigma_0..sigma_(n-1) by |c_m| d_m error units after the lower terms are refitted)"""
    err = weights(sig, rel, n, scale, basis.dps)
    return [lsq(basis, basis.cols[m], err, n, m)[2] for m in range(1, n)]


def n_visible(d, thr):
    """number of leading terms visible at coefficient size 1/thr: the first m with d_m < thr"""
    return next((m for m, x in enumerate(d, start=1) if x < thr), len(d) + 1)


def density_peak(c, ex):
    """largest |(1-s)^ex sum c_m T_m(2s-1)| on s = j/200, j < 200 (as #548)"""
    s = np.arange(200) / 200
    v = np.polynomial.chebyshev.chebval(2 * s - 1, np.array([complex(z) for z in c])) * (1 - s) ** complex(ex)
    return float(np.max(np.abs(v)))


def row(basis, sig, rel, n, M, scale):
    c, chi, l2 = twin_fit(basis, sig, rel, n, M, scale)
    return {"M": M, "chi": chi, "l2": l2, "c_max": max(float(abs(z)) for z in c),
            "density_max": density_peak(c, basis.ex)}


def sizes(rows, n):
    m_min = next((r["M"] for r in rows if r["chi"] <= 1), None)
    m_lo = next((r["M"] for r in rows if r["l2"] <= math.sqrt(n)), None)
    return m_min, m_lo


def scan(basis, sig, rel, n, scale, Ms, log=None):
    rows = []
    for M in Ms:
        rows.append(row(basis, sig, rel, n, M, scale))
        if log:
            r = rows[-1]
            log(f"    M {M:2d}: chi {r['chi']:.3e}  l2 {r['l2']:.3e}  max|c| {r['c_max']:.2e}  "
                f"max|density| {r['density_max']:.2f}")
    return rows


# ---------------------------------------------------------------- planted chord densities of known size
def planted_target(base, M_true, mut=None):
    """chord2's chord and exponent with c_m = 0.8^m e^(i m), m < M_true, by the plant's quadrature moments"""
    mp.mp.dps = pg.DPS_PLANT
    c = [mp.mpf("0.8") ** m * mp.expjpi(mp.mpf(m) / mp.pi) for m in range(M_true)]
    if mut == "planted":
        c[-1] = mp.mpc(0)
    mom = base.moments(M_true)
    lc = [mp.fsum(c[m] * mom[m][k] for m in range(M_true)) for k in range(NMAX)]
    s = pp.polymul(base.pref, lc, NMAX)
    return [z / s[0] for z in s]


def planted_check(base, basis, rel, mut=None, log=print):
    """the size finder recovers M_true: l2 > sqrt(n) at M_true - 1 (no smaller twin) and chi <= 1 at M_true"""
    ok, out = True, {}
    for M_true in PLANTED:
        sig = planted_target(base, M_true, mut)
        for n in (NS[0], NMAX):
            for scale in (1.0, 0.1):
                lo, hi = row(basis, sig, rel, n, M_true - 1, scale), row(basis, sig, rel, n, M_true, scale)
                good = lo["l2"] > math.sqrt(n) and hi["chi"] <= 1
                ok &= good
                out[f"{M_true}_{n}_{scale:g}"] = {"below": lo, "at": hi, "pass": good}
                log(f"planted M_true {M_true:2d}, n {n}, F {scale:g}: l2 at M_true - 1 {lo['l2']:.2e} "
                    f"(> {math.sqrt(n):.2f}), chi at M_true {hi['chi']:.2e} (<= 1): {'PASS' if good else 'FAIL'}")
    return ok, out


# ---------------------------------------------------------------- selftest
def selftest(mut=None):
    ok = True
    wf = 1.0
    base, sig_c = plant(wf, "chord2")
    # 1: exact chord moments against the plant's quadrature moments
    mp.mp.dps = pg.DPS_PLANT
    qm = base.moments(NMAX)
    basis = Basis(base, DPS_FIT, mut)
    with mp.workdps(DPS_FIT):
        ex = chord_moments(base.P0, base.P1, base.ex, NMAX, NMAX - 1)
        if mut == "moment":
            ex[0][26] *= 1 + mp.mpf("1e-8")
        e1 = max(abs(ex[m][k] - qm[m][k]) / abs(ex[m][k]) for m in range(NMAX) for k in range(NMAX))
    print(f"selftest 1: exact (Beta) chord moments against tanh-sinh, m, k < {NMAX}: max rel diff {float(e1):.1e} "
          f"(pass < 1e-40)")
    ok &= e1 < mp.mpf("1e-40")
    # 2: the plants' extra orders against the rule with half the step
    h = mp.mpf(1) / (8 if mut == "quad" else 64)
    e2 = 0
    for kind in ("arcjump", "chord2"):
        pl, sig = plant(wf, kind)
        sig2 = resampled(pl, h)
        e = max(abs(sig2[k] - sig[k]) / abs(sig[k]) for k in range(NMAX))
        e2 = max(e2, e)
        print(f"selftest 2: {kind} sigma_0..sigma_{NMAX - 1}, tanh-sinh step 1/32 against 1/{int(1 / h)}: max rel diff "
              f"{float(e):.1e} (pass < 1e-35)")
    ok &= e2 < mp.mpf("1e-35")
    # 3: the error model's tail fit
    rel, info = err_model(wf, mut)
    print(f"selftest 3: log-linear tail fit of rel_k on k = {TAIL[0]}..{TAIL[1]}: slope {info['slope_dex']:.4f} dex, "
          f"largest residual {info['tail_resid_dex']:.3f} dex (pass <= 0.05); rel_30 = {info['rel_30']:.2e}")
    ok &= info["tail_resid_dex"] <= 0.05
    # 4: same-class control at n = NMAX: chord2 refitted at M = 6
    r = row(basis, sig_c, rel, NMAX, M_CONTROL, 1.0)
    print(f"selftest 4: chord2 refitted at M = 6 against sigma_0..sigma_{NMAX - 1}: chi {r['chi']:.1e} (pass <= 1e-3)")
    ok &= r["chi"] <= 1e-3
    # 5: precision of the fit at a large size
    _, arcj = plant(wf, "arcjump")
    hi = Basis(base, DPS_CHECK)
    a, b = row(basis, arcj, rel, NMAX, 27, 1.0), row(hi, arcj, rel, NMAX, 27, 1.0)
    e5 = abs(a["chi"] - b["chi"]) / b["chi"]
    print(f"selftest 5: arcjump twin M 27, n {NMAX}: chi {a['chi']:.6e} at {DPS_FIT} digits, {b['chi']:.6e} at "
          f"{DPS_CHECK}: rel diff {e5:.1e} (pass <= 1e-3)")
    ok &= e5 <= 1e-3
    # 6: the size finder on planted chord densities
    ok6, _ = planted_check(base, basis, rel, mut)
    ok &= ok6
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------- run
def run(outdir, mut=None):
    logf = open(os.path.join(outdir, "run" + (f"_mutate_{mut}" if mut else "") + ".log"), "a")
    rowsf = open(os.path.join(outdir, "rows" + (f"_mutate_{mut}" if mut else "") + ".jsonl"), "a")

    def log(s):
        print(s)
        logf.write(s + "\n")
        sys.stdout.flush()
        logf.flush()

    log(f"run {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC, mutate {mut}")
    rec = {"frozen": FROZEN, "mutate": mut, "errors": {}, "targets": {}, "controls": {}}
    ok_controls = True
    for wf in pg.NODES:
        base, sig_c = plant(wf, "chord2")
        basis = Basis(base, DPS_FIT, mut)
        check = Basis(base, DPS_CHECK, mut)
        rel, info = err_model(wf)
        rec["errors"][wf] = {"rel": rel, **info}
        log(f"wf {wf}: error tail slope {info['slope_dex']:.4f} dex/order, residual {info['tail_resid_dex']:.3f}; "
            f"rel_22 {info['rel_22']:.2e}, rel_30 {info['rel_30']:.2e} (extrapolated; a quadratic tail gives "
            f"{info['linear_over_quadratic_30']:.2f} times less)")
        kinds = ("arcjump", "arc") if wf == 1.0 else ("arcjump",)
        targets = {k: plant(wf, k)[1] for k in kinds}
        for scale in SCALES:
            for n in NS:
                c0 = row(basis, sig_c, rel, n, M_CONTROL, scale)
                good = c0["chi"] <= 1e-3
                if scale in (1.0, 0.1):
                    ok_controls &= good
                rec["controls"].setdefault(wf, {}).setdefault(scale, {})[n] = {**c0, "pass": good}
                d = visibility(basis, targets["arcjump"], rel, n, scale)
                vis = {"d": d, "visible_c1": n_visible(d, 1.0), "visible_c1e3": n_visible(d, 1e-3)}
                rec["visibility"] = rec.get("visibility", {})
                rec["visibility"].setdefault(wf, {}).setdefault(scale, {})[n] = vis
                log(f"wf {wf} F {scale:g} n {n}: C0 chord2 at M 6 chi {c0['chi']:.1e} {'OK' if good else 'BAD'}; "
                    f"terms visible at |c_m| = 1: {vis['visible_c1']}, at |c_m| = 1e3: {vis['visible_c1e3']}")
                for kind, sig in targets.items():
                    t0 = time.time()
                    rows = scan(basis, sig, rel, n, scale, range(M_CONTROL, n + 1))
                    m_min, m_lo = sizes(rows, n)
                    prec = []
                    for M in (m_min - 1, m_min) if m_min else ():
                        r = next((x for x in rows if x["M"] == M), None)
                        if r is None:
                            continue
                        rc = row(check, sig, rel, n, M, scale)
                        d = abs(rc["chi"] - r["chi"]) / rc["chi"]
                        prec.append({"M": M, "chi_check": rc["chi"], "l2_check": rc["l2"], "rel_diff": d})
                    good_p = bool(prec) and all(p["rel_diff"] <= 1e-3 for p in prec)
                    if scale in (1.0, 0.1):
                        ok_controls &= good_p
                    at = next((x for x in rows if x["M"] == m_min), None)
                    res = {"rows": rows, "M_min": m_min, "M_lo": m_lo, "precision": prec, "C1_pass": good_p}
                    rec["targets"].setdefault(kind, {}).setdefault(wf, {}).setdefault(scale, {})[n] = res
                    rowsf.write(json.dumps(pg.jsonable({"wf": wf, "F": scale, "n": n, "kind": kind, **res})) + "\n")
                    rowsf.flush()
                    log(f"wf {wf} F {scale:g} n {n} {kind:8s}: M_min {m_min} M_lo {m_lo} (n - M_min "
                        f"{n - m_min if m_min else '-'}); chi at M_min - 1, M_min: "
                        + ", ".join(f"{x['chi']:.2e}" for x in rows if m_min and x["M"] in (m_min - 1, m_min))
                        + (f"; max|density| at M_min {at['density_max']:.2f}" if at else "")
                        + f"; C1 {'OK' if good_p else 'BAD'} ({time.time() - t0:.0f}s)")
    # verdicts
    aj = rec["targets"]["arcjump"]
    v = {}
    for scale in (1.0, 0.1):
        mm = {wf: {n: aj[wf][scale][n]["M_min"] for n in NS} for wf in pg.NODES}
        near = all(mm[wf][n] is not None and mm[wf][n] >= n - 4 for wf in pg.NODES for n in (27, 31))
        sat = all(mm[wf][31] is not None and mm[wf][23] is not None and mm[wf][31] <= mm[wf][23] + 2
                  for wf in pg.NODES)
        v[scale] = {"M_min": mm, "near_n_minus_3": near, "saturates": sat,
                    "gap": {wf: {n: (n - mm[wf][n]) if mm[wf][n] else None for n in NS} for wf in pg.NODES}}
        log(f"F {scale:g}: M_min " + "; ".join(f"wf {wf}: " + " ".join(f"n{n} {mm[wf][n]}" for n in NS)
                                              for wf in pg.NODES)
            + f"  -> near n - 3: {near}, saturates: {sat}")
    ref = {}
    for wf in pg.NODES:
        rows = aj[wf][1.0][23]["rows"]
        ref[wf] = {M: {"chi": next(x["chi"] for x in rows if x["M"] == M), "ref": REF548[wf][M]} for M in REF548[wf]}
        for M, d in ref[wf].items():
            d["within_2x"] = 0.5 <= d["chi"] / d["ref"] <= 2
    c2 = all(d["within_2x"] for r in ref.values() for d in r.values())
    log("C2 reproduction of #548 at n 23, F 1: " + "; ".join(
        f"wf {wf} " + " ".join(f"M{M} {d['chi']:.3e}/{d['ref']:.3e}" for M, d in r.items()) for wf, r in ref.items())
        + f" -> {'OK' if c2 else 'DISCREPANCY'}")
    both = all(v[s]["near_n_minus_3"] or v[s]["saturates"] for s in (1.0, 0.1))
    rec.update({"verdict": v, "C2_reproduce": {"rows": ref, "pass": c2}, "controls_pass": ok_controls,
                "verdict_549": "no separation without a prior" if both else "open"})
    log(f"controls C0 and C1 pass: {ok_controls}; verdict on #549: {rec['verdict_549']}")
    json.dump(pg.jsonable(rec), open(os.path.join(outdir, "twin_count" + (f"_mutate_{mut}" if mut else "") + ".json"),
                                     "w"), indent=1)
    return ok_controls


def main():
    argv = sys.argv[1:]
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    if "--selftest" in argv:
        sys.exit(0 if selftest(mut) else 1)
    argv = [a for a in argv if not a.startswith("--")]
    if len(argv) != 2 or argv[0] != "run":
        raise SystemExit(__doc__)
    outdir = argv[1]
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    os.makedirs(outdir, exist_ok=True)
    sys.exit(0 if run(outdir, mut) else 1)


if __name__ == "__main__":
    main()
