#!/usr/bin/env python3
"""The fold's square root cancels in lam_g - r Delta (#819): the combination's series at #818's bases, typed by
hp/singularity_type.py's estimators, with exponent 3/2 expected when lam_g and Delta share one branch point and one
Puiseux variable.

    cd hp && python3 fold_cancel.py coeffs RAWDIR [--json=OUT]    the inputs: b1, b3, b5 of each series at each base
    cd hp && python3 fold_cancel.py planted RAWDIR [--json=OUT]   the plants of the two series' sizes (exits 1 on a fail)
    cd hp && python3 fold_cancel.py real RAWDIR [--json=OUT]      lam_g - r Delta at the bases, r and r (1 +- 0.1)
    cd hp && python3 fold_cancel.py calibrate RAWDIR [--json=OUT] r's range over the bases against a planted shift of
                                                                  lam_g's fold (#820; exits 1 on a fail)

RAWDIR is a clone of the raw-data repository (I603-win2/chain_nsr, as hp/lam_singularity.py reads it). The bases are
#818's 26, where lam_g's own series meet the fold criterion (hp/data/lam_singularity.json's hits).

coeffs: at each base, s = 259.9033 - D0, the series' orders k = 10..24 fitted by least squares (each order weighted by
1/|c_k|) as c_k = sum_j b_j s^(j/2) binom(j/2, k) (-1)^k s^-k over j = 1, 3, 5: the coefficients of sqrt(D* - D),
(D* - D)^(3/2) and (D* - D)^(5/2) at the fold. r = b1(lam_g)/b1(Delta).

planted: at each base, both series built from the medians of coeffs' b (the fold at 259.9033), #782's near
contamination (a second real square-root point at 260.17 and a pair at 259.47 +- 1.17i, 0.3 of the fold's amplitude,
+ for lam_g and - for Delta, so that it does not cancel), the real series' orders 1-9 beyond their fitted fold terms,
and relative noise 1e-12. X = lam_g - r Delta is typed for r = b1(lam_g)/b1(Delta) of the plant, r (1 +- 0.1) and
0.0203, and for a pair whose lam_g fold sits 0.01 above Delta's (r fitted as coeffs would). 'cancels': both estimators
put D* within 0.01 of 259.9033 with mu in [1.25, 1.75]; 'S': both mu in [0.25, 0.75]. planted passes when the plant's r
cancels at 4 or more bases, both r (1 +- 0.1) are S at 4 or more, and the shifted pair cancels at fewer than 4.
First run (2026-11-15, #819): FAIL. The plant's r cancels at no base: the ratio estimator reads mu 1.51 to 2.7, the
Dlog estimator 1.14 to 1.27. Plants without contamination and the regular part give mu 1.51 to 1.58 (ratio) and 1.500
(Dlog); the regular low orders move Dlog's pole to D 260.35-262.5, and the contamination lowers its mu to 1.08-1.13.
"""
import json
import os
import sys

import mpmath as mp
import numpy as np

import lam_frontier as LFR
import lam_singularity as LS

HERE = os.path.dirname(os.path.abspath(__file__))
DSTAR = LS.DSTAR
KS = range(10, 25)
JS = (1, 3, 5)
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:] if a.startswith("--"))
pos = [a for a in sys.argv[2:] if not a.startswith("--")]


def bases():
    hits = json.load(open(os.path.join(HERE, "data", "lam_singularity.json")))["hits"]
    return sorted(hits)


def basis(j, k, s):
    return s ** (mp.mpf(j) / 2) * mp.binomial(mp.mpf(j) / 2, k) * (-1) ** k / s ** k


def puiseux(c, D0, ks=KS, js=JS):
    """b_j of the fold's terms from c_k (c[k - 1] = c_k), weighted least squares; and the largest relative residual"""
    s = DSTAR - mp.mpf(D0)
    A = np.array([[float(basis(j, k, s) / c[k - 1]) for j in js] for k in ks])
    y = np.ones(len(ks))
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return [float(x) for x in b], float(np.max(np.abs(A @ b - y)))


def series(raw, D0):
    jp = os.path.join(raw, "I603-win2", "chain_nsr", "jet_pin3_D%.4f.npz" % D0)
    Db, c, M = LFR.load_ah_jet(jp)
    return LS.lam_series(Db, c, M), LS.delta_series(jp)


def cmd_coeffs(raw):
    mp.mp.dps = 200
    out = {}
    print("b1, b3, b5 of sqrt(D* - D), (D* - D)^(3/2), (D* - D)^(5/2) from c_10..c_24 (largest relative residual)")
    for D0 in bases():
        lam, dl = series(raw, D0)
        bl, el = puiseux(lam[1:], D0)
        bd, ed = puiseux(dl[1:], D0)
        out[D0] = dict(lam=bl, lam_res=el, delta=bd, delta_res=ed, r=bl[0] / bd[0])
        print("  D0 %.4f  lam_g %+.5e %+.4e %+.3e (%.1e) | Delta %+.5e %+.4e %+.3e (%.1e) | r %.6f" % (
            D0, *bl, el, *bd, ed, bl[0] / bd[0]))
    rs = [v["r"] for v in out.values()]
    print("r: median %.6f, %.6f to %.6f" % (float(np.median(rs)), min(rs), max(rs)))
    if "json" in args:
        json.dump(out, open(args["json"], "w"), indent=1)
    return 0


CONT = 0.3          # #782's near contamination, against the fold's amplitude b1 sqrt(s): + for lam_g, - for Delta


def fold_terms(b, s, N=24):
    """c_1..c_N of sum_j b_j (D* - D)^(j/2) about a base s below D*"""
    return [sum(bj * basis(j, k, s) for bj, j in zip(b, JS)) for k in range(1, N + 1)]


def contamination(amp, D0, shift=0.0, N=24):
    """amp times the coefficients of (1 - h/s2)^(1/2) + 2 Re (1 - h/s3)^(1/2), s2 = 260.17 - D0 and s3 =
    259.47 + 1.17i - D0 (#782's near plant), moved by shift"""
    s2 = mp.mpf(260.17 - D0 + shift)
    s3 = mp.mpc(259.47 - D0 + shift, 1.17)
    h = mp.mpf(1) / 2
    return [amp * (mp.binomial(h, k) * (-1) ** k * (1 / s2 ** k + 2 * mp.re(1 / s3 ** k))) for k in range(1, N + 1)]


def plant_pair(D0, bl, bd, reg_l, reg_d, shift=0.0, seed=819, common=0.0):
    """the two planted series c_1..c_24: the fold terms (lam_g's moved by shift, both folds by common), the
    contamination, the real series' low orders beyond their fitted fold terms (orders 1-9), relative noise 1e-12"""
    rng = np.random.default_rng(seed + int(round(D0 * 1e4)) % 100000)
    s = DSTAR - mp.mpf(D0)
    out = []
    for b, reg, sign, sh in ((bl, reg_l, 1, shift), (bd, reg_d, -1, 0.0)):
        f = fold_terms(b, s + sh + common)
        c = contamination(sign * CONT * b[0] * mp.sqrt(s + sh), D0, sh)
        out.append([(f[k] + c[k] + reg[k]) * (1 + 1e-12 * rng.standard_normal()) for k in range(24)])
    return out


def kind_of(D0, e):
    """'cancels' when both estimators put D* within 0.01 of DSTAR with mu in [1.25, 1.75]; 'S' when both mu lie in
    [0.25, 0.75]; otherwise 'neither'"""
    ok = all(x is not None for x in e)
    if ok and all(abs(D0 + x[0] - float(DSTAR)) <= LS.TOL and 1.25 <= x[1] <= 1.75 for x in e):
        return "cancels"
    if ok and all(0.25 <= x[1] <= 0.75 for x in e):
        return "S"
    return "neither"


def low_orders(raw, D0, coeffs):
    """the real series' orders 1-9 beyond their fitted fold terms (and 0 from order 10), lam_g's and Delta's"""
    lam, dl = series(raw, D0)
    s = DSTAR - mp.mpf(D0)
    out = []
    for c, b in ((lam[1:], coeffs["lam"]), (dl[1:], coeffs["delta"])):
        f = fold_terms(b, s)
        out.append([c[k] - f[k] if k < 9 else mp.mpf(0) for k in range(24)])
    return out


def type_X(D0, pl, pd, r):
    X = [a - r * b for a, b in zip(pl, pd)]
    e = LS.est(X)
    return dict(ratio=e[0], dlog=e[1], kind=kind_of(D0, e))


def cmd_planted(raw):
    mp.mp.dps = 200
    co = {float(k): v for k, v in json.load(open(args.get("coeffs", os.path.join(HERE, "data", "fold_cancel_coeffs.json")))).items()}
    med = lambda key, i: float(np.median([v[key][i] for v in co.values()]))      # noqa: E731
    bl = [med("lam", i) for i in range(3)]
    bd = [med("delta", i) for i in range(3)]
    r = bl[0] / bd[0]
    print("plants: lam_g b %s, Delta b %s (medians of coeffs), r %.6f; contamination +-%.1f" % (
        ", ".join("%+.4e" % x for x in bl), ", ".join("%+.4e" % x for x in bd), r, CONT))
    out, tally = {}, {}
    for D0 in sorted(co):
        reg_l, reg_d = low_orders(raw, D0, co[D0])
        pl, pd = plant_pair(D0, bl, bd, reg_l, reg_d)
        res = {"r": type_X(D0, pl, pd, r), "r+10%": type_X(D0, pl, pd, 1.1 * r), "r-10%": type_X(D0, pl, pd, 0.9 * r),
               "0.0203": type_X(D0, pl, pd, mp.mpf("0.0203"))}
        # lam_g's fold 0.01 above Delta's; r as coeffs would measure it (the fold terms fitted at DSTAR)
        ql, qd = plant_pair(D0, bl, bd, reg_l, reg_d, shift=0.01)
        rs = puiseux(ql, D0)[0][0] / puiseux(qd, D0)[0][0]
        res["shift"] = dict(type_X(D0, ql, qd, rs), r_fit=rs)
        out[D0] = res
        for k, v in res.items():
            tally.setdefault(k, {}).setdefault(v["kind"], 0)
            tally[k][v["kind"]] += 1
        f = lambda v: "%s (ratio mu %s, dlog mu %s)" % (v["kind"], "%+.3f" % v["ratio"][1] if v["ratio"] else "-",   # noqa: E731
                                                     "%+.3f" % v["dlog"][1] if v["dlog"] else "-")
        print("  D0 %.4f  r: %s | +10%%: %s | -10%%: %s | 0.0203: %s | shift: %s" % (
            D0, f(res["r"]), f(res["r+10%"]), f(res["r-10%"]), f(res["0.0203"]), f(res["shift"])))
    ok = (tally["r"].get("cancels", 0) >= LS.NCOUNT and tally["r+10%"].get("S", 0) >= LS.NCOUNT
          and tally["r-10%"].get("S", 0) >= LS.NCOUNT and tally["shift"].get("cancels", 0) < LS.NCOUNT)
    print("tally:", json.dumps(tally))
    print("planted:", "PASS" if ok else "FAIL")
    if "json" in args:
        json.dump(dict(b_lam=bl, b_delta=bd, r=r, tally=tally, bases=out), open(args["json"], "w"), indent=1)
    return 0 if ok else 1


DELTAS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)          # #820's shifts of lam_g's fold, judged (+delta, common 0)
DELTAS_X = (3e-6, 3e-5, 3e-4)                    # reported
EPS = (-4e-4, -2e-4, -1e-4, 0.0, 1e-4, 2e-4, 4e-4)   # a shift of both folds, reported (Delta's D* spread, #782/#818)


def rel_range(v):
    return (max(v) - min(v)) / abs(float(np.median(v)))


def cmd_calibrate(raw):
    """r's relative range over #818's 26 bases for lam_g's fold planted at 259.9033 + delta (Delta's at 259.9033), r
    fitted at every base as coeffs fits it. Judged: the range grows strictly with delta over DELTAS, and the delta = 0
    range lies below the real one. The nominal bound: the separations whose range exceeds the real one are excluded;
    the bound is the smallest such delta of DELTAS, and the largest delta below it is reported as allowed. Reported:
    -delta, DELTAS_X, and the robust range, the smallest over a common shift EPS of both folds."""
    mp.mp.dps = 200
    co = {float(k): v for k, v in json.load(open(args.get("coeffs", os.path.join(HERE, "data", "fold_cancel_coeffs.json")))).items()}
    real = rel_range([v["r"] for v in co.values()])
    med = lambda key, i: float(np.median([v[key][i] for v in co.values()]))      # noqa: E731
    bl = [med("lam", i) for i in range(3)]
    bd = [med("delta", i) for i in range(3)]
    reg = {D0: low_orders(raw, D0, co[D0]) for D0 in sorted(co)}

    def rng_of(delta, eps=0.0):
        rs = []
        for D0 in sorted(co):
            ql, qd = plant_pair(D0, bl, bd, reg[D0][0], reg[D0][1], shift=delta, common=eps)
            rs.append(puiseux(ql, D0)[0][0] / puiseux(qd, D0)[0][0])
        return rel_range(rs), rs

    print("real r: median %.6f, relative range %.3e over %d bases" % (float(np.median([v["r"] for v in co.values()])),
                                                                     real, len(co)))
    out = dict(real_range=real, judged={}, minus={}, extra={}, robust={})
    for d in DELTAS:
        rg, rs = rng_of(d)
        out["judged"][d] = dict(range=rg, r=rs)
        rob = min(rng_of(d, e)[0] for e in EPS)
        out["robust"][d] = rob
        line = "  delta %-7g range %.3e (r %.6f to %.6f) | robust over the common shift %.3e" % (d, rg, min(rs), max(rs), rob)
        if d:
            out["minus"][d] = rng_of(-d)[0]
            line += " | -delta %.3e" % out["minus"][d]
        print(line)
    for d in DELTAS_X:
        out["extra"][d] = rng_of(d)[0]
        print("  delta %-7g range %.3e (reported)" % (d, out["extra"][d]))
    rgs = [out["judged"][d]["range"] for d in DELTAS]
    mono = all(b > a for a, b in zip(rgs, rgs[1:]))
    floor = rgs[0] < real
    excl = [d for d in DELTAS if d and out["judged"][d]["range"] > real]
    allowed = [d for d in DELTAS if out["judged"][d]["range"] <= real]
    if mono and floor and excl:
        bound = min(excl)
        res = "|D*(lam_g) - D*(Delta)| < %g: the planted range exceeds the real %.2e from delta %g on; %g is allowed" % (
            bound, real, bound, max(d for d in allowed if d < bound))
    else:
        bound = None
        res = "not bounded (%s)" % ", ".join(x for x, ok in (("the range does not grow strictly", mono),
                                                             ("the delta = 0 range is not below the real one", floor),
                                                             ("no delta exceeds the real range", bool(excl))) if not ok)
    rob_excl = [d for d in DELTAS if d and out["robust"][d] > real]
    rob = min(rob_excl) if rob_excl else None
    print("monotonic: %s; floor %.3e below the real %.3e: %s" % (mono, rgs[0], real, floor))
    print("calibrate:", res)
    print("robust over the common shift (reported): %s" % ("excluded from delta %g on" % rob if rob else "nothing excluded"))
    out.update(monotonic=mono, floor=floor, bound=bound, robust_bound=rob, verdict=res)
    if "json" in args:
        json.dump(out, open(args["json"], "w"), indent=1)
    return 0 if mono and floor else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = dict(coeffs=cmd_coeffs, planted=cmd_planted, calibrate=cmd_calibrate).get(cmd)
    if fn is None or not pos:
        print(__doc__)
        sys.exit(2)
    sys.exit(fn(pos[0]))
