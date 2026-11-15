#!/usr/bin/env python3
"""The two folds' separation from free-D* fits, planted (#822): does #821's |D*(lam_g) - D*(Delta)| <= 8.3e-7 bound
the separation?

    cd hp && python3 fold_separation.py planted RAWDIR [--json=OUT]   the plants (exits 1 on a fail)
    cd hp && python3 fold_separation.py real RAWDIR --planted=P.json [--json=OUT]   the real separation and the bound

The fit is hp/near_amp.py's (orders 12-24, the fold's three terms and #782's near singularities as free amplitudes),
with D* free for each series at each base: the largest relative residual minimized over D* in [259.9020, 259.9045]
(bounded Brent, xatol 1e-9), as `near_amp.py posthoc` fits it. The bases are #818's 26.

planted (the issue's design, fixed before the real separations were read by this script): at each base, lam_g's series
with its fold at 259.9033 + eps + delta and Delta's at 259.9033 + eps, eps = 1e-6 and 1e-5 (a common offset) and
delta = 0, 1e-7, 1e-6, 1e-5. Each series from fold_cancel's median b, with its regular part (lam_g:
lam_singularity.regular; Delta: the same divided by r = 0.020646) and relative noise 1e-12. The recovered separation is
D*(lam_g) - D*(Delta). A base is usable when, at both eps, delta = 1e-6 and 1e-5 are recovered to 30% and delta = 0
reads below 1e-7 in magnitude. planted passes when 4 or more bases are usable; exit 1 otherwise.
Reported, not judged: -delta, and the near singularities planted at 0.01 of the fold's amplitude. --b7=X adds an
unmodelled fourth fold term X b1 (D* - D)^(7/2); it was not run, since a probe found that the minimizer alone leaves
residuals of the real fit's size (1.6e-8 to 6.4e-7 on a plant with exactly the fit's fold terms, against about 1e-11 at
the true D*).
First run (2026-11-15, #822): PASS, 23 of 26 bases usable. The minimizer misplaces each series' D* by up to 3.5e-6, but
the error is common to both and cancels in the separation. The real S is 8.29e-7 (at 259.4928), just above the
smallest recovered 1e-6 separation (8.07e-7), so the bound is 1e-5.

real: |D*(lam_g) - D*(Delta)| at the usable bases, its largest value S. The bound is |D*(lam_g) - D*(Delta)| < delta0,
delta0 the smallest planted delta (1e-7, 1e-6, 1e-5) whose smallest recovered separation over the usable bases and
both eps exceeds S. If none does, or planted fails, the separation is not bounded.
"""
import json
import os
import sys

import mpmath as mp
import numpy as np
from scipy.optimize import minimize_scalar

import fold_cancel as FC
import lam_singularity as LS
import near_amp as NA

HERE = os.path.dirname(os.path.abspath(__file__))
D0STAR = mp.mpf("259.9033")
EPS = (1e-6, 1e-5)
DELTAS = (0.0, 1e-7, 1e-6, 1e-5)
TOL = 0.3
FLOOR = 1e-7
R_MED = 0.020646
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:] if a.startswith("--"))
pos = [a for a in sys.argv[2:] if not a.startswith("--")]


def free_fit(c, D0):
    """near_amp.fit with D* free: (D*, the fit at D*)"""
    def res(Ds):
        NA.DSTAR = mp.mpf(Ds)
        return NA.fit(c, D0)["res"]
    m = minimize_scalar(res, bounds=(259.9020, 259.9045), method="bounded", options=dict(xatol=1e-9))
    NA.DSTAR = mp.mpf(m.x)
    f = NA.fit(c, D0)
    NA.DSTAR = D0STAR
    return float(m.x), f


def series_plant(D0, b, fold, reg, seed, near=0.0, b7=0.0):
    """c_1..c_24: sum_j b_j (fold - D)^(j/2) (+ b7 (fold - D)^(7/2)), the near singularities at near b1 sqrt(s), the
    regular part, relative noise 1e-12"""
    rng = np.random.default_rng(seed)
    s = mp.mpf(fold) - mp.mpf(D0)
    amp = near * b[0] * mp.sqrt(s)
    out = []
    for k in range(1, 25):
        v = sum(bj * FC.basis(j, k, s) for bj, j in zip(b, FC.JS)) + (b7 * FC.basis(7, k, s) if b7 else 0)
        if near:
            f2, f3 = NA.near_funcs(D0, k)
            v += amp * (f2 + 2 * f3.real)
        out.append((v + reg[k - 1]) * (1 + 1e-12 * rng.standard_normal()))
    return out


def medians():
    co = {float(k): v for k, v in json.load(open(os.path.join(HERE, "data", "fold_cancel_coeffs.json"))).items()}
    bl = [float(np.median([v["lam"][i] for v in co.values()])) for i in range(3)]
    bd = [float(np.median([v["delta"][i] for v in co.values()])) for i in range(3)]
    return sorted(co), bl, bd


B7 = 0.0    # set by --b7=X for the reported misfit variant (in units of each series' b1)


def separations(D0, bl, bd, reg_l, reg_d, eps, deltas, near=0.0, b7=0.0):
    seed = 822 + int(round(D0 * 1e4)) % 1000
    Dd, fd = free_fit(series_plant(D0, bd, D0STAR + eps, reg_d, seed + 1, near, b7 * bd[0]), D0)
    out = {}
    for d in deltas:
        Dl, fl = free_fit(series_plant(D0, bl, D0STAR + eps + d, reg_l, seed, near, b7 * bl[0]), D0)
        out[d] = dict(sep=Dl - Dd, Dl=Dl, Dd=Dd, res_l=fl["res"], res_d=fd["res"])
    return out


def cmd_planted(raw):
    mp.mp.dps = 200
    bases, bl, bd = medians()
    b7 = float(args.get("b7", 0.0))
    out, usable = {}, []
    print("plants: lam_g's fold at 259.9033 + eps + delta, Delta's at 259.9033 + eps, D* free in each fit")
    for D0 in bases:
        with mp.workdps(60):
            reg_l = LS.regular(D0, 24)
        reg_d = [x / R_MED for x in reg_l]
        row, ok = {}, True
        for eps in EPS:
            sp = separations(D0, bl, bd, reg_l, reg_d, eps, DELTAS + (-1e-6, -1e-5))
            row[eps] = sp
            ok &= all(abs(sp[d]["sep"] - d) <= TOL * d for d in (1e-6, 1e-5)) and abs(sp[0.0]["sep"]) < FLOOR
        # reported: the near singularities at 0.01, at eps 1e-6
        row["near0.01"] = separations(D0, bl, bd, reg_l, reg_d, 1e-6, DELTAS, near=0.01)
        if b7:
            row["b7"] = separations(D0, bl, bd, reg_l, reg_d, 1e-6, DELTAS, b7=b7)
        out[D0] = dict(row, usable=bool(ok))
        if ok:
            usable.append(D0)
        f = lambda sp: " ".join("%+.2e" % sp[d]["sep"] for d in DELTAS)                       # noqa: E731
        print("  D0 %.4f  eps 1e-6: %s | eps 1e-5: %s | -1e-6/-1e-5: %+.2e %+.2e | near 0.01: %s%s | %s" % (
            D0, f(row[1e-6]), f(row[1e-5]), row[1e-6][-1e-6]["sep"], row[1e-6][-1e-5]["sep"], f(row["near0.01"]),
            (" | b7: %s (res %.1e)" % (f(row["b7"]), row["b7"][0.0]["res_l"])) if b7 else "",
            "usable" if ok else "not usable"), flush=True)
    passed = len(usable) >= LS.NCOUNT
    print("planted: %d of %d bases usable (%s): %s" % (len(usable), len(out), ", ".join("%.4f" % x for x in usable),
                                                      "PASS" if passed else "FAIL"))
    if "json" in args:
        json.dump(dict(usable=usable, b7=b7, bases={str(k): {str(e): v for e, v in r.items()} if isinstance(r, dict) else r
                                                     for k, r in out.items()}), open(args["json"], "w"), indent=1)
    return 0 if passed else 1


def cmd_real(raw):
    mp.mp.dps = 200
    pl = json.load(open(args["planted"]))
    usable = pl["usable"]
    bases, _, _ = medians()
    out = {}
    print("real: D* free for lam_g and Delta at each base")
    for D0 in bases:
        lam, dl = FC.series(raw, D0)
        Dl, fl = free_fit(lam[1:], D0)
        Dd, fd = free_fit(dl[1:], D0)
        use = any(abs(D0 - u) < 1e-9 for u in usable)
        out[D0] = dict(Dl=Dl, Dd=Dd, sep=Dl - Dd, res_l=fl["res"], res_d=fd["res"], usable=bool(use))
        print("  D0 %.4f  D*(lam_g) %.7f D*(Delta) %.7f separation %+.2e (res %.1e, %.1e)%s" % (
            D0, Dl, Dd, Dl - Dd, fl["res"], fd["res"], "" if use else " (not usable)"))
    S = max((abs(v["sep"]) for v in out.values() if v["usable"]), default=None)
    bound = None
    if pl and len(usable) >= LS.NCOUNT and S is not None:
        for d in (1e-7, 1e-6, 1e-5):
            rec = min(pl["bases"][str(u)][str(e)][str(d)]["sep"] for u in usable for e in EPS)
            print("  delta %g: smallest recovered separation at the usable bases %.3e against S %.3e" % (d, rec, S))
            if rec > S:
                bound = d
                break
    v = ("|D*(lam_g) - D*(Delta)| < %g (the real largest separation at the usable bases, %.2e, is below every "
         "recovered separation of that plant)" % (bound, S) if bound else "not bounded (S %s)" % ("%.2e" % S if S else "-"))
    print("real:", v)
    if "json" in args:
        json.dump(dict(S=S, bound=bound, verdict=v, bases=out), open(args["json"], "w"), indent=1)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = dict(planted=cmd_planted, real=cmd_real).get(cmd)
    if fn is None or not pos:
        print(__doc__)
        sys.exit(2)
    sys.exit(fn(pos[0]))
