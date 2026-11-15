#!/usr/bin/env python3
"""Does the fold move with the plan? D* on #795's re-planned series by #829's two approximant families (#833).

    cd hp && python3 fold_plan.py planted RAWDIR [--json=OUT]
    cd hp && python3 fold_plan.py real RAWDIR --planted=PLANTED.json [--json=OUT]

RAWDIR is a clone of the raw-data repository. The kept plan's series are I603-win2/chain_nsr/jet_pin3_D*.npz (#603's
plan; #776's reproduction I776-dh370/repro equals them), the new plan's I795-dh370/newplan/jet_pin3_D*.npz (#795,
`anm_run.py --keep_plan 0` at #603's stored states, which satisfy the new plan's equations to max|F| 4.7e-36 and
5.5e-36). Both plans have the bases D 259.2582 and 259.4928 only, with the same base states.

Configurations, the same on both plans (hp/fold_mirror.py's forms, M1 the differential approximant continued around
D*, M2 the quadratic Pade):
  P   the pair: anchor 259.4928, partner 259.2582, at fold_mirror's two-base degree sets DEG2 (least squares)
  Sa  259.4928 alone, at the one-base sets DA_DEG1 (M1) and QP_DEG1 (M2)
  Sb  259.2582 alone, the same
Each gives D*, Delta*, a1 (and M1's mu) per approximant; a method's value is the median over its valid approximants.

planted (registered on #833 before the real D* was computed): #829's plant (fold_mirror.Plant, noise 1e-30) at each
configuration, D*'s error per method; the move control, the plant with D* + 0.010 against the plant, must read
"moves"; the no-move control, the plant with the contamination doubled (0.6 a1) against the plant, must read "does not
move". The main configuration is the one with the smallest plant D* error at which both methods are valid and both
controls read as required.

The error of a plan's D* in a configuration: the largest of |M1 - M2|, each method's half range over its approximants
and the plant's |D* error| there (spreads and plant errors, not bounds). The combined error of a difference is the sum
of the two errors. Verdict (the issue's): "moves with the plan" when |D*(new) - D*(kept)| > 100 x combined, "does not
move" when below 3 x, otherwise undetermined.

real: both plans at the three configurations; the verdict in the main configuration of PLANTED.json. Each plan's own
measured states (the shared base states): its 259.2582 series summed at 259.4928 (25 terms, the last term as the
truncation estimate), and Sb's + sheet at 259.4928 and Sa's at 259.2582 by both methods, against the state's Delta,
bound 1e-8 (#829's).
"""
import json
import os
import statistics
import sys

import mpmath as mp

import fold_mirror as FM

BASES = ("259.4928", "259.2582")
MOVE, CONT2 = 0.010, 0.6
TOL_P = 1e-8
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:] if a.startswith("--"))
pos = [a for a in sys.argv[2:] if not a.startswith("--")]
CONFIGS = {"P": (BASES[0], BASES[1]), "Sa": (BASES[0],), "Sb": (BASES[1],)}


def jet(raw, plan, D):
    sub = {"kept": ("I603-win2", "chain_nsr"), "new": ("I795-dh370", "newplan")}[plan]
    return FM.delta_series(os.path.join(raw, *sub, "jet_pin3_D%s.npz" % D))


def state_delta(raw, D):
    import numpy as np
    z = np.load(os.path.join(raw, "I603-win2", "chain_nsr", "anm_pin3_D%s.npz" % D))
    return mp.fsum(mp.mpf(float(z[k][-1])) for k in ("zh", "zl", "z2", "z3", "z4", "z5", "z6", "z7") if k in z.files)


def config(series, name):
    """per method: [(D*, Delta*, a1, mu)] over the valid approximants, and the + sheet at the other base"""
    keys = CONFIGS[name]
    inputs = [series[k] for k in keys]
    Db, c0 = inputs[0][0], inputs[0][1][0]
    other = series[BASES[1] if keys[0] == BASES[0] else BASES[0]][0]
    out = dict(M1=[], M2=[], plus_M1=[], plus_M2=[], why=[])
    two = len(keys) == 2
    for deg in (FM.DA_DEG2 if two else FM.DA_DEG1):
        try:
            polys, _ = FM.forms(inputs, deg, "DA")
            a = FM.DApprox(polys, Db, c0)
            if a.why:
                out["why"].append("M1 %s: %s" % (deg, a.why))
                continue
            cf = a.coeffs()
            out["M1"].append((float(cf[0]), float(mp.re(cf[1])), float(mp.re(cf[2])), float(mp.re(cf[6]))))
            if not two:
                out["plus_M1"].append(mp.re(a.walk(0, mp.mpc(c0), other - Db)[1]))
        except (RuntimeError, ZeroDivisionError, ValueError) as ex:
            out["why"].append("M1 %s: %s" % (deg, ex))
    for deg in (FM.QP_DEG2 if two else FM.QP_DEG1):
        try:
            polys, _ = FM.forms(inputs, deg, "QP")
            q = FM.QPade(polys, Db, c0)
            cf = q.coeffs() if q.ok else None
            if cf is None:
                out["why"].append("M2 %s: %s" % (deg, "no branch point near D*" if q.ok else "discriminant <= 0"))
                continue
            out["M2"].append((float(cf[0]), float(cf[1]), float(cf[2]), None))
            if not two:
                v = q.value(other, 1)
                if v is not None:
                    out["plus_M2"].append(v)
        except (ZeroDivisionError, ValueError) as ex:
            out["why"].append("M2 %s: %s" % (deg, ex))
    return out


def summary(res):
    """per method: median D*, Delta*, a1, mu and the half range of D*; None when a method has no valid approximant"""
    s = {}
    for m in ("M1", "M2"):
        v = res[m]
        if not v:
            s[m] = None
            continue
        ds = [x[0] for x in v]
        s[m] = dict(n=len(v), Dstar=statistics.median(ds), half=(max(ds) - min(ds)) / 2,
                    Delta=statistics.median(x[1] for x in v), a1=statistics.median(x[2] for x in v),
                    mu=statistics.median(x[3] for x in v) if m == "M1" else None)
    return s


def value_err(s, plant_err):
    """(D*, error): the mean of M1 and M2, the error the largest of |M1 - M2|, the half ranges and the plant's error"""
    if s["M1"] is None or s["M2"] is None:
        return None, None
    v = (s["M1"]["Dstar"] + s["M2"]["Dstar"]) / 2
    return v, max(abs(s["M1"]["Dstar"] - s["M2"]["Dstar"]), s["M1"]["half"], s["M2"]["half"], plant_err)


def verdict(d, comb):
    if comb is None:
        return "undetermined (a method has no valid approximant)"
    return "moves with the plan" if abs(d) > 100 * comb else "does not move" if abs(d) < 3 * comb else "undetermined"


def plant_series(raw, plant, seed0):
    """{base: (D_b, the plant's c_0..c_24)}, the same noise seed per base for every plant"""
    out = {}
    for i, D in enumerate(BASES):
        Db = jet(raw, "kept", D)[0]
        out[D] = (Db, plant.series(Db, seed0 + i))
    return out


def cmd_planted(raw):
    mp.mp.dps = FM.DPS
    plants = dict(base=FM.Plant(), move=FM.Plant(shift=MOVE), cont2=FM.Plant(cont=CONT2))
    out = {}
    for name in CONFIGS:
        r = {}
        for pk, pl in plants.items():
            s = summary(config(plant_series(raw, pl, 833), name))
            r[pk] = s
        truth = float(FM.DSTAR)
        err = max((abs(r["base"][m]["Dstar"] - truth) for m in ("M1", "M2") if r["base"][m]), default=None)
        valid = all(r[pk][m] is not None for pk in plants for m in ("M1", "M2"))
        line = dict(valid=valid, plant_err=err, summary=r)
        if valid:
            vb, eb = value_err(r["base"], err)
            vm, em = value_err(r["move"], err)
            vc, ec = value_err(r["cont2"], err)
            dm, dc = vm - vb, vc - vb
            line.update(move=dict(shift=dm, comb=eb + em, verdict=verdict(dm, eb + em),
                                  recovered=abs(dm - MOVE) <= eb + em),
                        cont2=dict(shift=dc, comb=eb + ec, verdict=verdict(dc, eb + ec)))
            line["controls_ok"] = (line["move"]["verdict"] == "moves with the plan" and line["move"]["recovered"]
                                   and line["cont2"]["verdict"] == "does not move")
        else:
            line["controls_ok"] = False
        out[name] = line
        print("%-2s plant D* error: %s | M1 %s, M2 %s approximants" % (
            name, "%.1e" % err if err is not None else "-",
            r["base"]["M1"]["n"] if r["base"]["M1"] else 0, r["base"]["M2"]["n"] if r["base"]["M2"] else 0), flush=True)
        if valid:
            print("     move control: recovered %+.6f (planted %+.3f, combined error %.1e): %s%s" % (
                line["move"]["shift"], MOVE, line["move"]["comb"], line["move"]["verdict"],
                "" if line["move"]["recovered"] else ", NOT within the error"))
            print("     no-move control (contamination doubled): %+.1e (combined error %.1e): %s" % (
                line["cont2"]["shift"], line["cont2"]["comb"], line["cont2"]["verdict"]))
            print("     controls %s" % ("as required" if line["controls_ok"] else "NOT as required"))
    ok = [n for n in CONFIGS if out[n]["controls_ok"]]
    main = min(ok, key=lambda n: out[n]["plant_err"]) if ok else None
    print("main configuration: %s" % (main or "none (no configuration passes both controls)"))
    if "json" in args:
        json.dump(dict(configs=out, main=main), open(args["json"], "w"), indent=1, default=float)
    return 0 if main else 1


def cmd_real(raw):
    mp.mp.dps = FM.DPS
    pl = json.load(open(args["planted"]))
    main = pl["main"]
    series = {plan: {D: jet(raw, plan, D) for D in BASES} for plan in ("kept", "new")}
    states = {D: state_delta(raw, D) for D in BASES}
    out = dict(main=main, configs={}, states={})
    print("each plan's own states (shared): the 259.2582 series summed at 259.4928, and the single-base + sheets")
    for plan in ("kept", "new"):
        Db, c = series[plan][BASES[1]]
        h = series[plan][BASES[0]][0] - Db
        tot = mp.fsum(x * h ** k for k, x in enumerate(c))
        e_sum, last = float(abs(tot - states[BASES[0]])), float(abs(c[-1] * h ** (len(c) - 1)))
        print("  %-4s direct sum: |sum - Delta(259.4928)| %.1e (last term %.1e)" % (plan, e_sum, last))
        out["states"][plan] = dict(direct=e_sum, last=last)
    for name in CONFIGS:
        res = {plan: config(series[plan], name) for plan in ("kept", "new")}
        sm = {plan: summary(res[plan]) for plan in res}
        perr = pl["configs"][name]["plant_err"] or 0.0
        ve = {plan: value_err(sm[plan], perr) for plan in sm}
        line = dict(summary=sm, why={p: res[p]["why"] for p in res})
        if len(CONFIGS[name]) == 1:
            other = BASES[1] if CONFIGS[name][0] == BASES[0] else BASES[0]
            for plan in res:
                for m in ("M1", "M2"):
                    vals = res[plan]["plus_" + m]
                    line.setdefault("plus", {}).setdefault(plan, {})[m] = (
                        float(statistics.median(abs(v - states[other]) for v in vals)) if vals else None)
        if all(ve[p][0] is not None for p in ve):
            d = ve["new"][0] - ve["kept"][0]
            comb = ve["new"][1] + ve["kept"][1]
            line.update(kept=ve["kept"], new=ve["new"], diff=d, comb=comb, verdict=verdict(d, comb))
        out["configs"][name] = line
        print("%-2s%s" % (name, " (main)" if name == main else ""))
        for plan in ("kept", "new"):
            s = sm[plan]
            f = lambda m: ("D* %.11f (+-%.1e, %d) Delta* %.12f a1 %.9f%s" % (   # noqa: E731
                s[m]["Dstar"], s[m]["half"], s[m]["n"], s[m]["Delta"], s[m]["a1"],
                " mu %.9f" % s[m]["mu"] if s[m]["mu"] is not None else "")) if s[m] else "none valid"
            print("   %-4s M1 %s" % (plan, f("M1")))
            print("        M2 %s" % f("M2"))
            if "plus" in line:
                print("        + sheet at the other base: M1 %s, M2 %s" % tuple(
                    "%.1e" % line["plus"][plan][m] if line["plus"][plan][m] is not None else "-" for m in ("M1", "M2")))
        if "diff" in line:
            print("   D*(new) - D*(kept) = %+.3e, combined error %.1e (kept %.1e, new %.1e): %s" % (
                line["diff"], line["comb"], ve["kept"][1], ve["new"][1], line["verdict"]))
    if main and "diff" in out["configs"][main]:
        c = out["configs"][main]
        print("verdict (main configuration %s): %s, D*(new) - D*(kept) = %+.3e against combined error %.1e" % (
            main, c["verdict"], c["diff"], c["comb"]))
    if "json" in args:
        json.dump(out, open(args["json"], "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = dict(planted=cmd_planted, real=cmd_real).get(cmd)
    if fn is None or not pos:
        print(__doc__)
        sys.exit(2)
    sys.exit(fn(pos[0]))
