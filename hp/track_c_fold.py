#!/usr/bin/env python3
"""Could track C miss a fold of the continuum near D 260? The largest planted square root its fit and checks pass (#835).

    cd hp && python3 track_c_fold.py run [--out=FILE] [--Ds=259,260,...]

The plant (#835): Delta_plant(D) = Delta_real(D) + a g(D) on track C's real grid and sigma (track_c.load_real), with
g = sqrt(D*_p - D) - r(D) and r the weighted least-squares fit of sqrt(D*_p - D) over all points by the form of track
C's central fit (1/(D-1), K 20). Every plant goes through track_c's own all_fits, levels and estimate.

Flagged (registered on #835 before any bound was computed): track C gives no estimate or its D0 is no longer 10, which
happens when fewer than 5 of the D_min-10 fits keep chi^2/dof <= 4 (how P1 and P2 showed, by an exploding E).
|a|_max(D*_p), per sign: the largest |a| with every plant from 0 to a unflagged. Each fit's weighted residual is linear
in a (least squares), so its chi^2 is a quadratic in a and its acceptance thresholds are exact; the bound is verified
through the full code path, unflagged at 0.9 |a|_max and flagged at 1.1 |a|_max. Reported beside it: |a| at which the
first of the 135 fits changes acceptance, and at 0.9 |a|_max the shift of Delta(250), Delta(300), Delta(400) and
Delta_inf against E, E, and the central fit's largest residual on D 150-245 (the real data's 2.76 sigma).
Checks: a = 0 reproduces the stored series rows (data/delta_series.csv) at D 300, 400 and the limit to their E;
a = 0.035 at D*_p 259.903 (the chain's own fold, #829) must be flagged.
"""
import csv
import json
import math
import os
import sys

import mpmath as mp
from flint import arb

import track_c as T

HERE = os.path.dirname(os.path.abspath(__file__))
args = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[2:])
DSP = [259, 259.5, 259.903, 260, 260.5, 261, 262, 263, 265, 267.5, 270]
A_CHAIN, DS_CHAIN = 0.035, 259.903
REG_FORM = ("1/(D-1)", 20)
D0_REAL, NMIN = 10, 5
TGT = [250, 300, 400, "inf"]


def g_of(real, Dsp):
    """g(D) = sqrt(D*_p - D) - r(D) on the real grid"""
    sq = {d: mp.sqrt(mp.mpf(Dsp) - mp.mpf(d)) for d in real}
    ft = T.fit({d: (sq[d], real[d][1], real[d][2]) for d in real}, REG_FORM[0], REG_FORM[1], min(real))
    return {d: sq[d] - mp.mpf(ft["ev"](d).mid().str(70, radius=False)) for d in real}


def planted(real, g, a):
    a = mp.mpf(a)
    return {d: (v + a * g[d], s, k) for d, (v, s, k) in real.items()}


def run(data):
    """track C on data: fits, D0, the estimates at TGT, the central fit's residuals on 150-245"""
    fits = T.all_fits(data)
    D0, used = T.levels(fits)
    est = {str(D): T.estimate(fits, D) for D in TGT}
    out = dict(D0=D0, nused=len(used), naccepted=sum(f["ok"] for f in fits),
               n10=sum(1 for f in fits if f["ok"] and f["dmin"] == 10), flagged=(D0 != D0_REAL))
    if est["300"] is not None:
        cen = est["300"]["central"]
        rr = [abs(v) for d, v in cen["resid"].items() if 150 <= d <= T.DMAX + 1e-9]
        out["resid_max"] = max(rr)
        out["est"] = {k: dict(value=e["value"].mid().str(30, radius=False), E=e["E"]) for k, e in est.items()}
    return out, fits


def thresholds(fits0, fitsg):
    """per fit: (var, K, dmin, accepted at 0, [the a where its acceptance changes])"""
    out = []
    for f0, fg in zip(fits0, fitsg):
        assert (f0["var"], f0["K"], f0["dmin"]) == (fg["var"], fg["K"], fg["dmin"])
        Ds = list(f0["resid"])
        A = sum(fg["resid"][d] ** 2 for d in Ds)
        B = sum(f0["resid"][d] * fg["resid"][d] for d in Ds)
        C = sum(f0["resid"][d] ** 2 for d in Ds)
        cap = T.CHI_MAX * max(1, f0["n"] - f0["K"] - 1)
        disc = B * B - A * (C - cap)
        roots = sorted(((-B - math.sqrt(disc)) / A, (-B + math.sqrt(disc)) / A)) if disc > 0 and A > 0 else []
        out.append(dict(var=f0["var"], K=f0["K"], dmin=f0["dmin"], ok0=C <= cap, roots=roots, A=A, B=B, C=C, cap=cap))
    return out


def accepted(t, a):
    return t["A"] * a * a + 2 * t["B"] * a + t["C"] <= t["cap"]


def bound(th, sign):
    """the largest |a| (sign +1 or -1) with at least NMIN D_min-10 fits accepted on all of [0, a]; and |a| at which
    the first of all the fits changes acceptance"""
    pts = sorted(abs(r) for t in th for r in t["roots"] if r * sign > 0)
    first_any = pts[0] if pts else math.inf
    p10 = sorted(abs(r) for t in th if t["dmin"] == D0_REAL for r in t["roots"] if r * sign > 0)
    amax = math.inf
    for p in p10:
        a_after = sign * p * (1 + 1e-9)
        if sum(1 for t in th if t["dmin"] == D0_REAL and accepted(t, a_after)) < NMIN:
            amax = p
            break
    setters = [("%s K%d" % (t["var"], t["K"])) for t in th if t["dmin"] == D0_REAL and t["ok0"]
               and not accepted(t, sign * amax * (1 + 1e-9))] if amax < math.inf else []
    return amax, first_any, setters


def stored():
    rows = {r["D"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", "delta_series.csv")))}
    pick = {"300": rows["300.0"], "400": rows["400.0"], "inf": rows["inf"]}
    return {k: (mp.mpf(r["Delta_q0"]) + mp.mpf(r["Delta_q1"]), float(r["E"])) for k, r in pick.items()}


def cmd_run():
    real = T.load_real()
    log = []

    def say(s):
        print(s, flush=True)
        log.append(s)
    base, fits0 = run(real)
    st = stored()
    rep = {k: abs(mp.mpf(base["est"][k]["value"]) - st[k][0]) for k in st}
    ok0 = all(rep[k] <= st[k][1] for k in st)
    say("a = 0: D0 %s, %d accepted, %d at D_min 10, %d used; against the stored rows: %s -> %s" % (
        base["D0"], base["naccepted"], base["n10"], base["nused"],
        ", ".join("%s %.1e (E %.2e)" % (k, float(rep[k]), st[k][1]) for k in st), "reproduced" if ok0 else "NOT reproduced"))
    out = dict(baseline=base, reproduced=ok0, reproduce_diff={k: float(v) for k, v in rep.items()}, bounds={})
    dsp = [float(x) for x in args["Ds"].split(",")] if "Ds" in args else DSP
    for Dsp in dsp:
        g = g_of(real, Dsp)
        _, fitsg = run({d: (g[d], real[d][1], real[d][2]) for d in real})
        th = thresholds(fits0, fitsg)
        rec = {}
        for sign in (1, -1):
            amax, first_any, setters = bound(th, sign)
            ver = {}
            if amax == math.inf:
                rec["+" if sign > 0 else "-"] = dict(amax=amax, first_any=first_any, setters=[], verified=False)
                say("D*_p %g a %s: never flagged on the fits' thresholds" % (Dsp, "+" if sign > 0 else "-"))
                continue
            for fac in (0.9, 1.1):
                r_, _ = run(planted(real, g, sign * fac * amax))
                ver[str(fac)] = r_
            v9 = ver["0.9"]
            shifts = {k: abs(float(mp.mpf(v9["est"][k]["value"]) - mp.mpf(base["est"][k]["value"]))) / base["est"][k]["E"]
                      for k in v9["est"]} if "est" in v9 else None
            good = (not ver["0.9"]["flagged"]) and ver["1.1"]["flagged"]
            rec["+" if sign > 0 else "-"] = dict(amax=amax, first_any=first_any, setters=setters, verified=good,
                                                 at09=ver["0.9"], at11=ver["1.1"], shift_over_E=shifts)
            say("D*_p %-7g a %s: |a|_max %.2e (set by the D_min-10 fits %s), first change of any fit %.2e; "
                "0.9x: D0 %s, %d at D_min 10, resid %.2f, shift/E %s; 1.1x: D0 %s, %d at D_min 10 -> %s" % (
                    Dsp, "+" if sign > 0 else "-", amax, ", ".join(setters), first_any, v9["D0"], v9["n10"],
                    v9.get("resid_max", float("nan")),
                    ", ".join("%s %.2f" % (k, v) for k, v in shifts.items()) if shifts else "-",
                    ver["1.1"]["D0"], ver["1.1"]["n10"], "verified" if good else "NOT verified"))
        out["bounds"][str(Dsp)] = rec
        if abs(Dsp - DS_CHAIN) < 1e-9:
            r_, _ = run(planted(real, g, A_CHAIN))
            out["chain_fold"] = r_
            say("a = %g at D*_p %g (the chain's fold): D0 %s, %d of 135 accepted, %d at D_min 10 -> %s" % (
                A_CHAIN, DS_CHAIN, r_["D0"], r_["naccepted"], r_["n10"], "flagged" if r_["flagged"] else "NOT flagged"))
    if "out" in args:
        json.dump(out, open(args["out"], "w"), indent=1, default=float)
    passed = ok0 and out.get("chain_fold", {}).get("flagged", False) and all(
        r["verified"] for rec in out["bounds"].values() for r in rec.values())
    say("checks: a = 0 %s; the chain's fold %s; bounds %s" % (
        "reproduces" if ok0 else "FAILS", "flagged" if out.get("chain_fold", {}).get("flagged") else "NOT flagged",
        "all verified" if all(r["verified"] for rec in out["bounds"].values() for r in rec.values()) else "NOT all verified"))
    return 0 if passed else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        sys.exit(cmd_run())
    raise SystemExit(__doc__)
