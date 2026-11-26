#!/usr/bin/env python3
"""#1095: Track C's 1/n series (track_c.py: the same variables, degrees, D_min levels, acceptance and estimator)
refitted with #1088's form-S direct solves added, so that Δ at D 2828-10^4 and Δ∞ come from a short extrapolation.

Direct points: every `direct_s.jsonl` record under the raw folders given (I1088-*), one per D, chosen as
export_delta_all.py's direct rows are: the full system, then a partial release, then the held system, each by the lowest
check residual. Records with "valid": false or "method": "fine" are skipped (the fine mesh converged to a wrong Delta,
#1088). Sigma, written before any fit is read:
- full system: max(err, 1e-13), err export_delta_all.direct_err's (its largest recorded term; 1e-13 keeps the dd
  points' own weight dominant at low D);
- held system: max(err, 1e-11), err the first-order held bias, the record's own or carried over from another record
  at the same D or the nearest lower D (export_delta_all.direct_err). That estimate exceeded the measured held->full
  change at D 300 (+1.4e-10 against -9.7e-12) and at D 1000 (+5.1e-10 against +2.0e-10), so it is a conservative bar,
  not a measured error. (Before 2026-11-26 a held record without its own estimate took 1e-11, which understated the
  held bias at D 2000-8000 by a factor of about 20.)
- a record with no err (a partial release, or a held solve with no estimate in reach) is not a point; a notice says so.

Planted truth first (`planted`), through the same code: truth = track_c.readme_formula on the real grid with the dd/qd/
chain points' own Gaussian noise, and at the direct points
  P0: Gaussian noise of their sigma; P5: +sigma (a systematic held bias of one sign); P6: -sigma;
pass when |fit - truth| <= E at D 10^4 and D inf in P0, P5 and P6 (E = max(h, s), as track_c); F = the largest
|fit - truth| / E, floored at 1, multiplies E in `real`. Written before the run; a planted failure stops the quote.

    cd hp && python3 track_c_direct.py planted RAW/I1088-dh370 RAW/I1088-dh610 RAW/I1088-mac-a [--out=FILE]
    cd hp && python3 track_c_direct.py real RAW/I1088-dh370 ... [--F=x] [--out=FILE]
    --mutate=shift moves the largest-D direct point by 1e-8 in `real`; the check must fire.
The check of `real`: every direct point's residual in the central fit at D 10^4, |fit - data| / sigma, is at most 3;
otherwise the run exits 1 and nothing is quoted. (The first control rule, "fits rejected or the estimate moves by more
than E", did not fire on the shifted point: one point among 1355 neither fails chi^2/dof nor moves the median fit, while
its residual reads -18.8 sigma; so the residual is the check.)
"""
import glob
import json
import os
import random
import sys

import mpmath as mp

sys.argv = [sys.argv[0], "real"] + sys.argv[1:]           # track_c parses its own args from argv[2:]
import track_c as T  # noqa: E402

_dps = mp.mp.dps
import export_delta_all as X  # noqa: E402  (the direct rows' err rule; its import sets mp.dps, restored below)
mp.mp.dps = _dps

TGT = T.TARGETS if "targets" in T.args else [2828.4271247461903, 4000.0, 5656.854249492381, 8000.0, 10000.0]
MUT = T.args.get("mutate")


def direct_points(folders):
    """{D: (Δ mpf, sigma, kind)} of the direct solves, and {D: source}"""
    recs = []
    for fo in folders:
        for f in sorted(glob.glob(os.path.join(fo, "**", "direct_s.jsonl"), recursive=True)):
            for n, line in enumerate(open(f), 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("valid") is False or r.get("method") == "fine":
                    continue                       # fine-mesh records: converged to a wrong Delta (#1088), never data
                recs.append((f"{f}:{n}", r))
    best = {}
    for src, r in recs:
        e, ek = X.direct_err(r, src, recs)
        if not e:
            print(f"direct: {src} (D {r['D_str']}, {r['system']}) left out: {ek}")
            continue
        D = round(float(r["D_str"]), 6)
        full = r["system"] == "full"
        s = max(float(e), 1e-13 if full else 1e-11)
        tier = 0 if full else 2 if r["system"] == "held" else 1
        key = (tier, r.get("F_check") if r.get("F_check") is not None else r.get("F_solve") or 1.0)
        if D not in best or key < best[D][0]:
            v = mp.fsum(mp.mpf(repr(float(x))) for x in r["Delta_parts"])
            best[D] = (key, (v, s, "direct_full" if full else "direct_held"), src)
    return {D: b[1] for D, b in best.items()}, {D: b[2] for D, b in best.items()}


def data_with(direct):
    d = T.load_real()
    d.update(direct)
    return d


def planted(direct, real):
    rng = random.Random(7)
    worst = 0.0
    rows = {}
    for kind in ("P0", "P5", "P6"):
        data = {}
        for d, (v, s, k) in real.items():
            t = T.readme_formula(mp.mpf(d))
            if k.startswith("direct"):
                noise = {"P0": rng.gauss(0, s), "P5": s, "P6": -s}[kind]
            else:
                noise = rng.gauss(0, s)
            data[d] = (t + mp.mpf(noise), s, k)
        fits = T.all_fits(data)
        D0, used = T.levels(fits)
        rec = {}
        for D in (10000.0, "inf"):
            est = T.estimate(fits, D)
            tv = T.readme_formula(mp.mpf(D)) if D != "inf" else T.readme_formula(mp.mpf("1e40"))
            err = abs(float(est["value"] - T.arb(mp.nstr(tv, 70))))
            ratio = err / est["E"] if est["E"] > 0 else float("inf")
            worst = max(worst, ratio)
            rec[str(D)] = dict(err=err, E=est["E"], h=est["h"], s=est["s"], ratio=ratio, n=est["n"])
            print("%s D %-7s: |fit - truth| %.2e  E %.2e  (h %.2e, s %.2e, %d fits, D0 %s)  ratio %.2f" % (
                kind, D, err, est["E"], est["h"], est["s"], est["n"], D0, ratio), flush=True)
        rows[kind] = rec
    F = max(1.0, worst)
    ok = worst <= 1.0
    print("planted: largest |fit - truth| / E = %.2f -> F = %.2f; %s" % (worst, F, "PASS" if ok else
          "FAIL (the estimator's E understates the error; the real result is quoted with F, and the failure reported)"))
    return F, ok, rows


def real_fit(direct, real, Fc):
    if MUT == "shift":
        Dm = max(direct)
        v, s, k = real[Dm]
        real[Dm] = (v + mp.mpf("1e-8"), s, k)
        print("MUTATION: the D %g direct point moved by +1e-8" % Dm)
    fits = T.all_fits(real)
    D0, used = T.levels(fits)
    print("real: %d points (%d direct), %d of %d fits accepted; D0 %s, %d used" % (
        len(real), len(direct), sum(f["ok"] for f in fits), len(fits), D0, len(used)), flush=True)
    out = {}
    for D in [1000.0, 1414.0, 2000.0] + TGT + ["inf"]:
        est = T.estimate(fits, D, Fc)
        if est is None:
            print("  D %s: no estimate (fewer than 5 accepted fits)" % D)
            continue
        out[str(D)] = dict(Delta=est["value"].mid().str(20, radius=False), E=est["E"], h=est["h"], s=est["s"], n=est["n"])
        print("  D %-18s: Delta = %s  +- %.2e  (h %.2e, s %.2e, %d fits)" % (
            D, est["value"].mid().str(18, radius=False), est["E"], est["h"], est["s"], est["n"]), flush=True)
    cen = T.estimate(fits, 10000.0, Fc)["central"]
    rd = {d: cen["resid"][d] for d in direct if d in cen["resid"]}
    print("  central fit (%s K%d D%d): residuals at the direct points, (fit - data)/sigma: %s" % (
        cen["var"], cen["K"], cen["dmin"], ", ".join("%g: %+.2f" % (d, rd[d]) for d in sorted(rd))))
    worst = max(abs(v) for v in rd.values())
    out["check"] = dict(max_abs_resid=worst, ok=worst <= 3.0)
    print("  check: largest |residual| at a direct point %.2f sigma: %s" % (
        worst, "PASS" if worst <= 3.0 else "FAIL (a direct point disagrees with the series; nothing is quoted)"))
    return out


def main():
    cmd = sys.argv[2] if len(sys.argv) > 2 else ""
    folders = [a for a in sys.argv[3:] if not a.startswith("--")]
    if cmd not in ("planted", "real") or not folders:
        raise SystemExit(__doc__)
    direct, files = direct_points(folders)
    print("direct points: " + ", ".join("D %g %s %s sigma %.1e" % (d, real_k, mp.nstr(v, 17), s)
                                         for d, (v, s, real_k) in sorted(direct.items())), flush=True)
    real = data_with(direct)
    if cmd == "planted":
        F, ok, rows = planted(direct, real)
        res = dict(F=F, ok=ok, rows=rows)
    else:
        res = real_fit(direct, real, float(T.args.get("F", "1")))
        if not res["check"]["ok"]:
            if "out" in T.args:
                json.dump(res, open(T.args["out"], "w"), indent=1, default=str)
            sys.exit(1)
    res["direct"] = {str(d): dict(Delta=mp.nstr(v, 20), sigma=s, kind=k, file=files[d]) for d, (v, s, k) in direct.items()}
    if "out" in T.args:
        json.dump(res, open(T.args["out"], "w"), indent=1)


if __name__ == "__main__":
    main()
