#!/usr/bin/env python3
"""The large-D chains' steps as tables for the monitoring page, read from the chains' logs (anm_run.py output) in the
raw-data repository.

    cd hp && python3 chain_monitor.py RAWDIR OUTDIR [--selftest]

OUTDIR/steps.csv has one row per corrector attempt of every chain in CHAINS: the chain, the run (log file), the base D
and the target D, the step a, whether it was accepted, the Newton steps, the smallest lam, the start and final max|F|,
the first |dz|, the horizon refinement's last rung residual, the base's series (|z_N|, a_max, fourth ratio, smallest
pivot), Delta to 17 digits plus its low part, the seconds of computing, the cumulative hours of computing of the
chain, and a verdict per check (OK, WATCH, BAD by BOUNDS) with the step's overall verdict, the worst of them.
OUTDIR/errorbar.csv has the error bar on Delta against D, the spread over the held fit's degree: |d8 - d6| and
|d8 - d10|, each with how it was taken (at that D, or interpolated from the neighbouring D of that chain), and their
larger (error_bar), from data/delta_fits.csv (export_delta_all.py), then past its last D the same at equal D from the
chains' logs (the clean chain against win's d6 and d10).

--selftest parses a planted log with one accepted and one refused attempt, checks every field and verdict, and with
--mutate=lam reads lam wrongly, which must fail. Exit 1 on a failed check.
"""
import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# the chains, each a lineage of logs in the order they ran (relative to RAWDIR)
CHAINS = [
    ("win2 clean chain", "win2", 8, ["I461-win2/crawlG.log", "I461-win2/crawlH.log", "I461-win2/crawlI.log",
                                     "I461-win2/crawlJ.log", "I461-win2/crawlK.log", "I461-win2/probe125.log",
                                     "I461-win2/probe0625.log", "I461-win2/probe03125.log", "I461-win2/crawlL.log"]),
    ("win d6", "win", 6, ["I462-win/chain_d6.log"]),
    ("win d10", "win", 10, ["I462-win/chain_d10.log"]),
    ("win2 first chain", "win2", 8, ["I461-win2/crawlB2.log", "I461-win2/crawlB3.log", "I461-win2/crawlB4.log",
                                     "I461-win2/crawlB5.log", "I461-win2/crawlC.log", "I461-win2/crawlD.log",
                                     "I461-win2/crawlE.log", "I461-win2/crawlF.log"]),
]

# (check, OK when value <= ok, WATCH when <= watch, else BAD); the coordinator's bounds of 2026-11-11 01:06 UTC
BOUNDS = {
    "newton": (4, 8),
    "final_F": (1e-60, 1e-40),
    "start_F": (1e-33, 1e-25),
    "first_dz": (1e-29, 1e-26),
    "z_N": (1e-27, 1e-24),
}
CHECKS = ["newton", "lam", "final", "start", "first_dz", "series"]

NUM = r"([-+0-9.eE]+|nan|inf)"
RE_STEP = re.compile(r"^step (\d+): D ([0-9.]+), pivots " + NUM + r"\.\." + NUM + r", \|z_1\| " + NUM + r", \|z_N\| " + NUM
                     + r", ratios 2-5 " + NUM + " " + NUM + " " + NUM + " " + NUM + r", a_max " + NUM + r", ([0-9.]+) s")
RE_RESEED = re.compile(r"^\s+a ([0-9.]+): re-seeded \((ok|FAILED[^)]*)\), ([0-9.]+) s")
RE_REFINE = re.compile(r"horizon data refined in octo-double, rung max\|R\| (.*)$")
RE_ON = re.compile(r"^\s+on (\d+): \|F\| " + NUM + r" -> " + NUM + r", \|dz\| " + NUM + r" -> " + NUM + r", lam " + NUM)
RE_ON_CONV = re.compile(r"^\s+on (\d+): \|F\| " + NUM + r" converged, Delta ([0-9.]+) ([-+0-9.eE]+)")
RE_ON_STOP = re.compile(r"^\s+on (\d+): \|F\| " + NUM + r", \|dz\| " + NUM + r": no trial step accepted")
RE_ON_STALL = re.compile(r"^\s+on (\d+): lam at or below")
RE_OD = re.compile(r"octo-double Newton: (\d+) steps, max\|F\| " + NUM + r", (converged|not converged)")
RE_END = re.compile(r"^\s+a ([0-9.]+) -> D ([0-9.]+): .* Delta ([0-9.]+) .*, ([0-9.]+) s$")


def verdict(check, v):
    """OK, WATCH or BAD of one check's value"""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    ok, watch = BOUNDS[check]
    return "OK" if v <= ok else ("WATCH" if v <= watch else "BAD")


def parse_log(text, chain, run):
    """the attempts of one log, in order: a list of dicts"""
    out, base, att = [], None, None
    for line in text.splitlines():
        m = RE_STEP.match(line)
        if m:
            g = m.groups()
            base = dict(D0=g[1], pmin=float(g[2]), pmax=float(g[3]), z_N=float(g[5]), ratio4=float(g[9]),
                        a_max=float(g[10]), t_series=float(g[11]), first_of_base=True)
            continue
        m = RE_RESEED.match(line)
        if m and base is not None:
            att = dict(chain=chain, run=run, base=base, a=m.group(1), reseed_ok=m.group(2) == "ok",
                       t_reseed=float(m.group(3)), ons=[], refine="", Delta="", Delta_lo="", newton=None, conv=None,
                       F_od=None, stop="")
            continue
        if att is None:
            continue
        m = RE_REFINE.search(line)
        if m:
            att["refine"] = m.group(1).split("->")[-1].strip()
            continue
        m = RE_ON.match(line)
        if m:
            att["ons"].append(dict(F0=float(m.group(2)), F1=float(m.group(3)), dz0=float(m.group(4)), lam=float(m.group(6))))
            continue
        m = RE_ON_CONV.match(line)
        if m:
            att["Delta"], att["Delta_lo"] = m.group(3), m.group(4)
            if not att["ons"]:
                att["ons"].append(dict(F0=float(m.group(2)), F1=float(m.group(2)), dz0=float("nan"), lam=1.0))
            continue
        m = RE_ON_STOP.match(line)
        if m:
            att["stop"] = "no trial step accepted"
            if not att["ons"]:
                att["ons"].append(dict(F0=float(m.group(2)), F1=float(m.group(2)), dz0=float(m.group(3)), lam=float("nan")))
            continue
        if RE_ON_STALL.match(line):
            att["stop"] = "stalled"
            continue
        m = RE_OD.search(line)
        if m:
            att["newton"], att["F_od"], att["conv"] = int(m.group(1)), float(m.group(2)), m.group(3) == "converged"
            continue
        m = RE_END.match(line)
        if m:
            att["D"], att["t_corr"] = m.group(2), float(m.group(4))
            if not att["Delta"]:
                att["Delta"] = m.group(3)
            out.append(att)
            att = None
    return out


def rows_of(attempts):
    """the table's rows, with the chain's cumulative hours of computing"""
    rows, hours = [], 0.0
    for at in attempts:
        b = at["base"]
        if b.pop("first_of_base", False):
            hours += b["t_series"] / 3600
        hours += (at["t_reseed"] + at["t_corr"]) / 3600
        ons = at["ons"]
        lams = [o["lam"] for o in ons if not math.isnan(o["lam"])]
        start_F = ons[0]["F0"] if ons else float("nan")
        first_dz = ons[0]["dz0"] if ons else float("nan")
        lam_min = min(lams) if lams else float("nan")
        accepted = bool(at["conv"])
        v = {
            "newton": verdict("newton", at["newton"]) if accepted else "BAD",
            "lam": ("OK" if lam_min >= 1 else ("WATCH" if lam_min >= 0.25 else "BAD")) if lams else "",
            "final": verdict("final_F", at["F_od"]) if accepted else "BAD",
            "start": verdict("start_F", start_F),
            "first_dz": verdict("first_dz", first_dz),
            "series": max((verdict("z_N", b["z_N"]),
                           "OK" if b["a_max"] > 2.5 else ("WATCH" if b["a_max"] >= 2.0 else "BAD"),
                           # the fourth ratio is a diagnostic: its fall from D 231 on comes with clean steps (#461)
                           "OK" if b["ratio4"] > 150 else "WATCH"),
                          key=["", "OK", "WATCH", "BAD"].index),
        }
        overall = max((x for x in v.values()), key=["", "OK", "WATCH", "BAD"].index)
        rows.append(dict(chain=at["chain"], run=at["run"], D_from=b["D0"], D_to=at["D"], a=at["a"],
                         accepted=int(accepted), newton=at["newton"] if at["newton"] is not None else "",
                         lam_min=f"{lam_min:g}" if lams else "", start_F=f"{start_F:.2e}", first_dz=f"{first_dz:.2e}",
                         final_F=f"{at['F_od']:.2e}" if at["F_od"] is not None else "", refine_last=at["refine"],
                         reseed_ok=int(at["reseed_ok"]), z_N=f"{b['z_N']:.2e}", a_max=f"{b['a_max']:.3f}",
                         ratio4=f"{b['ratio4']:.1f}", pivot_min=f"{b['pmin']:.1e}", Delta=at["Delta"],
                         Delta_lo=at["Delta_lo"], seconds=round(at["t_reseed"] + at["t_corr"]), hours=f"{hours:.3f}",
                         stop=at["stop"], **{f"v_{k}": v[k] for k in CHECKS}, verdict=overall))
    return rows


def errorbar(rows):
    """the error bar on Delta against D: |d8 - d6| and |d8 - d10| of data/delta_fits.csv (each at that D or
    interpolated, export_delta_all.py) and their larger, then the same at equal D from the chains' logs past its last
    D"""
    out = []
    with open(os.path.join(HERE, "data", "delta_fits.csv")) as f:
        for r in csv.DictReader(f):
            if float(r["D"]) >= 100:
                e = {k: f"{abs(float(r['diff_' + k])):.4e}" if r["diff_" + k] else "" for k in ("d6", "d10")}
                out.append(dict(D=r["D"], d6=e["d6"], d6_how=r["how_d6"], d10=e["d10"], d10_how=r["how_d10"],
                                error_bar=max((v for v in e.values() if v), key=float), source="delta_fits.csv"))
    last = max((float(x["D"]) for x in out), default=-math.inf)
    d = {}
    for r in rows:
        if r["accepted"] and r["chain"] in ("win2 clean chain", "win d6", "win d10") and r["Delta"]:
            from decimal import Decimal
            d.setdefault(round(float(r["D_to"]), 6), {})[r["chain"]] = (r["D_to"], Decimal(r["Delta"])
                                                                         + Decimal(r["Delta_lo"] or "0"))
    for D, v in sorted(d.items()):
        if D <= last + 1e-9 or "win2 clean chain" not in v or len(v) == 1:
            continue
        D_to, d8 = v["win2 clean chain"]
        e = {k: f"{abs(v['win ' + k][1] - d8):.4e}" if "win " + k in v else "" for k in ("d6", "d10")}
        out.append(dict(D=D_to, d6=e["d6"], d6_how="direct" if e["d6"] else "", d10=e["d10"],
                        d10_how="direct" if e["d10"] else "", error_bar=max((x for x in e.values() if x), key=float),
                        source="logs"))
    return out


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


PLANT = """step 0: D 232.63125, pivots 3.1e-21..1.1e+33, |z_1| 4.10e-04, |z_N| 5.38e-30, ratios 2-5 298.1 268.7 255.2 216.6, a_max 3.277, 390 s
    a 0.5000: re-seeded (ok), 17.1 s
      horizon data refined in octo-double, rung max|R| 7.29e-29 -> 5.69e-62 -> 4.17e-69 -> 1.93e-75
  on 0: |F| 6.17e-36 -> 2.58e-38, |dz| 7.1e-36 -> 3.9e-40, lam 1, dDelta +1.0e-36, piv 3.1e-21..1.1e+33, 20 s
  on 1: |F| 2.58e-38 -> 1.40e-80, |dz| 3.9e-40 -> 1.0e-79, lam 1, dDelta +1.0e-40, piv 3.1e-21..1.1e+33, 20 s
  on 2: |F| 1.40e-80 converged, Delta 1.68787 +1.0e-17
      octo-double Newton: 3 steps, max|F| 1.4e-80, converged; double-double |F| of its first two parts 4.0e+02
    a 0.500 -> D 233.13125: predicted |F| 4.0e+02, od 3 steps -> 4.0e+02 (tol 1e-20), Delta 1.687870000000 (formula diff -1.4e-06), moved 3.9e-34, 46 s
step 1: D 233.13125, pivots 2.0e-21..1.3e+33, |z_1| 4.10e-04, |z_N| 6.00e-30, ratios 2-5 298.0 268.0 255.0 210.0, a_max 3.200, 390 s
    a 0.5000: re-seeded (ok), 17.0 s
  on 0: |F| 6.00e-36 -> 7.41e-22, |dz| 3.8e-26 -> 2.9e-27, lam 0.0156, dDelta +1e-30, piv 2e-21..1e33, 60 s
  on 1: |F| 7.41e-22, |dz| 2.9e-27: no trial step accepted, stopping
      octo-double Newton: 2 steps, max|F| 7.4e-22, not converged; double-double |F| of its first two parts 4.0e+02
    a 0.500 -> D 233.63125: predicted |F| 4.0e+02, od 2 steps -> 4.0e+02 (tol 1e-20), Delta 1.687700000000 (formula diff -1.4e-06), moved 2.7e-27, 522 s
"""


def selftest(mutate=""):
    text = PLANT.replace("lam 0.0156", "lam 1") if mutate == "lam" else PLANT
    rows = rows_of(parse_log(text, "test", "plant.log"))
    want = [dict(D_to="233.13125", accepted=1, newton=3, lam_min="1", verdict="OK", v_newton="OK", v_first_dz="OK",
                 Delta="1.68787", refine_last="1.93e-75", hours=f"{(390 + 17.1 + 46) / 3600:.3f}"),
            dict(D_to="233.63125", accepted=0, newton=2, lam_min="0.0156", verdict="BAD", v_lam="BAD",
                 v_first_dz="BAD", v_start="OK", v_series="OK", stop="no trial step accepted")]
    bad = 0
    if len(rows) != 2:
        print(f"selftest: {len(rows)} rows, expected 2: FAIL")
        return 1
    for r, w in zip(rows, want):
        for k, v in w.items():
            ok = r[k] == v
            bad += not ok
            if not ok:
                print(f"selftest: {r['D_to']} {k} = {r[k]!r}, expected {v!r}")
    print(f"selftest: {'PASS' if not bad else 'FAIL'}{' (mutation ' + mutate + ')' if mutate else ''}")
    return 1 if bad else 0


def main(argv):
    if "--selftest" in argv:
        m = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), "")
        return selftest(m)
    raw, out = argv[0], argv[1]
    os.makedirs(out, exist_ok=True)
    rows = []
    for chain, runner, deg, logs in CHAINS:
        attempts = []
        for lg in logs:
            p = os.path.join(raw, lg)
            if os.path.exists(p):
                attempts += parse_log(open(p, errors="replace").read(), chain, os.path.basename(lg))
        for r in rows_of(attempts):
            r.update(runner=runner, fit_degree=deg)
            rows.append(r)
    for i, r in enumerate(rows):
        r["n"] = i + 1                                   # the row's key on the page
    write(os.path.join(out, "steps.csv"), rows)
    eb = errorbar(rows)
    write(os.path.join(out, "errorbar.csv"), eb)
    print(f"steps.csv: {len(rows)} attempts, {sum(r['accepted'] for r in rows)} accepted; errorbar.csv: {len(eb)} rows")
    for chain, *_ in CHAINS:
        acc = [r for r in rows if r["chain"] == chain and r["accepted"]]
        if acc:
            r = acc[-1]
            print(f"  {chain}: last accepted D {r['D_to']}, {r['verdict']}, hours {r['hours']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
