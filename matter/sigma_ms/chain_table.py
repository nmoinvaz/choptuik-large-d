#!/usr/bin/env python3
"""Delta(D) of the multishoot kernel's chain at one coupling against sigdss's values (#916), from the raw records,
into a csv and a markdown table.

    git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw     (once, beside the checkout)
    python3 matter/sigma_ms/chain_table.py ../choptuik-large-d-raw [--folder I916-mac] [--eta 0.3] \
        [--sigdss solver/matter/data/sigdss_m03up.jsonl,solver/matter/data/sigdss_m03hi.jsonl] \
        [--ref I913-mac] [--csv matter/sigma_ms/data/delta_D_eta0.3.csv] [--mutate closure|kink]
    python3 matter/sigma_ms/chain_table.py ../choptuik-large-d-raw --folder I917-mac --eta 1 --also I652-mac \
        --Dmin 10 --D0 10.475 --ref I652-mac,I845-mac,I913-mac \
        --sigdss solver/matter/data/sigdss_m1up.jsonl,solver/matter/data/sigdss_m1hi.jsonl \
        --csv matter/sigma_ms/data/delta_D_eta1.csv                                  (#917)

Inputs: <folder>/out/sm_records.jsonl (run_eta.py's records of the chain: at each D the converged record of this
coupling with the smallest max |F|), with the records of the --also folders from --Dmin on (an earlier part of the
chain), sigdss's march records (D, Delta, reg: its smoothness prior), and for the closure the records of the --ref
folders at the chain's starting D.

A point of the quad-double hand-over (#928, run_eta.py --qd --qd_accept: past D 12.4 the double-double Newton stalls
near 1e-27 on the band) counts with its quad-double residual and its quad-double Delta, and only when run_eta.py
accepted it. Such points are marked: the csv's method column reads qd and its handover_tol the double-double
stage's tolerance, and check (1) counts them apart.

Checks, each printed with its verdict; the exit status is 1 when one fails:
  (1) every D of the chain has a converged record with max |F| <= 1e-28;
  (2) closure: the record of --folder at the starting D, solved again from the chain's neighbours, agrees with every
      converged record of --ref there to 1e-20 in Delta;
  (3) Delta(D) is smooth: on the multiples of 0.0625 its third differences keep one sign, its fourth differences
      keep one sign, and the largest fourth difference is below 1% of the largest second difference. A jump of
      2e-8 or more at one point, to another solution or by a wrong digit, fails this.
      (Its first version asked that neighbouring second differences change by less than a tenth. That fails where
      the second difference goes to zero, 12% near D 10.9 at eta = 0.3, with no kink there.)
--mutate closure shifts the chain's value at the starting D by 1e-19 before check (2), which must then fail.
--mutate kink shifts the value at the middle grid point by 1e-7 before check (3), which must then fail.
"""
import argparse
import csv
import json
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
mp.mp.dps = 40


def is_qd(r):
    return bool(r.get("qd", {}).get("accepted"))


def delta(r):
    return mp.mpf(r["qd"]["Delta"]) if is_qd(r) else mp.mpf(r["Delta_hi"]) + mp.mpf(r["Delta_lo"])


def res(r):
    """max |F| of a record: the quad-double residual of an accepted quad-double point, else the double-double one"""
    return r["qd"]["F_final"] if is_qd(r) else r["F"]


def best(path, eta):
    """{D: the converged record of this coupling with the smallest max |F|}"""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if str(r["eta"]) != eta or not r["converged"] or r.get("M", 32) != 32:
            continue
        if "handover_tol" in r.get("qd", {}) and r["qd"]["handover_tol"] > 1e-28 and not is_qd(r):
            continue                                        # a hand-over run that the quad-double stage did not accept
        D = float(r["D"])
        if D not in out or res(r) < res(out[D]):
            out[D] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--folder", default="I916-mac")
    ap.add_argument("--eta", default="0.3")
    ap.add_argument("--sigdss", default="solver/matter/data/sigdss_m03up.jsonl,solver/matter/data/sigdss_m03hi.jsonl")
    ap.add_argument("--ref", default="I913-mac", help="comma-separated folders")
    ap.add_argument("--also", default="", help="comma-separated folders with an earlier part of the chain")
    ap.add_argument("--Dmin", type=float, default=0.0, help="the table starts here")
    ap.add_argument("--D0", type=float, default=10.5, help="the chain's starting D, where the closure is checked")
    ap.add_argument("--csv", default="")
    ap.add_argument("--mutate", default="", choices=["", "closure", "kink"])
    a = ap.parse_args()
    own = best(os.path.join(a.raw, a.folder, "out", "sm_records.jsonl"), a.eta)
    ch = dict(own)
    for fo in [x for x in a.also.split(",") if x]:
        for D, r in best(os.path.join(a.raw, fo, "out", "sm_records.jsonl"), a.eta).items():
            if D not in ch or res(r) < res(ch[D]):
                ch[D] = r
    ch = {D: r for D, r in ch.items() if D >= a.Dmin - 1e-12}
    Ds = sorted(ch)
    sig = {}
    for f in a.sigdss.split(","):
        name = os.path.basename(f).replace("sigdss_", "").replace(".jsonl", "")
        for line in open(os.path.join(ROOT, f)):
            r = json.loads(line)
            if Ds[0] - 1e-9 <= r["D"] <= Ds[-1] + 1e-9:
                sig.setdefault(r["D"], []).append((name, r["Delta"], r["reg"]))
    ok = True
    # (1)
    worst = max(res(ch[D]) for D in Ds)
    v1 = worst <= 1e-28
    ok &= v1
    qd = [D for D in Ds if is_qd(ch[D])]
    note = ""
    if qd:
        note = (f"; {len(qd)} of them in quad-double from D {qd[0]:g} on (hand-over at "
                f"{max(ch[D]['qd']['handover_tol'] for D in qd):g}), largest quad-double max |F| "
                f"{max(res(ch[D]) for D in qd):.2e}, largest double-double one "
                f"{max(res(ch[D]) for D in Ds if D not in qd):.2e}")
    print(f"(1) {len(Ds)} values of D from {Ds[0]:g} to {Ds[-1]:g}, each converged; largest max |F| {worst:.2e} "
          f"(<= 1e-28){note}: {'PASS' if v1 else 'FAIL'}")
    # (2)
    d0 = delta(own[a.D0]) + (mp.mpf("1e-19") if a.mutate == "closure" else 0)
    allref = []
    for fo in a.ref.split(","):
        allref += [json.loads(line) for line in open(os.path.join(a.raw, fo, "out", "sm_records.jsonl"))]
    allref = [r for r in allref if str(r["eta"]) == a.eta and float(r["D"]) == a.D0 and r["converged"]
              and r["F"] <= 1e-30 and r.get("M", 32) == 32]
    gap = max(abs(d0 - delta(r)) for r in allref) if allref else mp.inf
    v2 = bool(allref) and gap <= mp.mpf("1e-20")
    ok &= v2
    print(f"(2) closure at D {a.D0:g}: the chain's {mp.nstr(d0, 25)} against {len(allref)} records of {a.ref} with "
          f"max |F| <= 1e-30: largest difference {mp.nstr(gap, 3)} (<= 1e-20): {'PASS' if v2 else 'FAIL'}")
    # (3) on the multiples of 0.0625 only, which must follow one another without a gap
    step = min(b - c for b, c in zip(Ds[1:], Ds))
    grid = [D for D in Ds if abs(D / 0.0625 - round(D / 0.0625)) < 1e-9]
    assert all(abs(q - p - 0.0625) < 1e-9 for p, q in zip(grid, grid[1:])), "a multiple of 0.0625 is missing"
    vals = [delta(ch[D]) for D in grid]
    if a.mutate == "kink":
        vals[len(vals) // 2] += mp.mpf("1e-7")

    def diff(v):
        return [q - p for p, q in zip(v, v[1:])]
    d2 = diff(diff(vals))
    d3 = diff(d2)
    d4 = diff(d3)
    one3 = all(x > 0 for x in d3) or all(x < 0 for x in d3)
    one4 = all(x > 0 for x in d4) or all(x < 0 for x in d4)
    ratio = max(abs(x) for x in d4) / max(abs(x) for x in d2)
    v3 = one3 and one4 and ratio < 0.01
    ok &= v3
    h = mp.mpf("0.0625")
    print(f"(3) smoothness on the {len(grid)} grid points (step 0.0625; smallest step of all points {step:g}): the "
          f"second derivative runs from {mp.nstr(d2[0] / h ** 2, 4)} to {mp.nstr(d2[-1] / h ** 2, 4)}; third "
          f"differences of one sign {one3}, fourth of one sign {one4}, largest fourth over largest second "
          f"{mp.nstr(ratio, 3)} (< 0.01): {'PASS' if v3 else 'FAIL'}")
    rows = []
    for D in Ds:
        de = delta(ch[D])
        for name, sd_, reg in sig.get(D, [(None, None, None)]):
            rows.append(dict(D=D, Delta=mp.nstr(de, 30), max_F=res(ch[D]), sigdss=name or "", sigdss_Delta=sd_ or "",
                             sigdss_prior=reg if reg is not None else "",
                             kernel_minus_sigdss=(float(de) - sd_) if sd_ else "",
                             method="qd" if is_qd(ch[D]) else "dd",
                             handover_tol=ch[D]["qd"]["handover_tol"] if is_qd(ch[D]) else ""))
    print("\n| D | Δ, kernel | max \\|F\\| | sigdss's Δ (march, prior) | kernel minus sigdss |")
    print("|---|---|---|---|---|")
    for r in rows:
        if r["sigdss"]:
            print(f"| {r['D']:.12g} | {r['Delta'][:18]} | {r['max_F']:.1e} | {r['sigdss_Delta']:.7f} ({r['sigdss']}, "
                  f"{r['sigdss_prior']:g}) | {r['kernel_minus_sigdss']:+.2e} |")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{len(rows)} rows -> {a.csv}")
    print(f"\nCHAIN TABLE{' (mutation ' + a.mutate + ')' if a.mutate else ''}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
