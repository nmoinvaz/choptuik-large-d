#!/usr/bin/env python3
"""Delta(eta) of the multishoot kernel at M = 32 and 48 against sigdss's k_c 23 and 39 (#639 acceptance 2-4), from the
raw records, into matter/sigma_ms/data/delta_eta.csv and a markdown table.

    git clone https://github.com/nmoinvaz/choptuik-large-d-raw ../choptuik-large-d-raw     (once, beside the checkout)
    python3 matter/sigma_ms/tables.py ../choptuik-large-d-raw [--csv matter/sigma_ms/data/delta_eta.csv]

Inputs: I639-mac/data/sm_records.jsonl (this kernel; the last converged record of each eta and M), and sigdss's
refine records sigdss_refine_*_kc{23,39}_eta*.jsonl of I623-win, I630-win, I631-win and I633-win.

Acceptance, written in the issue before the runs: (2) at eta = inf, Delta agrees with the l = 1 multiplet's and
sigdss's k_c 39 value 0.4598611019 within sigdss's own truncation, its k_c 31 to 39 change 9.83e-8 (the l = 1 records,
solver/matter/data/l1_D4_kc31.jsonl and kc39); (3) at eta = 5, 7 and 10, |Delta(M = 48) - Delta(k_c 39)| is below
|Delta(k_c 39) - Delta(k_c 23)|. Each row's verdict is printed and the exit status is 1 when one fails.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
ETAS = ["5", "6", "7", "8.5", "10", "14", "20", "inf"]
mp.mp.dps = 40


def eta_key(e):
    """sigdss's eta labels ('10.0000', '8.5', 'inf') as ours"""
    if e == "inf":
        return "inf"
    v = float(e)
    return f"{v:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--csv", default="")
    ap.add_argument("--Ms", default="32,48,64")
    a = ap.parse_args()
    Ms = [int(m) for m in a.Ms.split(",")]
    ours = {}
    for line in open(os.path.join(a.raw, "I639-mac", "data", "sm_records.jsonl")):
        r = json.loads(line)
        if r.get("converged"):
            ours[(r["eta"], r["M"])] = r
    sig = {}
    for d in ("I623-win", "I630-win", "I631-win", "I633-win"):
        for f in glob.glob(os.path.join(a.raw, d, "data", "sigdss_refine_*_kc*_eta*.jsonl")):
            m = re.search(r"_kc(\d+)_eta([0-9.]+|inf)\.jsonl$", f)
            r = json.loads(open(f).readline())
            sig[(eta_key(m.group(2)), int(m.group(1)))] = r["Delta"]
    l1 = {kc: json.loads(open(os.path.join(ROOT, "solver", "matter", "data", f"l1_D4_kc{kc}.jsonl")).readline())["Delta"]
          for kc in (23, 31, 39)}
    rows, ok = [], True
    for e in ETAS:
        row = dict(eta=e)
        for M in Ms:
            r = ours.get((e, M))
            # the double-double value hi + lo (the records' "Delta" string of runs before the fix of run_eta.py
            # holds the hi part only: mpmath was at its default 53 bits there)
            row[f"Delta_M{M}"] = mp.nstr(mp.mpf(r["Delta_hi"]) + mp.mpf(r["Delta_lo"]), 32) if r else ""
            row[f"F_M{M}"] = f"{r['F']:.1e}" if r else ""
            row[f"constraint_M{M}"] = f"{r['constraint']:.1e}" if r else ""
        if row["Delta_M32"] and row.get("Delta_M48"):
            row["M48_minus_M32"] = mp.nstr(mp.mpf(row["Delta_M48"]) - mp.mpf(row["Delta_M32"]), 10)
        if row.get("Delta_M48") and row.get("Delta_M64"):
            row["M64_minus_M48"] = mp.nstr(mp.mpf(row["Delta_M64"]) - mp.mpf(row["Delta_M48"]), 10)
        row["Delta_kc23"] = sig.get((e, 23), "")
        row["Delta_kc39"] = sig.get((e, 39), "")
        # the acceptance compares M = 48 (the issue's highest M; M = 64 is the check of its error)
        best = row.get("Delta_M48") or row.get("Delta_M32")
        if best and row["Delta_kc39"] != "":
            diff = mp.mpf(best) - mp.mpf(row["Delta_kc39"])
            row["best_minus_kc39"] = mp.nstr(diff, 10)
            row["kc39_minus_kc23"] = f"{row['Delta_kc39'] - row['Delta_kc23']:.4e}" if row["Delta_kc23"] != "" else ""
            if e == "inf":
                bound = abs(l1[39] - l1[31])
                v = abs(diff) <= bound and abs(mp.mpf(best) - mp.mpf(l1[39])) <= bound
                row["verdict"] = f"{'PASS' if v else 'FAIL'} (2): |d| <= l=1 k_c 31->39 change {bound:.2e}"
                ok &= v
            elif e in ("5", "7", "10"):
                bound = abs(row["Delta_kc39"] - row["Delta_kc23"])
                v = abs(diff) <= bound
                row["verdict"] = f"{'PASS' if v else 'FAIL'} (3): |d| <= k_c 23->39 change {bound:.2e}"
                ok &= v
        rows.append(row)
    cols = ["eta"] + [f"{p}_M{M}" for M in Ms for p in ("Delta", "F", "constraint")] + \
        ["M48_minus_M32"] + (["M64_minus_M48"] if 64 in Ms else []) + \
        ["Delta_kc23", "Delta_kc39", "best_minus_kc39", "kc39_minus_kc23", "verdict"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    print(f"l = 1 multiplet: k_c 23 {l1[23]}, 31 {l1[31]}, 39 {l1[39]}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        # the checks (I639-mac/checks/): a second start and the quad-double Newton, against the chain's run at the same
        # eta and M; Delta of a quad-double state from its four parts in the saved state
        import numpy as np
        cpath = os.path.join(a.raw, "I639-mac", "checks", "sm_records.jsonl")
        if os.path.exists(cpath):
            with open(os.path.join(os.path.dirname(a.csv), "checks.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["tag", "eta", "M", "seed", "F", "Delta", "minus_chain", "qd_F", "Delta_qd", "qd_minus_dd"])
                for line in open(cpath):
                    r = json.loads(line)
                    D = mp.mpf(r["Delta_hi"]) + mp.mpf(r["Delta_lo"])
                    c = ours[(r["eta"], r["M"])]
                    Dc = mp.mpf(c["Delta_hi"]) + mp.mpf(c["Delta_lo"])
                    qF, Dq, dq = "", "", ""
                    if "qd" in r:
                        z = np.load(os.path.join(a.raw, "I639-mac", "checks", f"sm_{r['tag']}.npz"))
                        with mp.workdps(80):
                            Q = sum(mp.mpf(float(z[k][-1])) for k in ("zh", "zl", "z2", "z3"))
                            Dq, dq = mp.nstr(Q, 64), mp.nstr(Q - D, 6)
                        qF = f"{r['qd']['F'][-1]:.2e}"
                    seed = r["seed"].get("sigdss") or r["seed"].get("src")
                    w.writerow([r["tag"], r["eta"], r["M"], seed, f"{r['F']:.2e}", mp.nstr(D, 32), mp.nstr(D - Dc, 6), qF, Dq, dq])
        # the planted check's runs (planted.py), one row per run, beside the table
        pc = ["M", "mutate", "steps", "F_star", "dd_converged", "dd_newton_steps", "dd_err", "dd_dDelta", "dd_F_last",
              "dd_pass", "qd_converged", "qd_F_last", "qd_dDelta_dd", "qd_dDelta", "qd_pass"]
        with open(os.path.join(os.path.dirname(a.csv), "planted.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(pc)
            for line in open(os.path.join(a.raw, "I639-mac", "planted", "planted_records.jsonl")):
                r = json.loads(line)
                dd, qd = r["dd"], r["qd"]
                w.writerow([r["M"], r["mutate"] or "none", r["steps"], f"{r['F_star']:.3e}", dd["converged"], dd["steps"],
                            f"{dd['err']:.3e}", f"{dd['dDelta']:.3e}", f"{dd['F'][-1]:.3e}", dd["pass_"], qd["converged"],
                            f"{qd['F'][-1]:.3e}", f"{qd.get('dDelta_dd', float('nan')):.3e}", f"{qd['dDelta']:.3e}", qd["pass_"]])
    print("ACCEPTANCE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
