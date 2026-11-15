#!/usr/bin/env python3
"""The horizon series' order ks of the runs behind notes/FACTORED_HORIZON.md's node law of D 40-61, against 1.9 Re s
(#734): the note's 4.4 keeps f = 1 to 1% from ks of about 1.9 Re s, and reads f = 1 at D 40-61.

    cd hp && python3 ks_vs_res.py [--csv=data/ks_vs_res.csv] [--mutate=ks]

The runs:
  node      the 15 runs of solver/beyond_lc/hp_node_test.py (notes/BEYOND_LIGHT_CONE.md section 5, the node law of
            LARGE_D_WALL.md: max|dDelta/dF| against e^(n [Phi(x_m) - Phi(0.7)]) over 13 orders), records in
            solver/beyond_lc/hp_node_test.jsonl. The records carry no ks; the script's --ks default, 160 since its only
            commit (46d03ba0), is the one they ran at, since BEYOND_LIGHT_CONE.md's command passes none.
  state     every double-double scan state of hp/data/states at D 40 to 61, the node runs' inputs among them, at the
            ks of its meta.
Re s is hp/horizon_radius.py's: n / (2 (1 + (n - 1)(1 - mean(a_h)^2))) - 1, n = D - 2, a_h the horizon block of the
state's leading part (offset 3M + 4MK). A node run takes its input state's Re s (Delta agrees between the layouts,
BEYOND_LIGHT_CONE.md 5).
A run is flagged when ks < 1.9 Re s. The check is the note's reading, that no run is flagged; exit 1 when one is.
Control: --mutate=ks takes every ks as 40, which flags every run, and must exit 1.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODE_KS = 160
args = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))


def re_s(path):
    z = np.load(path)
    m = json.loads(str(z["meta"]))
    M, K = int(m["M"]), len(m["nodes"])
    n = float(m["D"]) - 2.0
    ah = np.asarray(z["zh"], dtype=float)[3 * M + 4 * M * K:3 * M + 4 * M * K + M]
    return float(m["D"]), int(m["ks"]), n / (2.0 * (1.0 + (n - 1.0) * (1.0 - float(np.mean(ah)) ** 2))) - 1.0


def main():
    mut = args.get("mutate") == "ks"
    rows = []
    states = {}
    for f in sorted(glob.glob(os.path.join(HERE, "data", "states", "up*_D*.npz"))):
        z = np.load(f)
        if "meta" not in z.files:
            continue
        D = float(json.loads(str(z["meta"]))["D"])
        if 39.9 <= D <= 61.1:
            states[os.path.basename(f)[:-4]] = re_s(f)
    for name, (D, ks, rs) in sorted(states.items(), key=lambda x: x[1][0]):
        rows.append(dict(run="state", name=name, D=D, x_last="", ks=40 if mut else ks, ks_source="meta", Re_s=rs))
    for line in open(os.path.join(ROOT, "solver", "beyond_lc", "hp_node_test.jsonl")):
        if not line.strip():
            continue
        r = json.loads(line)
        D, _, rs = states[r["state"]]
        rows.append(dict(run="node", name=r["state"], D=D, x_last=r["x_last"], ks=40 if mut else NODE_KS,
                         ks_source="script default", Re_s=rs))
    for r in rows:
        r["ks_over_Re_s"] = r["ks"] / r["Re_s"]
        r["flag"] = "ks < 1.9 Re s" if r["ks"] < 1.9 * r["Re_s"] else ""
    print("run    name            D        x_last  ks   Re s    1.9 Re s  ks/Re s  flag")
    for r in rows:
        print("%-6s %-15s %8.4f  %-6s  %3d  %6.2f  %7.2f   %5.2f   %s" % (
            r["run"], r["name"], r["D"], r["x_last"], r["ks"], r["Re_s"], 1.9 * r["Re_s"], r["ks_over_Re_s"], r["flag"]))
    nflag = sum(bool(r["flag"]) for r in rows)
    lo = min(r["ks_over_Re_s"] for r in rows)
    print("%d runs, %d flagged; the smallest ks/Re s %.2f%s" % (len(rows), nflag, lo, " (--mutate=ks)" if mut else ""))
    ok = nflag == 0
    out = args.get("csv")
    if out:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("check:", "pass" if ok else "FAIL", "(no run with ks < 1.9 Re s)" if ok else "(%d runs with ks < 1.9 Re s)" % nflag)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
