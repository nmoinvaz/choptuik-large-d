#!/usr/bin/env python3
"""The prediction error per unit step of anm_run.py's chain: for each base of the records, c = |F_pred| / a over its
attempts (the step a halved after each failed correction), the spread of c across them, and whether each was accepted.
A c constant across a means |F_pred| is linear in a, an error of the series' first-order coefficient; the last lines
give the largest accepted and the smallest rejected |F_pred|, the corrector's reach in predicted residual.

    python3 anm_pred_slope.py RAWDIR FOLDER [FOLDER ...]     (folders of anm_run.jsonl, e.g. anm-mac/run_trend_pin3_qd6)
"""
import json
import os
import sys


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    raw, folders = sys.argv[1], sys.argv[2:]
    ok, bad = [], []
    print("folder  D0  attempts (a, |F_pred|, accepted)  c min..max  spread")
    for fo in folders:
        with open(os.path.join(raw, fo, "anm_run.jsonl")) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        for r in recs:
            att = [x for x in r.get("attempts", []) if x.get("F_pred") is not None and x["F_pred"] == x["F_pred"]]
            if not att:
                continue
            cs = [x["F_pred"] / x["a"] for x in att]
            for x in att:
                (ok if x["accepted"] else bad).append((x["F_pred"], fo, r["D0"], x["a"]))
            desc = ", ".join(f"({x['a']:.4g}, {x['F_pred']:.2e}, {'y' if x['accepted'] else 'n'})" for x in att)
            print(f"{os.path.basename(fo)}  {r['D0']:.4f}  {desc}  c {min(cs):.2e}..{max(cs):.2e}  {max(cs) / min(cs):.3f}")
    if ok:
        m = max(ok)
        print(f"largest accepted |F_pred| {m[0]:.2e} ({os.path.basename(m[1])}, D0 {m[2]:.4f}, a {m[3]:.4g})")
    if bad:
        m = min(bad)
        print(f"smallest rejected |F_pred| {m[0]:.2e} ({os.path.basename(m[1])}, D0 {m[2]:.4f}, a {m[3]:.4g})")
    print(f"{len(ok)} accepted, {len(bad)} rejected attempts")


if __name__ == "__main__":
    main()
