#!/usr/bin/env python3
"""The full equations' grip on each held harmonic, from held_schur.py's JSON files, as one CSV.

    cd hp && python3 held_grip.py RAWDIR FILE.json [FILE.json ...] [-o data/held_grip.csv]

FILE is a path inside the raw-data repository (e.g. I479-mac/schur_D79.7657_exact.json), written by
`held_schur.py --jac=exact --json=...`. One row per singular pair of the Schur complement S = dg/dh: the state's D, the
harmonic its direction mostly lies on, the singular value from the exact Jacobian and from central differences at eps
and eps/2, the full Newton step along it, and the source file. Refuses a file made without --jac=exact, one whose
held_schur check failed, and one where a harmonic's two singular values (its cosine and sine) differ by more than 1%
(exit 1).
"""
import argparse
import csv
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "held_grip.csv"))
    a = ap.parse_args()
    rows = []
    for rel in a.files:
        d = json.load(open(os.path.join(a.raw, rel)))
        if d.get("jac") != "exact" or not d.get("sigma_exact"):
            sys.exit(f"{rel}: not made with --jac=exact")
        # each harmonic's cosine and sine give a near-equal pair of singular values; a broken pair means the
        # arithmetic ran out of digits (quad-double past D 190)
        pairs = {}
        for r in d["directions"]:
            pairs.setdefault(r["harmonic"], []).append(float(r["sigma"]))
        bad = {k: v for k, v in pairs.items() if len(v) != 2 or abs(v[0] - v[1]) > 1e-2 * max(v)}
        if bad:
            sys.exit(f"{rel}: broken cosine-sine pairs " + ", ".join(f"harmonic {k}: " + " ".join(f"{x:.3e}" for x in v)
                                                                     for k, v in sorted(bad.items())))
        log = os.path.join(a.raw, rel[:-len(".json")] + ".log")
        if os.path.exists(log) and "check: pass" not in open(log).read():
            sys.exit(f"{rel}: held_schur's check did not pass ({log})")
        for i, r in enumerate(d["directions"]):
            rows.append(dict(D=f"{float(d['D']):.6f}", harmonic=r["harmonic"], sigma_exact=f"{float(r['sigma']):.6e}",
                             sigma_eps=f"{float(d['sigma_eps'][i]):.6e}", sigma_eps2=f"{float(d['sigma_eps2'][i]):.6e}",
                             newton_step=f"{float(r['step']):.6e}", state=d["state"], source=rel))
    with open(a.o, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.o}: {len(rows)} rows from {len(a.files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
