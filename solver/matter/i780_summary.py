#!/usr/bin/env python3
"""#780: the tables of COMPLEX_IMAG_DD.md section 6 from the raw folder I780-mac, with the acceptance verdicts.

    python3 i780_summary.py RAWDIR I95_COUNTS     (I95_COUNTS = <raw>/I95-mac/imag/counts.jsonl)

Prints, per D: the largest phase difference between complex_imag.py and complex_imag_dd.py (--dd=0) over the double
counts' points, the windings of both, the sample in full double-double (dd all) against both, and the light-cone
control. Exits 1 when a phase differs by more than 1e-3 rad, a winding differs, or the control does not change B0.
"""
import json
import math
import os
import sys

LIMIT = 1e-3


def recs(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main():
    raw, i95 = sys.argv[1], sys.argv[2]
    ok = True
    double = {(r["state"], r["box_name"]): r for r in recs(i95)
              if r.get("mutate") is None and r.get("plant") is None and r["state"].startswith(("up_D10", "up2_D20"))}
    dd0c = {(r["state"], r["box_name"]): r for r in recs(os.path.join(raw, "counts_dd0.jsonl"))}
    for d, state in (("D20", "up2_D20.0437.npz"), ("D10", "up_D10.0000.npz")):
        rs = [r for r in recs(os.path.join(raw, "dd0_%s.jsonl" % d)) if r.get("kind") == "eval" and "dphase" in r]
        w = max(rs, key=lambda r: abs(r["dphase"]))
        good = abs(w["dphase"]) <= LIMIT
        ok &= good
        print("%s: %d points, largest |dphase| %.2e rad at %s, largest |d ln|det|| %.2e: %s" % (
            d, len(rs), abs(w["dphase"]), w["lam"], max(abs(r["dln"]) for r in rs), "OK" if good else "BAD"))
        for box in ("B0", "U0", "FULL"):
            a, b = double[(state, box)], dd0c[(state, box)]
            same = round(a["winding_g"]) == round(b["winding_g"]) and abs(b["winding_g"] - round(b["winding_g"])) < 1e-6
            ok &= same
            print("   %-4s winding: double %.13f, light-cone block in double-double %.13f (evaluations %d, %d): %s" % (
                box, a["winding_g"], b["winding_g"], a["evaluations"], b["evaluations"], "equal" if same else "DIFFER"))
        dd0 = {tuple(r["lam"]): r for r in rs}
        for r in recs(os.path.join(raw, "ddall_%s.jsonl" % d)):
            r0 = dd0.get(tuple(r["lam"]))
            prop = math.remainder(r["lndet"][1] - r0["lndet"][1], 2 * math.pi) if r0 else float("nan")
            print("   full double-double at %-42s against double %+.2e rad, against dd 0 %+.2e rad" % (
                r["lam"], r["dphase"], prop))
    c = recs(os.path.join(raw, "control_lightcone.jsonl"))[-1]
    fires = round(c["winding_g"]) != 1
    ok &= fires
    print("control: complex_imag.py --mutate=lightcone (P_1 x 1.1), %s B0: winding %.13f against 1: %s" % (
        c["state"], c["winding_g"], "fires" if fires else "DOES NOT FIRE"))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
