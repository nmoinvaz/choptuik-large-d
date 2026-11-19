#!/usr/bin/env python3
"""P2c review (#1017): this review's (H8) numbers (p2c_h8.py records) against b8m_h8.py's logs, after the own numbers
were committed (2ed9a316), with the bars of P2_remaining.md 5.1 (P3), (P4) and the control K2.

    python3 p2c_h8_compare.py DIR > DIR/logs/compare.log      # DIR: the raw-data folder I1017-mac
"""
import json
import os
import sys

from mpmath import mp, mpf

mp.prec = 128
HERE = os.path.dirname(os.path.abspath(__file__))
PROOF = os.path.dirname(HERE)
NAMES = dict(P="P", Phi="Phi", C="C", theta="th")


def b8(name):
    for line in open(os.path.join(PROOF, name)):
        if line.startswith("{"):
            return json.loads(line)
    raise SystemExit(f"no record in {name}")


def mine(d, tag):
    with open(os.path.join(d, f"p2c_{tag}.json")) as fh:
        return json.load(fh)


def num(x):
    """a number, or the upper end of an interval string '[a, b]' (this review's records keep bounds that way)"""
    s = str(x).strip()
    if s.startswith("["):
        s = s[1:-1].split(",")[1]
    return mpf(s)


def rel(a, b):
    return abs(num(a) - num(b)) / abs(num(b))


def main():
    d = sys.argv[1]
    ok = True
    for d0, b8name in (("0", "b8m_h8_K4_L73.89.log"), ("1e7", "b8m_h8_K4_L73.89_d0_1e7.log")):
        m, B = mine(d, f"h8_d0_{d0}"), b8(b8name)
        print(f"== d0 = {d0}: this review (p2c_h8_d0_{d0}.json) against {b8name}")
        print(f"{'':6s} {'part':6s} {'here':>14s} {'b8m_h8':>14s} {'relative':>10s}  bar, reading")
        for c in ("P", "Phi", "C", "theta"):
            for part, key in (("T", "delta_T"), ("R", "delta_R"), ("delta", "delta")):
                a, b = m[part if part != "delta" else "delta"][c], B[key][NAMES[c]]
                r = rel(a, b)
                if part == "T" and c in ("Phi", "C", "theta"):
                    good = r <= mpf("1e-3")
                    note = f"1e-3 (P3): {'PASS' if good else 'FAIL'}"
                    ok &= good or d0 != "0"
                elif part == "R" and c in ("P", "C", "theta"):
                    good = r <= mpf("1e-2")
                    note = f"1e-2 (P3): {'PASS' if good else 'FAIL'}"
                    ok &= good
                elif part == "R" and c == "Phi":
                    ratio = num(a) / num(b)
                    good = mpf("0.1") <= ratio <= 10
                    note = (f"factor 10 ({'P4' if d0 != '0' else 'P3'}): ratio {mp.nstr(ratio, 6)}, "
                            f"{'PASS' if good else 'FAIL'}")
                    ok &= good
                elif part == "T" and c == "P":
                    note = "expected far below (B8's is a ball width around 0)"
                elif part == "delta" and c == "Phi" and d0 != "0":
                    note = "(P4): reported"
                else:
                    note = ""
                print(f"{c:6s} {part:6s} {mp.nstr(num(a), 9):>14s} {mp.nstr(num(b), 9):>14s} "
                      f"{mp.nstr(r, 3):>10s}  {note}")
    # the centre run against b8m_h8_centre.log's first line (printed to 6 digits)
    pt = mine(d, "h8_d0_0_point")
    line = open(os.path.join(PROOF, "b8m_h8_centre.log")).readline()
    vals = dict(x.split() for x in line.split("delta = ")[1].strip().split(", "))
    print("== the box centre: this review's T parts against b8m_h8_centre.log (6 digits printed)")
    for c in ("P", "Phi", "C", "theta"):
        a, b = mp.nstr(num(pt["T"][c]), 6), vals[NAMES[c]]
        same = mpf(a) == mpf(b)
        ok &= same
        print(f"  {c:6s} here {a:>12s}  b8m_h8 {b:>12s}  {'equal' if same else 'DIFFERENT'}")
    # K2: h_S x (1 + 1e-3)
    k2, m0, B = mine(d, "h8_d0_0_mut_hS"), mine(d, "h8_d0_0"), b8("b8m_h8_K4_L73.89.log")
    r0, r2 = rel(m0["R"]["P"], B["delta_R"]["P"]), rel(k2["R"]["P"], B["delta_R"]["P"])
    flag = r2 > mpf("1e-2")
    print(f"== K2 (h_S x (1 + 1e-3)): R_P here {mp.nstr(num(k2['R']['P']), 9)} against b8m_h8 "
          f"{mp.nstr(num(B['delta_R']['P']), 9)}, relative {mp.nstr(r2, 3)} (unperturbed {mp.nstr(r0, 3)})")
    print(f"   flagged at the protocol's 1e-2 bar: {flag}. A bar between the unperturbed agreement and "
          f"{mp.nstr(r2, 2)} would flag it.")
    print("comparison (P3, P4):", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
