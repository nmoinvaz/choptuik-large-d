#!/usr/bin/env python3
"""Fits of the CSS's oscillatory growth rate along the track in powers of 1/n (matter/COMPLEX_CRITICAL.md section 2).

    python3 solver/matter/complex_osc_fit.py ../choptuik-large-d-raw/I95-mac

lambda_osc(D) from css/track_k0*.jsonl (complex_css.py track); for D >= DMIN, least-squares polynomials of degree K in
1/n for Re lambda_osc and for Im lambda_osc / omega.  The spread over (DMIN, K) is a spread over fit choices, not an
error bound.  K = 3, 4 and DMIN = 15, 20, 30 (a first version with K = 2 and DMIN = 10 failed the control below: a
quadratic cannot absorb the planted 1/n^3 term; K = 2 was dropped).  Planted control: the same fits on Re = 0.53 - 0.9/n + 3/n^2 + 10/n^3 - 40/n^5, Im/omega = 2 - 0.6/n + 1/n^2 + 5/n^5 (exact
values at the track's D) must return 0.53 and 2 to 1e-3 for the fits used; exit 1 otherwise.
"""
import json
import os
import sys

import numpy as np


def load(R):
    pts = []
    for f in ("css/track_k0.jsonl", "css/track_k0_b.jsonl"):
        p = os.path.join(R, f)
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l)
                if r.get("kind") == "track" and r["mode"] == 0:
                    pts.append((r["D"], r["lam"][0], r["lam"][1] / r["omega"]))
    pts.sort()
    return np.array(pts)


def fits(D, y, dmins=(15, 20, 30), Ks=(3, 4)):
    out = {}
    for dmin in dmins:
        sel = D >= dmin
        if sel.sum() < 6:
            continue
        x = 1.0 / (D[sel] - 2)
        for K in Ks:
            c = np.polyfit(x, y[sel], K)
            out[(dmin, K)] = c[-1]
    return out


def main():
    P = load(sys.argv[1])
    D, re, imw = P[:, 0], P[:, 1], P[:, 2]
    n = D - 2
    ok = True
    for name, y, planted in (("Re lambda_osc", re, 0.53 - 0.9 / n + 3 / n ** 2 + 10 / n ** 3 - 40 / n ** 5),
                             ("Im lambda_osc / omega", imw, 2 - 0.6 / n + 1 / n ** 2 + 5 / n ** 5)):
        f = fits(D, y)
        fp = fits(D, planted)
        vals = np.array(list(f.values()))
        print("%s at 1/n = 0: %s" % (name, ", ".join("D>=%d K=%d: %.4f" % (k[0], k[1], v) for k, v in f.items())))
        print("   range %.4f .. %.4f (spread over fit choices)" % (vals.min(), vals.max()))
        target = 0.53 if name.startswith("Re") else 2.0
        dev = max(abs(v - target) for v in fp.values())
        print("   planted control: largest deviation of the fits from %.2f: %.1e" % (target, dev))
        ok = ok and dev < 1e-3
    print("planted control", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
