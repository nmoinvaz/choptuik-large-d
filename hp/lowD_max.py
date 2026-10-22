#!/usr/bin/env python3
"""The maximum of Delta(D) near D = 3.756 from the double-double points at M = 80 (data/lowD_dd_local_max.jsonl, the
local run T35 part D; the D = 3.70 reference point from data/lowD_dd.jsonl), by least-squares polynomials in D.

    cd hp && python3 lowD_max.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    pts = {}
    for line in open(os.path.join(HERE, "data", "lowD_dd_local_max.jsonl")):
        r = json.loads(line)
        pts[r["D"]] = r["Delta"]
    for line in open(os.path.join(HERE, "data", "lowD_dd.jsonl")):
        r = json.loads(line)
        if abs(r["D"] - 3.7) < 1e-9 and r["M"] == 80:
            ref = r["Delta"]
    res = []
    for label, data, deg in (("quartic, 3.72-3.80", pts, 4), ("quartic, 3.70-3.80", {**pts, 3.7: ref}, 4),
                             ("quintic, 3.70-3.80", {**pts, 3.7: ref}, 5)):
        D = np.array(sorted(data)); y = np.array([data[d] for d in D])
        c = np.polyfit(D - 3.75, y, deg)
        roots = [r.real + 3.75 for r in np.roots(np.polyder(c)) if abs(r.imag) < 1e-12 and 3.72 < r.real + 3.75 < 3.78]
        dm = roots[0]; ym = np.polyval(c, dm - 3.75)
        resid = np.abs(y - np.polyval(c, D - 3.75)).max()
        res.append((dm, ym))
        print(f"{label}: D_max = {dm:.6f}, Delta_max = {ym:.10f} (max residual {resid:.1e})")
    ds, ys = [r[0] for r in res], [r[1] for r in res]
    print(f"spread over the fits: D {max(ds) - min(ds):.1e}, Delta {max(ys) - min(ys):.1e}")


if __name__ == "__main__":
    main()
