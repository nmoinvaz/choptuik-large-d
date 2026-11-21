#!/usr/bin/env python3
"""Ecker, Ecker, Grumiller & Jechtl's published low-D data (github.com/EckerChristian/CriticalSpacetimeCrystal, read
only, not included here) against the sonic closing (notes/LOW_D_SONIC.md, stage 0).

    git clone --depth 1 https://github.com/EckerChristian/CriticalSpacetimeCrystal ecc
    python3 -I ecker_gap.py ecc

Their inner code marches V with the denominator f - x (code/delta/inner/derivs.f; f_x = (D - 3) f (1 - 1/a^2) / x is
our c_x, and f = 1 at x = 1, so f is our c and x is our x). Prints (1) min over their exported tau samples of f - x at
each exported x of plots/fields_<D>_0.json, the smallest over x and where; (2) per D of complete_data/CppData/*.json,
the tau points, the left and right step counts, the matching point and Delta, lowest D first; (3) every D below 3.046
in any of their plots/*.dat files.
"""
import glob
import json
import os
import sys


def main():
    ecc = sys.argv[1]
    for fn in sorted(glob.glob(os.path.join(ecc, "plots", "fields_*_0.json"))):
        d = json.load(open(fn))
        xs = sorted([k for k in d if k.replace(".", "").isdigit()], key=float)
        best = None
        for k in xs:
            f = d[k]["f"]
            x = float(k)
            g = [fi - x for fi in f]
            j = min(range(len(g)), key=g.__getitem__)
            if best is None or g[j] < best[0]:
                best = (g[j], x, j, len(g))
        print(f"{os.path.basename(fn)}: Delta {d['Delta']}, {len(xs)} x samples; min (f - x) {best[0]:.4e} at x {best[1]:.6f} "
              f"(tau sample {best[2]} of {best[3]})")
    rows = []
    for fn in sorted(glob.glob(os.path.join(ecc, "complete_data", "CppData", "*.json"))):
        d = json.load(open(fn))
        for k, v in d.items():
            rows.append((float(k), v.get("Ntau"), v.get("NLeft"), v.get("NRight"), v.get("XLeft"), v.get("XMid"),
                         v.get("XRight"), v.get("Converged"), v.get("mismatchNorm"), v["Initial_Conditions"]["Delta"],
                         os.path.basename(fn)))
    rows.sort()
    print("CppData (D, Ntau, NLeft, NRight, XLeft, XMid, XRight, converged, mismatch, Delta, file), lowest 6 D and every"
          " 10th:")
    for i, r in enumerate(rows):
        if i < 6 or i % 10 == 0:
            print("  ", r)
    print(f"  lowest D in CppData: {rows[0][0]}")
    low = []
    for fn in sorted(glob.glob(os.path.join(ecc, "plots", "*.dat"))):
        for line in open(fn):
            p = line.split()
            try:
                D = float(p[0])
            except (ValueError, IndexError):
                continue
            if 2.5 < D < 3.046:
                low.append((os.path.basename(fn), line.strip()))
    print("plots/*.dat rows with 2.5 < D < 3.046:")
    for fn, line in low:
        print("  ", fn, line)


if __name__ == "__main__":
    main()
