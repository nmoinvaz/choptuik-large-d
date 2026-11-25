#!/usr/bin/env python3
"""The harmonics of the centre field p1 at two full-system solutions, for the film's scenes on harmonics.

    <python with numpy and mpmath> video/harmonics_data.py RAWDIR [-o video/harmonics.csv] [--mutate=drop13]

RAWDIR is a clone of the raw-data repository (AGENTS.md). The states and the Fourier sums are those of
hp/plot_wave_family.py (hp/held_fit.py's full-system states, every harmonic solved): the lowest state at D >= 38, the
wave that figure builds up harmonic by harmonic, and the state nearest D = 60.96, where the first wall stood.
One row per state and odd harmonic k: D, Delta, k, the cosine and sine coefficients over the echo phase
theta = 2 pi tau / Delta, and the size sqrt(a^2 + b^2). Checks that the harmonics rebuild the stored samples, exit 1 if
not. --mutate=drop13 leaves harmonic 13 out of the rebuild, which must fail (and writes no file).
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hp"))
import plot_wave_family as wf  # noqa: E402
from held_fit import full_states, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("-o", default=os.path.join(HERE, "harmonics.csv"))
    ap.add_argument("--mutate", default="")
    a = ap.parse_args()
    full = [p for p in full_states(a.raw) if float(p["D"]) >= 38.0]
    picks = [full[0], min(full, key=lambda p: abs(float(p["D"]) - 60.96))]
    ok, rows = True, []
    for p in picks:
        s = wf.state(float(p["D"]), "full", p["path"])
        D, z, m, delta = load(p["path"])
        f = np.asarray(z, float)[:wf.M]
        th = np.pi * np.arange(wf.M) / wf.M
        err = float(np.max(np.abs(wf.wave(s["co"], th, kmax=11 if a.mutate == "drop13" else None) - f)))
        ok &= err < 1e-12 * float(np.max(np.abs(f)))
        print(f"D {s['D']:.6f}  Delta {s['delta']:.10f}  rebuild error {err:.1e}  {os.path.relpath(p['path'], a.raw)}")
        for k in sorted(s["co"]):
            ak, bk = s["co"][k]
            rows.append((s["D"], s["delta"], k, ak, bk, math.hypot(ak, bk)))
            print(f"   k {k:2d}  size {math.hypot(ak, bk):.3e}")
    if a.mutate:
        print(f"mutation {a.mutate}: rebuild check {'PASS' if ok else 'FAIL'} (must fail)")
        sys.exit(0 if ok else 1)
    with open(a.o, "w") as fh:
        fh.write("D,Delta,k,cos,sin,size\n")
        for r in rows:
            fh.write(f"{r[0]:.6f},{r[1]:.12f},{r[2]},{r[3]:.17e},{r[4]:.17e},{r[5]:.6e}\n")
    print(f"{len(rows)} rows to {os.path.relpath(a.o)}; rebuild check {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
