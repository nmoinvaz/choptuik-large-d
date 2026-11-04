#!/usr/bin/env python3
"""The centre profile F = sqrt(n) dp/dx(x = 0) of every given state, as largeD_F.py computes it for eleven states:
|F_1| (the first harmonic, the selection rule's amplitude A at D = infinity), F_3/F_1, F_5/F_1, <F^2> and
n (1 - <c(0)>) (Proposition 1: the last two share one limit), with D and Delta. One CSV row per state, appended to OUT
(states already in OUT are skipped, so a stopped run resumes).
    cd hp && python3 largeD_F_all.py OUT.csv STATE.npz [STATE.npz ...]
"""
import csv
import os
import sys
import numpy as np
from largeD_profiles import solver_for, harmonics

out, states = sys.argv[1], sys.argv[2:]
done = set()
if os.path.exists(out):
    done = {r["state"] for r in csv.DictReader(open(out))}
new = not os.path.exists(out)
with open(out, "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(["state", "D", "M", "Delta", "F1", "F3_over_F1", "F5_over_F1", "F2_mean", "n_one_minus_c0"])
    for f in states:
        key = os.path.relpath(f, os.environ.get("RAW", "/"))
        if key in done:
            continue
        try:
            s, zh, zl, meta = solver_for(f)
        except Exception as e:                       # a layout this solver does not take: say so, keep going
            print(f"skip {key}: {e}", flush=True)
            continue
        M, D = meta["M"], meta["D"]
        n = D - 2
        ch, cl = s.series(zh, zl, 0)
        C = (np.asarray(ch) + np.asarray(cl)).reshape(-1, 4, M)
        Fh = harmonics(C[1, 0] * np.sqrt(n), True)
        amp = np.abs(Fh)
        w.writerow([key, f"{D:.10g}", M, f"{zh[-1] + zl[-1]:.17g}", f"{amp[1]:.15g}", f"{amp[3] / amp[1]:.15g}",
                    f"{amp[5] / amp[1]:.15g}", f"{np.mean((C[1, 0] * np.sqrt(n)) ** 2):.15g}",
                    f"{n * (1 - np.mean(C[0, 3])):.15g}"])
        fh.flush()
        print(f"{key}: D {D:.4f} |F_1| {amp[1]:.10f} F3/F1 {amp[3] / amp[1]:.6f}", flush=True)
