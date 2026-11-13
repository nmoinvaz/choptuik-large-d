#!/usr/bin/env python3
"""Is the complex mode matrix smooth in lambda at the ulp scale (#605)? At a point, the double-double matrix is
assembled at Re lambda, one f64 ulp up and two ulps up (CSolver.mode_matrix). For a smooth assembly each entry's
two-ulp change d2 is twice its one-ulp change d1; the entries where |d2 - 2 d1| > |d1| / 2 are counted, located by
block of 4M rows and columns, and the largest is printed.

    cd hp && python3 modecount_smooth.py STATE.npz M KF|none RE IM [--joint] [--mutate=lo]

--joint prints the counted entries' (row block, column block) table, and within a block their field and tau lane.
--mutate=lo, the control: the matrices' low parts are dropped, as an assembly in f64 would leave them, so the one-ulp
changes are whole ulps of the high parts and most changed entries must count as non-linear.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modecount_common import setup
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) != 5:
    sys.exit(__doc__)
path, M, re0, im0 = args[0], int(args[1]), float(args[3]), float(args[4])
kf = None if args[2] == "none" else int(args[2])
mutate = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
if mutate not in (None, "lo"):
    sys.exit(f"--mutate takes lo, not {mutate}")
st = setup(path, M=M, threads=4, kf=kf, fmax=1e-15, log=lambda *a: None)
def mat(re_):
    rh, rl, ih, il = st.c.mode_matrix(st.zh, st.zl, re_, im0)
    if mutate == "lo":
        rl, il = np.zeros_like(rl), np.zeros_like(il)
    return (np.asarray(rh), np.asarray(rl), np.asarray(ih), np.asarray(il))
def diff(A, B):
    # exact-enough difference of two double-double complex matrices, in f64 (each part difference is exact for nearby values)
    return ((B[0] - A[0]) + (B[1] - A[1])) + 1j * ((B[2] - A[2]) + (B[3] - A[3]))
r1 = float(np.nextafter(re0, np.inf)); r2 = float(np.nextafter(r1, np.inf))
A0, A1, A2 = mat(re0), mat(r1), mat(r2)
d1, d2 = diff(A0, A1), diff(A0, A2)
scale = np.abs(A0[0] + 1j * A0[2])
n = d1.shape[0]
mask = np.abs(d1) > 1e-30 * (scale + 1e-300)
rel = np.abs(d2 - 2 * d1) / np.maximum(np.abs(d1), 1e-300)
print(f"{path.split('/')[-1]} at {re0}+{im0}i, n {n}: entries changed by one ulp {mask.sum()} of {n*n}")
for q in (0.5, 0.9, 0.99, 1.0):
    print(f"  quantile {q}: |d2 - 2 d1| / |d1| = {np.quantile(rel[mask], q):.2e}")
bad = mask & (rel > 0.5)
print(f"  entries with |d2 - 2 d1| > |d1|/2: {bad.sum()}")
if bad.sum():
    rows, cols = np.nonzero(bad)
    blk = 4 * M
    print(f"  their rows by block of 4M: {np.bincount(rows // blk)}; cols by block: {np.bincount(cols // blk)}")
    i = np.argmax(np.where(bad, np.abs(d1) / np.maximum(scale, 1e-300), 0))
    r, c = divmod(i, n)
    print(f"  largest: row {r}, col {c}: |A| {scale[r,c]:.2e}, |d1| {abs(d1[r,c]):.2e}, |d2| {abs(d2[r,c]):.2e}")
    if "--joint" in sys.argv:
        # the (row block, column block) pairs, and within a block of 4M the field (p, m, a, c = i // M) and the tau
        # lane (i % M) of the rows and the columns (#612)
        nb = (n + blk - 1) // blk
        joint = np.zeros((nb, nb), dtype=int)
        np.add.at(joint, (rows // blk, cols // blk), 1)
        print(f"  (row block, column block) counts, {joint.sum()} in all:")
        for a in range(nb):
            print("    " + " ".join(f"{x:7d}" for x in joint[a]))
        for lab, idx in (("rows", rows), ("cols", cols)):
            w = idx % blk
            print(f"  {lab} by field (p, m, a, c): {np.bincount(w // M, minlength=4)}; by tau lane: "
                  f"{np.bincount(w % M, minlength=M)}")
