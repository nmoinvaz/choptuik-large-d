#!/usr/bin/env python3
"""Is the complex mode matrix smooth in lambda at the ulp scale (#605)? At a point, the double-double matrix is
assembled at Re lambda, one f64 ulp up and two ulps up (CSolver.mode_matrix). For a smooth assembly each entry's
two-ulp change d2 is twice its one-ulp change d1; the entries where |d2 - 2 d1| > |d1| / 2 are counted, located by
block of 4M rows and columns, and the largest is printed.

    cd hp && python3 modecount_smooth.py STATE.npz M KF RE IM
"""
sys.path.insert(0, "/Users/nathan/Source/choptuik-large-d/hp")
from modecount_common import setup
path, M, kf, re0, im0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
st = setup(path, M=M, threads=4, kf=kf, fmax=1e-15, log=lambda *a: None)
def mat(re_):
    rh, rl, ih, il = st.c.mode_matrix(st.zh, st.zl, re_, im0)
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
