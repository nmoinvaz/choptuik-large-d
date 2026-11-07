#!/usr/bin/env python3
"""The control D2 of certify_M2.py, per block, and its midpoint form (review of proof/A_M2.md, proof/A_M2_review.md).

For a saved C^2 run and the finite differences of certify_M2_fd.py (converted to the run's chart as d2check does),
per (row block, pair block): the largest |FD|, radius / |FD|, |FD - mid| / |FD|, 3 err / |FD| (err the Richardson
estimate), the largest contribution to |K_i| / w_i, and the midpoint ratio |FD - mid| / (3 err + 1e-10 |FD|), which
leaves the enclosure's radius out. D2 passes when the FD value lies in the enclosure (within its error); the midpoint
form also fails when a wrong term widens the radius as much as it moves the midpoint (the w0 and chart mutations at
M = 5). The assembly is review_M2_indep.assemble (arb, 768 bits), not certify_M2's.

    python3 review_M2_d2mid.py data/A_M2_box_M5_1e-10.json data/A_M2_fd_M5.json        (M = 5 runs only)
Exit status 1 if the midpoint ratio exceeds 1 anywhere.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import review_M2_indep as R  # noqa: E402
from flint import arb  # noqa: E402

bx = json.load(open(sys.argv[1]))
assert bx["M"] == 5, "M = 5 runs only (review_M2_indep's layout)"
fd = json.load(open(sys.argv[2]))
Y, G1, G2, g0 = R.assemble(bx)
m = 6
D2 = np.array(fd["D2"])
err = np.nan_to_num(np.array(fd["err"]), nan=0.0)
G = [[float(arb(x).mid()) for x in r] for r in bx["Gamma"]]
# Dgt(0) = l V0 midpoints (for the chart conversion, as d2check does)
S = bx["sets"]
n = len(S["c"])
ell = R.jump_rows(n)
LV0 = R.lohner_apply(ell, S, "cV", "QV", "qV")
D1 = np.array([[float(LV0[i][j].mid()) for j in range(m)] for i in range(m)])
D2c = D2.copy()
for j in range(2):
    for k in range(2):
        for i in range(2, m):
            D2c[:, j, k] += D1[:, i] * G[i - 2][j + k]
Yn = np.array([[float(Y[i, j].mid()) for j in range(m)] for i in range(m)])
w = [float(bx["h"])] * 2 + [float(bx["w"][0])] * 2 + [float(bx["w"][1])] * 2
blk = lambda i: "o" if i < 2 else ("z3" if i < 4 else "z5")
rows = {}
for p, (j, k) in enumerate(R.pairs()):
    yfd = Yn @ D2c[:, j, k]
    yerr = np.abs(Yn) @ err[:, j, k]
    for i in range(m):
        z = G2[i][p]
        mid, rad = float(z.mid()), float(z.rad())
        dev = abs(yfd[i] - mid)
        contrib = (1 if j == k else 2) * (abs(mid) + rad) * w[j] * w[k] / w[i]
        key = (blk(i), blk(j) + blk(k))
        r = rows.setdefault(key, [0, 0, 0, 0, 0, 0])
        r[0] = max(r[0], rad / (abs(yfd[i]) + 1e-300))
        r[1] = max(r[1], dev / (abs(yfd[i]) + 1e-300))
        r[2] = max(r[2], contrib)
        r[3] = max(r[3], abs(yfd[i]))
        r[4] = max(r[4], 3 * yerr[i] / (abs(yfd[i]) + 1e-300))
        r[5] = max(r[5], dev / (3 * yerr[i] + 1e-10 * abs(yfd[i]) + 1e-300))
print(f"{'rows':4s} {'pairs':6s} {'max|FD|':>10s} {'rad/|FD|':>10s} {'dev/|FD|':>10s} {'FDerr/|FD|':>10s} {'max contrib':>11s} {'dev/(3err+1e-10|FD|)':>21s}")
for key in sorted(rows):
    r = rows[key]
    print(f"{key[0]:4s} {key[1]:6s} {r[3]:10.2e} {r[0]:10.2e} {r[1]:10.2e} {r[4]:10.2e} {r[2]:11.2e} {r[5]:21.2e}")
print('max dev / (3 err + 1e-10 |FD|) over all entries:', max(r[5] for r in rows.values()))
sys.exit(1 if max(r[5] for r in rows.values()) > 1 else 0)
