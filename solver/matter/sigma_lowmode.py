#!/usr/bin/env python3
"""Real eigenvalues of f_1 in a window (default -1.2 .. 0.95), e.g. the least-stable decaying mode lambda_{-1},
whose crossing of 0 marks a fold of the branch.  Input: a solution (D, eta, q, c0, s1) from a path file.

    python3 sigma_lowmode.py D ETA q c0 s1 [lo hi N]
"""
import sys
import json
import numpy as np
from sigma_css import CSS, real_roots, resonances

D, eta = float(sys.argv[1]), float(sys.argv[2])
z0 = tuple(float(v) for v in sys.argv[3:6])
lo, hi, N = (float(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8])) if len(sys.argv) > 6 else (-1.2, 0.95, 44)
S = CSS(D, eta)
z = S.solve(z0)
roots, ls, ds = real_roots(S, lo, hi, N)
checked = []
for r in roots:
    rm, sv = S.mode_check(r)
    checked.append((r, rm))
rec = dict(D=D, eta=eta, q=z[0], c0=z[1], s1=z[2], F=S.F, window=[lo, hi], roots=[c[0] for c in checked],
           RM=[c[1] for c in checked], poles=resonances(S.n, lo, hi))
print(json.dumps(rec), flush=True)
with open('data/sigma_lowmodes.jsonl', 'a') as f:
    f.write(json.dumps(rec) + '\n')
