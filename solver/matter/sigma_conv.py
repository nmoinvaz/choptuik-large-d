#!/usr/bin/env python3
"""Numerical convergence of the CSS solver: the f_1 background and its growing mode at one (D, eta), varying the
matching point, the Taylor orders at both ends and the integrator tolerance.

    python3 sigma_conv.py D ETA q c0 s1 lam0
"""
import sys
import json
import numpy as np
from sigma_css import CSS

D, eta = float(sys.argv[1]), float(sys.argv[2])
z0 = tuple(float(v) for v in sys.argv[3:6])
lam0 = float(sys.argv[6])
out = []
for (xm, Kc, Ks, rtol) in [(0.5, 24, 30, 1e-12), (0.4, 24, 30, 1e-12), (0.6, 24, 30, 1e-12), (0.5, 16, 20, 1e-12),
                           (0.5, 32, 40, 1e-12), (0.5, 24, 30, 1e-10)]:
    S = CSS(D, eta, xm=xm, Kc=Kc, Ks=Ks, rtol=rtol, atol=rtol * 0.1)
    z = S.solve(z0)
    lam = S.find_mode(lam0).real
    rec = dict(D=D, eta=eta, xm=xm, Kc=Kc, Ks=Ks, rtol=rtol, q=z[0], c0=z[1], s1=z[2], lam=lam, F=S.F)
    print(json.dumps(rec), flush=True)
    out.append(rec)
lams = np.array([r['lam'] for r in out])
qs = np.array([r['q'] for r in out])
print("spread: lam %.2e  q %.2e" % (lams.max() - lams.min(), qs.max() - qs.min()))
with open('data/sigma_conv.jsonl', 'a') as f:
    for r in out:
        f.write(json.dumps(r) + '\n')
