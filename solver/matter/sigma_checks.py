#!/usr/bin/env python3
"""Two point checks for SIGMA.md: the flat f_1 at D = 7 against Biernat-Bizon-Maliborski (d = 6), and f_1 at
D = 6 -+ 1e-3 (the resonant dimension) for eta = 0.14 and 0.3, where flat space has exactly U(1) = pi/2, U'(1) = 0,
lambda = 3.

    python3 sigma_checks.py
"""
import json
import numpy as np
from sigma_css import CSS

out = open('data/sigma_checks.jsonl', 'a')


def rec(tag, D, eta, z, S, lam):
    r = dict(tag=tag, D=D, eta=eta, q=z[0], c0=z[1], U1=S.U1, V1=S.V1, F=S.F, lam=lam)
    print(json.dumps(r), flush=True)
    out.write(json.dumps(r) + '\n')
    out.flush()


S = CSS(7.0, 0.0)
z = S.solve((6.71508, 1.0, 1.53534))
rec('flat_D7', 7.0, 0.0, z, S, S.find_mode(2.42624).real)
# D = 6 -+ 1e-3 at eta = 0.14 (start from the path solution at D = 6.0002) and eta = 0.3
z6 = np.array([5.997271245211825, 1.0474540251669937, 1.5707877405815123])
for eta in (0.14, 0.3):
    zz = z6
    if eta != 0.14:
        for e in np.linspace(0.14, eta, 5)[1:]:
            S = CSS(5.999, e)
            zz = S.solve(zz)
    for D in (5.999, 6.001):
        S = CSS(D, eta)
        z = S.solve(zz)
        rec('D6', D, eta, z, S, S.find_mode(3.0).real)
