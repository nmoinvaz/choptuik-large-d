#!/usr/bin/env python3
"""Pseudo-arclength continuation of a CSS branch in (ln q, c0, U(1), D) at fixed eta: follows f_1 through the place
where fixed-D continuation fails, to tell a fold (D turns back) from a blow-up (q -> infinity at finite D).  Records
lambda_1 and, every few steps, the real modes in (-1.2, 0.95) (lambda_{-1} -> 0 at a fold).

    python3 sigma_fold.py NAME ETA D0 q0 c00 U10 lam0 [h] [steps] [dir]
"""
import sys
import json
import math
import numpy as np
from sigma_css import CSS, real_roots

name, eta = sys.argv[1], float(sys.argv[2])
D0, q0, c00, U10, lam0 = (float(v) for v in sys.argv[3:8])
h = float(sys.argv[8]) if len(sys.argv) > 8 else 0.05
steps = int(sys.argv[9]) if len(sys.argv) > 9 else 40
dirn = float(sys.argv[10]) if len(sys.argv) > 10 else 1.0
out = 'data/sigma_fold_%s.jsonl' % name


def F(y):
    lq, c0, U1, D = y
    S = CSS(D, eta)
    S.frozen = False
    S._ss_key = None
    return S.mismatch((math.exp(lq), c0, U1))


def jac(y):
    J = np.empty((3, 4))
    for j in range(4):
        e = 1e-6 * max(1.0, abs(y[j]))
        yp = y.copy(); yp[j] += e
        ym = y.copy(); ym[j] -= e
        J[:, j] = (F(yp) - F(ym)) / (2 * e)
    return J


def tangent(J, t_old=None):
    t = np.linalg.svd(J)[2][-1]
    if t_old is not None and np.dot(t, t_old) < 0:
        t = -t
    return t / np.linalg.norm(t)


# start: converge at fixed D
S = CSS(D0, eta)
z = S.solve((q0, c00, U10))
y = np.array([math.log(z[0]), z[1], z[2], D0])
J = jac(y)
t = tangent(J)
if t[3] * dirn < 0:
    t = -t
lam = S.find_mode(lam0).real
print(json.dumps(dict(start=True, D=D0, q=z[0], lam=lam, t=list(t))), flush=True)
for k in range(steps):
    yp = y + h * t
    yn = yp.copy()
    ok = False
    for it in range(12):
        Fv = F(yn)
        Jn = jac(yn)
        Mb = np.vstack([Jn, t])
        rb = -np.r_[Fv, np.dot(t, yn - yp)]
        d = np.linalg.solve(Mb, rb)
        yn = yn + d
        if np.linalg.norm(d) < 1e-11 * (1 + np.linalg.norm(yn)):
            ok = True
            break
    Fn = np.linalg.norm(F(yn))
    if not ok and Fn > 1e-9:
        h *= 0.5
        print(json.dumps(dict(step=k, retry=True, h=h, F=Fn)), flush=True)
        if h < 1e-4:
            break
        continue
    y = yn
    t = tangent(jac(y), t)
    lq, c0, U1, D = y
    S = CSS(D, eta)
    zz = S.solve((math.exp(lq), c0, U1))
    lam = S.find_mode(lam).real
    rec = dict(name=name, eta=eta, D=D, q=math.exp(lq), c0=c0, U1=U1, F=float(S.F), lam=lam, gamma=1 / lam,
               dD_ds=t[3], RM_rel=S.mode_check(lam)[0])
    if k % 3 == 0:
        roots, _, _ = real_roots(S, -1.2, 0.95, 24)
        rec['low_modes'] = roots
    print(json.dumps(rec), flush=True)
    with open(out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
