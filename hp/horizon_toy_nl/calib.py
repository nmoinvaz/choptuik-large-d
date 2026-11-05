#!/usr/bin/env python3
"""Task 1: the toy's radius law, by hp/pole_test.py's estimator. b_h scaled by 1 + delta, series to 130 orders,
R = u0 / (t_K / t_{K-40})^(1/40), t_k = max(|b_k|, |y_k|) u0^k; fit R = a0 + b ln(1/delta); C = b N (N = n - 1);
Domb-Sykes on b_k: rho_k = A + B/k over the last 60 orders, alpha = 1 + B/A (1 for a simple pole).
Also the true singularity: the first zero of w = 1/b on the real axis, by integrating toward u0 and beyond (the pole
of b on the positive axis for delta > 0)."""
import json, math, sys
sys.path.insert(0, '.')
import mpmath as mp
from mpmath import mpf
import numpy as np
from toy_nl import Toy

DPS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
SS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["17.5", "22.5", "27.5", "31.5", "34.5", "48.5", "70.5", "98.5"]
DEL = [1e-8, 1e-6, 1e-4, 1e-2]
out = open(f"calib_{DPS}.jsonl", "w")
print(f"dps {DPS}; columns: s, D = 2s + 4, N, delta, R (ratio test), R_DS (Domb-Sykes 1/A), alpha_DS, neg frac")
for s in SS:
    mp.mp.dps = DPS
    T = Toy(s, DPS)
    N = float(T.N)
    pts = []
    for d in DEL:
        bh = T.ref["bh"] * (1 + mpf(d))
        B, Y, dB, dY, nt, conv, be, et = T.hseries(bh, T.u0, tang=False, kfix=130, coeffs=True)
        u0 = 0.3
        t = [max(abs(float(be[k])), abs(float(et[k]))) * u0 ** k for k in range(len(be))]
        K = len(t) - 1
        R = u0 / (t[K] / t[K - 40]) ** (1 / 40)
        bk = [float(x) for x in be]
        ks = np.arange(K - 59, K + 1)
        rho = np.array([bk[k] / bk[k - 1] for k in ks])
        neg = float(np.mean(rho < 0))
        Bc, A = np.polyfit(1.0 / ks, rho, 1)
        alpha = 1 + Bc / A
        pts.append((math.log(1 / d), R))
        rec = dict(s=float(s), D=2 * float(s) + 4, N=N, dps=DPS, delta=d, R=R, R_ds=1 / A, alpha=alpha, neg=neg)
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"{s:>5} {2*float(s)+4:6.1f} {N:6.1f} {d:7.0e} R {R:.4f} R_DS {1/A:.4f} alpha {alpha:6.2f} neg {neg:.2f}", flush=True)
    x, y = zip(*pts)
    b, a0 = np.polyfit(x, y, 1)
    resid = max(abs(np.polyval([b, a0], x) - np.array(y)))
    dmax = {u: math.exp(-(u - a0) / b) for u in (0.3, 0.2, 0.15, 0.1)}
    rec = dict(s=float(s), D=2 * float(s) + 4, N=N, dps=DPS, a0=a0, b=b, C=b * N, R0N=a0 * N, resid=resid, delta_max=dmax)
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(f"{s:>5} line: R = {a0:.4f} + {b:.5f} ln(1/delta); C = b N = {b*N:.3f}, R0 N = {a0*N:.2f}, worst resid {resid:.4f}; "
          f"delta_max(u0 0.3) {dmax[0.3]:.1e}, (0.15) {dmax[0.15]:.1e}, (0.1) {dmax[0.1]:.1e}", flush=True)
