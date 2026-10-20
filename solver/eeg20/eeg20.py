"""Ecker-Ecker-Grumiller Eq. (20) on the double-double solutions.

Their NLO condition (PRL 136, 191401, Eq. 20): beta'' + 3 beta' = 0 (tau-derivatives) at the zeros of beta(tau) = Pi(x=0).
In our variables beta is proportional to F(theta) = sqrt(n) dp/dx(x=0), theta = omega tau, so the condition reads
omega F'' + 3 F' = 0 at the zeros of F, i.e. Delta_pred = -2 pi F''/(3 F') there. For each D: F's zeros, F' and F'' there
(from the trigonometric interpolant of the centre-series slope on the tau grid), Delta_pred vs the actual Delta, and
the relative size of the NLO term (omega^2 F'' + 3 omega F')/(3 omega F').
    PYTHONPATH=<repo>/hp/python python3 eeg20.py
"""
import os
import glob, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hp"))
from largeD_profiles import solver_for, harmonics
H = "/home/claude/dss/hp_scan/"   # the runs' saved states (not in the repo; hp/data/seeds holds a few)
rows = []
for t in (12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60):
    fs = sorted(glob.glob(H + f"up*_D{t}.*.npz")) or sorted(glob.glob(H + f"up*_D{t-1}.*.npz"))
    if not fs:
        continue
    s, zh, zl, meta = solver_for(fs[0])
    M, D = meta["M"], meta["D"]; n = D - 2
    Delta = float(zh[-1] + zl[-1]); w = 2 * np.pi / Delta
    ch, cl = s.series(zh, zl, 0)
    C = (np.asarray(ch) + np.asarray(cl)).reshape(-1, 4, M)
    Fh = harmonics(C[1, 0] * np.sqrt(n), True)       # rfft / M of the full-period samples (2M points)
    ks = np.arange(len(Fh))
    # F(theta) = sum_k Re(Fh_k e^{i k theta}) (rfft convention: full = sum_k (X_k e^{ik theta} + c.c.)/(2M), X_k/M = Fh_k)
    def ev(th, d=0):
        return np.real(np.sum(Fh[1:] * (1j * ks[1:]) ** d * np.exp(1j * ks[1:] * th)))
    # zeros of F on [0, pi): bracket and bisect
    grid = np.linspace(0, np.pi, 2001)
    vals = np.array([ev(t) for t in grid])
    zs = []
    for i in range(len(grid) - 1):
        if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
            a, b = grid[i], grid[i + 1]
            for _ in range(60):
                c = 0.5 * (a + b)
                if ev(a) * ev(c) <= 0: b = c
                else: a = c
            zs.append(0.5 * (a + b))
    for z0 in zs:
        F1, F2, F3 = ev(z0, 1), ev(z0, 2), ev(z0, 3)
        Dp = -2 * np.pi * F2 / (3 * F1)
        rel = (w * w * F2 + 3 * w * F1) / (3 * w * F1)
        rows.append((D, Delta, z0, F1, F2, Dp, rel))
        print(f"D {D:7.3f} Delta {Delta:.6f}: zero of F at theta = {z0:.6f}, F' {F1:+.6f}, F'' {F2:+.3e}, F''' {F3:+.4f}; "
              f"Delta_pred = -2pi F''/(3F') = {Dp:+.4e}; (w^2F''+3wF')/(3wF') = {rel:+.4e}", flush=True)
    # symmetry about the zero: odd part check F(z0+d) + F(z0-d)
    d = 0.3
    print(f"      symmetry: [F(z0+d)+F(z0-d)]/[F(z0+d)-F(z0-d)] at d = 0.3: {(ev(zs[0]+d)+ev(zs[0]-d))/(ev(zs[0]+d)-ev(zs[0]-d)):+.3e}; "
          f"|F_k| k=1,3,5: {abs(Fh[1]):.4f} {abs(Fh[3]):.4f} {abs(Fh[5]):.5f}; phase(F3) - 3 phase(F1) = "
          f"{np.angle(Fh[3]) - 3 * np.angle(Fh[1]):+.4f}", flush=True)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eeg20_rows.npy"), np.array(rows))
