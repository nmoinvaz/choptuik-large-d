"""Which rows to drop when the weak harmonics of p1 are held (#1090): at a start, the condition number (rows scaled)
of the held system for each choice of the dropped rows (harmonics k of the p, m, U, V or c rows at x*), and a few
damped Newton steps of each held system, float64.

Usage: python3 pairing_test.py START.json --D=200 --M=8 --hold=5,7 [--a=8]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mobius_s as ms  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
js = json.load(open([a for a in sys.argv[1:] if not a.startswith("--")][0]))
D, M = float(opts["D"]), int(opts["M"])
hold = [int(v) for v in opts["hold"].split(",")]
y = np.concatenate([ms.resample_odd(np.array(js["p1"]), M), ms.resample_even(np.array(js["c0"]), M),
                    ms.resample_odd(np.array(js["Q0"]), M), [js["Delta"]]])
th = np.pi * np.arange(M) / M
cols, lab = [], []
for k in range(1, M, 2):
    cols += [np.cos(k * th), np.sin(k * th)]
    lab += [(k, "c"), (k, "s")]
T = np.array(cols).T
Ti = np.linalg.inv(T)
# even fields (c rows) in harmonics 0, 2, ..., on the half grid of period pi
ce, lab_e = [], []
for k in range(0, M, 2):
    ce.append(np.cos(k * th))
    lab_e.append((k, "c"))
    if 0 < k < M:
        ce.append(np.sin(k * th))
        lab_e.append((k, "s"))
ce.append(np.cos(M * th))
lab_e.append((M, "c"))
Te = np.array(ce[:M]).T
lab_e = lab_e[:M]
Tei = np.linalg.inv(Te)
xs = 1 / np.sqrt(2.0)
L = 3 * M + 1
B = np.eye(L)
B[:M, :M] = T
f = ms.FormS(D, M, a=float(opts.get("a", 8.0)), kb=150)
z0 = np.linalg.solve(B, y)
choices = {
    "p": np.block([[Ti / 2, Ti * xs / 2], [np.zeros((M, M)), np.eye(M)]]),
    "m": np.block([[Ti / 2, -Ti * xs / 2], [np.zeros((M, M)), np.eye(M)]]),
    "U": np.block([[Ti, np.zeros((M, M))], [np.zeros((M, M)), np.eye(M)]]),
    "V": np.block([[np.zeros((M, M)), Ti], [np.eye(M), np.zeros((M, M))]]),
}
F0 = f.residual(y)
J0 = f.jacobian(y)
for name, A2 in list(choices.items()) + [("c", None)]:
    A = np.eye(L)
    if A2 is not None:
        A[:2 * M, :2 * M] = A2
        drop = [i for i, (k, _) in enumerate(lab) if k in hold]
    else:
        A[2 * M:3 * M, 2 * M:3 * M] = Tei
        drop = [2 * M + i for i, (k, _) in enumerate(lab_e) if k in [h - 1 for h in hold]]  # pair h_k with c's k-1
    cols_drop = [i for i, (k, _) in enumerate(lab) if k in hold]
    keep_r = [i for i in range(L) if i not in drop]
    keep_c = [i for i in range(L) if i not in cols_drop]
    J = A @ J0 @ B
    Jk = J[np.ix_(keep_r, keep_c)]
    Jr = Jk / np.linalg.norm(Jk, axis=1)[:, None]
    sv = np.linalg.svd(Jr, compute_uv=False)
    z = z0.copy()
    hist = []
    for it in range(5):
        Fz = A @ f.residual(B @ z)
        Jz = A @ f.jacobian(B @ z) @ B
        dz = np.linalg.solve(Jz[np.ix_(keep_r, keep_c)], -Fz[keep_r])
        lam = 1.0
        for _ in range(10):
            zt = z.copy()
            zt[keep_c] += lam * dz
            Ft = A @ f.residual(B @ zt)
            if np.max(np.abs(Ft[keep_r])) < np.max(np.abs(Fz[keep_r])):
                break
            lam *= 0.5
        z = zt
        hist.append("%.1e" % np.max(np.abs(Ft[keep_r])))
    print("drop %s rows: held cond %.1e (sv_min %.1e); damped Newton %s; Delta %.10f"
          % (name, sv[0] / sv[-1], sv[-1], " ".join(hist), (B @ z)[3 * M]), flush=True)
Jr = J0 / np.linalg.norm(J0, axis=1)[:, None]
sv = np.linalg.svd(Jr, compute_uv=False)
print("full system: cond %.1e (sv_min %.1e)" % (sv[0] / sv[-1], sv[-1]))
