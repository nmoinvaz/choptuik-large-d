"""A start for form S at higher M: a lower-M solution resampled to M, p1's harmonics from --from_k up taken from the
closed-form large-D profile (SECOND_ORDER.md §1.2: F_n = F + K1/(2n) + K2/(2n^2), p1 = F_n/sqrt(n), every odd
harmonic <= 31; solver/inner_pert/data/closed_form_K.json), rotated to the solution's time origin by its fundamental,
and Delta from Track C at D (#1090). Then float64 damped Newton and the float64 split Newton on the full system, as a
test of whether such a start lies in Newton's basin.

Usage: python3 seed_closed.py SOL.json TRACKC.json --D=200 --M=8 [--from_k=7] [--a=8] [--iters=12] [--out=FILE.json]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mobius_s as ms  # noqa: E402
from lsnewton64 import ls_solve  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
fn, tcf = [a for a in sys.argv[1:] if not a.startswith("--")][:2]
D, M = float(opts["D"]), int(opts["M"])
from_k = int(opts.get("from_k", 7))
a = float(opts.get("a", 8.0))
js = json.load(open(fn))
tc = json.load(open(tcf))["values"]
cf = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inner_pert", "data",
                                 "closed_form_K.json")))
n = D - 2
p1 = ms.resample_odd(np.array(js["p1"]), M)
c0 = ms.resample_even(np.array(js["c0"]), M)
Q0 = ms.resample_odd(np.array(js["Q0"]), M)
# harmonics of p1: C_nu with p1 = sum 2 Re(C_nu e^{i nu theta}), theta_j = pi j / M (antiperiodic doubled grid)
full = np.concatenate([p1, -p1])
C = np.fft.rfft(full) / (2 * M)
Fn = {h["nu"]: complex(*h["F"]) + complex(*h["K1"]) / (2 * n) + complex(*h["K2"]) / (2 * n * n) for h in cf["harmonics"]}
th0 = np.angle(C[1]) - np.angle(Fn[1])
scale = abs(C[1]) / abs(Fn[1])
G = C.copy()
for k in range(from_k, M, 2):
    G[k] = scale * Fn[k] * np.exp(1j * k * th0)
p1s = np.fft.irfft(G * (2 * M), n=2 * M)[:M]
Dl = float(tc["%g" % D]["Delta"]) if "%g" % D in tc else js["Delta"]
y = np.concatenate([p1s, c0, Q0, [Dl]])
print("seed: |C_k| of the solution %s; closed-form |C_k| %s; Delta %.12f"
      % (" ".join("%.2e" % abs(C[k]) for k in range(1, M, 2)), " ".join("%.2e" % abs(G[k]) for k in range(1, M, 2)), Dl))
f = ms.FormS(D, M, a=a, kb=150)
F = f.residual(y)
print("start: max|F| %.2e" % np.max(np.abs(F)), flush=True)
for it in range(int(opts.get("iters", 12))):
    J = f.jacobian(y)
    dy = np.linalg.lstsq(J, -F, rcond=None)[0]
    lam, ok = 1.0, False
    for _ in range(14):
        Ft = f.residual(y + lam * dy)
        if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < np.max(np.abs(F)):
            ok = True
            break
        lam *= 0.5
    if not ok:
        break
    y, F = y + lam * dy, Ft
    print("  Newton %2d: max|F| %.2e lambda %g Delta %.12f" % (it, np.max(np.abs(F)), lam, y[3 * M]), flush=True)
    if np.max(np.abs(F)) < 1e-14:
        break
if np.max(np.abs(F)) > 1e-14:
    y, r = ls_solve(f.residual, y, f.jacobian(y), tau=1e-6, tol=1e-15, log=print)
    F = f.residual(y)
print("end: max|F| %.2e Delta %.12f (Track C %s, M-truncation %+.2e)" % (np.max(np.abs(F)), y[3 * M], tc.get("%g" % D, {}).get("Delta"),
      y[3 * M] - float(tc["%g" % D]["Delta"]) if "%g" % D in tc else float("nan")))
if "out" in opts:
    json.dump(dict(D=D, M=M, a=a, Delta=float(y[3 * M]), maxF=float(np.max(np.abs(F))), p1=list(y[:M]),
                   c0=list(y[M:2 * M]), Q0=list(y[2 * M:3 * M])), open(opts["out"], "w"), indent=1)
