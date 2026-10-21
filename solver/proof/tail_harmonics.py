#!/usr/bin/env python3
"""Gap (b) diagnostics from the amplitude expansion (every harmonic exact): the tuned harmonics f_nu(A) at the
selected point, their decay in nu, and the leading-order ratio q_nu = f_{nu+2}^{(nu+2)} / f_nu^{(nu)}.

f["k_nu"] in scan25phys/sigF_w*_K25_J750.json is the A^k coefficient of the e^{i nu theta} harmonic of F
(F = A sin(theta) + sum_{nu>=3} f_nu(A) e^{i nu theta} + c.c., F_1 = A/(2i)).

    python3 tail_harmonics.py [path-to-json] [A]
"""
import json
import sys
import cmath
import math

path = sys.argv[1] if len(sys.argv) > 1 else "../inner_amp/scan25phys/sigF_w1.00000_K25_J750.json"
A = float(sys.argv[2]) if len(sys.argv) > 2 else 2.7196
r = json.load(open(path))
K = r["K"]
f = {tuple(int(t) for t in k.split("_")): complex(*v) for k, v in r["f"].items()}
F1 = A / 2j
print(f"wf {r['wf']}  K {K}  A {A}")
print(" nu   leading q_nu = f_{nu+2}^{(nu+2)}/f_nu^{(nu)}     |q| A^2     f_nu(A)/F_1 (summed to A^K)     |F_nu/F_{nu-2}|   terms")
prev = F1
for nu in range(3, K + 1, 2):
    s = sum(f[(k, nu)] * A ** k for k in range(nu, K + 1, 2) if (k, nu) in f)
    nterm = len([k for k in range(nu, K + 1, 2) if (k, nu) in f])
    q = f[(nu + 2, nu + 2)] / f[(nu, nu)] if (nu + 2, nu + 2) in f else None
    qs = f"{q.real:+.6f}{q.imag:+.6f}i   {abs(q) * A * A:.5f}" if q is not None else " " * 30
    print(f"{nu:3d}   {qs:40s}  {(s / F1).real:+.3e}{(s / F1).imag:+.3e}i   {abs(s / prev):.5f}   {nterm}")
    prev = s
# convergence of each f_nu(A) series in A: ratio of successive terms at A
print("\nsuccessive-term ratios |f_nu^{(k+2)} A^2 / f_nu^{(k)}| at A (convergence of the A-series of each harmonic):")
for nu in (3, 5, 7, 9, 11):
    rs = [abs(f[(k + 2, nu)] * A * A / f[(k, nu)]) for k in range(nu, K - 1, 2) if (k, nu) in f and (k + 2, nu) in f]
    print(f"  nu {nu:2d}: " + " ".join(f"{x:.3f}" for x in rs))
