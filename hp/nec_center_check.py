#!/usr/bin/env python3
"""The zero structure that Ecker, Ecker & Grumiller's NEC-line angle alpha = 2 arccot(D - 1) assumes (PRD 112,
L021505 (2025), arXiv:2411.09233; notes/LIT_REVIEW.md): the centre field v1(tau) (U/x at the centre, our p1 up to
normalisation, lowD_ecker.py) has two simple zeros per period. Their other two assumptions, Taylor-expandability at
the centre and a zero average, our solver imposes (the centre series; odd harmonics only), so they are not tested
here. For each state: p1's odd harmonics from the half grid (the first M unknowns), the zeros of p1 on a fine grid
over one period, and each zero's simplicity |p1'| / max|p1'|. Exit 1 if any state has other than two zeros or a
zero with simplicity below --min (default 0.1). --mutate adds 2 max|p1| sin(3 theta) to p1: the third harmonic then
dominates and six zeros per period appear, so the check must fail (the control).
    cd hp && python3 nec_center_check.py FILE.npz [FILE.npz ...] [--min 0.1] [--mutate]
"""
import json
import sys
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
mn = float(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 0.1
if "--min" in sys.argv:
    args.remove(sys.argv[sys.argv.index("--min") + 1])
mutate = "--mutate" in sys.argv
bad = 0
print("| state | D | M | zeros per period | least simplicity | |F3/F1| |")
print("|---|---|---|---|---|---|")
for p in args:
    d = np.load(p)
    m = json.loads(str(d["meta"]))
    M = int(m["M"])
    f = np.asarray(d["zh"][:M], float) + np.asarray(d["zl"][:M], float)
    N = 2 * M
    th = 2 * np.pi * np.arange(M) / N
    ks = np.arange(1, M, 2)
    a = np.array([4 / N * np.sum(f * np.cos(k * th)) for k in ks])
    b = np.array([4 / N * np.sum(f * np.sin(k * th)) for k in ks])
    t = np.linspace(0, 2 * np.pi, 20001)[:-1]
    g = (a[:, None] * np.cos(ks[:, None] * t) + b[:, None] * np.sin(ks[:, None] * t)).sum(0)
    gp = (ks[:, None] * (-a[:, None] * np.sin(ks[:, None] * t) + b[:, None] * np.cos(ks[:, None] * t))).sum(0)
    if mutate:
        amp = np.abs(g).max()
        g = g + 2.0 * amp * np.sin(3 * t)
        gp = gp + 6.0 * amp * np.cos(3 * t)
    s = np.sign(g)
    idx = np.where(s != np.roll(s, -1))[0]
    simp = [abs(gp[i]) / np.abs(gp).max() for i in idx]
    least = min(simp) if simp else 0.0
    r31 = float(np.hypot(a[1], b[1]) / np.hypot(a[0], b[0]))
    ok = len(idx) == 2 and least >= mn
    bad += not ok
    print(f"| {p.split('/')[-1]} | {m['D']:.6g} | {M} | {len(idx)} | {least:.3f} | {r31:.4f} |" + ("" if ok else " FAIL |"))
print(f"{len(args) - bad} of {len(args)} states pass")
sys.exit(1 if bad else 0)
