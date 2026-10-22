"""Is a given Pade pole of a Borel transform stable when the number of terms changes? (proof/B3_borel.md, Section 5)

For a series of a b3_outer_*.py JSON file it lists, for K in --Ks, the [L/L] Pade poles (L = (K-1)//2, exact doublets
dropped) within distance --r of a target point. A genuine singularity keeps a pole near the same place for every K; a
pole drawn by the approximant (a cut wandering through the region outside the disc of convergence) moves.

    python3 b3_pole_stability.py file.json NAME --x=0.8 --target=1.653,-0.696 [--r=0.4] [--Ks=100,90,80,70]
"""
import json
import sys

import mpmath as mp

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FN, NAME = [a for a in sys.argv[1:] if not a.startswith("--")][:2]
mp.mp.dps = 120
tr, ti = (mp.mpf(v) for v in ARGS["target"].split(","))
target = mp.mpc(tr, ti)
R = mp.mpf(ARGS.get("r", "0.4"))
d = json.load(open(FN))
key = [k for k in d["series"] if abs(float(k) - float(ARGS["x"])) < 1e-9][0]
cs0 = [mp.mpc(re, im) for re, im, r in d["series"][key][NAME]]
for K in [int(v) for v in ARGS.get("Ks", "100,90,80,70").split(",")]:
    cs = cs0[:K]
    while cs[0] == 0:
        cs = cs[1:]
    b = [cs[k] / mp.factorial(k) for k in range(len(cs))]
    sc = abs(b[-1]) ** (-mp.mpf(1) / (len(b) - 1))
    bs = [bk * sc ** k for k, bk in enumerate(b)]
    L = (len(bs) - 1) // 2
    p, q = mp.pade(bs[:2 * L + 1], L, L)
    P = [z * sc for z in mp.polyroots(q[::-1], maxsteps=3000, extraprec=600)]
    Z = [z * sc for z in mp.polyroots(p[::-1], maxsteps=3000, extraprec=600)]
    near = sorted([z for z in P if abs(z - target) < R and min(abs(z - w) for w in Z) / abs(z) > 1e-30],
                  key=lambda z: abs(z - target))
    print(f"{NAME} x = {ARGS['x']} K = {K} (L = {L}): poles within {R} of {mp.nstr(target, 5)}: "
          + (", ".join(mp.nstr(z, 5) for z in near) if near else "none"), flush=True)
