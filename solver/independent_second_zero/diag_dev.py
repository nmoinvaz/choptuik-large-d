"""Development diagnostic on the planted family (not a control): every qhp/pade approximant's zero, admitted or not,
against the exact zero, with and without the 1e-12 noise.

    python3 diag_dev.py WP SEED NOISE
"""
import sys
import os

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lapl  # noqa: E402
import planted  # noqa: E402

wp, seed, rel = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
fam = planted.Family(wp)
w = mp.mpf("1.10")
xt = fam.x1(w)
mp.mp.dps = 50
sig = fam.sigma(w)
if rel > 0:
    sig = planted.noisy(sig, seed, rel)
mp.mp.dps = lapl.DPS
c = lapl.laplace_coeffs(sig)
zds, gds = lapl.domb_sykes(c, len(c) - 3)
print("true x1(1.10)", mp.nstr(xt, 12), "zeta_DS", mp.nstr(zds, 6), "noise", rel)
for tot in (21, 20, 19):
    nd = tot + 2
    for (p, q, r) in lapl.qhp_triples(tot):
        ct, info = lapl.qhp_extend(c[:nd], p, q, r, 220)
        ok, why = lapl.admissible(c, ct, zds, 13.5, info, nd)
        x1, zi = lapl.find_zero(ct, 13.5)
        print(f"qhp {p},{q},{r} nd {nd} ok {ok} err {mp.nstr(x1 - xt, 3)} growth {mp.nstr(lapl.growth(ct, 220), 4)} "
              f"g60 {mp.nstr(lapl.growth(ct, 60), 4)} cond {mp.nstr(info['cond'], 3)} {why}")
for tot in (22, 21, 20):
    nd = tot + 1
    for (L, M) in lapl.pade_pairs(tot):
        ct, info = lapl.pade_extend(c[:nd], L, M, 220)
        ok, why = lapl.admissible(c, ct, zds, 13.5, info, nd)
        x1, zi = lapl.find_zero(ct, 13.5)
        print(f"pade {L},{M} nd {nd} ok {ok} err {mp.nstr(x1 - xt, 3)} growth {mp.nstr(lapl.growth(ct, 220), 4)} {why}")
