"""Exact lambda = 1 mode of the inner problem: delta Z = X Z with X = xi d_xi + omega d_theta (time translation of the
singularity). Checks that series_pert at lam = 1, f = F + omega F' gives a_j = (2j+1) d_j + omega d_j' and
h_j = 2j g_j + omega g_j' exactly (here with gravity, F = 2.72 sin theta + 0.2 sin 3 theta, M = 4)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inner_borel"))
import mpmath as mp
from flint import acb, ctx
from pert import series_pert
from inner_stokes import W0, to_mp
ctx.prec = 600
w = acb(str(W0))
A, B = acb("2.72"), acb("0.2")
M = 4
F = {1: acb(0, -0.5) * A, -1: acb(0, 0.5) * A, 3: acb(0, -0.5) * B, -3: acb(0, 0.5) * B}
I = acb(0, 1)
f = {mu: v + w * I * mu * v for mu, v in F.items()}
d, a, g, h = series_pert(F, f, w, 1, 25, M, prec=600)
err = 0
for j in range(26):
    for mu in range(-M, M + 1):
        k = mu + M
        pred = (2 * j + 1) * d[j][k] + w * I * mu * d[j][k]
        hp = 2 * j * g[j][k] + w * I * mu * g[j][k]
        for A_, P_ in ((a[j][k], pred), (h[j][k], hp)):
            sc = max(1e-300, float(abs(to_mp(P_))))
            if sc > 1e-200:
                err = max(err, float(abs(to_mp(A_ - P_))) / sc)
print("lambda = 1 mode X Z: max relative deviation over j <= 25, all harmonics:", err)
