"""Checks of pert.series_pert: (1) a = derivative of the background along the formal family (finite difference in F)
at lam = 0 for f = dF; (2) theta-translation: f = F' at lam = 0 gives a_j = d_j' exactly."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inner_borel"))
import mpmath as mp
from flint import acb
from pert import series_pert
from inner_stokes import W0, to_mp
A = mp.mpf("2.542148"); w = mp.mpf("0.776994") * W0
J, M = 20, 2
F = {1: -0.5j * float(A), -1: 0.5j * float(A)}
# theta-translation: F' = A cos -> harmonics A/2 each
Fp = {1: 0.5 * float(A), -1: 0.5 * float(A)}
d, a, g, h = series_pert(F, Fp, float(w), 0, J, M)
I = acb(0, 1)
err = 0
for j in (1, 5, 10, 20):
    for mu in (-1, 1):
        dd = d[j][mu + M] * I * mu   # d/dtheta of harmonic mu
        err = max(err, float(abs(to_mp(a[j][mu + M]) - to_mp(dd)) / abs(to_mp(dd))))
print("theta-translation zero mode, lam = 0: max rel |a_j - d_j'| =", err)
# finite-difference check: f = dF = sin theta * eps (amplitude change), lam = 0
eps = 1e-30
F2 = {1: -0.5j * (float(A)) , -1: 0.5j * float(A)}
from flint import ctx
ctx.prec = 900
Fa = {1: acb(0, -0.5) * (acb(str(A)) + acb(eps)), -1: acb(0, 0.5) * (acb(str(A)) + acb(eps))}
Fb = {1: acb(0, -0.5) * acb(str(A)), -1: acb(0, 0.5) * acb(str(A))}
dA, _, _, _ = series_pert(Fa, {1: 0, -1: 0}, float(w), 0, J, M)
dB, aB, _, _ = series_pert(Fb, {1: acb(0, -0.5), -1: acb(0, 0.5)}, float(w), 0, J, M)
err = 0
for j in (1, 5, 10, 20):
    fd = (dA[j][1 + M] - dB[j][1 + M]) / acb(eps)
    err = max(err, float(abs(to_mp(fd) - to_mp(aB[j][1 + M])) / abs(to_mp(aB[j][1 + M]))))
print("amplitude direction, lam = 0: finite difference vs perturbation series, max rel err =", err)
