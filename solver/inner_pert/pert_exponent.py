"""Diagnostic: effective late-order exponents of the perturbation series in the one-harmonic inner model. Decompose
(a_{j,-1}, a_{j,1}) on the transport eigenvectors (B's transport_modes), c_{j,k}, and print
gamma_eff,k(j) = c_{j+1,k} / (2 c_{j,k}) - j, which tends to the mode's exponent if one exponent dominates.
    python3 pert_exponent.py lam profile(c|s) [J]"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inner_borel"))
import mpmath as mp
from flint import acb
from pert import series_pert
from inner_stokes import W0, to_mp, transport_modes
mp.mp.dps = 120
A = mp.mpf("2.54214805"); wf = mp.mpf("0.77699401"); w = W0 * wf
lam, pr = sys.argv[1], sys.argv[2]
J = int(sys.argv[3]) if len(sys.argv) > 3 else 160
Fh = {1: -0.5j * A, -1: 0.5j * A}
nus, E, V = transport_modes(Fh, w, 1)
gam = [mp.mpf(1) / 2 - E[k] / 2 for k in range(len(nus))]
print("nus", nus, "background exponents", [mp.nstr(g, 8) for g in gam])
Vi = V ** -1
Fa = {1: acb(0, -0.5) * acb(str(A)), -1: acb(0, 0.5) * acb(str(A))}
prof = {"c": {1: acb(0.5), -1: acb(0.5)}, "s": {1: acb(0, -0.5), -1: acb(0, 0.5)}, "0": {1: 0, -1: 0}}[pr]
d, a, _, _ = series_pert(Fa, prof, str(w), lam, J, 2)
src = d if pr == "0" else a
c = []
for j in range(J + 1):
    vec = mp.matrix([to_mp(src[j][nu + 2]) for nu in nus])
    c.append(Vi * vec)
for j in (40, 80, 120, J - 1):
    ge = [c[j + 1][k] / (2 * c[j][k]) - j for k in range(2)]
    print(f"j {j}: gamma_eff mode0 {mp.nstr(ge[0], 8):>26s}  mode1 {mp.nstr(ge[1], 8):>26s}   |c0| {mp.nstr(abs(c[j][0]), 3)} |c1| {mp.nstr(abs(c[j][1]), 3)}",
          flush=True)
