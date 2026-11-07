"""Planted control for the second zero x1. Plant: S(x) = (1 - x/x0)(1 - x/x1) L(x)/L(0), L(x) = int_0^1 rho(s) e^{x g(s)}
g'(s) ds on an arc g through the real [10/12] Borel-Pade poles (64-point Gauss-Legendre, so L is a finite exponential
sum and its zeros are exact), rho(s) = (1-s)^(beta-2) s^(pa) (two endpoint exponents). Taylor coefficients to k = 22
with 1e-12 relative noise, then the same Borel-Pade zero search as for the real data.
Criteria (fixed before the run): PLANT-2 (x1 planted): every ring-free sum [6/16]..[10/12] returns a zero within 0.05 of x1.
PLANT-1 (control, x1 not planted): no sum returns a zero within 0.5 of x1 = 12.75+1.12i. Exit 1 if either fails."""
import sys
import numpy as np
import mpmath as mp
import zerolib as Z
mp.mp.dps = 60
rng = np.random.default_rng(7)
x0 = mp.mpc("7.39408", "0"); x1 = mp.mpc("12.755", "1.118")
beta = mp.mpc("4.16", "-1.60"); pa = mp.mpc("-0.5", "0")
# arc: near end 0.705+0.055i, through 0.935+0.463i, far end 1.279+1.081i (the k <= 12 kappa)
P = [mp.mpc("0.705", "0.055"), mp.mpc("0.935", "0.463"), mp.mpc("1.279", "1.081")]
g = lambda s: P[0] * (1 - s) * (1 - 2 * s) + 4 * P[1] * s * (1 - s) + P[2] * s * (2 * s - 1)
dg = lambda s: P[0] * (4 * s - 3) + 4 * P[1] * (1 - 2 * s) + P[2] * (4 * s - 1)
nodes, wts = zip(*[(mp.mpf(a), mp.mpf(b)) for a, b in zip(*np.polynomial.legendre.leggauss(64))])
# map [-1, 1] -> [0, 1] with a substitution s = u^2 near 0 and 1 - (1-s)... keep plain: weights include rho
S_nodes = [(1 + u) / 2 for u in nodes]; W = [w / 2 for w in wts]
T = [g(s) for s in S_nodes]
R = [w * (1 - s) ** (beta - 2) * s ** pa * dg(s) for s, w in zip(S_nodes, W)]
L = lambda x: mp.fsum(r * mp.exp(t * x) for t, r in zip(T, R))
L0 = L(0)
lcoef = [mp.fsum(r * t ** k for t, r in zip(T, R)) / mp.factorial(k) / L0 for k in range(23)]

def times(c, z):
    return [c[0]] + [c[k] - c[k - 1] / z for k in range(1, len(c))]

ok = True
for label, zs in (("PLANT-2 (x0, x1 planted)", [x0, x1]), ("PLANT-1 (x0 only; control)", [x0])):
    c = lcoef
    for z in zs:
        c = times(c, z)
    c = [ck * (1 + 1e-12 * complex(*rng.normal(size=2))) for ck in c]
    true_f = lambda x, zs=zs: mp.fprod([1 - x / z for z in zs]) * L(x) / L0
    # true zeros of the plant in |x| < 16 (exact exponential sum)
    tz, tw = Z.zeros_in_disc(lambda X: np.vectorize(lambda v: complex(true_f(mp.mpc(v))))(X), true_f, R=16, h=0.5)
    print(f"{label}: true zeros |x|<16: " + "  ".join(Z.cnum(z, 4) for z in tz) + f"  (winding {tw})")
    near = []
    for m in range(6, 11):
        ts, rs, c0 = Z.borel_pade(c, m, 22 - m)
        f = Z.bp_func(ts, rs, c0); fn = Z.bp_numpy(ts, rs, c0)
        zz, w = Z.zeros_in_disc(fn, f, R=16, h=0.2)
        d1 = min(abs(z - x1) for z in zz) if zz else mp.inf
        d0 = min(abs(z - x0) for z in zz) if zz else mp.inf
        near.append(float(d1))
        print(f"   [{m}/{22-m}] zeros |x|<16: " + "  ".join(Z.cnum(z, 4) for z in zz) + f";  dist to x0 {float(d0):.1e}, to x1 {float(d1):.3g}")
    if label.startswith("PLANT-2"):
        ok &= max(near) < 0.05
    else:
        ok &= min(near) > 0.5
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
