"""Zeros of the Borel-Pade sums of the real sigma_0..sigma_22 (omega = omega_inf) in |x| < R, x = A^2."""
import sys, json, time
import mpmath as mp
import zerolib as Z
mp.mp.dps = 60
R = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
sig = Z.load(1.0)
K = len(sig) - 1
kap, beta, C = Z.ratio_law(sig, lo=12)
print(f"K = {K}; ratio law (k = 12..21): kappa = {Z.cnum(kap,4)}, beta = {Z.cnum(beta,3)}, C = {Z.cnum(C,3)}")
out = {"ratio_law": [str(kap), str(beta), str(C)], "sums": {}}
for m in range(5, 12):
    n = K - m
    t0 = time.time()
    ts, rs, c0 = Z.borel_pade(sig, m, n)
    f = Z.bp_func(ts, rs, c0); fn = Z.bp_numpy(ts, rs, c0)
    zs, wind = Z.zeros_in_disc(fn, f, R=R, h=0.2)
    poles = sorted(ts, key=lambda t: float(mp.re(t)))
    print(f"[{m}/{n}] poles: " + ", ".join(Z.cnum(t, 3) for t in poles))
    print(f"   max|t| = {max(float(abs(t)) for t in ts):.4f}; zeros found {len(zs)}, winding on |x|={R}: {wind}")
    for z in zs:
        print(f"     x = {Z.cnum(z,6)}  |x| = {float(abs(z)):.4f}  arg = {float(mp.arg(z))*180/3.14159265:.2f} deg  A = {Z.cnum(mp.sqrt(z),5)}")
    out["sums"][f"{m}/{n}"] = {"poles": [str(t) for t in poles], "residues": [str(r) for r in rs], "c0": str(c0),
                               "zeros": [str(z) for z in zs], "winding": wind, "seconds": time.time() - t0}
json.dump(out, open(f"../real_zeros_R{int(R)}.json", "w"), indent=1)
