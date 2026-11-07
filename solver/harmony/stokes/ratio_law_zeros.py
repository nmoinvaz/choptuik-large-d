"""(a) Zeros of the pure ratio-law function S_RL(x) = sum_k kappa^k x^k / Gamma(k + beta) = E_{1,beta}(kappa x)
(Mittag-Leffler), for (kappa, beta) of the k <= 12 calibration (paper) and of k = 21 (1/increment); against the
two-endpoint asymptotic zeros e^z z^(2-beta) Gamma(beta-1) = 1, z = kappa x (endpoints 0 and kappa of the Borel segment).
Control: the asymptotic formula must reproduce the exact zeros of E to O(1/|z|) at large m, and must fail if beta is
mutated by +0.5.
(b) Two-endpoint spacing: x1 from x0 by (kappa - zeta)(x1 - x0) + (1 - beta - q) ln(x1/x0) = 2 pi i, q = 0, against
the Borel-Pade x1 = 12.755+1.118i."""
import sys
import mpmath as mp
import zerolib as Z
mp.mp.dps = 40

def E(z, beta, N=400):
    return mp.nsum(lambda k: z ** k / mp.gamma(k + beta), [0, mp.inf]) if abs(z) > 60 else \
        mp.fsum(z ** k / mp.gamma(k + beta) for k in range(N))

def asym_zero(m, beta):
    z = 2j * mp.pi * m
    for _ in range(60):
        z = 2j * mp.pi * m + (beta - 2) * mp.log(z) - mp.loggamma(beta - 1)
    return z

ok = True
cases = {"k<=12 (paper)": (mp.mpc("1.279", "1.081"), mp.mpc("4.16", "-1.60")),
         "k=21 (1/incr)": (mp.mpc("1.563", "1.125"), mp.mpc("6.849", "-3.514"))}
for lab, (kap, beta) in cases.items():
    f = lambda x: E(kap * x, beta)
    print(f"\n{lab}: kappa = {Z.cnum(kap,3)}, beta = {Z.cnum(beta,3)}")
    rows = []
    for m in list(range(-4, 0)) + list(range(1, 6)):
        za = asym_zero(m, beta)
        try:
            ze = mp.findroot(lambda z: E(z, beta), za, tol=mp.mpf(10) ** -25, maxsteps=100)
        except Exception as e:
            ze = None
        xa = za / kap
        xe = ze / kap if ze is not None else None
        rows.append((m, xa, xe))
        print(f"  m={m:+d}: asymptotic x = {Z.cnum(xa,4)}   exact x = {Z.cnum(xe,4) if xe is not None else '-'}   "
              f"|z| = {float(abs(za)):.1f}  rel diff {float(abs(xa-xe)/abs(xe)) if xe is not None else float('nan'):.2e}  A = {Z.cnum(mp.sqrt(xe),3) if xe is not None else '-'}")
    if lab.startswith("k<=12"):
        big = [r for r in rows if abs(r[0]) >= 4 and r[2] is not None]
        ok &= all(abs(xa - xe) / abs(xe) < 0.02 for _, xa, xe in big)
        # mutation: beta + 0.5 in the asymptotic formula only
        mut = [abs(asym_zero(m, beta + 0.5) / kap - xe) / abs(xe) for m, xa, xe in big]
        print("  control (beta+0.5 in the formula): rel diffs", [f"{float(v):.2e}" for v in mut])
        ok &= min(mut) > 0.02
    # nearest zero to the real-axis point 7.394
    xs = [r[2] for r in rows if r[2] is not None]
    xn = min(xs, key=lambda x: abs(x - 7.394))
    print(f"  zero nearest x0 = 7.394: {Z.cnum(xn,4)} (A = {Z.cnum(mp.sqrt(xn),3)}); smallest |x|: {float(min(abs(x) for x in xs)):.3f}")

print("\n(b) two-endpoint spacing, q = 0: x1 predicted from x0 = 7.39408")
x0 = mp.mpf("7.39408"); x1bp = mp.mpc("12.755", "1.118")
for lab, kap, beta in (("k<=12 kappa, beta", mp.mpc("1.279", "1.081"), mp.mpc("4.16", "-1.60")),
                       ("k=21 kappa, beta", mp.mpc("1.563", "1.125"), mp.mpc("6.849", "-3.514"))):
    for zl, zeta in (("[6/6] near pole 0.749+0.155i", mp.mpc("0.749", "0.155")), ("K22 near pole 0.705+0.055i", mp.mpc("0.705", "0.055"))):
        F = lambda x: (kap - zeta) * (x - x0) + (1 - beta) * mp.log(x / x0) - 2j * mp.pi
        x1 = mp.findroot(F, x1bp)
        print(f"  {lab}; {zl}: x1 = {Z.cnum(x1,3)}  (Borel-Pade 12.755+1.118i; miss {float(abs(x1-x1bp)):.2f})")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
