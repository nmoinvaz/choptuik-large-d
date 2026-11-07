"""Two small checks.
(1) Flat space (the trivial CSS background of the scalar problem): self-similar modes e^{s tau} regular at the centre are
    analytic at the light cone iff C(n,s) = 0 (README Proposition 2), i.e. iff 1/(Gamma((1+s)/2) Gamma((2+s)/2)) = 0.
    Count those zeros in a box by the argument principle and report whether all are real (no complex pair, hence no Hopf
    bifurcation from flat space).  --mutate shifts alpha by 0.3i: the zeros must leave the real axis (check must fail).
(2) Oren & Sari (2009) large-beta slope 2 + sqrt(2 gamma/(gamma-1)) against their quoted 4.830, 4.450, 4.237."""
import sys
import mpmath as mp
mp.mp.dps = 30
mut = 0.3j if '--mutate' in sys.argv else 0
f = lambda s: mp.rgamma((1 + s) / 2 + mut) * mp.rgamma((2 + s) / 2)
re0, re1, im1 = -3.5, 1.0, 10.0
corners = [mp.mpc(re0, -im1), mp.mpc(re1, -im1), mp.mpc(re1, im1), mp.mpc(re0, im1), mp.mpc(re0, -im1)]
wind = 0
for a, b in zip(corners[:-1], corners[1:]):
    wind += mp.quad(lambda t: mp.diff(f, a + (b - a) * t) / f(a + (b - a) * t) * (b - a), [0, 0.25, 0.5, 0.75, 1])
nz = wind / (2j * mp.pi)
roots = []
for g in (-1, -2, -3):
    try:
        roots.append(mp.findroot(f, mp.mpc(g + 0.05, 0.05)))
    except Exception:
        pass
allreal = all(abs(mp.im(r)) < 1e-20 for r in roots) and len(roots) == round(float(mp.re(nz)))
print("zeros of 1/(Gamma(a)Gamma(b)) in box: %.6f; roots: %s; all real: %s" % (float(mp.re(nz)), [mp.nstr(r, 8) for r in roots], allreal))
n, Om = 2, 2 * mp.pi / mp.mpf('3.4454524022278212')
C = mp.gamma((n + 1) / 2) * mp.gamma(1j * Om + 1 - n / 2) * f(1j * Om) if not mut else 0
print("|C(n=2, s=i*Omega_C)| = %s (nonzero)" % mp.nstr(abs(C), 6))
for g, q in ((mp.mpf(4) / 3, 4.830), (mp.mpf(3) / 2, 4.450), (mp.mpf(5) / 3, 4.237)):
    print("gamma=%s: 2+sqrt(2g/(g-1)) = %s  (Oren-Sari slope %.3f)" % (mp.nstr(g, 4), mp.nstr(2 + mp.sqrt(2 * g / (g - 1)), 6), q))
sys.exit(0 if allreal else 1)
