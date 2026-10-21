"""Where are the Borel singularities of the outer 1/n series? (hypotheses H-a1, H-a2 of proof/C_selection.md)

Runs the outer recursion of solver/largeD_late_orders.py (F = A sin theta, omega, higher omega_k and K_k = 0) for the
M-harmonic model (all fields projected to harmonics |j| <= M after every order) and for the untruncated recursion, and
reads the late orders of
  - D_k (p - m), harmonic 1, at x = 0.5 and 1.0 (complex series in eps),
  - B_{k+1} (b - 1), harmonic 0 (a real series) and harmonic 2, at x = 0.5, 0.8, 0.95,
  - the highest harmonic kept (M for p - m, M - 1 for b - 1; 7 and 6 untruncated), Borel-Pade only,
by the Domb-Sykes form c_{k+1}/c_k = (k + g)/a: 1/a is the slope of the ratio in k, so a (complex) is the nearest Borel
singularity in eps (its argument: is it on the positive real axis?). For a real series, singularities come as a real one
(ratio of fixed sign) or a conjugate pair (ratio oscillates, with period 2 pi / arg a).

    python3 outer_borel.py M K [A] [w/w_inf]        M = 0 means untruncated (jmax = 2k + 3)
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import largeD_late_orders as L

M = int(sys.argv[1]) if len(sys.argv) > 1 else 5
K = int(sys.argv[2]) if len(sys.argv) > 2 else 26
A = float(sys.argv[3]) if len(sys.argv) > 3 else 2.72
wf = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
om = wf * 2 * np.pi / 1.64389375
th = 2 * np.pi * np.arange(L.NT) / L.NT
jmax = (lambda k: 2 * k + 3) if M == 0 else (lambda k: M)
S, D, G, B, st = L.run(A * np.sin(th), om, K, verbose=False, jmax_of=jmax)


def harm(p, xv, j):
    return np.fft.fft(L.at(p, xv))[j] / L.NT


def domb_sykes(c, label):
    ks = np.arange(len(c))
    r = np.array([c[k + 1] / c[k] if c[k] != 0 else np.nan for k in range(len(c) - 1)])
    # slope of r_k in k over the last few orders -> 1/a ; intercept -> g/a
    out = []
    for k0 in range(len(r) - 6, len(r) - 1):
        s = r[k0 + 1] - r[k0]
        a = 1 / s
        g = r[k0] * a - k0
        out.append((k0, a, g))
    k0, a, g = out[-1]
    sgn = np.sign(np.real(c[-8:])) if np.all(np.abs(np.imag(c)) < 1e-300) else None
    print(f"  {label:34s} a = {a.real:+.4f}{a.imag:+.4f}i  |a| {abs(a):.4f}  arg(a)/pi {np.angle(a) / np.pi:+.3f}"
          f"   (prev: " + ", ".join(f"{abs(x[1]):.3f}@{np.angle(x[1]) / np.pi:+.2f}pi" for x in out[-4:-1]) + ")"
          + (f"  signs of last 8: {''.join('+' if s > 0 else '-' for s in sgn)}" if sgn is not None else ""))


print(f"M {M if M else 'inf'}  K {K}  A {A}  w/w_inf {wf}")
for xv in (0.5, 1.0):
    domb_sykes([harm(D[k], xv, 1) for k in range(1, K + 1)], f"D_k harmonic 1 at x = {xv}")
for xv in (0.5, 0.8, 0.95):
    domb_sykes([harm(B[k + 1], xv, 0).real for k in range(1, K + 1)], f"B_k+1 harmonic 0 at x = {xv}")
    domb_sykes([harm(B[k + 1], xv, 2) for k in range(1, K + 1)], f"B_k+1 harmonic 2 at x = {xv}")


def borel_pade_poles(c, label, nmax=4):
    """poles of a diagonal Pade approximant of the Borel transform sum c_k zeta^k / k! (k from 1): the nearest Borel
    singularities in eps, with their arguments (more robust than ratios when two singularities interfere)."""
    import mpmath as mp
    mp.mp.dps = 30
    b = [mp.mpc(complex(ck)) / mp.factorial(k + 1) for k, ck in enumerate(c)]   # c[0] is order 1
    sc = abs(b[len(b) // 2]) ** (-1.0 / (len(b) // 2)) if abs(b[len(b) // 2]) > 0 else 1
    bs = [bk * mp.mpf(sc) ** k for k, bk in enumerate(b)]                      # rescale zeta -> zeta * sc
    L = (len(bs) - 1) // 2
    p, q = mp.pade(bs, L, len(bs) - 1 - L)
    roots = mp.polyroots(q[::-1], maxsteps=400, extraprec=300) if len(q) > 1 else []
    zeros = mp.polyroots(p[::-1], maxsteps=400, extraprec=300) if len(p) > 1 else []
    # drop Froissart doublets (a pole with a zero within 1e-4 of it: noise in the double-precision data; they move or vanish
    # when K changes. Genuine poles here have their nearest zero at relative distance >= 1e-2, doublets at 1e-6 to 1e-19)
    good = [r for r in roots if not zeros or min(abs(r - z) for z in zeros) > 1e-4 * max(1, abs(r))]
    nd = len(roots) - len(good)
    good = sorted([complex(r) * float(sc) for r in good], key=abs)[:nmax]
    print(f"  {label:34s} Borel-Pade poles: " + ", ".join(f"|a| {abs(r):.4f} arg {np.angle(r) / np.pi:+.3f}pi" for r in good)
          + f"   ({nd} doublets dropped)")


print("Borel-Pade (diagonal, all orders):")
for xv in (0.5, 1.0):
    borel_pade_poles([harm(D[k], xv, 1) for k in range(1, K + 1)], f"D_k harmonic 1 at x = {xv}")
for xv in (0.5, 0.8, 0.95):
    borel_pade_poles([harm(B[k + 1], xv, 0).real for k in range(1, K + 1)], f"B_k+1 harmonic 0 at x = {xv}")
# the highest harmonics kept (|j| = M for p - m, M - 1 for b - 1; 7 and 6 untruncated), where the README's
# "x-independent singularity driven by high harmonics" would show first
jD = M if M else 7
if jD > 1:
    k0 = (jD - 1) // 2      # first order at which harmonic jD can be non-zero
    for xv in (0.5, 1.0):
        borel_pade_poles([harm(D[k], xv, jD) for k in range(k0 + 1, K + 1)], f"D_k harmonic {jD} at x = {xv}")
    borel_pade_poles([harm(B[k + 1], 0.8, jD - 1) for k in range(k0 + 1, K + 1)], f"B_k+1 harmonic {jD - 1} at x = 0.8")
