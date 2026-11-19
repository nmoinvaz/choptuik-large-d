#!/usr/bin/env python3
"""The l-multiplet's M = 3 inner selection at D = infinity: nested.py's nested solve on inner_ell's model (#994).

Residuals. As inner_np.residual_demixed: the jump J(theta) at xi1 is multiplied by e^{-Pi_l(xi1, theta)}, taken on the
label theta0 = theta - 2 w ln xi1 and normalized by e^{1/(2 xi1^2)} xi1^(2l) / (2 pi |K'(i nu w + l)|). Pi_l is the
fast mode's leading exponent shift from the slow lapse. With y = 2 F xi and xi c_xi = (xi^2/2)^l c E, the slow lapse is
c = 1 + 2^(1-l) F^2 xi^(2l+2) / (2l+2), the fast exponent is int c^2 / xi^3 along theta = theta0 + 2 w ln xi, and

    Pi_l(xi, theta) = sum_k (F^2)_k e^{i k theta} 2^(2-l) xi^(2l) / ((2l + 2)(2l + 2 i k w)),

all k for l >= 1. At l = 0 the k = 0 term is the log xi^(2<F^2>) (the dressing) and the rest is inner_np's
Pi = (1/w) int (F^2 - <F^2>), so l = 0 is residual_demixed exactly. For l = 1 Pi_l is O(A^2 xi1^2), of order 1 at the
l = 1 zero, and its error leaks the harmonic-3 condition into the harmonic-1 one with weight ~|K'_3 / K'_1|, which at
w/w_inf 2.5 is ~2e12 (against ~1e5 for l = 0 at w_inf).

Solve (nested.py's, with hier.py's polish and Schur correction). Inner: at fixed (A, w) solve r_3 = 0 for F_3 on a
ladder of radii, from one where the harmonic-3 jump is linear for any F_3 near the guess out to xi1, each warm-started
(the jump left over at one radius is resolved ~|K'_3| e^{-1/(2 xi^2)} times better at the next). Outer: Newton on
(A, w/w_inf) for the Schur-corrected r_1 - K_13 K_33^-1 r_3 at xi1 (K by finite differences after the inner solve), so
that the ulp-level r_3 does not leak into r_1. A genuine zero does not move with xi1.

    python3 nested_ell.py ident                   demixed_ell at l = 0 against inner_np.residual_demixed (bit for bit)
    python3 nested_ell.py zero L A wf [--xi1=X,...] [--F3=re,im] [--expect=A,wf] [--mutate=grav]
        l = 0: the planted M = 3 zero (FIRST_ORDER.md: A 2.7264759, w/w_inf 1.015603095) at xi1 0.175, 0.2, 0.225
        l >= 1: xi1 from inner_ell.r1_for at tau 1e-6, 1e-5, 1e-4 (the harmonic-1 jump's size against the slow
        solution), pass: the same zero at every xi1 (relative 1e-6) and Newton steps there below 1e-7;
        --expect: that zero (relative 1e-6)
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inner_np as I    # noqa: E402
import inner_ell as E   # noqa: E402

T0 = time.time()
MUT = E.MUT
opt = E.opt


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def demixed_ell(x, M, ell, radii, ell_grav=None, NTH=64):
    """the de-mixed residuals at several radii from one integration: dict r1 -> [Re r_1, Im r_1, Re r_3, ...]"""
    w = I.W0 * x[1]
    Fh = I.Fh_from(x, M)
    lg = ell if ell_grav is None else ell_grav
    m = E.InnerEll(M, w, Fh, ell=ell, ell_grav=ell_grav)
    J, info = m.jumps(tuple(sorted(set(radii))), rtol=E.RTOL)
    th = 2 * np.pi * np.arange(NTH) / NTH
    kk = np.fft.fftfreq(NTH, 1 / NTH)
    Fth = sum(v * np.exp(1j * mu * th) for mu, v in Fh.items()).real
    F2h = np.fft.fft(Fth ** 2) / NTH
    if lg == 0:
        Pi0 = np.real(np.fft.ifft(np.where(kk != 0, F2h / (1j * kk + (kk == 0)), 0)) * NTH) / w
    out = {}
    for r1, Jd in J.items():
        if lg == 0:                             # as inner_np.residual_demixed, operation for operation
            Pi, sc = Pi0, np.exp(1 / (2 * r1 * r1)) * r1 ** (-2 * I.F2mean(Fh))
        else:
            coef = 2.0 ** (2 - lg) * r1 ** (2 * lg) / ((2 * lg + 2) * (2 * lg + 2j * kk * w))
            Pi = np.real(np.fft.ifft(F2h * coef) * NTH)
            sc = np.exp(1 / (2 * r1 * r1))
        if ell:
            sc = sc * r1 ** (2 * ell)               # |xi^(-2 s)| = xi^(-2 l) of the fast mode
        Jth = sum(Jd[k + m.L] * np.exp(1j * k * th) for k in range(-M, M + 1) if k % 2)
        q = np.fft.fft(Jth * np.exp(-Pi)) / NTH
        v = []
        for nu in range(1, M + 1, 2):
            z = q[nu % NTH] * r1 ** (2j * nu * w) * sc / (2 * np.pi * abs(I.Kp(1j * nu * w + ell)))
            v += [z.real, z.imag]
        out[r1] = np.array(v)
    demixed_ell.info = info
    return out


def tau_nu(nu, wf, ell, r):
    """|flat jump of harmonic nu| / |slow y_nu| per unit F_nu at real r"""
    w = I.W0 * wf
    return 2 * np.pi * abs(I.Kp(1j * nu * w + ell) * E.fast_lin_ell(nu, w, ell, r)) / r


def ladder(wf, ell, xi1, start=1e-4, step=1e6):
    """radii for the harmonic-3 solve: tau_3 = start, start*step, ... and finally xi1"""
    out, t = [], start
    while True:
        lo, hi = 0.092, 0.35
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if tau_nu(3, wf, ell, mid) > t else (mid, hi)
        r = round(0.5 * (lo + hi), 6)
        if r >= xi1:
            break
        out.append(r)
        t *= step
    return out + [xi1]


class Nested:
    def __init__(self, ell, xi1, rungs, ell_grav=None):
        self.ell, self.xi1, self.rungs, self.eg = ell, xi1, rungs, ell_grav
        self.nev = 0

    def res(self, x, radii):
        self.nev += 1
        return demixed_ell(x, 3, self.ell, list(radii) + [self.xi1], self.eg)

    def inner(self, A, wf, v):
        """solve r_3 = 0 for v = (B3, C3) on the ladder; returns v, r_1 (Schur-corrected) and |r_3| at xi1"""
        for r in self.rungs:
            f = lambda u, r=r: self.res(np.array([A, wf, u[0], u[1]]), [r])[r][2:]
            sol = least_squares(f, v, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=60)
            v = sol.x
        x = np.array([A, wf, v[0], v[1]])
        R0 = self.res(x, [])[self.xi1]
        K = np.zeros((4, 2))
        for k in range(2):
            h = 1e-9 * max(abs(v[k]), 1e-3 * A)
            xp = x.copy()
            xp[2 + k] += h
            K[:, k] = (self.res(xp, [])[self.xi1] - R0) / h
        dv = np.linalg.solve(K[2:], R0[2:])
        r1 = R0[:2] - K[:2] @ dv
        return v, r1[0] + 1j * r1[1], float(np.abs(R0[2:]).max()), float(np.abs(K[:2]).max() / np.abs(K[2:]).max())

    def solve(self, A, wf, v, its=12, stop=1e-10, floor=1e-7):
        """Newton on (A, w/w_inf) until the relative step is below stop, or below floor and no smaller than the step
        before (r_1's noise floor: no residual bar); returns A, wf, v, r_1 and the last relative step"""
        rel, prev = np.inf, np.inf
        for it in range(its):
            v, R, r3, leak = self.inner(A, wf, v)
            F3 = v[0] / 2j + v[1] / 2
            hA, hw = 1e-6 * A, 1e-6 * wf
            _, RA, _, _ = self.inner(A + hA, wf, v)
            _, Rw, _, _ = self.inner(A, wf + hw, v)
            Jm = np.array([[(RA - R).real / hA, (Rw - R).real / hw], [(RA - R).imag / hA, (Rw - R).imag / hw]])
            step = -np.linalg.solve(Jm, [R.real, R.imag])
            rel = max(abs(step[0]) / A, abs(step[1]) / wf)
            log(f"   it {it}: A {A:.9f} w/w_inf {wf:.9f}  r_1 {R:.4e}  |r_3| {r3:.1e}  |K_13|/|K_33| {leak:.1e}  "
                f"F3/F1 {F3 / (A / 2j):.7f}  relative step {rel:.1e}  ({self.nev} residuals)")
            if rel < stop or (rel < floor and rel >= prev):
                break
            prev = rel
            lam = min(1.0, 0.2 / rel)                                       # at most 20% per step
            A, wf = A + lam * step[0], wf + lam * step[1]
        return A, wf, v, R, rel


def ident():
    """demixed_ell at l = 0 against inner_np.residual_demixed, at an M = 3 point, bit for bit"""
    x = np.array([2.7265, 1.0156, 0.29, 0.0215])
    worst = 0.0
    for r1 in (0.175, 0.2):
        a = I.residual_demixed(x, 3, r1=r1)
        b = demixed_ell(x, 3, 0, [r1])[r1]
        if "ident" in MUT:
            b = b * (1 + 1e-15)
        d = float(np.abs(a - b).max() / np.abs(a).max())
        worst = max(worst, d)
        print(f"xi1 {r1}: residual_demixed {a}\n          demixed_ell      {b}   relative difference {d:.1e}")
    good = worst == 0.0
    print(f"identity at l = 0 (bit for bit): {'OK' if good else 'BAD'}")
    return good


def zero(ell, A, wf):
    eg = (1 if ell == 0 else 0) if "grav" in MUT else None
    f3 = [float(t) for t in opt("F3", "0.0532,0.0041" if ell == 0 else "0,0").split(",")]
    F3 = (A / 2j) * (f3[0] + 1j * f3[1])
    v0 = np.array([-2 * F3.imag, 2 * F3.real])
    if opt("xi1", None) is not None:
        xis = [float(t) for t in opt("xi1", "").split(",")]
    elif ell == 0:
        xis = [0.175, 0.2, 0.225]
    else:
        xis = [E.r1_for(wf, ell, t) for t in (1e-6, 1e-5, 1e-4)]
    log(f"l {ell}, M 3, start A {A}, w/w_inf {wf}, F3/F1 {f3[0]}+{f3[1]}i; xi1 {xis}; delta {I.DELTA:.6f}, rtol "
        f"{E.RTOL:.0e}, margin {E.MARGIN}" + (f"; MUTATION: gravity weight l = {eg}" if eg is not None else ""))
    zs = []
    for xi1 in xis:
        rungs = [xi1] if ell == 0 else ladder(wf, ell, xi1)
        nz = Nested(ell, xi1, rungs, eg)
        log(f"xi1 {xi1}: harmonic-3 ladder {rungs}")
        A1, w1, v1, R, rel = nz.solve(A, wf, v0)
        F3 = v1[0] / 2j + v1[1] / 2
        zs.append((A1, w1, rel))
        log(f"RESULT xi1 {xi1}: A {A1:.9f}  w/w_inf {w1:.9f}  (Delta {1.64389375 / w1:.7f})  F3/F1 "
            f"{F3 / (A1 / 2j):.7f}  |r_1| {abs(R):.1e}  last relative step {rel:.1e}")
        A, wf, v0 = A1, w1, v1
    z = np.array([q[:2] for q in zs])
    spread = np.max(np.abs(z - z[0]), axis=0) / np.abs(z[0])
    print(f"relative spread over xi1: A {spread[0]:.1e}, w/w_inf {spread[1]:.1e}")
    if ell == 0:
        dA, dw = abs(z[1][0] - 2.7264759), abs(z[1][1] - 1.015603095)
        good = dA < 1e-5 and dw < 1e-6
        print(f"planted (l = 0 M = 3, FIRST_ORDER.md's A 2.7264759, w/w_inf 1.015603095; bars 1e-5, 1e-6): |dA| {dA:.1e}, "
              f"|dw| {dw:.1e}  ({'OK' if good else 'BAD'})")
    else:
        good = spread.max() < 1e-6 and max(q[2] for q in zs) < 1e-7
        print(f"the same zero at every xi1 (relative spread < 1e-6, last Newton steps < 1e-7 relative): "
              f"{'OK' if good else 'BAD'}")
    ex = opt("expect", None)
    if ex is not None:
        e = np.array([float(t) for t in ex.split(",")])
        d = np.abs(z[0] - e) / np.abs(e)
        ok = d.max() < 1e-6
        good &= ok
        print(f"expected zero A {e[0]}, w/w_inf {e[1]} (relative 1e-6): relative |dA| {d[0]:.1e}, |dw| {d[1]:.1e}  "
              f"({'OK' if ok else 'BAD'})")
    return good


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ident":
        good = ident()
    elif cmd == "zero":
        rest = [a for a in sys.argv[2:] if not a.startswith("--")]
        good = zero(int(rest[0]), float(rest[1]), float(rest[2]))
    print("PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
