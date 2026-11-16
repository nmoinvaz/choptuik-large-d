#!/usr/bin/env python3
"""Planted truth for the deflated Newton smdriver.newton_band (#845): a toy system with a known root and a known
near-null band, solved through the same function as the kernel's runs.

The toy, in n = 12 unknowns with a band of k = 3 directions. With fixed orthogonal U, V (f64, from a seeded QR) and
w = V^T (z - z*), split into the strong coordinates s (the first 9) and the band coordinates b (the last 3),

    F(z) = U G(w),   G_i = sig_i s_i + s_i^2 / 2 + kappa c_i sum_j b_j^4          (strong rows, sig_i in [0.5, 3])
                     G_j = sig_j (b_j + b_j^3) + |s|^2                            (band rows, sig_j = 1e-8, 1e-9, 1e-10)

with kappa = 1e-6. So z* is a root, the Jacobian there has the singular values sig, and F hardly sees a move along the
band until it is of order 10, where the quartic term takes over. The toy is evaluated in mpmath at 50 digits and
handed over in double-double, with the four methods that newton_band uses of the kernel's Solver (residual, newton,
chord, jacobian). The start is z* + V w0 with |s| = 0.15 and b = (0.3, -0.2, 0.25). Its strong offset feeds the band
rows at second order, |s|^2 = 0.0225, so Newton's step has components of 1e6 along the band.

Plan, written before the runs. All three solves start at the same point, with tol 1e-28.
  A  newton_band converges: max |F| < 1e-28 within 40 iterations, at max |z - z*| < 1e-18 (tol over the smallest sig).
  B  plain Newton with backtracking (smdriver.newton_solve, the solve of the kernel's runs) stalls: not converged,
     max |F| still above half its first value.
  C  the control, newton_band with band=False (the same loop with nothing deflated): not converged within the same
     40 iterations, with max |F| still above 1e-14, which is halfway in orders between the start and tol.
  D  the kernel itself on its own planted problem (planted.py, M = 8, the same start), through the deflated path:
     newton_band with band_tol 1e-2 and a first radius of 1e-5, so that its first steps are deflated. It must take at
     least one resolved and one band step and reach planted.py's marks, max |F| < 1e-28 at max |z - z*| < 1e-26 and
     |Delta - Delta*| < 1e-28, within 40 iterations. (That Jacobian has no near-null band, condition 8e3: D checks the
     kernel's residual, newton, chord and jacobian through the new code, not the band.)
Exit 0 when A, B, C and D hold, 1 otherwise.
(The first version of this plan asked of C what it asks of B. The control halves its step up to 30 times where
newton_solve stops at 10, so it creeps instead of stopping: max |F| 0.18 -> 2.6e-4 in 40 iterations and 5e-9 in 200,
while its band coordinates wander from 0.3 out to 1.7. That is the walk along the band of the kernel's runs, and C
now asks for it in the terms above. A first version of newton_band itself, a trust region on |F|_2 in the singular
basis, also walked: its allowed step shrank with |F|, 40 iterations left max |F| at 1e-11 to 1e-23, and it was
replaced by the deflated Newton before any kernel run.)

Controls of this test (each must give exit 1):
  --mutate noband    A is run with band=False, so A must fail
  --mutate easy      the band's sig_j are 1 and kappa is 0: plain Newton then converges, so B must fail
  --mutate root      A's result is compared with z* moved by 1e-12 in its first component, so A must fail
  --mutate gamma     the kernel's gamma recursion mutated (planted.py's control): D must fail

    PYTHONPATH=<site> python3 planted_band.py [--mutate NAME] [--threads 2] [--out FILE.json]
"""
import argparse
import json
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import planted as PL  # noqa: E402
import smdriver as sd  # noqa: E402
from multidouble import dd_from_mp  # noqa: E402

N, K = 12, 3
TOL = 1e-28


class Toy:
    """the toy system with the kernel Solver's residual, newton and jacobian (double-double parts in and out)"""

    def __init__(self, easy=False, seed=845, dps=50):
        mp.mp.dps = dps
        rng = np.random.default_rng(seed)
        self.U = mp.matrix(np.linalg.qr(rng.standard_normal((N, N)))[0].tolist())
        self.V = mp.matrix(np.linalg.qr(rng.standard_normal((N, N)))[0].tolist())
        self.sig = [mp.mpf(float(x)) for x in np.linspace(0.5, 3.0, N - K)] + \
            ([mp.mpf(1)] * K if easy else [mp.mpf("1e-8"), mp.mpf("1e-9"), mp.mpf("1e-10")])
        self.kappa = mp.mpf(0) if easy else mp.mpf("1e-6")
        self.c = [mp.mpf(float(x)) for x in rng.uniform(0.5, 1.5, N - K)]
        self.zs = mp.matrix([mp.mpf(float(x)) for x in rng.uniform(-1.0, 1.0, N)])
        s0 = rng.standard_normal(N - K)
        self.w0 = mp.matrix([mp.mpf(float(x)) for x in 0.15 * s0 / np.linalg.norm(s0)] + [mp.mpf("0.3"), mp.mpf("-0.2"),
                                                                                           mp.mpf("0.25")])

    def start(self):
        return self.to_dd(self.zs + self.V * self.w0)

    @staticmethod
    def to_dd(v):
        h, lo = dd_from_mp([v[i] for i in range(len(v))])
        return np.asarray(h, float), np.asarray(lo, float)

    @staticmethod
    def to_mp(z):
        return mp.matrix([sum(mp.mpf(float(p[i])) for p in z) for i in range(len(z[0]))])

    def G(self, w):
        s2 = sum(w[i] ** 2 for i in range(N - K))
        b4 = sum(w[j] ** 4 for j in range(N - K, N))
        return mp.matrix([self.sig[i] * w[i] + w[i] ** 2 / 2 + self.kappa * self.c[i] * b4 for i in range(N - K)]
                         + [self.sig[j] * (w[j] + w[j] ** 3) + s2 for j in range(N - K, N)])

    def dG(self, w):
        A = mp.zeros(N, N)
        for i in range(N - K):
            A[i, i] = self.sig[i] + w[i]
            for j in range(N - K, N):
                A[i, j] = 4 * self.kappa * self.c[i] * w[j] ** 3
                A[j, i] = 2 * w[i]
        for j in range(N - K, N):
            A[j, j] = self.sig[j] * (1 + 3 * w[j] ** 2)
        return A

    def _F(self, z):
        return self.U * self.G(self.V.T * (self.to_mp(z) - self.zs))

    def _J(self, z):
        return self.U * self.dG(self.V.T * (self.to_mp(z) - self.zs)) * self.V.T

    def residual(self, z):
        return list(self.to_dd(self._F(z)))

    def jacobian(self, zh, zl):
        J = self._J([zh, zl])
        h, lo = dd_from_mp([J[i, j] for i in range(N) for j in range(N)])
        return np.asarray(h, float).reshape(N, N), np.asarray(lo, float).reshape(N, N)

    def newton(self, z):
        F, J = self._F(z), self._J(z)
        self.J = J                      # kept for chord, as the kernel keeps its factors
        _, _, Um = mp.lu(J)
        piv = [abs(float(Um[i, i])) for i in range(N)]
        return list(self.to_dd(mp.lu_solve(J, -F))), list(self.to_dd(F)), min(piv), max(piv)

    def chord(self, z):
        F = self._F(z)
        return list(self.to_dd(mp.lu_solve(self.J, -F))), list(self.to_dd(F))


def kernel_check(threads, mutate_gamma):
    """D: planted.py's problem and start at M = 8, solved by newton_band on its deflated path"""
    mp.mp.dps = 50
    M, nodes = 8, [0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.9]
    pp = PL.planted_problem(M)
    s = sd.Solver(M, 2.0, 0.0, 1.0, 0.0, *sd.dmat_odd(M), *sd.phase_weights(M), nodes, [], 30, 40, 40, threads)
    s.set_planted(pp["beta"][0], pp["beta"][1], *pp["coef"])
    zh, zl = PL.planted_z(pp, nodes)
    rng = np.random.default_rng(1)
    pert = 1e-3 * rng.uniform(-1, 1, len(zh)) * np.maximum(np.abs(zh), 0.1)
    pert[-1] = 1e-3 * zh[-1]
    z0h = zh + pert
    s.set_segs(s.plan(z0h, np.zeros_like(zl), 1e-32, 0.05))
    if mutate_gamma:
        s.set_mutate(PL.MUT["gamma"])
    rh, rl, h = sd.newton_band(s, z0h, np.zeros_like(zl), tol=TOL, it_max=40, band_tol=1e-2, radius=1e-5, verbose=False)
    D = dict(converged=bool(h[-1].get("converged")), F=h[-1]["F"], iterations=len(h) - 1,
             resolved=sum(r.get("mode") == "resolved" for r in h), band=sum(r.get("mode") == "band" for r in h),
             deflated=max([r.get("k", 0) for r in h]), err=float(np.max(np.abs((rh - zh) + (rl - zl)))),
             dDelta=abs(float((rh[-1] - zh[-1]) + (rl[-1] - zl[-1]))))
    D["ok"] = D["converged"] and D["resolved"] >= 1 and D["band"] >= 1 and D["err"] < 1e-26 and D["dDelta"] < 1e-28
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutate", default="", choices=["", "noband", "easy", "root", "gamma"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    toy = Toy(easy=a.mutate == "easy")
    zh0, zl0 = toy.start()
    F0 = sd.fnorm(toy.residual([zh0, zl0]))
    Jh, Jl = toy.jacobian(zh0, zl0)
    sv = np.linalg.svd(Jh + Jl, compute_uv=False)
    dz0 = toy.newton([zh0, zl0])[0]
    print(f"toy: n = {N}, band of {K}; at the start max |F| = {F0:.2e}, singular values {sv[0]:.2e} down to "
          f"{sv[-3]:.1e}, {sv[-2]:.1e}, {sv[-1]:.1e}; Newton's step |dz_N| = {np.linalg.norm(dz0[0] + dz0[1]):.2e}")
    zref = toy.zs.copy()
    if a.mutate == "root":
        zref[0] += mp.mpf("1e-12")

    def err(zh, zl):
        d = toy.to_mp([zh, zl]) - zref
        return float(max(abs(d[i]) for i in range(N)))

    print("A  newton_band" + (" with band=False (mutation noband)" if a.mutate == "noband" else ""))
    zh, zl, hA = sd.newton_band(toy, zh0, zl0, tol=TOL, it_max=40, band=a.mutate != "noband")
    A = dict(converged=bool(hA[-1].get("converged")), F=hA[-1]["F"], iterations=len(hA) - 1, err=err(zh, zl))
    A["ok"] = A["converged"] and A["err"] < 1e-18
    print(f"   converged {A['converged']}, max |F| {A['F']:.2e}, {A['iterations']} steps, max |z - z*| {A['err']:.2e}: "
          f"{'PASS' if A['ok'] else 'FAIL'}")
    print("B  plain Newton with backtracking (newton_solve)")
    zh, zl, hB = sd.newton_solve(toy, zh0, zl0, tol=TOL, it_max=40)
    B = dict(converged=bool(hB[-1].get("converged")), F=sd.fnorm(toy.residual([zh, zl])), records=len(hB),
             failed=hB[-1].get("failed"))
    B["ok"] = (not B["converged"]) and B["F"] > 0.5 * F0
    print(f"   converged {B['converged']}, max |F| {B['F']:.2e} (first {F0:.2e}), ended on {B['failed']}: "
          f"{'PASS (it stalls)' if B['ok'] else 'FAIL (it does not stall)'}")
    print("C  the control: newton_band with band=False")
    zh, zl, hC = sd.newton_band(toy, zh0, zl0, tol=TOL, it_max=40, band=False, verbose=False)
    wC = toy.V.T * (toy.to_mp([zh, zl]) - toy.zs)
    C = dict(converged=bool(hC[-1].get("converged")), F=sd.fnorm(toy.residual([zh, zl])), records=len(hC),
             failed=hC[-1].get("failed"), band_end=[float(wC[j]) for j in range(N - K, N)])
    C["ok"] = (not C["converged"]) and C["F"] > 1e-14
    print(f"   converged {C['converged']}, max |F| {C['F']:.2e} (first {F0:.2e}) after {C['records'] - 1} steps, band "
          f"coordinates {', '.join(f'{x:.2f}' for x in C['band_end'])}: "
          f"{'PASS (it does not converge)' if C['ok'] else 'FAIL (it converges)'}")
    print("D  the kernel's planted problem (planted.py, M = 8) through the deflated path")
    D = kernel_check(a.threads, a.mutate == "gamma")
    print(f"   converged {D['converged']}, max |F| {D['F']:.2e}, {D['iterations']} steps ({D['resolved']} resolved, "
          f"{D['band']} band, {D['deflated']} directions deflated), max |z - z*| {D['err']:.2e}, |dDelta| "
          f"{D['dDelta']:.2e}: {'PASS' if D['ok'] else 'FAIL'}")
    ok = A["ok"] and B["ok"] and C["ok"] and D["ok"]
    print(f"PLANTED BAND{' (mutation ' + a.mutate + ')' if a.mutate else ''}: {'PASS' if ok else 'FAIL'}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(mutate=a.mutate, n=N, band=K, F_start=F0, sv_start=[float(x) for x in sv],
                           dz_newton_start=float(np.linalg.norm(dz0[0] + dz0[1])), A=A, B=B, C=C, D=D, ok=ok,
                           history_A=[{k: v for k, v in h.items() if k not in ("piv", "t", "t_svd")} for h in hA]), fh,
                      indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
