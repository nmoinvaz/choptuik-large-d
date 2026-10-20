"""Follow a zero of the inner Stokes jump (selection condition J = 0) as harmonics are added, by homotopy.

Level M -> M + 2: the new harmonics (odd M+2 in Dd, sigma; even M+3 in c, E) enter every product with weight lam
(inner_np.Inner); lam goes 0 -> 1 in adaptive steps, and at each step the zero (A, w/w_inf, B3, C3, ..., B_M+2, C_M+2)
of the normalized jump r_nu = J_nu e^{1/(2 xi1^2)} xi1^{-2<F^2>} / (2 pi |K'_nu|) (nu = 1..M+2) is refound by
Levenberg-Marquardt from the previous one. At lam = 1 the zero is re-checked at a second radius (a true multiplier zero
does not depend on xi1).

    python3 -u continue_M.py [Mmax] [procs] [x0...] > cont.log   start: the M = 1 zero (2.542148, 0.776994)
(uses the de-mixed residuals inner_np.residual_demixed; never pipe through tail, the log is the progress record)

procs > 1 computes the finite-difference Jacobian columns in parallel (multiprocessing); one residual costs about
0.6 s (M = 1), 0.8 s (M = 3), 1.5 s (M = 5) measured on one cloud core with gravity (M = 9 linear: 1.2 s).
Output lines "LEVEL M lam ... x" are machine-readable; the final table is printed at the end.
"""
import sys
import time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
import inner_np as I

R1, R1CHECK = 0.2, 0.25


def _res(args):
    x, M, M0, lam, r1 = args
    try:
        return I.residual_demixed(np.asarray(x), M, M0, lam, r1)
    except Exception:
        return None


class Solver:
    def __init__(self, procs=1):
        self.pool = Pool(procs) if procs > 1 else None

    def jac(self, x, M, M0, lam, f0):
        h = 1e-6 * np.maximum(1.0, np.abs(x))
        args = [(x + h[i] * np.eye(len(x))[i], M, M0, lam, R1) for i in range(len(x))]
        cols = self.pool.map(_res, args) if self.pool else [_res(a) for a in args]
        return np.array([(c - f0) / h[i] for i, c in enumerate(cols)]).T

    def solve(self, x0, M, M0, lam, maxit=12):
        x = np.array(x0, float)
        f = I.residual_demixed(x, M, M0, lam, R1)
        mu = 1e-3
        for it in range(maxit):
            Jm = self.jac(x, M, M0, lam, f)
            g = Jm.T @ f
            Hm = Jm.T @ Jm
            while True:
                step = -np.linalg.solve(Hm + mu * np.diag(np.diag(Hm)), g)
                xn = x + step
                try:
                    fn = I.residual_demixed(xn, M, M0, lam, R1)
                except Exception:
                    fn = None
                if fn is not None and np.linalg.norm(fn) < np.linalg.norm(f):
                    x, f, mu = xn, fn, max(mu / 10, 1e-12)
                    break
                mu *= 10
                if mu > 1e8:
                    return x, f, False
            if np.abs(f).max() < 1e-9 or np.abs(step).max() < 1e-11:
                return x, f, True
        return x, f, np.abs(f).max() < 1e-6


def describe(x, M):
    A, wf = x[0], x[1]
    rat = [np.hypot(x[2 + 2 * i], x[3 + 2 * i]) / A for i in range((M - 1) // 2)]
    F2 = A * A / 2 + sum((x[2 + 2 * i] ** 2 + x[3 + 2 * i] ** 2) / 2 for i in range((M - 1) // 2))
    w = wf * I.W0
    return (f"A {A:.6f}  w/w_inf {wf:.6f}  w {w:.5f}  Delta {2 * np.pi / w:.5f}  <F^2> {F2:.4f}  " +
            "  ".join(f"F{2 * i + 3}/F1 {r:.5f}" for i, r in enumerate(rat)))


if __name__ == "__main__":
    Mmax = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    x = np.array([float(v) for v in sys.argv[3:]]) if len(sys.argv) > 3 else np.array([2.542148, 0.776994])
    M = (len(x) - 1)
    S = Solver(procs)
    t0 = time.time()
    x, f, ok = S.solve(x, M, M, 1.0)
    print(f"LEVEL {M} lam 1.0 ok {ok} |r| {np.abs(f).max():.1e} x {list(np.round(x, 8))}", flush=True)
    table = [(M, x.copy())]
    while M < Mmax:
        M0, M = M, M + 2
        x = np.concatenate([x, [0.0, 0.0]])
        lam, dlam = 0.0, 0.125
        while lam < 1.0 - 1e-12:
            lt = min(1.0, lam + dlam)
            xs, fs, ok = S.solve(x, M, M0, lt)
            jumpsize = np.abs(xs - x).max()
            if ok and jumpsize < 0.5:
                x, lam = xs, lt
                print(f"LEVEL {M} lam {lam:.4f} ok |r| {np.abs(fs).max():.1e} x {list(np.round(x, 8))}  "
                      f"[{time.time() - t0:.0f} s]", flush=True)
                dlam = min(0.25, dlam * 1.5)
            else:
                dlam /= 2
                print(f"   step to lam {lt:.4f} failed (ok {ok}, jump {jumpsize:.2e}); dlam -> {dlam:.4f}", flush=True)
                if dlam < 1e-3:
                    print("   continuation stalled", flush=True)
                    break
        if lam < 1.0 - 1e-12:
            break
        chk = I.residual_demixed(x, M, M, 1.0, R1CHECK)
        print(f"LEVEL {M} done: {describe(x, M)}   residual at xi1 = {R1CHECK}: {np.abs(chk).max():.1e}", flush=True)
        table.append((M, x.copy()))
    print("\nM  | " + "result")
    for Mv, xv in table:
        print(f"{Mv:2d} | {describe(xv, Mv)}")
    print(f"target (D = infinity): A ~ 2.72, w/w_inf = 1 (w 3.82214, Delta 1.64389), F3/F1 ~ 0.06-0.08, <F^2> ~ 3.65")
