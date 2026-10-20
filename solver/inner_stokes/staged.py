"""Staged continuation of the inner selection zero to more harmonics (faster alternative to continue_M.py):
at each new level M -> M + 2, first solve the new harmonic's own condition S_{M+2} = 0 for (B, C) with the old unknowns
fixed, then solve the full square system (de-mixed residuals, inner_np.residual_demixed) by Levenberg-Marquardt, and
re-check the zero at a second radius.
    python3 -u staged.py Mmax [x0 ...]          (default start: M = 1 zero 2.542148 0.776994)"""
import sys
import time
import numpy as np
from scipy.optimize import least_squares
import inner_np as I

R1, R1C = 0.2, 0.25
Mmax = int(sys.argv[1]) if len(sys.argv) > 1 else 5
x = np.array([float(v) for v in sys.argv[2:]]) if len(sys.argv) > 2 else np.array([2.542148, 0.776994])
M = len(x) - 1
t0 = time.time()


def full(xv, Mv, r1=R1):
    return I.residual_demixed(xv, Mv, r1=r1)


def show(tag, xv, Mv, f):
    A, wf = xv[0], xv[1]
    rats = [np.hypot(xv[2 + 2 * i], xv[3 + 2 * i]) / A for i in range((Mv - 1) // 2)]
    F2 = A * A / 2 + sum((xv[2 + 2 * i] ** 2 + xv[3 + 2 * i] ** 2) / 2 for i in range((Mv - 1) // 2))
    print(f"{tag} M {Mv}: A {A:.6f} w/w_inf {wf:.6f} (w {wf * I.W0:.5f}, Delta {2 * np.pi / (wf * I.W0):.5f}) <F^2> {F2:.4f} "
          + " ".join(f"F{2 * i + 3}/F1 {r:.5f}" for i, r in enumerate(rats)) + f"  |r| {np.abs(f).max():.1e}  [{time.time() - t0:.0f} s]",
          flush=True)


sol = least_squares(lambda v: full(v, M), x, method="lm", xtol=1e-12, ftol=1e-14)
x = sol.x
show("ZERO", x, M, sol.fun)
while M < Mmax:
    M += 2
    xo = x.copy()
    def new_only(bc):
        r = full(np.concatenate([xo, bc]), M)
        return r[-2:]
    s1 = least_squares(new_only, [0.0, 0.0], method="lm", xtol=1e-12, ftol=1e-14)
    x = np.concatenate([xo, s1.x])
    show("STAGE1", x, M, full(x, M))
    sol = least_squares(lambda v: full(v, M), x, method="lm", xtol=1e-12, ftol=1e-14, max_nfev=400)
    x = sol.x
    show("ZERO", x, M, sol.fun)
    chk = full(x, M, R1C)
    print(f"   check at xi1 = {R1C}: |r| {np.abs(chk).max():.1e} (status {sol.status})", flush=True)
