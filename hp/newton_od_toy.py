#!/usr/bin/env python3
"""octo_corrector.newton_od on a toy solver with one weak direction and no quadratic term, the structure of #655's
class A refusals: F(y) = (sigma y0 + T/6 y0^3, y1 + k/2 y0^2), sigma 1e-6, T 6, k 1e4, eight-part iterates whose parts
1-7 are zero. Far from the root Newton contracts by 2/3 per step, the rate at a triple root. It checks newton_od's two
trial tests and its printed lines without the kernel (#644, #666).

    cd hp && python3 newton_od_toy.py [--mutate=quiet]

Pass (exit 0): from y0 = 0.3 both tests (own, natural) converge below 1e-14; the default test refuses at least one
trial and prints it at verbose 1; and kantorovich_log.parse_log reads every printed trial line of both runs. Control:
--mutate=quiet runs newton_od at verbose 0, so no line is printed and the check must fail.

The strong-row chord (#674): chord_strong from the start keeps the first step's weak coordinate y0 to 1e-12
relative and solves the strong row to 1e-12 of its value after the step, and newton_od with strong_chord=1 converges.
Control: --mutate=noproj keeps the weak part of each chord correction, which moves y0, and the check must fail.
"""
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kantorovich_log import parse_log  # noqa: E402
import octo_corrector  # noqa: E402
from octo_corrector import chord_strong, newton_od  # noqa: E402

SIG, T, K = 1e-6, 6.0, 1e4
ZERO = np.zeros(2)


def F(y, k=K):
    return np.array([SIG * y[0] + T / 6 * y[0] ** 3, y[1] + k / 2 * y[0] ** 2])


def J(y, k=K):
    return np.array([[SIG + T / 2 * y[0] ** 2, 0.0], [k * y[0], 1.0]])


class Toy:
    """the octo-double solver interface newton_od uses: residual_o, newton_o and olu_solve (the LU of the last newton_o)"""

    size = 2

    def __init__(self, k=K):
        self.Jl = None
        self.k = k

    def residual_o(self, z, mutate=0):
        return [F(np.asarray(z[0], dtype=float), self.k)] + [ZERO] * 7

    def newton_o(self, z, mutate=0):
        y = np.asarray(z[0], dtype=float)
        self.Jl = J(y, self.k)
        f = F(y, self.k)
        piv = np.abs(np.diag(np.linalg.qr(self.Jl)[1]))
        return [-np.linalg.solve(self.Jl, f)] + [ZERO] * 7, f, ZERO, float(piv.min()), float(piv.max())

    def olu_solve(self, rhs, trans=False):
        return [np.linalg.solve(self.Jl.T if trans else self.Jl, np.asarray(rhs[0], dtype=float))] + [ZERO] * 7


HEADER = ("step 0: D 250.0, pivots 1.0e-28..1.0e+34, |z_1| 3.50e-04, |z_N| 1.00e-25, ratios 2-5 1 1 1 1, a_max 2.000, "
          "1 s\n    a 0.5000: re-seeded (ok), 0.1 s\n")
FOOTER = ("    a 0.500 -> D 250.5: predicted |F| 1.00e+00, od 1 steps -> 1.00e-15 (tol 1.0e-17), Delta 1.0 (formula diff "
          "+0.0e+00), moved 1.0e-30, 1 s\n")


def run(accept, verbose):
    y0 = 0.3
    z = [np.array([y0, -K / 2 * y0 ** 2 * 1.001])] + [ZERO] * 7
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        z, hist, conv = newton_od(Toy(), z, tol=1e-14, it_max=40, accept=accept, verbose=verbose)
    return conv, buf.getvalue()


def chord_check():
    """chord_strong from a nearly singular start (y0 = 1e-3, coupling k = 1) against the full first step: the weak
    coordinate v^T y kept (v the start Jacobian's weakest right singular vector) to 1e-12 of the step, and the
    residual off the weakest left vector u cut to 1e-12 of its value after the step; then newton_od with the chord
    converges and accepts it"""
    k, y0 = 1.0, 1e-3
    z = [np.array([y0, -k / 2 * y0 ** 2 * 1.001])] + [ZERO] * 7
    toy = Toy(k)
    d = toy.newton_o(z)[0]
    Uj, _, Vt = np.linalg.svd(J(np.asarray(z[0]), k))
    u, v = Uj[:, -1], Vt[-1]
    y_step = np.asarray(z[0]) + np.asarray(d[0])
    with contextlib.redirect_stdout(io.StringIO()):
        zc, ok_c, info = chord_strong(toy, z, d, 1, it_max=10, tol=1e-30, tol_dz=1e-30)
    yc = np.asarray(zc[0])
    off_u = lambda f: f - u * (u @ f)  # noqa: E731
    kept = abs(v @ (yc - y_step)) / np.abs(d[0]).max()
    cut = np.abs(off_u(F(yc, k))).max() / np.abs(off_u(F(y_step, k))).max()
    with contextlib.redirect_stdout(io.StringIO()):
        z2, hist, conv = newton_od(Toy(k), z, tol=1e-20, it_max=40, accept="natural", strong_chord=1)
    used = bool(hist and hist[0].get("chord", {}).get("accepted"))
    good = ok_c and kept <= 1e-12 and cut <= 1e-12 and conv and used
    print(f"chord: {len(info['steps'])} steps; weak coordinate moved {kept:.1e} of the step (<= 1e-12), residual off u "
          f"cut to {cut:.1e} (<= 1e-12); newton_od with the chord converged {conv}, chord accepted {used}: "
          f"{'pass' if good else 'FAIL'}")
    return good


def main():
    quiet = "--mutate=quiet" in sys.argv[1:]
    if "--mutate=noproj" in sys.argv[1:]:
        octo_corrector.CHORD_MUT["noproj"] = True
    ok = True
    for accept in ("own", "natural"):
        conv, log = run(accept, 0 if quiet else 1)
        printed = sum(" trial lam " in ln for ln in log.splitlines())
        refused = sum(" trial lam " in ln and "refused" in ln for ln in log.splitlines())
        atts = parse_log(HEADER + log + FOOTER)
        parsed = len(atts[0]["trials"]) if len(atts) == 1 else -1
        steps = sum(ln.startswith("  on ") for ln in log.splitlines())
        good = conv and printed > 0 and parsed == printed and (accept != "own" or refused > 0)
        ok &= good
        print(f"{accept}: converged {conv} in {steps} lines; trial lines printed {printed} (refused {refused}), parsed "
              f"{parsed}: {'pass' if good else 'FAIL'}")
    ok &= chord_check()
    mut = " ".join(a for a in sys.argv[1:] if a.startswith("--mutate="))
    print(f"newton_od_toy{' ' + mut if mut else ''}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
