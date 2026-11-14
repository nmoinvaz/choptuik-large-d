#!/usr/bin/env python3
"""driver.lm_solve's handover to newton_solve on toy solvers, without a kernel run. F(z) = (1e-14 (z0 - 1), g(z1)) with
(A) g = 0, whose Jacobian [[1e-14, 0], [0, 0]] is singular everywhere, so newton_solve's first factorization raises
the kernel's ValueError (a singular pivot), and (B) g(z1) = z1 - 1/2, regular. From (0, 0) LM's mu = 0 step lands on
the root, below newton_after, and hands over.

    cd hp && python3 lm_handover_toy.py [--mutate=raise]

It runs this checkout's driver (python/hpkernel/driver.py) with the installed kernel package.
Pass (exit 0): (A) lm_solve returns, with |F|_2 <= 1e-27 and the failed handover in its history; (B) it hands over,
its history ends with Newton's records, and |F|_2 <= 1e-27. Control: --mutate=raise gives (A) the handover's outcome
before this check, newton_solve from the state LM reached, which raises; the check must fail (exit 1).
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import hpkernel  # noqa: E402,F401  (the installed package: its compiled kernel, for the driver's relative import)

spec = importlib.util.spec_from_file_location("hpkernel.driver", os.path.join(HERE, "python", "hpkernel", "driver.py"))
drv = importlib.util.module_from_spec(spec)
sys.modules["hpkernel.driver"] = drv
spec.loader.exec_module(drv)


class Toy:
    """F(z) = (1e-14 (z0 - 1), z1 - 1/2 if regular else 0), z in double-double (the lo parts stay zero here)"""
    m = 8
    size = 2

    def __init__(self, regular):
        self.regular = regular

    def residual(self, zh, zl):
        z = np.asarray(zh) + np.asarray(zl)
        return np.array([1e-14 * (z[0] - 1.0), z[1] - 0.5 if self.regular else 0.0]), np.zeros(2)

    def lm_steps(self, zh, zl, mus):
        """(J^T J + mu diag(J^T J)) dz = -J^T F row by row (J diagonal); a zero row gives a zero step"""
        Fh, Fl = self.residual(zh, zl)
        j = np.array([1e-14, 1.0 if self.regular else 0.0])
        steps = [(np.array([-Fh[i] / (j[i] * (1.0 + mu)) if j[i] else 0.0 for i in range(2)]), np.zeros(2))
                 for mu in mus]
        return steps, Fh, Fl

    def newton(self, zh, zl):
        if not self.regular:
            raise ValueError("singular or non-finite pivot at column 1 (0)")
        Fh, Fl = self.residual(zh, zl)
        return np.array([-Fh[0] / 1e-14, -Fh[1]]), np.zeros(2), Fh, Fl, 1e-14, 1.0


def fnorm(s, zh, zl):
    return float(np.linalg.norm(s.residual(zh, zl)[0]))


def main():
    mutate = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
    fails = []
    z0 = (np.zeros(2), np.zeros(2))

    s = Toy(regular=False)
    if mutate == "raise":
        zr = (np.array([1.0, 0.0]), np.zeros(2))       # the state LM reaches, from which the handover ran Newton
        try:
            out = drv.newton_solve(s, *zr, tol=1e-27, it_max=8, verbose=False)
        except ValueError as e:
            out = e
    else:
        out = drv.lm_solve(s, *z0, tol=1e-27, verbose=False)
    if isinstance(out, Exception):
        ok, what = False, f"raised {out!r}"
    else:
        zh, zl, hist = out
        failed = [h["handover_failed"] for h in hist if "handover_failed" in h]
        ok = fnorm(s, zh, zl) <= 1e-27 and len(failed) == 1
        what = f"|F|_2 {fnorm(s, zh, zl):.1e}, failed handovers {failed}"
    print(f"(A) singular Jacobian at the root: {what}: {'PASS' if ok else 'FAIL'}")
    fails += [] if ok else ["A"]

    s = Toy(regular=True)
    zh, zl, hist = drv.lm_solve(s, *z0, tol=1e-27, verbose=False)
    newton = [h for h in hist if not h.get("lm")]
    ok = fnorm(s, zh, zl) <= 1e-27 and bool(newton) and not hist[-1].get("lm") \
        and not any("handover_failed" in h for h in hist)
    print(f"(B) regular Jacobian: |F|_2 {fnorm(s, zh, zl):.1e}, {len(hist) - len(newton)} LM and {len(newton)} Newton "
          f"records: {'PASS' if ok else 'FAIL'}")
    fails += [] if ok else ["B"]
    print(f"failed: {', '.join(fails) if fails else 'none'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
