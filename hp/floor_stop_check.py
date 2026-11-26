"""Planted check of newton_o_s's floor stop and newton_ladder's hand-over (form_s.py, #1088).

A model system F(z) = z + f_p 0.8^k (k the evaluation count, f_p the planted floor of the residual's precision:
1e-20 in double-double, 1e-40 in quad-double, 1e-60 in octo-double) with J = 1 and Delta's gain 1e8. From z = 1e-2 the
double-double stage reaches its floor in one step, and the next step cuts max |F| by 5x, so it must stop
there (floor) and hand over to quad-double, which converges on Delta's bound 1e-15 in one step. Pass: the dd stage
ends after 2 steps with floor set, the qd stage converges in 1, the run ends converged.
Control --mutate=nofloor (floor_cut 0): the dd stage runs to its cap at the floor and the run must fail.

    python3 hp/floor_stop_check.py [--mutate=nofloor]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import form_s  # noqa: E402

FLOOR = {2: 1e-20, 4: 1e-40, 8: 1e-60}
L = 6


class Model:
    def __init__(self):
        self.k = 0
        self.vec = np.linspace(1.0, 2.0, L)

    def residual_s(self, parts):
        z = np.asarray(parts[0]) + np.asarray(parts[1]) + (np.asarray(parts[2]) if len(parts) > 2 else 0.0)
        F = z + FLOOR[len(parts)] * self.vec * 0.8 ** self.k
        self.k += 1
        return [F] + [np.zeros(L) for _ in range(len(parts) - 1)]


def fake_step(s, z8, F, gain=False, jscale=1.0, parts=4):
    d = [-np.asarray(F[0]), -np.asarray(F[1])] + [np.zeros(L) for _ in range(6)]
    return (d, 1e8) if gain else d


def main(argv):
    mutate = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), "none")
    form_s.qj_step = fake_step
    z8 = [np.full(L, 1e-2)] + [np.zeros(L) for _ in range(7)]
    z8, conv, recs = form_s.newton_ladder(Model(), z8, 1e-300, ["dd", "qd"], 8, 2, 4, None, verbose=True,
                                          delta_tol=1e-15, jac="dd", floor_cut=0.0 if mutate == "nofloor" else 10.0)
    dd = [r for r in recs if r["res"] == "dd"]
    qd = [r for r in recs if r["res"] == "qd"]
    ok = conv and len(dd) == 2 and dd[-1]["floor"] and len(qd) == 1 and not qd[-1]["floor"]
    print(f"dd steps {len(dd)} (floor {dd[-1]['floor'] if dd else None}), qd steps {len(qd)}, converged {conv}: "
          f"{'PASS' if ok else 'FAIL'} (mutate {mutate})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
