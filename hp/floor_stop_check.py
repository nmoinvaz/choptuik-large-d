"""Planted check of newton_o_s's floor stop, newton_ladder's hand-over and its chord steps (form_s.py, #1088).

A model system F(z) = z + f_p 0.8^k (k the evaluation count, f_p the planted floor of the residual's precision:
1e-20 in double-double, 1e-40 in quad-double, 1e-60 in octo-double) with J = 1 and Delta's gain 1e8. From z = 1e-2 the
double-double stage reaches its floor in one step, and the next step cuts max |F| by 5x, so it must stop
there (floor) and hand over to quad-double, which converges on Delta's bound 1e-15 in one chord step (the last dd
Jacobian, no new one formed). Pass: the dd stage ends after 2 steps with floor set and 2 Jacobians formed, the qd
stage converges in 1 chord step with none formed, the run ends converged.
Second case, --stale: the kept Jacobian is planted 2x off (a chord step takes half the Newton step, cutting max |F|
by 2x), so the qd stage's chord step must re-form J (reform set), and the fresh step converges: pass when the qd
stage takes 2 steps, the first a chord step with reform, the second a fresh one, and the run converges.
Control --mutate=nofloor (floor_cut 0): the dd stage runs to its cap at the floor and the check must fail.

    python3 hp/floor_stop_check.py [--stale] [--mutate=nofloor]
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


def make_step(stale, formed):
    def fake_step(s, z8, F, gain=False, jscale=1.0, parts=4, jstore=None, reuse=False, solve="dd"):
        if reuse and jstore is not None and "J" in jstore:
            scale = jstore["J"]
        else:
            formed.append(1)
            scale = 1.0
            if jstore is not None:
                jstore.update(J=0.5 if stale else 1.0, gain=1e8)
        d = [-scale * np.asarray(F[0]), -scale * np.asarray(F[1])] + [np.zeros(L) for _ in range(6)]
        return (d, 1e8) if gain else d
    return fake_step


def main(argv):
    mutate = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), "none")
    stale = "--stale" in argv
    formed = []
    form_s.qj_step = make_step(stale, formed)
    z8 = [np.full(L, 1e-2)] + [np.zeros(L) for _ in range(7)]
    n0 = []

    def on_step(z, rec):
        n0.append((rec["res"], len(formed)))
    z8, conv, recs = form_s.newton_ladder(Model(), z8, 1e-300, ["dd", "qd"], 8, 2, 4, None, verbose=True,
                                          delta_tol=1e-15, jac="dd", floor_cut=0.0 if mutate == "nofloor" else 10.0,
                                          on_step=on_step)
    dd = [r for r in recs if r["res"] == "dd"]
    qd = [r for r in recs if r["res"] == "qd"]
    formed_dd = max([n for res, n in n0 if res == "dd"], default=0)
    if stale:
        ok = (conv and len(dd) == 2 and dd[-1]["floor"] and len(qd) == 2 and qd[0]["chord"] and qd[0]["reform"]
              and not qd[1]["chord"] and len(formed) == formed_dd + 1)
    else:
        ok = (conv and len(dd) == 2 and dd[-1]["floor"] and formed_dd == 2 and len(qd) == 1 and qd[0]["chord"]
              and len(formed) == 2)
    print(f"dd steps {len(dd)} (floor {dd[-1]['floor'] if dd else None}), qd steps {len(qd)} (chord "
          f"{[r['chord'] for r in qd]}, reform {[r['reform'] for r in qd]}), Jacobians formed {len(formed)}, converged "
          f"{conv}: {'PASS' if ok else 'FAIL'} ({'stale J, ' if stale else ''}mutate {mutate})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
