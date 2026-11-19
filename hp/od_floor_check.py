"""Check of newton_od's floor rule (octo_corrector.floor_accept, anm_run.py --od_floor_delta; #966).

    python3 od_floor_check.py                  every case must give its expected outcome: exit 0
    python3 od_floor_check.py --mutate=nodelta the rule without its dDelta test: case C converges, so exit 1
    python3 od_floor_check.py --mutate=off     the rule never fires: case A stalls, so exit 1

A mock system drives newton_od itself: its newton_o returns the same residual and Newton correction at every point,
so a state at a residual floor (max |F| above tol, nothing that a step could lower) either ends by the floor rule at
iteration 0 or refuses every trial and stalls. Case A is #966's D 251.2845 state of tol_od 1e-34 (raw I949-dh170,
c3b_ks293_D251.2845.log): max |F| 3.5e-35, its pinned correction 3.095e-39 in max norm and in Delta, which the
re-solve at 1.5e-35 confirmed. tol is 1.5e-35, as that re-solve's.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import octo_corrector as oc  # noqa: E402

N = 12
TOL = 1.5e-35


class Mock:
    """newton_o at every point: max |F| = F, a correction of max norm dz whose Delta part is dD"""

    def __init__(self, F, dz, dD):
        self.F, self.dz, self.dD = F, dz, dD

    def newton_o(self, z, mutate=0):
        d = [np.zeros(N) for _ in range(8)]
        d[0][0] = self.dz
        d[0][-1] = self.dD
        return (tuple(d), np.full(N, self.F), np.zeros(N), 1e-5, 1.0)


CASES = [  # name, F, dz, dD, floor_delta, floor_dz, expected converged, expected by the floor rule
    ("A: #966 D 251.2845 at tol_od 1e-34, rule on", 3.5e-35, 3.095e-39, 3.095e-39, 1e-30, 1e-30, True, True),
    ("B: the same, rule off", 3.5e-35, 3.095e-39, 3.095e-39, 0.0, 1e-30, False, False),
    ("C: dDelta 1e-20 (only the Delta test can refuse)", 3.5e-35, 1e-20, 1e-20, 1e-30, 1.0, False, False),
    ("D: max |F| 1e-25, above floor_F", 1e-25, 3.095e-39, 3.095e-39, 1e-30, 1e-30, False, False),
    ("E: max |dz| 1e-20, above floor_dz", 3.5e-35, 1e-20, 3.095e-39, 1e-30, 1e-30, False, False),
    ("F: max |F| 1e-36, below tol", 1e-36, 3.095e-39, 3.095e-39, 1e-30, 1e-30, True, False),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mutate", default="", choices=["", "nodelta", "off"])
    a = ap.parse_args()
    if a.mutate == "nodelta":
        oc.floor_accept = lambda F, dz, dD, b, Fm, dzm: bool(b > 0 and F <= Fm and dz <= dzm)
    elif a.mutate == "off":
        oc.floor_accept = lambda *args: False
    z0 = tuple(np.ones(N) if q == 0 else np.zeros(N) for q in range(8))
    ok_all = True
    for name, F, dz, dD, fdel, fdz, want, want_floor in CASES:
        _, hist, conv = oc.newton_od(Mock(F, dz, dD), z0, tol=TOL, it_max=3, verbose=False,
                                     floor_delta=fdel, floor_F=1e-30, floor_dz=fdz)
        by_floor = bool(hist and hist[-1].get("floor"))
        ok = conv == want and by_floor == want_floor
        ok_all &= ok
        print(f"{name}: converged {conv} (want {want}), by the floor rule {by_floor} (want {want_floor}), "
              f"{len(hist)} records: {'ok' if ok else 'WRONG'}")
    print(("PASS" if ok_all else "FAIL") + (f" (--mutate={a.mutate})" if a.mutate else ""))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
