#!/usr/bin/env python3
"""Delta's exact differences between eight-part states, at 140 digits (#595).

    cd hp && python3 delta_diff.py RAWDIR STATE EXACT [ROUNDED]

STATE is the start, EXACT its base polish on the exact projector's rows, ROUNDED (optional) its base polish on the
rounded projector's rows from the same start and plan (the control): all relative to RAWDIR. Printed: each one's
Delta and its polish record, Delta(EXACT) - Delta(STATE), Delta(ROUNDED) - Delta(STATE) and Delta(EXACT) -
Delta(ROUNDED), the shift due to the projector alone (the start's other inconsistencies, its plan and held rows,
cancel there). Exit 1 unless each polish converged to max |F| below 1e-60.
"""
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octo_corrector import load_o  # noqa: E402

mp.mp.dps = 140


def main():
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    raw, paths = sys.argv[1], sys.argv[2:]
    D, ok = [], True
    for p in paths:
        z, m = load_o(os.path.join(raw, p))
        D.append(mp.fsum(mp.mpf(float(q[-1])) for q in z))
        bp = m.get("base_polish")
        if bp is not None:
            ok &= bool(bp.get("converged")) and float(bp["max_F"][1]) < 1e-60
        print(f"{p}: Delta {mp.nstr(D[-1], 40)}; polish {bp}", flush=True)
    print(f"Delta(exact) - Delta(state)   {mp.nstr(D[1] - D[0], 12)}")
    if len(D) == 3:
        print(f"Delta(rounded) - Delta(state) {mp.nstr(D[2] - D[0], 12)}")
        print(f"Delta(exact) - Delta(rounded) {mp.nstr(D[1] - D[2], 12)}  (the projector's shift)")
    print("check:", "pass" if ok else "FAIL (a polish did not converge below 1e-60)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
