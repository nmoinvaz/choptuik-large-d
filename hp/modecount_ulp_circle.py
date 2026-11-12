#!/usr/bin/env python3
"""hp/modecount_circle.py with one entry of the differentiation matrix moved, to test how a circle's result depends on
its input's last bits (#498, #552).

    cd hp && python3 modecount_ulp_circle.py I J SHIFT PART -- <the arguments of modecount_circle.py>
    python3 hp/modecount_ulp_circle.py --selftest

SHIFT is +1 or -1, or 0, which passes the matrix as it is and must reproduce the plain run bit for bit. PART is:
  * hi: one ulp of the high part (np.nextafter), a relative change of about 2^-53, the perturbation of #498, #506 and
    #520 (I506-macm3/ulp_circle.py);
  * lo: one unit in the last place of the double-double value, the low part moved by spacing(hi) 2^-53, a relative
    change of about 2^-106, the size of double-double's own rounding (as hp/modecount_pivots.py moves it).
The hook is the module attribute modecount_cx.CSolver, which modecount_common looks up when it builds the solver, so
nothing else changes. It prints the entry's parts before and after. --selftest checks the move on planted entries.
"""
import os
import runpy
import sys

import numpy as np


def moved(dh, dl, i, j, shift, part):
    """copies of (dh, dl) with entry (i, j) moved by one unit of PART in the direction of SHIFT"""
    dh2, dl2 = np.array(dh, dtype=float, copy=True), np.array(dl, dtype=float, copy=True)
    if shift:
        if part == "hi":
            dh2[i, j] = np.nextafter(dh2[i, j], np.inf if shift > 0 else -np.inf)
        else:
            dl2[i, j] += np.sign(shift) * np.spacing(dh2[i, j]) * 2.0 ** -53
    return dh2, dl2


def selftest():
    """shift 0 leaves both parts; hi moves one ulp of the high part only; lo moves the low part only, by a relative
    2^-106 of the value to within a factor 2; the sign follows SHIFT"""
    import mpmath as mp
    mp.mp.dps = 50
    rng = np.random.default_rng(3)
    dh = rng.uniform(-40, 40, (6, 6))
    dl = dh * rng.uniform(-1, 1, (6, 6)) * 2.0 ** -54
    ok = True
    for (i, j) in ((1, 2), (4, 0)):
        a, b = moved(dh, dl, i, j, 0, "lo")
        ok &= bool(np.array_equal(a, dh) and np.array_equal(b, dl))
        for s in (1, -1):
            a, b = moved(dh, dl, i, j, s, "hi")
            ok &= bool(a[i, j] == np.nextafter(dh[i, j], s * np.inf) and np.array_equal(b, dl)
                       and np.count_nonzero(a != dh) == 1)
            a, b = moved(dh, dl, i, j, s, "lo")
            v0 = mp.mpf(dh[i, j]) + mp.mpf(dl[i, j])
            v1 = mp.mpf(a[i, j]) + mp.mpf(b[i, j])
            rel = (v1 - v0) / abs(v0)
            ok &= bool(np.array_equal(a, dh) and np.count_nonzero(b != dl) == 1
                       and mp.sign(rel) == s and 2.0 ** -107 <= abs(rel) <= 2.0 ** -104)
    print("selftest:", "pass" if ok else "FAIL")
    return 0 if ok else 1


def main():
    if sys.argv[1:] == ["--selftest"]:
        return selftest()
    if len(sys.argv) < 6 or sys.argv[5] != "--":
        sys.exit(__doc__)
    i, j, shift, part = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    if shift not in (-1, 0, 1) or part not in ("hi", "lo"):
        sys.exit("SHIFT is -1, 0 or 1 and PART hi or lo")
    hp = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, hp)
    sys.path.insert(0, os.path.join(hp, "modecount_cx", "build"))
    import modecount_cx
    orig = modecount_cx.CSolver

    def patched(M, nh, nl, dh, dl, ph, pl, *args, **kw):
        dh2, dl2 = moved(dh, dl, i, j, shift, part)
        print(f"modecount_ulp_circle: dh[{i},{j}] ({dh[i, j]!r}, {dl[i, j]!r}) -> ({dh2[i, j]!r}, {dl2[i, j]!r}) "
              f"(shift {shift:+d}, part {part})", flush=True)
        return orig(M, nh, nl, dh2, dl2, ph, pl, *args, **kw)

    modecount_cx.CSolver = patched
    script = os.path.join(hp, "modecount_circle.py")
    sys.argv = [script] + sys.argv[6:]
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
