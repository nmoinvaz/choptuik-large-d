"""#1076: argcount's --m3mp on two cached points without value, one blocked by the 10 sigma guard and one with S_3 not
tuned below 1e-8, from fu022mp's pass-2 cache (raw I1021-macm3/fu022mp/m3/points_M3_r0.17.jsonl at efed06f1).

    python m3mp_check.py CACHE [--procs=2]       exit 0 when both checks pass (about 25 min on the Mac)

1. Without --m3mp both points stay without value and the cache is unchanged.
The points are numpy floats, as argcount's walk passes them (a plain repr of one is not a number string under numpy 2).
2. With --m3mp at least one gets a value, the rows are cached with fu = mp, and on reloading the cache neither point is
   wanted again and the same points have values. Works on copies of CACHE in a fresh temporary folder."""
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import argcount as ac  # noqa: E402

import numpy as np  # noqa: E402

P = [(np.float64(3.3185956), np.float64(0.2250978975)), (np.float64(3.2248456), np.float64(1.08154321))]   # as the walk
                                                                                                     # passes them


def main():
    src = sys.argv[1]
    procs = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--procs=")), "2"))
    T = tempfile.mkdtemp(prefix="m3mp_check_")
    n_src = sum(1 for _ in open(src))
    c0 = os.path.join(T, "off.jsonl")
    shutil.copy(src, c0)
    F0 = ac.Multiplier(3, c0, 1, r1=0.17, m3fix=True)
    F0.raw(P)
    off = [F0.vals[F0.key(p)] for p in P]
    ok1 = all(not math.isfinite(abs(v)) for v in off) and sum(1 for _ in open(c0)) == n_src
    print(f"1. without --m3mp: values {off}, cache unchanged: {sum(1 for _ in open(c0)) == n_src}: {'PASS' if ok1 else 'FAIL'}")
    c1 = os.path.join(T, "on.jsonl")
    shutil.copy(src, c1)
    F1 = ac.Multiplier(3, c1, procs, r1=0.17, m3fix=True, m3mp=True)
    wanted = [F1.mp_wanted(F1.key(p)) for p in P]
    F1.raw(P)
    on = [F1.vals[F1.key(p)] for p in P]
    F2 = ac.Multiplier(3, c1, 1, r1=0.17, m3fix=True, m3mp=True)
    again = [F2.mp_wanted(F2.key(p)) for p in P]
    same = all(math.isfinite(abs(F2.vals[F2.key(p)])) == math.isfinite(abs(v)) for p, v in zip(P, on))
    ok2 = all(wanted) and any(math.isfinite(abs(v)) for v in on) and not any(again) and same
    print(f"2. with --m3mp: wanted {wanted}, values {on}; reloaded: wanted again {again}, same points valued {same}: "
          f"{'PASS' if ok2 else 'FAIL'}")
    print(f"m3mp check: {'PASS' if ok1 and ok2 else 'FAIL'} (work folder {T})")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
