#!/usr/bin/env python3
"""Issue #396: two plan_print.py records (two machines, the same state and M) compared entry by entry: the nodes, the
step counts per segment, and per segment how many points differ, the largest difference in ulps and in x.

    python3 hp/plan_diff.py A.json B.json [A2.json B2.json ...]

Exit 0 when every pair is bit-identical, 1 otherwise.
"""
import json
import sys

import numpy as np


def ulps(p, q):
    return int(abs(int(np.float64(p).view(np.int64)) - int(np.float64(q).view(np.int64))))


def diff(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    same = A["sha256"] == B["sha256"]
    print(f"== {a} against {b}: M {A['M']}, |F| {A['F']:.3e} and {B['F']:.3e}; nodes equal {A['nodes'] == B['nodes']}; "
          f"plans {'bit-identical' if same else 'differ'}")
    for i, (sa, sb) in enumerate(zip(A["segs"], B["segs"])):
        xa, xb = [float.fromhex(x) for x in sa], [float.fromhex(x) for x in sb]
        if len(xa) != len(xb):
            print(f"  segment {i}: {len(xa) - 1} against {len(xb) - 1} steps")
            continue
        u = [ulps(p, q) for p, q in zip(xa, xb)]
        print(f"  segment {i}: {len(xa) - 1} steps, {sum(x > 0 for x in u)} points differ, largest {max(u)} ulp, "
              f"largest |dx| {max(abs(p - q) for p, q in zip(xa, xb)):.1e}")
    return same


def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2:
        sys.exit(__doc__)
    ok = all([diff(a, b) for a, b in zip(args[::2], args[1::2])])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
