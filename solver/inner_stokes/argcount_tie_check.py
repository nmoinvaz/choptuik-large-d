"""#1076: argcount's extended-precision rows are stored under the key the walk asked for, also at a decimal tie.

    python3 argcount_tie_check.py [--mutate=pykey]      exit 0 iff the stored key equals the walk's key at every point

The walk's points are numpy floats, whose round() differs from a Python float's at a tie (3.3192059515625 rounds to
...562 as a numpy float and ...563 as a Python float). value_mp is replaced by a stand-in that returns the row
value_point would (A = float of the repr), so the check runs in a second. --mutate=pykey keys the rows from their own
Python floats, as argcount did before the fix: the check must fail."""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import argcount as AC  # noqa: E402
import value_mp as VM  # noqa: E402


def fake(args):
    (A, wf, f3re, f3im), kw = args
    return dict(A=float(A), wf=float(wf), re=1.0, im=0.0, info=0.0, F3re=float(f3re), F3im=float(f3im), ymax=1.0,
                sig=1e-30, fu="mp", mp_dps=[34, 42], valued=True, R_over_sig=1e30, seconds=0.0)


def main():
    mutate = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--mutate=")), None)
    VM._batch_one = fake
    pts = [(np.float64(3.3192059515625), np.float64(0.22529321)), (np.float64(3.5526043890625), np.float64(0.41279321)),
           (np.float64(2.5), np.float64(0.3))]
    d = tempfile.mkdtemp(prefix="tie_")
    cache = os.path.join(d, "c.jsonl")
    with open(cache, "w") as fh:                    # float64 records without a value, as the walk left them
        for p in pts:
            k = AC.Multiplier.key(p)
            fh.write(json.dumps(dict(A=k[0], wf=k[1], r1=0.17, re=float("nan"), im=float("nan"), info=1.0,
                                     F3re=0.1, F3im=0.0, ymax=1.0, fu="f64")) + "\n")
    AC.MP_KEY_FROM_ROW = mutate == "pykey"
    m = AC.Multiplier(3, cache, 1, r1=0.17, m3mp=True)
    m.mp_batch(pts)
    ok = all(np.isfinite(m.vals.get(AC.Multiplier.key(p), np.nan)) for p in pts)
    m2 = AC.Multiplier(3, cache, 1, r1=0.17, m3mp=True)               # and after a reload from the cache file
    ok2 = all(m2.extra.get(AC.Multiplier.key(p), {}).get("fu") == "mp" for p in pts)
    print(f"tie check{' (mutate ' + mutate + ')' if mutate else ''}: every point valued under the walk's key: {ok}; "
          f"after reload: {ok2}: {'PASS' if ok and ok2 else 'FAIL'}")
    return 0 if ok and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
