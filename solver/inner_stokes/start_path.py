"""#1080: an M-harmonic inner start at a point (A1, w_f1) by continuation in (A, w_f) from a point (A0, w_f0) where the
fixed-(A, w_f) solve is known: the straight path in n steps, each step a hier.py solve (--fixA --schur, numba backend
of hier_nb.py) of r_3 = ... = r_M = 0 from the linear extrapolation of the last two solutions.

Why: at the second excited M = 3 zero x2 the fixed-(A, w_f) M = 5 solve from (B_3, C_3, 0, 0) or the seed file's B_5,
C_5 (about 0.03) does not converge (#1080): the M = 5 solution has B_5 near 0.49 and B_3 0.088 above its M = 3 value,
and the level-3 and level-5 Newton steps over that distance cross a near-fold of r_5 in (B_5, C_5). Along a path from
x1, whose M = 5 solution is known near its M = 3 harmonics (macm3's x1fm), each step moves the harmonics by about 0.01.

    python -u start_path.py M A0 WF0 A1 WF1 --v0=B3,C3[,B5,C5,...] [--n=20] [--xi1=0.2] [--floor_max=1e-3]
                            [--h3=1e-4] [--out=ROWS.jsonl] [--ref=B3,C3,... --tol=T] [--mutate=end]
  --h3=REL      level-3 finite-difference step (hier_nb.py's --h3; hier's 1e-6 makes the C_3 column noise near x2,
                where C_3 is about -0.002 and the float64 floor of r_3 about 5e-2)
  --ref, --tol  the check: exit 0 iff every component of the end's harmonics is within T of --ref
  --mutate=end  the control: the path ends at A1 + 0.01 (a point off the reference), so the check must fail
Rows (one per step): t, A, wf, |r_1|, |r_3|, ..., |r_M| (Schur-corrected, at xi1), the solved harmonics v (repr), seconds.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    argv = sys.argv[1:]
    o = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    if "h3" in o:                                   # before hier_nb is imported (it reads the environment)
        os.environ["HIER_NB_H3"] = o["h3"]
    import hier_nb  # noqa: F401                    numba backend and LazyHier, installed into hier
    import hier as H
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 5 or "v0" not in o:
        print(__doc__)
        return 2
    M = int(args[0])
    P0 = np.array([float(args[1]), float(args[2])])
    P1 = np.array([float(args[3]), float(args[4])])
    if o.get("mutate") == "end":
        P1 = P1 + np.array([0.01, 0.0])
    v = np.array([float(t) for t in o["v0"].split(",")])
    assert len(v) == M - 1, "--v0 needs B_3, C_3, ..., B_M, C_M"
    n, xi1 = int(o.get("n", 20)), float(o.get("xi1", 0.2))
    fmax = float(o.get("floor_max", 1e-3))
    out = o.get("out")
    hist = []
    for i in range(n + 1):
        t = i / n
        p = P0 + t * (P1 - P0)
        vp = v if len(hist) < 2 else 2 * hist[-1] - hist[-2]      # linear predictor in t
        h = H.Hier(M, xi1, top=3, verbose=0)
        h.schur = True
        h.floor_max = fmax                         # used by hier.py versions that have --floor_max (main's)
        t0 = time.time()
        x, r = h.solve_from(3, np.concatenate([p, vp]))
        v = x[2:].copy()
        hist.append(v)
        row = dict(t=t, A=float(p[0]), wf=float(p[1]), r=[float(np.hypot(r[k], r[k + 1])) for k in range(0, M + 1, 2)],
                   v=[float(t_) for t_ in v], sec=round(time.time() - t0, 1))
        print(json.dumps(row), flush=True)
        if out:
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
    if "ref" in o:
        ref = np.array([float(t_) for t_ in o["ref"].split(",")])
        d = np.abs(v - ref)
        ok = bool(np.all(d <= float(o.get("tol", 1e-6))))
        print(f"check: end harmonics {v.tolist()} against --ref, max difference {d.max():.3e} "
              f"(tol {o.get('tol', 1e-6)}): {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
