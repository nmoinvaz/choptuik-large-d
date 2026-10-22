"""Outer (A, w) Newton for the M-harmonic selection from fixed-(A, w) solves, when hier.py's own level-1 Newton cannot be
used.

Why: at M = 7 (matching radius 0.15) each fixed-(A, w) solve (hier_nb.py --fixA) converges levels 3, 5, 7 to their
noise floors and reports the Schur-corrected r_1 with noise ~0.1, but hier's level-1 Newton re-solves level 3 from
linear predictions after steps of 1e-3 and stops those nested solves at |r_3| ~ 1e-3, where the leakage into r_1 exceeds
the step's effect, so no outer step is accepted (logs/hier_nb_M7_full_xi015.log). Here every point is a complete
fixed-(A, w) solve; the Jacobian is a forward difference over the first three points (steps 2e-3 in A, 1e-3 in
w/w_inf, so |dr_1| ~ 5-10 against noise ~0.1), every later point updates it by Broyden's rule, and the Newton step of
the last point is printed. A point is one log's FIXED-A line (A, w/w_inf, r_1 corrected).

    python3 m7_outer.py LOG_BASE LOG_DA LOG_DW [LOG_NEXT ...]
"""
import re
import sys
import numpy as np


def point(fn):
    txt = open(fn).read()
    m = re.findall(r"FIXED-A M (\d+): A ([0-9.]+)\s+w/w_inf ([0-9.]+).*?r_1 ([+-][0-9.]+e[+-]\d+)([+-][0-9.]+e[+-]\d+)i", txt)
    if not m:
        sys.exit(f"{fn}: no FIXED-A line")
    M, A, w, a, b = m[-1]
    return int(M), np.array([float(A), float(w)]), np.array([float(a), float(b)])


def main():
    pts = [point(fn) for fn in sys.argv[1:]]
    (M, p0, r0), (_, pa, ra), (_, pw, rw) = pts[:3]
    J = np.column_stack([(ra - r0) / (pa[0] - p0[0]), (rw - r0) / (pw[1] - p0[1])])
    print(f"M {M}; points (A, w/w_inf, r_1):")
    for (_, p, r), fn in zip(pts, sys.argv[1:]):
        print(f"  {p[0]:.9f} {p[1]:.9f}   r_1 {r[0]:+.4f}{r[1]:+.4f}i   ({fn})")
    print(f"forward-difference Jacobian (Re/Im rows; columns A, w/w_inf):\n{np.array2string(J, precision=1)}"
          f"   singular values {np.array2string(np.linalg.svd(J, compute_uv=False), precision=1)}")
    p, r = p0, r0
    for (_, pn, rn) in pts[3:]:                          # Broyden updates from the later points
        dp, dr = pn - p, rn - r
        J = J + np.outer(dr - J @ dp, dp) / (dp @ dp)
        p, r = pn, rn
    best = min(pts, key=lambda t: np.linalg.norm(t[2]))
    step = -np.linalg.solve(J, best[2])
    nxt = best[1] + step
    print(f"Newton step from the point with the smallest |r_1| ({np.linalg.norm(best[2]):.3f}): dA {step[0]:+.3e}, "
          f"d(w/w_inf) {step[1]:+.3e} -> A {nxt[0]:.9f}, w/w_inf {nxt[1]:.9f}")


if __name__ == "__main__":
    main()
