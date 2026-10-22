"""Outer (A, w) Newton for the M-harmonic selection from fixed-(A, w) solves, when hier.py's own level-1 Newton cannot be
used.

Why: at M = 7 (matching radius 0.15) each fixed-(A, w) solve (hier_nb.py --fixA) converges levels 3, 5, 7 and reports
the Schur-corrected r_1, but hier's level-1 Newton re-solves level 3 from linear predictions after steps of 1e-3 and
stops those nested solves at |r_3| ~ 1e-3, where the leakage into r_1 exceeds the step's effect, so no outer step is
accepted (logs/hier_nb_M7_full_xi015.log). Here every point is a complete fixed-(A, w) solve.

Two modes.
  newton: the Jacobian is a forward difference over the first three points (steps 2e-3 in A, 1e-3 in w/w_inf), every
          later point updates it by Broyden's rule, and the Newton step from the point with the smallest |r_1| is
          printed. Used to walk to the zero.
  local:  the zero from a local Jacobian at a centre point. The centre's r_1 is the mean of its repeated solves; only
          solves that converged level 3 to |r_3| < 1e-6 are used (at xi1 = 0.15 the Schur-corrected r_1 carries an
          error of roughly 1e5-1e6 |r_3|, measured from repeated solves at one point). The error bar is the repeats'
          half range mapped through J^-1, plus the same noise's effect on J's columns.

    python3 m7_outer.py LOG_BASE LOG_DA LOG_DW [LOG_NEXT ...]
    python3 m7_outer.py --local CENTER_LOG [CENTER_LOG ...] -- LOG_DA LOG_DW
"""
import re
import sys
import numpy as np


def point(fn, with_r3=False):
    txt = open(fn).read()
    m = re.findall(r"FIXED-A M (\d+): A ([0-9.]+)\s+w/w_inf ([0-9.]+).*?r_1 ([+-][0-9.]+e[+-]\d+)([+-][0-9.]+e[+-]\d+)i", txt)
    if not m:
        sys.exit(f"{fn}: no FIXED-A line")
    M, A, w, a, b = m[-1]
    out = (int(M), np.array([float(A), float(w)]), np.array([float(a), float(b)]))
    if with_r3:
        r3 = re.findall(r"F_3/F_1 = .*? r_3 ([0-9.]+e[+-]\d+)", txt)
        out = out + (float(r3[-1]),)
    return out


def local(args):
    k = args.index("--")
    cen = [point(f, True) for f in args[:k]]
    offs = [point(f, True) for f in args[k + 1:]]
    for (_, p, r, r3), fn in zip(cen + offs, args[:k] + args[k + 1:]):
        print(f"  {p[0]:.9f} {p[1]:.9f}   r_1 {r[0]:+.4f}{r[1]:+.4f}i   |r_3| {r3:.1e}"
              f"{'' if r3 < 1e-6 else '  (rejected)'}   ({fn})")
    good = [c for c in cen if c[3] < 1e-6]
    if not good or any(o[3] >= 1e-6 for o in offs):
        sys.exit("not enough solves with |r_3| < 1e-6")
    p0 = good[0][1]
    R = np.array([c[2] for c in good])
    r0 = R.mean(axis=0)
    noise = 0.5 * float(np.ptp(R, axis=0).max()) if len(good) > 1 else float("nan")
    (_, pa, ra, _), (_, pw, rw, _) = offs
    J = np.column_stack([(ra - r0) / (pa[0] - p0[0]), (rw - r0) / (pw[1] - p0[1])])
    sv = np.linalg.svd(J, compute_uv=False)
    step = -np.linalg.solve(J, r0)
    z = p0 + step
    hmin = min(abs(pa[0] - p0[0]), abs(pw[1] - p0[1]))
    dz_noise = noise / sv[-1]
    dz_J = np.linalg.norm(step) * 2 * noise / (sv[-1] * hmin)
    print(f"centre r_1 = {r0[0]:+.4f}{r0[1]:+.4f}i from {len(good)} solves, noise (half range) {noise:.3f}")
    print(f"local Jacobian (Re/Im rows; A, w/w_inf):\n{np.array2string(J, precision=1)}   singular values "
          f"{np.array2string(sv, precision=1)}")
    print(f"zero: A {z[0]:.6f}, w/w_inf {z[1]:.6f}   (step {step[0]:+.2e}, {step[1]:+.2e}); error estimate "
          f"{dz_noise + dz_J:.1e} in (A, w/w_inf) ({dz_noise:.1e} from the centre's noise, {dz_J:.1e} from J)")


def newton(args):
    pts = [point(fn) for fn in args]
    (M, p0, r0), (_, pa, ra), (_, pw, rw) = pts[:3]
    J = np.column_stack([(ra - r0) / (pa[0] - p0[0]), (rw - r0) / (pw[1] - p0[1])])
    print(f"M {M}; points (A, w/w_inf, r_1):")
    for (_, p, r), fn in zip(pts, args):
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
    if sys.argv[1] == "--local":
        local(sys.argv[2:])
    else:
        newton(sys.argv[1:])
