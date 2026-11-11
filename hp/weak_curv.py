#!/usr/bin/env python3
"""The pinned system's curvature on its weak subspace at a stored state (#499).

    cd hp && python3 weak_curv.py RAWDIR STATE [--k=4] [--its=16] [--threads=2] [--h=1e-20] [--seed=2] [--json=OUT]
        [--mutate=offdiag]

The pinned system (the held offset at the state's own coefficients, renode_o.pinned, as anm_run.py's base polish
forms it, on the state's stored plan) is factored once in octo-double (Solver.newton_o, the LU kept). Its k weakest
right singular vectors come from near_null.subspace in f64. An f64 basis lies off the weak subspace by about 1e-16,
which J's strong directions turn into a linear term near 1e18 |a|. So the basis V is refined by --refine=3 steps of
block inverse iteration through the octo-double LU, orthonormal at 140 digits, and every step from z is formed in eight
parts. The weak left subspace is U = orth(J^-T V). With J^-T V = U R
and V orthonormal, U^T J V = R^-T exactly, the reduced Jacobian B, whose singular values are sigma_1..k. The second
derivatives F''(v_j, v_k) are central second differences of the octo-double pinned residual at step h (three points
on the diagonal, four off it), and Q_ijk = u_i . F''(v_j, v_k) is the reduced curvature.

Printed: sigma from the iteration and from B, max|F''(v_j, v_k)| with the share of its norm on U, the matrices Q_i,
rho = sigma_min / max_i ||Q_i||_2, and |J^-1 F''(v_j, v_k)| with its largest, omega, Deuflhard's affine-covariant
curvature constant on span(V) in max norms: Newton's local region there is about |dz| < 2 / omega. Inside a ball of radius rho on span(V), the linear part of the reduced residual
exceeds its quadratic part. Newton's step from a residual r along U lies inside it only when |r| is below about
sigma_min rho.

Checks (exit 1 on failure):
  (a) F''(v_j, v_k) from h and from h/10 agrees to 1% (max norm of the difference over that of F''). It fails for
      --h=1e-12, where the higher orders dominate. The difference is the next order's, so it also gives the radius
      r_q on span(V) at which the next order equals the quadratic one.
  (b) The quadratic model predicts the residual at a planted point z + V a, with a random and |a| = 2 h, to 1%:
      F(z + V a) - F(z) against U B a + 1/2 F''(V a, V a). Control --mutate=offdiag drops the off-diagonal
      F''(v_j, v_k) from the model, and (b) must fail.
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freeze_round import mpv, parts8  # noqa: E402
from hpkernel.driver import make_solver  # noqa: E402
from multidouble import od_axpy  # noqa: E402
from near_null import subspace  # noqa: E402
from octo_corrector import load_o, o8  # noqa: E402
from renode_o import pinned  # noqa: E402

mp.mp.dps = 140


def mgs(W):
    """modified Gram-Schmidt of the mpf vectors W in order: (Q, R) with W_j = sum_i R_ij Q_i"""
    Q, k = [], len(W)
    R = [[mp.mpf(0)] * k for _ in range(k)]
    for j, w in enumerate(W):
        v = list(w)
        for i, q in enumerate(Q):
            R[i][j] = mp.fsum(a * b for a, b in zip(q, v))
            v = [a - R[i][j] * b for a, b in zip(v, q)]
        R[j][j] = mp.sqrt(mp.fsum(a * a for a in v))
        Q.append([a / R[j][j] for a in v])
    return Q, R


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(args) != 2:
        sys.exit(__doc__)
    raw, st = args
    k, its, h = int(opts.get("k", 4)), int(opts.get("its", 16)), float(opts.get("h", 1e-20))
    mutate = opts.get("mutate", "")
    if mutate not in ("", "offdiag"):
        sys.exit(f"--mutate={mutate}: offdiag")
    z, m = load_o(os.path.join(raw, st))
    M, nodes = int(m["M"]), list(m["nodes"])
    kf = int(m.get("kfreeze", 3))
    D = m.get("D_str", repr(float(m["D"])))
    s = make_solver(D, M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130),
                    threads=int(opts.get("threads", 2)), kfreeze=kf)
    s.set_segs(m["segs"] if m.get("segs") else s.plan(z[0], z[1], 1e-32, 0.05))
    P = pinned(s, z, M, kf)
    L = len(z[0])
    zero = np.zeros(L)
    t0 = time.time()
    out = s.newton_o(list(z))
    print(f"{st}: D {float(m['D']):.10g}, L {L}, octo-double LU {time.time() - t0:.0f} s, pivots {out[3]:.1e}..{out[4]:.1e}",
          flush=True)

    def solve(b, transpose=False):
        x = s.olu_solve([np.ascontiguousarray(b)] + [zero] * 7, transpose)
        return np.asarray(x[0]) + np.asarray(x[1])

    V, hist = subspace(solve, L, k, its)
    # An f64 basis is off the weak subspace by about 1e-16, and J's strong directions (to 3e34) turn that into a linear
    # term of about 1e18 |a|. So the basis is refined by block inverse iteration through the octo-double LU, kept
    # orthonormal at 140 digits, and every perturbation is formed in eight parts.
    t0 = time.time()
    X = mgs([[mp.mpf(float(x)) for x in V[:, j]] for j in range(k)])[0]
    for _ in range(int(opts.get("refine", 3))):
        X = mgs([mpv(o8(s.olu_solve(list(o8(s.olu_solve(list(parts8(x)), True))))), range(L)) for x in X])[0]
    Um, Rm = mgs([mpv(o8(s.olu_solve(list(parts8(x)), True)), range(L)) for x in X])
    R = np.array([[float(Rm[i][j]) for j in range(k)] for i in range(k)])
    U = np.array([[float(v) for v in u] for u in Um]).T
    B = np.linalg.inv(R).T
    svB = np.linalg.svd(B, compute_uv=False)[::-1]
    print("  sigma (iteration) " + ", ".join(f"{x:.4e}" for x in hist[-1])
          + "; sigma (B, refined basis) " + ", ".join(f"{x:.4e}" for x in svB) + f" ({time.time() - t0:.0f} s)", flush=True)

    F0 = P.residual_o(z)
    rec = dict(state=st, D=float(m["D"]), k=k, h=h, sigma=[float(x) for x in hist[-1]], sigma_B=[float(x) for x in svB])

    def at(c):
        """the pinned residual at z + sum_j c_j x_j, c a dict {j: coefficient}, the step in eight parts"""
        p = [mp.fsum(mp.mpf(cj) * X[j][r] for j, cj in c.items()) for r in range(L)]
        return P.residual_o(o8(od_axpy(1.0, *parts8(p), *z)))

    def comb(terms, scale=1.0):
        """scale times the sum of c * F over (c, F), the F in eight parts, in octo-double: (its f64 value, its eight
        parts)"""
        acc = [zero] * 8
        for c, F in terms:
            acc = o8(od_axpy(float(c), *o8(F), *acc))
        acc = o8(od_axpy(scale, *acc, *([zero] * 8)))
        return np.asarray(acc[0]) + np.asarray(acc[1]), acc

    def second(hh):
        """F''(v_j, v_k) for j <= k by central differences at step hh: {(j, k): (f64, eight parts)}"""
        out = {}
        for j in range(k):
            out[(j, j)] = comb([(1, at({j: hh})), (-2, F0), (1, at({j: -hh}))], 1 / hh ** 2)
            for q in range(j + 1, k):
                out[(j, q)] = comb([(1, at({j: hh, q: hh})), (-1, at({j: hh, q: -hh})), (-1, at({j: -hh, q: hh})),
                                    (1, at({j: -hh, q: -hh}))], 1 / (4 * hh ** 2))
        return out

    t0 = time.time()
    S1, S2 = second(h), second(h / 10)
    D1, D2 = {p: v[0] for p, v in S1.items()}, {p: v[0] for p, v in S2.items()}
    rel = {p: float(np.abs(D1[p] - D2[p]).max() / np.abs(D1[p]).max()) for p in D1}
    da = max(rel.values())
    ok_a = bool(da < 1e-2)
    # the difference is the O(h^2) term, c h^2 (1 - 1/100) relative; the next order equals F'' at 1/sqrt(c)
    r_q = min(h / np.sqrt(max(v, 1e-300) / 0.99) for v in rel.values())
    print(f"  (a) F'' from h {h:.0e} and h/10: max relative difference {da:.2e} (bound 1e-2, {time.time() - t0:.0f} s): "
          f"{'pass' if ok_a else 'FAIL'}; the next order equals F'' at |a| about {r_q:.1e}", flush=True)
    for (j, q), d in sorted(D1.items()):
        share = float(np.linalg.norm(U.T @ d) / np.linalg.norm(d))
        print(f"  F''(v{j + 1}, v{q + 1}): max {np.abs(d).max():.3e}, share on U {share:.2e}", flush=True)
    rec["Fpp"] = {f"{j + 1},{q + 1}": dict(max=float(np.abs(d).max()), share_U=float(np.linalg.norm(U.T @ d) / np.linalg.norm(d)))
                  for (j, q), d in D1.items()}

    def Fpp(j, q):
        return D1[(min(j, q), max(j, q))]

    Q = np.array([[[float(U[:, i] @ Fpp(j, q)) for q in range(k)] for j in range(k)] for i in range(k)])
    qn = [float(np.linalg.norm(Q[i], 2)) for i in range(k)]
    rho = float(svB[0] / max(qn))
    for i in range(k):
        print(f"  Q_{i + 1} (||.||_2 {qn[i]:.3e}): " + "; ".join(" ".join(f"{x:+.2e}" for x in row) for row in Q[i]), flush=True)
    print(f"  rho = sigma_min / max ||Q_i|| = {rho:.3e}; sigma_min rho = {svB[0] * rho:.3e}", flush=True)
    rec.update(Q=Q.tolist(), Q_norm=qn, rho=rho, sigma_rho=float(svB[0] * rho))

    # the curvature as Newton's step sees it: J^-1 F''(v_j, v_k) through the octo-double LU, from the eight-part F''
    # (an f64 F'' would come back from J^-1 as its rounding times 1/sigma). omega is Deuflhard's affine-covariant
    # constant on span(V) in max norms; Newton's local region there is about |dz| < 2 / omega
    K = {p: float(np.abs(np.asarray(x[0]) + np.asarray(x[1])).max())
         for p, x in ((p, o8(s.olu_solve(list(v[1])))) for p, v in S1.items())}
    omega = max(K.values())
    print("  |J^-1 F''(v_j, v_k)|: " + ", ".join(f"{j + 1}{q + 1} {v:.2e}" for (j, q), v in sorted(K.items()))
          + f"; omega {omega:.3e}, 2 / omega {2 / omega:.2e}", flush=True)
    rec.update(K={f"{j + 1},{q + 1}": v for (j, q), v in K.items()}, omega=omega)

    # (b) the model at a planted point z + V a
    rng = np.random.default_rng(int(opts.get("seed", 2)))
    a = rng.standard_normal(k)
    a *= 2 * h / np.linalg.norm(a)
    actual = comb([(1, at({j: float(a[j]) for j in range(k)})), (-1, F0)])[0]
    quad = sum(a[j] * a[q] * Fpp(j, q) for j in range(k) for q in range(k)
               if mutate != "offdiag" or j == q) / 2
    model = U @ (B @ a) + quad
    rb = float(np.abs(actual - model).max() / np.abs(actual).max())
    ok_b = bool(rb < 1e-2)
    print(f"  (b) planted point |a| {np.linalg.norm(a):.0e}: max|F(z + V a) - F(z)| {np.abs(actual).max():.3e}, model's "
          f"relative error {rb:.2e} (bound 1e-2): {'pass' if ok_b else 'FAIL'}", flush=True)
    rec.update(check_a=dict(rel=da, ok=ok_a, rel_pairs={f"{j + 1},{q + 1}": v for (j, q), v in rel.items()}),
               r_quad=float(r_q), check_b=dict(rel=rb, ok=ok_b, a_norm=float(np.linalg.norm(a))), mutate=mutate)
    ok = ok_a and ok_b
    print("check:", "pass" if ok else "FAIL")
    if opts.get("json"):
        with open(opts["json"], "w") as fh:
            json.dump(rec, fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
