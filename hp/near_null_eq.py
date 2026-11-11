#!/usr/bin/env python3
"""The weakest directions of the chain's Newton system with its rows and columns equilibrated, at a stored state (#502).

    cd hp && python3 near_null_eq.py RAWDIR STATE [--k=4] [--its=16] [--threads=2] [--tol=1e-12] [--plant=SEED]
        [--refine=3] [--jv=1] [--json=OUT] [--mutate=rows_only|sigma]

The pinned system is factored once in octo-double exactly as near_null.py does it (the same make_solver call and plan,
Solver.newton_o, the LU kept), and J is the exact octo-double Jacobian (Solver.jacobian_o), whose f64 leading part
gives the norms. Diagonal row and column scalings D_r, D_c come from Ruiz's iteration [1], which divides every row and
column by the square root of its norm until all norms are within --tol of 1, first in the max norm and then in the
2-norm. Singular values of J_eq = D_r J D_c come from near_null.subspace on (J_eq^T J_eq)^-1, with the solves
J_eq^-1 b = D_c^-1 J^-1 D_r^-1 b and J_eq^-T b = D_r^-1 J^-T D_c^-1 b through the octo-double LU (Solver.olu_solve).
Printed for each stage, as near_null prints them: sigma_1..k, their convergence (the change over the last two
iterations), the top (block, harmonic) components and the Delta share of v_i (the scaled unknowns) and of D_c v_i (the
direction in z), and the ranges of D_r and D_c.

The max-norm stage's fixed point is not unique. (1, a; b, 1) with a, b < 1 is equilibrated, and so is (1, ta; b/t, 1)
for b <= t <= 1/a, so the matrix it reaches depends on where it starts, and a column scaling of J moves its sigma. The
2-norm stage is Ruiz's 1-norm iteration on |a_ij|^2, whose limit, every row and column of D_r J D_c of unit 2-norm, is
the doubly stochastic scaling of |J|^2. That scaling is unique when |J|^2 has total support [2], so its sigma depend on
J's rows and columns only up to their scale. The 2-norm stage starts from the max-norm stage's scalings, and the
2-norm stage's sigma are the result.

Check (exit 1 on failure), --plant=SEED: a planted column scaling. J S with S_j = 10^u_j, u_j uniform in [-20, 20]
(numpy default_rng(SEED)), is equilibrated from |J_0| S by the same two stages, and its sigma come from the same
subspace iteration (the same start) with the solves (J S)^-1 b = S^-1 J^-1 b and (J S)^-T b = J^-T S^-1 b. The 2-norm
stage's sigma_1..k of J S must equal those of J to 1e-3 relative. The max-norm stage's are printed beside them and are
not part of the check. Control: --mutate=rows_only scales the rows only, for J and for J S, and the check must fail.

Check (exit 1 on failure), --jv=1: ||J_eq v_1|| / sigma_1 within 1e-2 of 1, as near_null.py checks it. v_1 is refined
by --refine steps of inverse iteration through the LU with the scalings applied at 140 digits (mpmath), and J_eq v_1 is
D_r times the eight-part Jacobian times D_c v_1, summed at 140 digits. Control: --mutate=sigma reports sigma_1 1.05
times too large to the comparison, and the check must fail.

[1] D. Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in matrices", Rutherford Appleton
    Laboratory report RAL-TR-2001-034, 2001.
[2] R. Sinkhorn and P. Knopp, "Concerning nonnegative matrices and doubly stochastic matrices", Pacific J. Math. 21
    (1967) 343-348.
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
from near_null import harmonics, layout, subspace  # noqa: E402
from octo_corrector import load_o, o8  # noqa: E402

mp.mp.dps = 140


def ruiz(A, tol, itmax=(1000, 1000000), cols=True):
    """Ruiz's iteration on the nonnegative f64 matrix A, the max-norm stage and then the 2-norm stage, each to tol:
    {stage: (D_r, D_c, iterations, the largest |1 - norm| of a row or column)}. cols=False scales the rows only."""
    n, m = A.shape
    dr, dc = np.ones(n), np.ones(m)
    out = {}
    for it in range(itmax[0] + 1):
        B = A * dr[:, None] * dc[None, :]
        r, c = B.max(axis=1), B.max(axis=0)
        err = max(np.abs(1 - r).max(), np.abs(1 - c).max() if cols else 0.0)
        if err < tol or it == itmax[0]:
            break
        dr /= np.sqrt(r)
        if cols:
            dc /= np.sqrt(c)
    out["max"] = (dr.copy(), dc.copy(), it, float(err))
    # 2-norm stage on the max-norm stage's matrix, whose entries are at most 1: its squares neither overflow nor lose a
    # row's or a column's leading entries. er, ec are the 2-norm stage's own factors.
    B2 = (A * dr[:, None] * dc[None, :]) ** 2
    er, ec = np.ones(n), np.ones(m)
    for it in range(itmax[1] + 1):
        r2 = er ** 2 * (B2 @ ec ** 2)
        c2 = ec ** 2 * (er ** 2 @ B2)
        err = max(np.abs(1 - np.sqrt(r2)).max(), np.abs(1 - np.sqrt(c2)).max() if cols else 0.0)
        if err < tol or it == itmax[1]:
            break
        er /= r2 ** 0.25
        if cols:
            ec /= c2 ** 0.25
    out["2"] = (dr * er, dc * ec, it, float(err))
    return out


def scaled(solve, R, C):
    """the solve of R J C from the solve of J: (R J C)^-1 b = C^-1 J^-1 R^-1 b, (R J C)^-T b = R^-1 J^-T C^-1 b"""
    def f(b, transpose=False):
        return solve(b / C, True) / R if transpose else solve(b / R) / C
    return f


def components(v, blocks, M):
    """near_null's split of the unit vector v: (the (amplitude, block, harmonic) components sorted, the block shares,
    the Delta share)"""
    comps = sorted(((a, name, h) for name, o, par in blocks for h, a in harmonics(v[o:o + M], par).items()), reverse=True)
    share = {name: float(np.sum(v[o:o + M] ** 2)) for name, o, _ in blocks}
    return comps, share, float(v[-1] ** 2)


def report(tag, V, hist, C, blocks, M):
    """print and return sigma_1..k with their convergence and the components of v_i and of D_c v_i"""
    sig = hist[-1]
    conv = np.abs(hist[-1] - hist[-2]) / hist[-1]
    rec = []
    for i in range(V.shape[1]):
        row = dict(sigma=float(sig[i]), conv=float(conv[i]))
        line = f"  {tag} sigma_{i + 1} {sig[i]:.4e} (converged to {conv[i]:.0e})"
        for where, v in (("scaled", V[:, i]), ("in z", C * V[:, i] / np.linalg.norm(C * V[:, i]))):
            comps, share, delta = components(v, blocks, M)
            top = ", ".join(f"{name} k{h} {a:.2f}" for a, name, h in comps[:5])
            fields = sorted(share.items(), key=lambda x: -x[1])[:3]
            line += (f"\n      {where}: delta share {delta:.1e}; blocks {', '.join(f'{n} {w:.2f}' for n, w in fields)}; "
                     f"top {top}")
            row[where] = dict(delta=delta, blocks=share, top=[(n, h, a) for a, n, h in comps[:8]])
        print(line, flush=True)
        rec.append(row)
    return rec


def jv_ratio(s, J, R, C, v, refine):
    """||R J C x|| / ||x|| for x, v refined by inverse iteration through the octo-double LU, all at 140 digits"""
    L = len(v)
    Rm = [mp.mpf(float(a)) for a in R]
    Cm = [mp.mpf(float(a)) for a in C]
    x = [mp.mpf(float(a)) for a in v]
    for _ in range(refine):
        y = mpv(o8(s.olu_solve(list(parts8([a / c for a, c in zip(x, Cm)])), True)), range(L))
        y = mpv(o8(s.olu_solve(list(parts8([a / (r * r) for a, r in zip(y, Rm)])))), range(L))
        y = [a / c for a, c in zip(y, Cm)]
        nrm = mp.sqrt(mp.fsum(a * a for a in y))
        x = [a / nrm for a in y]
    u = [a * c for a, c in zip(x, Cm)]
    nz = [c for c in range(L) if u[c] != 0]
    ju = [Rm[r] * mp.fsum(mp.fsum(mp.mpf(float(J[q][r, c])) for q in range(8) if J[q][r, c] != 0) * u[c] for c in nz)
          for r in range(L)]
    return mp.sqrt(mp.fsum(a * a for a in ju)) / mp.sqrt(mp.fsum(a * a for a in x))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(args) != 2:
        sys.exit(__doc__)
    raw, st = args
    k, its, tol = int(opts.get("k", 4)), int(opts.get("its", 16)), float(opts.get("tol", 1e-12))
    mutate = opts.get("mutate", "")
    if mutate not in ("", "rows_only", "sigma"):
        sys.exit(f"--mutate={mutate}: rows_only or sigma")
    cols = mutate != "rows_only"
    z, m = load_o(os.path.join(raw, st))
    z = [np.ascontiguousarray(np.asarray(p, dtype=float)) for p in z]
    M, nodes = int(m["M"]), list(m["nodes"])
    D = m.get("D_str", repr(float(m["D"])))
    s = make_solver(D, M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130),
                    threads=int(opts.get("threads", 2)), kfreeze=int(opts.get("kfreeze", m.get("kfreeze", 3))))
    s.set_segs(s.plan(z[0], z[1], 1e-32, 0.05))
    L = len(z[0])
    blocks = layout(M, len(nodes))
    t0 = time.time()
    out = s.newton_o(z)
    print(f"{st}: D {float(m['D']):g}, L {L}, octo-double LU {time.time() - t0:.0f} s, pivots {out[3]:.1e}..{out[4]:.1e}",
          flush=True)
    t0 = time.time()
    J = [np.asarray(p) for p in s.jacobian_o(z)]
    A = np.abs(J[0])
    print(f"  jacobian_o {time.time() - t0:.0f} s; |J| entries {A[A > 0].min():.1e}..{A.max():.1e}, row max-norms "
          f"{A.max(axis=1).min():.1e}..{A.max(axis=1).max():.1e}, column max-norms {A.max(axis=0).min():.1e}.."
          f"{A.max(axis=0).max():.1e}", flush=True)
    zero = np.zeros(L)

    def solve(b, transpose=False):
        x = s.olu_solve([np.ascontiguousarray(b)] + [zero] * 7, transpose)
        return np.asarray(x[0]) + np.asarray(x[1])

    def equilibrate(Am, tag):
        t = time.time()
        eq = ruiz(Am, tol, cols=cols)
        for stage, (dr, dc, n_it, err) in eq.items():
            print(f"  {tag}, {stage}-norm stage: {n_it} iterations, largest |1 - norm| {err:.1e}; D_r {dr.min():.2e}.."
                  f"{dr.max():.2e}, D_c {dc.min():.2e}..{dc.max():.2e}", flush=True)
        print(f"  {tag}: equilibration {time.time() - t:.0f} s", flush=True)
        return eq

    rec = dict(state=st, D=float(m["D"]), L=L, pivots=[out[3], out[4]], tol=tol, k=k, its=its, mutate=mutate, stages={})
    eq = equilibrate(A, "J")
    res = {}
    for stage, (dr, dc, n_it, err) in eq.items():
        V, hist = subspace(scaled(solve, dr, dc), L, k, its)
        res[stage] = (V, hist)
        rec["stages"][stage] = dict(iterations=n_it, err=err, D_r=[float(dr.min()), float(dr.max())],
                                    D_c=[float(dc.min()), float(dc.max())], vectors=report(f"J, {stage}-norm", V, hist, dc, blocks, M))
    rec["D_r"], rec["D_c"] = eq["2"][0].tolist(), eq["2"][1].tolist()
    ok = True

    if "plant" in opts:
        S = 10.0 ** np.random.default_rng(int(opts["plant"])).uniform(-20, 20, L)
        print(f"  planted S = 10^u, u uniform in [-20, 20], seed {opts['plant']}: {S.min():.1e}..{S.max():.1e}", flush=True)
        eqS = equilibrate(A * S[None, :], "J S")
        plant = {}
        for stage, (dr, dc, n_it, err) in eqS.items():
            V, hist = subspace(scaled(scaled(solve, np.ones(L), S), dr, dc), L, k, its)
            vecs = report(f"J S, {stage}-norm", V, hist, S * dc, blocks, M)
            rel = float(np.max(np.abs(hist[-1] / res[stage][1][-1] - 1)))
            # the scalings themselves: D_r' = t D_r and S D_c' = D_c / t at a unique fixed point
            lr = np.log10(dr / eq[stage][0])
            lc = np.log10(S * dc / eq[stage][1])
            spread = float(max(np.ptp(lr), np.ptp(lc)))
            print(f"  planted, {stage}-norm stage: max_i |sigma_i(J S) / sigma_i(J) - 1| = {rel:.2e}; spread of "
                  f"log10(D_r'/D_r) and log10(S D_c'/D_c) {spread:.1e}", flush=True)
            plant[stage] = dict(rel=rel, spread=spread, iterations=n_it, err=err, vectors=vecs)
        ok_p = bool(plant["2"]["rel"] < 1e-3)
        print(f"  check (planted column scaling, 2-norm stage, bound 1e-3): {'pass' if ok_p else 'FAIL'}"
              + (" [control rows_only]" if mutate == "rows_only" else ""), flush=True)
        rec["plant"] = dict(seed=int(opts["plant"]), ok=ok_p, stages=plant)
        ok = ok and ok_p

    if int(opts.get("jv", 1)):
        t0 = time.time()
        dr, dc = eq["2"][:2]
        V, hist = res["2"]
        jn = jv_ratio(s, J, dr, dc, V[:, 0], int(opts.get("refine", 3)))
        ratio = float(jn / (hist[-1][0] * (1.05 if mutate == "sigma" else 1.0)))
        ok_j = bool(abs(ratio - 1) < 1e-2)
        print(f"  ||J_eq v_1|| = {mp.nstr(jn, 6)} at 140 digits ({time.time() - t0:.0f} s); ||J_eq v_1|| / sigma_1 = "
              f"{ratio:.6f} (bound 1e-2): {'pass' if ok_j else 'FAIL'}", flush=True)
        rec["jv"] = dict(norm=float(jn), ratio=ratio, ok=ok_j)
        ok = ok and ok_j

    print("check:", "pass" if ok else "FAIL")
    rec["ok"] = ok
    if opts.get("json"):
        with open(opts["json"], "w") as fh:
            json.dump(rec, fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
