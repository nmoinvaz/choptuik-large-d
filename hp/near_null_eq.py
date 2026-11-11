#!/usr/bin/env python3
"""The weakest directions of the chain's Newton system with its rows and columns equilibrated, at a stored state (#502).

    cd hp && python3 near_null_eq.py RAWDIR STATE [--k=4] [--its=16] [--threads=2] [--tol=1e-12] [--plant=SEED]
        [--refine=3] [--jv=1] [--json=OUT] [--mutate=rows_only|sigma]

The pinned system is factored once in octo-double exactly as near_null.py does it (the same make_solver call and plan,
Solver.newton_o, the LU kept), and J is the exact octo-double Jacobian (Solver.jacobian_o), whose f64 leading part
gives the norms. Diagonal row and column scalings D_r, D_c make J_eq = D_r J D_c, whose singular values come from
near_null.subspace on (J_eq^T J_eq)^-1, with the solves J_eq^-1 b = D_c^-1 J^-1 D_r^-1 b and J_eq^-T b = D_r^-1 J^-T
D_c^-1 b through the octo-double LU (Solver.olu_solve). Printed for each scaling, as near_null prints them: sigma_1..k,
their convergence (the change over the last two iterations), the top (block, harmonic) components and the Delta share
of v_i (the scaled unknowns) and of D_c v_i (the direction in z), and the ranges of D_r and D_c.

Three scalings. "Ruiz" is Ruiz's iteration [1] from J: every row and column divided by the square root of its
max-norm until all are within --tol of 1. Its fixed point is not unique. (1, a; b, 1) with a, b < 1 is equilibrated,
and so is (1, ta; b/t, 1) for b <= t <= 1/a, so the matrix it reaches depends on where it starts, and a column scaling
of J moves its sigma, at any tolerance (the spread of its scalings under the planted scaling below is printed at three
tolerances). "Curtis-Reid" is Curtis and Reid's scaling [2]: rho, gamma minimize the sum over J's nonzeros of
(log|J_ij| - rho_i - gamma_j)^2, D_r = e^-rho, D_c = e^-gamma, from the normal equations (dense, minimum-norm least
squares). The problem is linear in the logs and its solution is unique up to D_r t, D_c / t on a connected pattern,
so D_r J D_c does not depend on the scales of J's rows and columns. "Curtis-Reid + Ruiz", the result, is Ruiz's
iteration started from the Curtis-Reid scaling: its start does not depend on those scales, so neither does its fixed
point, and every row and column of J_eq has max-norm 1 to --tol.

Check (exit 1 on failure), --plant=SEED: a planted column scaling. J S with S_j = 10^u_j, u_j uniform in [-20, 20]
(numpy default_rng(SEED)), is scaled from |J_0| S by the same three methods, and its sigma come from the same
subspace iteration (the same start) with the solves (J S)^-1 b = S^-1 J^-1 b and (J S)^-T b = J^-T S^-1 b. The
result's sigma_1..k of J S must equal those of J to 1e-3 relative. The other two scalings' are printed beside them and
are not part of the check. Control: --mutate=rows_only scales the rows only (rho alone in Curtis-Reid, rows alone in
Ruiz), for J and for J S, and the check must fail.

Check (exit 1 on failure), --jv=1: ||J_eq v_1|| / sigma_1 within 1e-2 of 1 for the result, as near_null.py checks it.
v_1 is refined by --refine steps of inverse iteration through the LU with the scalings applied at 140 digits (mpmath),
and J_eq v_1 is D_r times the eight-part Jacobian times D_c v_1, summed at 140 digits. Control: --mutate=sigma reports
sigma_1 1.05 times too large to the comparison, and the check must fail.

[1] D. Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in matrices", Rutherford Appleton
    Laboratory report RAL-TR-2001-034, 2001.
[2] A. R. Curtis and J. K. Reid, "On the automatic scaling of matrices for Gaussian elimination", J. Inst. Maths
    Applics 10 (1972) 118-124.
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
METHODS = ("Ruiz", "Curtis-Reid", "Curtis-Reid + Ruiz")
RESULT = METHODS[2]


def ruiz(A, tol, dr=None, dc=None, cols=True, itmax=1000):
    """Ruiz's iteration in the max norm on the nonnegative f64 matrix A from the scalings dr, dc (ones by default):
    (D_r, D_c, iterations, the largest |1 - max-norm| of a row or column). cols=False scales the rows only."""
    dr = np.ones(A.shape[0]) if dr is None else dr.copy()
    dc = np.ones(A.shape[1]) if dc is None else dc.copy()
    for it in range(itmax + 1):
        B = A * dr[:, None] * dc[None, :]
        r, c = B.max(axis=1), B.max(axis=0)
        err = max(np.abs(1 - r).max(), np.abs(1 - c).max() if cols else 0.0)
        if err < tol or it == itmax:
            break
        dr /= np.sqrt(r)
        if cols:
            dc /= np.sqrt(c)
    return dr, dc, it, float(err)


def curtis_reid(A, cols=True):
    """Curtis and Reid's scaling of the nonnegative f64 matrix A: (D_r, D_c) = (e^-rho, e^-gamma) with rho, gamma
    minimizing the sum over the nonzeros of (log a_ij - rho_i - gamma_j)^2. cols=False fits rho alone (gamma = 0)."""
    P = A > 0
    lA = np.where(P, np.log(np.where(P, A, 1.0)), 0.0)
    nr, nc = P.sum(axis=1), P.sum(axis=0)
    if not cols:
        return np.exp(-lA.sum(axis=1) / nr), np.ones(A.shape[1])
    Pf = P.astype(float)
    N = np.block([[np.diag(nr.astype(float)), Pf], [Pf.T, np.diag(nc.astype(float))]])
    x = np.linalg.lstsq(N, np.concatenate([lA.sum(axis=1), lA.sum(axis=0)]), rcond=None)[0]
    n = A.shape[0]
    return np.exp(-x[:n]), np.exp(-x[n:])


def scalings(A, tol, cols=True):
    """{method: (D_r, D_c, Ruiz iterations, the largest |1 - max-norm| of a row or column of D_r A D_c)}"""
    out = {"Ruiz": ruiz(A, tol, cols=cols)}
    dr, dc = curtis_reid(A, cols)
    B = A * dr[:, None] * dc[None, :]
    out["Curtis-Reid"] = (dr, dc, 0, float(max(np.abs(1 - B.max(axis=1)).max(), np.abs(1 - B.max(axis=0)).max())))
    out[RESULT] = ruiz(A, tol, dr, dc, cols=cols)
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


def report(tag, V, hist, C, blocks, M, full=True):
    """print and return sigma_1..k with their convergence and (full) the components of v_i and of the direction C v_i"""
    sig = hist[-1]
    conv = np.abs(hist[-1] - hist[-2]) / hist[-1]
    rec = []
    for i in range(V.shape[1]):
        row = dict(sigma=float(sig[i]), conv=float(conv[i]))
        line = f"    {tag} sigma_{i + 1} {sig[i]:.4e} (converged to {conv[i]:.0e})"
        for where, v in (("scaled", V[:, i]), ("in z", C * V[:, i] / np.linalg.norm(C * V[:, i]))):
            comps, share, delta = components(v, blocks, M)
            row[where] = dict(delta=delta, blocks=share, top=[(n, h, a) for a, n, h in comps[:8]])
            if full:
                top = ", ".join(f"{name} k{h} {a:.2f}" for a, name, h in comps[:5])
                fields = sorted(share.items(), key=lambda x: -x[1])[:3]
                line += (f"\n        {where}: delta share {delta:.1e}; blocks "
                         f"{', '.join(f'{n} {w:.2f}' for n, w in fields)}; top {top}")
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
        # (R J C)^-1 (R J C)^-T x = C^-1 J^-1 R^-2 J^-T C^-1 x
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


def spread(eq, eqS, S):
    """the largest spread of log10(D_r'/D_r) and log10(S D_c'/D_c): 0 when J S's scaling is J's up to D_r t, D_c / t"""
    return float(max(np.ptp(np.log10(eqS[0] / eq[0])), np.ptp(np.log10(S * eqS[1] / eq[1]))))


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
          f"{A.max(axis=0).max():.1e}" + (" [control rows_only: rows scaled only]" if not cols else ""), flush=True)
    zero = np.zeros(L)

    def solve(b, transpose=False):
        x = s.olu_solve([np.ascontiguousarray(b)] + [zero] * 7, transpose)
        return np.asarray(x[0]) + np.asarray(x[1])

    def show(tag, eqs):
        for name, (dr, dc, n_it, err) in eqs.items():
            print(f"  {tag}, {name}: {n_it} Ruiz iterations, largest |1 - max-norm| {err:.1e}; D_r {dr.min():.2e}.."
                  f"{dr.max():.2e}, D_c {dc.min():.2e}..{dc.max():.2e}", flush=True)

    rec = dict(state=st, D=float(m["D"]), L=L, pivots=[out[3], out[4]], tol=tol, k=k, its=its, mutate=mutate,
               methods={})
    t0 = time.time()
    eqs = scalings(A, tol, cols)
    print(f"  scalings {time.time() - t0:.1f} s", flush=True)
    show("J", eqs)
    res = {}
    for name, (dr, dc, n_it, err) in eqs.items():
        t0 = time.time()
        V, hist = subspace(scaled(solve, dr, dc), L, k, its)
        print(f"  J, {name} ({time.time() - t0:.0f} s):", flush=True)
        res[name] = (V, hist)
        rec["methods"][name] = dict(iterations=n_it, err=err, D_r=[float(dr.min()), float(dr.max())],
                                    D_c=[float(dc.min()), float(dc.max())],
                                    vectors=report(name, V, hist, dc, blocks, M))
    rec["D_r"], rec["D_c"] = eqs[RESULT][0].tolist(), eqs[RESULT][1].tolist()
    ok = True

    if "plant" in opts:
        S = 10.0 ** np.random.default_rng(int(opts["plant"])).uniform(-20, 20, L)
        AS = A * S[None, :]
        print(f"  planted S = 10^u, u uniform in [-20, 20], seed {opts['plant']}: {S.min():.1e}..{S.max():.1e}", flush=True)
        ladder = {}
        for t in (1e-8, 1e-12, 1e-14):
            ladder[t] = spread(ruiz(A, t, cols=cols), ruiz(AS, t, cols=cols), S)
        print("  Ruiz from J, spread of the scalings under S by tolerance: "
              + ", ".join(f"tol {t:.0e} {v:.2e}" for t, v in ladder.items()), flush=True)
        eqS = scalings(AS, tol, cols)
        show("J S", eqS)
        plant = {}
        for name, (dr, dc, n_it, err) in eqS.items():
            V, hist = subspace(scaled(scaled(solve, np.ones(L), S), dr, dc), L, k, its)
            vecs = report(f"J S, {name}", V, hist, S * dc, blocks, M, full=False)
            rel = float(np.max(np.abs(hist[-1] / res[name][1][-1] - 1)))
            sp = spread(eqs[name], eqS[name], S)
            print(f"  planted, {name}: max_i |sigma_i(J S) / sigma_i(J) - 1| = {rel:.2e}; spread of the scalings "
                  f"{sp:.1e}", flush=True)
            plant[name] = dict(rel=rel, spread=sp, iterations=n_it, err=err, vectors=vecs)
        ok_p = bool(plant[RESULT]["rel"] < 1e-3)
        print(f"  check (planted column scaling, {RESULT}, bound 1e-3): {'pass' if ok_p else 'FAIL'}"
              + (" [control rows_only]" if not cols else ""), flush=True)
        rec["plant"] = dict(seed=int(opts["plant"]), ok=ok_p, ruiz_ladder={f"{t:.0e}": v for t, v in ladder.items()},
                            methods=plant)
        ok = ok and ok_p

    if int(opts.get("jv", 1)):
        t0 = time.time()
        dr, dc = eqs[RESULT][:2]
        V, hist = res[RESULT]
        jn = jv_ratio(s, J, dr, dc, V[:, 0], int(opts.get("refine", 3)))
        ratio = float(jn / (hist[-1][0] * (1.05 if mutate == "sigma" else 1.0)))
        ok_j = bool(abs(ratio - 1) < 1e-2)
        print(f"  {RESULT}: ||J_eq v_1|| = {mp.nstr(jn, 6)} at 140 digits ({time.time() - t0:.0f} s); "
              f"||J_eq v_1|| / sigma_1 = {ratio:.6f} (bound 1e-2): {'pass' if ok_j else 'FAIL'}"
              + (" [control sigma]" if mutate == "sigma" else ""), flush=True)
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
