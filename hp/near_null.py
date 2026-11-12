#!/usr/bin/env python3
"""The weakest directions of the chain's Newton system at a stored state, and where in the unknowns they live.

    cd hp && python3 near_null.py RAWDIR STATE [--k=4] [--its=16] [--threads=2] [--kfreeze=K] [--keep_plan=1]
        [--json=OUT] [--planted=1 [--w=1e-30]] [--mutate=scale_p1|sigma|scale_block]

The pinned system (the state's kfreeze, as anm_run.py solves it) is factored once in octo-double (Solver.newton_o, the
LU kept), on a plan made again at the state (Solver.plan at 1e-32, 0.05); --keep_plan=1 takes the state's stored plan
instead, the one anm_run.py's base polish keeps. Subspace iteration on (J^T J)^-1 = J^-1 J^-T, each step one
transpose solve and one solve per vector (Solver.olu_solve), with the vectors orthonormalized in f64, converges to the
right singular vectors of the k smallest singular values; sigma_i is 1 / sqrt(|x_i|) for the normalized v_i, and the
change of sigma_i over the last two iterations is printed as its convergence. Each vector is then split over the
layout of z (hpkernel.driver: p1, c0 | node_i: p, m, a, c | m_h, a_h | Delta), and each block's share is projected on
its harmonics over the half grid: odd harmonics for p1, p, m and m_h, even ones for c0, a, c and a_h. Printed:
sigma_i, its convergence, the five largest (block, harmonic) components and the share of the norm in Delta.

--kfreeze=K factors the system with a different freeze (1: harmonic 3 of p1 held too), the Jacobian at the same
state, to test whether a weak direction is a given harmonic.

The first Newton step at the state is printed and stored in two forms, from the same LU. "step" is newton_o's dz =
-J^-1 F, where the kernel's frozen rows read p1's held harmonics themselves and so drive them to zero. "pinned step"
is dz + J^-1 c (pinning.PinnedSolver, octo_corrector.PinnedO), with c the held offset at the state's own coefficients
of p1's harmonics (kfreeze, 13] in eight parts: the step of anm_run.py's base polish (--held_o 1, the fit's increment
zero at the state's D), on its system when the plan is the stored one. For each step: max |dz|, |dz|, its Delta
component, V^T dz on the k vectors of the subspace iteration and the share |V^T dz| / |dz| of the step on their span,
then for each block of z its max, its 2-norm and its three largest harmonics.

Check (exit 1 on failure): sigma_1 converged to 1e-3 relative, and ||J v_1|| / sigma_1 within 1e-2 of 1. v_1 is
refined in octo-double by three more steps of inverse iteration (scaled with od_axpy of multidouble's Python package),
and J v_1 is the exact octo-double Jacobian (Solver.jacobian_o, forward-mode duals) times v_1 summed in mpmath at 140
digits: with ||J|| near 1e34 and sigma near 1e-27 the product needs about 77 digits, which a vector in f64 or a central
difference of the residual cannot give. Control: --mutate=sigma reports sigma_1 1.05 times too large to the
comparison, and the check must fail (at D 242.36 the ratio is then 0.952). The check does not see a mis-applied solve
of a few percent: --mutate=scale_p1, the p1 rows of every right-hand side times 1.05, moves the ratio by 0.6% only
(1.0058 at D 242.36), because the weakest vector hardly changes; a single row times 1.01 moves it less (#488).

--planted=1 checks the printed step against a planted one (exit 1 on failure, #501). With w = |w| (v_1 + g) / sqrt(2),
g a random unit vector (seed 2) and |w| from --w, the system G(x) = F(x) - F(z) has the root z exactly. At the planted
start z - w (od_axpy, exact in eight parts) its residual is -J w + H(w, w) / 2, a chosen vector in the Jacobian's range,
and its Newton step newton_o(z - w).dz + J^-1 F(z) (the LU at z - w, F(z) from residual_o in eight parts) is w + J^-1
H(w, w) / 2, whatever F(z) is. That step goes through the pinned step's path (dz + J^-1 c, with c = F(z) on every row)
and is printed. For every block the printed step, its norm, max and harmonics, and its Delta component and V^T dz,
must equal w's to 1e-10 relative. Its verdict has its own line, "planted check:", and the last line, "check:", then
reads pass only if both checks pass. Control: --mutate=scale_block multiplies the m_h block of every printed step by
1.01, and the planted check must fail.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hpkernel.driver import make_solver  # noqa: E402
from octo_corrector import load_o  # noqa: E402


def layout(M, K):
    """(name, start, parity) of each block of z: parity 1 for odd harmonics, 0 for even"""
    out = [("p1", 0, 1), ("c0", M, 0)]
    for i in range(K):
        for j, (f, par) in enumerate((("p", 1), ("m", 1), ("a", 0), ("c", 0))):
            out.append((f"node{i}.{f}", 2 * M + 4 * M * i + j * M, par))
    out += [("m_h", 2 * M + 4 * M * K, 1), ("a_h", 3 * M + 4 * M * K, 0)]
    return out


def harmonics(f, parity):
    """|coefficient| of each harmonic of the parity class on the half grid theta_j = pi j / M"""
    M = len(f)
    th = np.pi * np.arange(M) / M
    out = {}
    for k in range(parity, M, 2):
        a = (1 if k == 0 else 2) / M * np.sum(f * np.cos(k * th))
        b = 2 / M * np.sum(f * np.sin(k * th)) if k else 0.0
        out[k] = float(np.hypot(a, b))
    return out


def subspace(solve, L, k, its, seed=1):
    """the k weakest right singular vectors of the factored J by subspace iteration on (J^T J)^-1, with solve(b,
    transpose) applying J^-1 or J^-T to an f64 vector: (V, the estimates of sigma at each iteration)"""
    rng = np.random.default_rng(seed)
    V = np.linalg.qr(rng.standard_normal((L, k)))[0]
    hist = []
    for _ in range(its):
        X = np.column_stack([solve(solve(V[:, i], True)) for i in range(k)])
        hist.append(1 / np.sqrt(np.linalg.norm(X, axis=0)))
        V = np.linalg.qr(X)[0]
    return V, hist


MUT_BLOCK = "m_h"   # the block --mutate=scale_block scales in every printed step


def printed(d8, blocks, M, mutate=""):
    """the step that is printed: the f64 sum of an eight-part step's two leading parts (the control scales one block)"""
    d = np.asarray(d8[0], dtype=float) + np.asarray(d8[1], dtype=float)
    if mutate == "scale_block":
        o = {n: o for n, o, _ in blocks}[MUT_BLOCK]
        d[o:o + M] *= 1.01
    return d


def step_report(d, V, blocks, M, ntop=3):
    """max |d|, |d|, the Delta component, V^T d and the share of |d| on span(V); per block its max, 2-norm and ntop
    largest harmonics"""
    c = V.T @ d
    nd = float(np.linalg.norm(d))
    out = dict(max=float(np.abs(d).max()), norm=nd, delta=float(d[-1]), vt=[float(x) for x in c],
               share=float(np.linalg.norm(c) / nd) if nd else 0.0, blocks={})
    for name, o, par in blocks:
        b = d[o:o + M]
        top = sorted(((a, h) for h, a in harmonics(b, par).items()), reverse=True)[:ntop]
        out["blocks"][name] = dict(max=float(np.abs(b).max()), norm=float(np.linalg.norm(b)),
                                   top=[(h, a) for a, h in top])
    return out


def print_step(label, r):
    vt = " ".join(f"{x:+.2e}" for x in r["vt"])
    print(f"  {label}: max {r['max']:.3e}, norm {r['norm']:.3e}, Delta {r['delta']:+.3e}; V^T dz {vt}, "
          f"share on span(V) {r['share']:.4f}")
    for name, b in r["blocks"].items():
        hs = ", ".join(f"k{h} {a:.2e}" for h, a in b["top"])
        print(f"    {name:<9} max {b['max']:.2e}, norm {b['norm']:.2e}; {hs}")


def planted_errors(d, r, w, V, blocks, M):
    """relative error of the printed step d and of its report r against the planted step w: per block the vector,
    its norm, max and reported harmonics; the Delta component; V^T d and the share on span(V)"""
    err = {}
    for name, o, par in blocks:
        wb = w[o:o + M]
        nb, mb, hw = np.linalg.norm(wb), np.abs(wb).max(), harmonics(wb, par)
        rb = r["blocks"][name]
        err[name] = float(max(np.linalg.norm(d[o:o + M] - wb) / nb, abs(rb["norm"] - nb) / nb, abs(rb["max"] - mb) / mb,
                              max(abs(a - hw[h]) for h, a in rb["top"]) / max(hw.values())))
    err["Delta"] = float(max(abs(d[-1] - w[-1]), abs(r["delta"] - w[-1])) / abs(w[-1]))
    cw = V.T @ w
    sw = np.linalg.norm(cw) / np.linalg.norm(w)
    err["V^T"] = float(max(np.linalg.norm(np.asarray(r["vt"]) - cw) / np.linalg.norm(cw), abs(r["share"] - sw) / sw))
    return err


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(args) < 2:
        sys.exit(__doc__)
    raw, st = args
    k = int(opts.get("k", 4))
    its = int(opts.get("its", 16))
    z, m = load_o(os.path.join(raw, st))
    z = [np.ascontiguousarray(np.asarray(p, dtype=float)) for p in z]
    M, nodes = int(m["M"]), list(m["nodes"])
    D = m.get("D_str", repr(float(m["D"])))
    kf = int(opts.get("kfreeze", m.get("kfreeze", 3)))
    s = make_solver(D, M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130),
                    threads=int(opts.get("threads", 2)), kfreeze=kf)
    keep_plan = opts.get("keep_plan") == "1"
    s.set_segs(m["segs"] if keep_plan else s.plan(z[0], z[1], 1e-32, 0.05))
    L = len(z[0])
    t0 = time.time()
    out = s.newton_o(z)
    print(f"{st}: D {float(m['D']):g}, L {L}, octo-double LU {time.time() - t0:.0f} s, pivots {out[3]:.1e}..{out[4]:.1e}", flush=True)
    # the first Newton step: newton_o's dz, and dz + J^-1 c with the base polish's held offset c (eight parts)
    from multidouble import od_axpy
    from pinning import PinnedSolver, band_coeffs, band_field
    ks_held = [h for h in range(kf + 1, 14) if h % 2 == 1]
    hc = band_coeffs(z[0][:M], z[1][:M], M, ks_held, exact=True, rest=[p[:M] for p in z[2:]])
    hf = band_field(hc, ks_held, M, parts=8)
    c8 = [np.ascontiguousarray(p) for p in PinnedSolver(s, M, hf[0], hf[1], h_rest=hf[2:]).c8]
    dz8 = [np.asarray(p) for p in out[0]]
    dpin8 = od_axpy(1.0, *[np.asarray(p) for p in s.olu_solve(c8)], *dz8)
    zero = np.zeros(L)
    scale = np.ones(L)
    if opts.get("mutate") == "scale_p1":
        scale[:M] = 1.05

    def solve(b, transpose=False):
        x = s.olu_solve([np.ascontiguousarray(b * scale)] + [zero] * 7, transpose)
        return np.asarray(x[0]) + np.asarray(x[1])

    V, hist = subspace(solve, L, k, its)
    sig = hist[-1]
    conv = np.abs(hist[-1] - hist[-2]) / hist[-1]
    # v_1 refined in octo-double, then ||J v_1|| from the exact Jacobian at 140 digits: it must be sigma_1
    import mpmath as mp
    mp.mp.dps = 140
    x8 = [np.ascontiguousarray(V[:, 0])] + [zero] * 7
    for _ in range(3):
        y = s.olu_solve([np.ascontiguousarray(x8[0] * scale)] + [np.ascontiguousarray(p) for p in x8[1:]], True)
        y = s.olu_solve([np.ascontiguousarray(np.asarray(y[0]) * scale)] + [np.ascontiguousarray(p) for p in y[1:]])
        nrm = float(np.linalg.norm(np.asarray(y[0])))
        x8 = [np.ascontiguousarray(p) for p in od_axpy(1.0 / nrm, *[np.asarray(p) for p in y], *([zero] * 8))]
    t1 = time.time()
    J = [np.asarray(p) for p in s.jacobian_o(z)]
    xv = [mp.fsum(mp.mpf(float(x8[q][c])) for q in range(8)) for c in range(L)]
    nz = [c for c in range(L) if xv[c] != 0]
    jv = [mp.fsum(mp.fsum(mp.mpf(float(J[q][r, c])) for q in range(8) if J[q][r, c] != 0) * xv[c] for c in nz) for r in range(L)]
    jn = mp.sqrt(mp.fsum(a * a for a in jv)) / mp.sqrt(mp.fsum(a * a for a in xv))
    ratio = float(jn / (sig[0] * (1.05 if opts.get("mutate") == "sigma" else 1.0)))
    print(f"  ||J v_1|| = {mp.nstr(jn, 6)} from the exact Jacobian at 140 digits ({time.time() - t1:.0f} s)", flush=True)
    blocks = layout(M, len(nodes))
    rec = {"state": st, "D": float(m["D"]), "pivots": [out[3], out[4]], "vectors": []}
    for i in range(k):
        v = V[:, i]
        comps = []
        for name, o, par in blocks:
            for h, a in harmonics(v[o:o + M], par).items():
                comps.append((a, name, h))
        comps.sort(reverse=True)
        top = ", ".join(f"{name} k{h} {a:.2f}" for a, name, h in comps[:5])
        share = {name: float(np.sum(v[o:o + M] ** 2)) for name, o, _ in blocks}
        fields = sorted(share.items(), key=lambda x: -x[1])[:3]
        print(f"  sigma_{i + 1} {sig[i]:.3e} (converged to {conv[i]:.0e}); delta share {v[-1] ** 2:.1e}; "
              f"blocks {', '.join(f'{n} {w:.2f}' for n, w in fields)}; top {top}")
        rec["vectors"].append(dict(sigma=float(sig[i]), conv=float(conv[i]), top=[(n, h, a) for a, n, h in comps[:8]],
                                   blocks=share, delta=float(v[-1] ** 2)))
    assert len(blocks) * M + 1 == L and all(o == j * M for j, (_, o, _) in enumerate(blocks)), "layout does not tile z"
    mut = opts.get("mutate", "")
    rec["plan"] = "stored" if keep_plan else "new"
    for key, label, d8 in (("step", "step dz = -J^-1 F (newton_o)", dz8),
                           ("step_pinned", "pinned step dz + J^-1 c (held offset at the state's coefficients)", dpin8)):
        rec[key] = step_report(printed(d8, blocks, M, mut), V, blocks, M)
        print_step(label, rec[key])
    ok_p = True
    if opts.get("planted") == "1":
        # G(x) = F(x) - F(z) has the root z; its Newton step at z - w is w to first order (docstring)
        wn = float(opts.get("w", 1e-30))
        g = np.random.default_rng(2).standard_normal(L)
        w = (V[:, 0] + g / np.linalg.norm(g)) * (wn / np.sqrt(2))
        w8 = [np.ascontiguousarray(w)] + [zero] * 7
        F8 = [np.ascontiguousarray(np.asarray(p)) for p in s.residual_o(z)]
        zp = [np.ascontiguousarray(p) for p in od_axpy(-1.0, *w8, *z)]
        t2 = time.time()
        outp = s.newton_o(zp)
        dp8 = od_axpy(1.0, *[np.asarray(p) for p in s.olu_solve(F8)], *[np.asarray(p) for p in outp[0]])
        dp = printed(dp8, blocks, M, mut)
        rp = step_report(dp, V, blocks, M)
        print(f"  planted start z - w, |w| {np.linalg.norm(w):.3e} (v_1 and a random unit vector), max|F(z)| "
              f"{np.abs(F8[0]).max():.2e}, octo-double LU {time.time() - t2:.0f} s", flush=True)
        print_step("planted step newton_o(z - w).dz + J^-1 F(z)", rp)
        err = planted_errors(dp, rp, w, V, blocks, M)
        worst = sorted(err.items(), key=lambda x: -x[1])
        ok_p = bool(worst[0][1] <= 1e-10)
        print(f"  planted step against w, relative: largest {', '.join(f'{n} {e:.1e}' for n, e in worst[:5])}; "
              f"whole vector {np.linalg.norm(dp - w) / np.linalg.norm(w):.1e}")
        print("planted check:", "pass" if ok_p else "FAIL", "(every block within 1e-10)")
        rec["planted"] = dict(w_norm=float(np.linalg.norm(w)), seed=2, F_z_max=float(np.abs(F8[0]).max()),
                              step=rp, errors=err, mutate=mut, ok=ok_p)
    ok = bool(conv[0] < 1e-3 and abs(ratio - 1) < 1e-2) and ok_p
    print(f"  ||J v_1|| / sigma_1 = {ratio:.6f}")
    print("check:", "pass" if ok else "FAIL")
    if opts.get("json"):
        json.dump(rec, open(opts["json"], "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
