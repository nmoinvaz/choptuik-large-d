#!/usr/bin/env python3
"""The weakest directions of the chain's Newton system at a stored state, and where in the unknowns they live.

    cd hp && python3 near_null.py RAWDIR STATE [--k=4] [--its=16] [--threads=2] [--kfreeze=K] [--json=OUT]
        [--mutate=scale_p1|sigma]

The pinned system (the state's kfreeze, as anm_run.py solves it) is factored once in octo-double (Solver.newton_o, the
LU kept). Subspace iteration on (J^T J)^-1 = J^-1 J^-T, each step one transpose solve and one solve per vector
(Solver.olu_solve), with the vectors orthonormalized in f64, converges to the right singular vectors of the k smallest
singular values; sigma_i is 1 / sqrt(|x_i|) for the normalized v_i, and the change of sigma_i over the last two
iterations is printed as its convergence. Each vector is then split over the layout of z (hpkernel.driver: p1, c0 |
node_i: p, m, a, c | m_h, a_h | Delta), and each block's share is projected on its harmonics over the half grid: odd
harmonics for p1, p, m and m_h, even ones for c0, a, c and a_h. Printed: sigma_i, its convergence, the five largest
(block, harmonic) components and the share of the norm in Delta.

--kfreeze=K factors the system with a different freeze (1: harmonic 3 of p1 held too), the Jacobian at the same
state, to test whether a weak direction is a given harmonic.

Check (exit 1 on failure): sigma_1 converged to 1e-3 relative, and ||J v_1|| / sigma_1 within 1e-2 of 1. v_1 is
refined in octo-double by three more steps of inverse iteration (scaled with od_axpy of multidouble's Python package),
and J v_1 is the exact octo-double Jacobian (Solver.jacobian_o, forward-mode duals) times v_1 summed in mpmath at 140
digits: with ||J|| near 1e34 and sigma near 1e-27 the product needs about 77 digits, which a vector in f64 or a central
difference of the residual cannot give. Control: --mutate=sigma reports sigma_1 1.05 times too large to the
comparison, and the check must fail (at D 242.36 the ratio is then 0.952). The check does not see a mis-applied solve
of a few percent: --mutate=scale_p1, the p1 rows of every right-hand side times 1.05, moves the ratio by 0.6% only
(1.0058 at D 242.36), because the weakest vector hardly changes; a single row times 1.01 moves it less (#488).
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
    s = make_solver(D, M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130),
                    threads=int(opts.get("threads", 2)), kfreeze=int(opts.get("kfreeze", m.get("kfreeze", 3))))
    s.set_segs(s.plan(z[0], z[1], 1e-32, 0.05))
    L = len(z[0])
    t0 = time.time()
    out = s.newton_o(z)
    print(f"{st}: D {float(m['D']):g}, L {L}, octo-double LU {time.time() - t0:.0f} s, pivots {out[3]:.1e}..{out[4]:.1e}", flush=True)
    zero = np.zeros(L)
    scale = np.ones(L)
    if opts.get("mutate") == "scale_p1":
        scale[:M] = 1.05

    def solve(b, transpose=False):
        x = s.olu_solve([np.ascontiguousarray(b * scale)] + [zero] * 7, transpose)
        return np.asarray(x[0]) + np.asarray(x[1])

    rng = np.random.default_rng(1)
    V = np.linalg.qr(rng.standard_normal((L, k)))[0]
    hist = []
    for it in range(its):
        X = np.column_stack([solve(solve(V[:, i], True)) for i in range(k)])
        norms = np.linalg.norm(X, axis=0)
        hist.append(1 / np.sqrt(norms))
        V = np.linalg.qr(X)[0]
    sig = hist[-1]
    conv = np.abs(hist[-1] - hist[-2]) / hist[-1]
    # v_1 refined in octo-double, then ||J v_1|| from the exact Jacobian at 140 digits: it must be sigma_1
    import mpmath as mp
    from multidouble import od_axpy
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
    ok = bool(conv[0] < 1e-3 and abs(ratio - 1) < 1e-2)
    print(f"  ||J v_1|| / sigma_1 = {ratio:.6f}")
    print("check:", "pass" if ok else "FAIL")
    if opts.get("json"):
        json.dump(rec, open(opts["json"], "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
