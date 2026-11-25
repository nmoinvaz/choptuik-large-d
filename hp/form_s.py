#!/usr/bin/env python3
"""Form S of the DSS system (hp/src/form_s.rs, #1088): B = 1/a^2 slaved to p, m, c by the momentum constraint at every x,
so the unknowns are p1, c0 | node i: p, m, c | m_h | Delta (3M(K+1) + 1; form A's 4M(K+1) + 1 without a and a_h).

    python3 form_s.py check STATE.npz [STATE.npz ...] [--kfreeze none|13] [--plan_tol 1e-32] [--it_max 12]
        [--tol 1e-28] [--threads 2] [--mutate none|flux]
    python3 form_s.py cond STATE.npz [...] [--kfreeze 13] [--parts 2|8] [--w 8]

check: a converged form-A state (double-double) mapped to form S (a and a_h dropped), its form-S residual, then Newton
in double-double on form S to --tol; prints Delta_S - Delta_A and Delta_S against hp/data/delta_all.csv at the
state's D. --mutate flux (form_s.rs FS_MUT_FLUX: q's (x - c) and (x + c) swapped in the slaving) must move Delta.
cond: the Jacobian's LU pivots and weakest singular values, form A against form S, at the state's own layout.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from hpkernel.driver import dd_add, dd_from_mp, load, make_solver  # noqa: E402


def dmat_even(M, dps=40):
    """Folded d/dtheta for even fields (f(theta + pi) = f(theta)) on the half grid of N = 2M points, double-double
    (hi, lo), row-major: the full grid's trigonometric-interpolation derivative with both halves added"""
    with mp.workdps(dps):
        N = 2 * M
        out = []
        for i in range(M):
            for j in range(M):
                v = mp.mpf(0)
                for jj in (j, j + M):
                    if jj != i:
                        v += mp.mpf(0.5) * (-1) ** (i - jj) * mp.cot((i - jj) * mp.pi / N)
                out.append(v)
        hi, lo = dd_from_mp(out)
    return hi.reshape(M, M), lo.reshape(M, M)


def make_solver_s(D, M, nodes, kt=36, kc=40, ks=120, threads=0, kfreeze=None, segs=None):
    """make_solver with the even-sector d/dtheta set, for the form-S entry points (residual_s, jacobian_s, newton_s)"""
    s = make_solver(D, M, nodes=nodes, kt=kt, kc=kc, ks=ks, threads=threads, kfreeze=kfreeze)
    if segs:
        s.set_segs(segs)
    dh, dl = dmat_even(M)
    s.set_dmat_even(np.ascontiguousarray(dh), np.ascontiguousarray(dl))
    return s


def a_to_s(parts, M, K):
    """form A's vector (any number of part arrays) to form S's: each node's a and the horizon's a_h dropped"""
    out = []
    for z in parts:
        z = np.asarray(z, dtype=float)
        pieces = [z[:2 * M]]
        for i in range(K):
            o = 2 * M + 4 * M * i
            pieces += [z[o:o + 2 * M], z[o + 3 * M:o + 4 * M]]
        o = 2 * M + 4 * M * K
        pieces += [z[o:o + M], z[-1:]]
        out.append(np.ascontiguousarray(np.concatenate(pieces)))
    return out


def blocks_s(F, M, K):
    """max |F| per block of form S's rows (centre, segments, horizon; then the phase row)"""
    v = np.asarray(F[0]) + np.asarray(F[1])
    return [float(np.abs(v[b * 3 * M:(b + 1) * 3 * M]).max()) for b in range(K + 1)] + [float(abs(v[-1]))]


def newton_dd(s, z, tol=1e-28, it_max=12, halvings=8, verbose=True, mutate=0):
    """Newton in double-double on form S from z (hi, lo): step from newton_s, halved while max |F| does not fall"""
    zh, zl = (np.ascontiguousarray(np.asarray(p, dtype=float)) for p in z)
    F = s.residual_s([zh, zl], mutate)
    nF = float(np.abs(np.asarray(F[0]) + np.asarray(F[1])).max())
    hist = [dict(it=0, F=nF)]
    for it in range(it_max):
        if nF <= tol:
            break
        t0 = time.time()
        d, _, pmin, pmax = s.newton_s([zh, zl], mutate)
        dh, dl = np.asarray(d[0]), np.asarray(d[1])
        lam, acc = 1.0, False
        for _ in range(halvings + 1):
            th, tl = dd_add(zh, zl, lam * dh, lam * dl)
            th, tl = np.ascontiguousarray(th), np.ascontiguousarray(tl)
            Ft = s.residual_s([th, tl], mutate)
            nt = float(np.abs(np.asarray(Ft[0]) + np.asarray(Ft[1])).max())
            if np.isfinite(nt) and nt < nF:
                acc = True
                break
            lam *= 0.5
        rec = dict(it=it + 1, F=nt if acc else nF, dz=float(np.abs(dh).max()), dDelta=float(dh[-1]), lam=lam if acc else 0.0,
                   pmin=pmin, pmax=pmax, s=round(time.time() - t0, 1))
        hist.append(rec)
        if verbose:
            print(f"    it {it + 1}: |F| {nF:.3e} -> {rec['F']:.3e}, |dz| {rec['dz']:.2e}, dDelta {rec['dDelta']:+.3e}, "
                  f"lam {rec['lam']:g}, pivots {pmin:.1e}..{pmax:.1e}, {rec['s']} s", flush=True)
        if not acc:
            break
        zh, zl, nF = th, tl, nt
    return (zh, zl), nF <= tol, hist


def delta_csv(D, path=os.path.join(HERE, "data", "delta_all.csv")):
    """(Delta, err, kind) of delta_all.csv's row at D (within 1e-6), or None"""
    import csv
    for r in csv.DictReader(open(path)):
        if abs(float(r["D"]) - D) < 1e-6:
            with mp.workdps(40):
                return mp.mpf(r["Delta_q0"]) + mp.mpf(r["Delta_q1"] or 0), float(r["err"]), r["kind"]
    return None


def opts(argv, d):
    o, pos = dict(d), []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            k = a[2:]
            if "=" in k:
                k, v = k.split("=", 1)
            else:
                v, i = argv[i + 1], i + 1
            o[k] = v
        else:
            pos.append(a)
        i += 1
    return pos, o


def cmd_check(argv):
    pos, o = opts(argv, dict(kfreeze="none", plan_tol="1e-32", it_max="12", tol="1e-28", threads="2", mutate="none"))
    worst = 0.0
    for path in pos:
        zh, zl, meta = load(path)
        M, nodes = int(meta["M"]), list(meta["nodes"])
        K = len(nodes)
        with mp.workdps(60):
            D = mp.mpf(meta.get("D_str", repr(meta["D"])))
        Dn = D
        mut = 51 if o["mutate"] == "flux" else 0
        kf = None if o["kfreeze"] == "none" else int(o["kfreeze"])
        sa = make_solver(D, M, nodes=nodes, kt=meta.get("kt", 36), kc=meta.get("kc", 40), ks=meta.get("ks", 120),
                         threads=int(o["threads"]), kfreeze=kf)
        segs = sa.plan(zh, zl, float(o["plan_tol"]), 0.05)
        sa.set_segs(segs)
        Fa = sa.residual(zh, zl)
        nFa = float(np.abs(np.asarray(Fa[0]) + np.asarray(Fa[1])).max())
        s = make_solver_s(Dn, M, nodes, kt=meta.get("kt", 36), kc=meta.get("kc", 40), ks=meta.get("ks", 120),
                          threads=int(o["threads"]), kfreeze=kf, segs=segs)
        z = a_to_s([zh, zl], M, K)
        t0 = time.time()
        F = s.residual_s(z, mut)
        nF = float(np.abs(np.asarray(F[0]) + np.asarray(F[1])).max())
        print(f"{os.path.basename(path)}: D {float(D):.6g}, M {M}, K {K}, ks {meta.get('ks')}, kfreeze {kf}, "
              f"{sum(len(g) - 1 for g in segs)} steps; form A |F| {nFa:.2e}; form S at the form-A state |F| {nF:.2e} "
              f"(blocks {' '.join(f'{b:.1e}' for b in blocks_s(F, M, K))}), {time.time() - t0:.1f} s", flush=True)
        zs, conv, hist = newton_dd(s, z, float(o["tol"]), int(o["it_max"]), mutate=mut)
        with mp.workdps(60):
            dA = mp.mpf(float(zh[-1])) + mp.mpf(float(zl[-1]))
            dS = mp.mpf(float(zs[0][-1])) + mp.mpf(float(zs[1][-1]))
            row = delta_csv(float(D))
            diff = dS - dA
            dcsv = dS - row[0] if row else None
        worst = max(worst, abs(float(diff)))
        print(f"  {'converged' if conv else 'NOT converged'} in {len(hist) - 1} steps: Delta_S {mp.nstr(dS, 20)}, "
              f"Delta_S - Delta_A {mp.nstr(diff, 3)}"
              + (f", - delta_all ({row[2]}, err {row[1]:.1e}) {mp.nstr(dcsv, 3)}" if row else ""), flush=True)
    return 0


def main():
    cmd = sys.argv[1]
    return dict(check=cmd_check)[cmd](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
