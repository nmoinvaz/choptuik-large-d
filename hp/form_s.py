#!/usr/bin/env python3
"""Form S of the DSS system (hp/src/form_s.rs, #1088): B = 1/a^2 slaved to p, m, c by the momentum constraint at every x,
so the unknowns are p1, c0 | node i: p, m, c | m_h | Delta (3M(K+1) + 1; form A's 4M(K+1) + 1 without a and a_h).

    python3 form_s.py check STATE.npz [STATE.npz ...] [--kfreeze none|13] [--plan_tol 1e-32] [--it_max 12]
        [--tol 1e-28] [--threads 2] [--mutate none|flux]
    python3 form_s.py cond STATE.npz [...] [--kfreeze 13] [--w 10] [--w2 4] [--iters 6] [--forms A,S] [--segs state|plan]
        [--ks state|N]
    python3 form_s.py jump OUTDIR --D 250 --states 'GLOB,...' [--Dmin 226 --Dmax 245.64 --deg 3 --kfreeze 13
        --plan_tol 1e-40 --ks sigma|auto|N --tol 1e-40 --it_max 16 --halvings 8 --threads 2 --delta series --mutate none|step
        --junk 0 --tol_dz 1e-45 --ls 0 --outer 8 --held 0]

jump: the rescaled guess of jump.py (p, m, p1, m_h times sqrt(n), c0 and c as (v - 1) n, fitted in 1/n; Delta from
Track C's series), in form S, so with no a and no horizon re-seed; then Newton in octo-double on form S (newton_s),
each step halved while max |F| does not fall. One record per step in OUTDIR/jump_s.jsonl, the state in
OUTDIR/jump_s_D<D>.npz (octo_corrector.save_o, form S's layout, meta form "S"). --mutate step flips each step's sign.
--junk W freezes the W weakest directions without Delta content (newton_o_s); --ls 1 with --junk W runs ls_solve (all W
weakest frozen in the inner solves, then steps along the Delta-carrying ones, --outer times at most).

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
            try:
                Ft = s.residual_s([th, tl], mutate)
                nt = float(np.abs(np.asarray(Ft[0]) + np.asarray(Ft[1])).max())
            except BaseException as e:          # a trial whose slaving LU is not finite (pyo3's PanicException)
                if isinstance(e, KeyboardInterrupt):
                    raise
                nt = float("nan")
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
    pos, o = opts(argv, dict(kfreeze="none", plan_tol="1e-32", it_max="12", tol="1e-28", threads="2", mutate="none", out=""))
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
        if o.get("out") and conv:
            from octo_corrector import save_o
            out = os.path.join(o["out"], f"s_D{float(D):.4f}.npz")
            zero = np.zeros(len(zs[0]))
            save_o(out, list(zs) + [zero] * 6, dict(meta, form="S", segs=[list(map(float, g)) for g in segs], kfreeze=kf,
                                                  source=path, parts=2))
            print(f"  saved {out}", flush=True)
    return 0


def cmd_planted(argv):
    """a converged form-S state (form_s.py check --out) with every unknown but Delta moved by rel (relative, smooth in
    tau, a random sign per field block, seed), solved again by newton_dd. Pass: converged and Delta within --pass of the state's.
    --mutate step (each Newton step taken with its sign flipped) must fail."""
    from octo_corrector import load_o
    pos, o = opts(argv, dict(rel="1e-8", seed="1", tol="1e-28", it_max="12", threads="2", mutate="none", **{"pass": "1e-26"}))
    ok_all = True
    for path in pos:
        z8, meta = load_o(path)
        M, nodes = int(meta["M"]), list(meta["nodes"])
        with mp.workdps(60):
            D = mp.mpf(meta.get("D_str", repr(meta["D"])))
        s = make_solver_s(D, M, nodes, kt=meta.get("kt", 36), kc=meta.get("kc", 40), ks=meta.get("ks", 120),
                          threads=int(o["threads"]), kfreeze=meta.get("kfreeze"), segs=meta["segs"])
        zh, zl = np.array(z8[0]), np.array(z8[1])
        # a smooth relative move: each field block times rel (1 + cos(2 theta) / 2) with a random sign per block (pi-periodic,
        # so that odd fields stay odd and even ones even); a move with
        # every harmonic would also move p1's highest harmonics, which the full system barely fixes (its junk floor)
        rng = np.random.default_rng(int(o["seed"]))
        th = np.pi * np.arange(M) / M
        nb = (len(zh) - 1) // M
        r = np.concatenate([float(o["rel"]) * rng.choice([-1, 1]) * (1 + 0.5 * np.cos(2 * th)) for _ in range(nb)] + [np.zeros(1)])
        ph, pl = dd_add(zh, zl, zh * r, np.zeros(len(zh)))
        if o["mutate"] == "step":
            zt, conv, hist = newton_dd_flip(s, (ph, pl), float(o["tol"]), int(o["it_max"]))
        else:
            zt, conv, hist = newton_dd(s, (ph, pl), float(o["tol"]), int(o["it_max"]))
        with mp.workdps(60):
            d = (mp.mpf(float(zt[0][-1])) + mp.mpf(float(zt[1][-1]))) - (mp.mpf(float(zh[-1])) + mp.mpf(float(zl[-1])))
        ok = conv and abs(d) <= mp.mpf(o["pass"])
        ok_all = ok_all and ok
        print(f"{os.path.basename(path)}: moved by {o['rel']}; {'converged' if conv else 'NOT converged'} in {len(hist) - 1}; "
              f"Delta - the state's {mp.nstr(d, 3)}: {'PASS' if ok else 'FAIL'}{' (--mutate step)' if o['mutate'] == 'step' else ''}",
              flush=True)
    return 0 if ok_all else 1


def newton_dd_flip(s, z, tol, it_max):
    """newton_dd with every step's sign flipped (planted's control)"""
    zh, zl = (np.ascontiguousarray(np.asarray(p, dtype=float)) for p in z)
    hist = [dict(it=0)]
    for it in range(it_max):
        d, _, _, _ = s.newton_s([zh, zl])
        th, tl = dd_add(zh, zl, -np.asarray(d[0]), -np.asarray(d[1]))
        zh, zl = np.ascontiguousarray(th), np.ascontiguousarray(tl)
        hist.append(dict(it=it + 1))
    F = s.residual_s([zh, zl])
    return (zh, zl), float(np.abs(np.asarray(F[0]) + np.asarray(F[1])).max()) <= tol, hist


def weakest(solve, L, w, iters, w2, seed=0):
    """the w weakest right singular values of J from solve(v, transpose) = J^-1 v or J^-T v (leading part): subspace
    iteration on (J^T J)^-1 with Rayleigh-Ritz, then w2 more on the complement of those the f64 Rayleigh-Ritz resolves
    (eigenvalues above 1e-12 of the largest), deflated. Returns [(sigma, vector)], weakest first."""
    rng = np.random.default_rng(seed)

    def apply(v):
        return solve(solve(v, True), False)

    def rr(V, Vd=None):
        W = np.column_stack([apply(V[:, j]) for j in range(V.shape[1])])
        if Vd is not None:
            W = W - Vd @ (Vd.T @ W)
        H = V.T @ W
        ev, U = np.linalg.eigh((H + H.T) / 2)
        return ev, V @ U

    V = np.linalg.qr(rng.standard_normal((L, w)))[0]
    for _ in range(iters):
        V = np.linalg.qr(np.column_stack([apply(V[:, j]) for j in range(w)]))[0]
    ev, X = rr(V)
    keep = ev > 1e-12 * ev.max()
    out = [(1 / np.sqrt(e), X[:, j]) for j, e in enumerate(ev) if keep[j]]
    if w2:
        Vd = np.linalg.qr(X[:, keep])[0]
        V2 = np.linalg.qr(rng.standard_normal((L, w2)))[0]
        for _ in range(iters + 2):
            V2 = V2 - Vd @ (Vd.T @ V2)
            V2 = np.column_stack([apply(V2[:, j]) for j in range(w2)])
            V2 = np.linalg.qr(V2 - Vd @ (Vd.T @ V2))[0]
        ev2, X2 = rr(V2, Vd)
        out += [(1 / np.sqrt(e), X2[:, j]) for j, e in enumerate(ev2) if e > 0]
    out.sort(key=lambda t: t[0])
    return out


def cmd_cond(argv):
    """the conditioning of form A and form S at the same state, layout, orders and plan (module docstring)"""
    from octo_corrector import load_o, o8
    from pinning import band_coeffs
    pos, o = opts(argv, dict(kfreeze="13", plan_tol="1e-32", threads="2", w="10", w2="4", iters="6", forms="A,S",
                             segs="state", ks="state"))
    for path in pos:
        z8, meta = load_o(path)
        M, nodes = int(meta["M"]), list(meta["nodes"])
        K = len(nodes)
        with mp.workdps(60):
            D = mp.mpf(meta.get("D_str", repr(meta["D"])))
        kf = None if o["kfreeze"] == "none" else int(o["kfreeze"])
        ks_c = meta.get("ks", 120) if o["ks"] == "state" else int(o["ks"])
        kw = dict(kt=meta.get("kt", 36), kc=meta.get("kc", 40), ks=ks_c, threads=int(o["threads"]), kfreeze=kf)
        sa = make_solver(D, M, nodes=nodes, **kw)
        segs = meta["segs"] if (o["segs"] == "state" and meta.get("segs")) else sa.plan(z8[0], z8[1], float(o["plan_tol"]), 0.05)
        sa.set_segs(segs)
        print(f"{os.path.basename(path)}: D {float(D):.6g}, M {M}, K {K}, kt {kw['kt']}, kc {kw['kc']}, ks {kw['ks']}, "
              f"kfreeze {kf}, {sum(len(g) - 1 for g in segs)} Taylor steps", flush=True)
        for form in o["forms"].split(","):
            t0 = time.time()
            if form == "A":
                s, z, L = sa, o8(z8), sa.size
                d, Fh, Fl, pmin, pmax = s.newton_o(list(z))
                Fv = np.asarray(Fh) + np.asarray(Fl)
            else:
                s = make_solver_s(D, M, nodes, segs=segs, **kw)
                z, L = a_to_s(list(z8), M, K), s.size_s
                d, F, pmin, pmax = s.newton_s(z)
                Fv = np.asarray(F[0]) + np.asarray(F[1])
            t_lu = time.time() - t0
            zero = np.zeros(L)

            def solve(v, tr, s=s):
                return np.asarray(s.olu_solve([np.ascontiguousarray(v)] + [zero] * 7, tr)[0])
            eD = np.zeros(L)
            eD[-1] = 1.0
            g = solve(eD, True)
            ws = weakest(solve, L, int(o["w"]), int(o["iters"]), int(o["w2"]))
            print(f"  form {form}: L {L}, octo-double LU {t_lu:.0f} s, max|F| {np.abs(Fv).max():.2e}, pivots {pmin:.3e} .. "
                  f"{pmax:.3e} (ratio {pmax / pmin:.2e}); Delta's gain sum|dDelta/dF| {np.abs(g).sum():.3e} "
                  f"(max {np.abs(g).max():.2e}); first-order dDelta {-(g @ Fv):.3e}", flush=True)
            for sig, x in ws[:int(o["w"]) + int(o["w2"])]:
                co = band_coeffs(x[:M], np.zeros(M), M, list(range(1, 32, 2)))
                amps = [abs(a) + abs(b) for a, b in co]
                k = 2 * int(np.argmax(amps)) + 1
                hz = x[2 * M + (4 if form == "A" else 3) * M * K:-1]
                print(f"    sigma {sig:.3e}  |Delta comp| {abs(x[-1]):.2e}  p1 part {np.linalg.norm(x[:M]):.2e} "
                      f"(harmonic {k})  horizon data {np.linalg.norm(hz):.2e}", flush=True)
    return 0


def junk_basis(s, L, w, iters=6, delta_max=1e-2):
    """the w weakest right singular directions of the last newton_s LU whose Delta component is below delta_max
    (weakest(), f64), orthonormal, with their sigmas and those of the weak ones kept"""
    zero = np.zeros(L)

    def solve(v, tr):
        return np.asarray(s.olu_solve([np.ascontiguousarray(v)] + [zero] * 7, tr)[0])
    ws = weakest(solve, L, w, iters, 0)
    junk = [(sg, x) for sg, x in ws if abs(x[-1]) < delta_max]
    kept = [(sg, x) for sg, x in ws if abs(x[-1]) >= delta_max]
    V = np.linalg.qr(np.column_stack([x for _, x in junk]))[0] if junk else np.zeros((L, 0))
    junk_basis.kept = kept
    return V, [sg for sg, _ in junk], [sg for sg, _ in kept]


def ls_solve(s, z8, w, outer_it, inner_it, halvings, tol_dz, log=None, verbose=True, delta_tol=1e-30):
    """A Lyapunov-Schmidt split of Newton's method on form S: inner solves with the w weakest right singular directions
    all frozen (newton_o_s, junk_w w, delta_max 2: the strong equations solved with those coordinates fixed), then one
    step along the weak directions that carry Delta (|Delta component| >= 1e-2) only: the full Newton step at the inner
    solution projected on them, the junk directions (no Delta) staying frozen. Repeated until that step moves Delta by
    less than delta_tol. Returns (z8, converged, outer records)."""
    from multidouble import od_axpy
    from octo_corrector import delta_digits, o8
    z8 = o8(z8)
    L = len(z8[0])
    zero8 = np.zeros(L)
    outs = []
    conv = False
    for outer in range(outer_it):
        t0 = time.time()
        z8, _, recs = newton_o_s(s, z8, 0.0, inner_it, halvings, log, junk_w=w, tol_dz=tol_dz, delta_max=2.0, verbose=verbose)
        d, F, pmin, pmax = s.newton_s(list(z8))
        nF = float(np.abs(np.asarray(F[0]) + np.asarray(F[1])).max())
        junk_basis(s, L, w)
        kept = junk_basis.kept
        dv = np.asarray(d[0]) + np.asarray(d[1])
        if kept:
            Vd = np.linalg.qr(np.column_stack([x for _, x in kept]))[0]
            c = Vd @ (Vd.T @ dv)
        else:
            c = np.zeros(L)
        z8 = o8(od_axpy(1.0, np.ascontiguousarray(c), *([zero8] * 7), *z8))
        F2 = s.residual_s(list(z8))
        rec = dict(outer=outer, inner_steps=len(recs), F_inner=nF, weak_kept=[float(sg) for sg, _ in kept],
                   step=float(np.abs(c).max()), dDelta=float(c[-1]), F_after=float(np.abs(np.asarray(F2[0]) + np.asarray(F2[1])).max()),
                   Delta=delta_digits(z8, 40), s=round(time.time() - t0, 1),
                   utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        outs.append(rec)
        if log:
            with open(log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        if verbose:
            print(f"  outer {outer}: inner {len(recs)} steps to |F| {nF:.2e}; weak Delta directions "
                  f"{', '.join(f'{x:.1e}' for x in rec['weak_kept'])}; step {rec['step']:.2e}, dDelta {rec['dDelta']:+.3e}, "
                  f"|F| after {rec['F_after']:.2e}; Delta {rec['Delta']} ({rec['s']} s)", flush=True)
        if abs(rec["dDelta"]) < delta_tol:
            conv = True
            break
    return z8, conv, outs


def newton_o_s(s, z8, tol, it_max, halvings, log=None, mutate="none", verbose=True, info=None, junk_w=0, tol_dz=1e-45,
               delta_max=1e-2):
    """Newton in octo-double on form S from the eight-part z8: newton_s's step, halved while max |F| does not fall;
    one record per step to log (jsonl); returns (z8, converged, records). junk_w > 0: the step's components along the
    weakest junk_w right singular directions with no Delta content (junk_basis, formed at each LU) are removed, so
    those coordinates stay where they are; a trial is then accepted when its max |F| or its own projected correction
    falls, and the iteration ends converged when the projected correction is below tol_dz (the residual left is the
    frozen directions')."""
    from multidouble import od_axpy
    from octo_corrector import delta_digits, o8
    z8 = o8(z8)
    L = len(z8[0])
    F = s.residual_s(list(z8))
    nF = float(np.abs(np.asarray(F[0]) + np.asarray(F[1])).max())
    recs = []
    if verbose:
        print(f"  start: max |F| {nF:.3e}, Delta {delta_digits(z8, 40)}", flush=True)
    conv = nF <= tol
    zero8 = np.zeros(L)

    def proj(d, V):
        if V is None or V.shape[1] == 0:
            return d
        c = V @ (V.T @ (np.asarray(d[0]) + np.asarray(d[1])))
        return o8(od_axpy(-1.0, np.ascontiguousarray(c), *([zero8] * 7), *o8(d)))
    st = None
    for it in range(it_max):
        if conv:
            break
        t0 = time.time()
        if st is None:
            d, _, pmin, pmax = s.newton_s(list(z8))
            V, sj, sk = junk_basis(s, L, junk_w, delta_max=delta_max) if junk_w else (None, [], [])
            if junk_w and verbose:
                print(f"    {V.shape[1]} junk directions frozen (sigma {min(sj) if sj else 0:.1e}..{max(sj) if sj else 0:.1e}), "
                      f"weak ones with Delta content kept: {', '.join(f'{x:.1e}' for x in sk)}", flush=True)
            d = proj(o8(d), V)
        else:
            d, pmin, pmax, V = st
        st = None
        t_lu = time.time() - t0
        sign = -1.0 if mutate == "step" else 1.0
        dz = float(np.abs(np.asarray(d[0])).max())
        dD = float(np.asarray(d[0])[-1])
        if junk_w and dz < tol_dz:
            conv = True
            recs.append(dict(it=it, F=nF, dz=dz, junk_floor=True))
            if verbose:
                print(f"  it {it}: projected correction {dz:.2e} below {tol_dz:.0e}: converged at the frozen directions' "
                      f"residual |F| {nF:.2e}", flush=True)
            break
        lam, acc, trials = 1.0, False, []
        for _ in range(halvings + 1):
            zt = o8(od_axpy(sign * lam, *d, *z8))
            Ft = s.residual_s(list(zt))
            nt = float(np.abs(np.asarray(Ft[0]) + np.asarray(Ft[1])).max())
            tr = dict(lam=lam, F=nt)
            if np.isfinite(nt) and nt < nF:
                acc = True
            elif junk_w and np.isfinite(nt):
                # the frozen directions' residual need not fall: judge the trial by its own projected correction
                dt, _, pt0, pt1 = s.newton_s(list(zt))
                Vt, _, _ = junk_basis(s, L, junk_w, delta_max=delta_max)
                dtp = proj(o8(dt), Vt)
                dzt = float(np.abs(np.asarray(dtp[0])).max())
                tr["dz_own"] = dzt
                if dzt < dz:
                    acc = True
                    st = (dtp, pt0, pt1, Vt)
            trials.append(tr)
            if acc:
                break
            lam *= 0.5
        rec = dict(it=it, F=nF, dz=dz, dDelta=dD, pmin=pmin, pmax=pmax, lu_s=round(t_lu, 1), trials=trials, accepted=acc,
                   s=round(time.time() - t0, 1))
        if acc:
            z8, nF = zt, nt
            conv = nF <= tol
        rec.update(F_new=nF, Delta=delta_digits(z8, 40), utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        recs.append(rec)
        if log:
            with open(log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        if verbose:
            print(f"  it {it}: |F| {rec['F']:.3e} -> {nF:.3e}, |dz| {dz:.2e}, dDelta {dD:+.3e}, lam {lam if acc else 0:g}, "
                  f"pivots {pmin:.1e}..{pmax:.1e}, {rec['s']} s (LU {t_lu:.0f} s), Delta {rec['Delta']}", flush=True)
        if not acc:
            break
    return z8, conv, recs


def cmd_jump(argv):
    import glob
    import jump
    from octo_corrector import delta_digits, load_o, o8, save_o
    from raw_out import pop_allow, require_raw
    argv = list(argv)
    allow = pop_allow(argv)
    pos, o = opts(argv, dict(D="250", states="", Dmin="0", Dmax="1e9", deg="3", kfreeze="13", plan_tol="1e-40", hmax="0.05",
                             ks="sigma", kc="146", threads="2", delta="series", tol="1e-40", it_max="16", halvings="8",
                             mutate="none", junk="0", tol_dz="1e-45", ls="0", outer="8", held="0"))
    outdir = pos[0]
    os.makedirs(outdir, exist_ok=True)
    log = os.path.join(outdir, "jump_s.jsonl")
    require_raw(log, allow=allow)
    paths = sorted(set(p for g in o["states"].split(",") for p in glob.glob(g)))
    rows = []
    for p in paths:
        z, m = load_o(p)
        if float(o["Dmin"]) <= m["D"] <= float(o["Dmax"]):
            rows.append((m["D"], z, m))
    rows.sort(key=lambda r: r[0])
    m0 = rows[-1][2]
    M, nodes = int(m0["M"]), list(m0["nodes"])
    K = len(nodes)
    Ds = np.array([r[0] for r in rows])
    Y = np.array([jump.scale(sum(np.asarray(p) for p in r[1][:2]), r[0], M, K) for r in rows])
    with mp.workdps(60):
        Dt = mp.mpf(o["D"])
    deg = int(o["deg"])
    t = 1.0 / (Ds - 2.0)
    tt = 1.0 / (float(Dt) - 2.0)
    c = np.polyfit(t, Y, deg)
    y = np.array([np.polyval(c[:, j], tt) for j in range(Y.shape[1])])
    c1 = np.polyfit(t, Y, max(deg - 1, 0))
    y1 = np.array([np.polyval(c1[:, j], tt) for j in range(Y.shape[1])])
    zA = jump.unscale(y, float(Dt), M, K)
    spread = np.abs(jump.unscale(y1, float(Dt), M, K) - zA)
    print(f"guess at D {o['D']} from {len(rows)} states, D {Ds[0]:.4f} .. {Ds[-1]:.4f}, degree {deg} in 1/n; degree "
          f"{deg - 1} differs by {spread[:-1].max():.2e} (unknowns), {spread[-1]:.2e} (Delta)", flush=True)
    if o["ks"] == "auto":
        ah = zA[jump.groups(M, K)["ah"]]
        ks, ks_s, ks_m = jump.ks_rule(float(Dt), gbar=(float(Dt) - 3.0) * float(np.mean(ah * ah - 1.0)))
        print(f"  ks {ks} (sigma rule {ks_s}, form A's mass-mode tail rule {ks_m})", flush=True)
    elif o["ks"] == "sigma":
        ah = zA[jump.groups(M, K)["ah"]]
        ks = jump.ks_rule(float(Dt), gbar=(float(Dt) - 3.0) * float(np.mean(ah * ah - 1.0)))[1]
        print(f"  ks {ks} (the sigma rule)", flush=True)
    else:
        ks = int(o["ks"])
    kf = None if o["kfreeze"] == "none" else int(o["kfreeze"])
    kw = dict(kt=int(m0.get("kt", 36)), kc=int(o["kc"]), ks=ks, threads=int(o["threads"]), kfreeze=kf)
    sa = make_solver(Dt, M, nodes=nodes, **kw)
    zero = np.zeros(len(zA))
    segs = sa.plan(np.ascontiguousarray(zA), zero, float(o["plan_tol"]), float(o["hmax"]))
    segs = [list(map(float, g)) for g in segs]
    s = make_solver_s(Dt, M, nodes, segs=segs, **kw)
    z8 = [a_to_s([zA], M, K)[0]] + [np.zeros(s.size_s) for _ in range(7)]
    if o["delta"] != "fit":
        # Track C's series row of delta_all.csv, or a value given as a number (Track C's at a D without a row)
        sd = jump.series_delta(o["D"]) if o["delta"] == "series" else (mp.mpf(o["delta"]), 0.0)
        if sd is None:
            raise SystemExit(f"no series row at D {o['D']} in delta_all.csv")
        with mp.workdps(60):
            v = mp.mpf(sd[0])
            for q in range(8):
                f = float(v)
                z8[q][-1] = f
                v -= mp.mpf(f)
        print(f"  Delta from Track C's series: {mp.nstr(sd[0], 16)} +- {sd[1]:.2e}; the fit's {zA[-1]:.12f}", flush=True)
    meta = dict(D=float(Dt), D_str=mp.nstr(Dt, 50), M=M, nodes=nodes, kt=kw["kt"], kc=kw["kc"], ks=ks, kfreeze=kf, form="S",
                parts=8, segs=segs, plan_tol=float(o["plan_tol"]), deg=deg, states=[r[2].get("source", "") for r in rows][:0],
                source="form_s.py jump")
    F = s.residual_s(list(o8(z8)))
    print(f"  {sum(len(g) - 1 for g in segs)} Taylor steps; the guess's max |F| per block "
          + " ".join(f"{b:.1e}" for b in blocks_s(F, M, K)), flush=True)
    with open(log, "a") as fh:
        fh.write(json.dumps(dict(start="guess", D_str=meta["D_str"], ks=ks, kc=kw["kc"], kfreeze=kf, deg=deg,
                                 states=len(rows), D_range=[float(Ds[0]), float(Ds[-1])], tol=float(o["tol"]),
                                 mutate=o["mutate"], threads=int(o["threads"]),
                                 utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))) + "\n")
    s_full = s
    if int(o["held"]):
        # p1's harmonics 5-13 held at the guess's own values (form S's PinnedO), the rest solved
        sh = make_solver_s(Dt, M, nodes, segs=segs, **dict(kw, kfreeze=3))
        s = HeldS(sh, M, K, held_field(z8, M))
        meta.update(held="5-13 at the guess's values", kfreeze=3)
        print("  p1's harmonics 5-13 held at the guess's values (kfreeze 3 with the held offset)", flush=True)
    if int(o["ls"]):
        z8, conv, recs = ls_solve(s, z8, int(o["junk"]), int(o["outer"]), int(o["it_max"]), int(o["halvings"]),
                                  float(o["tol_dz"]), log)
    else:
        z8, conv, recs = newton_o_s(s, z8, float(o["tol"]), int(o["it_max"]), int(o["halvings"]), log, o["mutate"],
                                    junk_w=int(o["junk"]), tol_dz=float(o["tol_dz"]))
    F = s.residual_s(list(z8))
    meta.update(solved=bool(conv), F=float(np.abs(np.asarray(F[0]) + np.asarray(F[1])).max()))
    if int(o["held"]):
        # the full system's (kfreeze 13) first-order Delta correction at the held solution: -e_Delta^T J^-1 F
        Ff = s_full.residual_s(list(z8))
        s_full.newton_s(list(z8))
        L = s_full.size_s
        eD = np.zeros(L)
        eD[-1] = 1.0
        g = np.asarray(s_full.olu_solve([eD] + [np.zeros(L)] * 7, True)[0])
        Fv = np.asarray(Ff[0]) + np.asarray(Ff[1])
        dD1 = float(-(g @ Fv))
        meta.update(full_F=float(np.abs(Fv).max()), full_dDelta_first_order=dD1)
        print(f"  the full system (kfreeze 13) at the held solution: max |F| {np.abs(Fv).max():.3e}, first-order "
              f"dDelta {dD1:+.3e} (Delta's gain sum |dDelta/dF| {np.abs(g).sum():.2e})", flush=True)
        with open(log, "a") as fh:
            fh.write(json.dumps(dict(full_F=float(np.abs(Fv).max()), full_dDelta_first_order=dD1,
                                     gain=float(np.abs(g).sum()))) + "\n")
    out = os.path.join(outdir, f"jump_s_D{float(Dt):.4f}.npz")
    save_o(out, z8, meta)
    Dl = delta_digits(z8, 40)
    print(f"  {'converged' if conv else 'NOT converged'}; Delta {Dl}; saved {out}", flush=True)
    sd = jump.series_delta(o["D"])
    if sd is not None:
        with mp.workdps(60):
            print(f"  Delta - Track C {mp.nstr(mp.mpf(Dl) - sd[0], 6)} (Track C's +- {sd[1]:.2e})", flush=True)
    return 0 if conv else 1


def held_field(z8, M, ks=(5, 7, 9, 11, 13)):
    """the held field: p1's harmonics ks of the eight-part z8, from all its parts at 160 digits, as eight parts on the
    half grid (pinning.band_coeffs, band_field)"""
    from pinning import band_coeffs, band_field
    co = band_coeffs(np.asarray(z8[0])[:M], np.asarray(z8[1])[:M], M, list(ks), exact=True,
                     rest=[np.asarray(p)[:M] for p in z8[2:]])
    return band_field(co, list(ks), M, parts=8)


class HeldS:
    """form S with p1's harmonics 5-13 held (pinning.PinnedO's offset on the form-S methods): the solver has kfreeze 3,
    whose horizon block's p rows read P_low (mismatch) + P_high p1; the offset c (the held field h in those rows, eight
    parts) makes them P_low (mismatch) + P_high p1 - h, so p1's harmonics 5-13 are held at h and 15 and up at zero.
    F - c in the residual, dz + J^-1 c in the step (the LU newton_s kept)."""

    def __init__(self, s, M, K, h8):
        from octo_corrector import o8
        self.s = s
        L = s.size_s
        r0 = K * 3 * M
        c8 = [np.zeros(L) for _ in range(8)]
        for q in range(8):
            c8[q][r0:r0 + M] = np.asarray(h8[q])
        self.c8 = o8(c8)

    @property
    def size_s(self):
        return self.s.size_s

    def olu_solve(self, b, transpose=False):
        return self.s.olu_solve(b, transpose)

    def _minus_c(self, F):
        from multidouble import od_axpy
        from octo_corrector import o8
        if len(F) == 8:
            return list(o8(od_axpy(-1.0, *self.c8, *o8(F))))
        return list(dd_add(np.asarray(F[0]), np.asarray(F[1]), -self.c8[0], -self.c8[1]))

    def residual_s(self, z, mutate=0):
        return self._minus_c(self.s.residual_s(z, mutate))

    def newton_s(self, z, mutate=0):
        from multidouble import od_axpy
        from octo_corrector import o8
        d, F, pmin, pmax = self.s.newton_s(z, mutate)
        if len(z) == 8:
            x = o8(self.s.olu_solve(list(self.c8), False))
            d = list(o8(od_axpy(1.0, *x, *o8(d))))
        else:
            raise ValueError("HeldS: the held step needs eight parts (the octo-double LU)")
        return d, self._minus_c(F), pmin, pmax


def main():
    cmd = sys.argv[1]
    return dict(check=cmd_check, cond=cmd_cond, jump=cmd_jump, planted=cmd_planted)[cmd](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
