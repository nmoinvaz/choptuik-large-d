"""NOT VALID FOR DELTA (#1088): the fine mesh does not impose the selection. The SSH's non-analytic mode decays by about
e^-2 per segment toward 0.95, so the series there cannot exclude it; its amplitude is a near-null direction (Delta's
gain 1e19) and Delta drifts with the guess's error along it (+6.7e-9 at D 2200, +4.6e-8 at D 2828). Its records carry
valid false. The tools (path_s, the sparse Jacobian) remain useful.

Form S on a fine mesh toward the horizon (#1088): nodes from x 0.7 outward with 1 - x geometric, ratio 1 - c/n (the
SSH's (1 - x)^(n/2) mode changes by about e^(c/2) per segment), to x_end (0.95) where the Moebius horizon series converges
in about 100 terms; the Jacobian assembled block by block (Solver.jacobian_sub_s) in sparse storage and factored by
SuperLU, so the cost is linear in the node count. At x_l = 0.7 the series' sensitivity to m_h grows like 10^(0.0087 n)
and jumps fail past D 2200 (raw I1088-dh610/j2420_l2); summed at 0.95 its gain from m_h is 0.05.

    python3 form_s_fine.py jump OUTDIR --D 2828 --states_s S1,S2,S3 [--deg 2 --mob 8 --x_end 0.95 --c 4 --ks 150]

The guess: each source state's solution carried from its node 0.7 outward along the target's mesh (Solver.path_s, the
(1 - x)^(n/2) mode decays outward, so the path is stable), then every unknown fitted in 1/n as form_s.py jump does.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spl  # noqa: E402

from form_s import (HeldS, a_to_s, blocks_s, direct_line, ham_check, held_field, make_solver_s, opts, s_to_a,  # noqa: E402
                    scale_s, unscale_s)
from hpkernel.driver import make_solver  # noqa: E402

M = 32


def mesh(D, x0=0.7, x_end=0.95, c=4.0):
    """the outer nodes after x0: 1 - x_(j+1) = (1 - x_j)(1 - c/n), the first at or past x_end the last"""
    n = float(D) - 2.0
    r = 1.0 - c / n
    xs, u = [], 1.0 - x0
    while 1.0 - u < x_end:
        u *= r
        xs.append(1.0 - u)
    return xs


def substeps(xa, xs, sub):
    """Taylor points from xa through xs, sub steps per interval; the indices of xs's points among them"""
    pts, rec = [xa], []
    a = xa
    for b in xs:
        for i in range(1, sub + 1):
            pts.append(a + (b - a) * i / sub)
        rec.append(len(pts) - 1)
        a = b
    return pts, rec


def inner_solver(D, meta, threads, mob):
    """a solver on a state's first five nodes (0.3 ... 0.7), for path_s from node 0.7"""
    nodes = list(meta["nodes"])[:5]
    segs = list(meta["segs"])[:4]
    return make_solver_s(D, M, nodes, kt=int(meta.get("kt", 36)), kc=int(meta["kc"]), ks=50, threads=threads,
                         segs=segs, mob=mob or 8.0)


def inner_z(z8, meta):
    """a state's unknowns on its first five nodes (p1, c0, nodes 0-4, m_h, Delta), eight parts"""
    K = len(meta["nodes"])
    head = 2 * M + 3 * M * 5
    out = []
    for q in range(8):
        v = np.asarray(z8[q])
        out.append(np.ascontiguousarray(np.concatenate([v[:head], v[2 * M + 3 * M * K:]])))
    return out


def outer_values(D, z8, meta, xs, sub, threads):
    """a state's solution at the points xs (past 0.7) by Taylor steps from its node 0.7: (len(xs), 3M), float64"""
    s = inner_solver(D, meta, threads, float(meta.get("hmob", 0.0) or 0.0))
    pts, rec = substeps(0.7, xs, sub)
    zi = inner_z(z8, meta)
    P = s.path_s([np.ascontiguousarray(p) for p in zi[:4]], pts, rec)
    return np.asarray(P[0]) + np.asarray(P[1])


def sparse_jacobian(s, z, K, L):
    """form S's Jacobian (the solver s, projected rows included) in CSC from its blocks' sub-Jacobians at z's
    double-double parts, the leading part as float64"""
    rows_i, cols_i, vals = [], [], []
    zz = [np.ascontiguousarray(z[0]), np.ascontiguousarray(z[1])]
    gcol = L - 1
    p1 = list(range(0, M))
    c0 = list(range(M, 2 * M))
    mh = list(range(2 * M + 3 * M * K, 2 * M + 3 * M * K + M))

    def node(i):
        o = 2 * M + 3 * M * i
        return list(range(o, o + 3 * M))
    for b in range(K + 1):
        r = list(range(b * 3 * M, (b + 1) * 3 * M))
        if b == 0:
            c = p1 + c0 + node(0) + [gcol]
        elif b < K:
            c = node(b - 1) + node(b) + [gcol]
        else:
            c = p1 + mh + node(K - 1) + [gcol]
        J = np.asarray(s.jacobian_sub_s(zz, r, c)[0])
        ii, jj = np.nonzero(J)
        rows_i.append(np.asarray(r)[ii])
        cols_i.append(np.asarray(c)[jj])
        vals.append(J[ii, jj])
    c = p1 + [gcol]
    J = np.asarray(s.jacobian_sub_s(zz, [L - 1], c)[0])
    ii, jj = np.nonzero(J)
    rows_i.append(np.full(len(ii), L - 1))
    cols_i.append(np.asarray(c)[jj])
    vals.append(J[ii, jj])
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(cols_i))), shape=(L, L))


def newton_sparse(sh, s, z8, K, tol_delta, it_max, halvings, log=None, jscale=1.0, verbose=True, res="dd"):
    """held Newton on the fine mesh: the residual of sh in double-double, the Jacobian of s (sparse_jacobian) in float64
    factored by SuperLU, the step halved while max |F| does not fall; stopped when Delta's first-order bound gain |F|
    (gain = sum |J^-T e_Delta|) is at most tol_delta"""
    from multidouble import od_axpy
    from octo_corrector import delta_digits, o8
    z8 = o8(z8)
    L = len(z8[0])
    zero = np.zeros(L)

    npart = 8 if res == "od" else 2

    def resid(z):
        F = sh.residual_s([np.ascontiguousarray(z[q]) for q in range(npart)])
        return np.asarray(F[0]) + np.asarray(F[1])
    F = resid(z8)
    nF = float(np.abs(F).max())
    recs, conv, gain = [], False, float("nan")
    if verbose:
        print(f"  start: max |F| {nF:.3e}, Delta {delta_digits(z8, 40)}", flush=True)
    for it in range(it_max):
        t0 = time.time()
        J = sparse_jacobian(s, z8, K, L)
        tj = time.time() - t0
        t1 = time.time()
        lu = spl.splu((J * jscale).tocsc(), permc_spec="COLAMD")
        dz = -lu.solve(F)
        eD = np.zeros(L)
        eD[-1] = 1.0
        g = spl.splu(J.tocsc(), permc_spec="COLAMD").solve(eD, trans="T") if jscale != 1.0 else lu.solve(eD, trans="T")
        gain = float(np.abs(g).sum())
        tl = time.time() - t1
        cond = None
        if it == 0:
            # a 1-norm condition estimate of J (Hager-Higham, scipy's onenormest of J^-1 with the LU)
            Ji = spl.LinearOperator((L, L), matvec=lu.solve, rmatvec=lambda x: lu.solve(x, trans="T"), dtype=float)
            cond = float(spl.onenormest(J) * spl.onenormest(Ji))
        lam, acc, trials = 1.0, False, []
        for _ in range(halvings + 1):
            zt = o8(od_axpy(lam, np.ascontiguousarray(dz), *([zero] * 7), *z8))
            try:
                Ft = resid(zt)
                nt = float(np.abs(Ft).max())
            except BaseException as e:  # a trial whose slaving or horizon LU fails
                if isinstance(e, KeyboardInterrupt):
                    raise
                Ft, nt = None, float("inf")
            trials.append(dict(lam=lam, F=nt))
            if np.isfinite(nt) and nt < nF:
                acc = True
                break
            lam *= 0.5
        rec = dict(it=it, F=nF, dz=float(np.abs(dz).max()), dDelta=float(dz[-1]), gain=gain, nnz=int(J.nnz), cond1=cond,
                   jac_s=round(tj, 1), lu_s=round(tl, 1), trials=trials, accepted=acc, s=round(time.time() - t0, 1))
        if acc:
            z8, F, nF = zt, Ft, nt
        rec.update(F_new=nF, bound=gain * nF, Delta=delta_digits(z8, 40),
                   utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        recs.append(rec)
        if log:
            with open(log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        if verbose:
            print(f"  it {it}: |F| {rec['F']:.3e} -> {nF:.3e}, |dz| {rec['dz']:.2e}, dDelta {rec['dDelta']:+.3e}, "
                  f"lam {lam if acc else 0:g}, Delta's gain {gain:.2e}, gain |F| {gain * nF:.2e}, nnz {J.nnz}, "
                  + (f"cond_1 {cond:.2e}, " if cond else "")
                  + f"{rec['s']} s (J {tj:.0f} s, LU {tl:.0f} s), Delta {rec['Delta']}", flush=True)
        if not acc:
            break
        if gain * nF <= tol_delta:
            conv = True
            break
    return z8, conv, recs


def cmd_jump(argv):
    import glob
    import jump
    from octo_corrector import delta_digits, load_o, o8, save_o
    from raw_out import pop_allow, require_raw
    argv = list(argv)
    allow = pop_allow(argv)
    pos, o = opts(argv, dict(D="2828", states_s="", deg="2", mob="8", x_end="0.95", c="4", ks="150", sub="4",
                             kc="250", threads="2", plan_tol="1e-40", hmax="0.05", delta="fit", tc="", it_max="10",
                             halvings="8", delta_tol="1e-15", jscale="1", commit="", ham="1", res="od", held_from=""))
    outdir = pos[0]
    os.makedirs(outdir, exist_ok=True)
    log = os.path.join(outdir, "fine.jsonl")
    require_raw(log, allow=allow)
    t_all = time.time()
    with mp.workdps(60):
        Dt = mp.mpf(o["D"])
    n = float(Dt) - 2.0
    thr = int(o["threads"])
    xo = mesh(float(Dt), 0.7, float(o["x_end"]), float(o["c"]))
    sub = int(o["sub"])
    print(f"D {o['D']}: {len(xo)} nodes past 0.7 (1 - x times 1 - {o['c']}/n, to {xo[-1]:.6f}); Moebius a {o['mob']}, "
          f"ks {o['ks']}", flush=True)
    rows = []
    for p in sorted(set(q for g in o["states_s"].split(",") if g for q in glob.glob(g))):
        z, m = load_o(p)
        t0 = time.time()
        with mp.workdps(60):
            Dm = mp.mpf(m["D_str"])
        ov = outer_values(Dm, z, m, xo, sub, thr)
        zi = inner_z(z, m)
        rows.append((float(m["D"]), np.asarray(zi[0]) + np.asarray(zi[1]), ov, m))
        print(f"  source D {m['D']:g}: carried along the mesh, max |p| at the last node {np.abs(ov[-1, :M]).max():.4e} "
              f"({time.time() - t0:.0f} s)", flush=True)
    rows.sort(key=lambda r: r[0])
    m0 = rows[-1][3]
    nodes_in = list(m0["nodes"])[:5]
    K = 5 + len(xo)
    nodes = nodes_in + xo
    Ds = np.array([r[0] for r in rows])
    tt = 1.0 / (float(Dt) - 2.0)
    deg = int(o["deg"])

    def fit(Y):
        cfs = np.polyfit(1.0 / (Ds - 2.0), Y, deg)
        return np.array([np.polyval(cfs[:, j], tt) for j in range(Y.shape[1])])
    Yi = np.array([scale_s(r[1], r[0], M, 5) for r in rows])
    yi = unscale_s(fit(Yi), float(Dt), M, 5)

    def sc_outer(v, D):
        nn = D - 2.0
        w = v.copy()
        w[:, :2 * M] *= np.sqrt(nn)
        w[:, 2 * M:] = (w[:, 2 * M:] - 1.0) * nn
        return w.ravel()
    Yo = np.array([sc_outer(r[2], r[0]) for r in rows])
    yo = fit(Yo).reshape(len(xo), 3 * M)
    yo[:, :2 * M] /= np.sqrt(n)
    yo[:, 2 * M:] = 1.0 + yo[:, 2 * M:] / n
    head = 2 * M + 3 * M * 5
    zs = np.concatenate([yi[:head], yo.ravel(), yi[head:head + M], yi[-1:]])
    L = len(zs)
    z8 = [zs] + [np.zeros(L) for _ in range(7)]
    if o["delta"] != "fit":
        sd = jump.series_delta(o["D"]) if o["delta"] == "series" else (mp.mpf(o["delta"]), 0.0)
        with mp.workdps(60):
            v = mp.mpf(sd[0])
            for q in range(8):
                f = float(v)
                z8[q][-1] = f
                v -= mp.mpf(f)
    z8 = o8(z8)
    zA = s_to_a(zs, float(Dt), M, K, nodes)
    sa = make_solver(Dt, M, nodes=nodes, kt=int(m0.get("kt", 36)), kc=int(o["kc"]), ks=int(o["ks"]), threads=thr)
    t0 = time.time()
    segs = [list(map(float, g)) for g in sa.plan(np.ascontiguousarray(zA), np.zeros(len(zA)), float(o["plan_tol"]),
                                                    float(o["hmax"]))]
    print(f"  {K} nodes, {sum(len(g) - 1 for g in segs)} Taylor steps (plan {time.time() - t0:.0f} s), {L} unknowns",
          flush=True)
    kw = dict(kt=int(m0.get("kt", 36)), kc=int(o["kc"]), ks=int(o["ks"]), threads=thr, segs=segs, mob=float(o["mob"]))
    s = make_solver_s(Dt, M, nodes, kfreeze=3, **kw)
    # p1's harmonics 5-13 held at the guess's values, or at another state's (a control on the same held system)
    h8 = held_field(load_o(o["held_from"])[0], M) if o["held_from"] else held_field(z8, M)
    sh = HeldS(s, M, K, h8)
    F = sh.residual_s([np.ascontiguousarray(z8[0]), np.ascontiguousarray(z8[1])])
    b = blocks_s(F, M, K)
    Fv = np.abs(np.asarray(F[0]) + np.asarray(F[1]))
    seg_max = max(b[1:K])
    print(f"  the guess: centre {b[0]:.1e}, segments up to {seg_max:.1e} (the first past 0.7: {b[5]:.1e}), horizon "
          f"{b[K]:.1e}, phase {b[K + 1]:.1e}: the basin test, horizon against the largest other block "
          f"{b[K] / max(b[0], seg_max):.2e}", flush=True)
    meta = dict(D=float(Dt), D_str=mp.nstr(Dt, 50), M=M, nodes=nodes, kt=kw["kt"], kc=kw["kc"], ks=kw["ks"], kfreeze=3,
                form="S", hmob=float(o["mob"]), parts=8, segs=segs, plan_tol=float(o["plan_tol"]), deg=deg,
                held=f"5-13 at {os.path.basename(o['held_from'])}'s values" if o["held_from"] else "5-13 at the guess's values",
                source="form_s_fine.py jump", mesh=dict(x0=0.7, x_end=float(o["x_end"]),
                                                                                         c=float(o["c"]), sub=sub))
    with open(log, "a") as fh:
        fh.write(json.dumps(dict(start="guess", D_str=meta["D_str"], K=K, L=L, blocks_max=[float(x) for x in b],
                                 states=[r[0] for r in rows], utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))) + "\n")
    z8, conv, recs = newton_sparse(sh, s, z8, K, float(o["delta_tol"]), int(o["it_max"]), int(o["halvings"]), log,
                                   jscale=float(o["jscale"]), res=o["res"])
    F4 = sh.residual_s([np.ascontiguousarray(z8[q]) for q in range(4)])
    Fc = float(np.abs(np.asarray(F4[0]) + np.asarray(F4[1])).max())
    gain = recs[-1]["gain"] if recs else float("nan")
    meta.update(solved=bool(conv), F=Fc, check="qd", delta_gain=gain, delta_bound_first_order=gain * Fc)
    print(f"  end state: quad-double max |F| {Fc:.3e}; Delta's first-order bound {gain * Fc:.2e} (gain {gain:.2e})",
          flush=True)
    hc = None
    if int(o["ham"]):
        hc = ham_check(z8, meta, threads=thr)
        print("  Hamiltonian code check: segment blocks' a rows up to "
              f"{max(r['a'] for r in hc.values()):.1e}, p/m/c up to {max(max(r['p'], r['m'], r['c']) for r in hc.values()):.1e}",
              flush=True)
        meta.update(ham_check={str(k): r for k, r in hc.items()})
    out = os.path.join(outdir, f"fine_D{float(Dt):.4f}.npz")
    save_o(out, z8, meta)
    Dl = delta_digits(z8, 40)
    print(f"  {'converged' if conv else 'NOT converged'}; Delta {Dl}; saved {out} ({time.time() - t_all:.0f} s in all)",
          flush=True)
    tc = None
    if o["tc"]:
        tc = dict(value=o["tc"].split(":")[0], err=float(o["tc"].split(":")[1]), source="given")
        with mp.workdps(60):
            print(f"  Delta - Track C {mp.nstr(mp.mpf(Dl) - mp.mpf(tc['value']), 6)} (+- {tc['err']:.2e})", flush=True)
    if conv:
        rec = direct_line(z8, meta, out, "held", recs[-1]["F_new"], o["res"], Fc, "qd", bound=gain * Fc, ham=meta.get("ham_check"),
                          tc=tc, commit=o["commit"], extra=dict(jac="f64 sparse (SuperLU)", newton_steps=len(recs),
                                                                mesh=meta["mesh"], fine=True, method="fine", valid=False,
                                                                invalid_reason="the fine mesh does not impose the "
                                                                "selection (#1088)"))
        with open(os.path.join(outdir, "direct_s.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return 0 if conv else 1


def main():
    cmd = sys.argv[1]
    return dict(jump=cmd_jump)[cmd](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
