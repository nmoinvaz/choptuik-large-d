#!/usr/bin/env python3
"""Bit-identity of the evaluations that hp's drivers no longer repeat: each reuse against the evaluation it replaces,
on a stored eight-part state in its chain's own pinned system (octo_corrector.truth_system), every part of every
number compared by its bits.

    cd hp && python3 dup_eval_check.py RAW STATE.npz [--threads 9] [--sites base,pinned,near_null,reseed] [--N 4]
        [--its 2] [--mutate base|pinned|near_null|reseed]

base (anm_run.py base_lu_o): the corrector's last newton_o at z on the solver, as newton_od leaves it, then base_lu_o
must keep that LU, and the series z_1..z_N from it (coefficients_o) must equal the series after a newton_o at z on a
fresh solver, with the same pivots. The key must refuse z one ulp off, mutate 10, other tangents, and the solver after
set_segs.
pinned (octo_corrector.PinnedO): newton_o with its kernel newton_o's residual (kept_residual) against the residual
formed again (residual_o), at z: the outputs, and the residual in eight parts; then newton_od (the natural test) for
--its steps from z moved by 1e-12 relative (plant_start), every newton_o and residual_o call's input and output on
both paths. The key must refuse z one ulp off.
near_null (near_null.py): newton_o(keep_jacobian=True) against newton_o, the kept Jacobian (jacobian_o_kept) against
jacobian_o, newton_o_residual against residual_o, all in eight parts, and near_null.subspace's vectors and sigma
estimates (k 4, 6 iterations) through each solver's LU. The key must refuse z one ulp off, and a newton_o that did
not keep its Jacobian.
reseed (reseed.py jac_block): on the Reseeder's u0 rung solver at the state's own two-node point, Solver.jacobian_sub
against the full Jacobian's block (jacobian, jacobian_q, jacobian_o with 1 and 2 tangents) for the rung rows by the
horizon-data columns, and every row by columns holding the targets' -1, the phase row's weights, Delta and a repeat;
then the rung solves invert (double-double), polish_q (quad-double, 3 steps) and refine_o (2 octo-double corrections)
from the state's horizon data moved by 1e-20 relative, with jac_block against jac_block_full, every result compared.

--mutate X, the control of site X, a one-ulp change of the reused quantity, which must make that site fail (exit 1):
base, the kept LU taken at z with Delta's leading part one ulp up and used without the key; pinned, one ulp of the
kept residual's first entry, leading part; near_null, one ulp of one entry of the kept Jacobian; reseed, one ulp of
the first entry of every block jacobian_sub returns. The script exits 1 when any site fails, 2 when the hpkernel
lacks the methods (a build before this change), and prints each site's times on both paths.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

NEW = ("olu_pivots_at", "newton_o_residual", "jacobian_o_kept", "jacobian_sub")


def parse(argv):
    opt = dict(threads=9, sites="base,pinned,near_null,reseed", N=4, its=2, mutate="")
    args = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            k, v = (a[2:].split("=", 1) + [None])[:2]
            if v is None:
                i += 1
                v = argv[i]
            opt[k] = type(opt[k])(v) if k in opt else v
        else:
            args.append(a)
        i += 1
    return args, opt


def bits(x):
    return np.ascontiguousarray(np.asarray(x, dtype=float)).view(np.uint64)


def ndiff(a, b):
    """the number of floats whose bits differ between two results (arrays, numbers, None, nested in tuples, lists and
    dicts); a mismatch of structure or shape counts one"""
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or a.keys() != b.keys():
            return 1
        return sum(ndiff(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))) or len(a) != len(b):
            return 1
        return sum(ndiff(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return int(not (a is None and b is None))
    if isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return int(a != b)
    xa, xb = bits(a), bits(b)
    if xa.shape != xb.shape:
        return 1
    return int(np.count_nonzero(xa != xb))


def timeless(h):
    """newton_od's history without its timings ('seconds', which differ between runs)"""
    if isinstance(h, dict):
        return {k: timeless(v) for k, v in h.items() if k != "seconds"}
    if isinstance(h, (list, tuple)):
        return [timeless(v) for v in h]
    return h


def ulp_up(a, i, part=0):
    """a copy of the part arrays a with entry i of the given part one ulp up"""
    out = [np.array(p, dtype=float) for p in a]
    out[part][i] = np.nextafter(out[part][i], np.inf)
    return out


class Report:
    def __init__(self):
        self.fails = []

    def check(self, site, what, ok, detail=""):
        print(f"  {site}: {what}: {'ok' if ok else 'FAIL'}{' (' + detail + ')' if detail else ''}", flush=True)
        if not ok:
            self.fails.append(f"{site}: {what}")

    def same(self, site, what, a, b):
        n = ndiff(a, b)
        self.check(site, what, n == 0, f"{n} floats differ" if n else "bit for bit")


def system(raw, path, threads):
    """the state in eight parts, its meta and its chain's pinned system (s, PinnedSolver, PinnedQ, PinnedO)"""
    from octo_corrector import o8_any, truth_system
    z, meta, sysm, _ = truth_system(raw, path, threads=threads)
    return [np.ascontiguousarray(p) for p in o8_any(z)], meta, sysm


def site_base(raw, path, opt, rep):
    """anm_run.py's series base: the corrector's kept LU against a newton_o at z"""
    from anm_run import base_lu_o, coefficients_o
    z, meta, (s0, _, _, _) = system(raw, path, opt["threads"])
    t0 = time.time()
    st0 = s0.newton_o(z)
    Z0, _ = coefficients_o(s0, z, opt["N"])
    t_old = time.time() - t0
    _, _, (s1, _, _, _) = system(raw, path, opt["threads"])
    zc = ulp_up(z, len(z[0]) - 1) if opt["mutate"] == "base" else z
    s1.newton_o(zc)                                   # the corrector's last newton_o, as newton_od leaves it
    t0 = time.time()
    if opt["mutate"] == "base":
        pmin, pmax, how = st0[3], st0[4], "kept"       # the control: the LU at z + ulp used without the key
    else:
        pmin, pmax, how = base_lu_o(s1, z)
    Z1, _ = coefficients_o(s1, z, opt["N"])
    t_new = time.time() - t0
    print(f"  base: series to N {opt['N']}: newton_o and series {t_old:.1f} s, kept LU and series {t_new:.1f} s", flush=True)
    rep.check("base", "the corrector's LU is kept at z", how == "kept", how)
    rep.same("base", "pivots", (st0[3], st0[4]), (pmin, pmax))
    rep.same("base", f"series z_0..z_{opt['N']}, eight parts", Z0, Z1)
    # the key: a state one ulp off, another mutate or tangents, and a new plan must each refuse the kept LU
    if opt["mutate"] != "base":
        other = 1 if s1.octo_tangents != 1 else 2
        refused = dict(ulp=s1.olu_pivots_at(ulp_up(z, len(z[0]) - 1)), mutate=s1.olu_pivots_at(z, mutate=10),
                       tangents=s1.olu_pivots_at(z, tangents=other))
        s1.set_segs(s1.get_segs())
        refused["set_segs"] = s1.olu_pivots_at(z)
        rep.check("base", "the key refuses z one ulp off, mutate 10, other tangents, set_segs",
                  all(v is None for v in refused.values()), ", ".join(k for k, v in refused.items() if v is not None))


class Rec:
    """the solver wrapper newton_od runs on, recording every newton_o and residual_o call's input and output"""

    def __init__(self, so):
        self.so, self.calls = so, []

    def __getattr__(self, name):
        return getattr(self.so, name)

    def newton_o(self, z, mutate=0):
        out = self.so.newton_o(z, mutate=mutate)
        self.calls.append(("newton_o", [np.array(p) for p in z], out))
        return out

    def residual_o(self, z, mutate=0):
        out = self.so.residual_o(z, mutate=mutate)
        self.calls.append(("residual_o", [np.array(p) for p in z], out))
        return out


def site_pinned(raw, path, opt, rep):
    """PinnedO.newton_o's residual from its newton_o against residual_o, and newton_od's iterates"""
    import octo_corrector as oc
    from reseed import Reseeder
    kept = oc.PinnedO.kept_residual

    def old_path(self, z, mutate=0):
        return None

    def mutated(self, z, mutate=0):
        F = kept(self, z, mutate=mutate)
        return None if F is None else ulp_up(F, 0)
    z, meta, (s, _, _, so) = system(raw, path, opt["threads"])
    try:
        if opt["mutate"] == "pinned":
            oc.PinnedO.kept_residual = mutated
        t0 = time.time()
        new = so.newton_o(z)
        t_new = time.time() - t0
        F_kept = so.kept_residual(z)
        oc.PinnedO.kept_residual = old_path
        t0 = time.time()
        old = so.newton_o(z)
        t_old = time.time() - t0
    finally:
        oc.PinnedO.kept_residual = kept
    print(f"  pinned: PinnedO.newton_o at z with the kept residual {t_new:.1f} s, with residual_o {t_old:.1f} s", flush=True)
    rep.check("pinned", "the kernel kept newton_o's residual", F_kept is not None)
    rep.same("pinned", "newton_o's outputs (dz, F hi, F lo, pivots)", new, old)
    if opt["mutate"] != "pinned":
        rep.same("pinned", "newton_o_residual against residual_o, eight parts", F_kept, s.residual_o(z))
        rep.check("pinned", "the key refuses z one ulp off", s.newton_o_residual(ulp_up(z, len(z[0]) - 1)) is None)
    # newton_od from z moved off it, on both paths, every call recorded
    rs = Reseeder(meta, oc.D_of(meta), threads=opt["threads"])
    zs, _ = oc.plant_start(z, rs, 1e-12, 1, 0)
    runs = {}
    for label in ("new", "old"):
        _, _, (_, _, _, so) = system(raw, path, opt["threads"])
        rec = Rec(so)
        try:
            if label == "old":
                oc.PinnedO.kept_residual = old_path
            elif opt["mutate"] == "pinned":
                oc.PinnedO.kept_residual = mutated
            t0 = time.time()
            zf, hist, conv = oc.newton_od(rec, zs, tol=1e-40, it_max=opt["its"], verbose=False, accept="natural")
            runs[label] = (zf, hist, conv, rec.calls, time.time() - t0)
        finally:
            oc.PinnedO.kept_residual = kept
    print(f"  pinned: newton_od {opt['its']} steps, {len(runs['new'][3])} calls: {runs['new'][4]:.1f} s with the kept "
          f"residual, {runs['old'][4]:.1f} s without; max|F| {[h['F'] for h in runs['new'][1]]}", flush=True)
    rep.check("pinned", "newton_od took steps", len(runs["new"][1]) > 0 and len(runs["new"][3]) > 1)
    rep.same("pinned", "newton_od's calls (inputs and outputs), final z and history",
             [runs["new"][3], runs["new"][0], timeless(runs["new"][1]), runs["new"][2]],
             [runs["old"][3], runs["old"][0], timeless(runs["old"][1]), runs["old"][2]])


def site_near_null(raw, path, opt, rep):
    """near_null.py's one Jacobian: newton_o's kept against jacobian_o, and the subspace iteration on both LUs"""
    from near_null import subspace
    z, _, (s0, _, _, _) = system(raw, path, opt["threads"])
    t0 = time.time()
    st0 = s0.newton_o(z)
    J0 = s0.jacobian_o(z)
    t_old = time.time() - t0
    F0 = s0.residual_o(z)
    _, _, (s1, _, _, _) = system(raw, path, opt["threads"])
    t0 = time.time()
    st1 = s1.newton_o(z, keep_jacobian=True)
    J1 = s1.jacobian_o_kept(z)
    t_new = time.time() - t0
    F1 = s1.newton_o_residual(z)
    print(f"  near_null: newton_o and jacobian_o {t_old:.1f} s, newton_o keeping its Jacobian {t_new:.1f} s", flush=True)
    rep.check("near_null", "the Jacobian was kept at z", J1 is not None)
    if J1 is None:
        return
    J1 = [np.array(p) for p in J1]
    if opt["mutate"] == "near_null":
        r, c = len(z[0]) // 2, len(z[0]) // 3
        J1[0][r, c] = np.nextafter(J1[0][r, c], np.inf)
    rep.same("near_null", "newton_o's step with and without keep_jacobian", st0, st1)
    rep.same("near_null", "the kept Jacobian against jacobian_o, eight parts", J0, J1)
    rep.same("near_null", "newton_o_residual against residual_o, eight parts", F0, F1)
    L, zero = len(z[0]), np.zeros(len(z[0]))

    def solver(s):
        def solve(b, transpose=False):
            x = s.olu_solve([np.ascontiguousarray(b)] + [zero] * 7, transpose)
            return np.asarray(x[0]) + np.asarray(x[1])
        return solve
    rep.same("near_null", "subspace vectors and sigma estimates", subspace(solver(s0), L, 4, 6), subspace(solver(s1), L, 4, 6))
    if opt["mutate"] != "near_null":
        s1.newton_o(z)
        rep.check("near_null", "the key refuses z one ulp off and a newton_o that kept no Jacobian",
                  s1.jacobian_o_kept(ulp_up(z, L - 1)) is None and s1.jacobian_o_kept(z) is None)


def rung_point(rs, z, parts):
    """the u0 rung solver's two-node point from the state z (the centre data, node 0, the last node as node 1, the
    horizon data and Delta), as Reseeder.polish_q and refine_o put it, in the first `parts` parts"""
    M, a, o = rs.M, rs.aoff, rs.off
    n0, n1, nk = o["node"](0), o["node"](1), o["node"](rs.K - 1)
    out = []
    for q in range(parts):
        b, zq = np.zeros(a["L"]), np.asarray(z[q], dtype=float)
        b[:2 * M] = zq[:2 * M]
        b[a["node"](0):a["node"](1)] = zq[n0:n1]
        b[a["node"](1):a["node"](1) + 4 * M] = zq[nk:nk + 4 * M]
        b[a["mh"]:a["mh"] + 2 * M] = zq[o["mh"]:o["mh"] + 2 * M]
        b[-1] = zq[-1]
        out.append(np.ascontiguousarray(b))
    return out


def site_reseed(raw, path, opt, rep):
    """the rung blocks from jacobian_sub against the full Jacobian's, and the rung solves on both"""
    import reseed
    from multidouble import od_axpy
    from octo_corrector import D_of, o8, q4_of_o
    z, meta, _ = system(raw, path, opt["threads"])
    rs = reseed.Reseeder(meta, D_of(meta), threads=opt["threads"], qd=True, od=2)
    M, a = rs.M, rs.aoff
    s = rs.rung_solver(rs.us[-1])
    sub = reseed.jac_block

    def mutated(s_, z_, rows_, cols_):
        B = [np.array(p) for p in sub(s_, z_, rows_, cols_)]
        B[0][0, 0] = np.nextafter(B[0][0, 0], np.inf)
        return B
    if opt["mutate"] == "reseed":
        reseed.jac_block = mutated
    try:
        rows, cols = rs.rung_rows(), np.r_[a["mh"]:a["mh"] + 2 * M]
        L, t = a["L"], a["node"](1)
        mix_rows = np.r_[L - 1:-1:-1, 3, 3]
        mix_cols = np.array([L - 1, t + 2, 1, t + M + 5, a["node"](0) + 3, 0, t + 2, a["mh"] + 1])
        for parts, name in ((2, "double-double"), (4, "quad-double")):
            p = rung_point(rs, z, parts)
            for rr, cc, what in ((rows, cols, "rung block"), (mix_rows, mix_cols, "all rows, mixed columns")):
                t0 = time.time()
                new = reseed.jac_block(s, p, rr, cc)
                t1 = time.time()
                old = reseed.jac_block_full(s, p, rr, cc)
                t2 = time.time()
                print(f"  reseed: {name} {what}: jacobian_sub {t1 - t0:.2f} s, full {t2 - t1:.2f} s", flush=True)
                rep.same("reseed", f"{name} {what}", new, old)
        p8 = rung_point(rs, z, 8)
        for tg in (1, 2):
            old = [np.asarray(q)[np.ix_(rows, cols)] for q in s.jacobian_o(p8, tangents=tg)]
            new = s.jacobian_sub(p8, [int(r) for r in rows], [int(c) for c in cols])
            if opt["mutate"] == "reseed":
                new = [np.array(q) for q in new]
                new[0][0, 0] = np.nextafter(new[0][0, 0], np.inf)
            rep.same("reseed", f"octo-double rung block against jacobian_o with {tg} tangents", new, old)
        # the rung solves from the state's horizon data moved by 1e-20 relative, on both paths
        sl = slice(rs.off["mh"], rs.off["mh"] + 2 * M)
        rng = np.random.default_rng(3)
        dz = np.zeros(len(z[0]))
        dz[sl] = 1e-20 * rng.uniform(0.5, 1.0, 2 * M) * z[0][sl]
        zero = np.zeros(len(dz))
        zm = o8(od_axpy(1.0, dz, *([zero] * 7), *z))
        z4 = q4_of_o(zm)
        d4 = tuple(np.ascontiguousarray(np.asarray(p)[sl]) for p in z4)
        nk = rs.off["node"](rs.K - 1)
        th, tl = np.array(zm[0][nk:nk + 4 * M]), np.array(zm[1][nk:nk + 4 * M])
        res = {}
        for label in ("new", "old"):
            if label == "old":
                reseed.jac_block = reseed.jac_block_full
            t0 = time.time()
            inv = rs.invert(rs.us[-1], np.array(zm[0]), np.array(zm[1]), th, tl, np.array(zm[0][sl]), np.array(zm[1][sl]))
            t1 = time.time()
            pq = rs.polish_q(z4, d4, it_max=3)
            t2 = time.time()
            ro = rs.refine_o(zm, steps=2)
            t3 = time.time()
            res[label] = (inv, pq, ro)
            print(f"  reseed: rung solves ({label}): invert {t1 - t0:.1f} s ({len(inv[2]) - 1} its), polish_q {t2 - t1:.1f} s, "
                  f"refine_o {t3 - t2:.1f} s; |R| {inv[2][-1]['R']:.2e}, {pq[1][-1]['R']:.2e}, {ro[1][-1]:.2e}", flush=True)
            reseed.jac_block = mutated if opt["mutate"] == "reseed" else sub
        for k, what in enumerate(("invert (dd)", "polish_q (qd)", "refine_o (od)")):
            rep.same("reseed", f"{what}: data, history, convergence", res["new"][k], res["old"][k])
    finally:
        reseed.jac_block = sub


def main():
    args, opt = parse(sys.argv[1:])
    if len(args) != 2:
        sys.exit(__doc__)
    raw, path = args
    import mpmath as mp
    mp.mp.dps = 50                  # anm_run.py's precision, which its import sets
    import hpkernel
    from hpkernel.driver import Solver
    print(f"hpkernel {hpkernel.__file__}; mutate '{opt['mutate']}'", flush=True)
    missing = [n for n in NEW if not hasattr(Solver, n)]
    if missing:
        print(f"this hpkernel has no Solver.{', '.join(missing)}: build hp from this branch (hp/build.sh)", flush=True)
        return 2
    rep = Report()
    sites = dict(base=site_base, pinned=site_pinned, near_null=site_near_null, reseed=site_reseed)
    for name in opt["sites"].split(","):
        t0 = time.time()
        print(f"{name}:", flush=True)
        sites[name](raw, path, opt, rep)
        print(f"  {name}: {time.time() - t0:.0f} s", flush=True)
    print("check:", "pass" if not rep.fails else "FAIL (" + "; ".join(rep.fails) + ")", flush=True)
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
