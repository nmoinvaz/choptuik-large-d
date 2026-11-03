#!/usr/bin/env python3
"""Route 1's prototype tests T1-T4 (notes/QUAD_DOUBLE.md 13.4) on the kernel's stored double-double states, in f64
(dss_colloc.py beside dss.py). Each test appends its records to OUT.jsonl and prints its reading.

  t1   consistency: the collocation rows on the double-double series' samples, Nu = 16, 24, 32, 48
  t2   the basin: Newton from the state with the horizon datum a_h scaled by 1 + delta (and a random vector on
       m_h, a_h), both blocks; the series block must reproduce 12.2 (a residual that is not finite or explodes
       beyond its basin), the collocation must converge from every start
  t3   the pivots: the smallest singular values of the Jacobian at the polished solution, both blocks, with the last
       node at 0.7, and the collocation (and the series where its series converges) with the last node at 0.6, 0.5
  t4   agreement: Delta from the polished f64 solutions of both blocks against the double-double value, Nu = 16..48;
       the mutations p0sign, nogauge, dut, each of which must fire

    python3 colloc_tests.py OUT.jsonl TEST STATE.npz [STATE.npz ...] [--Nu 24] [--deltas 1e-6,1e-4,1e-3,1e-2]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dss_colloc import DSSColloc, DSSMulti, colloc_state_from_kernel, load_state, solvers  # noqa: E402


def norm(F):
    return float(np.linalg.norm(F)) if np.all(np.isfinite(F)) else float("inf")


def newton(s, z, tol, maxit=12, verbose=False):
    """Newton with dss.py's line search; returns (z, |F| history, converged, note)"""
    hist = []
    try:
        for it in range(maxit):
            F, J = s.jacobian(z)
            nF = norm(F)
            hist.append(nF)
            if verbose:
                print(f"      it {it:2d} |F| {nF:.3e} Delta {z[-1]:.12f}", flush=True)
            if not np.isfinite(nF):
                return z, hist, False, "residual not finite"
            if nF < tol:
                return z, hist, True, "converged"
            dz = np.linalg.lstsq(J, -F, rcond=None)[0]
            t = 1.0
            for _ in range(14):
                zt = z + t * dz
                nFt = norm(s.residual(zt[None, :])[0])
                if nFt < nF * (1 - 0.1 * t) or t < 1e-4:
                    break
                t *= 0.5
            if not (nFt < nF):
                return zt, hist + [nFt], False, "no decrease"
            z = zt
        F = s.residual(z[None, :])[0]
        hist.append(norm(F))
        return z, hist, hist[-1] < tol, "iterations exhausted"
    except (np.linalg.LinAlgError, ValueError) as ex:
        return z, hist, False, f"{type(ex).__name__}: {ex}"


def floor_of(s, z):
    return norm(s.residual(z[None, :])[0])


def polish(s, z, tol_factor=0.5, maxit=8):
    """Newton from the double-double state until f64 stalls: the f64 solution of this formulation"""
    f0 = floor_of(s, z)
    z1, hist, ok, note = newton(s, z, tol=tol_factor * f0, maxit=maxit)
    return z1, hist


def run_t1(out, states, Nus, threads):
    for path in states:
        zh, meta = load_state(path)
        for Nu in Nus:
            sm, sc = solvers(meta, Nu=Nu)
            H, K, L = sc.layout()
            zc = colloc_state_from_kernel(path, sc, zh, threads=threads)
            F = sc.residual(zc[None, :])[0]
            Fm = sm.residual(zh[None, :])[0]
            blk = 4 * H
            rec = dict(test="t1", state=os.path.basename(path), D=meta["D"], Nu=Nu, coll_rows_max=float(np.abs(F[blk * (K + 1):-1]).max()),
                       match_rows_max=float(np.abs(F[:blk * (K + 1)]).max()), series_horizon_rows_max=float(np.abs(Fm[blk * K:blk * (K + 1)]).max()),
                       series_F=norm(Fm), coll_F=norm(F))
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(f"t1 D {meta['D']:.2f} Nu {Nu}: collocation rows max {rec['coll_rows_max']:.1e}; the series block's horizon rows {rec['series_horizon_rows_max']:.1e}", flush=True)


def perturb_series(sm, z, delta, kind, rng):
    H, K, L = sm.layout()
    o_s = 2 * H + 4 * H * K
    z = z.copy()
    if kind == "mass":
        z[o_s + H:o_s + 2 * H] *= 1 + delta
    else:
        v = z[o_s:o_s + 2 * H]
        r = rng.standard_normal(2 * H)
        z[o_s:o_s + 2 * H] = v + delta * np.linalg.norm(v) * r / np.linalg.norm(r)
    return z


def perturb_coll(sc, z, delta, kind, rng):
    H, K, L = sc.layout()
    oc = sc.offsets()["coll"]
    z = z.copy()
    if kind == "mass":
        z[oc + 2 * H:oc + 3 * H] *= 1 + delta                 # a at the horizon point
    else:
        v = z[oc + H:oc + 3 * H]                              # m, a at the horizon point
        r = rng.standard_normal(2 * H)
        z[oc + H:oc + 3 * H] = v + delta * np.linalg.norm(v) * r / np.linalg.norm(r)
    return z


def run_t2(out, states, Nu, deltas, threads):
    rng = np.random.default_rng(0)
    for path in states:
        zh, meta = load_state(path)
        sm, sc = solvers(meta, Nu=Nu)
        zc0 = colloc_state_from_kernel(path, sc, zh, threads=threads)
        fm, fc = floor_of(sm, zh), floor_of(sc, zc0)
        print(f"t2 D {meta['D']:.2f}: f64 floors |F| series {fm:.1e}, collocation {fc:.1e}; Delta(dd) {zh[-1]:.12f}", flush=True)
        for kind in ("mass", "random"):
            for delta in deltas:
                for name, s, z0, f0 in (("series", sm, zh, fm), ("colloc", sc, zc0, fc)):
                    zp = perturb_series(sm, z0, delta, kind, rng) if name == "series" else perturb_coll(sc, z0, delta, kind, rng)
                    t = time.time()
                    F0 = norm(s.residual(zp[None, :])[0])
                    z1, hist, ok, note = newton(s, zp, tol=10 * f0, maxit=14)
                    rec = dict(test="t2", state=os.path.basename(path), D=meta["D"], Nu=Nu, block=name, kind=kind, delta=delta,
                               F_start=F0, hist=hist, converged=bool(ok), note=note, Delta=float(z1[-1]), Delta_dd=float(zh[-1]),
                               dDelta=float(z1[-1] - zh[-1]), seconds=time.time() - t)
                    out.write(json.dumps(rec) + "\n"); out.flush()
                    h = " ".join(f"{x:.1e}" for x in hist[:8]) + (" ..." if len(hist) > 8 else "")
                    print(f"   {kind:6s} delta {delta:.0e} {name:7s}: start |F| {F0:.1e}; {h}; {'CONVERGED' if ok else 'FAILED'} ({note}); "
                          f"Delta - dd {z1[-1] - zh[-1]:+.1e}; {time.time() - t:.0f} s", flush=True)


def svals(s, z):
    F, J = s.jacobian(z)
    sv = np.linalg.svd(J, compute_uv=False)
    return sv


def run_t3(out, states, Nu, threads):
    for path in states:
        zh, meta = load_state(path)
        nodes0 = list(meta["nodes"])
        D = meta["D"]
        for last in (len(nodes0), len(nodes0) - 1, len(nodes0) - 2):
            nodes = nodes0[:last]
            u0 = 1 - nodes[-1]
            H = meta["M"]
            # the state in this layout: the nodes kept, the horizon data and Delta the same
            K0 = len(nodes0)
            zs = np.concatenate([zh[:2 * H + 4 * H * last], zh[2 * H + 4 * H * K0:]])
            sm, sc = solvers(meta, Nu=Nu, nodes=nodes)
            rec = dict(test="t3", state=os.path.basename(path), D=D, Nu=Nu, nodes=nodes, u0=u0)
            # the series block, where its series converges at u0 (its residual finite and at the floor)
            Fm = sm.residual(zs[None, :])[0]
            if np.all(np.isfinite(Fm)) and norm(Fm) < 1e-3:
                z1, hist = polish(sm, zs)
                sv = svals(sm, z1)
                rec["series"] = dict(F=hist[-1], smin=float(sv[-1]), s5=[float(x) for x in sv[-5:]], below_1e10=int((sv < 1e-10).sum()), below_1e8=int((sv < 1e-8).sum()), n=len(sv))
            else:
                rec["series"] = dict(F=norm(Fm), note="the series does not converge at this u0")
            try:
                zc = colloc_state_from_kernel(path, sc, zs, threads=threads)
                Fc = sc.residual(zc[None, :])[0]
                if not np.all(np.isfinite(Fc)) or norm(Fc) > 1e-3:
                    rec["colloc"] = dict(F=norm(Fc), note="the double-double series does not converge at these points")
                else:
                    z1, hist = polish(sc, zc)
                    sv = svals(sc, z1)
                    rec["colloc"] = dict(F=hist[-1], smin=float(sv[-1]), s5=[float(x) for x in sv[-5:]], below_1e10=int((sv < 1e-10).sum()), below_1e8=int((sv < 1e-8).sum()), n=len(sv))
            except Exception as ex:  # noqa: BLE001
                rec["colloc"] = dict(note=f"{type(ex).__name__}: {ex}")
            out.write(json.dumps(rec) + "\n"); out.flush()
            fmt = lambda r: (f"smin {r['smin']:.1e}, below 1e-10: {r['below_1e10']}, below 1e-8: {r['below_1e8']} of {r['n']}" if "smin" in r else r.get("note"))
            print(f"t3 D {D:.2f} last node {nodes[-1]} (u0 {u0:.2f}): series {fmt(rec['series'])}; collocation {fmt(rec['colloc'])}", flush=True)


def run_t4(out, states, Nus, threads):
    for path in states:
        zh, meta = load_state(path)
        D = meta["D"]
        sm, _ = solvers(meta, Nu=Nus[0])
        zs, hist = polish(sm, zh)
        rec = dict(test="t4", state=os.path.basename(path), D=D, Delta_dd=float(zh[-1]), series=dict(F=hist[-1], Delta=float(zs[-1]), dDelta=float(zs[-1] - zh[-1])), colloc={})
        print(f"t4 D {D:.2f}: Delta(dd) {zh[-1]:.12f}; series f64 polish: |F| {hist[-1]:.1e}, Delta - dd {zs[-1] - zh[-1]:+.1e}", flush=True)
        for Nu in Nus:
            _, sc = solvers(meta, Nu=Nu)
            zc = colloc_state_from_kernel(path, sc, zh, threads=threads)
            z1, hist = polish(sc, zc)
            rec["colloc"][str(Nu)] = dict(F=hist[-1], Delta=float(z1[-1]), dDelta=float(z1[-1] - zh[-1]))
            print(f"   collocation Nu {Nu}: |F| {hist[-1]:.1e}, Delta - dd {z1[-1] - zh[-1]:+.1e}", flush=True)
        # mutations at Nus[0]
        rec["mutations"] = {}
        for mut in ("p0sign", "nogauge", "dut"):
            _, scm = solvers(meta, Nu=Nus[0], mutate=mut)
            zc = colloc_state_from_kernel(path, scm, zh, threads=threads)
            F0 = norm(scm.residual(zc[None, :])[0])
            z1, hist, ok, note = newton(scm, zc, tol=10 * rec["colloc"][str(Nus[0])]["F"], maxit=10)
            try:
                sv = np.linalg.svd(scm.jacobian(z1)[1], compute_uv=False)
                smin = float(sv[-1])
            except Exception:  # noqa: BLE001
                smin = None
            m = dict(F_start=F0, F=hist[-1] if hist else None, converged=bool(ok), note=note, dDelta=float(z1[-1] - zh[-1]), smin=smin)
            rec["mutations"][mut] = m
            print(f"   mutation {mut}: start |F| {F0:.1e}, end |F| {m['F']:.1e}, {note}, Delta - dd {m['dDelta']:+.1e}, smin {smin}", flush=True)
        out.write(json.dumps(rec) + "\n"); out.flush()


def main():
    args = list(sys.argv[1:])
    Nu, deltas, threads = 24, (1e-6, 1e-4, 1e-3, 1e-2), 2
    if "--Nu" in args:
        i = args.index("--Nu"); Nu = int(args[i + 1]); del args[i:i + 2]
    if "--deltas" in args:
        i = args.index("--deltas"); deltas = tuple(float(x) for x in args[i + 1].split(",")); del args[i:i + 2]
    if "--threads" in args:
        i = args.index("--threads"); threads = int(args[i + 1]); del args[i:i + 2]
    outp, test, states = args[0], args[1], args[2:]
    with open(outp, "a") as out:
        if test == "t1":
            run_t1(out, states, (16, 24, 32, 48), threads)
        elif test == "t2":
            run_t2(out, states, Nu, deltas, threads)
        elif test == "t3":
            run_t3(out, states, Nu, threads)
        elif test == "t4":
            run_t4(out, states, (16, 24, 32, 48), threads)
        elif test == "t3p":
            run_t3p(out, states, Nu, threads)
        elif test == "t4p":
            run_t4p(out, states, (16, 24, 32), threads)
        elif test == "t2p":
            run_t2p(out, states, Nu, deltas, threads)
        else:
            raise SystemExit(f"unknown test {test}")


# ---------------------------------------------------------------- the pin (kfreeze): p1 harmonics > KF removed, as the kernel does
KF = 13


def p1_keep_basis(s):
    """orthonormal basis (H x nkeep) of p1's odd harmonics k <= KF in the half-sample representation"""
    H, N = s.N // 2, s.N
    tau = np.arange(N) * 2 * np.pi / N
    cols = []
    for k in range(1, KF + 1, 2):
        for f in (np.cos, np.sin):
            v = f(k * tau)[:H]
            cols.append(v / np.linalg.norm(v))
    B = np.array(cols).T
    Q, _ = np.linalg.qr(B)
    return Q


def pinned_rows(s, is_colloc):
    """the row indices of the p matching at the last node (the horizon block's p rows), which the pin projects"""
    H, K, L = s.layout()
    r0 = 4 * H * K                      # the horizon block's matching rows: p first
    return list(range(r0, r0 + H))


def reduce(s, F, J, is_colloc):
    """(F_r, J_r, expand): the system with p1 restricted to harmonics <= KF and the horizon p-matching rows projected
    onto them; expand maps a reduced step back to the full unknowns"""
    H, K, L = s.layout()
    Q = p1_keep_basis(s)                                   # H x nk
    nk = Q.shape[1]
    rows = pinned_rows(s, is_colloc)
    # columns: p1 -> Q y
    Jc = np.concatenate([J[:, :H] @ Q, J[:, H:]], axis=1)
    # rows: the H p-matching rows -> nk projected rows
    keep = np.ones(J.shape[0], bool); keep[rows] = False
    Pr = Q.T @ J[rows, :]                                  # nk x L, projected rows (full columns)
    Prc = np.concatenate([Pr[:, :H] @ Q, Pr[:, H:]], axis=1)
    Jr = np.concatenate([Jc[keep], Prc], axis=0)
    Fr = np.concatenate([F[keep], Q.T @ F[rows]])
    def expand(dy):
        return np.concatenate([Q @ dy[:nk], dy[nk:]])
    return Fr, Jr, expand


def newton_pinned(s, z, tol, is_colloc, maxit=12):
    hist = []
    try:
        for it in range(maxit):
            F, J = s.jacobian(z)
            Fr, Jr, expand = reduce(s, F, J, is_colloc)
            nF = norm(Fr)
            hist.append(nF)
            if not np.isfinite(nF):
                return z, hist, False, "residual not finite"
            if nF < tol:
                return z, hist, True, "converged"
            dy = np.linalg.solve(Jr, -Fr) if Jr.shape[0] == Jr.shape[1] else np.linalg.lstsq(Jr, -Fr, rcond=None)[0]
            dz = expand(dy)
            t = 1.0
            for _ in range(14):
                zt = z + t * dz
                Ft = s.residual(zt[None, :])[0]
                nFt = norm(reduce(s, Ft, J, is_colloc)[0]) if np.all(np.isfinite(Ft)) else float("inf")
                if nFt < nF * (1 - 0.1 * t) or t < 1e-4:
                    break
                t *= 0.5
            if not (nFt < nF):
                return zt, hist + [nFt], False, "no decrease"
            z = zt
        F = s.residual(z[None, :])[0]
        hist.append(norm(reduce(s, F, s.jacobian(z)[1], is_colloc)[0]))
        return z, hist, hist[-1] < tol, "iterations exhausted"
    except (np.linalg.LinAlgError, ValueError) as ex:
        return z, hist, False, f"{type(ex).__name__}: {ex}"


def run_t3p(out, states, Nu, threads):
    """T3 with the pin: the singular values of the reduced square systems"""
    for path in states:
        zh, meta = load_state(path)
        nodes0 = list(meta["nodes"]); D = meta["D"]; H = meta["M"]; K0 = len(nodes0)
        for last in (K0, K0 - 1, K0 - 2):
            nodes = nodes0[:last]
            zs = np.concatenate([zh[:2 * H + 4 * H * last], zh[2 * H + 4 * H * K0:]])
            sm, sc = solvers(meta, Nu=Nu, nodes=nodes)
            rec = dict(test="t3p", state=os.path.basename(path), D=D, Nu=Nu, nodes=nodes, u0=1 - nodes[-1])
            for name, s, z0, isc in (("series", sm, zs, False), ("colloc", sc, None, True)):
                try:
                    if isc:
                        z0 = colloc_state_from_kernel(path, sc, zs, threads=threads)
                    F = s.residual(z0[None, :])[0]
                    if not np.all(np.isfinite(F)) or norm(F) > 1e-3:
                        rec[name] = dict(note="the series does not converge at this u0"); continue
                    z1, hist, ok, note = newton_pinned(s, z0, tol=0.5 * norm(reduce(s, F, s.jacobian(z0)[1], isc)[0]), is_colloc=isc, maxit=6)
                    F, J = s.jacobian(z1)
                    Fr, Jr, _ = reduce(s, F, J, isc)
                    sv = np.linalg.svd(Jr, compute_uv=False)
                    rec[name] = dict(F=norm(Fr), shape=list(Jr.shape), smin=float(sv[-1]), s5=[float(x) for x in sv[-5:]], below_1e10=int((sv < 1e-10).sum()), below_1e8=int((sv < 1e-8).sum()), below_1e6=int((sv < 1e-6).sum()), Delta=float(z1[-1]), dDelta=float(z1[-1] - zh[-1]))
                except Exception as ex:  # noqa: BLE001
                    rec[name] = dict(note=f"{type(ex).__name__}: {ex}")
            out.write(json.dumps(rec) + "\n"); out.flush()
            fmt = lambda r: (f"smin {r['smin']:.1e} ({r['shape'][0]} rows), below 1e-10/1e-8/1e-6: {r['below_1e10']}/{r['below_1e8']}/{r['below_1e6']}, Delta - dd {r['dDelta']:+.1e}" if "smin" in r else r.get("note"))
            print(f"t3p D {D:.2f} last node {nodes[-1]} (u0 {1 - nodes[-1]:.2f}): series {fmt(rec['series'])}; collocation {fmt(rec['colloc'])}", flush=True)


def run_t4p(out, states, Nus, threads):
    for path in states:
        zh, meta = load_state(path); D = meta["D"]
        sm, _ = solvers(meta, Nu=Nus[0])
        F = sm.residual(zh[None, :])[0]
        zs, hist, ok, note = newton_pinned(sm, zh, tol=0.5 * norm(reduce(sm, F, sm.jacobian(zh)[1], False)[0]), is_colloc=False, maxit=8)
        rec = dict(test="t4p", state=os.path.basename(path), D=D, Delta_dd=float(zh[-1]), series=dict(F=hist[-1], dDelta=float(zs[-1] - zh[-1])), colloc={})
        print(f"t4p D {D:.2f}: series pinned polish |F| {hist[-1]:.1e}, Delta - dd {zs[-1] - zh[-1]:+.1e}", flush=True)
        for Nu in Nus:
            _, sc = solvers(meta, Nu=Nu)
            zc = colloc_state_from_kernel(path, sc, zh, threads=threads)
            F = sc.residual(zc[None, :])[0]
            z1, hist, ok, note = newton_pinned(sc, zc, tol=0.5 * norm(reduce(sc, F, sc.jacobian(zc)[1], True)[0]), is_colloc=True, maxit=8)
            rec["colloc"][str(Nu)] = dict(F=hist[-1], dDelta=float(z1[-1] - zh[-1]), hist=hist)
            print(f"   collocation Nu {Nu} pinned polish: |F| {hist[-1]:.1e}, Delta - dd {z1[-1] - zh[-1]:+.1e}", flush=True)
        out.write(json.dumps(rec) + "\n"); out.flush()


def run_t2p(out, states, Nu, deltas, threads):
    rng = np.random.default_rng(0)
    for path in states:
        zh, meta = load_state(path); D = meta["D"]
        sm, sc = solvers(meta, Nu=Nu)
        zc0 = colloc_state_from_kernel(path, sc, zh, threads=threads)
        fl = {}
        for name, s, z0, isc in (("series", sm, zh, False), ("colloc", sc, zc0, True)):
            F = s.residual(z0[None, :])[0]
            z1, hist, ok, note = newton_pinned(s, z0, tol=0.5 * norm(reduce(s, F, s.jacobian(z0)[1], isc)[0]), is_colloc=isc, maxit=6)
            fl[name] = hist[-1]
        print(f"t2p D {D:.2f}: pinned floors series {fl['series']:.1e}, collocation {fl['colloc']:.1e}", flush=True)
        for kind in ("mass", "random"):
            for delta in deltas:
                for name, s, z0, isc in (("series", sm, zh, False), ("colloc", sc, zc0, True)):
                    zp = perturb_series(sm, z0, delta, kind, rng) if name == "series" else perturb_coll(sc, z0, delta, kind, rng)
                    t = time.time()
                    F0 = norm(s.residual(zp[None, :])[0])
                    z1, hist, ok, note = newton_pinned(s, zp, tol=10 * fl[name], is_colloc=isc, maxit=12)
                    rec = dict(test="t2p", state=os.path.basename(path), D=D, Nu=Nu, block=name, kind=kind, delta=delta, F_start=F0, hist=hist,
                               converged=bool(ok), note=note, dDelta=float(z1[-1] - zh[-1]), seconds=time.time() - t)
                    out.write(json.dumps(rec) + "\n"); out.flush()
                    h = " ".join(f"{x:.1e}" for x in hist[:8]) + (" ..." if len(hist) > 8 else "")
                    print(f"   {kind:6s} delta {delta:.0e} {name:7s}: start |F| {F0:.1e}; {h}; {'CONVERGED' if ok else 'FAILED'} ({note}); Delta - dd {z1[-1] - zh[-1]:+.1e}; {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
