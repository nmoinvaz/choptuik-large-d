#!/usr/bin/env python3
"""Predictor diagnostic for the large-D continuation: the residual of each predictor's seed at D1 + dD, before any
corrector step.

    python3 predictor_diag.py OUT.jsonl S1.npz,...,Sn.npz [--truth T1.npz,...] [--dDs 1e-4,3e-4,1e-3,3e-3,1e-2]
        [--kt 36 --kc 146 --ks 130 --kfreeze 13 --tol_step 1e-32 --hmax 0.05] [--hold 7] [--fd_h 1e-11,1e-12] [--pin_tangent 11,7]
        [--threads N] [--save_seeds DIR] [--require_controls] [--mutate=tangent_sign|seed_offset]

S1..Sn: converged states in one layout, oldest first; Sn is the base point D1. --truth: converged states at later D
in the same layout; each adds its D to the steps and records each seed's distance to it. The residual is
QdhSolver's (the horizon block in quad-double), at D1 + dD on a step plan made from the base state.

Seeds (all formed as increments from the base state z1, so no f64 rounding of z enters):
  const          z1
  lag            driver.predict (f64 Lagrange through the last 4 points; scan_hp --predictor lagrange)
  lag_dd         pinning.predict_dd (the same in double-double; scan_hp --predictor lagrange_dd)
  lag_dd_hold    lag_dd with p1's odd harmonics above --hold kept at z1's values
  euler_fd7      z1 + dD t, t = driver.tangent (dF/dD by a central difference, h = 1e-7, f64; scan_hp --predictor
                 hermite's tangent)
  euler_h<h>     z1 + dD t, dF/dD by the 4th-order difference on the f64 steps D1 + kh (k = ±1, ±2) of the quad-double-horizon
                 residual, summed in double-double, t = -J^-1 dF/dD with the double-double LU at z1
  euler_h<h>_hold  the same with p1's odd harmonics above --hold kept at z1's values
  euler_pin<K>   z1 + dD t, t the tangent of the kfreeze = K system (--pin_tangent; dF/dD as euler_h at --pin_h):
                 p1's harmonics above K stay at z1's values and out of the solve
  neg_<t>        z1 - dD t (control C2; t the pinned tangent of the smallest K, else the first --fd_h one)
  truth_inc      at a --truth step, z1 + (z_true - z1) with the increment in double-double (control C1)

Controls (F(z1, D) is far from linear in D here: const's |F| grows like dD^6-8 at D = 62-68, so neither control
assumes linearity):
  C1  the harness, at each --truth step: the truth state's residual and truth_inc's are within 1e3 of the floor (the
      step's D and plan are right, and seeds reach the residual unchanged), and lag_dd's |F| is below 1e-3 of
      const's (the seeds are formed from the history)
  C2  the tangent, at its own step --c2_dD (default 1e-9, where F(z1, D) is still linear in D: at 1e-6 it is 20 times
      its linear part at D = 67.7): |F(neg)| / |F(const)| is 1.5-2.5 (z1 - dD t twice as far off as z1)
--mutate=seed_offset adds 1e-9 |z1| to every seed and --mutate=wrong_D evaluates every step at D1 + 2 dD (C1 must fire
for both); --mutate=tangent_sign flips t (C2 must fire where it is testable).
With --require_controls the exit code is 1 when C1 or C2 fails or C1 cannot be tested.
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from hpkernel.driver import QdhSolver, load, load_parts, make_solver, plow_odd, predict, tangent  # noqa: E402
from pinning import dd_add, predict_dd  # noqa: E402
from multidouble import dd_scale  # noqa: E402


def as_f(r):
    return np.asarray(r[0]) + np.asarray(r[1])


def blocks(F, M):
    """max |F| per residual block (4M rows each; the last block is the horizon), then the final row"""
    nb = (len(F) - 1) // (4 * M)
    return [float(np.abs(F[b * 4 * M:(b + 1) * 4 * M]).max()) for b in range(nb)] + [float(abs(F[-1]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("states")
    ap.add_argument("--truth", default="")
    ap.add_argument("--dDs", default="1e-4,3e-4,1e-3,3e-3,1e-2")
    ap.add_argument("--kt", type=int, default=36)
    ap.add_argument("--kc", type=int, default=146)
    ap.add_argument("--ks", type=int, default=130)
    ap.add_argument("--kfreeze", type=int, default=13)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--hold", type=int, default=7)
    ap.add_argument("--fd_h", default="1e-11,1e-12")
    ap.add_argument("--pin_tangent", default="", help="comma list of K: also the tangent of the kfreeze = K system")
    ap.add_argument("--pin_h", type=float, default=1e-12, help="--pin_tangent: the difference step")
    ap.add_argument("--c2_dD", type=float, default=1e-9, help="control C2's step")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save_seeds", default="")
    ap.add_argument("--require_controls", action="store_true")
    ap.add_argument("--mutate", default="", choices=["", "tangent_sign", "seed_offset", "wrong_D"])
    a = ap.parse_args()
    t_start = time.time()

    paths = [p for p in a.states.split(",") if p]
    pts = []
    for p in paths:
        zh, zl, meta = load(p)
        pts.append((meta["D"], zh, zl, meta))
    D1, z1h, z1l, meta1 = pts[-1]
    M, nodes = meta1["M"], list(meta1["nodes"])
    for p, (_, _, _, m) in zip(paths, pts):
        if m["M"] != M or list(m["nodes"]) != nodes:
            raise SystemExit(f"{p}: layout differs from the base's (M {M}, nodes {nodes})")
    z1 = z1h + z1l
    L = len(z1h)
    kw = dict(nodes=nodes, kt=a.kt, kc=a.kc, ks=a.ks, threads=a.threads, kfreeze=a.kfreeze)

    # base: plan, double-double LU at z1, its quad-double-horizon residual (the floor)
    s1 = make_solver(D1, M, **kw)
    segs = s1.plan(z1h, z1l, a.tol_step, a.hmax)
    s1.set_segs(segs)
    s1.newton(z1h, z1l)
    q1 = QdhSolver(s1)
    z2, z3 = load_parts(paths[-1])
    q1._parts[q1._key(z1h, z1l)] = (z2, z3)
    F1 = as_f(q1.residual(z1h, z1l))
    floor = float(np.abs(F1).max())
    print(f"base D={D1} M={M} nodes={nodes} kfreeze={a.kfreeze}: |F| {floor:.2e} {['%.1e' % v for v in blocks(F1, M)]}"
          f" ({time.time() - t_start:.0f}s)", flush=True)

    def q_at(D, kf=a.kfreeze):
        s = make_solver(D, M, **dict(kw, kfreeze=kf))
        s.set_segs(segs)
        q = QdhSolver(s)
        q._parts[q._key(z1h, z1l)] = (z2, z3)
        return q

    def dfdd(h, kf=a.kfreeze):
        """dF/dD at z1 (double-double) of the kfreeze = kf residual, by the 4th-order difference on the steps actually
        taken: D1 + k h rounds to f64 (spacing 1.4e-14 near D = 68), so the weights are those of the derivative at 0
        of the quartic through the five points"""
        ks_ = (-2, -1, 1, 2)
        hk = np.array([(D1 + k * h) - D1 for k in ks_])
        V = np.vander(np.r_[hk, 0.0], 5, increasing=True)
        w = np.linalg.solve(V.T, np.r_[0.0, 1.0, 0.0, 0.0, 0.0])
        F0h, F0l = (np.asarray(v) for v in q_at(D1, kf).residual(z1h, z1l))
        dh, dl = np.zeros(L), np.zeros(L)
        for wk, k in zip(w[:4], ks_):
            r = q_at(D1 + k * h, kf).residual(z1h, z1l)
            # F(D1 + h_k) - F(D1) in double-double first (nearly equal values), then the weight
            eh, el = dd_add(np.asarray(r[0]), np.asarray(r[1]), -F0h, -F0l)
            eh, el = dd_scale(np.asarray(eh), np.asarray(el), wk)
            dh, dl = (np.asarray(v) for v in dd_add(dh, dl, eh, el))
        return dh, dl

    # tangents
    tans = {}
    t0 = time.time()
    tans["fd7"] = tangent(lambda D: make_solver(D, M, **kw), s1, D1, z1h, z1l, h=1e-7)
    for hs in [v for v in a.fd_h.split(",") if v]:
        dh, dl = dfdd(float(hs))
        th, tl = s1.lu_solve(-dh, -dl)
        tans[f"h{hs}"] = np.asarray(th) + np.asarray(tl)
    for K in [int(v) for v in a.pin_tangent.split(",") if v]:
        # the tangent of the kfreeze = K system: p1's harmonics above K drop out of the unknowns (their rows are
        # P_high p1, independent of D), so t keeps them at z1's values and the solve never sees their near-null
        # directions
        sK = make_solver(D1, M, **dict(kw, kfreeze=K))
        sK.set_segs(segs)
        sK.newton(z1h, z1l)
        dh, dl = dfdd(a.pin_h, K)
        th, tl = sK.lu_solve(-dh, -dl)
        tans[f"pin{K}"] = np.asarray(th) + np.asarray(tl)
    if a.mutate == "tangent_sign":
        tans = {k: -v for k, v in tans.items()}
    names = list(tans)
    # C2's tangent: the pinned one with the smallest K if any (the unpinned ones are not finite at large D)
    pins = sorted((int(k[3:]), k) for k in names if k.startswith("pin"))
    tneg = pins[0][1] if pins else (names[1] if len(names) > 1 else names[0])
    lo, lo2 = plow_odd(M, a.hold)
    P = np.asarray(lo) + np.asarray(lo2)
    for k in names:
        t = tans[k]
        print(f"tangent {k}: |t| {np.abs(t).max():.3e}: p1 low {np.abs(P @ t[:M]).max():.3e}, p1 held "
              f"{np.abs(t[:M] - P @ t[:M]).max():.3e}, rest {np.abs(t[M:-1]).max():.3e}, dDelta/dD {t[-1]:+.10e}",
              flush=True)
    if len(names) > 2:
        d = tans[names[1]] - tans[names[2]]
        print(f"tangent {names[1]} - {names[2]}: max {np.abs(d).max():.3e} (p1 {np.abs(d[:M]).max():.3e}, Delta "
              f"{abs(d[-1]):.3e})", flush=True)
    print(f"tangents: {time.time() - t0:.0f}s", flush=True)

    def hold(inc):
        """inc with p1's odd harmonics above --hold removed (those of the seed stay at z1's)"""
        out = inc.copy()
        out[:M] = P @ inc[:M]
        return out

    hist_z = [(p[0], p[1] + p[2]) for p in pts]
    hist_dd = [(p[0], p[1], p[2]) for p in pts]

    def seeds(Dn, tr=None):
        dD = Dn - D1
        incs = {"const": np.zeros(L)}
        if len(pts) > 1:
            incs["lag"] = predict(hist_z, Dn) - z1          # f64, as scan_hp: its rounding of z is part of it
            lh, ll = predict_dd(hist_dd, Dn, 4)
            ld = (np.asarray(lh) - z1h) + (np.asarray(ll) - z1l)
            incs["lag_dd"] = ld
            incs["lag_dd_hold"] = hold(ld)
        for k in names:
            incs[f"euler_{k}"] = dD * tans[k]
            if k != "fd7" and not k.startswith("pin"):
                incs[f"euler_{k}_hold"] = hold(dD * tans[k])
        incs[f"neg_{tneg}"] = -dD * tans[tneg]
        if tr is not None:
            incs["truth_inc"] = tuple(np.asarray(v) for v in dd_add(tr[0], tr[1], -z1h, -z1l))
        out = {}
        for n, inc in incs.items():
            if n == "lag":
                zp = predict(hist_z, Dn)
                out[n] = (zp, np.zeros(L))
            else:
                ih, il = inc if isinstance(inc, tuple) else (inc, np.zeros(L))
                out[n] = tuple(np.asarray(v) for v in dd_add(z1h, z1l, ih, il))
            if a.mutate == "seed_offset":
                out[n] = tuple(np.asarray(v) for v in dd_add(out[n][0], out[n][1], 1e-9 * np.abs(z1), np.zeros(L)))
        return out

    truths = {}
    for p in [v for v in a.truth.split(",") if v]:
        th_, tl_, mt = load(p)
        if mt["M"] != M or list(mt["nodes"]) != nodes:
            raise SystemExit(f"{p}: layout differs from the base's")
        truths[round(mt["D"], 6)] = (th_, tl_, p)
    steps = sorted(set([round(D1 + float(v), 6) for v in a.dDs.split(",") if v] + list(truths)))

    if a.save_seeds:
        os.makedirs(a.save_seeds, exist_ok=True)
    recs = []
    with open(a.out, "a") as fo:
        for Dn in steps:
            t0 = time.time()
            sn = make_solver(D1 + 2 * (Dn - D1) if a.mutate == "wrong_D" else Dn, M, **kw)
            sn.set_segs(segs)
            qn = QdhSolver(sn)
            row = dict(D1=D1, D=Dn, dD=Dn - D1, floor=floor, mutate=a.mutate, seeds={})
            tr = truths.get(Dn)
            if tr is not None:
                Ft = as_f(qn.residual(tr[0], tr[1]))
                row["truth"] = dict(file=tr[2], F=float(np.abs(Ft).max()), blocks=blocks(Ft, M))
            for n, (sh, sl) in seeds(Dn, tr).items():
                F = as_f(qn.residual(sh, sl))
                r = dict(F=float(np.abs(F).max()), blocks=blocks(F, M))
                if tr is not None:
                    d = (sh - tr[0]) + (sl - tr[1])
                    r["dist"] = dict(p1_low=float(np.abs(P @ d[:M]).max()), p1_held=float(np.abs(d[:M] - P @ d[:M]).max()),
                                     rest=float(np.abs(d[M:-1]).max()), Delta=float(d[-1]))
                row["seeds"][n] = r
                if a.save_seeds:
                    m = dict(meta1, D=Dn, predictor=n, D_base=D1)
                    np.savez(os.path.join(a.save_seeds, f"seed_{n}_D{Dn:.6f}.npz"), zh=sh, zl=sl, meta=json.dumps(m))
            row["seconds"] = round(time.time() - t0, 1)
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            recs.append(row)
            tline = f", truth |F| {row['truth']['F']:.1e}" if tr is not None else ""
            print(f"D={Dn} (dD {Dn - D1:.3e}){tline}, {row['seconds']:.0f}s", flush=True)
            for n, r in row["seeds"].items():
                dist = (f"  dist p1_low {r['dist']['p1_low']:.1e} p1_held {r['dist']['p1_held']:.1e} rest "
                        f"{r['dist']['rest']:.1e} Delta {r['dist']['Delta']:+.1e}") if "dist" in r else ""
                print(f"  {n:18s} |F| {r['F']:.2e}  horizon {r['blocks'][-2]:.1e}  inner {max(r['blocks'][:-2]):.1e}"
                      f"{dist}", flush=True)

    # controls
    neg = [k for k in recs[0]["seeds"] if k.startswith("neg_")][0]
    tr_rows = [r for r in recs if "truth" in r]
    c1 = bool(tr_rows)
    for r in tr_rows:
        f_t, f_c, f_l = r["truth"]["F"], r["seeds"]["const"]["F"], r["seeds"]["lag_dd"]["F"]
        f_i = r["seeds"]["truth_inc"]["F"]
        ok_r = f_t <= 1e3 * floor and f_i <= 1e3 * floor and f_l <= 1e-3 * f_c
        c1 &= ok_r
        print(f"C1 at D={r['D']}: truth |F| {f_t:.1e}, truth_inc {f_i:.1e} (floor {floor:.1e}), lag_dd / const {f_l / f_c:.1e}: "
              f"{'PASS' if ok_r else 'FIRE'}", flush=True)
    if not tr_rows:
        print("C1: not testable (no --truth)", flush=True)
    Dc = D1 + a.c2_dD
    qc = q_at(Dc)
    fc = float(np.abs(as_f(qc.residual(z1h, z1l))).max())
    nh_, nl_ = (np.asarray(v) for v in dd_add(z1h, z1l, -(Dc - D1) * tans[tneg], np.zeros(L)))
    fn = float(np.abs(as_f(qc.residual(nh_, nl_))).max())
    ratio = fn / fc
    c2 = bool(np.isfinite(ratio)) and 1.5 <= ratio <= 2.5
    print(f"C2 (|F({neg})| / |F(const)| at dD {Dc - D1:.1e}, const |F| {fc:.1e}): {ratio:.3f}: "
          f"{'PASS' if c2 else 'FIRE'}", flush=True)
    print(f"done in {time.time() - t_start:.0f}s", flush=True)
    if a.require_controls and not (c1 and c2):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
