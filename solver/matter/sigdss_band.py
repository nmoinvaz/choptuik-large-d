#!/usr/bin/env python3
"""How well does a sigdss state fix Delta? The width of Delta at the residual floor (#859).

    python3 solver/matter/sigdss_band.py width STATE --D D --eps EPS [--kc 15] [--json OUT] [--mutate dcol]
    python3 solver/matter/sigdss_band.py walk STATE --D D --eps EPS [--kc 15] [--maxjac 100] [--dtol 1e-9]
        [--settle 3] [--out OUT.npy] [--json OUT]
    python3 solver/matter/sigdss_band.py profile STATE --D D --eps EPS --deltas D1,D2,... [--kc 15] [--maxjac 30]
        [--settle 3] [--json OUT] [--mutate dfree]

STATE is a sigdss solution (.npy), or NPZ@D for the entry of a sigdss_run.py continuation file (v, Z) at that D.

width: at STATE, sigdss's complex-step Jacobian J (sigdss_solver.jacobian, with the RK steps frozen from STATE as
sigdss_run.py does) and |F| (2-norm). g = |e_Delta^T J^-1|, the largest move of Delta per unit residual, by the SVD
(sqrt(sum_i V[Delta, i]^2 / s_i^2)) and by a solve with J^T. W = g |F| is how far Delta can move at the state's own
residual. Printed beside it: the three smallest singular values with their vectors' Delta part, and Newton's Delta
step -(J^-1 F)_Delta. Check (exit 1): the two g agree to 1e-6 relative. Control: --mutate dcol scales J's Delta column
by 1e-6, which raises g by 1e6 where the Delta column is not near-null; the coordinator's acceptance reads it.

walk: Levenberg-Marquardt (sigdss_solver.lm, the steps frozen once from STATE, no stall stop), one Jacobian at a time,
until Delta moves less than --dtol in each of --settle (3) Jacobians in a row (settled), a Jacobian does not lower |F|
(stalled), or --maxjac Jacobians. Per Jacobian: |F|, Delta, its change.

profile: the least |F| at fixed Delta (#907). For each Delta of --deltas in the order given, Delta is set and held,
and Levenberg-Marquardt runs on the other unknowns only (J without its Delta column; the steps frozen once from STATE,
as in walk), from the previous point's minimizer (the first from STATE). It stops when |F| falls by less than 1% in
each of --settle Jacobians in a row, when a Jacobian does not lower |F|, or at --maxjac. A solution of the system
shows as a dip of the least |F| to the residual floor; two solutions as two dips. Per point: the least |F|, the
Jacobians used, and the three smallest singular values of J without its Delta column. Check (exit 1): every point
ends with Delta exactly at its target. Control: --mutate dfree keeps the Delta column, so Delta moves and the check
fails.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sigdss_solver import lm, make  # noqa: E402


def load_state(spec):
    if "@" in spec:
        path, D = spec.rsplit("@", 1)
        d = np.load(path)
        i = int(np.argmin(np.abs(np.asarray(d["v"], float) - float(D))))
        if abs(float(d["v"][i]) - float(D)) > 1e-9:
            raise SystemExit(f"sigdss_band.py: {path} has no entry at D {D}")
        return np.asarray(d["Z"][i], float)
    return np.asarray(np.load(spec), float)


def width(S, z, mutate=""):
    F, J = S.jacobian(z)
    if mutate == "dcol":
        J = J.copy()
        J[:, -1] *= 1e-6
    nF = float(np.linalg.norm(F))
    U, s, Vt = np.linalg.svd(J)
    vD = Vt[:, -1]                                   # each right singular vector's Delta part
    g_svd = float(np.sqrt(np.sum((vD / s) ** 2)))
    e = np.zeros(len(z)); e[-1] = 1.0
    row = np.linalg.solve(J.T, e)                    # e_Delta^T J^-1
    g_solve = float(np.linalg.norm(row))
    dD = float(-(row @ F))
    small = [dict(s=float(s[-k]), vDelta=float(vD[-k])) for k in (1, 2, 3)]
    return dict(F=nF, Delta=float(z[-1]), g_svd=g_svd, g_solve=g_solve, W=g_svd * nF, newton_dDelta=dD,
                smax=float(s[0]), smallest=small, agree=abs(g_svd - g_solve) <= 1e-6 * g_svd)


def lm_fixed(S, z, maxjac, settle, keep_delta=False, t0=None, log=print):
    """Levenberg-Marquardt on z[:-1] with z[-1] (Delta) held, sigdss_solver.lm's damping set and chord corrections"""
    mus = np.concatenate([[0.0], 10.0 ** np.arange(-12, 0.5, 1.0)])
    t0 = t0 or time.time()
    nv = len(z) if keep_delta else len(z) - 1
    F = S.residual(z[None, :])[0]
    nF = float(np.linalg.norm(F))
    hist, low = [], None
    for it in range(maxjac):
        _, J = S.jacobian(z)
        U, s, Vt = np.linalg.svd(J[:, :nv], full_matrices=False)
        if low is None:
            low = [float(x) for x in s[-3:][::-1]] + [float(s[0])]
        filt = np.stack([s / (s * s + (mu * s[0]) ** 2) for mu in mus])

        def step(Fb):
            d = np.zeros((Fb.shape[0], len(z)))
            d[:, :nv] = -((U.T @ Fb.T).T * filt) @ Vt
            return d
        Zt = z[None, :] + step(np.repeat(F[None, :], len(mus), 0))
        best = (nF, None)
        for c in range(3):
            Ft = S.residual(Zt)
            nt = np.linalg.norm(Ft, axis=1)
            for q in range(len(mus)):
                if np.isfinite(nt[q]) and nt[q] < best[0]:
                    best = (float(nt[q]), Zt[q].copy())
            if c == 2:
                break
            Ft[~np.isfinite(nt)] = 0.0
            Zt = Zt + step(Ft)
        if best[1] is None:
            hist.append(dict(jac=it, F=nF, stalled=True))
            break
        prev, z = nF, best[1]
        F = S.residual(z[None, :])[0]
        nF = float(np.linalg.norm(F))
        hist.append(dict(jac=it, F=nF, Delta=float(z[-1])))
        log(f"    jac {it}: |F| {nF:.4e} Delta {z[-1]:.12f} ({time.time() - t0:.0f} s)")
        rel = [h for h in hist if "Delta" in h]
        if len(rel) > settle and all(rel[-k - 1]["F"] - rel[-k]["F"] < 0.01 * rel[-k - 1]["F"]
                                     for k in range(1, settle + 1)):
            break
    return z, nF, hist, low


def main(argv):
    if len(argv) < 2 or argv[0] not in ("width", "walk", "profile"):
        raise SystemExit(__doc__)
    cmd, spec = argv[0], argv[1]
    opt = dict(D="", eps="", kc="15", json="", mutate="", maxjac="100", dtol="1e-9", out="", settle="3", deltas="")
    for k in opt:
        if "--" + k in argv:
            opt[k] = argv[argv.index("--" + k) + 1]
    D, eps, kc = float(opt["D"]), float(opt["eps"]), int(opt["kc"])
    z = load_state(spec)
    S = make(D, eps, z, kc=kc)
    t0 = time.time()
    rec = dict(cmd=cmd, state=spec, D=D, eps=eps, kc=kc, mutate=opt["mutate"])
    ok = True
    if cmd == "width":
        r = width(S, z, opt["mutate"])
        rec.update(r)
        ok = r["agree"]
        sm = ", ".join(f"{x['s']:.3e} (Delta part {x['vDelta']:+.3e})" for x in r["smallest"])
        print(f"sigdss_band width {spec}: D {D}, eps {eps}, kc {kc}{', --mutate ' + opt['mutate'] if opt['mutate'] else ''}"
              f"\n  Delta {r['Delta']:.12f}, |F| {r['F']:.4e}, smax {r['smax']:.4e}; smallest {sm}"
              f"\n  g = |e_Delta^T J^-1| {r['g_svd']:.4e} (SVD), {r['g_solve']:.4e} (solve): "
              f"{'agree' if r['agree'] else 'DISAGREE'} to 1e-6"
              f"\n  W = g |F| = {r['W']:.4e}; Newton's Delta step {r['newton_dDelta']:+.4e} ({time.time() - t0:.0f} s)",
              flush=True)
    elif cmd == "profile":
        targets = [float(x) for x in opt["deltas"].split(",") if x]
        if not targets:
            raise SystemExit("sigdss_band.py profile: --deltas D1,D2,... is required")
        maxjac = int(opt["maxjac"]) if "--maxjac" in argv else 30
        print(f"sigdss_band profile {spec}: D {D}, eps {eps}, kc {kc}, start Delta {z[-1]:.12f}, {len(targets)} points"
              f"{', --mutate ' + opt['mutate'] if opt['mutate'] else ''}", flush=True)
        pts = []
        for d in targets:
            z = z.copy()
            z[-1] = d
            z, nF, hist, low = lm_fixed(S, z, maxjac, int(opt["settle"]), opt["mutate"] == "dfree", t0)
            held = float(z[-1]) == d
            ok = ok and held
            pts.append(dict(target=d, Delta=float(z[-1]), F=nF, jac=len(hist), held=held, s_low=low[:3], smax=low[3]))
            print(f"  Delta {d:.6f}: least |F| {nF:.4e} after {len(hist)} Jacobians; smallest singular values "
                  f"{', '.join(f'{x:.2e}' for x in low[:3])} of {low[3]:.2e}; Delta held {held} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        rec.update(points=pts)
        best = min(pts, key=lambda p: p["F"])
        print(f"sigdss_band profile: least |F| {best['F']:.4e} at Delta {best['target']:.6f}; every Delta held: {ok}",
              flush=True)
    else:
        hist, stalled = [], False
        nF0 = float(np.linalg.norm(S.residual(z[None, :])[0]))
        for it in range(int(opt["maxjac"])):
            zn, nF, _ = lm(S, z, tol=0.0, maxjac=1, verbose=False, stall=10 ** 9)
            dD = float(zn[-1] - z[-1])
            hist.append(dict(jac=it, F=float(nF), Delta=float(zn[-1]), dDelta=dD))
            print(f"  jac {it}: |F| {nF:.4e} Delta {zn[-1]:.12f} change {dD:+.3e} ({time.time() - t0:.0f} s)", flush=True)
            # a Jacobian that does not lower |F| (a step of rounding size, or none) is a stall, not a settled Delta
            stalled = np.array_equal(zn, z) or not nF < nF0
            if stalled:
                print("  no decrease of |F|: stalled", flush=True)
                break
            z, nF0 = zn, float(nF)
            # settled: the last --settle Jacobians each moved Delta by less than --dtol (one short LM step is not enough:
            # #892 stopped on +2.7e-10 while Delta fell 2.3e-5 per Jacobian)
            quiet = len(hist) >= int(opt["settle"]) and all(abs(h["dDelta"]) < float(opt["dtol"])
                                                             for h in hist[-int(opt["settle"]):])
            if quiet:
                break
        rec.update(hist=hist, Delta=float(z[-1]), F=hist[-1]["F"] if hist else None, stalled=bool(stalled),
                   settled=bool(hist) and not stalled and len(hist) >= int(opt["settle"])
                   and all(abs(h["dDelta"]) < float(opt["dtol"]) for h in hist[-int(opt["settle"]):]))
        print(f"sigdss_band walk {spec}: {len(hist)} Jacobians, Delta {z[-1]:.12f}, |F| {rec['F']:.4e}, "
              f"{'settled' if rec['settled'] else 'stalled' if stalled else 'NOT settled'} (dtol {opt['dtol']})",
              flush=True)
        if opt["out"]:
            np.save(opt["out"], z)
    if opt["json"]:
        with open(opt["json"], "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
