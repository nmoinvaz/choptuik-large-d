#!/usr/bin/env python3
"""How well does a sigdss state fix Delta? The width of Delta at the residual floor (#859).

    python3 solver/matter/sigdss_band.py width STATE --D D --eps EPS [--kc 15] [--json OUT] [--mutate dcol]
    python3 solver/matter/sigdss_band.py walk STATE --D D --eps EPS [--kc 15] [--maxjac 100] [--dtol 1e-9]
        [--out OUT.npy] [--json OUT]

STATE is a sigdss solution (.npy), or NPZ@D for the entry of a sigdss_run.py continuation file (v, Z) at that D.

width: at STATE, sigdss's complex-step Jacobian J (sigdss_solver.jacobian, with the RK steps frozen from STATE as
sigdss_run.py does) and |F| (2-norm). g = |e_Delta^T J^-1|, the largest move of Delta per unit residual, by the SVD
(sqrt(sum_i V[Delta, i]^2 / s_i^2)) and by a solve with J^T. W = g |F| is how far Delta can move at the state's own
residual. Printed beside it: the three smallest singular values with their vectors' Delta part, and Newton's Delta
step -(J^-1 F)_Delta. Check (exit 1): the two g agree to 1e-6 relative. Control: --mutate dcol scales J's Delta column
by 1e-6, which raises g by 1e6 where the Delta column is not near-null; the coordinator's acceptance reads it.

walk: Levenberg-Marquardt (sigdss_solver.lm, the steps frozen once from STATE, no stall stop), one Jacobian at a time,
until Delta moves less than --dtol in one Jacobian or --maxjac Jacobians. Per Jacobian: |F|, Delta, its change.
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


def main(argv):
    if len(argv) < 2 or argv[0] not in ("width", "walk"):
        raise SystemExit(__doc__)
    cmd, spec = argv[0], argv[1]
    opt = dict(D="", eps="", kc="15", json="", mutate="", maxjac="100", dtol="1e-9", out="")
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
    else:
        hist = []
        for it in range(int(opt["maxjac"])):
            zn, nF, _ = lm(S, z, tol=0.0, maxjac=1, verbose=False, stall=10 ** 9)
            dD = float(zn[-1] - z[-1])
            hist.append(dict(jac=it, F=float(nF), Delta=float(zn[-1]), dDelta=dD))
            print(f"  jac {it}: |F| {nF:.4e} Delta {zn[-1]:.12f} change {dD:+.3e} ({time.time() - t0:.0f} s)", flush=True)
            if np.array_equal(zn, z):
                print("  no decrease: stopped", flush=True)
                break
            z = zn
            if abs(dD) < float(opt["dtol"]):
                break
        rec.update(hist=hist, Delta=float(z[-1]), F=hist[-1]["F"] if hist else None, settled=bool(hist) and
                   abs(hist[-1]["dDelta"]) < float(opt["dtol"]))
        print(f"sigdss_band walk {spec}: {len(hist)} Jacobians, Delta {z[-1]:.12f}, |F| {rec['F']:.4e}, "
              f"{'settled' if rec['settled'] else 'NOT settled'} (dtol {opt['dtol']})", flush=True)
        if opt["out"]:
            np.save(opt["out"], z)
    if opt["json"]:
        with open(opt["json"], "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
