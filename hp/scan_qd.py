#!/usr/bin/env python3
"""Continuation in D with the quad-double solver (notes/QUAD_DOUBLE.md, section 12).

    python3 scan_qd.py OUT.jsonl START.npz D_END [--history A.npz,B.npz] [--dD 0.02] [--dD_min 1e-4] [--dD_max 0.5]
        [--grow 1.5] [--kfreeze 13] [--tol 1e-50] [--tol_accept 1e-44] [--it_max 10] [--threads N] [--tag qd]

START.npz and the --history states: solutions in driver.save (two parts) or driver.save_q (four parts) format, in
increasing distance from D_END last. Each is first converged in quad-double at its own D with the full system
(kfreeze), saved and appended to OUT like the scan's points, so the predictor history holds quad-double solutions. Each step predicts by Lagrange extrapolation through
the last four converged points, in quad-double (driver.predict_q; one point: the point itself), plans the Taylor steps
from the prediction (Solver.plan, as scan_hp.py does), and runs driver.newton_qd. A point is converged when max |F| <=
--tol, or when Newton stalls (no decrease along its step: the quad-double floor) with max |F| <= --tol_accept. A
converged point is appended to OUT and saved as <tag>_D<D>.npz (save_q) next to OUT, and dD grows by --grow up to
--dD_max; a failure halves dD, and below --dD_min the scan stops. Every attempt goes to OUT's _attempts.jsonl. OUT
must be in a raw-data folder (raw_out.py; --allow_unraw writes anyway).
"""
import argparse
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from hpkernel import qd_axpy  # noqa: E402
from hpkernel.driver import load_q, make_solver, newton_qd, predict_q, save_q, tangent_q  # noqa: E402
from raw_out import FLAG, HELP, require_raw  # noqa: E402


PLAN = {}        # the Taylor steps in use: D where they were planned, and the segments


def solver_at(D, a, meta):
    return make_solver(D, meta["M"], nodes=list(meta["nodes"]), kt=a.kt or meta.get("kt", 36), kc=a.kc or meta.get("kc", 146),
                       ks=a.ks or meta.get("ks", 130), threads=a.threads, kfreeze=a.kfreeze)


def solve_at(D, z, a, meta):
    """(z, hist, converged, |F|, seconds, Taylor steps, solver) of driver.newton_qd at D from z. The Taylor steps are
    planned from z (Solver.plan) when none are kept or D has moved --replan or more since they were planned; otherwise
    the kept ones are reused, so that the discrete system changes smoothly with D (a new plan changes it by the
    steps' truncation, which moved Delta by 9e-28 at D = 72.8; raw/T81 cont7)."""
    s = solver_at(D, a, meta)
    if not PLAN or abs(D - PLAN["D"]) >= a.replan:
        PLAN.update(D=D, segs=s.plan(z[0], z[1], a.tol_step, a.hmax))
    s.set_segs(PLAN["segs"])
    t0 = time.time()
    z, hist, conv = newton_qd(s, z, tol=a.tol, it_max=a.it_max, verbose=a.verbose)
    f = hist[-1].get("F_new", hist[-1]["F"])
    stalled = not conv and hist[-1].get("lam") == 0.0
    ok = conv or (stalled and f <= a.tol_accept)
    return z, hist, ok, f, round(time.time() - t0, 1), sum(len(g) - 1 for g in s.get_segs()), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("start")
    ap.add_argument("D_end", type=float)
    ap.add_argument("--history", default="")
    ap.add_argument("--dD", type=float, default=0.02)
    ap.add_argument("--dD_min", type=float, default=1e-4)
    ap.add_argument("--dD_max", type=float, default=0.5)
    ap.add_argument("--grow", type=float, default=1.5)
    ap.add_argument("--kfreeze", type=int, default=13)
    ap.add_argument("--kt", type=int, default=0, help="0: the start file's")
    ap.add_argument("--kc", type=int, default=0, help="0: the start file's")
    ap.add_argument("--ks", type=int, default=0, help="0: the start file's")
    ap.add_argument("--tol", type=float, default=1e-50)
    ap.add_argument("--tol_accept", type=float, default=1e-44)
    ap.add_argument("--tol_step", type=float, default=1e-32)
    ap.add_argument("--hmax", type=float, default=0.05)
    ap.add_argument("--replan", type=float, default=0.5, help="replan the Taylor steps when D has moved this much")
    ap.add_argument("--predictor", default="auto", choices=["auto", "lagrange", "tangent"],
                    help="auto: the tangent -J^-1 dF/dD (driver.tangent_q) while fewer than 3 points are converged, "
                    "then Lagrange through the last four")
    ap.add_argument("--it_max", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="qd")
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument(FLAG, action="store_true", help=HELP)
    a = ap.parse_args()
    att = a.out[:-6] + "_attempts.jsonl" if a.out.endswith(".jsonl") else a.out + ".attempts"
    require_raw(a.out, att, allow=a.allow_unraw)
    outdir = os.path.dirname(os.path.abspath(a.out))

    def log(path, rec):
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    hist = []
    meta = None
    for path in [p for p in a.history.split(",") if p] + [a.start]:
        z, m = load_q(path)
        meta = meta or m
        if (m["M"], list(m["nodes"])) != (meta["M"], list(meta["nodes"])):
            raise SystemExit(f"{path}: M and nodes must match the others'")
        print(f"converging {path} at D = {m['D']} in quad-double", flush=True)
        z, h, ok, f, sec, nst, s = solve_at(m["D"], z, a, meta)
        rec = dict(D=m["D"], start=path, ok=ok, F=f, steps=len(h), seconds=sec, Delta=float(z[0][-1] + z[1][-1]),
                   Delta_q=[float(z[q][-1]) for q in range(4)], hist=h)
        log(att, rec)
        if not ok:
            raise SystemExit(f"{path} did not converge in quad-double (|F| {f:.2e})")
        out = os.path.join(outdir, f"{a.tag}_D{m['D']:.6f}.npz")
        save_q(out, z, dict(meta, D=m["D"], kfreeze=a.kfreeze, tol=a.tol, solver="qd"))
        log(a.out, {k: v for k, v in rec.items() if k != "hist"} | dict(state=os.path.basename(out)))
        hist.append((m["D"], z))
    D, dD = hist[-1][0], a.dD
    tan = None
    sign = 1.0 if a.D_end > D else -1.0
    while (a.D_end - D) * sign > 1e-12:
        Dn = D + sign * min(dD, abs(a.D_end - D))
        use_tan = a.predictor == "tangent" or (a.predictor == "auto" and len(hist) < 3)
        if use_tan:
            if tan is None or tan[0] != D:
                tan = (D, tangent_q(s, lambda Dx: solver_at(Dx, a, meta), D, hist[-1][1]))
            zp = qd_axpy(Dn - D, *tan[1], *hist[-1][1])
        else:
            zp = predict_q(hist, Dn)
        z, h, ok, f, sec, nst, s_new = solve_at(Dn, zp, a, meta)
        rec = dict(D=Dn, dD=Dn - D, ok=ok, F=f, steps=len(h), seconds=sec, taylor_steps=nst, plan_D=PLAN["D"],
                   predictor="tangent" if use_tan else "lagrange", Delta=float(z[0][-1] + z[1][-1]),
                   Delta_q=[float(z[q][-1]) for q in range(4)], hist=h)
        log(att, rec)
        print(f"D {Dn:.6f} (dD {Dn - D:+.2e}): {'ok' if ok else 'FAILED'} |F| {f:.2e}, {len(h)} steps, {sec} s, "
              f"Delta {z[0][-1]:.17f} {z[1][-1]:+.6e}", flush=True)
        if ok:
            D, s = Dn, s_new
            hist.append((D, z))
            hist = hist[-6:]
            path = os.path.join(outdir, f"{a.tag}_D{D:.6f}.npz")
            save_q(path, z, dict(meta, D=D, kfreeze=a.kfreeze, tol=a.tol, solver="qd"))
            log(a.out, {k: v for k, v in rec.items() if k != "hist"} | dict(state=os.path.basename(path)))
            dD = min(dD * a.grow, a.dD_max)
        else:
            dD *= 0.5
            if dD < a.dD_min:
                raise SystemExit(f"stopped at D = {D}: dD below {a.dD_min}")
    print(f"reached D = {D}", flush=True)


if __name__ == "__main__":
    main()
