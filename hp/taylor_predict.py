#!/usr/bin/env python3
"""Measurement C of notes/QUAD_DOUBLE.md 13.13: a high-degree Taylor predictor built from the stored points of one
layout, tested where the cubic predictor failed. The state z(D) is fitted over T75's window (D 70.63-73.27, Chebyshev
degree 16) and extrapolated to D_t = D_last + h; at each extrapolated state the kernel's horizon series' radius
(12.4's ratio test), the residual, and the double-double Newton of the series block (the production newton_solve,
kfreeze 13) are run. The cubic Lagrange predictor through the last four stored points is the control at the same h.
Delta is compared with the rational formula (solver/formula_all.delta).

    cd hp && python3 taylor_predict.py RAWDIR OUT.jsonl [--folder T75] [--degree 16] [--hs 0.5,1,2,3,4,5] [--threads N]
"""
import glob
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from numpy.polynomial import chebyshev as C  # noqa: E402
from hpkernel.driver import make_solver, newton_solve, rnorm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "solver"))
from pole_test import ratio_radius  # noqa: E402
# the rational formula of README "The formula" (solver/data/formula_coefficients.json, as tabulated there)
_C = json.load(open(os.path.join(HERE, "..", "solver", "data", "formula_coefficients.json")))
_A = [float(v) for v in _C["a"]]
_B = [1.0] + [float(v) for v in _C["b"]]


def formula_delta(D):
    e = 1.0 / (D - 2)
    return sum(c * e ** k for k, c in enumerate(_A)) / sum(c * e ** k for k, c in enumerate(_B))


def load_window(raw, folder):
    paths = sorted(glob.glob(os.path.join(raw, folder, "*.npz")))
    Ds, Z, layout, meta0 = [], [], None, None
    for p in paths:
        d = np.load(p); m = json.loads(str(d["meta"]))
        key = (m["M"], tuple(m["nodes"]), len(d["zh"]))
        if layout is None:
            layout, meta0 = key, m
        if key != layout:
            continue
        Ds.append(m["D"]); Z.append(d["zh"] + d["zl"])
    Ds = np.array(Ds); Z = np.array(Z)
    o = np.argsort(Ds)
    return Ds[o], Z[o], meta0


def main():
    args = list(sys.argv[1:])
    folder, degree, hs, threads = "T75", 16, (0.5, 1.0, 2.0, 3.0, 4.0, 5.0), 2
    for flag, conv in (("--folder", str), ("--degree", int), ("--threads", int)):
        if flag in args:
            i = args.index(flag); v = conv(args[i + 1]); del args[i:i + 2]
            folder, degree, threads = (v if flag == "--folder" else folder), (v if flag == "--degree" else degree), (v if flag == "--threads" else threads)
    if "--hs" in args:
        i = args.index("--hs"); hs = tuple(float(x) for x in args[i + 1].split(",")); del args[i:i + 2]
    raw, out = args[0], args[1]
    Ds, Z, m = load_window(raw, folder)
    M, nodes = m["M"], list(m["nodes"])
    lo, hi = Ds[0], Ds[-1]; mid, half = (lo + hi) / 2, (hi - lo) / 2
    V = C.chebvander((Ds - mid) / half, degree)
    coef = np.linalg.lstsq(V, Z, rcond=None)[0]
    D_last = Ds[-1]
    print(f"{len(Ds)} states, D {lo:.4f}..{hi:.4f}, M {M}, nodes {nodes}; Chebyshev degree {degree}; last state D {D_last:.6f} Delta {Z[-1][-1]:.12f}", flush=True)
    kf = 13 if D_last >= 35 else None
    # one segment plan, from the last converged state: a predicted state whose apparent horizon lies inside
    # the domain (section 13.6) has no convergent plan, and the kernel's planner then never returns
    s0 = make_solver(D_last, M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130), threads=threads, kfreeze=kf)
    z0 = np.ascontiguousarray(Z[-1], dtype=float)
    segs0 = s0.plan(z0, np.zeros_like(z0), 1e-32, 0.05)
    print("segment plan taken from the last state", flush=True)
    with open(out, "a") as f:
        for h in hs:
            Dt = D_last + h
            preds = {"taylor16": C.chebval((Dt - mid) / half, coef)}
            # the control: the cubic Lagrange predictor through the last four states
            x4, z4 = Ds[-4:], Z[-4:]
            w = np.array([np.prod([(Dt - x4[j]) / (x4[i] - x4[j]) for j in range(4) if j != i]) for i in range(4)])
            preds["cubic"] = (w[:, None] * z4).sum(axis=0)
            for name, zp in preds.items():
                t0 = time.time()
                zh = np.ascontiguousarray(zp, dtype=float); zl = np.zeros_like(zh)
                s = make_solver(Dt, M, nodes=nodes, kt=m.get("kt", 36), kc=m.get("kc", 146), ks=m.get("ks", 130), threads=threads, kfreeze=kf)
                rec = dict(folder=folder, degree=degree, D_last=float(D_last), h=h, D=float(Dt), predictor=name)
                try:
                    s.set_segs(segs0)
                    hh, hl = s.series(zh, zl, 1, True)
                    c = (np.asarray(hh) + np.asarray(hl)).reshape(-1, 4, M)
                    R, _ = ratio_radius(c, 1 - nodes[-1], s.series_scale(1))
                    Fh, Fl = s.residual(zh, zl)
                    F0 = float(rnorm(Fh, Fl))
                    rec.update(R_pred=R, F_pred=F0, Delta_pred=float(zh[-1]))
                    print(f"h {h:.1f} (D {Dt:.4f}) {name:9s}: radius at the predicted state {R if R else float('nan'):.3f}, |F| {F0:.1e}", flush=True)
                    if np.isfinite(F0):
                        zh2, zl2, hist = newton_solve(s, zh, zl, tol=1e-25, it_max=12, verbose=False)
                        Fh, Fl = s.residual(zh2, zl2)
                        Ff = float(rnorm(Fh, Fl))
                        fd = float(formula_delta(Dt)) if formula_delta else None
                        rec.update(converged=bool(Ff <= 1e-25), F_final=Ff, steps=len(hist), Delta=float(zh2[-1] + zl2[-1]),
                                   Delta_formula=fd, dDelta_formula=(float(zh2[-1] + zl2[-1]) - fd) if fd else None, seconds=time.time() - t0)
                        print(f"      Newton: {len(hist)} steps, |F| {F0:.1e} -> {Ff:.1e}, {'CONVERGED' if rec['converged'] else 'not converged'}; "
                              f"Delta {rec['Delta']:.12f}" + (f" (formula {fd:.6f}, diff {rec['dDelta_formula']:+.1e})" if fd else "") + f"; {time.time() - t0:.0f} s", flush=True)
                    else:
                        rec.update(converged=False, note="residual not finite at the predicted state", seconds=time.time() - t0)
                        print("      residual not finite: no Newton", flush=True)
                except Exception as ex:  # noqa: BLE001
                    rec.update(converged=False, note=f"{type(ex).__name__}: {ex}", seconds=time.time() - t0)
                    print(f"      {type(ex).__name__}: {ex}", flush=True)
                f.write(json.dumps(rec) + "\n"); f.flush()


if __name__ == "__main__":
    main()
