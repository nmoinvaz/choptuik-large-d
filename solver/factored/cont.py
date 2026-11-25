"""Form S at a list of D in double-double, each from a predictor built on the previous solutions (#1088).

The predictor is linear in 1/n through the last two solutions in the scaled unknowns sqrt(n) p1, sqrt(n) Q0,
n (c0 - 1) and Delta (with one solution: those held fixed, Delta moved by the 1/D fit of the table). Each D is then
solved by fsolve.py's float64 damped Newton (a few iterations) and lsnewton.ls_solve in double-double with the exact
dual Jacobian formed again before every outer step (--outer 1, up to --rounds rounds). A step that does not reach
--tol is logged as failed and the run stops. This is a continuation in D, used to measure the conditioning at
solutions; a single step of any size is the jump of #1088 Track A.

Usage: python3 cont.py --start=SOL.json --Ds=110,125,150 --out=DIR [--M=8] [--rounds=12] [--tol=1e-24] [--kb=...]
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ddnp import DD  # noqa: E402
from fseries_sdd import FactoredSDD  # noqa: E402
from fsolve import conditioning, solve  # noqa: E402
from lsnewton import ls_solve, scaled_svd  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
M = int(opts.get("M", 8))
Ds = [float(v) for v in opts["Ds"].split(",")]
OUT = opts["out"]
rounds = int(opts.get("rounds", 12))
tol = float(opts.get("tol", 1e-24))
kb = int(opts["kb"]) if "kb" in opts else None
log = lambda s: print(s, flush=True)  # noqa: E731


def load(fn):
    js = json.load(open(fn))
    y = np.concatenate([js["p1"], js["c0"], js["Q0"], [js["Delta"]]])
    yl = np.concatenate([js.get("p1_lo", [0] * M), js.get("c0_lo", [0] * M), js.get("Q0_lo", [0] * M),
                         [js.get("Delta_lo", 0.0)]])
    return js["D"], DD(y, yl)


def scaled(D, y):
    n = D - 2
    v = y.f().copy()
    v[0:M] *= np.sqrt(n)
    v[2 * M:3 * M] *= np.sqrt(n)
    v[M:2 * M] = (v[M:2 * M] - 1) * n
    return v


def unscaled(D, v):
    n = D - 2
    y = v.copy()
    y[0:M] /= np.sqrt(n)
    y[2 * M:3 * M] /= np.sqrt(n)
    y[M:2 * M] = 1 + y[M:2 * M] / n
    return y


hist_sol = [load(opts["start"])]
t0 = time.time()
for D in Ds:
    n = D - 2
    if len(hist_sol) >= 2:
        (Da, ya), (Db, yb) = hist_sol[-2], hist_sol[-1]
        sa, sb = scaled(Da, ya), scaled(Db, yb)
        tt = (1 / n - 1 / (Db - 2)) / (1 / (Db - 2) - 1 / (Da - 2))
        pred = unscaled(D, sb + tt * (sb - sa))
    else:
        Db, yb = hist_sol[-1]
        sb = scaled(Db, yb)
        sb[-1] += 10.37 * (1 / D - 1 / Db) - 49.17 * (1 / D**2 - 1 / Db**2)
        pred = unscaled(D, sb)
    log("== D %.6f: predictor Delta %.12f (%.0f s)" % (D, pred[-1], time.time() - t0))
    f64 = np.concatenate([pred[:3 * M], np.zeros(M), pred[3 * M:]])
    with np.errstate(all="ignore"):
        f64, J64, fz, h64, secs = solve(D, M, f64, iters=int(opts.get("f64iters", 4)), log=lambda s: log("  f64 " + s),
                                        form="S", kb=kb)
    y = DD(np.concatenate([f64[:3 * M], f64[4 * M:]]))
    f = FactoredSDD(D, M, kb=kb)
    lsh = []
    ok = False
    for rnd in range(rounds):
        J = f.jacobian(y)
        y, F, h1, S = ls_solve(f.residual, y, J, tau=float(opts.get("tau", 1e-8)), outer=1, log=lambda s: log("  " + s))
        lsh += h1
        nF = float(np.max(np.abs(F.h)))
        log("  round %d: max|F| %.3e Delta %.16f (%.0f s)" % (rnd, nF, float(y.h[-1]), time.time() - t0))
        if nF < tol:
            ok = True
            break
    J = f.jacobian(y)
    r, c, U, Ssol, Vt = scaled_svd(J)
    Jr = (J.h + J.l) / r[:, None]
    e = np.zeros(3 * M + 1)
    e[-1] = 1.0
    grip = float(np.linalg.norm(np.linalg.lstsq(Jr.T, e, rcond=None)[0]))
    log("D %.6f %s: max|F| %.3e Delta %.16f ; scaled sv max %.3e min %.3e ; smallest %s ; grip %.3e (%.0f s)" % (
        D, "solved" if ok else "NOT solved", nF, float(y.h[-1]), Ssol[0], Ssol[-1], " ".join("%.2e" % v for v in Ssol[-6:]),
        grip, time.time() - t0))
    js = {"D": D, "M": M, "kc": f.kc, "ks": f.ks, "kb": kb, "form": "S-dd", "solved": ok, "p1": list(y.h[0:M]),
          "c0": list(y.h[M:2 * M]), "Q0": list(y.h[2 * M:3 * M]), "Delta": float(y.h[-1]), "p1_lo": list(y.l[0:M]),
          "c0_lo": list(y.l[M:2 * M]), "Q0_lo": list(y.l[2 * M:3 * M]), "Delta_lo": float(y.l[-1]), "maxF": nF,
          "ls_hist": lsh, "sv_scaled": list(map(float, Ssol)), "grip": grip, "predictor_Delta": float(pred[-1]),
          "seconds": time.time() - t0}
    json.dump(js, open("%s/Scont_D%g_M%d.json" % (OUT, D, M), "w"), indent=1)
    if not ok:
        log("stop: D %.6f not solved" % D)
        break
    hist_sol.append((D, y))
