"""Continuation in D of form S with the Moebius horizon variable in double-double (mobius_sdd.py) (#1090).

Per D: a seed extrapolated in 1/n from the previous two solutions (p1 sqrt(n), Q0 sqrt(n), (c0 - 1) n, Delta), a
float64 pre-solve (mobius_s.py with lsnewton64.py), then the split Newton of lsnewton.py (Track B) in double-double with
the exact forward-mode Jacobian. Recorded per D (one JSON with hi and lo parts): max|F|, Delta, the LS history, the
scaled singular values and the grip |e_Delta^T J^-1| (unit rows) at the solution, the series tails and the times.

Usage: python3 solve_dd_cont.py START.json --M=10 --Ds=100,110,... --out=DIR [--a=1] [--kb=150] [--tau=1e-8]
                                [--outer=6] [--rounds=2] [--tol=1e-27] [--ks_extra=0] [--pre=newton|ls]
                                [--pre_iters=4]
  START.json: a form-S json with hi/lo parts (Track B's Sdd_D100_M10.json, or a file written here).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ddnp as d  # noqa: E402
from ddnp import DD  # noqa: E402
import mobius_s as ms  # noqa: E402
import mobius_sdd as msd  # noqa: E402
from lsnewton import ls_solve, scaled_svd  # noqa: E402
from lsnewton64 import ls_solve as ls_solve64  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
start = [a for a in sys.argv[1:] if not a.startswith("--")][0]
M = int(opts.get("M", 10))
Ds = [float(v) for v in opts["Ds"].split(",")]
out = opts["out"]
a = float(opts.get("a", 1.0))
kb = int(opts.get("kb", 150))
tau = float(opts.get("tau", 1e-8))
tol = float(opts.get("tol", 1e-27))
rounds = int(opts.get("rounds", 2))
os.makedirs(out, exist_ok=True)
logf = open(os.path.join(out, "solve_dd_cont.log"), "a")


def say(s):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


js = json.load(open(start))
D0, M0 = js["D"], js["M"]
get = lambda k: np.array(js[k]) + np.array(js.get(k + "_lo", [0.0] * len(js[k])))  # noqa: E731
if M0 == M:
    yh = np.concatenate([js["p1"], js["c0"], js["Q0"], [js["Delta"]]])
    yl = np.concatenate([js.get("p1_lo", [0] * M), js.get("c0_lo", [0] * M), js.get("Q0_lo", [0] * M), [js.get("Delta_lo", 0)]])
else:
    yh = np.concatenate([ms.resample_odd(get("p1"), M), ms.resample_even(get("c0"), M), ms.resample_odd(get("Q0"), M),
                         [js["Delta"] + js.get("Delta_lo", 0.0)]])
    yl = np.zeros_like(yh)


def scaled(y, D):
    n = D - 2
    return np.concatenate([y[:M] * np.sqrt(n), (y[M:2 * M] - 1) * n, y[2 * M:3 * M] * np.sqrt(n), [y[3 * M]]])


def unscaled(s, D):
    n = D - 2
    return np.concatenate([s[:M] / np.sqrt(n), 1 + s[M:2 * M] / n, s[2 * M:3 * M] / np.sqrt(n), [s[3 * M]]])


say("# %s  dd continuation  M %d  a %g  kb %d  tau %.0e  tol %.0e  start %s (D %g, M %d)"
    % (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), M, a, kb, tau, tol, os.path.basename(start), D0, M0))
hist = []
queue = list(Ds)
refine = {}
while queue:
    D = queue.pop(0)
    t0 = time.time()
    ks = None if "ks_extra" not in opts else int((D - 2) + 14 * np.sqrt(D - 2) + 80) + int(opts["ks_extra"])
    if len(hist) >= 2:
        (Da, sa), (Db, sb) = hist[-2], hist[-1]
        xa, xb, x = 1 / (Da - 2), 1 / (Db - 2), 1 / (D - 2)
        y = DD(unscaled(sb + (sb - sa) * (x - xb) / (xb - xa), D))
    elif len(hist) == 1:
        y = DD(unscaled(hist[-1][1], D))
    else:
        y = DD(yh, yl) if abs(D - D0) < 1e-12 else DD(unscaled(scaled(yh + yl, D0), D))
    # float64 pre-solve, only when the seed is not already at the float64 floor: damped Newton steps that are kept only
    # when they lower max|F|, then (--pre=ls) the float64 split Newton; at M >= 10 the weakest directions lie below
    # float64, so the split Newton there is optional and guarded
    g = ms.FormS(D, M, a=a, kb=kb, ks=ks)
    with np.errstate(all="ignore"):
        y64 = y.f()
        r64 = float(np.max(np.abs(g.residual(y64))))
        if r64 > 1e-12 and np.isfinite(r64):
            for it in range(int(opts.get("pre_iters", 4))):
                try:
                    F64 = g.residual(y64)
                    dy = np.linalg.lstsq(g.jacobian(y64), -F64, rcond=None)[0]
                except np.linalg.LinAlgError:
                    break
                lam, done = 1.0, False
                for _ in range(10):
                    Ft = g.residual(y64 + lam * dy)
                    if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < np.max(np.abs(F64)):
                        y64, done = y64 + lam * dy, True
                        break
                    lam *= 0.5
                if not done:
                    break
            if opts.get("pre", "newton") == "ls":
                try:
                    y64, _ = ls_solve64(g.residual, y64, g.jacobian(y64), tau=1e-6, tol=1e-15, log=lambda s: None)
                except np.linalg.LinAlgError:
                    pass
            r64n = float(np.max(np.abs(g.residual(y64))))
            if np.isfinite(r64n) and r64n < r64:
                y, r64 = DD(y64), r64n
    say("D %.2f: float64 pre-solve max|F| %.2e Delta %.12f (%.0f s)" % (D, r64, y.f()[3 * M], time.time() - t0))
    f = msd.FormSDD(D, M, a=a, kb=kb, ks=ks)
    lsh = []
    F = f.residual(y)
    for rnd in range(rounds):
        if float(np.max(np.abs(F.f()))) < tol:
            break
        J = f.jacobian(y)
        y, F, h1, S = ls_solve(f.residual, y, J, tau=tau, outer=int(opts.get("outer", 6)), log=lambda s: say("    " + s), tol=tol)
        lsh += h1
    nF = float(np.max(np.abs(F.f())))
    J = f.jacobian(y)
    r, c, U, Ssol, Vt = scaled_svd(J)
    Jf = J.f() / r[:, None]
    ev = np.zeros(3 * M + 1)
    ev[3 * M] = 1.0
    grip = float(np.linalg.norm(np.linalg.lstsq(Jf.T, ev, rcond=None)[0]))
    ce, ho = f.last
    rec = dict(D=D, n=D - 2, M=M, a=a, ks=f.ks, kc=f.kc, kb=kb, maxF=nF, Delta=float(y.h[3 * M]), Delta_lo=float(y.l[3 * M]),
               ls_hist=lsh, sv_scaled=[float(v) for v in Ssol], grip=grip, tail_h=float(ho["tail"]), tail_c=float(ce["tail"]),
               f64_maxF=r64, seconds=time.time() - t0,
               p1=list(y.h[:M]), c0=list(y.h[M:2 * M]), Q0=list(y.h[2 * M:3 * M]),
               p1_lo=list(y.l[:M]), c0_lo=list(y.l[M:2 * M]), Q0_lo=list(y.l[2 * M:3 * M]))
    json.dump(rec, open(os.path.join(out, "sdd_D%g_M%d_a%g.json" % (D, M, a)), "w"), indent=1)
    say("D %8.2f  max|F| %.2e  Delta %.16f (%+.1e)  sv_min %.1e  grip %.1e  tails %.0e/%.0e  ks %d  %.0f s"
        % (D, nF, rec["Delta"], rec["Delta_lo"], Ssol[-1], grip, rec["tail_h"], rec["tail_c"], f.ks, rec["seconds"]))
    if nF < 1e-20:
        hist.append((D, scaled(y.f(), D)))
    else:
        Dl = hist[-1][0] if hist else D0
        key = round(D, 6)
        refine[key] = refine.get(key, 0) + 1
        if refine[key] > 3 or not hist:
            say("not converged at D %g after %d refinements; stop" % (D, refine[key] - 1))
            break
        Dm = round(2 + 1 / (0.5 * (1 / (Dl - 2) + 1 / (D - 2))), 2)
        say("not converged at D %g; inserting D %g (midpoint in 1/n)" % (D, Dm))
        queue = [Dm, D] + queue
