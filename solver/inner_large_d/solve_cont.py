"""Continuation in D of form S with the Moebius horizon variable (mobius_s.py), in float64 (#1090).

Each D is solved by Newton with the complex-step Jacobian, from a seed extrapolated in 1/n from the previous two
solutions in the scaled variables p1 sqrt(n), Q0 sqrt(n), (c0 - 1) n and Delta (the first step rescales the start).
At each solution it records max|F|, Delta, the Newton history, the scaled singular values (rows and columns at unit
norm), the grip |e_Delta^T J^-1| with unit rows, the series tails, and the amplification of relative noise in the
horizon sum (a = 1, and a = 0 where --noise0 lists the D).

Usage: python3 solve_cont.py START.json --M=6 --Ds=100,150,... --out=DIR [--a=1] [--tol=5e-15] [--iters=12]
                             [--noise0=100,150,200] [--kb=150] [--accept=1e-11] [--noise_as=1,8,32] [--a_rule=400] [--mutate=a0]
  --mutate=a0 solves with a = 0 (Track B's variable): the control, which must lose the solution as D grows.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mobius_s as ms  # noqa: E402
from lsnewton64 import ls_solve  # noqa: E402

opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
start = [a for a in sys.argv[1:] if not a.startswith("--")][0]
M = int(opts.get("M", 6))
Ds = [float(v) for v in opts["Ds"].split(",")]
out = opts["out"]
a = 0.0 if opts.get("mutate") == "a0" else float(opts.get("a", 1.0))
tol = float(opts.get("tol", 5e-15))
iters = int(opts.get("iters", 3))
kb = int(opts.get("kb", 150))
noise0 = [float(v) for v in opts.get("noise0", "").split(",") if v]
noise_as = [float(v) for v in opts.get("noise_as", "").split(",") if v]  # extra Moebius parameters for the noise test, every D
accept = float(opts.get("accept", 1e-11))
arule = float(opts.get("a_rule", 0))  # with --a_rule=N: a = max(a, n/N) at each D (the coefficients' exponent range)  # max|F| at which a D counts as solved and seeds the next
os.makedirs(out, exist_ok=True)

js = json.load(open(start))
D0, M0 = js["D"], js["M"]
p1 = np.array(js["p1"]) + np.array(js.get("p1_lo", [0] * M0))
c0 = np.array(js["c0"]) + np.array(js.get("c0_lo", [0] * M0))
Q0 = np.array(js["Q0"]) + np.array(js.get("Q0_lo", [0] * M0))
Dl0 = js["Delta"] + js.get("Delta_lo", 0.0)
if M0 != M:
    p1, c0, Q0 = ms.resample_odd(p1, M), ms.resample_even(c0, M), ms.resample_odd(Q0, M)


def scaled(y, D):
    n = D - 2
    return np.concatenate([y[:M] * np.sqrt(n), (y[M:2 * M] - 1) * n, y[2 * M:3 * M] * np.sqrt(n), [y[3 * M]]])


def unscaled(s, D):
    n = D - 2
    return np.concatenate([s[:M] / np.sqrt(n), 1 + s[M:2 * M] / n, s[2 * M:3 * M] / np.sqrt(n), [s[3 * M]]])


def noise_amp(f, y, eta=1e-13, reps=2):
    om = 2 * np.pi / y[3 * M]
    base = f.horizon(y[2 * M:3 * M], om)["U"]
    f.noise = (["U", "V"], eta, np.random.default_rng(7))
    d = max(np.max(np.abs(f.horizon(y[2 * M:3 * M], om)["U"] - base)) for _ in range(reps))
    f.noise = None
    return float(d / eta / np.max(np.abs(base)))


hist = []  # (D, scaled solution)
y = np.concatenate([p1, c0, Q0, [Dl0]])
hist_start = (D0, scaled(y, D0))
log = open(os.path.join(out, "solve_cont.log"), "a")


def say(s):
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


say("# %s  M %d  a %g  kb %d  tol %.1e  start %s (D %g, M %d)" % (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                                                                  M, a, kb, tol, os.path.basename(start), D0, M0))
queue = list(Ds)
refine = {}
while queue:
    D = queue.pop(0)
    t0 = time.time()
    if len(hist) >= 2:
        (Da, sa), (Db, sb) = hist[-2], hist[-1]
        xa, xb, x = 1 / (Da - 2), 1 / (Db - 2), 1 / (D - 2)
        s = sb + (sb - sa) * (x - xb) / (xb - xa)
    elif len(hist) == 1:
        s = hist[-1][1]
    else:
        s = hist_start[1]
    y = unscaled(s, D)
    aD = max(a, (D - 2) / arule) if arule else a
    f = ms.FormS(D, M, a=aD, kb=kb)
    hstory = []
    F = f.residual(y)
    best = (np.max(np.abs(F)), y.copy())
    hstory.append(float(np.max(np.abs(F))))
    with np.errstate(all="ignore"):
        for it in range(iters):
            if not np.all(np.isfinite(F)):
                break
            J = f.jacobian(y)
            dy = np.linalg.lstsq(J, -F, rcond=None)[0]
            lam, ok = 1.0, False
            for _ in range(8):
                yn = y + lam * dy
                Fn = f.residual(yn)
                if np.all(np.isfinite(Fn)) and np.max(np.abs(Fn)) < max(np.max(np.abs(F)), 1e-300) * (1 - 0.1 * lam):
                    ok = True
                    break
                lam *= 0.5
            if not ok:
                break
            y, F = yn, Fn
            r = float(np.max(np.abs(F)))
            hstory.append(r)
            if np.isfinite(r) and r < best[0]:
                best = (r, y.copy())
            if r < tol or (len(hstory) > 3 and r > 0.5 * hstory[-3] and r < 1e-12):
                break
    r, y = best
    if r > tol and np.isfinite(r) and r < 1e-3:
        J = f.jacobian(y)
        try:
            yl, rl = ls_solve(f.residual, y, J, tau=float(opts.get("tau", 1e-6)), tol=tol, log=say)
            if np.isfinite(rl) and rl < r:
                y, r = yl, rl
        except np.linalg.LinAlgError:
            say("    LS: linear algebra failed; kept the Newton point")
        hstory.append(float(r))
    with np.errstate(all="ignore"):
        F = f.residual(y)
        J = f.jacobian(y)
    try:
        if not (np.all(np.isfinite(F)) and np.all(np.isfinite(J))):
            raise np.linalg.LinAlgError("non-finite residual or Jacobian")
        rn = np.linalg.norm(J, axis=1)
        Jr = J / rn[:, None]
        cn = np.linalg.norm(Jr, axis=0)
        sv = np.linalg.svd(Jr / cn[None, :], compute_uv=False)
        grip = float(np.linalg.norm(np.linalg.solve(Jr.T, np.eye(len(y))[3 * M])))
    except np.linalg.LinAlgError as ex:
        say("D %.2f: no finite solution (%s)" % (D, ex))
        Dl = hist[-1][0] if hist else D0
        key = round(D, 6)
        refine[key] = refine.get(key, 0) + 1
        if refine[key] > 4 or not hist:
            say("not converged at D %g after %d refinements; stop" % (D, refine[key] - 1))
            break
        Dm = round(2 + 1 / (0.5 * (1 / (Dl - 2) + 1 / (D - 2))), 2)
        say("not converged at D %g; inserting D %g (midpoint in 1/n)" % (D, Dm))
        queue = [Dm, D] + queue
        continue
    amp1 = noise_amp(f, y)
    amp0 = None
    if D in noise0:
        g0 = ms.FormS(D, M, a=0.0, kb=kb)
        with np.errstate(all="ignore"):
            amp0 = noise_amp(g0, y)
    amp_as = {}
    for aa in noise_as:
        ga = ms.FormS(D, M, a=aa, kb=kb)
        with np.errstate(all="ignore"):
            amp_as["%g" % aa] = noise_amp(ga, y)
    ce, ho = f.last
    rec = dict(D=D, n=D - 2, M=M, a=aD, ks=f.ks, kc=f.kc, kb=kb, maxF=float(np.max(np.abs(F))), Delta=float(y[3 * M]),
               newton=hstory, sv_min=[float(v) for v in sv[-6:]], sv_max=float(sv[0]), grip=grip, noise_amp=amp1,
               noise_amp_a0=amp0, noise_amp_as=amp_as, tail_h=float(ho["tail"]), tail_c=float(ce["tail"]),
               p1_harm=[float(v) for v in ms.harmonics_odd(y[:M])], seconds=time.time() - t0,
               p1=list(y[:M]), c0=list(y[M:2 * M]), Q0=list(y[2 * M:3 * M]))
    json.dump(rec, open(os.path.join(out, "sol_D%g_M%d_a%g.json" % (D, M, aD)), "w"), indent=1)
    say("D %8.2f  max|F| %.2e  Delta %.12f  steps %d  sv_min %.1e  grip %.1e  noise %.1e%s  tails %.0e/%.0e  ks %d  %.0f s"
        % (D, rec["maxF"], rec["Delta"], len(hstory) - 1, sv[-1], grip, amp1,
           ("" if amp0 is None else " (a=0: %.1e)" % amp0) + "".join(" (a=%s: %.1e)" % kv for kv in amp_as.items()), rec["tail_h"], rec["tail_c"], f.ks, rec["seconds"]))
    if rec["maxF"] < accept:
        hist.append((D, scaled(y, D)))
    else:
        Dl = hist[-1][0] if hist else D0
        key = round(D, 6)
        refine[key] = refine.get(key, 0) + 1
        if refine[key] > 4 or not hist:
            say("not converged at D %g after %d refinements; stop" % (D, refine[key] - 1))
            break
        Dm = 2 + 1 / (0.5 * (1 / (Dl - 2) + 1 / (D - 2)))
        Dm = round(Dm, 2)
        say("not converged at D %g; inserting D %g (midpoint in 1/n)" % (D, Dm))
        queue = [Dm, D] + queue
