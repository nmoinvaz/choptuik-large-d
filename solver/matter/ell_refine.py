"""Refine continuation solutions at higher resolution and compute the growth rate.

For each D: take data/sol_{src}_D{D}.npy (kc_src; waits for it), regrid to kc, re-solve with RK stiffness target kappa,
save data/sol_{tag}_D{D}.npy, and find the physical growth rate (root of det M(lambda), bracketed around the value
extrapolated from the previous points, refined by brentq).
usage: python ell_refine.py src kc_src tag kc l "D1,D2,..." lam0 [--kappa 0.5] [--rk 1] [--no-gamma] [--tol 1e-11]
Records: data/{tag}.jsonl."""
import os, sys, json, time, argparse
import numpy as np
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD, lm

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("kc_src", type=int); ap.add_argument("tag"); ap.add_argument("kc", type=int)
ap.add_argument("l", type=int); ap.add_argument("Ds"); ap.add_argument("lam0", type=float)
ap.add_argument("--kappa", type=float, default=0.5); ap.add_argument("--rk", type=int, default=1)
ap.add_argument("--Ks", type=int, default=16); ap.add_argument("--u0", type=float, default=0.01)
ap.add_argument("--no-gamma", action="store_true"); ap.add_argument("--tol", type=float, default=1e-11)
ap.add_argument("--wait", type=float, default=7000)
ap.add_argument("--reg", type=float, default=0.0)
a = ap.parse_args()
out = os.path.join(HERE, "data", f"{a.tag}.jsonl")
hist = []
if os.path.exists(out):
    for line in open(out):
        r = json.loads(line)
        if r.get("lam"):
            hist.append((r["D"], r["lam"]))


def lam_root(S, z, guess):
    cache = {}

    def sd(l):
        if l not in cache:
            cache[l] = np.linalg.slogdet(S.mode_matrix(z, l))
        return cache[l]
    for w in (0.02, 0.06, 0.15, 0.35):
        grid = guess * (1 + w * np.array([-1.0, 0.0, 1.0]))
        sg = [sd(l)[0] for l in grid]
        ch = [i for i in range(2) if sg[i] != sg[i + 1]]
        if ch:
            i = ch[0] if len(ch) == 1 else min(ch, key=lambda i: abs(0.5 * (grid[i] + grid[i + 1]) - guess))
            ref = 0.5 * (sd(grid[i])[1] + sd(grid[i + 1])[1])
            r = brentq(lambda l: sd(l)[0] * np.exp(sd(l)[1] - ref), grid[i], grid[i + 1], xtol=2e-8 * guess, rtol=1e-12)
            return r, len(cache), len(ch)
    return None, len(cache), 0


done = {json.loads(l)["D"] for l in open(out)} if os.path.exists(out) else set()
skipf = os.path.join(HERE, "data", f"{a.tag}_skip.txt")      # optional: D values to leave out (edited by hand)
for D in [float(v) for v in a.Ds.split(",")]:
    skip = {float(v) for v in open(skipf).read().split()} if os.path.exists(skipf) else set()
    if D in done or D in skip:
        continue
    f = os.path.join(HERE, "data", f"sol_{a.src}_D{D:.4f}.npy")
    tw = time.time()
    while not os.path.exists(f):
        if time.time() - tw > a.wait:
            print(f"gave up waiting for {f}", flush=True); sys.exit(0)
        time.sleep(30)
    time.sleep(2)
    z0 = np.load(f)
    t0 = time.time()
    mk = lambda kc, z: DSSEll(kc=kc, nodes=NODES_STD, n=D - 2, ell=a.l, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0, rk=a.rk)
    S = mk(a.kc, z0)
    z = S.regrid_from(z0, mk(a.kc_src, z0)) if a.kc != a.kc_src else z0
    S.freeze_from(z, a.kappa)
    F0 = float(np.linalg.norm(S.residual(z[None, :])[0]))
    z, nF, _ = lm(S, z, tol=a.tol, maxjac=12, verbose=False, prep=lambda zz: S.refreeze(zz, a.kappa),
                  reg=a.reg)
    S = mk(a.kc, z); steps = S.freeze_from(z, a.kappa)
    z, nF, _ = lm(S, z, tol=a.tol, maxjac=4, verbose=False, reg=a.reg)
    rec = dict(D=D, l=a.l, kc=a.kc, kappa=a.kappa, rk=a.rk, reg=a.reg, u0=a.u0, Ks=a.Ks, src=a.src, Delta=float(z[-1]), Delta_src=float(z0[-1]),
               F=float(nF), F_start=F0, steps=int(sum(steps.values())), constraint=S.constraint(z),
               lam_gauge=S.gauge_lambda(z))
    np.save(os.path.join(HERE, "data", f"sol_{a.tag}_D{D:.4f}.npy"), z)
    if not a.no_gamma and nF < 1e-8:
        if len(hist) >= 2:
            (D1, l1), (D2, l2) = hist[-2], hist[-1]
            guess = l2 + (l2 - l1) * (D - D2) / (D2 - D1)
        elif hist:
            guess = hist[-1][1]
        else:
            guess = a.lam0
        lam, nev, nch = lam_root(S, z, guess)
        rec.update(lam_guess=guess, lam=lam, gamma=(1 / lam if lam else None), n_eval=nev, n_signchanges=nch)
        if lam:
            hist.append((D, lam))
    rec["seconds"] = round(time.time() - t0)
    print(json.dumps(rec), flush=True)
    open(out, "a").write(json.dumps(rec) + "\n")
