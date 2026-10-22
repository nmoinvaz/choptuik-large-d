"""Growth rate of the physical mode along a D-scan of saved solutions (data/sol_{tag}_D{D}.npy), run alongside the
continuation (waits for each file). The root of det M(lambda) is bracketed around the value extrapolated from the
previous points (3-point brackets of width 2%, 6%, 15%, 35%) and refined by brentq.
usage: python ell_gamma_series.py tag l kc "D1,D2,..." lam0 [--Ks 16] [--u0 0.01] [--rk 1] [--kappa 0.5] [--N N]
Writes data/{tag}_gamma.jsonl."""
import os, sys, json, time, argparse
import numpy as np
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dss_ell import DSSEll, NODES_STD

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("l", type=int); ap.add_argument("kc", type=int); ap.add_argument("Ds")
ap.add_argument("lam0", type=float)
ap.add_argument("--Ks", type=int, default=16); ap.add_argument("--u0", type=float, default=0.01)
ap.add_argument("--rk", type=int, default=1); ap.add_argument("--kappa", type=float, default=0.5)
ap.add_argument("--N", type=int, default=None); ap.add_argument("--wait", type=float, default=7200)
a = ap.parse_args()
out = os.path.join(HERE, "data", f"{a.tag}_gamma.jsonl")
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
            r = brentq(lambda l: sd(l)[0] * np.exp(sd(l)[1] - ref), grid[i], grid[i + 1], xtol=1e-9 * guess, rtol=1e-12)
            return r, len(cache), len(ch)
    return None, len(cache), 0


for D in [float(v) for v in a.Ds.split(",")]:
    if any(abs(D - h[0]) < 1e-9 for h in hist):
        continue
    f = os.path.join(HERE, "data", f"sol_{a.tag}_D{D:.4f}.npy")
    t_wait = time.time()
    while not os.path.exists(f):
        if time.time() - t_wait > a.wait:
            print(f"gave up waiting for {f}", flush=True); sys.exit(0)
        time.sleep(30)
    time.sleep(2)
    z = np.load(f)
    t0 = time.time()
    S = DSSEll(kc=a.kc, nodes=NODES_STD, n=D - 2, ell=a.l, Delta_ref=z[-1], Ks=a.Ks, u0=a.u0, rk=a.rk, N=a.N)
    S.freeze_from(z, a.kappa)
    if len(hist) >= 2:
        (D1, l1), (D2, l2) = hist[-2], hist[-1]
        guess = l2 + (l2 - l1) * (D - D2) / (D2 - D1)
    elif hist:
        guess = hist[-1][1]
    else:
        guess = a.lam0
    lam, nev, nch = lam_root(S, z, guess)
    rec = dict(D=D, l=a.l, kc=a.kc, rk=a.rk, kappa=a.kappa, Delta=float(z[-1]), lam_guess=guess, lam=lam,
               gamma=(1 / lam if lam else None), n_eval=nev, n_signchanges=nch, lam_gauge=S.gauge_lambda(z),
               seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True)
    open(out, "a").write(json.dumps(rec) + "\n")
    if lam:
        hist.append((D, lam))
