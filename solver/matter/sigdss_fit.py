"""Fits of lambda(D) = 1/gamma(D) in powers of 1/n (n = D - 2) for the sigma-model DSS branch, with lambda_inf free and
with lambda_inf = 2 imposed, and the same for Delta(D).

    lambda = lam_inf + sum_{k=1..m} c_k n^-k          (free: m + 1 parameters)
    lambda = 2       + sum_{k=1..m+1} c_k n^-k        (fixed, same number of parameters)

usage: python sigdss_fit.py FILE.jsonl [FILE2.jsonl ...] --eps EPS [--kc KC] [--Dmin a,b,..] [--mmax 5] [--key lam]
Takes, for each D, the record with the largest kc (or the given kc) that has the key. Prints a table and writes
data/sigdss_fit_eta{eta}.json."""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+"); ap.add_argument("--eps", type=float, required=True)
ap.add_argument("--kc", type=int, default=None); ap.add_argument("--Dmin", default="4,5,6,7,8")
ap.add_argument("--Dmax", type=float, default=1e9)
ap.add_argument("--mmax", type=int, default=5); ap.add_argument("--key", default="lam")
ap.add_argument("--out", default=None)
ap.add_argument("--plant", default=None,
                help="planted-truth control: 'lam_inf,c1,c2,...' replaces the data by lam_inf + sum c_k n^-k on the same D")
ap.add_argument("--noise", type=float, default=0.0, help="with --plant: Gaussian noise of this size (seed 1)")
a = ap.parse_args()

pts = {}
for f in a.files:
    for line in open(f):
        r = json.loads(line)
        if abs(r.get("eps", -1) - a.eps) > 1e-3 or r.get(a.key) is None:
            continue
        if a.key == "lam" and r.get("last_step") is not None and r["last_step"] > 1e-3:
            continue                        # first (secant) root finder stopped at maxit: not converged
        if a.kc and r["kc"] != a.kc:
            continue
        D = round(r["D"], 6)
        if D > a.Dmax:
            continue
        if D not in pts or r["kc"] > pts[D]["kc"]:
            pts[D] = r
Ds = np.array(sorted(pts))
y = np.array([pts[D][a.key] for D in Ds])
n = Ds - 2
if a.plant:
    cp = [float(v) for v in a.plant.split(",")]
    y = sum(c * n ** -k for k, c in enumerate(cp)) + a.noise * np.random.default_rng(1).standard_normal(len(n))
    print(f"PLANTED: {a.key} = " + " + ".join(f"{c:g} n^-{k}" for k, c in enumerate(cp)) + f", noise {a.noise:g}")
print(f"eps = {a.eps}: {len(Ds)} points, D = {Ds.min()}..{Ds.max()}")
for D, v in zip(Ds, y):
    print(f"  D {D:8.4f}  {a.key} {v:.10f}  kc {pts[D]['kc']}")
out = dict(eps=a.eps, key=a.key, D=Ds.tolist(), y=y.tolist(), fits=[])


def fit(nn, yy, m, fixed):
    if fixed is None:
        A = np.stack([nn ** -k for k in range(0, m + 1)], 1)
        c, *_ = np.linalg.lstsq(A, yy, rcond=None)
        return c, A @ c
    A = np.stack([nn ** -k for k in range(1, m + 2)], 1)
    c, *_ = np.linalg.lstsq(A, yy - fixed, rcond=None)
    return np.concatenate([[fixed], c]), fixed + A @ c


print(f"\n{'Dmin':>6} {'pts':>4} {'m':>2} | {'free: inf':>12} {'rms':>9} {'c1':>9} | {'fixed 2: rms':>12} {'c1':>9}")
for Dmin in [float(v) for v in a.Dmin.split(",")]:
    sel = Ds >= Dmin - 1e-9
    for m in range(1, a.mmax + 1):
        if sel.sum() < m + 3:
            continue
        cf, yf = fit(n[sel], y[sel], m, None)
        cx, yx = fit(n[sel], y[sel], m, 2.0 if a.key == "lam" else None)
        rf = float(np.sqrt(np.mean((yf - y[sel]) ** 2)))
        rx = float(np.sqrt(np.mean((yx - y[sel]) ** 2)))
        print(f"{Dmin:6.2f} {int(sel.sum()):4d} {m:2d} | {cf[0]:12.6f} {rf:9.2e} {cf[1]:9.4f} | {rx:12.2e} {cx[1]:9.4f}")
        out["fits"].append(dict(Dmin=Dmin, npts=int(sel.sum()), m=m, free_inf=float(cf[0]), free_rms=rf,
                                free_coef=cf.tolist(), fixed_rms=rx, fixed_coef=cx.tolist()))
eta = "inf" if a.eps == 0 else f"{1 / a.eps:g}"
fn = a.out or os.path.join(HERE, "data", f"sigdss_fit_{a.key}_eta{eta}.json")
json.dump(out, open(fn, "w"), indent=1)
print("written", fn)
