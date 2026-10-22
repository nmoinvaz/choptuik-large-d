"""Large-D fits of the l-multiplet's Delta(D) and lambda(D) = 1/gamma(D) in powers of 1/n (n = D - 2) and 1/D.

Data: data/l1_kc23r.jsonl (kc 23 refinement and growth rates) plus the D = 4 values (kc 39 Delta, kc 23 lambda).
For lambda: lambda_inf free, and lambda_inf = 2 imposed (the s-translation value of the l = 0 scalar), with the same
number of fitted coefficients; rms residuals and the extrapolations are printed for several fit windows and degrees,
and the spread over the stable fits is the quoted error bar. Writes data/ell_fit.json.
usage: python ell_fit.py [tag=l1_kc23r] [Dmin_list=6,8,10] [maxdeg=5]"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "l1_kc23r"
Dmins = [float(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "6,8,10").split(",")]
maxdeg = int(sys.argv[3]) if len(sys.argv) > 3 else 5

rows = {}
for line in open(os.path.join(HERE, "data", f"{tag}.jsonl")):
    r = json.loads(line)
    rows[r["D"]] = r
D = np.array(sorted(rows))
Dl = np.array([rows[d]["Delta"] for d in D])
lam = np.array([rows[d].get("lam") or np.nan for d in D])
print(f"{len(D)} points, D = {D.min()} .. {D.max()}")
for d, x, l in zip(D, Dl, lam):
    print(f"  D {d:6.2f}  Delta {x:.9f}  lambda {l:.7f}  gamma {1 / l:.7f}" if np.isfinite(l) else f"  D {d:6.2f}  Delta {x:.9f}")


def fit(x, y, deg, fixed=None):
    """y = c0 + c1 x + ... + c_deg x^deg (c0 = fixed if given). Returns coefficients (c0 first), rms."""
    if fixed is None:
        A = np.stack([x ** k for k in range(deg + 1)], 1)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ c
    else:
        A = np.stack([x ** k for k in range(1, deg + 1)], 1)
        c, *_ = np.linalg.lstsq(A, y - fixed, rcond=None)
        res = y - fixed - A @ c
        c = np.concatenate([[fixed], c])
    return c, float(np.sqrt(np.mean(res ** 2)))


out = dict(tag=tag, D=D.tolist(), Delta=Dl.tolist(), lam=lam.tolist(), fits=[])
for var in ("1/n", "1/D"):
    for Dmin in Dmins:
        m = (D >= Dmin) & np.isfinite(lam)
        x = 1 / (D[m] - 2) if var == "1/n" else 1 / D[m]
        npts = int(m.sum())
        print(f"\nvariable {var}, D >= {Dmin} ({npts} points)")
        for deg in range(2, maxdeg + 1):
            if deg + 1 > npts - 1:
                break
            cf, rf = fit(x, lam[m], deg)
            c2, r2 = fit(x, lam[m], deg + 1, fixed=2.0)       # same number of free coefficients
            cd, rd = fit(x, Dl[m], deg)
            print(f"  deg {deg}: lambda_inf free {cf[0]:.5f} (rms {rf:.1e}) | lambda_inf = 2: rms {r2:.1e} "
                  f"(ratio {r2 / rf:.1f}) | Delta_inf {cd[0]:.5f} (rms {rd:.1e})")
            out["fits"].append(dict(var=var, Dmin=Dmin, deg=deg, npts=npts, lam_inf=float(cf[0]), rms_free=rf,
                                    rms_fixed2=r2, Delta_inf=float(cd[0]), rms_Delta=rd,
                                    lam_coef=cf.tolist(), Delta_coef=cd.tolist()))
# comparison with the l = 0 growth rate at the same D (project data: kc 39 RK x2 scan, and the Rust gamma table;
# D = 9 by linear interpolation between 8.0 and 9.048)
SOLVER = os.path.join(HERE, "..")
lam0 = {}
for f in ("data/scan_ne6_lowD_kc39_rk2.jsonl", "data/gamma_rust.jsonl"):
    for line in open(os.path.join(SOLVER, f)):
        r = json.loads(line)
        if r.get("lam") and r["D"] not in lam0:
            lam0[r["D"]] = r["lam"]


def lam0_at(d):
    if d in lam0:
        return lam0[d]
    ks = sorted(lam0)
    lo = max(k for k in ks if k < d); hi = min(k for k in ks if k > d)
    return lam0[lo] + (lam0[hi] - lam0[lo]) * (d - lo) / (hi - lo)


print("\nl = 1 against l = 0 at the same D:")
dl = []
for d, l in zip(D, lam):
    if np.isfinite(l):
        l0 = lam0_at(d); dl.append((d - 2, l - l0))
        print(f"  D {d:5.2f}  lambda_0 {l0:.6f}  lambda_1 - lambda_0 {l - l0:.6f}  n (lambda_1 - lambda_0) {(d - 2) * (l - l0):.4f}")
nn = np.array([x[0] for x in dl]); dd = np.array([x[1] for x in dl])
print("  delta = c + a/n (+ b/n^2) on consecutive windows:")
fits_delta = []
for k in range(2, 4):
    for i in range(len(nn) - k + 1):
        A = np.stack([nn[i:i + k] ** -j for j in range(k)], 1)
        c = np.linalg.solve(A, dd[i:i + k])
        fits_delta.append(dict(n=nn[i:i + k].tolist(), c=float(c[0]), coef=c.tolist()))
        print(f"    n = {nn[i:i + k].tolist()}: c = {c[0]:+.4f}, coefficients {np.round(c, 4).tolist()}")
out["lam0"] = {str(d): lam0_at(d) for d in D}
out["delta_fits"] = fits_delta
json.dump(out, open(os.path.join(HERE, "data", "ell_fit.json"), "w"), indent=1)
