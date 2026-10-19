"""gamma(D) from the complex-step mode matrix: lambda0 = root of det M(lambda) (bracketed scan + brentq),
cross-checked by the regularised sigma_min dip.  usage: gamma_root.py out.jsonl xK Ks "D:file;D:file;..." lam_guess"""
import sys, json, time, numpy as np, warnings
from scipy.optimize import brentq, minimize_scalar
warnings.filterwarnings("ignore")
import fast; fast.install()
from spec_long import make_long
from spec_c import freeze_steps, mode_matrix_cs
from newton_tsvd import harmonic_weights
out, xk, KS = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]); lam_guess = float(sys.argv[5])
nodes = [x for x in [0.02, 0.05, 0.1, 0.16, 0.23, 0.3, 0.4, 0.5, 0.6] if x <= xk + 1e-9]
hist = []
for item in sys.argv[4].split(";"):
    Ds, f = item.split(":"); D = float(Ds); t0 = time.time()
    z = np.load(f); S = freeze_steps(make_long(23, D - 2, nodes=nodes, Ks=KS), z[-1])
    w = harmonic_weights(S, 4)[:-1]
    cache = {}
    def sd(lam):
        if lam not in cache: cache[lam] = np.linalg.slogdet(mode_matrix_cs(S, z, lam))
        return cache[lam]
    if len(hist) >= 2: g0 = hist[-1][1] + (hist[-1][1] - hist[-2][1]) * (D - hist[-1][0]) / (hist[-1][0] - hist[-2][0])
    elif hist: g0 = hist[-1][1]
    else: g0 = lam_guess
    lam = None
    for wdt in (0.004, 0.012, 0.03):
        grid = np.linspace(g0 - wdt, g0 + wdt, 5); sg = [sd(l)[0] for l in grid]
        ch = [i for i in range(4) if sg[i] != sg[i + 1]]
        if ch:
            i = min(ch, key=lambda i: abs(grid[i] - g0)); ref = 0.5 * (sd(grid[i])[1] + sd(grid[i + 1])[1])
            lam = brentq(lambda l: sd(l)[0] * np.exp(sd(l)[1] - ref), grid[i], grid[i + 1], xtol=1e-10, rtol=1e-13)
            nch = len(ch); break
    def smin(l):
        M = mode_matrix_cs(S, z, l); smax = np.linalg.norm(M, 2)
        return np.linalg.svd(np.vstack([M, 1e-4 * smax * np.diag(w)]), compute_uv=False)[-1]
    c = lam if lam else g0
    grid = c + np.linspace(-0.003, 0.003, 7); vals = [smin(l) for l in grid]; i = int(np.argmin(vals))
    i = min(max(i, 1), 5)
    try:
        r = minimize_scalar(smin, bracket=(grid[i - 1], grid[i], grid[i + 1]), tol=1e-7)
    except ValueError:
        r = type("R", (), dict(x=grid[int(np.argmin(vals))], fun=min(vals)))()
    rec = dict(D=D, xK=xk, Ks=KS, lam_det=lam, gamma=(1 / lam if lam else None), n_signchanges=(nch if lam else 0),
               lam_dip=float(r.x), smin_dip=float(r.fun), Delta=float(z[-1]), seconds=round(time.time() - t0))
    print(json.dumps(rec), flush=True); open(out, "a").write(json.dumps(rec) + "\n")
    if lam: hist.append((D, lam))
