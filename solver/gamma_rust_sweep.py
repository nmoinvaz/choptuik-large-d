"""gamma(D) for every reliable solution (D <= 25) with the Rust growth-mode matrix."""
import os, json, time, numpy as np, warnings
from scipy.optimize import brentq
warnings.filterwarnings("ignore")
from spec_long import make_long
from spec_c import freeze_steps
from rustkernel import RustResidual
from rust_solver import NODES_L
pts = {}
for fn, pref in (("scan_long_kc23.jsonl", "long_D"), ("scan_long_pol.jsonl", "longP_D"), ("scan_longB.jsonl", "longB_D")):
    for l in open(fn):
        r = json.loads(l); D = r["D"]
        if D > 25.0: continue
        f = f"{pref}{D:.4f}.npy"
        if os.path.exists(f): pts[round(D, 4)] = (D, f)
out = open("gamma_rust.jsonl", "w"); hist = []
for k in sorted(pts):
    D, f = pts[k]; z = np.load(f); t0 = time.time()
    S = freeze_steps(make_long(23, D - 2, nodes=NODES_L, Ks=60), z[-1]); R = RustResidual(S, threads=2)
    g0 = hist[-1][1] + (hist[-1][1] - hist[-2][1]) * (D - hist[-1][0]) / (hist[-1][0] - hist[-2][0]) if len(hist) >= 2 else 2.2386
    cache = {}
    def sd(l):
        if l not in cache: cache[l] = np.linalg.slogdet(R.mode_matrix(z, l))
        return cache[l]
    lam = None
    for w in (0.003, 0.01, 0.03):
        grid = np.linspace(g0 - w, g0 + w, 5); sg = [sd(l)[0] for l in grid]
        ch = [i for i in range(4) if sg[i] != sg[i + 1]]
        if ch:
            i = min(ch, key=lambda i: abs(grid[i] - g0)); ref = 0.5 * (sd(grid[i])[1] + sd(grid[i + 1])[1])
            lam = brentq(lambda l: sd(l)[0] * np.exp(sd(l)[1] - ref), grid[i], grid[i + 1], xtol=1e-11, rtol=1e-14); nch = len(ch); break
    rec = dict(D=D, lam=lam, gamma=1 / lam if lam else None, n_signchanges=nch if lam else 0, Delta=float(z[-1]), source=f, seconds=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
    if lam: hist.append((D, lam))
