"""D-continuation with the nonlinear-elimination solver (ne_solver.py).

Phase 1 re-solves the stored x_K = 0.6 solutions (longB_D*.npy) from D_from upward; phase 2 continues past the
last one with a cubic predictor up to D_end. For every point: Delta, the residual split (strong / weak / junk),
the weak singular values, and gamma = 1/lambda from the Rust growth-mode matrix.

usage: python3 scan_ne.py out.jsonl D_from D_end h hmax [xK=0.6|0.7] [seed prefixes, comma separated] [kc=23]
Seeds are kc = 23 solutions; for another kc they are regridded. Threads per process: env NE_THREADS (2).
env NE_SEED_MAX: re-solve only the seeds up to this D, then continue from those solutions.
Run from the directory holding gamma_rust.jsonl; seed prefixes may include a directory (../seeds/long7_D).
(D_end below the last seed: re-solve only, no continuation)
"""
import glob, json, os, sys, time, warnings
import numpy as np
from scipy.optimize import brentq
warnings.filterwarnings("ignore")
import fast; fast.install()
from rust_solver import make, NODES_L
from spec_long import make_long
from ne_solver import ne_solve

out, D_from, D_end, h, hmax = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
XK = float(sys.argv[6]) if len(sys.argv) > 6 else 0.6
NODES = NODES_L + ([0.7] if XK > 0.65 else [])
SEED, OUTP = ("long7_D", "ne7_D") if XK > 0.65 else ("longB_D", "neB_D")
PREFIXES = sys.argv[7].split(",") if len(sys.argv) > 7 else [SEED]
KC = int(sys.argv[8]) if len(sys.argv) > 8 else 23
THREADS = int(os.environ.get("NE_THREADS", "2"))
if KC != 23:
    OUTP = OUTP.replace("_D", f"_kc{KC}_D")
_g = [json.loads(l) for l in open("gamma_rust.jsonl")]
_gD = np.array([r["D"] for r in _g if r.get("lam")]); _gl = np.array([r["lam"] for r in _g if r.get("lam")])


def lam_table(D):
    """lambda from the D <= 24.9 Rust gamma sweep, extrapolated linearly from its last points beyond."""
    if D <= _gD[-1]:
        return float(np.interp(D, _gD, _gl))
    c = np.polyfit(_gD[-6:], _gl[-6:], 1)
    return float(np.polyval(c, D))


def gamma_of(R, z, g0):
    cache = {}
    def sd(l):
        if l not in cache:
            cache[l] = np.linalg.slogdet(R.mode_matrix(z, l))
        return cache[l]
    for w in (0.003, 0.01, 0.03):
        grid = np.linspace(g0 - w, g0 + w, 5)
        sg = [sd(l)[0] for l in grid]
        ch = [i for i in range(4) if sg[i] != sg[i + 1]]
        if ch:
            i = min(ch, key=lambda i: abs(grid[i] - g0))
            ref = 0.5 * (sd(grid[i])[1] + sd(grid[i + 1])[1])
            return brentq(lambda l: sd(l)[0] * np.exp(sd(l)[1] - ref), grid[i], grid[i + 1], xtol=1e-11, rtol=1e-14)
    return None


def extrap(Ds, vals, Dt):
    Ds, vals = Ds[-4:], vals[-4:]
    o = 0
    for i in range(len(Ds)):
        w = 1.0
        for j in range(len(Ds)):
            if j != i:
                w *= (Dt - Ds[j]) / (Ds[i] - Ds[j])
        o = o + w * vals[i]
    return o


def solve(D, z0, lam_guess, t0, src, Delta_old=None):
    S, R = make(D, z0[-1], nodes=NODES, threads=THREADS, kc=KC)
    if len(z0) != S.layout()[2]:            # a kc = 23 seed for another kc
        z0 = S.regrid(z0, make_long(23, D - 2, nodes=NODES, Ks=60))
    z, info = ne_solve(R, z0)
    h_ = info["hist"][-1]
    ok = h_[2] < 1e-12 and h_[1] < 1e-9 and info["F"] < 1e-7
    lam = gamma_of(R, z, lam_guess) if ok else None
    rec = dict(D=D, xK=XK, Delta=float(z[-1]), Delta_old=Delta_old, F=info["F"], strong=h_[1], weak=h_[2], junk=h_[3],
               n_weak=info["n_weak"], weak_s=[float(f"{x:.2e}") for x in info["weak_s"]], n_outer=len(info["hist"]),
               lam=lam, gamma=(1 / lam if lam else None), ok=bool(ok), source=src, seconds=round(time.time() - t0))
    return z, rec


hist = []
seeds = sorted((float(f[len(p):-4]), f) for p in PREFIXES for f in glob.glob(p + "*.npy") if f[len(p):-4].replace(".", "").isdigit())
SEED_MAX = float(os.environ.get("NE_SEED_MAX", "inf"))   # re-solve seeds only up to here, then continue
seeds = [(D, f) for D, f in seeds if D_from - 1e-9 <= D <= SEED_MAX]
lam_g = 2.0
for D, f in seeds:
    t0 = time.time()
    z0 = np.load(f)
    lam_g = extrap([r[0] for r in hist], [r[2] for r in hist], D) if len(hist) >= 4 and D > _gD[-1] else lam_table(D)
    z, rec = solve(D, z0, lam_g, t0, f, Delta_old=float(z0[-1]))
    print(json.dumps(rec), flush=True)
    open(out, "a").write(json.dumps(rec) + "\n")
    np.save(f"{OUTP}{D:.4f}.npy", z)
    if rec["ok"] and rec["lam"]:
        hist.append([D, z, rec["lam"]])

D = hist[-1][0] if hist else D_end
while D < D_end - 1e-9:
    Dt = min(D + h, D_end)
    t0 = time.time()
    Ds = [r[0] for r in hist]
    z, rec = solve(Dt, extrap(Ds, [r[1] for r in hist], Dt), extrap(Ds, [r[2] for r in hist], Dt), t0, "continuation")
    if not rec["ok"] or not rec["lam"]:
        h /= 2
        print(f"step to D={Dt:.4f} failed (|F| {rec['F']:.1e}, weak {rec['weak']:.1e}); h -> {h}", flush=True)
        if h < 0.02:
            print("giving up", flush=True)
            break
        continue
    print(json.dumps(rec), flush=True)
    open(out, "a").write(json.dumps(rec) + "\n")
    np.save(f"{OUTP}{Dt:.4f}.npy", z)
    hist.append([Dt, z, rec["lam"]])
    D = Dt
    h = min(h * 1.3, hmax)
