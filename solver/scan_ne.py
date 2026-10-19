"""D-continuation with the nonlinear-elimination solver (ne_solver.py).

Phase 1 re-solves the stored x_K = 0.6 solutions (longB_D*.npy) from D_from upward; phase 2 continues past the
last one with a cubic predictor up to D_end. For every point: Delta, the residual split (strong / weak / junk),
the weak singular values, and gamma = 1/lambda from the Rust growth-mode matrix.

usage: python3 scan_ne.py out.jsonl D_from D_end h hmax [xK=0.6|0.7] [seed prefixes, comma separated] [kc=23]
Seeds are kc = 23 solutions; for another kc they are regridded. Threads per process: env NE_THREADS (2).
env NE_SEED_MAX: re-solve only the seeds up to this D, then continue from those solutions.
env NE_JUNK_H: step of the junk solve's forward-difference Jacobian (default 1e-2; 1e-3 reaches the bottom of the flat
  valley at D ~ 29, where 1e-2 can stop 1e-7 short in Delta). A comma list (1e-2,3e-3,1e-3) is tried in turn.
env NE_RK: multiply all RK4 step counts (x_K = 0.7 needs 4).
env NE_NJUNK: the junk set is this many smallest singular directions, not s/s_max < 3e-14 (x_K 0.7 kc 23: 18). At a
    continuation predictor the junk singular values smear above 3e-14 and the misfiled ones stall the junk solve.
env NE_KS: horizon series terms (default 60; x_K = 0.5 needs ~120).
env NE_SMOOTH=1: after each solve, fix the junk directions by smoothness (smooth_junk.py) and re-solve, until
  Delta changes by < 1e-10 (at most 8 rounds).
env NE_JUNK: solve the junk equations too (junk_solve.py) on a pool of this many processes. Without it the junk
  coordinates keep whatever values the start had, which moves Delta by up to 1e-5 at D = 23 and more above.
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
from dd_solver import dd_solve
from smooth_junk import smooth_junk

out, D_from, D_end, h, hmax = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
XK = float(sys.argv[6]) if len(sys.argv) > 6 else 0.6
NODES = [x for x in NODES_L + [0.7] if x <= XK + 1e-9]           # x_K = 0.5, 0.6 or 0.7
KS = int(os.environ.get("NE_KS", "60"))                          # horizon series terms (x_K = 0.5: ~120)
# a longer series has larger coefficients: f64 rounding of z leaves strong residuals ~1e-9 (harmless for Delta:
# x_K = 0.4 with 3.6e-9 agrees with x_K = 0.5 to 4e-10 at D = 23)
STRONG_TOL = float(os.environ.get("NE_STRONG_TOL", "1e-8" if KS > 60 else "1e-9"))
SEED = "long7_D" if XK > 0.65 else "longB_D"
OUTP = {6: "neB_D"}.get(int(round(XK * 10)), f"ne{int(round(XK * 10))}_D")   # ne4_D, ne5_D, neB_D, ne7_D
PREFIXES = sys.argv[7].split(",") if len(sys.argv) > 7 else [SEED]
KC = int(sys.argv[8]) if len(sys.argv) > 8 else 23
THREADS = int(os.environ.get("NE_THREADS", "2"))
if KC != 23:
    OUTP = OUTP.replace("_D", f"_kc{KC}_D")
SMOOTH = os.environ.get("NE_SMOOTH", "0") == "1"
if SMOOTH:
    OUTP = OUTP.replace("_D", "_sj_D")
JUNK = int(os.environ.get("NE_JUNK", "0"))
NJUNK = int(os.environ["NE_NJUNK"]) if os.environ.get("NE_NJUNK") else None   # fixed junk count (ne_solve n_junk)
if JUNK:
    OUTP = OUTP.replace("_D", "_js_D")
RK = int(os.environ.get("NE_RK", "1"))
if RK != 1:
    OUTP = OUTP.replace("_D", f"_rk{RK}_D")
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


def convert_seed(z0, D):
    """kc = 23 seed on NODES_L or NODES_L + [0.7] -> this setup: drop node blocks beyond x_K, regrid to KC."""
    for src in (NODES_L, NODES_L + [0.7], NODES):
        S0 = make_long(23, D - 2, nodes=src, Ks=60)
        sizes, K, L = S0.layout()
        if L == len(z0):
            break
    else:
        raise ValueError(f"seed of length {len(z0)} matches no known layout")
    if len(NODES) > len(src):                  # x_K 0.6 seed for x_K 0.7: node 0.7 from the horizon series at u = 0.3
        from rust_solver import convert_06_to_07
        z0 = convert_06_to_07(D, z0)
        src = NODES_L + [0.7]
        S0 = make_long(23, D - 2, nodes=src, Ks=60)
        sizes, K, L = S0.layout()
    offs = np.concatenate([[0], np.cumsum(sizes)])
    j0, j1 = 2 + 4 * len(NODES), 2 + 4 * K
    z0 = np.r_[z0[:offs[j0]], z0[offs[j1]:]]
    if KC != 23:
        z0 = make_long(KC, D - 2, nodes=NODES, Ks=60).regrid(z0, make_long(23, D - 2, nodes=NODES, Ks=60))
    return z0


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
    S, R = make(D, z0[-1], nodes=NODES, threads=THREADS, kc=KC, Ks=KS)
    if len(z0) != S.layout()[2]:            # a seed with other nodes (drop the outer node blocks) and/or kc = 23
        z0 = convert_seed(z0, D)
    if np.linalg.norm(R(z0)) > 1e-8:           # predictor or old seed far off: damped LM first, then NE polishes
        z0, _, _, _ = dd_solve(R, z0, maxjac=12, use_dd=False, stall=3)
    z, info = ne_solve(R, z0, n_junk=NJUNK)
    if SMOOTH:                                   # alternate until Delta stops moving (3-5 rounds at D = 23)
        for _ in range(8):
            d_prev = z[-1]
            z, _, _ = smooth_junk(R, S, z)
            if np.linalg.norm(R(z)) > 1e-8:      # the junk move is only linearly null: its 2nd-order residual
                z, _, _, _ = dd_solve(R, z, maxjac=20, use_dd=False, stall=4, cut=3e-14)   # LM that leaves junk alone
            z, info = ne_solve(R, z, n_junk=NJUNK)
            if abs(z[-1] - d_prev) < 1e-10:
                break
    h_ = info["hist"][-1]
    F, junk, js = info["F"], h_[3], {}
    if JUNK:
        from junk_solve import junk_solve
        Delta_ne = float(z[-1])
        z, ji = junk_solve(D, z, NODES, KC, JUNK, Ks=KS, h=[float(x) for x in os.environ.get("NE_JUNK_H", "1e-2").split(",")],
                           n_junk=NJUNK)
        F, junk = ji["F"], ji["r"]
        js = dict(js=True, Delta_ne=Delta_ne, junk_ne=ji["r0"], n_junk=ji["n_junk"], junk_its=ji["its"])
    ok = h_[2] < 1e-12 and h_[1] < STRONG_TOL and F < 1e-7
    lam = gamma_of(R, z, lam_guess) if ok else None
    rec = dict(D=D, xK=XK, kc=KC, Ks=KS, rk=RK, smooth=SMOOTH, Delta=float(z[-1]), Delta_old=Delta_old, F=F, strong=h_[1], weak=h_[2], junk=junk,
               n_weak=info["n_weak"], weak_s=[float(f"{x:.2e}") for x in info["weak_s"]], n_outer=len(info["hist"]),
               lam=lam, gamma=(1 / lam if lam else None), ok=bool(ok), source=src, **js, seconds=round(time.time() - t0))
    return z, rec


if __name__ == "__main__":                     # junk_solve's process pool re-imports this module
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
