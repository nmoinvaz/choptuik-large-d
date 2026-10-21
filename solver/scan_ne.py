"""D-continuation with the nonlinear-elimination solver (ne_solver.py).

Phase 1 re-solves the stored x_K = 0.6 solutions (longB_D*.npy) from D_from upward; phase 2 continues past the
last one with a cubic predictor up to D_end. For every point: Delta, the residual split (strong / weak / junk),
the weak singular values, and gamma = 1/lambda from the Rust growth-mode matrix.

usage: python3 scan_ne.py out.jsonl D_from D_end h hmax [xK=0.6|0.7] [seed prefixes, comma separated] [kc=23]
(D_end < D_from: downward sweep, seeds between env NE_SEED_MIN and D_from re-solved from the top down, then continued
down to D_end)
Seeds are kc = 23 solutions; for another kc they are regridded. Threads per process: env NE_THREADS (2).
env NE_SEED_MAX: re-solve only the seeds up to this D, then continue from those solutions.
env NE_SEED_XK: the matching point of the seeds' layout, when it differs from x_K (e.g. x_K 0.7 seeds for an x_K 0.65
    run: the two layouts have the same length). The seed is taken back to the x_K 0.6 layout and, for x_K > 0.6, gets
    a node at x_K from the horizon series at u = 1 - x_K.
env NE_JUNK_H: step of the junk solve's forward-difference Jacobian (default 1e-2; 1e-3 reaches the bottom of the flat
  valley at D ~ 29, where 1e-2 can stop 1e-7 short in Delta). A comma list (1e-2,3e-3,1e-3) is tried in turn.
env NE_RK: multiply all RK4 step counts (x_K = 0.7 needs 4).
env NE_NJUNK: in the junk solve, the junk set is this many smallest singular directions, not s/s_max < 3e-14
    (x_K 0.7 kc 23: 18). Off the valley bottom the junk singular values smear above 3e-14 and the misfiled ones stall
    the junk solve (D = 30.98). The first NE solve keeps the cut.
env NE_JUNK_DAMP=1: the junk solve tries Levenberg-damped steps instead of shortened Gauss-Newton steps.
env NE_JUNK_DAMP_MU: comma list of the damping values mu / sigma_max tried (default 0,1e-3,1e-2,0.1,0.3,1).
env NE_PRED_JUNK0=1: the continuation predictor extrapolates only the non-junk coordinates and keeps the junk
    coordinates of the last solution (SVD there). The junk coordinates are not smooth in D (each point stops at a
    different place in the flat valley: 0.1 apart between 30.32 and 30.58), so extrapolating them throws the
    predictor 0.16 (linear) to 0.36 (quadratic) off in the junk directions.
env NE_KS: horizon series terms (default 60; x_K = 0.5 needs ~120).
env NE_LAM_EXTRA: comma list of jsonl files whose (D, lam) records join the gamma_rust.jsonl table of lambda guesses
  (that table starts at D = 7.75, so a sweep that starts below it, e.g. from the D = 4 seeds, needs
  scan_ne6_lowD_kc47.jsonl here: with the table's lambda(7.75) as the guess the growth-mode search misses, gamma is
  None, the point does not enter the history and the continuation never starts).
env NE_SEED_KC: the seeds are native solutions on this run's nodes at that kc (e.g. 47 for neB_kc47_rk2_D*), regridded
  to KC instead of being treated as kc 23 seeds (a seed already at KC on these nodes is used as it is anyway).
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
NODES = [x for x in NODES_L if x <= XK + 1e-9] + ([XK] if XK > NODES_L[-1] + 1e-9 else [])   # x_K 0.4 .. 0.7
KS = int(os.environ.get("NE_KS", "60"))                          # horizon series terms (x_K = 0.5: ~120)
# a longer series has larger coefficients: f64 rounding of z leaves strong residuals ~1e-9 (harmless for Delta:
# x_K = 0.4 with 3.6e-9 agrees with x_K = 0.5 to 4e-10 at D = 23)
STRONG_TOL = float(os.environ.get("NE_STRONG_TOL", "1e-8" if KS > 60 else "1e-9"))
WEAK_TOL = float(os.environ.get("NE_WEAK_TOL", "1e-12"))       # acceptance cut on the weak residual (default as before)
SEED = "long7_D" if XK > 0.65 else "longB_D"
_xt = int(round(XK * 100))
OUTP = {60: "neB_D"}.get(_xt, f"ne{_xt // 10}_D" if _xt % 10 == 0 else f"ne{_xt}_D")   # ne4_D, ne5_D, neB_D, ne65_D, ne7_D
SEED_XK = float(os.environ["NE_SEED_XK"]) if os.environ.get("NE_SEED_XK") else None
SEED_KC = int(os.environ["NE_SEED_KC"]) if os.environ.get("NE_SEED_KC") else None
PREFIXES = sys.argv[7].split(",") if len(sys.argv) > 7 else [SEED]
KC = int(sys.argv[8]) if len(sys.argv) > 8 else 23
THREADS = int(os.environ.get("NE_THREADS", "2"))
if KC != 23:
    OUTP = OUTP.replace("_D", f"_kc{KC}_D")
SMOOTH = os.environ.get("NE_SMOOTH", "0") == "1"
if SMOOTH:
    OUTP = OUTP.replace("_D", "_sj_D")
JUNK = int(os.environ.get("NE_JUNK", "0"))
NJUNK = int(os.environ["NE_NJUNK"]) if os.environ.get("NE_NJUNK") else None   # fixed junk count in the junk solve
PRED_JUNK0 = os.environ.get("NE_PRED_JUNK0", "0") == "1"
JUNK_DAMP = os.environ.get("NE_JUNK_DAMP", "0") == "1"           # Levenberg-damped junk steps (junk_solve damp)
JUNK_DAMP_MU = [float(x) for x in os.environ.get("NE_JUNK_DAMP_MU", "0,1e-3,1e-2,0.1,0.3,1").split(",")]
if JUNK:
    OUTP = OUTP.replace("_D", "_js_D")
RK = int(os.environ.get("NE_RK", "1"))
RK_SEG = os.environ.get("NE_RK_SEG", "")                         # per-segment extra RK factors (rust_solver.make)
if RK != 1:
    OUTP = OUTP.replace("_D", f"_rk{RK}_D")
if RK_SEG:
    OUTP = OUTP.replace("_D", "_s" + RK_SEG.replace(",", "") + "_D")
_g = [json.loads(l) for l in open("gamma_rust.jsonl")]
for _f in filter(None, os.environ.get("NE_LAM_EXTRA", "").split(",")):
    _g += [json.loads(l) for l in open(_f)]
_g = sorted((r for r in _g if r.get("lam")), key=lambda r: r["D"])
_gD = np.array([r["D"] for r in _g]); _gl = np.array([r["lam"] for r in _g])


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


def convert_xk(z0, D, sxk):
    """kc 23 seed matched at x_K = sxk -> this setup: back to the x_K 0.6 layout (drop node blocks above 0.6), then a
    node at x_K > 0.6 from the horizon series at u = 1 - x_K (rust_solver.convert_06_to), or convert_seed below 0.6."""
    src = [x for x in NODES_L if x <= sxk + 1e-9] + ([sxk] if sxk > NODES_L[-1] + 1e-9 else [])
    sizes, K, L = make_long(23, D - 2, nodes=src, Ks=60).layout()
    if L != len(z0):
        raise ValueError(f"seed of length {len(z0)} is not an x_K = {sxk} kc 23 layout ({L})")
    offs = np.concatenate([[0], np.cumsum(sizes)])
    z6 = np.r_[z0[:offs[2 + 4 * len(NODES_L)]], z0[offs[2 + 4 * K]:]]
    if XK <= NODES_L[-1] + 1e-9:
        return convert_seed(z6, D)
    from rust_solver import convert_06_to
    z1 = convert_06_to(D, z6, XK, Ks=KS)
    if KC != 23:
        z1 = make_long(KC, D - 2, nodes=NODES, Ks=60).regrid(z1, make_long(23, D - 2, nodes=NODES, Ks=60))
    return z1


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


def predict(hist, Dt):
    """Continuation predictor: polynomial extrapolation of the last solutions; with NE_PRED_JUNK0 the junk
    coordinates (at the last solution) are kept instead of extrapolated."""
    Ds = [r[0] for r in hist]
    zp = extrap(Ds, [r[1] for r in hist], Dt)
    if PRED_JUNK0:
        Dl, zl = hist[-1][0], hist[-1][1]
        _, Rl = make(Dl, zl[-1], nodes=NODES, threads=THREADS, kc=KC, Ks=KS)
        _, J = Rl.jacobian(zl)
        _, s, Vt = np.linalg.svd(J)
        jm = np.arange(len(s)) >= len(s) - NJUNK if NJUNK else s / s[0] < 3e-14
        dz = zp - zl
        zp = zl + dz - Vt[jm].T @ (Vt[jm] @ dz)
    return zp


def solve(D, z0, lam_guess, t0, src, Delta_old=None):
    S, R = make(D, z0[-1], nodes=NODES, threads=THREADS, kc=KC, Ks=KS)
    if SEED_XK is not None and abs(SEED_XK - XK) > 1e-9 and src != "continuation":
        z0 = convert_xk(z0, D, SEED_XK)        # a seed matched elsewhere (lengths can coincide: 0.65 vs 0.7)
    elif len(z0) != S.layout()[2] and SEED_KC is not None and src != "continuation":
        z0 = make_long(KC, D - 2, nodes=NODES, Ks=60).regrid(z0, make_long(SEED_KC, D - 2, nodes=NODES, Ks=60))
    elif len(z0) != S.layout()[2]:            # a seed with other nodes (drop the outer node blocks) and/or kc = 23
        z0 = convert_seed(z0, D)
    if np.linalg.norm(R(z0)) > 1e-8:           # predictor or old seed far off: damped LM first, then NE polishes
        z0, _, _, _ = dd_solve(R, z0, maxjac=12, use_dd=False, stall=3)
    z, info = ne_solve(R, z0)
    if SMOOTH:                                   # alternate until Delta stops moving (3-5 rounds at D = 23)
        for _ in range(8):
            d_prev = z[-1]
            z, _, _ = smooth_junk(R, S, z)
            if np.linalg.norm(R(z)) > 1e-8:      # the junk move is only linearly null: its 2nd-order residual
                z, _, _, _ = dd_solve(R, z, maxjac=20, use_dd=False, stall=4, cut=3e-14)   # LM that leaves junk alone
            z, info = ne_solve(R, z)
            if abs(z[-1] - d_prev) < 1e-10:
                break
    h_ = info["hist"][-1]
    F, junk, js = info["F"], h_[3], {}
    if JUNK:
        from junk_solve import junk_solve
        Delta_ne = float(z[-1])
        z, ji = junk_solve(D, z, NODES, KC, JUNK, Ks=KS, h=[float(x) for x in os.environ.get("NE_JUNK_H", "1e-2").split(",")],
                           n_junk=NJUNK, damp=JUNK_DAMP, damp_mu=JUNK_DAMP_MU)
        F, junk = ji["F"], ji["r"]
        js = dict(js=True, Delta_ne=Delta_ne, junk_ne=ji["r0"], n_junk=ji["n_junk"], junk_its=ji["its"])
    ok = h_[2] < WEAK_TOL and h_[1] < STRONG_TOL and F < 1e-7
    lam = gamma_of(R, z, lam_guess) if ok else None
    rec = dict(D=D, xK=XK, kc=KC, Ks=KS, rk=RK, rk_seg=RK_SEG or None, smooth=SMOOTH, Delta=float(z[-1]), Delta_old=Delta_old, F=F, strong=h_[1], weak=h_[2], junk=junk,
               n_weak=info["n_weak"], weak_s=[float(f"{x:.2e}") for x in info["weak_s"]], n_outer=len(info["hist"]),
               lam=lam, gamma=(1 / lam if lam else None), ok=bool(ok), source=src, **js, seconds=round(time.time() - t0))
    return z, rec


if __name__ == "__main__":                     # junk_solve's process pool re-imports this module
    hist = []
    seeds = sorted((float(f[len(p):-4]), f) for p in PREFIXES for f in glob.glob(p + "*.npy") if f[len(p):-4].replace(".", "").isdigit())
    SEED_MAX = float(os.environ.get("NE_SEED_MAX", "inf"))   # re-solve seeds only up to here, then continue
    DOWN = D_end < D_from                      # downward sweep: seeds from D_from down to NE_SEED_MIN, then continue down
    if DOWN:
        SEED_MIN = float(os.environ.get("NE_SEED_MIN", "-inf"))
        seeds = sorted([(D, f) for D, f in seeds if SEED_MIN - 1e-9 <= D <= D_from + 1e-9], reverse=True)
    else:
        seeds = [(D, f) for D, f in seeds if D_from - 1e-9 <= D <= SEED_MAX]
    lam_g = 2.0
    for D, f in seeds:
        t0 = time.time()
        z0 = np.load(f)
        lam_g = extrap([r[0] for r in hist], [r[2] for r in hist], D) if len(hist) >= 4 and (D > _gD[-1] or D < _gD[0]) else lam_table(D)
        z, rec = solve(D, z0, lam_g, t0, f, Delta_old=float(z0[-1]))
        print(json.dumps(rec), flush=True)
        open(out, "a").write(json.dumps(rec) + "\n")
        np.save(f"{OUTP}{D:.4f}.npy", z)
        if rec["ok"] and rec["lam"]:
            hist.append([D, z, rec["lam"]])

    D = hist[-1][0] if hist else D_end
    while (D > D_end + 1e-9) if DOWN else (D < D_end - 1e-9):
        Dt = max(D - h, D_end) if DOWN else min(D + h, D_end)
        t0 = time.time()
        Ds = [r[0] for r in hist]
        z, rec = solve(Dt, predict(hist, Dt), extrap(Ds, [r[2] for r in hist], Dt), t0, "continuation")
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
