"""Time BorderedCore's mpmath work per call at L = 769 (#1008's size), without the kernel: the pieces of newton_o
(to_mp_vec of a and b, g, den, dD, to_parts of the step, join) and of residual_o (split, g_eff, join), on random
eight-part vectors, not a state of the chain (notes/CRAWL.md section 16, "What the border costs now").

    cd hp && python3 border_cost.py . 5

The first argument is the directory that holds arclength.py, the second the number of repeats (default 5).
"""
import sys, time
import numpy as np
import mpmath as mp
sys.path.insert(0, sys.argv[1])
import arclength as A

L = 769
rng = np.random.default_rng(1)

def vec8(scale):
    x = rng.standard_normal(L) * scale
    parts = [x]
    for q in range(1, 8):
        parts.append(rng.standard_normal(L) * scale * 2.0 ** (-53 * q))
    return tuple(np.ascontiguousarray(p) for p in parts)

class FakeSys:
    L = L
    opt = dict(arc_dhold=1e-12)

reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
ts = dict(newton=[], residual=[], olu_hold=[])
with mp.workdps(A.DPS_SERIES):
    zp = A.to_mp_vec(vec8(1.0))
    tz = A.to_mp_vec(vec8(1e-4))
    core = A.BorderedCore(FakeSys(), zp, mp.mpf("259.3830708213711742098100558743816"), tz, mp.mpf("0.99999560"))
    for r in range(reps):
        z, d, b = vec8(1.0), vec8(1e-16), vec8(3e-3)
        y = core.join(mp.mpf("259.3830708213711742098100558743816"), z)
        t0 = time.perf_counter()
        # newton_o's mpmath part, as written in BorderedCore.newton_o
        D, zz = core.split(y)
        am, bm = A.to_mp_vec(d), A.to_mp_vec(b)
        g = core.g(D, zz)
        den = core.tD - A.vdot(core.tz, bm)
        dD = -(g + A.vdot(core.tz, am)) / den
        hold = bool(abs(dD) <= core.hold_max and abs(g) <= core.g_tol(D))
        dDr = mp.mpf(0) if hold else A.dd_round(D + dD) - D
        step = core.join(dDr, A.to_parts(A.vadd(am, bm, -dDr)))
        ge = core.g_eff(D, zz)
        ts["newton"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        D, zz = core.split(y)
        out = core.join(core.g_eff(D, zz), zz)
        ts["residual"].append(time.perf_counter() - t0)
for k, v in ts.items():
    if v:
        print(f"{k}: median {np.median(v):.3f} s, min {min(v):.3f}, max {max(v):.3f} over {len(v)}")
