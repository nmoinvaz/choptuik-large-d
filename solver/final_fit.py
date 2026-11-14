"""Delta_inf from all step-converged data: x_K 0.6 RK x2 (Richardson with RK x1 where available, junk-solved above
17.2) up to D = 30.2, plus x_K 0.7 RK x4 junk-solved above that. Weighted fits in x / max(x) for x = 1/D, 1/(D-1),
1/(D-2), several degrees and lower cuts; candidate limits forced for comparison.

    python3 ../final_fit.py            (from solver/data)
    python3 ../final_fit.py --range    the README's f64 range of Delta_inf on D <= 30.2 (limit_range)
    python3 ../final_fit.py --plant [--mutate]    planted-truth test of that range (exit 1 when a plant fails)
    python3 ../final_fit.py --plant --seeds N      the plants at noise seeds 1..N (informative, not graded)
    from final_fit import step_converged

--range: the free limit c0 of weighted_series for each of the README's 24 fit choices, variable 1/D, 1/(D-1) or
1/(D-2), degree 8-11, lo <= D <= 30.2 with lo = 10 or 12; the range is [min, max] of c0 over the fits with
chi2/dof <= 1, quoted as its centre and half-width.

--plant (plants and pass criteria fixed before the first run). Synthetic Delta(D) with a known limit c0 on the real D
grid of step_converged (D <= 30.2), with the real weights 1/sigma, through limit_range. Base: the stored formula
(data/formula_coefficients.json, P8/Q2 in 1/(D-2), in none of the three polynomial families), whose limit a0 =
1.6438975180 is exact. P2, P3 add c h(t), t = 1/D, h(0) = 0, |c h(0.1)| = 1e-3 (as hp/planted_dinf.py T2, T5):
    P1  none                          (the formula's own shape; plumbing)
    P2  t^3 / (1 + t/0.12)            pole at t = -0.12 (D = -8.3)
    P3  t^2 log(1 + t/0.05)           log branch point at t = -0.05 (D = -20)
Noise Gaussian, sigma 0.8 times the weights' sigma (the real fits' chi2/dof is about 0.63), fixed seeds.
Pass criteria: at least 10 of the 24 fits have chi2/dof <= 1, c0 lies in [min, max] of their limits, and within the
README's half-width 0.00008 of that range's centre. Control --mutate adds 3 x 0.00008 to every synthetic point and
keeps c0; every plant must then fail.
"""
import glob
import json
import os
import sys

import numpy as np

from data_best import load as old_load


def step_converged(Dmin=8.0, hi_from=30.2, hi_junk=1.2e-15, hi_refined=True):
    """(D, Delta, sigma, gamma) arrays: x_K 0.6 RK x2 (+ Richardson) up to hi_from, x_K 0.7 RK x3+ junk-solved above,
    only points driven to junk <= hi_junk (one left at ~2e-15 can sit ~2e-7 up the flat valley).
    Above hi_from the RK error of the last segment (x = 0.6 -> 0.7) grows ~2x per 0.8 in D (RK x3: +4.0e-7 at 34.18,
    +1.5e-6 at 35.78); runs with that segment refined (rk_seg, NE_RK_SEG) are step-converged (34.18: x24 = x48 to
    1e-10). Per D the best class wins (refined > RK x6+ > RK x3-4), then the lowest junk. hi_refined: use only refined
    runs above hi_from."""
    o = old_load()
    rk1 = {k: r["Delta"] for k, r in o.items() if k < 17.2}
    rk2, hi, gam = {}, {}, {}
    for f in sorted(glob.glob("scan_ne*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if not r.get("ok") or r.get("kc", 23) != 23 or r.get("smooth"):
                continue
            k = round(r["D"], 4)
            xk = r.get("xK", 0.6)
            rk = r.get("rk", 1)
            if abs(xk - 0.6) < 1e-9:
                if rk == 2:
                    rk2[k] = r["Delta"]
                    gam[k] = r.get("gamma")
                elif rk == 1 and (r.get("js") or (17.2 <= k <= 23.4)):
                    if r.get("js") or k not in rk1:
                        rk1[k] = r["Delta"]
            elif abs(xk - 0.7) < 1e-9 and rk >= 3 and r.get("js") and k > hi_from and r.get("junk", 1) <= hi_junk:
                cls = 2 if r.get("rk_seg") else 1 if rk >= 6 else 0
                if hi_refined and cls < 2:
                    continue
                if k not in hi or (cls, -r["junk"]) > (hi[k][2], -hi[k][1]):
                    hi[k] = (r["Delta"], r["junk"], cls)
                    gam[k] = r.get("gamma")
    D, y, s, g = [], [], [], []
    for k in sorted(rk2):
        if k < Dmin or k > hi_from:
            continue
        D.append(k)
        y.append(rk2[k] + ((rk2[k] - rk1[k]) / 15 if k in rk1 else 0.0))
        s.append(3e-11 if k < 22 else 2e-9 if k < 26 else 1.5e-9 if k < 29.5 else 3e-8)   # rms fit residual per range
        g.append(gam.get(k))
    for k in sorted(hi):
        D.append(k)
        y.append(hi[k][0])
        s.append(1.5e-7 * np.exp(0.27 * (k - 29.3)))   # flat-valley width: ~1.5e-7 at 29.3, >= 7e-7 at 34.98 (x_K 0.7)
        g.append(gam.get(k))
    return np.array(D), np.array(y), np.array(s), np.array([np.nan if v is None else v for v in g], float)


def weighted_series(x, y, w, m, c0=None):
    """Weighted polynomial fit in x / max(x); returns coefficients of x and chi2/dof."""
    sc = x.max()
    A = np.vander(x / sc, m + 1, increasing=True)
    if c0 is None:
        c, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        chi = np.sum(((A @ c - y) * w) ** 2) / (len(y) - m - 1)
    else:
        c, *_ = np.linalg.lstsq(A[:, 1:] * w[:, None], (y - c0) * w, rcond=None)
        chi = np.sum(((A[:, 1:] @ c - (y - c0)) * w) ** 2) / (len(y) - m)
        c = np.r_[c0, c]
    return c / sc ** np.arange(m + 1), chi


RANGE_VARS = (("1/D", lambda d: 1 / d), ("1/(D-1)", lambda d: 1 / (d - 1)), ("1/(D-2)", lambda d: 1 / (d - 2)))
RANGE_LOS, RANGE_DEGREES, RANGE_DMAX, RANGE_CHI = (10, 12), (8, 9, 10, 11), 30.2, 1.0
HALF = 0.00008                                         # the README's half-width of the f64 range
COEF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "formula_coefficients.json")
PLANTS = {"P1": (None, 0.0),
          "P2": (lambda t: t ** 3 / (1 + t / 0.12), 1e-3),
          "P3": (lambda t: t ** 2 * np.log1p(t / 0.05), 1e-3)}


def limit_range(D, y, s, dmax=RANGE_DMAX):
    """Rows (lo, variable, degree, c0, chi2/dof) of the README's 24 fit choices on lo <= D <= dmax, and the c0 of the
    rows with chi2/dof <= 1."""
    rows = []
    for lo in RANGE_LOS:
        m = (D >= lo) & (D <= dmax)
        for name, f in RANGE_VARS:
            for k in RANGE_DEGREES:
                c, chi = weighted_series(f(D[m]), y[m], 1 / s[m], k)
                rows.append((lo, name, k, c[0], chi))
    return rows, [r[3] for r in rows if r[4] <= RANGE_CHI]


def formula_delta(D):
    """The stored formula P8(e)/Q2(e), e = 1/(D-2), as formula_all.delta (read here, since formula_all imports this
    module), and its exact limit a0."""
    C = json.load(open(COEF))
    a = np.array([float(v) for v in C["a"]])
    b = np.r_[1.0, [float(v) for v in C["b"]]]
    e = 1 / (np.asarray(D, float) - 2)
    return np.polyval(a[::-1], e) / np.polyval(b[::-1], e), a[0]


def plant_one(D, s, base, c0, h, size, seed, mutate=False):
    """One plant through limit_range: (number of fits kept, min c0, max c0, pass)."""
    rng = np.random.default_rng(seed)
    y = base + (size / abs(h(0.1)) * h(1 / D) if h else 0.0) + rng.normal(0.0, 0.8 * s)
    if mutate:
        y = y + 3 * HALF                               # the limit moves and c0 is kept, so every plant must fail
    _, ok = limit_range(D, y, s)
    lo, hi = (min(ok), max(ok)) if ok else (np.nan, np.nan)
    return len(ok), lo, hi, bool(len(ok) >= 10 and lo <= c0 <= hi and abs((lo + hi) / 2 - c0) <= HALF)


def plant_grid():
    D, _, s, _ = step_converged()
    m = D <= RANGE_DMAX
    return (D[m], s[m]) + formula_delta(D[m])


def planted(mutate=False):
    """Run the plants through limit_range at their fixed seeds; True when every plant meets the pass criteria."""
    D, s, base, c0 = plant_grid()
    passed = True
    for i, (name, (h, size)) in enumerate(PLANTS.items()):
        n, lo, hi, good = plant_one(D, s, base, c0, h, size, 1 + i, mutate)
        centre = (lo + hi) / 2
        passed &= good
        print(f"{name}: {n} of 24 fits chi2/dof <= 1, c0 {lo:.6f} .. {hi:.6f}, centre - planted "
              f"{centre - c0:+.2e}, half-width {(hi - lo) / 2:.2e}, planted {'inside' if lo <= c0 <= hi else 'OUTSIDE'}"
              f"  {'pass' if good else 'FAIL'}")
    print(f"planted limit {c0:.10f}{' (mutated: +' + f'{3 * HALF:g}' + ' on every point)' if mutate else ''}: "
          f"{'every plant passes' if passed else 'a plant fails'}")
    return passed


def planted_seeds(n):
    """Each plant at noise seeds 1..n: how often it passes, how often c0 is inside the range, and the spread of the
    range's centre about c0 (informative, not graded)."""
    D, s, base, c0 = plant_grid()
    for name, (h, size) in PLANTS.items():
        r = [plant_one(D, s, base, c0, h, size, seed) for seed in range(1, n + 1)]
        off = np.array([(lo + hi) / 2 - c0 for k, lo, hi, _ in r if k])
        hw = np.array([(hi - lo) / 2 for k, lo, hi, _ in r if k])
        print(f"{name}: {sum(g for *_, g in r)} of {n} seeds pass, c0 inside {sum(lo <= c0 <= hi for k, lo, hi, _ in r if k)}, "
              f"no fit kept {sum(k == 0 for k, *_ in r)}; centre - planted {off.min():+.2e} .. {off.max():+.2e} "
              f"(rms {np.sqrt(np.mean(off ** 2)):.2e}), half-width median {np.median(hw):.2e}")


def main_range():
    D, y, s, _ = step_converged()
    rows, ok = limit_range(D, y, s)
    for lo, name, k, c0, chi in rows:
        print(f"D>={lo:2d} {name:8s} m={k:2d}  c0 {c0:.9f}  chi2/dof {chi:6.2f}{'' if chi <= RANGE_CHI else '  (dropped)'}")
    lo, hi = min(ok), max(ok)
    print(f"{len(ok)} of {len(rows)} fits with chi2/dof <= 1 on D <= {RANGE_DMAX}: c0 {lo:.6f} .. {hi:.6f}, "
          f"centre {(lo + hi) / 2:.6f}, half-width {(hi - lo) / 2:.6f}")


if __name__ == "__main__" and "--range" in sys.argv:
    main_range()
elif __name__ == "__main__" and "--plant" in sys.argv and "--seeds" in sys.argv:
    planted_seeds(int(sys.argv[sys.argv.index("--seeds") + 1]))
elif __name__ == "__main__" and "--plant" in sys.argv:
    sys.exit(0 if planted("--mutate" in sys.argv) else 1)
elif __name__ == "__main__":
    D, y, s, _ = step_converged()
    print(f"{len(D)} points: D {D.min():.2f}..{D.max():.2f} ({sum(D > 30.2)} x_K 0.7 RK x4 above 30.2)")
    cands = {"pi^2/6": np.pi ** 2 / 6, "1.6438": 1.6438, "1.6440": 1.6440, "1.6442": 1.6442}
    for lo in (8, 10, 12):
        m_ = D >= lo
        for xn, xf in (("1/D", lambda d: 1 / d), ("1/(D-1)", lambda d: 1 / (d - 1)), ("1/(D-2)", lambda d: 1 / (d - 2))):
            x = xf(D[m_])
            w = 1 / s[m_]
            row = [f"{weighted_series(x, y[m_], w, m)[0][0]:.6f}({weighted_series(x, y[m_], w, m)[1]:.1f})" for m in (8, 9, 10, 11)]
            cc = [f"{name}:{weighted_series(x, y[m_], w, 10, L)[1]:.1f}" for name, L in cands.items()]
            print(f"D>={lo:2d} {xn:8s} free c0(chi2/dof) m=8..11: " + " ".join(row) + "  | forced m=10 chi2/dof: " + " ".join(cc))
