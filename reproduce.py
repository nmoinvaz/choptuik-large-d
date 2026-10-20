#!/usr/bin/env python3
"""Reproduce the repository's headline numbers and checks, and say PASS / FAIL / SKIP for each.

    ./reproduce.sh            (checks the Python dependencies, then runs this script)
    python3 reproduce.py [--full]

Needs numpy, scipy, mpmath, sympy. Optional: python-flint (the Borel-Darboux check) and the compiled double-double
kernel hpkernel (the checks on saved solutions; build it with hp/build.sh, which needs Rust and maturin). Missing
optional parts are reported as SKIP. --full adds the late-order (Proposition 4) check, about two more minutes.

Measured on the development machine (2 cores, both busy with other jobs, run at nice -n 15); an idle laptop
should be two to four times faster:
  data, interpolation, formula     ~1 s
  Delta_inf (cleaning + 5 fits)    ~280 s  mpmath, 60 digits, every cleaned point 10 <= D <= 61.03 (not reduced;
                                           about 9 s per fit when the machine was idle)
  held-out test (3 fits)           ~25 s
  theory checks                    ~40 s   (the Proposition 1 formal check is ~30 s of it)
  saved solutions (hpkernel)       ~2 s
  total                            ~6 min  (--full: +2 min)
Nothing in the repository is written.
"""
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
HP = os.path.join(ROOT, "hp")
SOLVER = os.path.join(ROOT, "solver")
sys.path.insert(0, HP)
FULL = "--full" in sys.argv

results = []          # (check, expected, got, status)
T0 = time.time()


def record(name, expected, got, ok):
    status = "SKIP" if ok is None else ("PASS" if bool(ok) else "FAIL")
    results.append((name, expected, got, status))
    print(f"  [{status}] {name}: got {got} (expected {expected})", flush=True)


def section(title):
    print(f"\n== {title}  (t = {time.time() - T0:.0f} s)", flush=True)


def run(args, cwd, timeout=900):
    """Run a repository script, return its stdout (stderr appended)."""
    p = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.stdout + p.stderr


import mpmath as mp          # noqa: E402
import dd_fits_mp as F       # noqa: E402  (hp/dd_fits_mp.py: loading, Chebyshev fits at 60 digits, cleaning)

# ---------------------------------------------------------------------------------------------------------------
section("1. The double-double data (hp/data/delta_dd.csv)")
pts, _ = F.load_points()
Ds = sorted(pts)
record("points and range", "857 points, D = 8.25 .. 61.0264", f"{len(Ds)} points, D = {Ds[0]:.2f} .. {Ds[-1]:.4f}",
       len(Ds) >= 857 and Ds[0] <= 8.25 + 1e-9 and Ds[-1] >= 61.0)


def interp(D, k=8):
    """Lagrange interpolation of Delta at D from the k nearest points (60 digits)."""
    near = sorted(sorted(Ds, key=lambda d: abs(d - D))[:k])
    D = mp.mpf(D)
    v = mp.mpf(0)
    for i, di in enumerate(near):
        w = mp.mpf(1)
        for j, dj in enumerate(near):
            if j != i:
                w *= (D - dj) / (mp.mpf(di) - dj)
        v += w * pts[di]
    return v


TABLE = {10: "2.521078919562", 20: "2.112080154580", 30: "1.964957899258", 40: "1.888568197910",
         50: "1.841659503007", 60: "1.809889459975"}          # README, Values table (12 decimals)
worst = max(abs(interp(D) - mp.mpf(v)) for D, v in TABLE.items())
record("Values table, Delta(D) at D = 10..60 interpolated from the CSV", "within rounding, <= 1e-12",
       f"max |diff| {mp.nstr(worst, 2)}", worst <= 1e-12)

# README coefficient table (solver/formula_all.py), Delta = P8(eps)/Q2(eps), eps = 1/(D-2)
A = [1.6438975180, 4.0378045077e1, 2.5233285989e2, 2.0673502321e2, -3.5709850575e2, 3.1347320996e2,
     -4.1891374574e2, 3.2695252782e2, -1.6050467699e2]
B = [1, 1.8251453626e1, 6.8182050167e1]


def formula(D):
    e = 1 / (D - 2)
    return sum(a * e ** k for k, a in enumerate(A)) / sum(b * e ** k for k, b in enumerate(B))


fd = max(abs(formula(d) - float(pts[d])) for d in Ds)
record("rational formula vs every double-double point (8.25 <= D <= 61)", "<= 6e-8 (README: 5.8e-8)",
       f"max |diff| {fd:.2g}", fd <= 6e-8)
record("formula at D = 4 vs Martin-Garcia & Gundlach 3.445452402(3)", "|diff| <= 1e-7",
       f"{formula(4):.10f} (diff {abs(formula(4) - 3.445452402):.1g})", abs(formula(4) - 3.445452402) <= 1e-7)

# ---------------------------------------------------------------------------------------------------------------
section("2. Delta_inf from the analytic fits (hp/dd_fits_mp.py functions)")
Dall = [d for d in Ds if d >= 10]
Dc = F.clean(pts, Dall, verbose=False)
print(f"  cleaning drops {len(Dall) - len(Dc)} isolated points (README: 27 on 8.25..61.03), {len(Dc)} kept", flush=True)
lims = []
for model, deg in (("1/D", 16), ("1/D", 18), ("1/D", 20), ("1/n", 18), ("1/n", 20)):
    ev, res, _ = F.fit(pts, Dc, model, deg)
    L = F.limit(ev)
    lims.append(L)
    print(f"  {model:4s} degree {deg}: max residual {mp.nstr(res, 2):>8s}, limit {mp.nstr(L, 11)}", flush=True)
spread = max(lims) - min(lims)
ok = all(abs(L - mp.mpf("1.64389375")) <= mp.mpf("1e-8") for L in lims)
record("Delta_inf, 5 analytic fits on all cleaned points", "1.64389375 +- 1e-8 (README: 1.643893745-1.643893752)",
       f"{mp.nstr(min(lims), 10)} .. {mp.nstr(max(lims), 10)} (spread {mp.nstr(spread, 2)})", ok)

# ---------------------------------------------------------------------------------------------------------------
section("3. Held-out test: fit on D <= 45, predict 45 < D <= 61.03")
fitD = [d for d in Dc if d <= 45]
hold = [d for d in Dc if d > 45]
held = {}
for model, deg in (("1/D", 16), ("1/n", 18), ("n^-1/2", 20)):
    ev, res, _ = F.fit(pts, fitD, model, deg)
    held[model] = max(abs(ev(d) - pts[d]) for d in hold)
    print(f"  {model:7s} degree {deg}: held-out max error {mp.nstr(held[model], 2)} on {len(hold)} points, "
          f"limit {mp.nstr(F.limit(ev), 10)}", flush=True)
record("held-out error of the 1/D series (degree 16)", "<= 3e-12 (README: 5.2e-13 on the D <= 60.1 snapshot)",
       mp.nstr(held["1/D"], 2), held["1/D"] <= 3e-12)
record("half-integer series (n^-1/2, degree 20) predicts worse", ">= 10x the 1/D error",
       f"{mp.nstr(held['n^-1/2'] / held['1/D'], 3)}x", held["n^-1/2"] >= 10 * held["1/D"])

# ---------------------------------------------------------------------------------------------------------------
section("4. Theory checks")
out = run(["largeD_stokes_C.py"], SOLVER)
R = {}
for mu, n, re_, im_ in re.findall(r"mu (\d) n\s+([\d.]+): R =\s+\(([-\d.e]+) ([+-] ?[\d.e-]+)j\)", out):
    R[(int(mu), float(n))] = complex(float(re_), float(im_.replace(" ", "")))
r1 = R.get((1, 100000.0))
record("Proposition 2 / inner problem: C(n,s) vs the 2F0 Stokes constant, mu = 1, n = 1e5 (largeD_stokes_C.py)",
       "R -> -1 like 1/n: |R + 1| <= 2e-4", f"R = {r1:.7f}" if r1 is not None else "no output",
       r1 is not None and abs(r1 + 1) <= 2e-4)

out = run(["inner_np.py", "checklin", "3"], os.path.join(SOLVER, "inner_stokes"))
vals = re.findall(r"R_\d ([-\d.]+)[+-]", out)
record("lateral solutions without gravity, harmonics 1 and 3 (inner_stokes/inner_np.py checklin 3)",
       "R = 1.00000000", ", ".join(sorted(set(vals))) or "no output",
       bool(vals) and all(abs(float(v) - 1) <= 1e-7 for v in vals))

try:
    import flint  # noqa: F401
    out = run(["inner_stokes.py", "linear"], os.path.join(SOLVER, "inner_borel"))
    m = re.findall(r"with (\d+) corrections: .*rel err ([\d.e+-]+)", out)      # 6, 12, 20 correction terms
    record("Borel-Darboux linear Stokes constant, 20 correction terms (inner_borel/inner_stokes.py linear)",
           "rel err <= 1e-20 (README: 6e-27)", m[-1][1] if m else "no output", bool(m) and float(m[-1][1]) <= 1e-20)
except ImportError:
    record("Borel-Darboux linear Stokes constant", "rel err <= 1e-20", "python-flint not installed", None)

sys.path.insert(0, SOLVER)
import numpy as np                        # noqa: E402
import largeD_fast_transport as T         # noqa: E402
At = T.transported(0.9, 0.72, True)
errs = [np.abs(T.exact(n, 0.9, 0.72, True) - At).max() / np.abs(At).max() for n in (100, 200, 400)]
ratios = [errs[1] / errs[0], errs[2] / errs[1]]
record("Proposition 3: fast-mode transport with gravity, error at n = 100/200/400 (largeD_fast_transport.py)",
       "falls like 1/n (ratios ~0.5; README 0.57, 0.26, 0.13)",
       ", ".join(f"{e:.3g}" for e in errs) + f" (ratios {ratios[0]:.2f}, {ratios[1]:.2f})",
       all(0.4 <= r <= 0.6 for r in ratios))

try:
    import hpkernel  # noqa: F401
    have_hp = True
except ImportError:
    have_hp = False
out = run(["largeD_formal_check.py"], SOLVER)
rows = re.findall(r"n\s+(\d+): relative residuals .*x n\^3: ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", out)
if rows:
    first = {int(r[0]): [float(v) for v in r[1:]] for r in rows[:5]}      # the first omega block
    q = [first[800][i] / first[400][i] for i in range(4)]
    record("Proposition 1: formal solution in the exact equations, residual x n^3 at n = 400 vs 800 (largeD_formal_check.py)",
           "constant: ratios within 2% of 1", ", ".join(f"{v:.3f}" for v in q), all(abs(v - 1) <= 0.02 for v in q))
else:
    record("Proposition 1 formal check", "residual ~ n^-3", "did not run (needs hp/data/states)", None)

if FULL:
    out = run(["largeD_dingle_K.py"], SOLVER, timeout=1800)
    m = re.search(r"x\s+0\.9: .*Richardson K0/K_pred = \(([-\d.e]+) ([+-] ?[\d.e-]+)j\)", out)
    k = complex(float(m.group(1)), float(m.group(2).replace(" ", ""))) if m else None
    record("Proposition 4: Dingle constant of the linear late orders at x = 0.9 (largeD_dingle_K.py)",
           "K0/K_pred = 1 within 2e-3 (README: 1.0006)", f"{k:.5f}" if k is not None else "no output",
           k is not None and abs(k - 1) <= 2e-3)

# ---------------------------------------------------------------------------------------------------------------
section("5. Saved solutions (hp/data/states; needs hpkernel)")
if have_hp:
    from states import H                              # noqa: E402
    from largeD_profiles import solver_for, harmonics  # noqa: E402
    import glob                                         # noqa: E402
    rows = []
    for t in (20, 30, 40, 50, 60):
        f = sorted(glob.glob(H + f"up*_D{t}.*.npz"))[0]
        s, zh, zl, meta = solver_for(f)
        Mh, D = meta["M"], meta["D"]
        ch, cl = s.series(zh, zl, 0)
        C = (np.asarray(ch) + np.asarray(cl)).reshape(-1, 4, Mh)
        Fh = harmonics(C[1, 0] * np.sqrt(D - 2), True)
        rows.append((D, abs(Fh[3] / Fh[1])))
    x = np.array([1 / (D - 2) for D, _ in rows])
    y = np.array([r for _, r in rows])
    ext = np.polyval(np.polyfit(x, y, 2), 0)
    record("|F3/F1| of the echo profile at D = 60 (hp/largeD_Fratios.py)", "0.08027", f"{y[-1]:.5f}",
           abs(y[-1] - 0.08027) <= 1e-4)
    record("|F3/F1| extrapolated to D = infinity (degree 2 in 1/n, 5 points) vs the inner-problem prediction",
           "0.055 (README: 0.0552-0.0549 from 16 points)", f"{ext:.4f}", abs(ext - 0.055) <= 0.003)
else:
    record("checks on saved solutions", "hpkernel installed", "hpkernel not installed (build with hp/build.sh)", None)

# ---------------------------------------------------------------------------------------------------------------
print(f"\n== Summary ({time.time() - T0:.0f} s)")
for name, exp, got, st in results:
    print(f"  {st:4s}  {name}")
n_fail = sum(1 for r in results if r[3] == "FAIL")
n_skip = sum(1 for r in results if r[3] == "SKIP")
print(f"\n{len(results) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped")
sys.exit(1 if n_fail else 0)
