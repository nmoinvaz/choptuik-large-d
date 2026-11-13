#!/usr/bin/env python3
"""The D -> infinity behaviour of lam_g(D) (hp/data/lam_gauge.csv, from hp/lam_gauge.py table): least-squares
polynomials in t = 1/n (n = D - 2) or t = 1/D, lam_g = c0 + c1 t + c2 t^2 + ... + c_d t^d, in mpmath.

    python3 hp/lam_gauge_fit.py planted [--mutate=limit]     the planted-truth test (run first; exits 1 on a fail)
    python3 hp/lam_gauge_fit.py fit                          the fit of the real backbone

Post-hoc options of planted (after the real fit, reported as such): --sigma=S (the noise), --junk=D:offset,... (offsets
at grid points, as the I461 states' near the frontier), --step=D0:offset (an offset below D0, as between layouts).
--csv=PATH reads another table. --dlos=a,b,... (fit) changes the lower ends, a diagnostic outside the selection.

c0 and c1 are the same in both variables (1/n = 1/D + 2/D^2 + ...). c2 is quoted in 1/n, the 1/D fits' c2 converted
by c2(1/n) = c2(1/D) - 2 c1.

The points are the table's backbone rows (one state per D), thinned to D at least 1 apart from D = 20 up, so that the
dense pinned stretch near D 70 does not set the fit. The fit choices are the variable (1/n, 1/D), the lower end
D_lo (20, 30, 40, 60) and the degree d (3 to 14). Free fits leave c0 free, constrained ones fix c0 = 1. For each
variable and D_lo, the selection keeps the degrees whose rms residual is within a factor 3 of the smallest rms there.
The quoted value is the median over the selected fits and the quoted +- the half-width of their [min, max]: a spread
over fit choices, not a bound. The data's own scatter is far below the fits' residuals (the table's same-D spreads are
at most 3.8e-13).

Planted truth (frozen before the first fit of the real data, 2026-11-13). Synthetic lam(t) on the real thinned D grid,
t = 1/n, with c0 = 1, c1 = -7.4, c2 = 95 and a tail t^3 h(t) whose structure the polynomial does not contain, of the
size of the real function's high orders (from a coarse look at n(1 - lam_g), about 7.4 - 94/n). Gaussian noise of
sigma 1e-12, fixed seed.
    T1  h = -1200/(1 + t/0.077)                     a pole at t = -0.077 (n = -13)
    T2  h = -1200 (1 + t/0.06)^(-1/2)               a square-root branch point at t = -0.06
    T3  h = -1200 Re 1/(1 - t/ts), ts = 0.08 e^{2 pi i/3}   a complex pole pair at |t| = 0.08
    T4  h = -1200 log(1 + t/0.05)/(t/0.05)          a log branch point at t = -0.05
    T5  lam = 1 - 7.4 t/(1 + 12.84 t)               one pole, every order (c2 = 95.016)
    T6  T1 plus 30 t^2 log t                        non-analytic at t = 0: outside the model class (reported only)
Pass criteria (frozen): for T1-T5, in the free fits the planted c0, c1 and c2 lie inside the quoted [min, max], and in
the constrained fits (c0 = 1) the planted c1 and c2 do. The control --mutate=limit plants c0 = 1 + 1e-5 in T1-T5 and
grades against c0 = 1: the free fits' c0 range must then exclude 1 at every plant, so the test can see a limit 1e-5
away from 1 (exit 1).
"""
import csv
import os
import random
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "lam_gauge.csv")
mp.mp.dps = 40
VARS = ("n", "D")
DLOS = (20, 30, 40, 60)
DEGS = range(3, 15)
PLATEAU = 3
SPACING = 1.0
SIGMA = 1e-12
REF_DEG = 14         # the reference fit for the residuals: the selected c0 = 1 fit in 1/n from D = 20
BANDS = [(20, 40), (40, 67), (67, 74), (74, 120), (120, 180), (180, 226), (226, 246)]
C0, C1, C2, C3 = mp.mpf(1), mp.mpf("-7.4"), mp.mpf(95), mp.mpf(-1200)
TS = mp.mpf("0.08") * mp.expj(2 * mp.pi / 3)
TAILS = {
    "T1": lambda t: C3 / (1 + t / mp.mpf("0.077")),
    "T2": lambda t: C3 / mp.sqrt(1 + t / mp.mpf("0.06")),
    "T3": lambda t: C3 * mp.re(1 / (1 - t / TS)),
    "T4": lambda t: C3 * mp.log(1 + t / mp.mpf("0.05")) / (t / mp.mpf("0.05")),
}
B5 = mp.mpf("12.84")


def planted(name, t, c0):
    if name == "T5":
        return c0 - mp.mpf("7.4") * t / (1 + B5 * t)
    if name == "T6":
        return planted("T1", t, c0) + 30 * t ** 2 * mp.log(t)
    return c0 + C1 * t + C2 * t ** 2 + t ** 3 * TAILS[name](t)


def planted_coeffs(name, c0):
    if name == "T5":
        return c0, mp.mpf("-7.4"), mp.mpf("7.4") * B5
    return c0, C1, C2


def backbone(path=None):
    path = path or CSV
    rows = [r for r in csv.DictReader(x for x in open(path) if not x.startswith("#")) if r["role"] == "backbone"]
    pts = sorted((mp.mpf(r["D_str"]) if r["D_str"] else mp.mpf(float(r["D"])), mp.mpf(r["lam_g"])) for r in rows)
    out, last = [], None
    for D, lam in pts:
        if D < min(DLOS):
            continue
        if last is None or D - last >= SPACING:
            out.append((D, lam))
            last = D
    return out


def superseded(c, path=None):
    """(D, lam_g - fit, state) of the table's superseded rows (delta_all.csv's I461 states past D 225.13), against the
    polynomial c in 1/n"""
    rows = [r for r in csv.DictReader(x for x in open(path or CSV) if not x.startswith("#")) if r["role"] == "superseded"]
    out = []
    for r in rows:
        D = mp.mpf(r["D_str"]) if r["D_str"] else mp.mpf(float(r["D"]))
        out.append((D, mp.mpf(r["lam_g"]) - mp.polyval(c[::-1], 1 / (D - 2)), r["state"]))
    return sorted(out)


def lsq(ts, ys, d, fixed0=None):
    """coefficients c0..cd (c0 = fixed0 when given) and the rms residual; columns scaled by t_max^k"""
    tmax = max(ts)
    k0 = 0 if fixed0 is None else 1
    A = mp.matrix(len(ts), d + 1 - k0)
    b = mp.matrix(len(ts), 1)
    for i, (t, y) in enumerate(zip(ts, ys)):
        for k in range(k0, d + 1):
            A[i, k - k0] = (t / tmax) ** k
        b[i] = y - (fixed0 if fixed0 is not None else 0)
    x, res = mp.qr_solve(A, b)
    c = ([fixed0] if fixed0 is not None else []) + [x[k - k0] / tmax ** k for k in range(k0, d + 1)]
    return c, res / mp.sqrt(len(ts))


def fits(pts, fixed0=None):
    """every fit choice: dict(var, Dlo, d, c, rms), and the selection"""
    allf = []
    for var in VARS:
        for Dlo in DLOS:
            sel = [(D, y) for D, y in pts if D >= Dlo]
            ts = [1 / (D - 2) if var == "n" else 1 / D for D, _ in sel]
            ys = [y for _, y in sel]
            row = []
            for d in DEGS:
                if d + 1 > len(ts) // 4:
                    continue
                c, rms = lsq(ts, ys, d, fixed0)
                if var == "D":          # c2 and c3 in 1/n: 1/D = t - 2t^2 + 4t^3 - ..., t = 1/n
                    c = c[:2] + [c[2] - 2 * c[1], c[3] - 4 * c[2] + 4 * c[1]] + c[4:]
                row.append(dict(var=var, Dlo=Dlo, d=d, c=c, rms=rms, npts=len(ts)))
            best = min(f["rms"] for f in row)
            for f in row:
                f["sel"] = f["rms"] <= PLATEAU * best
            allf += row
    return allf


def summary(allf, k):
    v = sorted(f["c"][k] for f in allf if f["sel"])
    lo, hi = v[0], v[-1]
    med = v[len(v) // 2]
    return med, lo, hi


def report(allf, label, ks=(0, 1, 2)):
    out = {}
    for k in ks:
        med, lo, hi = summary(allf, k)
        out[k] = (med, lo, hi)
        print(f"  {label} c{k}: {mp.nstr(med, 12)} +- {mp.nstr((hi - lo) / 2, 2)}  "
              f"[{mp.nstr(lo, 12)}, {mp.nstr(hi, 12)}]  ({sum(1 for f in allf if f['sel'])} fits)")
    return out


def cmd_planted(argv):
    mut = "--mutate=limit" in argv
    sigma = SIGMA
    junk = {}
    step = (0, 0)
    for a in argv:
        if a.startswith("--sigma="):
            sigma = float(a.split("=", 1)[1])
        if a.startswith("--step="):           # an offset below D0, as between the backbone's layouts
            step = (float(a.split("=", 1)[1].split(":")[0]), mp.mpf(a.split("=", 1)[1].split(":")[1]))
        if a.startswith("--junk="):
            junk = {float(x.split(":")[0]): mp.mpf(x.split(":")[1]) for x in a.split("=", 1)[1].split(",")}
    grid = [D for D, _ in backbone()]
    c0p = mp.mpf(1) + (mp.mpf("1e-5") if mut else 0)
    ok_all = True
    for name in ("T1", "T2", "T3", "T4", "T5", "T6"):
        rng = random.Random(622)
        pts = [(D, planted(name, 1 / (D - 2), c0p) + mp.mpf(rng.gauss(0, sigma))
                + junk.get(round(float(D), 3), 0) + (step[1] if D < step[0] else 0)) for D in grid]
        want = planted_coeffs(name, mp.mpf(1))          # graded against c0 = 1 (the mutation plants 1 + 1e-5)
        print(f"{name}{' (mutation limit: planted c0 = 1 + 1e-5)' if mut else ''} (sigma {sigma:g}"
              f"{', junk at ' + str(sorted(junk)) if junk else ''}"
              f"{', step ' + mp.nstr(step[1], 2) + ' below D ' + str(step[0]) if step[0] else ''}): planted c0 1, c1 "
              f"{mp.nstr(want[1], 6)}, c2 {mp.nstr(want[2], 6)}")
        free = report(fits(pts), "free")
        okf = all(free[k][1] <= want[k] <= free[k][2] for k in (0, 1, 2))
        if name == "T6":
            print("  T6 is outside the model class: reported, not graded")
            continue
        if mut:
            ok = okf                                     # the control: must not pass
        else:
            con = report(fits(pts, fixed0=mp.mpf(1)), "c0=1", ks=(1, 2))
            okc = all(con[k][1] <= want[k] <= con[k][2] for k in (1, 2))
            ok = okf and okc
        print(f"  {name}: {'pass' if ok else 'FAIL'}")
        ok_all &= ok
    print("planted:", "PASS" if ok_all else "FAIL", "(mutation limit)" if mut else "")
    return 0 if ok_all else 1


def cmd_fit(argv):
    global DLOS
    for a in argv:
        if a.startswith("--dlos="):           # a diagnostic, not the quoted selection
            DLOS = tuple(float(x) for x in a.split("=", 1)[1].split(","))
    pts = backbone()
    print(f"{len(pts)} points, D {mp.nstr(pts[0][0], 8)} to {mp.nstr(pts[-1][0], 8)}")
    free = fits(pts)
    for f in free:
        if f["sel"]:
            print(f"  free {f['var']} D>={f['Dlo']} d {f['d']} ({f['npts']} pts): rms {mp.nstr(f['rms'], 2)}  "
                  f"c0 {mp.nstr(f['c'][0], 12)} c1 {mp.nstr(f['c'][1], 10)} c2 {mp.nstr(f['c'][2], 8)}")
    fr = report(free, "free")
    con = fits(pts, fixed0=mp.mpf(1))
    for f in con:
        if f["sel"]:
            print(f"  c0=1 {f['var']} D>={f['Dlo']} d {f['d']}: rms {mp.nstr(f['rms'], 2)}  "
                  f"c1 {mp.nstr(f['c'][1], 10)} c2 {mp.nstr(f['c'][2], 8)} c3 {mp.nstr(f['c'][3], 6)}")
    report(con, "c0=1", ks=(1, 2, 3))
    ts = [1 / (D - 2) for D, _ in pts]
    c, _ = lsq(ts, [y for _, y in pts], REF_DEG, fixed0=mp.mpf(1))
    res = [(D, y - mp.polyval(c[::-1], t)) for (D, y), t in zip(pts, ts)]
    print(f"residuals of the reference fit (c0 = 1, 1/n, D >= {DLOS[0]}, degree {REF_DEG}):")
    for lo, hi in BANDS:
        r = [float(e) for D, e in res if lo <= D < hi]
        if r:
            print(f"  D {lo}-{hi}: {len(r)} pts, rms {(sum(x * x for x in r) / len(r)) ** 0.5:.1e}, "
                  f"max {max(abs(x) for x in r):.1e}")
    for D, e in sorted(res, key=lambda x: -abs(x[1]))[:3]:
        print(f"  largest: D {mp.nstr(D, 10)} residual {mp.nstr(e, 3)}")
    for D, lam, st in superseded(c):
        print(f"  superseded {st}: D {mp.nstr(D, 10)}, off this fit by {mp.nstr(lam, 3)}")
    inside = fr[0][1] <= 1 <= fr[0][2]
    print(f"c0 = 1 {'inside' if inside else 'OUTSIDE'} the free fits' range")
    return 0


def main():
    global CSV
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    for a in sys.argv[2:]:
        if a.startswith("--csv="):            # another table, e.g. an earlier commit's
            CSV = a.split("=", 1)[1]
    if cmd == "planted":
        return cmd_planted(sys.argv[2:])
    if cmd == "fit":
        return cmd_fit(sys.argv[2:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
