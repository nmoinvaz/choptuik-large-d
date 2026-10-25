#!/usr/bin/env python3
"""Is driver.delta_correction's dot product -y.F accurate, and how far off was the f64 dot product it used until
2026-10-25? (comms 2026-10-25, T57: for a planted residual that cancels in y.F, the f64 dot product of y, |y| up to
1e15 at large D, lost the result.)

    cd hp && python3 delta_correction_check.py --selftest
    cd hp && python3 delta_correction_check.py [STATE.npz ...] [--run=T48-up70] [--csv=data/delta_dd.csv] [--every=1]
                                               [--threads=N] [--out=data/large_d/delta_correction_check.jsonl]

--selftest (no kernel calls: a stand-in solver whose lu_solve_t returns a planted double-double y): delta_correction
against the exact -y.F (fractions.Fraction of the double-double inputs), L = 1409 (M 32, 10 nodes), |y_i| from 1e-2 to
1e15 and |F_i| about 1e-22 as at D 67, in two cases:
  a. no cancellation (every y_i F_i > 0);
  b. cancellation: y_{2k+1} = y_{2k} times 0.5 .. 2 and F_{2k+1} = -y_{2k} F_{2k} / y_{2k+1} (rounded), so that y.F
     is about 2^-53 of sum |y_i F_i|.
  Pass in both: |c - exact| <= 2^-52 |exact| + 2^-100 sum |y_i F_i| (the result's rounding to f64 and the
  double-double dot product's error). Controls, each of which must fail that bound: the f64 dot product used before
  (f64_correction) in case b, and the exact value times (1 + 2^-49) in case a. Exit 1 otherwise.
States (a measurement, no pass/fail): at each saved state (default: those of the run's rows in --csv, every
--every-th; the solver as qdh_delta_check.py's: the state's kt, kc, ks, kfreeze 13, the Taylor plan at tol_step 1e-32
and hmax 0.05), F and the LU from Solver.newton at the state; delta_correction, the f64 dot product, their difference,
sum |y_i F_i| and the row's dDelta_est in the csv.
"""
import csv
import glob
import json
import os
import re
import sys
import time
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpkernel.driver import delta_correction  # noqa: E402

U = 2.0 ** -53


def f64_correction(s, Fh, Fl):
    """delta_correction as it was until 2026-10-25: the f64 dot product of the rounded vectors"""
    L = s.size
    e = np.zeros(L)
    e[-1] = 1.0
    yh, yl = s.lu_solve_t(e, np.zeros(L))
    return float(-(np.asarray(yh) + np.asarray(yl)) @ (np.asarray(Fh) + np.asarray(Fl)))


class Planted:
    """a stand-in for the Solver: lu_solve_t(e_Delta) returns the planted y"""

    def __init__(self, yh, yl):
        self.size, self.y = len(yh), (yh, yl)

    def lu_solve_t(self, bh, bl):
        assert bh[-1] == 1.0 and not np.any(bh[:-1]) and not np.any(bl)
        return self.y


def exact(yh, yl, Fh, Fl):
    """-y.F exactly, and sum |y_i F_i|, as Fractions"""
    tot, mag = Fraction(0), Fraction(0)
    for a, b, c, d in zip(yh.tolist(), yl.tolist(), Fh.tolist(), Fl.tolist()):
        p = (Fraction(a) + Fraction(b)) * (Fraction(c) + Fraction(d))
        tot += p
        mag += abs(p)
    return -tot, mag


def within(c, ex, mag):
    err = abs(Fraction(c) - ex)
    return err <= Fraction(2 * U) * abs(ex) + Fraction(2.0 ** -100) * mag, float(err)


def planted(case, L=1409, seed=1):
    rng = np.random.default_rng(seed)
    yh = 10.0 ** rng.uniform(-2, 15, L) * rng.choice([-1.0, 1.0], L)
    Fh = 1e-22 * rng.uniform(0.1, 1, L) * rng.choice([-1.0, 1.0], L)
    if case == "a":
        Fh = np.abs(Fh) * np.sign(yh)
    else:
        n = L // 2
        ev, od = slice(0, 2 * n, 2), slice(1, 2 * n, 2)
        yh[od] = yh[ev] * rng.uniform(0.5, 2, n) * rng.choice([-1.0, 1.0], n)
        Fh[od] = -yh[ev] * Fh[ev] / yh[od]
        Fh[2 * n:] = 0.0                         # L odd: the unpaired entry would carry the sum
    yl = yh * rng.uniform(-1, 1, L) * U / 2      # |lo| below half an ulp of hi
    Fl = Fh * rng.uniform(-1, 1, L) * U / 2
    return yh, yl, Fh, Fl


def selftest():
    bad = []
    for case in ("a", "b"):
        yh, yl, Fh, Fl = planted(case)
        s = Planted(yh, yl)
        ex, mag = exact(yh, yl, Fh, Fl)
        c, c64 = delta_correction(s, Fh, Fl), f64_correction(s, Fh, Fl)
        ok, err = within(c, ex, mag)
        ok64, err64 = within(c64, ex, mag)
        print(f"case {case}: exact {float(ex):+.6e}, sum |y_i F_i| {float(mag):.3e}; delta_correction error {err:.1e} "
              f"{'ok' if ok else 'WRONG'}; f64 dot product {c64:+.6e}, error {err64:.1e} "
              f"({'within' if ok64 else 'outside'} the bound)", flush=True)
        if not ok:
            bad.append(f"case {case}: delta_correction off by {err:.1e}")
        if case == "b" and ok64:
            bad.append("control: the f64 dot product passed case b (the case does not test the cancellation)")
        if case == "a":
            nudged = float(ex) * (1 + 2.0 ** -49)
            okn, errn = within(nudged, ex, mag)
            print(f"  control: exact (1 + 2^-49), error {errn:.1e}: {'passed (WRONG)' if okn else 'rejected, ok'}")
            if okn:
                bad.append("control: the nudged value passed case a")
    print("selftest", "ok" if not bad else "FAILED: " + "; ".join(bad), flush=True)
    return not bad


def states(paths, threads, out, rows):
    from hpkernel.driver import load, make_solver
    recs = []
    t0 = time.time()
    for i, path in enumerate(paths):
        zh, zl, meta = load(path)
        zh, zl = np.ascontiguousarray(zh), np.ascontiguousarray(zl)
        s = make_solver(meta["D"], meta["M"], nodes=meta["nodes"], kt=meta["kt"], kc=meta["kc"], ks=meta["ks"],
                        threads=threads, kfreeze=13)
        s.set_segs(s.plan(zh, zl, 1e-32, 0.05))
        _, _, Fh, Fl, _, _ = s.newton(zh, zl)
        c, c64 = delta_correction(s, Fh, Fl), f64_correction(s, Fh, Fl)
        L = s.size
        e = np.zeros(L)
        e[-1] = 1.0
        y = np.add(*s.lu_solve_t(e, np.zeros(L)))
        F = np.asarray(Fh) + np.asarray(Fl)
        r = rows.get(round(meta["D"], 4))
        rec = dict(state=os.path.basename(path), D=meta["D"], c=c, c_f64=c64, diff=c64 - c,
                   sum_abs=float(np.sum(np.abs(y * F))), y_max=float(np.abs(y).max()), F=float(np.abs(F).max()),
                   csv_dDelta_est=float(r["dDelta_est"]) if r else None)
        recs.append(rec)
        print(f"  D {meta['D']:.4f}: delta_correction {c:+.3e}, f64 {c64:+.3e}, diff {c64 - c:+.1e}, sum|y F| "
              f"{rec['sum_abs']:.1e}, |y| {rec['y_max']:.1e}, csv {rec['csv_dDelta_est'] or float('nan'):+.3e}  "
              f"({i + 1}/{len(paths)}, {time.time() - t0:.0f} s)", flush=True)
    d = np.array([abs(r["diff"]) for r in recs])
    rel = np.array([abs(r["diff"]) / max(abs(r["c"]), 1e-300) for r in recs])
    summ = dict(states=len(recs), max_diff=float(d.max()), median_diff=float(np.median(d)), max_rel=float(rel.max()),
                over_1pct=int(np.sum(rel > 0.01)), over_10pct=int(np.sum(rel > 0.1)), sec=round(time.time() - t0, 1),
                time=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"{len(recs)} states: |f64 - dd| max {summ['max_diff']:.1e}, median {summ['median_diff']:.1e}; relative max "
          f"{summ['max_rel']:.1e}, above 1%: {summ['over_1pct']}, above 10%: {summ['over_10pct']}", flush=True)
    with open(out, "a") as fo:
        fo.write(json.dumps(dict(summary=summ, rows=recs)) + "\n")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    if "selftest" in opts:
        sys.exit(0 if selftest() else 1)
    run = opts.get("run", "T48-up70")
    path = opts.get("csv", os.path.join(HERE, "data", "delta_dd.csv"))
    rows = {round(float(r["D"]), 4): r for r in csv.DictReader(open(path)) if r["run"] == run}
    if not args:
        files = {round(float(re.search(r"D([0-9.]+)\.npz", f).group(1)), 4): f
                 for f in glob.glob(os.path.join(HERE, "data", "large_d", "up70_D*.npz"))}
        args = [files[D] for D in sorted(rows) if D in files][::int(opts.get("every", 1))]
    states(args, int(opts.get("threads", 2)), opts.get("out", os.path.join(HERE, "data", "large_d",
                                                                           "delta_correction_check.jsonl")), rows)


if __name__ == "__main__":
    main()
