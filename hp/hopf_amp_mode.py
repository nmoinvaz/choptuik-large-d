#!/usr/bin/env python3
"""Two tests of the Hopf reading of the large-D limit (notes/ECHO_ANALOGUES.md; README, "What repeats with that
period"): the echo's amplitude falls and its shape becomes one harmonic as D grows, as a limit cycle born at a
supercritical Hopf bifurcation would, with the bifurcation at 1/D = 0.

(1) The amplitude law. A supercritical Hopf cycle has amplitude^2 proportional to the distance to the bifurcation, so
    A = max |p1| of hp/data/echo_shapes.csv should go like D^(-1/2): A sqrt(D), A sqrt(D - 2) and A D are tabulated.
(2) The amplitude mode. Besides the phase mode (lambda = 0, the tau translation) such a cycle has a Floquet exponent
    lambda_amp = -2 mu, mu the distance to the bifurcation, which tends to 0 from below as D grows. The growth-mode
    determinant det M(lambda) (Solver.mode_det as in hp/gamma_dd.py: perturbations e^{lambda tau} of the stored state,
    Delta held fixed) is scanned on the real axis at each state and every sign change refined by bisection. A zero
    near 0 that moves toward 0 with D is the signature; its absence in the scanned range weighs against the reading.

    python3 hopf_amp_mode.py RAWDIR OUT.jsonl [--threads 2] [--lo -1.2] [--hi 0.5] [--step 0.025] [--fine]

    The uniform grid skips the pair of points across 0 (the phase mode), so a zero between its last negative point and 0
    is invisible to it; --fine scans the geometric grid -0.025 * 2^-i, i = 0..6 (down to -3.9e-4) instead, where a
    simple zero shows as a sign change with a dip of ln|det|. --law: only the amplitude law (no kernel), with its fits:
    A sqrt(D) = a + b / D over D >= 40 (the Hopf law with its first correction) and the exponents of the harmonic
    content, power fraction outside the first harmonic ~ D^s and shape deviation from a sine ~ D^s, over the same D
    (a Hopf cycle with an odd nonlinearity has both corrections of relative size A^2 ~ 1/D: s = -2 for the power
    fraction, whose first term is the third harmonic, A^3, and s = -1 for the deviation).

RAWDIR is a clone of the raw-data repository; OUT.jsonl lies in a task folder of it (hp/raw_out.py).
"""
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from gamma_dd import best_solver  # noqa: E402
from raw_out import require_raw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# the states of hp/plot_echo.py (converged double-double states across D)
STATES = [
    "T79-win/down_D4.0000.npz",
    "T79-win/m64_D8.0000.npz",
    "T78/layoutB_D16.7600_M32.npz",
    "T77/relayout_test_D45.0507_n7.npz",
    "T12-win/scratch/t12/up17_D60.9626.npz",
    "T75/pin7_D73.2657.npz",
]


def amplitude_law():
    rows = [r for r in csv.DictReader(l for l in open(os.path.join(HERE, "data", "echo_shapes.csv")) if not l.startswith("#"))]
    out = []
    print("| D | A = max p1 | A sqrt(D) | A sqrt(D - 2) | A D | power outside the first harmonic | that over A^2 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        D, A, f1 = float(r["D"]), float(r["max_p1"]), float(r["first_harmonic_power_fraction"])
        rec = dict(D=D, A=A, A_sqrtD=A * math.sqrt(D), A_sqrtn=A * math.sqrt(D - 2), A_D=A * D, rest=1 - f1, rest_over_A2=(1 - f1) / A ** 2)
        rec["deviation"] = float(r["max_deviation_from_sine"])
        out.append(rec)
        print(f"| {D:g} | {A:.4g} | {rec['A_sqrtD']:.3f} | {rec['A_sqrtn']:.3f} | {rec['A_D']:.2f} | {rec['rest']:.4f} | {rec['rest_over_A2']:.4f} |")
    big = [r for r in out if r["D"] >= 40]
    x = np.array([1 / r["D"] for r in big]); D = np.array([r["D"] for r in big])
    b, a = np.polyfit(x, [r["A_sqrtD"] for r in big], 1)
    s_rest = np.polyfit(np.log(D), np.log([r["rest"] for r in big]), 1)[0]
    s_dev = np.polyfit(np.log(D), np.log([r["deviation"] for r in big]), 1)[0]
    s_A = np.polyfit(np.log(D), np.log([r["A"] for r in big]), 1)[0]
    fits = dict(D_min=40, n=len(big), A_sqrtD_limit=a, A_sqrtD_slope_1_over_D=b, A_exponent=s_A, rest_exponent=s_rest, deviation_exponent=s_dev)
    print(f"over D >= 40 ({len(big)} states): A ~ D^{s_A:+.3f}; A sqrt(D) = {a:.3f} {b:+.2f}/D; power outside the first harmonic ~ D^{s_rest:+.3f} "
          f"(Hopf: -2); deviation from a sine ~ D^{s_dev:+.3f} (Hopf: -1)")
    return dict(rows=out, fits=fits)


def scan(s, zh, zl, grid, log):
    cache = {}

    def sd(lam):
        if lam not in cache:
            t = time.time()
            sg, la, pmin, pmax = s.mode_det(zh, zl, float(lam), 0.0)
            cache[lam] = (sg, la)
            log(f"    lam {lam:+.6f}: sign {sg:+.0f} ln|det| {la:.4f} pivots {pmin:.1e}..{pmax:.1e} ({time.time() - t:.1f} s)")
        return cache[lam]

    zeros = []
    for a, b in zip(grid, grid[1:]):
        if a < 0 < b:
            continue                                   # the sign change across 0 is the phase mode
        sa, sb = sd(a)[0], sd(b)[0]
        if sa != sb:
            x, y = a, b
            for _ in range(40):
                m = 0.5 * (x + y)
                if sd(m)[0] == sa:
                    x = m
                else:
                    y = m
                if y - x < 1e-9:
                    break
            zeros.append(0.5 * (x + y))
            log(f"  zero of det M at lam = {0.5 * (x + y):.9f}")
    return zeros, [(g, cache[g][0], cache[g][1]) for g in grid], len(cache)


def main():
    args = list(sys.argv[1:])
    opt = dict(threads=2, lo=-1.2, hi=0.5, step=0.025)
    fine, law_only = "--fine" in args, "--law" in args
    args = [a for a in args if a not in ("--fine", "--law")]
    for k in list(opt):
        f = "--" + k
        if f in args:
            i = args.index(f); opt[k] = type(opt[k])(args[i + 1]); del args[i:i + 2]
    raw, out = args[0], args[1]
    require_raw(out)
    f = open(out, "a")
    law = amplitude_law()
    f.write(json.dumps(dict(kind="amplitude_law", **law)) + "\n"); f.flush()
    if law_only:
        f.close()
        return
    log = lambda m: print(m, flush=True)  # noqa: E731
    for path in STATES:
        d = np.load(os.path.join(raw, path)); meta = json.loads(str(d["meta"]))
        zh, zl = d["zh"].astype(float), d["zl"].astype(float)
        meta.setdefault("kc", 146); meta.setdefault("ks", 130); meta.setdefault("kt", 36)
        t0 = time.time()
        s, F, kf = best_solver(meta, zh, zl, opt["threads"], force="auto")
        if fine:
            grid = sorted(-0.025 * 2.0 ** -i for i in range(7))
            where = f"the fine grid {grid[0]:.3g} .. {grid[-1]:.3g}"
        else:
            grid = [opt["lo"] + i * opt["step"] for i in range(int(round((opt["hi"] - opt["lo"]) / opt["step"])) + 1)]
            grid = [g for g in grid if abs(g) > 1e-9]          # lambda = 0 is the exact phase mode
            where = f"[{opt['lo']}, {opt['hi']}] step {opt['step']}"
        log(f"{path}: D {meta['D']}, M {meta['M']}, kfreeze {kf}, |F| {F:.1e}; scanning det M(lam) on {where}")
        zeros, grid, nev = scan(s, zh, zl, grid, log)
        rec = dict(kind="scan", state=path, D=meta["D"], M=meta["M"], kfreeze=kf, F=F, lo=opt["lo"], hi=opt["hi"], step=opt["step"],
                   fine=fine, zeros=zeros, grid=grid, evaluations=nev, seconds=round(time.time() - t0, 1))
        f.write(json.dumps(rec) + "\n"); f.flush()
        log(f"  D {meta['D']}: zeros {['%.6f' % z for z in zeros]} ({nev} evaluations, {time.time() - t0:.0f} s)")
    f.close()


if __name__ == "__main__":
    main()
