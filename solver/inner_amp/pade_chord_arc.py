#!/usr/bin/env python3
"""Chord against arc from sigma_0..sigma_22: three discriminators of S_1's Borel geometry, judged on plants (issue #548).

#541's arc fraction R_k of the [k/k] poles separated its two plants by only 0.11-0.15 at k <= 11. The candidates here
use the same 23 coefficients (a_k = k! sigma_k, f(u) = sum a_k u^k, u = 1/t) and the same chord frame (P0 -> kappa_*,
coordinates s along it and d off it, the arc's offset profile d_arc(s); pade_geometry.Frame).

  H_DA   fraction of the 11 members of pade_farend.ENSEMBLE (second-order inhomogeneous differential approximants;
         Guttmann and Joyce 1972, J. Phys. A 5, L81; Hunter and Baker 1979, Phys. Rev. B 19, 3808) with a zero of Q2,
         a singular point of the approximant, on the middle of the arc: 0.3 <= s <= 0.7 and |d - d_arc(s)| <= 0.04.
         There the arc lies 0.09-0.11 off the chord, so such a zero is nearer the arc than the chord.
  H_HP   the same for the zeros of the discriminant p1^2 - 4 p0 p2 of quadratic Hermite-Pade approximants
         p0 + p1 f + p2 f^2 = O(u^n) (the branch points of the approximant; Shafer 1974, SIAM J. Numer. Anal. 11, 447),
         degrees (m, m, m) and (m - 1, m, m) for m = 6, 7, matched to a_0..a_(n-1) with n <= 23.
  R_w    #541's arc fraction pooled over k = 6..11 with the poles weighted by |residue|:
         sum |r| d d_arc(s) / sum |r| d_arc(s)^2 over the geometry poles (pade_geometry.geometry) with 0.05 <= s <= 0.95.

Each statistic is near 0 for a chord (no singularity off it) and near 1 when the singularities lie on the arc.

    python3 pade_chord_arc.py plants OUTDIR [--mutate=mirror|sigma7] [--noise_scale=F] [--allow_unraw]
    python3 pade_chord_arc.py bound OUTDIR [--allow_unraw]         noise-scale study, F = 1, 0.1, ..., 1e-4
    python3 pade_chord_arc.py real OUTDIR [--allow_unraw]          only after plants passed (reads plants.json)

Noise: relative complex Gaussian noise of the measured per-order errors (sigma_err_fit.py, as pade_geometry.py
--noise=measured), times F; the noise spread is the largest shift of a statistic over NSEED seeds. Not a bound.
Exit 1 when a frozen criterion fails.
"""
import json
import os
import sys
import time

import mpmath as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pade_farend as F                                                # noqa: E402

pg = F.pg
NSEED = 10
MID = (0.3, 0.7)
TOL = 0.04
HP_DEGS = [(m - drop, m, m) for m in (6, 7) for drop in (0, 1)]
FROZEN = {                          # written before the plants run of this script and before any real run
    "statistics": "H_DA, H_HP, R_w as in the docstring",
    "separation": "per node wf 0.99, 1.00, 1.01 and per statistic: S = X(arcjump) - X(chord2) >= 0.3 and S >= 3 nu, "
                  "nu the larger noise spread of X on the two plants at the measured per-order errors",
    "pass": "a statistic passes if the separation holds at all three nodes and its mutation fires",
    "mutation": "--mutate=mirror (the arc's profile reflected across the chord, d_arc -> -d_arc, in H_DA, H_HP and R_w) "
                "must give S < 0.3 at every node; --mutate=sigma7 (sigma_7 x 1.01) is reported, not judged",
    "real": "a real run only for a statistic that passes; otherwise the task reports the bound",
    "bound": "the noise-scale study: the largest F in 1, 0.1, ..., 1e-4 at which a statistic's separation holds at all "
             "three nodes, i.e. how much more accurate sigma_13..sigma_22 must be",
}
PRIMARY = ("chord2", "arcjump")
CONTEXT = ("arcmid", "arcsteps", "arc")


# ---------------------------------------------------------------- quadratic Hermite-Pade
def hp_branch_points(sig, degs):
    """zeros t = 1/u of the discriminant of p0 + p1 f + p2 f^2 = O(u^n), deg p_i = degs[i], p2(0) = 1"""
    m0, m1, m2 = degs
    n = m0 + m1 + m2 + 2
    if n > F.N_COEF:
        return None
    a = [mp.factorial(k) * sig[k] for k in range(n)]
    a2 = [sum(a[j] * a[k - j] for j in range(k + 1)) for k in range(n)]
    rows, rhs = [], []
    for k in range(n):
        row = [1 if k == i else 0 for i in range(m0 + 1)]
        row += [a[k - i] if k >= i else 0 for i in range(m1 + 1)]
        row += [a2[k - i] if k >= i else 0 for i in range(1, m2 + 1)]
        rows.append(row)
        rhs.append(-a2[k])
    x = mp.lu_solve(mp.matrix(rows), mp.matrix(rhs))
    p0 = [x[i] for i in range(m0 + 1)]
    p1 = [x[m0 + 1 + i] for i in range(m1 + 1)]
    p2 = [mp.mpc(1)] + [x[m0 + m1 + 2 + i] for i in range(m2)]
    deg = max(2 * m1, m0 + m2)
    disc = [mp.mpc(0)] * (deg + 1)
    for i, c in enumerate(p1):
        for j, e in enumerate(p1):
            disc[i + j] += c * e
    for i, c in enumerate(p0):
        for j, e in enumerate(p2):
            disc[i + j] -= 4 * c * e
    while len(disc) > 1 and disc[-1] == 0:
        disc.pop()
    return [1 / u for u in mp.polyroots(disc[::-1], maxsteps=400, extraprec=4 * mp.mp.prec) if u != 0]


# ---------------------------------------------------------------- statistics
def mid_arc_hit(ts, fr, sgn):
    for t in ts or []:
        s, d = fr.coords(t)
        if MID[0] <= s <= MID[1] and abs(d - sgn * fr.d_arc(s)) <= TOL:
            return True
    return False


def stats(sig, fr, sgn=1):
    """(H_DA, H_HP, R_w) of sigma_0..sigma_22 in the frame fr; sgn = -1 mirrors the arc's profile"""
    mp.mp.dps = F.DPS_DA
    sig = [mp.mpc(z) for z in sig[:F.N_COEF]]
    mem = F.members(2)
    h_da = sum(mid_arc_hit(F.da_zeros(sig, degs, lp), fr, sgn) for degs, lp in mem) / len(mem)
    h_hp = sum(mid_arc_hit(hp_branch_points(sig, d), fr, sgn) for d in HP_DEGS) / len(HP_DEGS)
    mp.mp.dps = pg.DPS_PADE
    sig120 = [mp.mpc(z) for z in sig]
    num = den = 0.0
    for k in pg.KS:
        for p in pg.pade(sig120, k):
            t = complex(p["t"])
            s, d = fr.coords(t)
            if p["dz"] < pg.TOL_DOUBLET or not (0.3 < abs(t) < 3 and -0.1 <= s <= 1.1 and abs(d) <= 0.5):
                continue
            if 0.05 <= s <= 0.95:
                w = float(abs(p["res"]))
                da = sgn * fr.d_arc(s)
                num += w * d * da
                den += w * da * da
    return h_da, h_hp, (num / den if den > 0 else None)


NAMES = ("H_DA", "H_HP", "R_w")


def with_noise(sig, fr, wf, scale, sgn=1):
    """clean statistics and their noise spreads"""
    clean = stats(sig, fr, sgn)
    pg.NOISE_MODE = "measured"
    eps = [e * scale for e in pg.noise_eps(wf)]
    spread = [0.0] * 3
    vals = []
    for seed in range(NSEED):
        x = stats(pg.noisy([mp.mpc(z) for z in sig[:F.N_COEF]], 548000 + seed, eps), fr, sgn)
        vals.append(x)
        for i in range(3):
            if x[i] is not None and clean[i] is not None:
                spread[i] = max(spread[i], abs(x[i] - clean[i]))
    return clean, spread, vals


def plant_sigma(wf, kind, real):
    pl = F.FarPlant(wf, kind, real[wf])
    return pl.sigma(), max(pl.resid)


def mutate(sig, mut):
    return pg.mutate(sig, mut) if mut == "sigma7" else sig


def run_plants(outdir, mut, scale=1.0, kinds=PRIMARY + CONTEXT, quiet=False):
    mp.mp.dps = F.DPS_DA
    real = pg.load_scan(pg.REAL["J1250"])
    sgn = -1 if mut == "mirror" else 1
    rec = {"frozen": FROZEN, "mutate": mut, "noise_scale": scale, "plants": {}}
    for wf in pg.NODES:
        fr = pg.Frame(wf)
        for kind in kinds if wf == 1.0 else PRIMARY:
            t0 = time.time()
            sig, res = plant_sigma(wf, kind, real)
            clean, spread, vals = with_noise(mutate(sig, mut), fr, wf, scale, sgn)
            rec["plants"].setdefault(kind, {})[wf] = {"clean": clean, "spread": spread, "noisy": vals,
                                                       "fit_resid_max": res}
            if not quiet:
                print(f"wf {wf} {kind:9s} " + "  ".join(f"{n} {'-' if c is None else f'{c:+.3f}'} (noise {s:.2f})"
                                                       for n, c, s in zip(NAMES, clean, spread))
                      + f"  ({time.time() - t0:.0f}s)")
                sys.stdout.flush()
    verdict = {}
    for i, n in enumerate(NAMES):
        per = {}
        for wf in pg.NODES:
            c, a = rec["plants"]["chord2"][wf], rec["plants"]["arcjump"][wf]
            if c["clean"][i] is None or a["clean"][i] is None:
                per[wf] = {"S": None, "nu": None, "ok": False}
                continue
            S = a["clean"][i] - c["clean"][i]
            nu = max(a["spread"][i], c["spread"][i])
            per[wf] = {"S": S, "nu": nu, "ok": S >= 0.3 and S >= 3 * nu}
        verdict[n] = {"nodes": per, "separates": all(p["ok"] for p in per.values()),
                      "S_lt_0.3_everywhere": all(p["S"] is None or p["S"] < 0.3 for p in per.values())}
        if not quiet:
            parts = []
            for wf, p in per.items():
                sv = "-" if p["S"] is None else f"{p['S']:+.3f}"
                nv = "-" if p["nu"] is None else f"{p['nu']:.3f}"
                parts.append(f"wf {wf} S {sv} nu {nv} {'OK' if p['ok'] else 'BAD'}")
            print(f"{n}: " + "; ".join(parts) + f"  -> {'separates' if verdict[n]['separates'] else 'does not separate'}")
    rec["verdict"] = verdict
    name = "plants" + (f"_mutate_{mut}" if mut else "") + (f"_noise{scale:g}" if scale != 1.0 else "") + ".json"
    json.dump(pg.jsonable(rec), open(os.path.join(outdir, name), "w"), indent=1)
    return rec


def run_bound(outdir):
    out = {}
    for scale in (1.0, 0.1, 0.01, 1e-3, 1e-4):
        rec = run_plants(outdir, None, scale, kinds=PRIMARY, quiet=True)
        out[scale] = {n: rec["verdict"][n] for n in NAMES}
        print(f"noise x {scale:g}: " + "; ".join(
            f"{n} " + " ".join(f"S {p['S']:+.2f}/nu {p['nu']:.2f}" if p["S"] is not None else "-"
                               for p in rec["verdict"][n]["nodes"].values())
            + (" separates" if rec["verdict"][n]["separates"] else "") for n in NAMES))
        sys.stdout.flush()
    best = {n: max((s for s in out if out[s][n]["separates"]), default=None) for n in NAMES}
    print("largest noise scale at which each statistic separates at all three nodes:", best)
    json.dump(pg.jsonable({"frozen": FROZEN, "scales": out, "largest_separating_scale": best}),
              open(os.path.join(outdir, "bound.json"), "w"), indent=1)
    return any(v is not None for v in best.values())


def selftest(mut=None):
    """(1) the Hermite-Pade discriminant of f = ((1 - k1 u)(1 - k2 u))^(1/2) exp(u/5) has zeros at both branch points
    1/k1, 1/k2; (2) H_DA and H_HP count a planted singularity on the middle of the arc: a chord density plus a
    log branch point at the arc's s = 1/2 (a density step on a short arc piece) gives 1, the chord density alone 0."""
    ok = True
    mp.mp.dps = F.DPS_DA
    k1, k2 = mp.mpc("1.28", "1.08"), mp.mpc("0.75", "0.16")
    def binom_series(k, al):        # (1 - k u)^al
        return [mp.binomial(al, n) * (-k) ** n for n in range(F.N_COEF)]
    b1, b2 = binom_series(k1, mp.mpf(1) / 2), binom_series(k2, mp.mpf(1) / 2)
    ex = [mp.mpf(1) / 5 ** n / mp.factorial(n) for n in range(F.N_COEF)]
    a1 = pg.pp.polymul(pg.pp.polymul(b1, b2, F.N_COEF), ex, F.N_COEF)
    if mut == "selftest_hp":
        a1[12] *= mp.mpf("1.01")
    sig = [a1[k] / mp.factorial(k) for k in range(F.N_COEF)]
    ts = hp_branch_points(sig, (7, 7, 7))
    e = max(min(abs(t - k) for t in ts) for k in (k1, k2))
    print(f"selftest 1: HP (7,7,7) discriminant zeros at both branch points to {float(e):.1e} (pass < 1e-3)")
    ok &= e < 1e-3
    # 2: chord2's density, and the same plus a step of weight 1 on the arc piece s in [0.5, 0.6]
    real = pg.load_scan(pg.REAL["J1250"])
    fr = pg.Frame(1.0)
    pl = F.FarPlant(1.0, "chord2", real[1.0])
    base = pl.sigma()
    mp.mp.dps = pg.DPS_PLANT
    arc = pg.pp.Arc(1.0, "arc")
    w = mp.mpf(1) if mut != "selftest_step" else mp.mpf(0)
    step = [mp.quad(lambda s: arc.g(s) ** k * arc.dg(s), [0.5, 0.6]) / mp.factorial(k) for k in range(F.N_COEF)]
    sig2 = [b + w * st / abs(step[0]) * mp.mpf("0.05") * abs(base[0]) for b, st in zip(base, step)]
    h0, h1 = stats(base, fr)[:2], stats(sig2, fr)[:2]
    print(f"selftest 2: chord2 H_DA {h0[0]:.2f} H_HP {h0[1]:.2f} (pass <= 0.2); with a planted step on the arc at "
          f"s = 0.5-0.6: H_DA {h1[0]:.2f} H_HP {h1[1]:.2f} (pass >= 0.5 for H_DA)")
    ok &= h0[0] <= 0.2 and h1[0] >= 0.5
    print("selftest:", "PASS" if ok else "FAIL")
    return ok


def main():
    argv = sys.argv[1:]
    from raw_out import pop_allow, require_raw
    allow = pop_allow(argv)
    mut = next((a.split("=", 1)[1] for a in argv if a.startswith("--mutate=")), None)
    scale = float(next((a.split("=", 1)[1] for a in argv if a.startswith("--noise_scale=")), 1.0))
    argv = [a for a in argv if not a.startswith("--")]
    if not argv or argv[0] == "selftest":
        sys.exit(0 if selftest(mut) else 1)
    stage, outdir = argv[0], argv[1]
    require_raw(os.path.join(outdir, "x.json"), allow=allow)
    os.makedirs(outdir, exist_ok=True)
    if stage == "plants":
        rec = run_plants(outdir, mut, scale)
        if mut == "mirror":
            ok = all(v["S_lt_0.3_everywhere"] for v in rec["verdict"].values())
            print("mirror mutation: S < 0.3 at every node for every statistic:", ok)
        else:
            ok = any(v["separates"] for v in rec["verdict"].values())
        sys.exit(0 if ok else 1)
    if stage == "bound":
        sys.exit(0 if run_bound(outdir) else 1)
    if stage == "real":
        v = json.load(open(os.path.join(outdir, "plants.json")))["verdict"]
        passing = [n for n in NAMES if v[n]["separates"]]
        if not passing:
            print("no statistic passed on the plants: no real run (frozen rule)")
            sys.exit(1)
        raise SystemExit("real stage: to be written only if a statistic passes")


if __name__ == "__main__":
    main()
