#!/usr/bin/env python3
"""eta_min and Delta_min of the sigma-model's Delta(eta) at D = 4 (#639 acceptance 4), from run_eta.py's records,
with the fit tested on planted data first (--selftest).

The fit, fixed before the data: Delta is analytic in eps = 1/eta at eps = 0 (eta = infinity is the l = 1 multiplet,
and kappa^2 = n eps enters the equations analytically), so Delta(eps) is fitted by polynomials in eps of degree
d = 4, 5, 6, 7 through the eight couplings eta = 5, 6, 7, 8.5, 10, 14, 20, inf (least squares below degree 7,
interpolation at 7), with every point weighted equally. The minimum of each polynomial on [1/10, 1/5] gives
eps_min (eta_min = 1/eps_min) and Delta_min. The quoted value is that of degree 6.

Two error bars, both reported, neither a rigorous enclosure:
  - the model spread: the range of eta_min (Delta_min) over d = 4..7 (an estimator range);
  - the resolution: the change of eta_min (Delta_min) between the fits to the M = 32 and the M = 48 data (the data's
    own resolution error, carried through the same fit).

--selftest runs the same code path on planted data with a known minimum: Delta_p(eps) = A - B eps + C eps^2 +
E eps^3 / (1 + 4 eps) with the shape of the real data (eta_min near 7, depth 5e-5), sampled at the same eps, with
noise of the size of the real M change (1e-9, seeded). Pass, written before the run: the degree-6 eta_min within
max(model spread, 1e-3) of the true one and Delta_min within max(spread, 1e-11). Controls that must fail: the
planted minimum moved outside [5, 10] (no minimum found in the window: the fit must say so, not return a point), and
one data point shifted by 1e-7 (the degree-6 fit's largest residual must exceed 20 times the noise, which the fit
reports as inconsistent data). Exit 1 when a check fails.

    python3 fit_min.py --records <raw>/I639-mac/data/sm_records.jsonl [--out fit.json]
    python3 fit_min.py --selftest
"""
import argparse
import json
import sys

import mpmath as mp
import numpy as np

ETAS = ["5", "6", "7", "8.5", "10", "14", "20", "inf"]
DEGREES = (4, 5, 6, 7)
QUOTED = 6
mp.mp.dps = 40


def eps_of(eta):
    return mp.mpf(0) if eta == "inf" else 1 / mp.mpf(eta)


def polyfit_mp(x, y, d):
    """least-squares polynomial of degree d in mpmath (normal equations via QR of the Vandermonde matrix)"""
    A = mp.matrix([[xi ** j for j in range(d + 1)] for xi in x])
    b = mp.matrix(y)
    if d + 1 == len(x):
        return list(mp.lu_solve(A, b))
    Q, R = mp.qr(A)
    c = mp.lu_solve(R[:d + 1, :d + 1], (Q.T * b)[:d + 1, 0])
    return list(c)


def pval(c, x):
    return sum(ci * x ** i for i, ci in enumerate(c))


def fit_min(eps, delta, d, lo=mp.mpf(1) / 10, hi=mp.mpf(1) / 5):
    """the minimum of the degree-d fit on [lo, hi]: (eps_min, Delta_min, max |residual|), or None when its derivative
    has no sign change from - to + there"""
    c = polyfit_mp(eps, delta, d)
    dc = [i * c[i] for i in range(1, len(c))]
    res = max(abs(pval(c, e) - y) for e, y in zip(eps, delta))
    grid = [lo + (hi - lo) * k / 200 for k in range(201)]
    dv = [pval(dc, g) for g in grid]
    for k in range(200):
        if dv[k] < 0 <= dv[k + 1]:
            r = mp.findroot(lambda e: pval(dc, e), (grid[k], grid[k + 1]), solver="anderson")
            return r, pval(c, r), res
    return None, None, res


def fit_all(eps, delta):
    out = {}
    for d in DEGREES:
        e, D, res = fit_min(eps, delta, d)
        out[d] = dict(eta_min=None if e is None else float(1 / e), Delta_min=None if D is None else mp.nstr(D, 20),
                      Delta_min_f=None if D is None else D, residual=float(res))
    return out


def summarize(fits):
    ok = [d for d in DEGREES if fits[d]["eta_min"] is not None]
    if QUOTED not in ok:
        return None
    em = [fits[d]["eta_min"] for d in ok]
    Dm = [fits[d]["Delta_min_f"] for d in ok]
    q = fits[QUOTED]
    return dict(eta_min=q["eta_min"], Delta_min=q["Delta_min"], Delta_min_f=q["Delta_min_f"],
                spread_eta=(min(em), max(em)), spread_Delta=(float(min(Dm) - q["Delta_min_f"]), float(max(Dm) - q["Delta_min_f"])),
                residual=q["residual"])


# ------------------------------------------------------------------ planted
def planted(eta_min_target=7.1):
    """Delta_p(eps) = A - B eps + C eps^2 + E eps^3/(1 + 4 eps), C fixed by the minimum at eps* = 1/eta_min_target"""
    A, B, E = mp.mpf("0.45986"), mp.mpf("7.5e-4"), mp.mpf("-1e-3")
    es = 1 / mp.mpf(eta_min_target)
    # d/deps: -B + 2 C eps + E (3 eps^2 (1 + 4 eps) - 4 eps^3)/(1 + 4 eps)^2 = 0 at es
    g = E * (3 * es ** 2 * (1 + 4 * es) - 4 * es ** 3) / (1 + 4 * es) ** 2
    C = (B - g) / (2 * es)
    f = lambda e: A - B * e + C * e ** 2 + E * e ** 3 / (1 + 4 * e)
    return f, es, f(es)


def selftest():
    eps = [eps_of(e) for e in ETAS]
    rng = np.random.default_rng(7)
    ok = True
    # pass case
    f, es, Ds = planted()
    noise = 1e-9
    delta = [f(e) + mp.mpf(noise * rng.standard_normal()) for e in eps]
    s = summarize(fit_all(eps, delta))
    tol_eta = max(s["spread_eta"][1] - s["spread_eta"][0], 1e-3)
    tol_D = max(s["spread_Delta"][1] - s["spread_Delta"][0], 1e-11)
    e_ok = abs(s["eta_min"] - float(1 / es)) <= tol_eta
    d_ok = abs(s["Delta_min_f"] - Ds) <= tol_D
    print(f"planted eta_min {float(1 / es):.6f}, Delta_min {mp.nstr(Ds, 15)}: fit {s['eta_min']:.6f} "
          f"(spread {s['spread_eta'][0]:.6f}..{s['spread_eta'][1]:.6f}), Delta_min {s['Delta_min']}, "
          f"residual {s['residual']:.1e} -> {'PASS' if e_ok and d_ok else 'FAIL'}")
    ok &= e_ok and d_ok
    # control: minimum outside the window
    f2, es2, _ = planted(12.0)
    s2 = summarize(fit_all(eps, [f2(e) for e in eps]))
    fired = s2 is None
    print(f"control (minimum at eta 12, outside [5, 10]): {'no minimum found, fires' if fired else 'a minimum was returned: did NOT fire'}")
    ok &= fired
    # control: one point shifted by 1e-7
    delta3 = list(delta)
    delta3[2] += mp.mpf("1e-7")
    s3 = summarize(fit_all(eps, delta3))
    fired3 = s3 is None or s3["residual"] > 20 * noise
    print(f"control (eta 7 shifted by 1e-7): degree-6 residual {s3['residual'] if s3 else float('nan'):.1e} "
          f"{'> 20 x noise, fires' if fired3 else 'within noise: did NOT fire'}")
    ok &= fired3
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    recs = [json.loads(l) for l in open(a.records)]
    out = {}
    for M in (32, 48):
        best = {}
        for r in recs:
            if r["M"] == M and r.get("converged") and r["eta"] in ETAS:
                best[r["eta"]] = r
        if set(best) != set(ETAS):
            print(f"M = {M}: missing {sorted(set(ETAS) - set(best))}")
            continue
        eps = [eps_of(e) for e in ETAS]
        delta = [mp.mpf(best[e]["Delta"]) for e in ETAS]
        fits = fit_all(eps, delta)
        s = summarize(fits)
        out[M] = dict(fits={d: dict((k, v) for k, v in f.items() if k != "Delta_min_f") for d, f in fits.items()},
                      summary=None if s is None else dict((k, v) for k, v in s.items() if k != "Delta_min_f"))
        if s:
            out[M]["_Dm"] = s["Delta_min_f"]
        for d in DEGREES:
            f = fits[d]
            print(f"M = {M}, degree {d}: eta_min {f['eta_min']}, Delta_min {f['Delta_min']}, residual {f['residual']:.1e}")
    if 32 in out and 48 in out and out[32]["summary"] and out[48]["summary"]:
        s32, s48 = out[32]["summary"], out[48]["summary"]
        res = dict(eta_min=s48["eta_min"], Delta_min=s48["Delta_min"],
                   model_spread_eta=s48["spread_eta"], model_spread_Delta=s48["spread_Delta"],
                   resolution_eta=s48["eta_min"] - s32["eta_min"],
                   resolution_Delta=float(out[48]["_Dm"] - out[32]["_Dm"]))
        out["result"] = res
        print(json.dumps(res, indent=1))
    for M in (32, 48):
        if M in out:
            out[M].pop("_Dm", None)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
