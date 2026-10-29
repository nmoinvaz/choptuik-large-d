#!/usr/bin/env python3
"""sigma_err of an amp_flint_par.py file recomputed from its fit_info, for the files whose stored sigma_err is inf.

amp_flint_par.py writes sigma_err[n] = S1_err[2n+1] / a1 + |sigma_n| e1 / a1 with S1_err[k] the plateau step
|S(Mf) - S(Mf - 1)| of the S_1 family at order k (fit_info["k_1"][1]), a1 = |S_1^(1)| and e1 = S1_err[1]. At 1280 bits
family 1_1's residual is 0.0 at every Mf, pick_plateau finds no falling pair and reports e1 = inf, so every sigma_err is
inf although the plateau steps at the higher orders are finite (notes/PLANTED_SELECTION.md, section 11.0, amendment of
2026-10-29). This script recomputes the sum with e1 = 0 in that case and says what it dropped: at 960 bits e1 is about
2e-274 absolute, and |sigma_n| e1 / a1 is below 1e-290 for every n here, invisible next to S1_err[2n+1] / a1 ~ 1e-27.
a1 is not stored in the files; it is read from a companion file at the same wf whose sigma_err is finite (--a1-from),
as fit_info["K_1"][1] / sigma_err[top], which is a1 up to a factor 1 + |sigma_top| e1 / S1_err[K] ~ 1 + 1e-265.

    python3 sigma_err_fit.py FILE --a1-from FILE2        # errors of FILE; exit 1 if err_top > 1e-10 |sigma_top|
    python3 sigma_err_fit.py FILE                        # FILE's own sigma_err must be finite (a1 from FILE itself)
    python3 sigma_err_fit.py --selftest                  # the recomputation reproduces every finite file's sigma_err

Exit 2 when the error is undefined: family 1_1 with err inf and a nonzero residual (a true missing plateau), another
family of the S_1 column with err inf, or the companion's wf differs. The selftest also runs a mutation (a1 scaled by
1 + 1e-6) and requires the comparison to fire.
"""
import argparse
import glob
import json
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
mp.mp.dps = 40
INF = float("inf")


def undefined(msg):
    print(f"{msg}: the error is undefined, exit 2", file=sys.stderr)
    sys.exit(2)


def a1_of(r):
    """|S_1^(1)| of a file with finite sigma_err, from its top-order plateau step and sigma_err."""
    K, top = r["K"], (r["K"] - 1) // 2
    e = r["sigma_err"][top]
    if not (e < INF):
        undefined(f"a1_of: sigma_err[{top}] is {e}, the file cannot supply a1")
    return r["fit_info"][f"{K}_1"][1] / e


def recompute(r, a1, e1=None):
    """[(k, sigma_k, err_k, dropped_k)] with err_k = S1_err[2k+1] / a1 + |sigma_k| e1 / a1.
    e1 None: the file's own, and when it is inf with residual 0.0, e1 = 0 (dropped_k then holds the bound of the
    dropped term with e1 = 2.3e-274, the largest 1_1 plateau step of the 960-bit files)."""
    K = r["K"]
    m1, err1, res1 = r["fit_info"]["1_1"]
    if e1 is None:
        if err1 < INF:
            e1, e1_bound = err1, 0.0
        elif res1 == 0.0:
            e1, e1_bound = 0.0, 2.3e-274
        else:
            undefined(f"family 1_1: err inf with residual {res1} (no plateau)")
    else:
        e1_bound = 0.0
    out = []
    for k, (re_, im) in enumerate(r["sigma_str"]):
        s = mp.mpc(re_, im)
        key = f"{2 * k + 1}_1"
        step = e1 if k == 0 else r["fit_info"][key][1]
        if not (step < INF):
            undefined(f"family {key}: err inf (residual {r['fit_info'][key][2]})")
        mag = float(abs(s))
        out.append((k, s, step / a1 + mag * e1 / a1, mag * e1_bound / a1))
    return out


def report(fn, fn_a1=None):
    r = json.load(open(fn))
    K, top = r["K"], (r["K"] - 1) // 2
    if fn_a1:
        r2 = json.load(open(fn_a1))
        if r2["wf"] != r["wf"]:
            undefined(f"--a1-from wf {r2['wf']} differs from the file's {r['wf']}")
        a1 = a1_of(r2)
        print(f"a1 = |S_1^(1)| = {a1:.15g} from {os.path.basename(fn_a1)} (K {r2['K']}, J {r2['J']}, {r2['prec']} bits)")
    else:
        a1 = a1_of(r)
        print(f"a1 = |S_1^(1)| = {a1:.15g} from the file itself")
    m1, err1, res1 = r["fit_info"]["1_1"]
    rows = recompute(r, a1)
    print(f"{os.path.basename(fn)}: wf {r['wf']}, K {K}, J {r['J']}, {r['prec']} bits; family 1_1 fit Mf {m1} err {err1} "
          f"resid {res1}" + ("; e1 := 0, the dropped term |sigma_k| e1 / a1 is bounded below" if err1 == INF else ""))
    print("  k  sigma_k (20 digits)                                     err_k      err_k/|sigma_k|  stored     dropped <=")
    for k, s, e, dropped in rows:
        print(f"  {k:2d}  {mp.nstr(s, 20):52s}  {e:.3e}  {e / float(abs(s)):.2e}         {r['sigma_err'][k]:.3g}   "
              f"{dropped:.0e}")
    k, s, e, _ = rows[top]
    rel = e / float(abs(s))
    ok = rel <= 1e-10
    print(f"gate: err{top} = {e:.3e} = {rel:.2e} |sigma_{top}| against 1e-10: {'PASS' if ok else 'FAIL'}")
    return ok


def selftest():
    fns = sorted(glob.glob(os.path.join(HERE, "scan*/sigF_*.json")))
    n = 0
    for fn in fns:
        r = json.load(open(fn))
        if "fit_info" not in r or not (r["sigma_err"][-1] < INF):
            continue
        a1 = a1_of(r)
        for k, s, e, _ in recompute(r, a1):
            if abs(e - r["sigma_err"][k]) > 1e-12 * r["sigma_err"][k]:
                raise SystemExit(f"selftest: {fn} k {k}: recomputed {e:.6e} against stored {r['sigma_err'][k]:.6e}")
        n += 1
    r = json.load(open(fns[-1]))
    fired = any(abs(e - r["sigma_err"][k]) > 1e-12 * r["sigma_err"][k] for k, s, e, _ in recompute(r, a1_of(r) * (1 + 1e-6)))
    if not fired:
        raise SystemExit("selftest: the mutation (a1 scaled by 1 + 1e-6) did not fire")
    print(f"selftest: {n} files with finite sigma_err reproduced to 1e-12 relative at every order; the a1 mutation fires")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("file", nargs="?")
    p.add_argument("--a1-from", dest="a1_from")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
    elif args.file:
        sys.exit(0 if report(args.file, args.a1_from) else 1)
    else:
        p.print_help()
