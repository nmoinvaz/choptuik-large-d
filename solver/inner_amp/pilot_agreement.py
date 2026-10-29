#!/usr/bin/env python3
"""Agreement of the pilots at wf 1.00, (a) J 1000 / 960 bits and (b) J 1250 / 1280 bits, order by order, and at K 39
the attribution pilot (d) J 1250 / 960 bits (notes/PLANTED_SELECTION.md, section 11.0 and its amendment of
2026-10-29). For each K it prints, at the top orders: |sigma_k(a) - sigma_k(b)|; (a)'s sigma_err, the plateau step
|S(Mf) - S(Mf - 1)| of its S_1 family at order 2k + 1 over |S_1^(1)| (amp_flint_par.py); their ratio, the pilot gate
(pass if at most 1); the difference relative to |sigma_k|; and (b)'s own plateau error read from its fit_info with the
order-1 term set to zero (its sigma_err is inf because family 1_1's residual underflows to 0.0 at 1280 bits and
pick_plateau reads a non-falling residual as no plateau; sigma_err_fit.py). Then the Mf histogram of the fits at the
top order for both, and the worst family fit error there. With (d): |a - d| (the J effect, same bits), |d - b| (the
precision effect, same J), (d)'s sigma_err, and per order the number of neighbour families whose Mf pick differs
between (d) and (b) and the largest relative difference of the tuning constants f_{k,nu} between (d) and (b) and
between (a) and (d) (the constants are added exactly into the later orders, so a different pick at order 33 moves
sigma_19).

    python3 pilot_agreement.py           # K 35, 37, 39 from scan{K}J1000/*_p960.json and scan{K}J1250/*.json
    python3 pilot_agreement.py 39        # one K
"""
import collections
import glob
import json
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
mp.mp.dps = 40


def load(pattern):
    fns = sorted(glob.glob(os.path.join(HERE, pattern)))
    if len(fns) != 1:
        raise SystemExit(f"{pattern}: {len(fns)} files")
    return json.load(open(fns[0]))


def sig(r, k):
    return mp.mpc(r["sigma_str"][k][0], r["sigma_str"][k][1])


def a1_of(a, K):
    # |S_1^(1)| from (a): its sigma_err at the top order is S_1's fit error at order K over |S_1^(1)| (e1 ~ 1e-274)
    return a["fit_info"][f"{K}_1"][1] / a["sigma_err"][(K - 1) // 2]


def err_b(b, a1, k):
    """(b)'s plateau error at sigma_k with the order-1 term set to zero."""
    return b["fit_info"][f"{2 * k + 1}_1"][1] / a1


def f_reldiff(r1, r2, K):
    """largest |f1 - f2| / |f2| over the tuning constants of order K, with the family."""
    fams = [key for key in r2["f"] if key.startswith(f"{K}_")]
    return max((abs(complex(*r1["f"][f]) - complex(*r2["f"][f])) / abs(complex(*r2["f"][f])), f) for f in fams)


def attribution(K=39):
    """the numbers section 11.0's amendment quotes: J effect, precision effect, (d)'s err19 and the mechanism."""
    a = load(f"scan{K}J1000/sigF_w1.0000_K{K}_J1000_p960.json")
    b = load(f"scan{K}J1250/sigF_w1.0000_K{K}_J1250_p1280.json")
    d = load(f"scan{K}J1250p960/sigF_w1.0000_K{K}_J1250_p960.json")
    top = (K - 1) // 2
    a1 = a1_of(a, K)
    out = {"a1": a1, "seconds_d": d["seconds"]}
    for k in range(top - 3, top + 1):
        sa, sb, sd = sig(a, k), sig(b, k), sig(d, k)
        out[f"ad{k}"], out[f"db{k}"], out[f"ab{k}"] = (float(abs(x - y)) for x, y in ((sa, sd), (sd, sb), (sa, sb)))
        out[f"err_a{k}"], out[f"err_d{k}"], out[f"err_b{k}"] = a["sigma_err"][k], d["sigma_err"][k], err_b(b, a1, k)
        out[f"mag_a{k}"] = float(abs(sa))
    out["exact_12_15"] = all(sig(d, k) == sig(b, k) for k in range(12, 16))
    for Kf in range(27, K + 1, 2):
        fams = [key for key in d["fit_info"] if key.startswith(f"{Kf}_") and key != f"{Kf}_1"]
        out[f"ndiff{Kf}"] = sum(d["fit_info"][f][0] != b["fit_info"][f][0] for f in fams)
        out[f"nfam{Kf}"] = len(fams)
        out[f"fdb{Kf}"], out[f"fdb_fam{Kf}"] = f_reldiff(d, b, Kf)
        out[f"fad{Kf}"], out[f"fad_fam{Kf}"] = f_reldiff(a, d, Kf)
    # the amended err19 at wf 1.00: (a)'s plateau error plus the pilot difference |a - b|
    out["err_amended"] = a["sigma_err"][top] + out[f"ab{top}"]
    out["err_amended_rel"] = out["err_amended"] / out[f"mag_a{top}"]
    out["margin_amended"] = 1e-10 / out["err_amended_rel"]
    return out


def report(K):
    a = load(f"scan{K}J1000/sigF_w1.0000_K{K}_J1000_p960.json")
    b = load(f"scan{K}J1250/sigF_w1.0000_K{K}_J1250_p1280.json")
    top = (K - 1) // 2
    a1 = a1_of(a, K)
    print(f"K {K}: (a) J {a['J']} / {a['prec']} bits, (b) J {b['J']} / {b['prec']} bits; |S_1^(1)| = {a1:.3f}; "
          f"(b)'s fit_info 1_1 = {b['fit_info']['1_1']}")
    print("  k   |a - b|    err_a      ratio   |a - b|/|sigma_k|  err_a/|sigma_k|  err_b (e1 := 0)")
    for k in range(top - 3, top + 1):
        sa, sb = sig(a, k), sig(b, k)
        d, ea = float(abs(sa - sb)), a["sigma_err"][k]
        mag = float(abs(sa))
        print(f"  {k:2d}  {d:.2e}   {ea:.2e}   {d / ea:.3f}   {d / mag:.2e}           {ea / mag:.2e}         "
              f"{err_b(b, a1, k):.2e}")
    for name, r in (("a", a), ("b", b)):
        fams = {key: v for key, v in r["fit_info"].items() if key.startswith(f"{K}_")}
        h = collections.Counter(v[0] for v in fams.values())
        worst = max(fams.items(), key=lambda kv: kv[1][1] if kv[1][1] != float("inf") else -1)
        print(f"  ({name}) Mf at order {K}: " + ", ".join(f"{m} x{c}" for m, c in sorted(h.items()))
              + f"; worst family {worst[0]} err {worst[1][1]:.1e}; S_1 family {fams[f'{K}_1']}")
    if not glob.glob(os.path.join(HERE, f"scan{K}J1250p960/sigF_w1.0000_K{K}_J1250_p960.json")):
        return
    t = attribution(K)
    print(f"  (d) J 1250 / 960 bits, {t['seconds_d'] / 60:.0f} min; (d) == (b) at sigma_12..15: {t['exact_12_15']}")
    print("  k   |a - d| (J)   |d - b| (bits)   |a - b|    err_a      err_d      err_b (e1 := 0)")
    for k in range(top - 3, top + 1):
        print(f"  {k:2d}  {t[f'ad{k}']:.2e}      {t[f'db{k}']:.2e}         {t[f'ab{k}']:.2e}   {t[f'err_a{k}']:.2e}   "
              f"{t[f'err_d{k}']:.2e}   {t[f'err_b{k}']:.2e}")
    print("  order: neighbour families with Mf(d) != Mf(b); max |f_d - f_b|/|f_b|; max |f_a - f_d|/|f_d|")
    for Kf in range(27, K + 1, 2):
        print(f"    {Kf}: {t[f'ndiff{Kf}']}/{t[f'nfam{Kf}']}; {t[f'fdb{Kf}']:.1e} ({t[f'fdb_fam{Kf}']}); "
              f"{t[f'fad{Kf}']:.1e} ({t[f'fad_fam{Kf}']})")
    print(f"  amended err{top} at wf 1.00 = err_a + |a - b| = {t['err_amended']:.3e} = {t['err_amended_rel']:.3e} "
          f"|sigma_{top}|, margin {t['margin_amended']:.2f} under 1e-10")


if __name__ == "__main__":
    for K in [int(x) for x in sys.argv[1:]] or (35, 37, 39):
        report(K)
