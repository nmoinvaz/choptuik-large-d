#!/usr/bin/env python3
"""Agreement of the two pilots at wf 1.00, (a) J 1000 / 960 bits and (b) J 1250 / 1280 bits, order by order
(notes/PLANTED_SELECTION.md, section 11.0a). For each K it prints, at the top four orders: |sigma_k(a) - sigma_k(b)|;
(a)'s sigma_err, the plateau error |S(Mf) - S(Mf - 1)| of its S_1 family at order 2k + 1 over |S_1^(1)|
(amp_flint_par.py); their ratio, the pilot gate (pass if at most 1); the difference relative to |sigma_k|; and (b)'s own
plateau error read from its fit_info with the order-1 term set to zero (its sigma_err is inf because family 1_1's
residual underflows to 0.0 at 1280 bits and pick_plateau reads a non-falling residual as no plateau). Then the Mf
histogram of the fits at the top order for both, and the worst family fit error there.

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


def report(K):
    a = load(f"scan{K}J1000/sigF_w1.0000_K{K}_J1000_p960.json")
    b = load(f"scan{K}J1250/sigF_w1.0000_K{K}_J1250_p1280.json")
    top = (K - 1) // 2
    # |S_1^(1)| from (a): its sigma_err at the top order is S_1's fit error at order K over |S_1^(1)| (e1 ~ 1e-274)
    a1 = a["fit_info"][f"{K}_1"][1] / a["sigma_err"][top]
    print(f"K {K}: (a) J {a['J']} / {a['prec']} bits, (b) J {b['J']} / {b['prec']} bits; |S_1^(1)| = {a1:.3f}; "
          f"(b)'s fit_info 1_1 = {b['fit_info']['1_1']}")
    print("  k   |a - b|    err_a      ratio   |a - b|/|sigma_k|  err_a/|sigma_k|  err_b (e1 := 0)")
    for k in range(top - 3, top + 1):
        sa, sb = sig(a, k), sig(b, k)
        d, ea = float(abs(sa - sb)), a["sigma_err"][k]
        eb = b["fit_info"][f"{2 * k + 1}_1"][1] / a1
        mag = float(abs(sa))
        print(f"  {k:2d}  {d:.2e}   {ea:.2e}   {d / ea:.3f}   {d / mag:.2e}           {ea / mag:.2e}         {eb:.2e}")
    for name, r in (("a", a), ("b", b)):
        fams = {key: v for key, v in r["fit_info"].items() if key.startswith(f"{K}_")}
        h = collections.Counter(v[0] for v in fams.values())
        worst = max(fams.items(), key=lambda kv: kv[1][1] if kv[1][1] != float("inf") else -1)
        print(f"  ({name}) Mf at order {K}: " + ", ".join(f"{m} x{c}" for m, c in sorted(h.items()))
              + f"; worst family {worst[0]} err {worst[1][1]:.1e}; S_1 family {fams[f'{K}_1']}")


if __name__ == "__main__":
    for K in [int(x) for x in sys.argv[1:]] or (35, 37, 39):
        report(K)
