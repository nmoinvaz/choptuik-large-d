#!/usr/bin/env python3
"""The dry run of the K = 19 stages (notes/PLANTED_SELECTION.md, section 11.0 item 0): a synthetic sigma_19 with a known
answer, written in the scan files' format, so that every stage of planted_k19.py runs on it before the real sigma_19.

The synthetic series is the C18-BP [7/11] sum's own (section 10.6: ring-free, type-passing, zero at 0.99999997): its
sigma_0..18 are the real ones, its sigma_19 at each node is the sum's, written with 20 digits (the scans' format) and
err19 = 1e-12 |sigma_19|. Pilot (b) gets the same sigma_19 with the real pilot's inf errors. The stages then run with
--data DIR; the pass criteria are section 11.0 item 0.

    python3 planted_k19_dry.py DIR     writes DIR/scan39J1000/ and DIR/scan39J1250/ (DIR is outside the repository)
"""
import copy
import glob
import json
import os
import sys

import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import planted_k14 as K14  # noqa: E402
import planted_k18 as K18  # noqa: E402

M, N = 7, 11                     # the completion whose sigma_19 is planted
K19 = 39                         # the K of the T88 scans


def main():
    out = sys.argv[1]
    if os.path.abspath(out).startswith(os.path.abspath(os.path.join(HERE, "..", ".."))):
        raise SystemExit("the dry run's files go outside the repository")
    real18 = K18.load_real18()
    nodes, funcs, info = K14.borel_pade(real18, M, N)
    truth, _ = K14.truth_of(funcs)
    print(f"C18-BP [{M}/{N}]: zero at {truth['im_x']:.9f}; reproduces sigma_0..18 to "
          f"{max(info[w]['reproduce_max_rel'] for w in info):.1e}")
    for src, dst, pilot in ((K18.SCAN18, "scan39J1000", False), (K18.PILOT18, "scan39J1250", True)):
        os.makedirs(os.path.join(out, dst), exist_ok=True)
        for fn in sorted(glob.glob(os.path.join(src, "*.json"))):
            r = copy.deepcopy(json.load(open(fn)))
            wf = float(r["wf"])
            s19 = nodes[wf][19]
            r["K"] = K19
            r["sigma_str"] = r["sigma_str"][:19] + [[mp.nstr(mp.re(s19), 20), mp.nstr(mp.im(s19), 20)]]
            r["sigma"] = r["sigma"][:19] + [[float(mp.re(s19)), float(mp.im(s19))]]
            r["sigma_err"] = r["sigma_err"][:19] + [float("inf") if pilot else 1e-12 * float(abs(s19))]
            if "sigma_rad" in r:
                r["sigma_rad"] = r["sigma_rad"][:19] + [r["sigma_rad"][18]]
            r["dry_run"] = f"synthetic sigma_19 from C18-BP [{M}/{N}] (planted_k19_dry.py); zero at {truth['im_x']:.9f}"
            name = os.path.basename(fn).replace("_K37_", f"_K{K19}_")
            with open(os.path.join(out, dst, name), "w") as f:
                json.dump(r, f)
            print(f"  {dst}/{name}: wf {wf}, sigma_19 = {mp.nstr(s19, 12)}, err19 {r['sigma_err'][19]:.2e}")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"# dry run of planted_k19.py: synthetic sigma_19 from C18-BP [{M}/{N}], zero at {truth['im_x']:.9f}\n")


if __name__ == "__main__":
    main()
