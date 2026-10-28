#!/usr/bin/env python3
"""The dry run of the K = 18 stages (notes/PLANTED_SELECTION.md, section 10.0 item 0): a synthetic sigma_18 with a known
answer, written in the scan files' format, so that every stage of planted_k18.py runs on it before the real sigma_18.

The synthetic series is the C17-BP [7/10] sum's own (section 9.6: ring-free, type-passing, zero at 1.0000002): its
sigma_0..17 are the real ones, its sigma_18 at each node is the sum's, written with 20 digits (the scans' format) and
err18 = 1e-12 |sigma_18|. Pilot (b) gets the same sigma_18 with the real pilot's inf errors. The stages then run with
--data DIR; the pass criteria are section 10.0 item 0.

    python3 planted_k18_dry.py DIR     writes DIR/scan37J1000/ and DIR/scan37J1250/ (DIR is outside the repository)
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
import planted_k17 as K17  # noqa: E402

M, N = 7, 10                     # the completion whose sigma_18 is planted
K18 = 37                         # the K of the T87 scans


def main():
    out = sys.argv[1]
    if os.path.abspath(out).startswith(os.path.abspath(os.path.join(HERE, "..", ".."))):
        raise SystemExit("the dry run's files go outside the repository")
    real17 = K17.load_real17()
    nodes, funcs, info = K14.borel_pade(real17, M, N)
    truth, _ = K14.truth_of(funcs)
    print(f"C17-BP [{M}/{N}]: zero at {truth['im_x']:.9f}; reproduces sigma_0..17 to "
          f"{max(info[w]['reproduce_max_rel'] for w in info):.1e}")
    for src, dst, pilot in ((K17.SCAN17, "scan37J1000", False), (K17.PILOT17, "scan37J1250", True)):
        os.makedirs(os.path.join(out, dst), exist_ok=True)
        for fn in sorted(glob.glob(os.path.join(src, "*.json"))):
            r = copy.deepcopy(json.load(open(fn)))
            wf = float(r["wf"])
            s18 = nodes[wf][18]
            r["K"] = K18
            r["sigma_str"] = r["sigma_str"][:18] + [[mp.nstr(mp.re(s18), 20), mp.nstr(mp.im(s18), 20)]]
            r["sigma"] = r["sigma"][:18] + [[float(mp.re(s18)), float(mp.im(s18))]]
            r["sigma_err"] = r["sigma_err"][:18] + [float("inf") if pilot else 1e-12 * float(abs(s18))]
            if "sigma_rad" in r:
                r["sigma_rad"] = r["sigma_rad"][:18] + [r["sigma_rad"][17]]
            r["dry_run"] = f"synthetic sigma_18 from C17-BP [{M}/{N}] (planted_k18_dry.py); zero at {truth['im_x']:.9f}"
            name = os.path.basename(fn).replace("_K35_", f"_K{K18}_")
            with open(os.path.join(out, dst, name), "w") as f:
                json.dump(r, f)
            print(f"  {dst}/{name}: wf {wf}, sigma_18 = {mp.nstr(s18, 12)}, err18 {r['sigma_err'][18]:.2e}")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"# dry run of planted_k18.py: synthetic sigma_18 from C17-BP [{M}/{N}], zero at {truth['im_x']:.9f}\n")


if __name__ == "__main__":
    main()
