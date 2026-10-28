#!/usr/bin/env python3
"""The dry run of the K = 17 stages (notes/PLANTED_SELECTION.md, section 9.0 item 0): a synthetic sigma_17 with a known
answer, written in the scan files' format, so that every stage of planted_k17.py runs on it before the real sigma_17.

The synthetic series is the C16-BP [7/9] sum's own (section 8.6: ring-free, type-passing, zero at 1.0000007): its
sigma_0..16 are the real ones, its sigma_17 at each node is the sum's, written with 20 digits (the scans' format) and
err17 = 1e-12 |sigma_17|. Pilot (b) gets the same sigma_17 with the real pilot's inf errors. The stages then run with
--data DIR; the pass criteria are section 9.0 item 0.

    python3 planted_k17_dry.py DIR     writes DIR/scan35J1000/ and DIR/scan35J1250/ (DIR is outside the repository)
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
import planted_k16 as K16  # noqa: E402

M, N = 7, 9                      # the completion whose sigma_17 is planted
K17 = 35                         # the K of the T86 scans


def main():
    out = sys.argv[1]
    if os.path.abspath(out).startswith(os.path.abspath(os.path.join(HERE, "..", ".."))):
        raise SystemExit("the dry run's files go outside the repository")
    real16 = K16.load_real16()
    nodes, funcs, info = K14.borel_pade(real16, M, N)
    truth, _ = K14.truth_of(funcs)
    print(f"C16-BP [{M}/{N}]: zero at {truth['im_x']:.9f}; reproduces sigma_0..16 to "
          f"{max(info[w]['reproduce_max_rel'] for w in info):.1e}")
    for src, dst, pilot in ((K16.SCAN16, "scan35J1000", False), (K16.PILOT16, "scan35J1250", True)):
        os.makedirs(os.path.join(out, dst), exist_ok=True)
        for fn in sorted(glob.glob(os.path.join(src, "*.json"))):
            r = copy.deepcopy(json.load(open(fn)))
            wf = float(r["wf"])
            s17 = nodes[wf][17]
            r["K"] = K17
            r["sigma_str"] = r["sigma_str"][:17] + [[mp.nstr(mp.re(s17), 20), mp.nstr(mp.im(s17), 20)]]
            r["sigma"] = r["sigma"][:17] + [[float(mp.re(s17)), float(mp.im(s17))]]
            r["sigma_err"] = r["sigma_err"][:17] + [float("inf") if pilot else 1e-12 * float(abs(s17))]
            if "sigma_rad" in r:
                r["sigma_rad"] = r["sigma_rad"][:17] + [r["sigma_rad"][16]]
            r["dry_run"] = f"synthetic sigma_17 from C16-BP [{M}/{N}] (planted_k17_dry.py); zero at {truth['im_x']:.9f}"
            name = os.path.basename(fn).replace("_K33_", f"_K{K17}_")
            with open(os.path.join(out, dst, name), "w") as f:
                json.dump(r, f)
            print(f"  {dst}/{name}: wf {wf}, sigma_17 = {mp.nstr(s17, 12)}, err17 {r['sigma_err'][17]:.2e}")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"# dry run of planted_k17.py: synthetic sigma_17 from C16-BP [{M}/{N}], zero at {truth['im_x']:.9f}\n")


if __name__ == "__main__":
    main()
