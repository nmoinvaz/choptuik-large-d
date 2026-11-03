#!/usr/bin/env python3
"""The dry run of the K = 22 stages (notes/PLANTED_SELECTION.md, section 14.0 item 0): a synthetic sigma_22 with a known
answer, written in the scan files' format, so that every stage of planted_k22.py runs on it before the real sigma_22.

The synthetic series is the C21-BP [6/15] sum's own (section 13.7: the first ring-free sum passing the type rule, zero at
1.000000009): its sigma_0..21 are the real ones (the 1280-bit node data, scan43J1250/), its sigma_22 at each node is the
sum's, written with 20 digits (the scans' format), err22 = 1e-12 |sigma_22| and the other errors the plateau errors
sigma_err_fit.py recomputes for the real files (finite, so planted_k22.py reads them as stored). The pilot file (J 1500,
from scan43J1500/) gets the same sigma_22, so item 1's pilot term is zero and the mutation `amend` tests it; its errors
stay inf as in the real pilot. The stages then run with --data DIR; the pass criteria are section 14.0 item 0.

    python3 planted_k22_dry.py DIR     writes DIR/scan45J1250/ and DIR/scan45J1500/ (DIR is outside the repository)
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
import planted_k21 as K21  # noqa: E402

M, N = 6, 15                     # the completion whose sigma_22 is planted (section 13.7's first ring-free, type-passing sum)


def main():
    out = sys.argv[1]
    if os.path.abspath(out).startswith(os.path.abspath(os.path.join(HERE, "..", ".."))):
        raise SystemExit("the dry run's files go outside the repository")
    real21 = K21.load_real21()
    nodes, funcs, info = K14.borel_pade(real21, M, N)
    truth, _ = K14.truth_of(funcs)
    print(f"C21-BP [{M}/{N}]: zero at {truth['im_x']:.9f}; reproduces sigma_0..21 to "
          f"{max(info[w]['reproduce_max_rel'] for w in info):.1e}")
    for src, dst, pilot in ((K21.SCAN21, "scan45J1250", False), (K21.PILOT21, "scan45J1500", True)):
        os.makedirs(os.path.join(out, dst), exist_ok=True)
        for fn in sorted(glob.glob(os.path.join(src, "*.json"))):
            r = copy.deepcopy(json.load(open(fn)))
            wf = float(r["wf"])
            s22 = nodes[wf][22]
            errs = None if pilot else K21.errs_of(r)              # the real plateau errors at k = 0..21
            r["sigma_str"] = r["sigma_str"][:22] + [[mp.nstr(mp.re(s22), 20), mp.nstr(mp.im(s22), 20)]]
            r["sigma"] = r["sigma"][:22] + [[float(mp.re(s22)), float(mp.im(s22))]]
            r["sigma_err"] = [float("inf")] * 23 if pilot else errs[:22] + [1e-12 * float(abs(s22))]
            if "sigma_rad" in r:
                r["sigma_rad"] = r["sigma_rad"][:22] + [r["sigma_rad"][21]]
            r["dry_run"] = f"synthetic sigma_22 from C21-BP [{M}/{N}] (planted_k22_dry.py); zero at {truth['im_x']:.9f}"
            name = os.path.basename(fn).replace("K43", "K45")
            with open(os.path.join(out, dst, name), "w") as f:
                json.dump(r, f)
            print(f"  {dst}/{name}: wf {wf}, sigma_22 = {mp.nstr(s22, 12)}, err22 {r['sigma_err'][22]:.2e}")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"# dry run of planted_k22.py: synthetic sigma_22 from C21-BP [{M}/{N}], zero at {truth['im_x']:.9f}\n")


if __name__ == "__main__":
    main()
