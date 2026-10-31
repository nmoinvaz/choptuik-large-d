#!/usr/bin/env python3
"""The dry run of the K = 20 stages (notes/PLANTED_SELECTION.md, section 12.0 item 0): a synthetic sigma_20 with a known
answer, written in the scan files' format, so that every stage of planted_k20.py runs on it before the real sigma_20.

The synthetic series is the C19-BP [6/13] sum's own (section 11.6: the first ring-free sum passing the type rule, zero at
1.00000009): its sigma_0..19 are the real ones (the 1280-bit node data, scan41J1250/), its sigma_20 at each node is the
sum's, written with 20 digits (the scans' format), err20 = 1e-12 |sigma_20| and the other errors the plateau errors
sigma_err_fit.py recomputes for the real files (finite, so planted_k20.py reads them as stored). The pilot file (J 1500)
gets the same sigma_20, so item 1's pilot term is zero and the mutation `amend` tests it; its errors stay inf as in the
real pilot. The stages then run with --data DIR; the pass criteria are section 12.0 item 0.

    python3 planted_k20_dry.py DIR     writes DIR/scan41J1250/ and DIR/scan41J1500/ (DIR is outside the repository)
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
import planted_k20 as K20  # noqa: E402

M, N = 6, 13                     # the completion whose sigma_20 is planted (section 11.6's first ring-free, type-passing sum)


def main():
    out = sys.argv[1]
    if os.path.abspath(out).startswith(os.path.abspath(os.path.join(HERE, "..", ".."))):
        raise SystemExit("the dry run's files go outside the repository")
    real19 = K20.load_real19()
    nodes, funcs, info = K14.borel_pade(real19, M, N)
    truth, _ = K14.truth_of(funcs)
    print(f"C19-BP [{M}/{N}]: zero at {truth['im_x']:.9f}; reproduces sigma_0..19 to "
          f"{max(info[w]['reproduce_max_rel'] for w in info):.1e}")
    for src, dst, pilot in ((K20.SCAN20, "scan41J1250", False), (K20.PILOT20, "scan41J1500", True)):
        os.makedirs(os.path.join(out, dst), exist_ok=True)
        for fn in sorted(glob.glob(os.path.join(src, "*.json"))):
            r = copy.deepcopy(json.load(open(fn)))
            wf = float(r["wf"])
            s20 = nodes[wf][20]
            errs = None if pilot else K20.errs_of(r)
            r["sigma_str"] = r["sigma_str"][:20] + [[mp.nstr(mp.re(s20), 20), mp.nstr(mp.im(s20), 20)]]
            r["sigma"] = r["sigma"][:20] + [[float(mp.re(s20)), float(mp.im(s20))]]
            r["sigma_err"] = [float("inf")] * 21 if pilot else errs[:20] + [1e-12 * float(abs(s20))]
            if "sigma_rad" in r:
                r["sigma_rad"] = r["sigma_rad"][:20] + [r["sigma_rad"][19]]
            r["dry_run"] = f"synthetic sigma_20 from C19-BP [{M}/{N}] (planted_k20_dry.py); zero at {truth['im_x']:.9f}"
            name = os.path.basename(fn)
            with open(os.path.join(out, dst, name), "w") as f:
                json.dump(r, f)
            print(f"  {dst}/{name}: wf {wf}, sigma_20 = {mp.nstr(s20, 12)}, err20 {r['sigma_err'][20]:.2e}")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"# dry run of planted_k20.py: synthetic sigma_20 from C19-BP [{M}/{N}], zero at {truth['im_x']:.9f}\n")


if __name__ == "__main__":
    main()
