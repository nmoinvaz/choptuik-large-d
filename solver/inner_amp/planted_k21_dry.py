#!/usr/bin/env python3
"""The dry run of the K = 21 stages (notes/PLANTED_SELECTION.md, section 13.0 item 0): a synthetic sigma_21 with a known
answer, written in the scan files' format, so that every stage of planted_k21.py runs on it before the real sigma_21.

The synthetic series is the C20-BP [6/14] sum's own (section 12.7: the first ring-free sum passing the type rule, zero at
1.000000028): its sigma_0..20 are the real ones (the 1280-bit node data, scan41J1250/), its sigma_21 at each node is the
sum's, written with 20 digits (the scans' format), err21 = 1e-12 |sigma_21| and the other errors the plateau errors
sigma_err_fit.py recomputes for the real files (finite, so planted_k21.py reads them as stored). The pilot file (J 1500,
from scan41J1500/) gets the same sigma_21, so item 1's pilot term is zero and the mutation `amend` tests it; its errors
stay inf as in the real pilot. The stages then run with --data DIR; the pass criteria are section 13.0 item 0.

    python3 planted_k21_dry.py DIR     writes DIR/scan43J1250/ and DIR/scan43J1500/ (DIR is outside the repository)
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

M, N = 6, 14                     # the completion whose sigma_21 is planted (section 12.7's first ring-free, type-passing sum)


def main():
    out = sys.argv[1]
    if os.path.abspath(out).startswith(os.path.abspath(os.path.join(HERE, "..", ".."))):
        raise SystemExit("the dry run's files go outside the repository")
    real20 = K20.load_real20()
    nodes, funcs, info = K14.borel_pade(real20, M, N)
    truth, _ = K14.truth_of(funcs)
    print(f"C20-BP [{M}/{N}]: zero at {truth['im_x']:.9f}; reproduces sigma_0..20 to "
          f"{max(info[w]['reproduce_max_rel'] for w in info):.1e}")
    for src, dst, pilot in ((K20.SCAN20, "scan43J1250", False), (K20.PILOT20, "scan43J1500", True)):
        os.makedirs(os.path.join(out, dst), exist_ok=True)
        for fn in sorted(glob.glob(os.path.join(src, "*.json"))):
            r = copy.deepcopy(json.load(open(fn)))
            wf = float(r["wf"])
            s21 = nodes[wf][21]
            errs = None if pilot else K20.errs_of(r)              # the real plateau errors at k = 0..20
            r["sigma_str"] = r["sigma_str"][:21] + [[mp.nstr(mp.re(s21), 20), mp.nstr(mp.im(s21), 20)]]
            r["sigma"] = r["sigma"][:21] + [[float(mp.re(s21)), float(mp.im(s21))]]
            r["sigma_err"] = [float("inf")] * 22 if pilot else errs[:21] + [1e-12 * float(abs(s21))]
            if "sigma_rad" in r:
                r["sigma_rad"] = r["sigma_rad"][:21] + [r["sigma_rad"][20]]
            r["dry_run"] = f"synthetic sigma_21 from C20-BP [{M}/{N}] (planted_k21_dry.py); zero at {truth['im_x']:.9f}"
            name = os.path.basename(fn).replace("K41", "K43")
            with open(os.path.join(out, dst, name), "w") as f:
                json.dump(r, f)
            print(f"  {dst}/{name}: wf {wf}, sigma_21 = {mp.nstr(s21, 12)}, err21 {r['sigma_err'][21]:.2e}")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"# dry run of planted_k21.py: synthetic sigma_21 from C20-BP [{M}/{N}], zero at {truth['im_x']:.9f}\n")


if __name__ == "__main__":
    main()
