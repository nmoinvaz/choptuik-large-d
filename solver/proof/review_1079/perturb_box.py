#!/usr/bin/env python3
"""certify_M.py with one last-bit perturbation planted (review of #1079, check 5: the macOS/Linux margin gap).

    REVIEW_PERTURB=none|qr|perron [REVIEW_SEED=k] [REVIEW_ULPS=n] python3 review_1079/perturb_box.py <certify_M.py arguments>

none    certify_M.main() unchanged: the control, which must reproduce the stored run of the same platform
qr      every entry of the float matrix Q returned by validated_int.qr_pivot multiplied by 1 +- 2^-52 (random signs,
        seeded). Q is any invertible float matrix used exactly, so the run stays a valid enclosure
perron  every Perron weight of fixed_point_M.perron multiplied by 1 +- 2^-52. Any positive weights are allowed, so the
        start data stay valid; only their radii move in the last digits
REVIEW_ULPS (default 1) scales the perturbation: the factor is 1 +- n 2^-52.
The run is rigorous in every mode. What is measured is how far the Krawczyk margins move.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import certify_M as CM  # noqa: E402
import validated_int as VI  # noqa: E402
import fixed_point_M as FPM  # noqa: E402

WHAT = os.environ.get("REVIEW_PERTURB", "none")
RNG = np.random.default_rng(int(os.environ.get("REVIEW_SEED", "1079")))
EPS = int(os.environ.get("REVIEW_ULPS", "1")) * 2.0 ** -52
COUNT = {"qr": 0, "perron": 0}
assert WHAT in ("none", "qr", "perron"), WHAT

if WHAT == "qr":
    _qr = VI.qr_pivot

    def qr_pivot(Am, wts=None):
        Q = _qr(Am, wts)
        COUNT["qr"] += 1
        return Q * (1.0 + (RNG.integers(0, 2, Q.shape) * 2 - 1) * EPS)
    VI.qr_pivot = qr_pivot
elif WHAT == "perron":
    _perron = FPM.perron

    def perron(L, sym=None):
        w = _perron(L, sym)
        COUNT["perron"] += 1
        return [float(v) * (1.0 + (int(RNG.integers(0, 2)) * 2 - 1) * EPS) for v in w]
    FPM.perron = perron

if __name__ == "__main__":
    import flint
    print(f"review_1079/perturb_box.py: REVIEW_PERTURB={WHAT}, seed {os.environ.get('REVIEW_SEED', '1079')}, relative size {EPS:.3e}; "
          f"python-flint {flint.__version__}, FLINT {flint.__FLINT_VERSION__}, numpy {np.__version__}; "
          f"NTAY, NJAC = {CM.NTAY}, {CM.NJAC}; path {[str(x) for x in CM.path_for(3)]}", flush=True)
    rc = CM.main()
    print(f"perturbed calls: qr_pivot {COUNT['qr']}, perron {COUNT['perron']}", flush=True)
    sys.exit(rc)
