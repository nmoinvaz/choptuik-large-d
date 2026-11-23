#!/usr/bin/env python3
"""certify_M.py with one last-bit perturbation planted (review of #1079, check 5: the macOS/Linux margin gap).

    REVIEW_PERTURB=none|qr|qrin|perron [REVIEW_SEED=k] [REVIEW_ULPS=n] python3 review_1079/perturb_box.py <certify_M.py arguments>

none    certify_M.main() unchanged: the control, which must reproduce the stored run of the same platform
qr      every entry of the float matrix Q returned by validated_int.qr_pivot multiplied by 1 +- 2^-52 (random signs,
        seeded). Q is any invertible float matrix used exactly, so the run stays a valid enclosure
qrin    every entry of the float matrix handed to numpy's QR multiplied by 1 +- 2^-52, so that Q is LAPACK's own
        output for a matrix one rounding away (orthogonal to rounding, as on any machine)
perron  every Perron weight of fixed_point_M.perron multiplied by 1 +- 2^-52. Any positive weights are allowed, so the
        start data stay valid; only their radii move in the last digits
REVIEW_ULPS (default 1) scales the perturbation: the factor is 1 +- n 2^-52.
REVIEW_TIES=1 (with any mode) counts, over the calls of qr_pivot, those whose pivot order rests on a near tie: two
        consecutive sorted column weights (norm times weight) within a relative 1e-12, among columns whose weight
        is above 1e-200. It prints the counts and the smallest gap seen.
REVIEW_STABLE=1 replaces the pivot order of qr_pivot by one that does not depend on last bits (a trial remedy).
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
assert WHAT in ("none", "qr", "qrin", "perron"), WHAT

if os.environ.get("REVIEW_STABLE"):
    # a trial remedy, not the code of record: the pivot order by the column weights rounded to 6 significant digits,
    # ties broken by the column index. Any order gives a valid Q; this one does not depend on the last bits of the
    # weights or on the sort's treatment of equal keys. Installed first, so that the perturbations act on top of it.
    def qr_stable(Am, wts=None):
        nrm = np.linalg.norm(Am, axis=0)
        if wts is not None:
            nrm = nrm * (np.asarray(wts) + 1e-300)
        key = np.array([float(f"{v:.5e}") for v in nrm])
        order = np.lexsort((np.arange(len(key)), -key))
        Q, _R = np.linalg.qr(Am[:, order])
        return Q
    VI.qr_pivot = qr_stable

if WHAT == "qr":
    _qr = VI.qr_pivot

    def qr_pivot(Am, wts=None):
        Q = _qr(Am, wts)
        COUNT["qr"] += 1
        return Q * (1.0 + (RNG.integers(0, 2, Q.shape) * 2 - 1) * EPS)
    VI.qr_pivot = qr_pivot
elif WHAT == "qrin":
    _qr = VI.qr_pivot

    def qr_pivot(Am, wts=None):
        COUNT["qr"] += 1
        return _qr(Am * (1.0 + (RNG.integers(0, 2, Am.shape) * 2 - 1) * EPS), wts)
    VI.qr_pivot = qr_pivot
elif WHAT == "perron":
    _perron = FPM.perron

    def perron(L, sym=None):
        w = _perron(L, sym)
        COUNT["perron"] += 1
        return [float(v) * (1.0 + (int(RNG.integers(0, 2)) * 2 - 1) * EPS) for v in w]
    FPM.perron = perron

TIES = {"calls": 0, "tied": 0, "min_gap": 1.0, "exact": 0}
if os.environ.get("REVIEW_TIES"):
    _qr0 = VI.qr_pivot

    def qr_ties(Am, wts=None):
        nrm = np.linalg.norm(Am, axis=0)
        if wts is not None:
            nrm = nrm * (np.asarray(wts) + 1e-300)
        v = np.sort(nrm)[::-1]
        v = v[v > 1e-200]
        TIES["calls"] += 1
        if len(v) > 1:
            gap = float(np.min((v[:-1] - v[1:]) / v[:-1]))
            TIES["min_gap"] = min(TIES["min_gap"], gap)
            TIES["tied"] += gap < 1e-12
            TIES["exact"] += gap == 0.0
        return _qr0(Am, wts)
    VI.qr_pivot = qr_ties

if __name__ == "__main__":
    import flint
    print(f"review_1079/perturb_box.py: REVIEW_PERTURB={WHAT}, seed {os.environ.get('REVIEW_SEED', '1079')}, relative size {EPS:.3e}; "
          f"python-flint {flint.__version__}, FLINT {flint.__FLINT_VERSION__}, numpy {np.__version__}; "
          f"NTAY, NJAC = {CM.NTAY}, {CM.NJAC}; stable pivot order {bool(os.environ.get('REVIEW_STABLE'))}; path {[str(x) for x in CM.path_for(3)]}", flush=True)
    rc = CM.main()
    if os.environ.get("REVIEW_TIES"):
        print(f"pivot order: {TIES['calls']} calls of qr_pivot, {TIES['tied']} with two column weights within a relative "
              f"1e-12 ({TIES['exact']} exactly equal); smallest relative gap {TIES['min_gap']:.3e}", flush=True)
    print(f"perturbed calls: qr_pivot {COUNT['qr']}, perron {COUNT['perron']}", flush=True)
    sys.exit(rc)
