#!/usr/bin/env python3
"""Importing sigma_err_fit through pade_geometry.noise_eps must leave mpmath's working precision alone (#683).

    cd solver/inner_amp && python3 dps_isolation_check.py [--mutate=global]

Sets mp.mp.dps = 150, calls noise_eps(1.0) in the measured noise mode (which imports sigma_err_fit and recomputes the
per-order errors), and exits 0 when the precision is still 150. Control: --mutate=global sets the precision to 40
inside the call, as sigma_err_fit's import did before #683, and the check must fail (exit 1).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402

import pade_geometry  # noqa: E402

mp.mp.dps = 150
pade_geometry.NOISE_MODE = "measured"
pade_geometry.noise_eps(1.0)
if "--mutate=global" in sys.argv[1:]:
    mp.mp.dps = 40
ok = mp.mp.dps == 150
print(f"dps after noise_eps: {mp.mp.dps} ({'pass' if ok else 'FAIL'}: 150 required)")
sys.exit(0 if ok else 1)
