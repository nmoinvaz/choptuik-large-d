#!/usr/bin/env bash
# Reproduce the headline numbers of this repository and report PASS / FAIL / SKIP for each check.
#
#   ./reproduce.sh          about 3 minutes (measured on a busy 2-core machine; a laptop should be faster)
#   ./reproduce.sh --full   adds the Proposition 4 late-order check (about 2 minutes more)
#
# What it checks (details and expected values in reproduce.py):
#   1. the double-double data hp/data/delta_dd.csv, the Values table and the rational formula against it;
#   2. Delta_inf from five analytic fits in 1/D on all cleaned points (mpmath, 60 digits);
#   3. the held-out test: fit on D <= 45, predict 45 < D <= 61 (analytic vs half-integer series);
#   4. theory checks: C(n,s) as a Stokes constant, the lateral-solution method, the Borel-Darboux constant
#      (needs python-flint), the transport of Proposition 3, the formal solution of Proposition 1;
#   5. the echo profile on saved solutions hp/data/states (needs the compiled kernel hpkernel: install Rust and
#      maturin, then run hp/build.sh).
# Nothing in the repository is written.
set -euo pipefail
cd "$(dirname "$0")"

python3 - <<'EOF'
import importlib.util, sys
need = ["numpy", "scipy", "mpmath", "sympy"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("missing Python packages: " + " ".join(missing) + "  (pip install " + " ".join(missing) + ")")
for m, why in (("flint", "python-flint: the Borel-Darboux check"), ("hpkernel", "hpkernel: checks on saved solutions (hp/build.sh)")):
    if importlib.util.find_spec(m) is None:
        print(f"optional, not found: {why} (reported as SKIP)")
EOF

exec python3 reproduce.py "$@"
