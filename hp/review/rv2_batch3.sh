#!/bin/bash
# Review RV2, third batch (one process at a time, nice 10): the M = 48 residual of the interpolated M = 32 state at
# D = 60 (does the M = 48 test have anything to correct?), and the last-node move with full Newton steps.
set -u
cd "$(dirname "$0")/../.."
R=hp/review
echo "== M = 48 residual of the double-double interpolated M = 32 state, D = 60"
nice -n 10 python3 $R/rv2_m48start.py hp/data/states/up13_D60.0020.npz 2>&1 | tail -3
echo "== last node 0.75, full Newton steps without line search"
nice -n 10 python3 $R/rv2_resolve.py hp/data/states/up13_D60.0020.npz --label nodes_last075_nols --nodes 0.3,0.4,0.5,0.6,0.75 \
    --corrector nols --its 6 --quiet --out $R/rv2_resolve_D60.jsonl 2>&1 | tail -1
