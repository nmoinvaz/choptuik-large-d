#!/bin/bash
# Review RV2: resolution / robustness re-solves of saved states (one process at a time, nice 10).
#   hp/review/rv2_batch.sh STATE.npz OUT.jsonl SET
# SET: 'full' (every variant), 'core' (M 48, kfreeze 15, Taylor, series, one junk kick)
set -u
cd "$(dirname "$0")/../.."
S=$1
O=$2
SET=${3:-full}
run() { nice -n 10 python3 hp/review/rv2_resolve.py "$S" --out "$O" --quiet "$@" 2>&1 | tail -1; }
run --label base --its 4
run --label kfreeze15 --kfreeze 15 --its 8
run --label taylor --kt 40 --tol_step 1e-34 --hmax 0.025 --its 6
run --label series --kc 520 --ks 150 --its 6
run --label junk13 --perturb 13:1e-8 --its 10
if [ "$SET" = full ]; then
  run --label junk11 --perturb 11:1e-8 --its 10
  run --label junk9 --perturb 9:1e-8 --its 10
  run --label nodes_last075 --nodes 0.3,0.4,0.5,0.6,0.75 --its 8
  run --label nodes_first025 --nodes 0.25,0.4,0.5,0.6,0.7 --its 8
fi
run --label M48 --M 48 --its 8
if [ "$SET" = full ]; then
  run --label kfreeze17 --kfreeze 17 --its 10
fi
