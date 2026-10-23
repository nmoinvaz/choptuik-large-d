#!/bin/bash
# The D = 3.7 lifts 256 -> 320 -> 384 (notes/HIGH_M.md; resumable as stage1.sh). Run from hp/ with PYTHONPATH as there.
set -u
cd "$(dirname "$0")/../.."
L=data/lowD_highM/logs
prev=data/lowD_states/highm_D3.7000_M256.npz
for M in 320 384; do
  nice -n 10 python3 -u highm_check.py lift --M=$M --start=$prev >> $L/lift_M$M.log 2>&1 || { echo "lift M $M failed"; exit 1; }
  prev=data/lowD_states/highm_D3.7000_M$M.npz
done
echo STAGE1_BIG_DONE
