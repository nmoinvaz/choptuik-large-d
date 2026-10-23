#!/bin/bash
# Stage 1 on the 15-node set 0.0025..0.9 (notes/HIGH_M.md 1.4): at D = 3.7 the M = 80 state's nodes 0.005..0.8 leave the
# series tails of an M = 192 start at 1e-20 (centre) and 1e-22 (horizon), and the horizon series diverges at M = 256.
# M = 128 and 160 re-noded and re-solved, then the lifts 160 -> 192 -> 256. Idempotent (resumes from checkpoints).
# Run from hp/ with PYTHONPATH pointing at a build of this branch.
set -u
cd "$(dirname "$0")/../.."
L=data/lowD_highM/logs
N15=0.0025,0.005,0.01,0.02,0.05,0.1,0.16,0.23,0.3,0.4,0.5,0.6,0.7,0.8,0.9
for M in 128 160; do
  nice -n 10 python3 -u highm_check.py lift --M=$M --start=data/lowD_states/highm_D3.7000_M$M.npz --nodes=$N15 --tag=highm15 >> $L/lift15_M$M.log 2>&1 || { echo "M $M failed"; exit 1; }
done
prev=data/lowD_states/highm15_D3.7000_M160.npz
for M in 192 256; do
  nice -n 10 python3 -u highm_check.py lift --M=$M --start=$prev --tag=highm15 >> $L/lift15_M$M.log 2>&1 || { echo "M $M failed"; exit 1; }
  prev=data/lowD_states/highm15_D3.7000_M$M.npz
done
echo STAGE1_N15_DONE
