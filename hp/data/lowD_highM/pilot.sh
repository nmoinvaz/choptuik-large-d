#!/bin/bash
# Stage 2 of notes/HIGH_M.md: the pilot toward D = 3 from D = 3.7 at M = 128 / 160 on the 15 nodes 0.0025..0.9.
# Resumable: rerun the same command after an interruption. Run from anywhere with PYTHONPATH at a build of this branch.
cd "$(dirname "$0")/../.."
nice -n 10 python3 -u lowD_highM.py --start_w data/lowD_states/highm15_D3.7000_M128.npz \
    --start_n data/lowD_states/highm15_D3.7000_M160.npz --Ms 128,160,192,256,320,384 --auto_nodes --tag lowDhm \
    --resume >> data/lowD_highM/logs/pilot.log 2>&1
