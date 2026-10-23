#!/bin/bash
# Stage 2 of notes/HIGH_M.md: the pilot toward D = 3 from D = 3.7 at M = 80 / 96 on the 15 nodes 0.0025..0.9 (the
# starting pair is the smallest that the M differences allow at D = 3.7; a first attempt at M = 128 / 160 went astray
# in the weak directions from the first, constant predictor: logs/pilot_attempt1_M128.log). The two start states are
# made first if missing (hp/highm_check.py lift, re-noded from the M = 80 state). Resumable: rerun the same command.
# Run from anywhere with PYTHONPATH at a build of this branch.
cd "$(dirname "$0")/../.."
N15=0.0025,0.005,0.01,0.02,0.05,0.1,0.16,0.23,0.3,0.4,0.5,0.6,0.7,0.8,0.9
L=data/lowD_highM/logs
nice -n 10 python3 -u highm_check.py lift --M=80 --start=data/lowD_states/lowDv_D3.7000_M80.npz --nodes=$N15 --tag=highm15 >> $L/lift15_M80.log 2>&1 || exit 1
nice -n 10 python3 -u highm_check.py lift --M=96 --start=data/lowD_states/highm15_D3.7000_M80.npz --tag=highm15 >> $L/lift15_M96.log 2>&1 || exit 1
nice -n 10 python3 -u lowD_highM.py --start_w data/lowD_states/highm15_D3.7000_M80.npz \
    --start_n data/lowD_states/highm15_D3.7000_M96.npz --Ms 80,96,128,160,192,256,320,384 --auto_nodes --tag lowDhm \
    --resume >> $L/pilot.log 2>&1
