#!/bin/sh
cd /home/claude/dss
OPENBLAS_NUM_THREADS=1 NE_SEED_MAX=24.0 nohup python3 scan_ne.py scan_ne7.jsonl 23 40 0.2 0.4 0.7 long7_D > scan_ne7.log 2>&1 &
