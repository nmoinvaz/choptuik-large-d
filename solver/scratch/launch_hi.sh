#!/bin/sh
cd /home/claude/dss
OPENBLAS_NUM_THREADS=1 NE_SEED_MAX=23.4 nohup python3 scan_ne.py scan_ne6b.jsonl 22.0 40 0.2 0.4 0.6 neB_D > scan_ne6b.log 2>&1 &
