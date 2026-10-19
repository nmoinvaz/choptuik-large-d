#!/bin/sh
cd /home/claude/dss
OPENBLAS_NUM_THREADS=1 NE_THREADS=1 NE_SEED_MAX=24.0 NE_SMOOTH=1 nohup python3 scan_ne.py scan_ne6_kc31_sj.jsonl 23 40 0.2 0.4 0.6 longB_D 31 > scan_ne6_kc31_sj.log 2>&1 &
OPENBLAS_NUM_THREADS=1 NE_THREADS=1 NE_SEED_MAX=24.0 NE_SMOOTH=1 nohup python3 scan_ne.py scan_ne7_kc31_sj.jsonl 23 40 0.2 0.4 0.7 long7_D 31 > scan_ne7_kc31_sj.log 2>&1 &
