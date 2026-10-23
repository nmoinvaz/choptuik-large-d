#!/bin/bash
# Stage 1 of notes/HIGH_M.md: the D = 3.7 lifts M = 80 -> 128 -> 160 -> 192 -> 256 (two passes each, checkpointed),
# the plan comparison at M = 128, and the checks (stage1_big.sh: the lifts to 320 and 384). Idempotent: a rerun after an interruption resumes
# from the checkpoints and skips what is done. Run from hp/ with PYTHONPATH pointing at a build of this branch and
# REF_MAIN at the directory with a build of the parent commit (for the bit-identity references).
set -u
cd "$(dirname "$0")/../.."
L=data/lowD_highM/logs
mkdir -p $L
prev=data/lowD_states/lowDv_D3.7000_M80.npz
for M in 128 160 192 256; do
  nice -n 10 python3 -u highm_check.py lift --M=$M --start=$prev >> $L/lift_M$M.log 2>&1 || { echo "lift M $M failed"; exit 1; }
  prev=data/lowD_states/highm_D3.7000_M$M.npz
done
nice -n 10 python3 -u highm_check.py lift --M=128 --no_replan --tag=highm_planstart --out=data/lowD_highM/lift_D3.7_planstart.jsonl >> $L/lift_M128_planstart.log 2>&1
[ -s $L/ref_mixed_compare.log ] && grep -q "PASS\|FAIL" $L/ref_mixed_compare.log || nice -n 10 python3 -u mixed_check.py ref --compare=$REF_MAIN/REF_mixed_main.npz > $L/ref_mixed_compare.log 2>&1
PYTHONPATH=$REF_MAIN/py_main nice -n 10 python3 -u highm_check.py ref --save=$REF_MAIN/REF_highm_main.json >> $L/ref_highm_save_main.log 2>&1
grep -q "PASS\|FAIL" $L/ref_highm_compare.log 2>/dev/null || nice -n 10 python3 -u highm_check.py ref --compare=$REF_MAIN/REF_highm_main.json > $L/ref_highm_compare.log 2>&1
grep -q "^PASS\|^FAIL" $L/block_check.log 2>/dev/null || nice -n 10 python3 -u highm_check.py block --Ms=16,48,80,128 --variants=f64x4,f64x1,dd4 --json=data/lowD_highM/block_check.json > $L/block_check.log 2>&1
grep -q "controls:" $L/block_controls.log 2>/dev/null || nice -n 10 python3 -u highm_check.py controls > $L/block_controls.log 2>&1
grep -q "PASS\|FAIL" $L/converge_M80.log 2>/dev/null || nice -n 10 python3 -u highm_check.py converge --M=80 --json=data/lowD_highM/converge_M80.json > $L/converge_M80.log 2>&1
echo STAGE1_DONE
