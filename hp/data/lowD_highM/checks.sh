#!/bin/bash
# The checks of notes/HIGH_M.md section 2 (idempotent: a check whose log ends in a verdict is skipped). Run from hp/
# with PYTHONPATH pointing at a build of this branch, REF_MAIN at the directory holding REF_mixed_main.npz,
# REF_highm_main.json and py_main/ (a build of the parent commit, a1973f7).
set -u
cd "$(dirname "$0")/../.."
L=data/lowD_highM/logs
D=data/lowD_highM
done_() { [ -s "$1" ] && tail -3 "$1" | grep -q "$2"; }
done_ $L/ref_mixed_compare.log "PASS\|FAIL" || nice -n 10 python3 -u mixed_check.py ref --compare=$REF_MAIN/REF_mixed_main.npz > $L/ref_mixed_compare.log 2>&1
PYTHONPATH=$REF_MAIN/py_main nice -n 10 python3 -u highm_check.py ref --save=$REF_MAIN/REF_highm_main.json >> $L/ref_highm_save_main.log 2>&1
cp $REF_MAIN/REF_highm_main.json $D/REF_highm_main_a1973f7.json
done_ $L/ref_highm_compare.log "PASS\|FAIL" || nice -n 10 python3 -u highm_check.py ref --compare=$D/REF_highm_main_a1973f7.json > $L/ref_highm_compare.log 2>&1
done_ $L/block_check_small.log "^PASS\|^FAIL" || nice -n 10 python3 -u highm_check.py block --Ms=16,48 --variants=f64x4,f64x1,dd4 --json=$D/block_check_small.json > $L/block_check_small.log 2>&1
done_ $L/block_check_large.log "^PASS\|^FAIL" || nice -n 10 python3 -u highm_check.py block --Ms=80,128 --variants=f64x4 --json=$D/block_check_large.json > $L/block_check_large.log 2>&1
done_ $L/block_controls.log "controls:" || nice -n 10 python3 -u highm_check.py controls > $L/block_controls.log 2>&1
done_ $L/converge_M80.log "PASS\|FAIL" || nice -n 10 python3 -u highm_check.py converge --M=80 --json=$D/converge_M80.json > $L/converge_M80.log 2>&1
echo CHECKS_DONE
