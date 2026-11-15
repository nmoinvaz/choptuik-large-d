#!/bin/bash
# The reviewer's rerun of the 32 runs of I427-mac/run_i427.sh (issue #867), from solver/proof/ of the checkout.
# Usage: review_867/rerun_i427.sh <python> <root of the raw-data clone, read only> <output directory>
# The commands are those of run_i427.sh. Only the log and JSON paths differ: they go to the output directory.
# exits.txt there has, per run, its exit status, the status it must have and its command.
PY=$1; ROOT=$2; OUT=$3
RAY=$ROOT/I38-macm3/lemma6b1_runs_M5_cert/ray.json
RAYW=$ROOT/I121-win2/lemma6b1_runs_M5_cert/ray.json
: > "$OUT/exits.txt"
run() {   # run <log name> <expected exit> <command...>
  local log=$1 want=$2; shift 2
  nice -n 10 "$@" > "$OUT/$log.log" 2>&1
  local rc=$?
  echo "$log: exit $rc (must be $want): $*" | sed "s|$PY|python3|; s|$ROOT/|RAW/|g; s|$OUT/|OUT/|g" >> "$OUT/exits.txt"
}
START="--start --rho0=405 --h=1e-18 --cauchy=1e-3"
for M in 1 3 5; do run planted_M$M 0 $PY fixed_point_M_b.py --planted --M=$M; done
for M in 1 3 5; do run planted_M${M}_edge 0 $PY fixed_point_M_b.py --planted --M=$M --rho0=204.1; done
for m in truth coef radius; do run planted_M5_mut_$m 3 $PY fixed_point_M_b.py --planted --M=5 --mutate=$m; done
run check_m1 0 $PY fixed_point_M_b_check.py m1
for m in coef sign scale; do run check_m1_mut_$m 3 $PY fixed_point_M_b_check.py m1 --mutate=$m; done
run lip_M1 0 $PY fixed_point_M_b_check.py lip --M=1
run lip_M3 0 $PY fixed_point_M_b_check.py lip --M=3
run lip_M5 0 $PY fixed_point_M_b_check.py lip --M=5
run lip_M5_sector 0 $PY fixed_point_M_b_check.py lip --M=5 --sector=0.2
run lip_M5_mut_drop 3 $PY fixed_point_M_b_check.py lip --M=5 --mutate=drop
run two_M5_60_105 0 $PY fixed_point_M_b.py --M=5 --two=60,105 --compare=$RAY
run two_M5_95_105 0 $PY fixed_point_M_b.py --M=5 --two=95,105 --compare=$RAY
run two_M5_60_105_mut_forcing 3 $PY fixed_point_M_b.py --M=5 --two=60,105 --compare=$RAY --mutate=forcing
run start_M5 0 $PY fixed_point_M_b.py --M=5 $START --compare=$RAY --json=$OUT/start_M5.json
run start_M5_sector 0 $PY fixed_point_M_b.py --M=5 $START --sector=0.2 --compare=$RAY --json=$OUT/start_M5_sector.json
run start_M5_k40_sector 0 $PY fixed_point_M_b.py --M=5 $START --sector=0.2 --k=40 --compare=$RAY --json=$OUT/start_M5_k40_sector.json
for m in radius coef sign scale rho0; do run start_M5_mut_$m 3 $PY fixed_point_M_b.py --M=5 $START --compare=$RAY --mutate=$m; done
run start_M5_vs_win2 0 $PY fixed_point_M_b.py --M=5 $START --compare=$RAYW
run start_M3_cert 0 $PY fixed_point_M_b.py --M=3 --start --rho0=262 --r0=0.1 --h=1.1e-9 --cauchy=1e-3 --compare=lemma6b1_runs_M3_cert/ray.json --json=$OUT/start_M3_cert.json
run start_M3_cert_mut_sign 3 $PY fixed_point_M_b.py --M=3 --start --rho0=262 --r0=0.1 --h=1.1e-9 --cauchy=1e-3 --compare=lemma6b1_runs_M3_cert/ray.json --mutate=sign
bad=$(grep -c -v -E 'exit ([0-9]+) \(must be \1\)' "$OUT/exits.txt")
echo "RERUN: $(wc -l < "$OUT/exits.txt" | tr -d ' ') runs, $bad with another exit status than required" | tee -a "$OUT/exits.txt"
[ "$bad" -eq 0 ]
