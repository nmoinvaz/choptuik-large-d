#!/bin/sh
# Review A3: the cheap mutations of proof/A_lemmas_M.md section 8 for step (i), the system and lemma S, run directly
# (mutations_M.py would overwrite lemmas_M_runs/mutations_M.json), plus fp_series on the certificate's own
# configuration (ray 5pi/16, r0 = 0.0741, diag), which the table has at M = 5 only. Run from solver/proof/review_a3/;
# writes run_controls2.log. "caught" = exit 0 for these scripts (each exits 0 when its mutation is caught).
cd "$(dirname "$0")/.." || exit 1
LOG=review_a3/run_controls2.log
: > "$LOG"
run() {
    echo "=== $*" >> "$LOG"
    nice -n 10 "$@" >> "$LOG" 2>&1
    echo "=== exit $?" >> "$LOG"
}
run python3 fixed_point_M.py --M=3 --mutate=series                                    # caught: exit 0
run python3 fixed_point_M.py --M=3 --mutate=series --r0=0.0741 --arg=5pi/16 --diag   # caught: exit 0 (not in the table)
run python3 fixed_point_M.py --M=3 --r0=0.0741 --arg=5pi/16 --diag                   # clean, same configuration: holds
run python3 fixed_point_M.py --M=3 --mutate=cdrop                                     # caught: exit 0
run python3 fixed_point_M.py --M=3 --mutate=diagpath                                  # refused: exit 0
run python3 lemma6_usys_M.py --mutate=label                                           # caught: exit 0
run python3 lemma6_sector_M.py --M=3 --rp=0.1 --k=20 --mutate=series                 # caught: exit 0
echo "=== done" >> "$LOG"
