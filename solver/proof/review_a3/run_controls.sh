#!/bin/sh
# Review A3: rerun the mutation controls that proof/A_M.md and proof/A_lemmas_M.md cite, the cheap ones only
# (seconds to a few minutes each). Run from solver/proof/review_a3/; writes run_controls.log next to this file.
# Each line records the command and its exit status; the expected verdict is in the comment after it.
cd "$(dirname "$0")/.." || exit 1
LOG=review_a3/run_controls.log
: > "$LOG"
run() {
    echo "=== $*" >> "$LOG"
    nice -n 10 "$@" >> "$LOG" 2>&1
    echo "=== exit $?" >> "$LOG"
}
run python3 certify_M_sys.py selftest --mutate=coupling          # must FAIL (exit 1)
run python3 certify_M.py krawczyk --box=data/A_M_box_M3_1e-10.json --hier                  # pass, exit 0
run python3 certify_M.py krawczyk --box=data/A_M_box_M3_1e-10.json --hier --mutate=hier    # exit 1
run python3 certify_M.py controls --box=data/A_M_box_M3_1e-10.json                         # exit 0
run python3 certify_M_recheck.py data/A_M_box_M3_1e-10.json                                 # exit 0
run python3 certify_M_recheck.py data/A_M_box_M3_1e-10.json --mutate                        # exit 1
run python3 certify_M.py startcheck 3                                                       # contains, exit 0
run python3 certify_M.py startcheck 3 --mutate                                              # exit 1
run python3 certify_M.py crosscheck 3                                                       # AGREE, exit 0
run python3 certify_M.py crosscheck 3 --mutate                                              # DISAGREE, exit 1
run python3 lemma6_usys_M.py                                                                # exit 0
echo "=== done" >> "$LOG"
