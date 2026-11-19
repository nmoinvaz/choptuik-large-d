#!/bin/bash
# #1025 on a second machine: k_c 63 at D 17 and 17.5 (re-solves of the Mac's k_c 55 states, with lambda), the k_c 63
# continuation from D 17.5 to 20 (ell_leap --fold, psi-constraint bar 1e-2), lambda at k_c 63 on its half-integer
# states, and the eigenvalues near lambda_1 (ell_eigs) on the Mac's k_c 47 and 55 states at D 16-17.5.
# Four processes with 2 BLAS threads each:
#   P1 k_c 63 starts, then the continuation; P2 lambda on the continuation's states; P3a, P3b the eigenvalue scans.
# Inputs: the raw clone's I1025-mac/out (states and lambda from the Mac). Outputs: this worktree's data/ (records,
# states) and $LOGS (logs); the publish step copies them into the raw clone's I1025-win/. No job writes in the raw clone.
# usage (bash; Git Bash on Windows), from solver/matter of a worktree of wip/ell1-highD:
#   RAW=<choptuik-large-d-raw clone> PY=<python> LOGS=<log folder> bash ell_win1025.sh
#   RAW=... LOGS=... bash ell_win1025.sh publish     (copies the outputs into $RAW/I1025-win/; then commit as usual)
set -u
: "${RAW:?set RAW}" "${LOGS:?set LOGS}"
cd "$(dirname "$0")"
OUT=$RAW/I1025-win
if [ "${1:-}" = "publish" ]; then
  mkdir -p $OUT/out
  cp $LOGS/*.log $OUT/
  cp data/l1kc63*1025.jsonl data/sol_l1kc63*1025_D*.npy data/eigs_win1025_*.jsonl $OUT/out/ 2>/dev/null
  [ -f $OUT/README.md ] || cat > $OUT/README.md <<'EOF'
# I1025-win: the l = 1 DSS solution at k_c 63 from D 17 to 20, its growth rate, and the eigenvalues near it on the Mac's k_c 47 and 55 states (logs, records, states)

Machine: winb (Windows 11, i7-1185G7, 8 threads). Issue: https://github.com/nmoinvaz/choptuik-large-d/issues/1025

Code: `solver/matter/ell_win1025.sh` on branch `wip/ell1-highD` of choptuik-large-d (ell_refine.py, ell_leap.py,
ell_eigs.py). Inputs: I1025-mac/out (the k_c 55 states at D 17 and 17.5, the k_c 47 states at D 16-17.5).

| file | what |
|---|---|
| `checks.log` | ell_eigs.py planted (exit 0) and its mutation --mutate=beyn (exit 1) |
| `kc63_start.log`, `out/l1kc63Fw1025.jsonl`, `out/sol_l1kc63Fw1025_D*.npy` | k_c 63 at D 17, 17.5 from k_c 55, with lambda; then lambda at D 18-20 |
| `leap_kc63.log`, `out/l1kc63W1025.jsonl`, `out/sol_l1kc63W1025_D*.npy` | ell_leap --fold at k_c 63, D 17.5 -> 20 |
| `lam_kc63.log` | lambda at k_c 63 on the continuation's half-integer states |
| `eigs_a.log`, `eigs_b.log`, `out/eigs_win1025_*.jsonl` | ell_eigs: eigenvalues inside a circle around lambda_1 (Beyn), with the state checks |
EOF
  exit 0
fi
: "${PY:?set PY}"
timeout --version > /dev/null 2>&1 || { echo "needs coreutils timeout (Git Bash's), not Windows' timeout.exe"; exit 1; }
mkdir -p $LOGS
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
run() { log=$1; shift; echo "== $(date -u '+%Y-%m-%d %H:%M:%S') $*" >> $LOGS/$log
        timeout 43200 "$PY" -u "$@" >> $LOGS/$log 2>&1; echo "== exit $? $(date -u '+%Y-%m-%d %H:%M:%S')" >> $LOGS/$log; }
OPT="--kappa 0.7 --u0 0.04 --Ks 30 --tol 1e-12"
EIG="--r 0 --N 24 --p 12 --dlam 0 --no_polish_complex"
IN=$RAW/I1025-mac/out
for f in sol_l1kc55Fs1025_D17.0000.npy sol_l1kc55Fs1025_D17.5000.npy sol_l1kc47Fr1025_D16.0000.npy \
         sol_l1kc47Fr1025_D16.5000.npy sol_l1kc47Fr1025_D17.0000.npy sol_l1kc47Fr1025_D17.5000.npy; do
  [ -f $IN/$f ] || { echo "missing input $IN/$f (pull the raw clone)"; exit 1; }
done
cp $IN/sol_l1kc55Fs1025_D17.0000.npy $IN/sol_l1kc55Fs1025_D17.5000.npy data/
run checks.log ell_eigs.py planted
run checks.log ell_eigs.py planted --mutate=beyn
# P1: k_c 63 starts (guess: lambda at k_c 55, D 17), then the continuation
( run kc63_start.log ell_refine.py l1kc55Fs1025 55 l1kc63Fw1025 63 1 17,17.5 2.025718689014619 $OPT --wait 60
  run leap_kc63.log ell_leap.py l1kc63W1025 1 63 data/sol_l1kc63Fw1025_D17.0000.npy 17 \
      data/sol_l1kc63Fw1025_D17.5000.npy 17.5 20 0.125 --tol 1e-12 --fmax 1e-8 --cmax 1e-2 --fold ) &
# P2: lambda at k_c 63 on the continuation's half-integer states, once the starts have theirs (same tag: the guess
# extrapolates from them); a state that does not appear within 2 h ends it
( t0=$(date +%s)
  until [ "$(grep -c '"lam": [0-9]' data/l1kc63Fw1025.jsonl 2>/dev/null || echo 0)" -ge 2 ]; do
    sleep 60; [ $(( $(date +%s) - t0 )) -gt 14400 ] && { echo "== gave up waiting for the k_c 63 starts" >> $LOGS/lam_kc63.log; exit 1; }
  done
  run lam_kc63.log ell_refine.py l1kc63W1025 63 l1kc63Fw1025 63 1 18,18.5,19,19.5,20 1.9 $OPT --wait 7200 ) &
# P3a, P3b: eigenvalues near lambda_1 on the Mac's states (lambda from I1025-mac/out/l1kc47Fr1025.jsonl and
# l1kc55Fs1025.jsonl)
( run eigs_a.log ell_eigs.py $IN/sol_l1kc47Fr1025_D17.5000.npy 17.5 47 1.8831582965844342 $EIG --check --out data/eigs_win1025_a.jsonl
  run eigs_a.log ell_eigs.py $IN/sol_l1kc47Fr1025_D17.0000.npy 17 47 2.03637873040092 $EIG --out data/eigs_win1025_a.jsonl
  run eigs_a.log ell_eigs.py $IN/sol_l1kc47Fr1025_D16.0000.npy 16 47 2.247129257178688 $EIG --out data/eigs_win1025_a.jsonl ) &
( run eigs_b.log ell_eigs.py $IN/sol_l1kc55Fs1025_D17.5000.npy 17.5 55 1.9522284485089008 $EIG --check --out data/eigs_win1025_b.jsonl
  run eigs_b.log ell_eigs.py $IN/sol_l1kc55Fs1025_D17.0000.npy 17 55 2.025718689014619 $EIG --out data/eigs_win1025_b.jsonl
  run eigs_b.log ell_eigs.py $IN/sol_l1kc47Fr1025_D16.5000.npy 16.5 47 2.1548214799814693 $EIG --out data/eigs_win1025_b.jsonl ) &
wait
echo "== $(date -u '+%Y-%m-%d %H:%M:%S') all done" >> $LOGS/checks.log
