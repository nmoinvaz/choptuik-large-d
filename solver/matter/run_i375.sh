#!/usr/bin/env bash
# #375: the imaginary sector's det M in double-double at D = 30.07, 45.05, 60.00 (complex_imag_dd.py), on a runner.
# Usage: run_i375.sh RAWDIR   (RAWDIR = <clone of choptuik-large-d-raw>/I375-<machine>, made beforehand with its README)
# Run from solver/matter/ of a checkout at origin/wip/xf-checks (or main once merged). One process per thread.
set -u
RAW=$1
mkdir -p "$RAW"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=${PY:-python3}
ST=../../hp/data/states
TOL=1e-14
{
  date -u '+start %Y-%m-%d %H:%M:%S UTC'
  $PY -c "import numpy, scipy, mpmath, sys; print(sys.version.split()[0], numpy.__version__, scipy.__version__, mpmath.__version__)"
  git rev-parse HEAD
  $PY ddnp.py selftest; echo "ddnp selftest exit $?"
  $PY ddnp.py selftest --mutate=twoprod; echo "ddnp control exit $? (must be 1)"
  $PY complex_imag_dd.py selftest; echo "complex_imag_dd selftest exit $?"
} > "$RAW/selftests.log" 2>&1
# phase 1 (12 processes): the tolerance acceptance (6 lambda x tol 1e-12, 1e-14 per D) and the nine box counts
for s in up6_D30.0688 up7_D45.0507 up13_D60.0020; do
  nice -n 10 $PY complex_imag_dd.py tolscan $ST/$s.npz --tols=1e-12,1e-14 --out="$RAW/tolscan_$s.jsonl" \
      > "$RAW/tolscan_$s.log" 2>&1 &
  for b in B0 U0 FULL; do
    nice -n 10 $PY complex_imag_dd.py count $ST/$s.npz $b --tol=$TOL --evlog="$RAW/evals_${s}_$b.jsonl" \
        --out="$RAW/count_${s}_$b.jsonl" > "$RAW/count_${s}_$b.log" 2>&1 &
  done
done
wait
date -u '+phase 1 done %Y-%m-%d %H:%M:%S UTC' >> "$RAW/selftests.log"
# phase 2 (6 processes): the controls. A pair planted at 3.1 +- 1.0i in FULL (reuses FULL's evaluations) and the
# mutated coupling (the P equation's (n/2) c (P + Q) times 1.1) in B0, at each D
for s in up6_D30.0688 up7_D45.0507 up13_D60.0020; do
  nice -n 10 $PY complex_imag_dd.py count $ST/$s.npz FULL --plant=3.1+1.0j --tol=$TOL \
      --evlog="$RAW/evals_${s}_FULL.jsonl" --out="$RAW/count_${s}_FULL_plant.jsonl" > "$RAW/count_${s}_FULL_plant.log" 2>&1 &
  nice -n 10 $PY complex_imag_dd.py count $ST/$s.npz B0 --mutate=coupling --tol=$TOL \
      --evlog="$RAW/evals_${s}_B0_coupling.jsonl" --out="$RAW/count_${s}_B0_coupling.jsonl" \
      > "$RAW/count_${s}_B0_coupling.log" 2>&1 &
done
wait
date -u '+phase 2 done %Y-%m-%d %H:%M:%S UTC' >> "$RAW/selftests.log"
