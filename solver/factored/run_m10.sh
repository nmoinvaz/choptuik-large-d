#!/bin/bash
# Form S at M 10 from hp's states (#1088): float64 damped Newton, then the dd split Newton, at each D given.
#   bash solver/factored/run_m10.sh OUT STATES "200:anm_pin3_D200.0000.npz 245.1366177558899:anm_pin3_D245.1366.npz"
{
set -u
OUT=$1
ST=$2
LIST=$3
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/Users/nathan/venv-314/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$HERE" || exit 1
for item in $LIST; do
  D=${item%%:*}
  f=${item##*:}
  tag=${D%%.*}
  nice -n 10 "$PY" fsolve.py --form=S --D=$D --M=10 --start="$ST/$f" --iters=12 --out="$OUT/S64_D${tag}_M10.json" > "$OUT/S64_D${tag}_M10.log" 2>&1
  nice -n 10 "$PY" solve_dd.py --D=$D --M=10 --start="$OUT/S64_D${tag}_M10.json" --f64iters=1 --outer=6 --rounds=3 --tol=1e-24 --out="$OUT/Sdd_D${tag}_M10.json" > "$OUT/Sdd_D${tag}_M10.log" 2>&1
done
exit 0
}
