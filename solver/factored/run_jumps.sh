#!/bin/bash
# Jumps in D for form S at M points (#1088): each D is solved from the previous D's solution rescaled (solve_dd.py,
# no continuation in D). The Delta seed is the reference value at D (table or series rows of hp/data/delta_all.csv,
# hard-coded below) plus the previous solution's offset from its own reference.
#   bash solver/factored/run_jumps.sh OUT START.json "D1:ref1 D2:ref2 ..." [M] [extra solve_dd options]
{
set -u
OUT=$1
prev=$2
LIST=$3
M=${4:-8}
shift 4 2>/dev/null || shift $#
EXTRA="$*"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/Users/nathan/venv-314/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$HERE" || exit 1
for item in $LIST; do
  D=${item%%:*}
  ref=${item##*:}
  seed=$("$PY" -c "import json,sys; j=json.load(open(sys.argv[1])); print(repr(float(sys.argv[3]) + j['Delta'] - float(j.get('ref', sys.argv[3]))))" "$prev" "$D" "$ref")
  out="$OUT/Sdd_D${D}_M${M}.json"
  nice -n 10 "$PY" solve_dd.py --D=$D --M=$M --start="$prev" --Delta=$seed --f64iters=4 --outer=6 --rounds=4 $EXTRA --out="$out" > "$OUT/Sdd_D${D}_M${M}.log" 2>&1
  if [ -s "$out" ]; then
    "$PY" -c "import json,sys; j=json.load(open(sys.argv[1])); j['ref']=float(sys.argv[2]); json.dump(j, open(sys.argv[1],'w'), indent=1)" "$out" "$ref"
    prev="$out"
  else
    echo "D $D: no solution file; stop" >> "$OUT/run_jumps.log"
    break
  fi
done
exit 0
}
