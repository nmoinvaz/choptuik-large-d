#!/bin/bash
# Track B of #1088: every check and solve of notes/LARGE_D_FACTORED.md, one process at a time, logs into OUT.
#   bash solver/factored/run_track_b.sh OUT STATES [stage...]
# OUT: a task folder of the raw-data repository (I1088-mac/); STATES: a folder with the three hp states
#   T138-win2/chain/anm_pin3_D100.0000.npz, T138-win2/chain_o/anm_pin3_D200.0000.npz and
#   I603-win2/chain/anm_pin3_D245.1366.npz of the raw-data repository, e.g.
#   git -C ../choptuik-large-d-raw show origin/main:T138-win2/chain/anm_pin3_D100.0000.npz > STATES/anm_pin3_D100.0000.npz
# stages: checks, s100, s200, s245, jumps (default: all)
{
set -u
OUT=$1
ST=$2
shift 2
STAGES=${*:-checks s100 s200 s245 jumps}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/Users/nathan/venv-314/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
run() { nice -n 10 "$PY" "$@"; }
cd "$HERE" || exit 1
for st in $STAGES; do
  case $st in
  checks)
    run derive_factored.py > "$OUT/derive.log" 2>&1
    for k in 1 2 3 4 5 6 7; do run derive_factored.py --mutate=$k > "$OUT/derive_mutate_$k.log" 2>&1; done
    run check_state.py "$ST/anm_pin3_D100.0000.npz" > "$OUT/check_state_D100.log" 2>&1
    run check_state.py "$ST/anm_pin3_D100.0000.npz" --mutate=sigma > "$OUT/check_state_D100_mutate_sigma.log" 2>&1
    run check_s.py "$ST/anm_pin3_D100.0000.npz" > "$OUT/check_s_D100.log" 2>&1
    run check_s.py "$ST/anm_pin3_D245.1366.npz" > "$OUT/check_s_D245.log" 2>&1
    for m in noslave evenD sigma; do run check_s.py "$ST/anm_pin3_D100.0000.npz" --mutate=$m > "$OUT/check_s_D100_mutate_$m.log" 2>&1; done
    run anchor_noise.py "$ST/anm_pin3_D100.0000.npz" > "$OUT/anchor_noise.log" 2>&1
    ;;
  s100)
    run fsolve.py --form=S --D=100 --M=8 --start="$ST/anm_pin3_D100.0000.npz" --iters=10 --out="$OUT/S64_D100_M8.json" > "$OUT/S64_D100_M8.log" 2>&1
    run solve_dd.py --D=100 --M=8 --start="$OUT/S64_D100_M8.json" --f64iters=1 --outer=8 --out="$OUT/Sdd_D100_M8.json" > "$OUT/Sdd_D100_M8.log" 2>&1
    ;;
  s200)
    run fsolve.py --form=S --D=200 --M=8 --start="$ST/anm_pin3_D200.0000.npz" --iters=12 --out="$OUT/S64_D200_M8.json" > "$OUT/S64_D200_M8.log" 2>&1
    run solve_dd.py --D=200 --M=8 --start="$OUT/S64_D200_M8.json" --f64iters=1 --outer=8 --out="$OUT/Sdd_D200_M8.json" > "$OUT/Sdd_D200_M8.log" 2>&1
    ;;
  s245)
    run fsolve.py --form=S --D=245.1366177558899 --M=8 --start="$ST/anm_pin3_D245.1366.npz" --iters=12 --out="$OUT/S64_D245_M8.json" > "$OUT/S64_D245_M8.log" 2>&1
    run solve_dd.py --D=245.1366177558899 --M=8 --start="$OUT/S64_D245_M8.json" --f64iters=1 --outer=8 --out="$OUT/Sdd_D245_M8.json" > "$OUT/Sdd_D245_M8.log" 2>&1
    ;;
  jumps)
    prev="$OUT/Sdd_D245_M8.json"
    for D in 263 300 400; do
      run solve_dd.py --D=$D --M=8 --start="$prev" --f64iters=15 --outer=10 --out="$OUT/Sdd_D${D}_M8.json" > "$OUT/Sdd_D${D}_M8.log" 2>&1
      [ -s "$OUT/Sdd_D${D}_M8.json" ] && prev="$OUT/Sdd_D${D}_M8.json"
    done
    ;;
  esac
done
exit 0
}
