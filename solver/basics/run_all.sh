#!/usr/bin/env bash
# Runs every check of notes/LARGE_D_BASICS.md with its control and writes the logs to OUT (#1086).
# Usage: bash solver/basics/run_all.sh OUT STATES [PYTHON]
#   OUT: a folder for the logs (the raw-data repository's I1086-mac/)
#   STATES: a folder holding the chains' anm_pin3_D*.npz, e.g. from
#     git -C ../choptuik-large-d-raw archive origin/main I603-win2/chain I603-win2/chain_nsr I841-dh370/chain \
#       I903-macm3/chain_k135 I903-mac/chain_fixed I966-dh370/ks_chain3b I461-win2/crawlB5 I461-win2/crawlC | tar -x -C STATES
# Run from the repository root.
{
set -u
OUT=$1
ST=$2
PY=${3:-python3}
mkdir -p "$OUT"
B=solver/basics
{ nice -n 10 "$PY" $B/derive.py; echo "exit $?"
  for k in 1 2 3 4 5; do echo "--mutate=$k:"; nice -n 10 "$PY" $B/derive.py --mutate=$k | grep -E "^$k\."; done; } > "$OUT/derive.txt" 2>&1
nice -n 10 "$PY" $B/free_field.py > "$OUT/free_field.txt" 2>&1
nice -n 10 "$PY" $B/free_field.py --mutate=A > "$OUT/free_field_mutate_A.txt" 2>&1
nice -n 10 "$PY" $B/order_laws.py > "$OUT/order_laws.txt" 2>&1
nice -n 10 "$PY" $B/order_laws.py --mutate=binom > "$OUT/order_laws_mutate_binom.txt" 2>&1
nice -n 10 "$PY" $B/order_laws.py --tails > "$OUT/order_laws_tails.txt" 2>&1
nice -n 10 "$PY" $B/order_laws.py --tails --mutate=binom > "$OUT/order_laws_tails_mutate_binom.txt" 2>&1
for r in ks293 chain_k135; do
  nice -n 10 "$PY" $B/sigma_vs_ks.py "$ST/" --ref=$r > "$OUT/sigma_vs_ks_ref_$r.txt" 2>&1
done
nice -n 10 "$PY" $B/sigma_vs_ks.py "$ST/" --ref=chain_k135 --mutate=g > "$OUT/sigma_vs_ks_ref_chain_k135_mutate_g.txt" 2>&1
nice -n 10 "$PY" $B/eeg_gap.py hp/data/delta_all.csv > "$OUT/eeg_gap.txt" 2>&1
nice -n 10 "$PY" $B/eeg_gap.py hp/data/delta_all.csv --mutate=shift > "$OUT/eeg_gap_mutate_shift.txt" 2>&1
nice -n 10 "$PY" $B/eeg_gap.py hp/data/delta_all.csv --at="$OUT/sigma_vs_ks_ref_ks293.txt" > "$OUT/chain_vs_dd_fit.txt" 2>&1
nice -n 10 "$PY" $B/oscillation_fit.py "$OUT/chain_vs_dd_fit.txt" > "$OUT/oscillation_fit.txt" 2>&1
nice -n 10 "$PY" $B/oscillation_fit.py "$OUT/chain_vs_dd_fit.txt" --dmin=256 > "$OUT/oscillation_fit_from256.txt" 2>&1
nice -n 10 "$PY" $B/oscillation_fit.py "$OUT/chain_vs_dd_fit.txt" --mutate=shuffle > "$OUT/oscillation_fit_mutate_shuffle.txt" 2>&1
nice -n 10 "$PY" $B/order_laws.py --tails --ks=130,135,142,150,160,200,240,250 \
  --Ds=150,225.13,232,233,237.13,249.01,253.80,259.9 > "$OUT/order_laws_tails_fixed_orders.txt" 2>&1
SL=(I461-win2/crawlC/anm_pin3_D238.7562.npz I903-macm3/chain_k135/anm_pin3_D226.1312.npz
    I966-dh370/ks_chain3b/leg15_D263.4862_ks293/anm_pin3_D263.5987.npz)
nice -n 10 "$PY" $B/slow_check.py "${SL[@]/#/$ST/}" > "$OUT/slow_check.txt" 2>&1
nice -n 10 "$PY" $B/slow_check.py "${SL[@]/#/$ST/}" --mutate=K2 > "$OUT/slow_check_mutate_K2.txt" 2>&1
PL=(I603-win2/chain_nsr/anm_pin3_D259.8997.npz I966-dh370/ks_chain3b/leg10_D256.1776_ks293/anm_pin3_D256.5265.npz
    I966-dh370/ks_chain3b/leg15_D263.4862_ks293/anm_pin3_D263.5987.npz)
nice -n 10 "$PY" $B/plan_stability.py "${PL[@]/#/$ST/}" > "$OUT/plan_stability.txt" 2>&1
nice -n 10 "$PY" $B/plan_stability.py "${PL[@]/#/$ST/}" --mutate=h2 > "$OUT/plan_stability_mutate_h2.txt" 2>&1
echo done
exit
}
