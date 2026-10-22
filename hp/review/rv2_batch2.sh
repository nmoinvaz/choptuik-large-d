#!/bin/bash
# Review RV2, second batch (one process at a time, nice 10): the kfreeze cut at several D, the independent residual
# at the second machine's last state, the equation check away from the solution, and the negative controls.
set -u
cd "$(dirname "$0")/../.."
R=hp/review
res() { nice -n 10 python3 $R/rv2_resolve.py "$@" --quiet 2>&1 | tail -1; }
ind() { nice -n 10 python3 $R/rv2_residual.py "$@" 2>&1 | tail -12; }
echo "== kfreeze 15 vs 13, full Newton steps (no line search)"
res hp/data/states/up13_D60.0020.npz --label kfreeze15_nols --kfreeze 15 --corrector nols --its 6 --out $R/rv2_resolve_D60.jsonl
echo "== independent residual: second machine's last state (D = 61.026449)"
ind hp/data/seeds/local/up23_D61.0264.npz --out $R/rv2_residual.jsonl
echo "== independent residual away from the solution (equation check), blocks 0, 1, 5"
ind hp/data/states/up13_D60.0020.npz --perturb 1e-6 --blocks 0,1,5 --out $R/rv2_residual.jsonl
echo "== negative control: Delta + 1e-24 in the independent evaluation only (horizon block)"
ind hp/data/states/up13_D60.0020.npz --break=delta --blocks 5 --out $R/rv2_residual.jsonl
echo "== negative control: n/2 c (p + m) coupling x (1 + 1e-15) in the independent evaluation (segment 1)"
ind hp/data/states/up13_D60.0020.npz --break=eq --blocks 1 --out $R/rv2_residual.jsonl
echo "== kfreeze 15 vs 13 at D = 50 and 40"
res hp/data/states/up8_D50.0075.npz --label kfreeze15_nols --kfreeze 15 --corrector nols --its 6 --out $R/rv2_resolve_D50.jsonl
res hp/data/states/up7_D40.0201.npz --label kfreeze15_nols --kfreeze 15 --corrector nols --its 6 --out $R/rv2_resolve_D40.jsonl
echo "== M 48 at D = 50"
res hp/data/states/up8_D50.0075.npz --label M48 --M 48 --its 8 --out $R/rv2_resolve_D50.jsonl
