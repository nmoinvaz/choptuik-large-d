#!/bin/bash
# Review of #1079, check 5: one box run of an excited zero with a planted last-bit perturbation, then the Krawczyk
# test with macm3's point file.
# usage: bash run_perturb.sh RAW ZERO WHAT [SEED [ULPS]]    from solver/proof
#   RAW   the raw-data clone (reads I1079-macm3/levers/data, writes I1079-mac-review/)
#   ZERO  x1 (box h 1e-12, w 1e-13) or x2 (h 1e-12, w 1e-12);  WHAT none | qr | qrin | perron
#   environment: REVIEW_NJAC=n another order of the step Jacobian (default 16; the tag gains _nj<n>);
#                REVIEW_TIES=1 counts near ties of the pivot order (the tag gains _ties);
#                REVIEW_STABLE=1 a pivot order with rounded weights and ties by index (a trial; the tag gains _stable)
{
RAW=$1; Z=$2; WHAT=$3; SEED=${4:-1079}; ULPS=${5:-1}
PY=${PY:-/Users/nathan/Source/choptuik-large-d/.venv/bin/python}
IN=$RAW/I1079-macm3/levers/data
OUT=$RAW/I1079-mac-review
case $Z in
  x1) W=1e-13 ;;
  x2) W=1e-12 ;;
  *) echo "unknown zero $Z"; exit 2 ;;
esac
T=${Z}E2
TAG=M3${T}_h1e-12_w${W}_${WHAT}_s${SEED}
[ "$ULPS" = 1 ] || TAG=${TAG}_u${ULPS}
[ -z "$REVIEW_NJAC" ] || TAG=${TAG}_nj${REVIEW_NJAC}
[ -z "$REVIEW_TIES" ] || TAG=${TAG}_ties
[ -z "$REVIEW_STABLE" ] || TAG=${TAG}_stable
export CERTIFY_M_NTAY=28 CERTIFY_M_NJAC=${REVIEW_NJAC:-16} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
unset CERTIFY_M_XI1
export REVIEW_PERTURB=$WHAT REVIEW_SEED=$SEED REVIEW_ULPS=$ULPS
mkdir -p "$OUT/out" "$OUT/logs"
BOX=$OUT/out/A_M_box_$TAG.json
[ ! -e "$BOX" ] || { echo "$BOX exists"; exit 2; }
echo "$(date -u '+%F %T') start $TAG" >> "$OUT/steps.txt"
nice -n 10 "$PY" review_1079/perturb_box.py box 3 --centre="$IN/A_M_refine_M3_$T.json" --use_out --h=1e-12 --w=$W \
    --out="$BOX" --log="$OUT/logs/A_M_box_$TAG.log" > "$OUT/logs/A_M_box_${TAG}_stdout.log" 2>&1
rc=$?
echo "$(date -u '+%F %T') box $TAG exit $rc: $(tail -1 "$OUT/logs/A_M_box_${TAG}_stdout.log" | cut -c1-200)" >> "$OUT/steps.txt"
[ $rc = 0 ] || exit 1
unset REVIEW_PERTURB REVIEW_SEED REVIEW_ULPS
nice -n 10 "$PY" certify_M.py krawczyk --box="$BOX" --point="$IN/A_M_point_M3$T.json" > "$OUT/logs/A_M_krawczyk_pt_$TAG.log" 2>&1
k=$?
echo "$(date -u '+%F %T') krawczyk $TAG (macm3's point file) exit $k: $(head -1 "$OUT/logs/A_M_krawczyk_pt_$TAG.log" | cut -c1-200)" >> "$OUT/steps.txt"
exit 0
}
