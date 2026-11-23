#!/bin/bash
# Review of #1079, finding 10: the planted truth of the C^1 sets at a box where its mutation can fire. A C^1 ray run
# of lemma6b1_cover_M.py at each excited centre with h = 1e-9 (a control only, not part of the covering of record,
# which is at h = 1e-11), then c1check (must be True) and c1check --mutate=nov (must be True: every moved point out).
# usage: bash review_1079/c1_control.sh RAW     from solver/proof; writes RAW/I1079-mac-review/
{
RAW=$1
PY=${PY:-/Users/nathan/Source/choptuik-large-d/.venv/bin/python}
OUT=$RAW/I1079-mac-review
H=$OUT/light_steps.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for Z in 1 2; do
  if [ $Z = 1 ]; then TAG=M3x1E2_h1e-12_w1e-13; else TAG=M3x2E2_h1e-12_w1e-12; fi
  XP=$("$PY" -I -c 'import json, sys; print(",".join(json.load(open(sys.argv[1]))["p"]))' "$RAW/I1079-macm3/levers/data/A_M_box_$TAG.json")
  COM="--M=3 --x=$XP --h=1e-9 --tag=_rv1079c$Z"
  nice -n 10 "$PY" lemma6b1_cover_M.py ray --c1 $COM > "$OUT/logs/C1CTRL_ray_exc$Z.log" 2>&1
  echo "$(date -u '+%F %T') C1CTRL_ray_exc$Z (h 1e-9) exit $?: $(tail -1 "$OUT/logs/C1CTRL_ray_exc$Z.log" | cut -c1-160)" >> "$H"
  nice -n 10 "$PY" lemma6b1_cover_M.py c1check $COM > "$OUT/logs/C1CTRL_c1check_exc$Z.log" 2>&1
  echo "$(date -u '+%F %T') C1CTRL_c1check_exc$Z (h 1e-9) exit $?: $(tail -1 "$OUT/logs/C1CTRL_c1check_exc$Z.log" | cut -c1-160)" >> "$H"
  nice -n 10 "$PY" lemma6b1_cover_M.py c1check $COM --mutate=nov > "$OUT/logs/C1CTRL_c1check_exc${Z}_nov.log" 2>&1
  echo "$(date -u '+%F %T') C1CTRL_c1check_exc${Z}_nov (h 1e-9) exit $?: $(tail -1 "$OUT/logs/C1CTRL_c1check_exc${Z}_nov.log" | cut -c1-160)" >> "$H"
done
exit 0
}
