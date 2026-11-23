#!/bin/bash
# Review of #1079: the short reruns on the reviewer's machine, one after the other (one process). Each line of
# $OUT/light_steps.txt has the command's name, its exit status and the line that carries the verdict.
# usage: bash review_1079/light_checks.sh RAW [STAGE ...]     from solver/proof; stages: xload h1 h3 incl h2 path
#   RAW the raw-data clone (reads I1079-macm3/ and I1079-dh610/, writes I1079-mac-review/)
{
RAW=$1; shift
STAGES=${*:-xload h1 h3 incl h2 path}
PY=${PY:-/Users/nathan/Source/choptuik-large-d/.venv/bin/python}
M=$RAW/I1079-macm3
DH=$RAW/I1079-dh610/out
OUT=$RAW/I1079-mac-review
LG=$OUT/logs
H=$OUT/light_steps.txt
mkdir -p "$LG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
unset CERTIFY_M_XI1 CERTIFY_M_NTAY CERTIFY_M_NJAC
run() {   # run NAME PATTERN CMD...: log to $LG/NAME.log; record the exit status and the last line matching PATTERN
    local name=$1 pat=$2; shift 2
    nice -n 10 "$PY" "$@" > "$LG/$name.log" 2>&1; local rc=$?
    echo "$(date -u '+%F %T') $name exit $rc: $(grep -E "$pat" "$LG/$name.log" | tail -1 | cut -c1-220)" >> "$H"
    return $rc
}
xof() { "$PY" -I -c 'import json, sys; print(",".join(json.load(open(sys.argv[1]))["p"]))' "$1"; }
for Z in 1 2; do
  if [ $Z = 1 ]; then TAG=M3x1E2_h1e-12_w1e-13; T=x1E2; else TAG=M3x2E2_h1e-12_w1e-12; T=x2E2; fi
  BOX=$M/levers/data/A_M_box_$TAG.json
  XP=$(xof "$BOX")
  for S in $STAGES; do
    case $S in
      xload)   # dh610's box and point files through this machine
        run XLdh_krawczyk_pt_$TAG "relative margins" certify_M.py krawczyk --box=$DH/data/A_M_box_$TAG.json --point=$DH/data/A_M_point_M3$T.json
        run XLdh_controls_pt_$TAG "CONTROLS|UNEXPECTED" certify_M.py controls --box=$DH/data/A_M_box_$TAG.json --point=$DH/data/A_M_point_M3$T.json
        run XLdh_recheck_$TAG "relative margins" certify_M_recheck.py $DH/data/A_M_box_$TAG.json
        run XLdh_recheck_${TAG}_mutate "relative margins" certify_M_recheck.py $DH/data/A_M_box_$TAG.json --mutate
        # macm3's files, and the box-only test (no --point) that certify_M_recheck.py mirrors
        run XLm3_krawczyk_pt_$TAG "relative margins" certify_M.py krawczyk --box=$BOX --point=$M/levers/data/A_M_point_M3$T.json
        run XLm3_krawczyk_boxonly_$TAG "relative margins" certify_M.py krawczyk --box=$BOX
        ;;
      h1)
        run H1_sector_$TAG "ALL CHECKS|contraction constant|old sector" lemma6_sector_M.py --M=3 --rp=0.1 --k=20 --box=1e-6 --Z=0.004489 --x=$XP
        run H1_sector_${TAG}_mutate_series "MUTATION" lemma6_sector_M.py --M=3 --rp=0.1 --k=20 --box=1e-6 --Z=0.004489 --x=$XP --mutate=series
        ;;
      h3)      # from macm3's stored backward enclosure B: the clean value again, and the coords mutation (not run before)
        IN=$M/levers/data/A_M_6a_${TAG}_in.json; B=$M/levers/lemmas_M_runs/lemma6a_${TAG}_B.json
        run H3_lemma6a_${TAG}_loadB "rho_6a" lemma6_jump_M.py --M=3 --box=$IN --x=$XP --load-B=$B
        run H3_lemma6a_${TAG}_coords "rho_6a|NOT established" lemma6_jump_M.py --M=3 --box=$IN --x=$XP --load-B=$B --mutate=coords
        run H3_lemma6a_${TAG}_partial "rho_6a|NOT established" lemma6_jump_M.py --M=3 --box=$IN --x=$XP --load-B=$B --mutate=partial
        ;;
      incl)
        run INCL_exc$Z "THEOREM A SETS|slack" theorem_A_sets_M.py --M=3 --cert=$BOX --ray=$M/h2/lemma6b1_runs_M3_exc$Z/ray.json
        run INCL_exc${Z}_selftest "SELFTEST" theorem_A_sets_M.py --M=3 --cert=$BOX --ray=$M/h2/lemma6b1_runs_M3_exc$Z/ray.json --selftest
        # the other zero's covering: must FAIL
        run INCL_exc${Z}_other "THEOREM A SETS" theorem_A_sets_M.py --M=3 --cert=$BOX --ray=$M/h2/lemma6b1_runs_M3_exc$((3 - Z))/ray.json
        ;;
      h2)      # the stored covering: the check again, a dropped tile, the planted truth and its mutation; row 11 rerun
        D=lemma6b1_runs_M3_rv1079s$Z; R=lemma6b1_runs_M3_rv1079r$Z
        [ -e $D ] || ln -s "$M/h2/lemma6b1_runs_M3_exc$Z" $D
        COM="--M=3 --x=$XP --h=1e-11"
        run H2_check_exc$Z "LEMMA 6b_1 CHECK|tiles glued" lemma6b1_cover_M.py check $COM --tag=_rv1079s$Z --json=$OUT/out/lemma6b1_check_M3_exc$Z.json
        run H2_check_exc${Z}_drop "LEMMA 6b_1 CHECK" lemma6b1_cover_M.py check $COM --tag=_rv1079s$Z --drop=5,6
        run H2_c1check_exc$Z "C1 PLANTED TRUTH" lemma6b1_cover_M.py c1check $COM --tag=_rv1079s$Z
        run H2_c1check_exc${Z}_nov "MUTATION nov" lemma6b1_cover_M.py c1check $COM --tag=_rv1079s$Z --mutate=nov
        mkdir -p $R && cp "$M/h2/lemma6b1_runs_M3_exc$Z/ray.json" $R/
        [ -e $R/row_11.json ] || run H2_row11_exc$Z "disc 11/11" lemma6b1_cover_M.py row --j=11 $COM --tag=_rv1079r$Z
        cp $R/row_11.json "$OUT/out/row_11_exc$Z.json"
        ;;
      path)    # krawczyk_test with a point file whose recorded xi1 is relabelled: is it refused?
        run PATHMIX_$TAG "PATHMIX" review_1079/defaults_check.py pathmix $BOX $M/levers/data/A_M_point_M3$T.json
        ;;
    esac
  done
done
echo "$(date -u '+%F %T') light checks done: $STAGES" >> "$H"
exit 0
}
