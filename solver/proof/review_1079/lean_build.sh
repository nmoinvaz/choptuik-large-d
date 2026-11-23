#!/bin/bash
# Review of #1079, check 4: build the Lean project of feat/lean-excited one module at a time (one lean process),
# then the whole target, then the review file's axioms and mutations.
# usage: bash lean_build.sh LEANDIR LOG     LEANDIR = lean/ of a checkout of feat/lean-excited, with
#                                           review_1079/Exc1079.lean copied to LEANDIR/review/
{
cd "$1" || exit 2
LOG=$2
: > "$LOG"
for m in InnerConstant Krawczyk KrawczykN M1capd M1capdWide M1arb M1arbWide M3cert M3certWide M5cert M5certWide \
         M3exc1 M3exc1Wide M3exc2 M3exc2Wide NonVacuous; do
  t0=$(date +%s)
  nice -n 10 lake build DSS.$m >> "$LOG" 2>&1; rc=$?
  echo "$(date -u '+%F %T') lake build DSS.$m exit $rc, $(( $(date +%s) - t0 )) s" >> "$LOG"
  [ $rc = 0 ] || exit 1
done
nice -n 10 lake build >> "$LOG" 2>&1; rc=$?
echo "$(date -u '+%F %T') lake build exit $rc" >> "$LOG"
[ $rc = 0 ] || exit 1
nice -n 10 lake env lean review/Exc1079.lean >> "$LOG" 2>&1; rc=$?
echo "$(date -u '+%F %T') lake env lean review/Exc1079.lean exit $rc" >> "$LOG"
exit $rc
}
