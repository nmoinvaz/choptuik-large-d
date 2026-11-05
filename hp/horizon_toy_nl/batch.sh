#!/bin/zsh
# sequential runs, one process at a time, each under 15 minutes: batch.sh FORM DPS S1 S2 ... [-- extra args]
form=$1; dps=$2; shift 2
extra=()
ss=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; extra=("$@"); break; fi
  ss+=("$1"); shift
done
for s in $ss; do
  timeout 870 nice -n 10 ${PYTHON:-python3} -u run_basin.py $form $dps $s $extra >> log_${form}_${dps}${INEXACT_START:+_inexact}.txt 2>&1
  echo "exit $? form $form dps $dps s $s" >> log_${form}_${dps}.txt
done
echo "BATCH DONE $form $dps"
