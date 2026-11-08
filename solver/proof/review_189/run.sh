#!/usr/bin/env bash
# run.sh LOG CMD ARGS...: run one command, its output to LOG, then its exit status and wall time appended to LOG and to
# exits.txt beside LOG (the #189 review's runs; call it under nice -n 10).
set -u
LOG=$1
shift
T0=$(date +%s)
"$@" > "$LOG" 2>&1
RC=$?
T1=$(date +%s)
echo "EXIT $RC wall $((T1 - T0)) s" >> "$LOG"
echo "$(basename "$LOG" .log): rc $RC, wall $((T1 - T0)) s, ended $(date -u '+%Y-%m-%d %H:%M:%S') UTC" >> "$(dirname "$LOG")/exits.txt"
exit $RC
