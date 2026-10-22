#!/bin/bash
# Run mode-count jobs one after another (single process, nice 10) and commit the data after each.
#   hp/modecount_queue.sh "STATE [opts]" "STATE [opts]" ...      (a job "f64 [opts]" runs modecount_f64.py)
# Each job is resumable: modecount_wind.py / modecount_f64.py reuse hp/data/modecount_<tag>_evals.jsonl.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
for job in "$@"; do
  set -- $job
  echo "=== $(date '+%F %T') job: $job"
  if [ "$1" = "f64" ]; then
    shift
    nice -n 10 python3 hp/modecount_f64.py "$@"
  else
    nice -n 10 python3 hp/modecount_wind.py "$@"
  fi
  rc=$?
  echo "=== $(date '+%F %T') exit $rc: $job"
  git add hp/modecount_*.py hp/modecount_*.sh hp/modecount_cx/Cargo.toml hp/modecount_cx/Cargo.lock \
      hp/modecount_cx/.gitignore hp/modecount_cx/src hp/data/modecount_* 2>/dev/null
  git commit -q -m "mode count: $job (exit $rc)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0153nshFY7Jok53P4PMRbvz1" && echo "committed $(git rev-parse --short HEAD)"
done
