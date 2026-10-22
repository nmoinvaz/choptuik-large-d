#!/bin/bash
# Run mode-count jobs one after another (single process, nice 10) and commit the data after each.
#   hp/modecount_queue.sh "STATE [opts]" "STATE [opts]" ...      (a job "f64 [opts]" runs modecount_f64.py,
#                                                                  "check STATE [opts]" modecount_check.py)
# Jobs are resumable (modecount_wind.py / modecount_f64.py reuse hp/data/modecount_<tag>_evals.jsonl), and a job that
# finished (listed in hp/data/modecount_queue.done) is skipped, so the same command can simply be started again.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
DONE=hp/data/modecount_queue.done
touch "$DONE"
for job in "$@"; do
  if grep -qxF "$job" "$DONE"; then
    echo "=== skip (done): $job"
    continue
  fi
  set -- $job
  echo "=== $(date '+%F %T') job: $job"
  if [ "$1" = "f64" ]; then
    shift
    nice -n 10 python3 hp/modecount_f64.py "$@"
  elif [ "$1" = "check" ]; then
    shift
    nice -n 10 python3 hp/modecount_check.py "$@" 2>&1 | tee -a hp/data/modecount_check.log
  else
    nice -n 10 python3 hp/modecount_wind.py "$@"
  fi
  rc=${PIPESTATUS[0]}
  echo "=== $(date '+%F %T') exit $rc: $job"
  [ "$rc" = 0 ] && echo "$job" >> "$DONE"
  git add hp/modecount_*.py hp/modecount_*.sh hp/modecount_cx/Cargo.toml hp/modecount_cx/Cargo.lock \
      hp/modecount_cx/.gitignore hp/modecount_cx/src hp/data/modecount_* 2>/dev/null
  git commit -q -m "mode count: $job (exit $rc)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0153nshFY7Jok53P4PMRbvz1" && echo "committed $(git rev-parse --short HEAD)"
done
