#!/bin/bash
# Review of #1077: the exporter must regenerate the 8 instance files of lean/DSS byte for byte.
#   bash lean/review/regen_check.sh WORKDIR [PYTHON]     from the repository root; exit 0 iff all 8 are identical
# WORKDIR is a scratch directory outside the repository; PYTHON an interpreter with python-flint (default python3).
{
set -u
S="$1"
PY="${2:-python3}"
E=lean/tools/export_krawczyk.py
P=solver/proof
mkdir -p "$S"
run() {  # mode source name
  "$PY" "$E" "$1" "$2" "$3" "$S/$3.lean" || exit 2
  "$PY" "$E" "$1" "$2" "$3Wide" "$S/$3Wide.lean" --widen=1000000 || exit 2
}
run capd "$P/capd_check/results/capd_enclosures_1e-9.json" M1capd
run arb "$P/validated_int_krawczyk_1e-9.json" M1arb
run certM "$P/data/A_M_box_M3_1e-10.json" M3cert
run certM2 "$P/data/A_M2_box_M5_1e-10.json" M5cert
bad=0
for f in M1capd M1capdWide M1arb M1arbWide M3cert M3certWide M5cert M5certWide; do
  if cmp -s "$S/$f.lean" "lean/DSS/$f.lean"; then echo "$f: identical"; else echo "$f: DIFFERS"; bad=1; fi
done
exit $bad
}
