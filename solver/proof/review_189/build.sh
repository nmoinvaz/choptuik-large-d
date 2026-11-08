#!/usr/bin/env bash
# Build capd_m5 against a CAPD build made by ../capd_check/build.sh (CAPD 6.1.0, commit 37057e58, multiprecision on).
# On Apple Silicon CAPD's default interval library (filib) needs x86 SSE, so CAPD is configured with
# -DCAPD_INTERVAL_TYPE=NATIVE (its own intervals, rounding through the ARM FPCR), and MPFR/GMP come from Homebrew:
#   CPATH=/opt/homebrew/include LIBRARY_PATH=/opt/homebrew/lib cmake -S $CAPD_DIR -B $CAPD_DIR/build \
#     -DCMAKE_BUILD_TYPE=Release -DCAPD_ENABLE_MULTIPRECISION=ON -DCAPD_INTERVAL_TYPE=NATIVE
#   cmake --build $CAPD_DIR/build -j 2 --target capd
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CAPD_DIR=${CAPD_DIR:-$HOME/.cache/capd_check/CAPD}
CFG="$CAPD_DIR/build/bin/capd-config"
EXTRA_I=${EXTRA_I:--I/opt/homebrew/include}
EXTRA_L=${EXTRA_L:--L/opt/homebrew/lib}
mkdir -p "$HERE/build"
# shellcheck disable=SC2046
c++ $("$CFG" --cflags) $EXTRA_I -o "$HERE/build/capd_m5" "$HERE/capd_m5.cpp" $("$CFG" --libs) $EXTRA_L -lmpfr -lgmp
echo "built $HERE/build/capd_m5"
