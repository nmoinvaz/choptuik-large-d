#!/usr/bin/env bash
# Build CAPD at a pinned commit (outside this repository) and then capd_check against it.
#
# CAPD (Computer Assisted Proofs in Dynamics, https://github.com/CAPDGroup/CAPD) is distributed under the GNU General
# Public License v3; see COPYING in its source tree. It is not vendored here: this script clones it.
#
#   ./build.sh                 # clone + build CAPD into $CAPD_DIR (default ~/.cache/capd_check/CAPD), then capd_check
#   CAPD_DIR=/some/dir ./build.sh
#   JOBS=4 ./build.sh
# Needs cmake, a C++17 compiler, git, and the MPFR and GMP headers (Debian/Ubuntu: apt install libmpfr-dev libgmp-dev):
# the point run uses CAPD's multiprecision intervals. Only the CAPD library and filib are built (no examples or
# tests): about 5 minutes on 2 cores.
set -euo pipefail

CAPD_URL=https://github.com/CAPDGroup/CAPD.git
CAPD_COMMIT=37057e5899bccaa6cd5701057b7e6ea5e232ef8b      # CAPD 6.1.0, master of 2026-10-18
HERE=$(cd "$(dirname "$0")" && pwd)
CAPD_DIR=${CAPD_DIR:-$HOME/.cache/capd_check/CAPD}
JOBS=${JOBS:-2}

if [ ! -d "$CAPD_DIR/.git" ]; then
  mkdir -p "$(dirname "$CAPD_DIR")"
  git clone "$CAPD_URL" "$CAPD_DIR"
fi
git -C "$CAPD_DIR" fetch --quiet origin "$CAPD_COMMIT" 2>/dev/null || true
git -C "$CAPD_DIR" checkout --quiet "$CAPD_COMMIT"
echo "CAPD at $(git -C "$CAPD_DIR" rev-parse HEAD)"

if [ ! -x "$CAPD_DIR/build/bin/capd-config" ] || [ ! -f "$CAPD_DIR/build/libcapd.a" ]; then
  cmake -S "$CAPD_DIR" -B "$CAPD_DIR/build" -DCMAKE_BUILD_TYPE=Release -DCAPD_ENABLE_MULTIPRECISION=ON
  cmake --build "$CAPD_DIR/build" -j "$JOBS" --target capd
fi

cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release -DCAPD_CONFIG="$CAPD_DIR/build/bin/capd-config"
cmake --build "$HERE/build" -j "$JOBS"
echo "built $HERE/build/capd_check"
echo "run:  $HERE/build/capd_check $HERE/data/start_point.txt $HERE/data/start_box_1e-9.txt $HERE/data/start_box_1e-8.txt"
