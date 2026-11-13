#!/bin/bash
# Build the sigma-model kernel (sigmams) for this machine and install it into a site directory, not into a venv:
#   PY=/path/to/python3 SITE=/path/to/site matter/sigma_ms/build.sh
# then run the scripts with PYTHONPATH=$SITE. Defaults: PY = python3 on PATH, SITE = matter/sigma_ms/site,
# CARGO_TARGET_DIR = matter/sigma_ms/target. FAST=1 builds without fat LTO and with 16 codegen units (quicker, less
# memory; hp found the same bits either way, #600, #604). On macOS 27 keep strip = false (Cargo.toml).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
PY=${PY:-$(command -v python3)}
SITE=${SITE:-$ROOT/site}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$ROOT/target}
export CARGO_PROFILE_RELEASE_STRIP=false
if [ "${FAST:-0}" = "1" ]; then
  export CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
fi
TAG=$("$PY" -c "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')")
OUT=$CARGO_TARGET_DIR/wheels
RUSTFLAGS="-C target-cpu=native" nice -n 10 "$PY" -m maturin build --release -i "$PY" --out "$OUT"
"$PY" -m pip install --target "$SITE" --no-deps --upgrade --force-reinstall "$OUT"/sigmams-0.1.0-"$TAG"-*.whl
PYTHONPATH=$SITE "$PY" -c "import sigmams; print('sigmams', sigmams.isa(), sigmams.__file__)"
