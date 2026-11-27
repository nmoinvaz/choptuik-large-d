#!/bin/bash
# Build and install ellrs (block 12 of the l = 1 mode matrix in Rust, #1102) for this machine's instruction set:
# multidouble's lanes and matrixmultiply's kernels follow the target features, so build on the machine that runs it.
#   ellrs/build.sh [SITE]       SITE: install into this folder (use PYTHONPATH=SITE) instead of the active venv
# On macOS the Ozaki products come from Accelerate's dgemm (ELLRS_FEATURES=accelerate, the default there), elsewhere
# from matrixmultiply; the products are exact, so the bits are the same either way. Needs Rust (cargo), maturin, and access to crates.io on the first build.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
PY=""
if [ -n "${VIRTUAL_ENV:-}" ]; then
  for c in "$VIRTUAL_ENV/bin/python" "$VIRTUAL_ENV/Scripts/python.exe"; do [ -x "$c" ] && PY=$c && break; done
fi
PY=${PY:-$(command -v python3 || command -v python)}
TAG=$("$PY" -c "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')")
# macOS defaults to Accelerate's dgemm (about 7x matrixmultiply's NEON kernel on an M5); ELLRS_FEATURES=none for it
[ "$(uname -s)" = Darwin ] && : "${ELLRS_FEATURES:=accelerate}"
FEAT=()
[ -n "${ELLRS_FEATURES:-}" ] && [ "$ELLRS_FEATURES" != none ] && FEAT=(--features "$ELLRS_FEATURES")
rm -f "$ROOT"/dist/ellrs-0.1.0-"$TAG"-*.whl
# strip=false in Cargo.toml (macOS 27's dyld); RUSTFLAGS for the native instruction set
RUSTFLAGS="-C target-cpu=native" CARGO_PROFILE_RELEASE_STRIP=false nice -n 10 maturin build --release ${FEAT[@]+"${FEAT[@]}"} -i "$PY" --out "$ROOT/dist"
WHL=$(ls "$ROOT"/dist/ellrs-0.1.0-"$TAG"-*.whl)
# pip, or uv where the venv has no pip
if "$PY" -m pip --version >/dev/null 2>&1; then PIP=("$PY" -m pip install --force-reinstall --no-deps)
else PIP=(uv pip install --python "$PY" --reinstall --no-deps); fi
if [ $# -ge 1 ]; then
  mkdir -p "$1"
  "${PIP[@]}" --target "$1" "$WHL"
  PYTHONPATH="$1" "$PY" -c "import ellrs; print('ellrs', ellrs.isa(), ellrs.gemm(), ellrs.__file__)"
else
  "${PIP[@]}" "$WHL"
  "$PY" -c "import ellrs; print('ellrs', ellrs.isa(), ellrs.gemm(), ellrs.__file__)"
fi
