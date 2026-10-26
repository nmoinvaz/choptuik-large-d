#!/bin/bash
# Build + install the double-double kernel for this machine's instruction set (AVX-512 / AVX2+FMA / portable are
# selected at compile time from the target features, so build on the machine that runs it).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
RUSTFLAGS="-C target-cpu=native" nice -n 2 maturin build --release --out "$ROOT/dist"
pip install --break-system-packages --force-reinstall --no-deps "$ROOT"/dist/hpkernel-0.1.0-*.whl
# the driver's double-double helpers: multidouble's Python package at the commit pyproject.toml names (needs Rust).
# Its version number does not change between commits, so pip would keep an installed older one ("already
# satisfied"): reinstall it every time (T63, comms 2026-10-26 02:04 UTC).
MD=$(sed -n 's/.*"\(multidouble @ [^"]*\)".*/\1/p' "$ROOT/pyproject.toml")
pip install --break-system-packages --force-reinstall --no-deps "$MD"
python3 -c "import hpkernel, multidouble; print('hpkernel', hpkernel.isa(), 'multidouble', multidouble.isa())"
