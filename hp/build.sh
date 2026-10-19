#!/bin/bash
# Build + install the double-double kernel for this machine's instruction set (AVX-512 / AVX2+FMA / portable are
# selected at compile time from the target features, so build on the machine that runs it).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
RUSTFLAGS="-C target-cpu=native" nice -n 2 maturin build --release --out "$ROOT/dist"
pip install --break-system-packages --force-reinstall --no-deps "$ROOT"/dist/hpkernel-0.1.0-*.whl
python3 -c "import hpkernel; print('hpkernel', hpkernel.isa())"
