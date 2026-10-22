#!/bin/bash
# Build the complex-lambda growth-mode determinant (hp/modecount_cx, a separate crate on top of the hp kernel) for this
# machine's instruction set and put the extension module at hp/modecount_cx/build/modecount_cx.so, where the
# modecount_*.py scripts import it from (nothing is pip-installed; hpkernel itself is untouched).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT/modecount_cx"
[ -f Cargo.lock ] || cp "$ROOT/Cargo.lock" Cargo.lock
RUSTFLAGS="-C target-cpu=native" nice -n 10 cargo build --release --offline
mkdir -p build
cp target/release/libmodecount_cx.so build/modecount_cx.so
python3 -c "import sys; sys.path.insert(0, 'build'); import modecount_cx; print('modecount_cx ok', modecount_cx.__file__)"
