#!/bin/bash
# Build the complex-lambda growth-mode determinant (hp/modecount_cx, a separate crate on top of the hp kernel) for this
# machine's instruction set and put the extension module at hp/modecount_cx/build/modecount_cx.so, where the
# modecount_*.py scripts import it from (nothing is pip-installed; hpkernel itself is untouched).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT/modecount_cx"
[ -f Cargo.lock ] || cp "$ROOT/Cargo.lock" Cargo.lock
# macOS: a Python extension leaves the interpreter's symbols to the loader (maturin passes the same flags)
EXTRA=""
[ "$(uname)" = Darwin ] && EXTRA="-C link-arg=-undefined -C link-arg=dynamic_lookup"
RUSTFLAGS="-C target-cpu=native $EXTRA" nice -n 10 cargo build --release --offline
# the active venv's python (as hp/build.sh), else python3 on PATH
PY=""
if [ -n "${VIRTUAL_ENV:-}" ]; then
  for c in "$VIRTUAL_ENV/bin/python" "$VIRTUAL_ENV/Scripts/python.exe"; do [ -x "$c" ] && PY=$c && break; done
fi
PY=${PY:-$(command -v python3 || command -v python)}
# name the module with the interpreter's ABI tag (modecount_cx.cpython-314-darwin.so): each Python imports its own
# build, and a build for another version is never loaded (a cp313 build segfaults under 3.14, #179)
EXT=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
mkdir -p build
# an untagged module from before would be imported by any interpreter without its own build: set it aside
[ -f build/modecount_cx.so ] && mv build/modecount_cx.so build/modecount_cx.so.untagged
for f in target/release/libmodecount_cx.so target/release/libmodecount_cx.dylib target/release/modecount_cx.dll; do
  if [ -f "$f" ]; then cp "$f" "build/modecount_cx$EXT"; fi
done
"$PY" -c "import sys; sys.path.insert(0, 'build'); import modecount_cx; print('modecount_cx ok', modecount_cx.__file__)"
