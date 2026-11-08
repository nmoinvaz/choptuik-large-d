#!/bin/bash
# Build + install the double-double kernel for this machine's instruction set (AVX-512 / AVX2+FMA / portable are
# selected at compile time from the target features, so build on the machine that runs it).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
# the environment's python: the active venv's (bin/ on Unix, Scripts/ on Windows, where a python3 on PATH may be another
# interpreter), else python3 or python on PATH
PY=""
if [ -n "${VIRTUAL_ENV:-}" ]; then
  for c in "$VIRTUAL_ENV/bin/python" "$VIRTUAL_ENV/Scripts/python.exe"; do [ -x "$c" ] && PY=$c && break; done
fi
PY=${PY:-$(command -v python3 || command -v python)}
# --break-system-packages (a system python on macOS) is unknown to pip before 23.0.1 (T108, a Windows venv's pip 21)
BSP=""
case "$("$PY" -m pip install --help)" in *--break-system-packages*) BSP=--break-system-packages ;; esac
# build for this interpreter and install only its wheel: dist/ may hold wheels of other Pythons (#179)
TAG=$("$PY" -c "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')")
RUSTFLAGS="-C target-cpu=native" nice -n 2 maturin build --release -i "$PY" --out "$ROOT/dist"
"$PY" -m pip install $BSP --force-reinstall --no-deps "$ROOT"/dist/hpkernel-0.1.0-"$TAG"-*.whl
# the driver's double-double helpers: multidouble's Python package at the commit pyproject.toml names (needs Rust).
# Its version number does not change between commits, so pip would keep an installed older one ("already
# satisfied"): reinstall it every time (T63, comms 2026-10-26 02:04 UTC).
MD=$(sed -n 's/.*"\(multidouble @ [^"]*\)".*/\1/p' "$ROOT/pyproject.toml")
"$PY" -m pip install $BSP --force-reinstall --no-deps "$MD"
"$PY" -c "import hpkernel, multidouble; print('hpkernel', hpkernel.isa(), 'multidouble', multidouble.isa())"
