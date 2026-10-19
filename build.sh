#!/bin/bash
# Build + install the main wheel, and build the two ablation variants into variants/ (same source):
#   main     : RUSTFLAGS from .cargo/config.toml (-C target-cpu=native), opt-level 3, LTO, 1 CGU
#   scalar   : + -C no-vectorize-loops -C no-vectorize-slp   (no auto-vectorisation)
#   auto512  : + -C target-feature=-prefer-256-bit            (LLVM auto-vectorises with zmm)
# All builds run at nice 2. Usage: ./build.sh [main|variants|all]
set -euo pipefail
ROOT=/home/claude/rustkernel
cd "$ROOT"
what=${1:-all}

if [[ $what == main || $what == all ]]; then
  nice -n 2 maturin build --release --out "$ROOT/dist"
  nice -n 2 pip install --break-system-packages --force-reinstall --no-deps "$ROOT"/dist/rustkernel-0.1.0-*.whl
fi

build_variant() {
  local name=$1 flags=$2
  RUSTFLAGS="$flags" CARGO_TARGET_DIR="$ROOT/target_$name" nice -n 2 maturin build --release --out "$ROOT/dist_$name"
  mkdir -p "$ROOT/variants/rustkernel_$name"
  python3 - "$ROOT" "$name" <<'EOF'
import sys, zipfile, glob
root, name = sys.argv[1], sys.argv[2]
w = glob.glob(f"{root}/dist_{name}/rustkernel-*.whl")[0]
z = zipfile.ZipFile(w)
for n in z.namelist():
    if n.startswith("rustkernel/") and (n.endswith(".so") or n.endswith("__init__.py")):
        open(f"{root}/variants/rustkernel_{name}/" + n.split("/")[-1], "wb").write(z.read(n))
print("variant", name, "->", f"{root}/variants/rustkernel_{name}")
EOF
}

if [[ $what == variants || $what == all ]]; then
  build_variant scalar "-C target-cpu=native -C no-vectorize-loops -C no-vectorize-slp"
  build_variant auto512 "-C target-cpu=native -C target-feature=-prefer-256-bit"
fi
