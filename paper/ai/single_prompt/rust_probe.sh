# Build check for SP_RUST=1: the vendored crates build offline in the trial's sandbox, and a crate not vendored fails.
set -u
cd /work
echo "PATH=$PATH CARGO_HOME=$CARGO_HOME"
cargo --version; rustc --version; cc --version | head -1
cargo new --quiet app && cd app
printf '%s\n' 'multishoot = "4"' 'multisolver = "0.2"' 'multidouble = "5"' >> Cargo.toml
cat > src/main.rs <<'R'
use multishoot::mobius::Mobius;
fn main() {
    let _m: Option<Mobius> = None;
    println!("built with multishoot, multisolver {}", std::any::type_name::<multidouble::od::OD>());
}
R
s=$(date +%s); cargo build --release 2>&1 | tail -3; echo "release build $(( $(date +%s) - s )) s, exit ${PIPESTATUS[0]}"
./target/release/app && echo "PROBE OK: run"
cd /work && cargo new --quiet control && cd control
echo 'serde = "1"' >> Cargo.toml
if cargo build 2>&1 | tail -2; [ "${PIPESTATUS[0]}" -ne 0 ]; then echo "CONTROL OK: an unvendored crate fails offline"; else echo "CONTROL BAD: serde resolved"; fi
