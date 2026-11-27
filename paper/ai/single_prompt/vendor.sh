{
set -euo pipefail
R=/home/nathan/sp-trials/rust
[ -e "$R" ] && { echo "$R exists already"; exit 2; }
mkdir -p "$R/seed/src"
cd "$R/seed"
cat > Cargo.toml <<'T'
[package]
name = "seed"
version = "0.0.0"
edition = "2021"

[dependencies]
multishoot = "4.2"
multisolver = "0.2.2"
multidouble = "5.1"

# multishoot main at 70fac930 (series::PeriodicLu, mobius::Mobius and the rest of #66), not yet on crates.io
[patch.crates-io]
multishoot = { git = "https://github.com/nmoinvaz/multishoot", rev = "70fac930736c4eb7b24bdfe599b2bf8da90c5c4c" }
T
echo 'fn main() {}' > src/main.rs
~/.cargo/bin/cargo vendor --versioned-dirs "$R/vendor" > "$R/vendor_config.toml"
ls "$R/vendor"
cat "$R/vendor_config.toml"
exit
}
