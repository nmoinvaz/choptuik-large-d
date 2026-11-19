//! hp/src/fastmul.rs's planted checks (`fastmul_check::check*`, also hp's test `fastmul_planted`) and its timings
//! against the naive octo-double product, on this machine, interleaved:
//!
//! ```text
//! cd hp/fastmul_bench && cargo run --release                 checks, then timings at n = 128, 251, 512 (7 reps)
//! cargo run --release -- --no-check --reps 5                 timings only
//! cargo run --release -- --dump DIR                          accuracy and time on factors dumped by hp's tests
//! ```
// hp's modules as they are; what only hp's series code uses is unused here
#[allow(dead_code)]
#[path = "../../src/fastmul.rs"]
mod fastmul;
#[path = "../../src/fastmul_check.rs"]
mod fastmul_check;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(fastmul_check::run(&args));
}
