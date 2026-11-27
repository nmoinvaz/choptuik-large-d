//! CPU time by kind of work, summed over threads, per phase (the horizon series, the RK4): the profile of
//! `Block12.profile()`. Phases never overlap, so one global phase index serves every thread.
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

pub const SLICE: usize = 0;
pub const GEMM: usize = 1;
pub const COMB: usize = 2;
pub const LU: usize = 3;
/// a phase's wall time (series) or its chunks' summed wall time (RK4, one thread per chunk)
pub const WALL: usize = 4;
/// f64 multiply-adds of the Ozaki products
pub const MADD: usize = 5;
pub const NK: usize = 6;
pub const NAMES: [&str; NK] = ["slice", "gemm", "combine", "lu", "wall", "madd"];

pub static PHASE: AtomicUsize = AtomicUsize::new(0);
static C: [[AtomicU64; NK]; 2] = [const { [const { AtomicU64::new(0) }; NK] }; 2];

thread_local! {
    /// set inside an RK4 chunk: its products and elementwise work stay on the chunk's thread
    pub static SEQ: Cell<bool> = const { Cell::new(false) };
}

pub fn seq() -> bool {
    SEQ.with(|s| s.get())
}

pub fn add_ns(k: usize, t0: Instant) {
    C[PHASE.load(Ordering::Relaxed)][k].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

pub fn add_count(k: usize, n: u64) {
    C[PHASE.load(Ordering::Relaxed)][k].fetch_add(n, Ordering::Relaxed);
}

/// (phase, kind) -> value (seconds, or the count for MADD); reset clears the counters
pub fn take(reset: bool) -> Vec<(usize, &'static str, f64)> {
    let mut out = Vec::new();
    for (p, row) in C.iter().enumerate() {
        for (k, c) in row.iter().enumerate() {
            let v = if reset { c.swap(0, Ordering::Relaxed) } else { c.load(Ordering::Relaxed) };
            out.push((p, NAMES[k], if k == MADD { v as f64 } else { v as f64 * 1e-9 }));
        }
    }
    out
}
