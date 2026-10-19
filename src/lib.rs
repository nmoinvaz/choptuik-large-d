//! Rust port of the DSSSpec multiple-shooting residual. The Python bindings are built with the
//! `python` feature (enabled by maturin).

pub mod dispatch;
pub mod model;
pub mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod tests;
