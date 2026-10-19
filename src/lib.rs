//! Rust port of the DSSSpec multiple-shooting residual. The Python bindings are built with the
//! `python` feature (enabled by maturin).

pub mod model;
pub mod scalar;

#[cfg(all(feature = "python", target_arch = "x86_64", target_feature = "avx512f"))]
mod avx;

#[cfg(feature = "python")]
mod python;
