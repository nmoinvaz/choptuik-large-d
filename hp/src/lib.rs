//! Double-double DSS solver kernel.
//!
//! * `v8`: 8-lane f64 vector (AVX-512 zmm, 2 x AVX2 ymm, 4 x SSE2 or NEON, or portable), chosen at compile time
//!   (multidouble's `native` backend).
//! * `dd`: scalar double-double; `dv`: 8-lane double-double built on `v8`.
//! * `lane`: the scalar abstraction the model is generic over: `DV` (values) and `Du` (value + one forward-mode
//!   tangent, for exact Jacobian columns); for the mixed-precision Newton's f64 Jacobian `F8`, `Fu` (f64), `Fun<N>`
//!   (f64 value, N tangents) and `Dtn<N>` (double-double value, N f64 tangents).
//! * `field`: a tau field on the half grid (M = 8 B points) as B lane vectors; Cauchy products, d/dtheta.
//! * `linalg`: dense double-double LU with partial pivoting (periodic solves, Newton), and the same blocked LU in f64.
//! * `model`: centre series, Taylor stepping in x, horizon series, residual and Jacobian (double-double or f64).
//! * `qsolve`: the residual, its exact Jacobian and the Newton step in quad-double (notes/QUAD_DOUBLE.md, section 12).
//! * `qd`: scalar quad-double (~64 digits); `qv`: 8-lane quad-double built on `v8`, bitwise the same per lane; `exact`:
//!   exact dyadic and decimal arithmetic for their decimal input/output and their tests. Not used by the solver yet
//!   (notes/QUAD_DOUBLE.md).
//!
//! `v8`, `dd`, `dv`, `qd`, `qv`, `exact` and `linalg`, the lane types, the field arithmetic (`field`) and the staircase
//! LU (`block`) are the multidouble crate's; this crate adds the model and the layout of its Newton system.
pub mod block;
pub mod field;
pub mod lane;
pub mod model;
pub mod qsolve;

// The arithmetic, the dual lanes and the dense LU live in the multidouble crate (crates.io; github.com/nmoinvaz/multidouble),
// split out of this one; re-exported under their old paths so the solver code reads as before. Its lane types are
// generic over the SIMD backend; the solver uses them on the backend the build's target features select
// (multidouble::native, `-C target-cpu=native` in hp/build.sh), so V8, DV, QV and the dual lanes here are those aliases.
pub use multidouble::{dd, exact, linalg, qd, simd};

/// 8 f64 lanes on the compile-time backend (multidouble's `v8`, with `V8` and `M8` its `native` aliases)
pub mod v8 {
    pub use multidouble::native::{M8, V8};
    pub use multidouble::v8::*;
}
/// 8-lane double-double on the compile-time backend
pub mod dv {
    pub use multidouble::dv::*;
    pub use multidouble::native::{DotAcc, DV};
}
/// 8-lane quad-double on the compile-time backend
pub mod qv {
    pub use multidouble::native::{QDotAcc, QV};
    pub use multidouble::qv::*;
}

#[cfg(test)]
mod qd_residual;
#[cfg(test)]
mod jvp_tests;

#[cfg(feature = "python")]
mod python;
