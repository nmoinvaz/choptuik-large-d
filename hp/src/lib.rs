//! Double-double DSS solver kernel.
//!
//! * `v8`: 8-lane f64 vector (AVX-512 zmm, 2 x AVX2 ymm, or portable), chosen at compile time.
//! * `dd`: scalar double-double; `dv`: 8-lane double-double built on `v8`.
//! * `lane`: the scalar abstraction the model is generic over: `DV` (values) and `Du` (value + one forward-mode
//!   tangent, for exact Jacobian columns); for the mixed-precision Newton's f64 Jacobian `F8`, `Fu` (f64), `Fun<N>`
//!   (f64 value, N tangents) and `Dtn<N>` (double-double value, N f64 tangents).
//! * `field`: a tau field on the half grid (M = 8 B points) as B lane vectors; Cauchy products, d/dtheta.
//! * `linalg`: dense double-double LU with partial pivoting (periodic solves, Newton), and the same blocked LU in f64.
//! * `model`: centre series, Taylor stepping in x, horizon series, residual and Jacobian (double-double or f64).
//! * `qd`: scalar quad-double (~64 digits); `qv`: 8-lane quad-double built on `v8`, bitwise the same per lane; `exact`:
//!   exact dyadic and decimal arithmetic for their decimal input/output and their tests. Not used by the solver yet
//!   (notes/QUAD_DOUBLE.md).
//!
//! `v8`, `dd`, `dv`, `qd`, `qv`, `exact` and `linalg`, and the lane types, are the multidouble crate's; this crate adds
//! the tau fields, the model, the block LU, the `lane::Lane` extension (`dmat_apply`) and `field::FieldSolve`.
pub mod block;
pub mod field;
pub mod lane;
pub mod model;

// The arithmetic, the dual lanes and the dense LU live in the multidouble crate (github.com/nmoinvaz/multidouble),
// split out of this one; re-exported under their old paths so the solver code reads as before.
pub use multidouble::{dd, dv, exact, linalg, qd, qv, v8};

#[cfg(test)]
mod qd_residual;
#[cfg(test)]
mod jvp_tests;

#[cfg(feature = "python")]
mod python;
