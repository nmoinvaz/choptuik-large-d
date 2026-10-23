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
pub mod block;
pub mod dd;
pub mod dv;
pub mod exact;
pub mod field;
pub mod lane;
pub mod linalg;
pub mod model;
pub mod qd;
pub mod qv;
pub mod v8;

#[cfg(test)]
mod qd_residual;
#[cfg(test)]
mod qd_tests;

#[cfg(feature = "python")]
mod python;
