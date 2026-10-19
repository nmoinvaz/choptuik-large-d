//! Double-double DSS solver kernel.
//!
//! * `v8`: 8-lane f64 vector (AVX-512 zmm, 2 x AVX2 ymm, or portable), chosen at compile time.
//! * `dd`: scalar double-double; `dv`: 8-lane double-double built on `v8`.
//! * `lane`: the scalar abstraction the model is generic over: `DV` (values) and `Du` (value + one forward-mode
//!   tangent, for exact Jacobian columns).
//! * `field`: a tau field on the half grid (M = 8 B points) as B lane vectors; Cauchy products, d/dtheta.
//! * `linalg`: dense double-double LU with partial pivoting (periodic solves, Newton).
//! * `model`: centre series, Taylor stepping in x, horizon series, residual and Jacobian.
pub mod dd;
pub mod dv;
pub mod field;
pub mod lane;
pub mod linalg;
pub mod model;
pub mod v8;

#[cfg(feature = "python")]
mod python;
